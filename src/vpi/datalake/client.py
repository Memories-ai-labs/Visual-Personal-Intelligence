"""HTTP client for the Memories.ai video datalake.

Design notes that are not obvious from the endpoint list:

* Auth is `Authorization: sk-mai-...` — the raw key, **no** `Bearer` prefix.
* 429 / 5xx / `409 video_not_ready` all carry `Retry-After`; we honour it rather
  than inventing our own backoff.
* Signed URLs (frames, clips, thumbnails) expire — 15 min for search
  thumbnails, 5 h for clips. Nothing here caches them; callers re-fetch.
* Cost is metered per call, so every call is recorded on an optional meter.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

from vpi.config import Settings, get_settings
from vpi.datalake import errors
from vpi.datalake.models import (
    Collection,
    Moment,
    Operation,
    SearchHit,
    SearchPage,
    Segment,
    Speaker,
    Video,
)

# Per-call prices, from the published pricing table. Used for the cost ledger the
# agent shows after each turn — not for billing.
PRICE_SEARCH = 0.008
PRICE_MOMENT = 0.008
PRICE_MOMENT_CLIP = 0.005
PRICE_DERIVED_READ = 0.001
PRICE_INDEX_PER_MINUTE = 0.04
RERANK_MULTIPLIER = 3.0

RETRY_STATUSES = {409, 429, 500, 502, 503, 504}
SEARCH_TARGETS = ("caption", "transcription", "summary", "title", "frame_embedding", "event")
MOMENT_EXPANDS = ("caption", "transcription", "frame", "clip", "speakers", "entities", "events")


class DataLakeClient:
    """Thin, typed wrapper over the datalake REST surface."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        meter: Callable[[str, float], None] | None = None,
        max_retries: int = 3,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or get_settings()
        if not self.settings.memories_api_key and transport is None:
            raise errors.AuthError(
                "MEMORIES_API_KEY is not set. Create a key at https://console.memories.ai "
                "and put it in .env, or run with VPI_DEMO=1 to use the bundled fixtures."
            )
        self._meter = meter
        self._max_retries = max_retries
        self._sleep = sleep
        self._client = httpx.Client(
            base_url=self.settings.datalake_base_url,
            headers={
                "Authorization": self.settings.memories_api_key,
                "User-Agent": "vpi/0.1 (+https://github.com/Memories-ai-labs/Visual-Personal-Intelligence)",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
            transport=transport,
        )

    # ---------------------------------------------------------------- plumbing

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> DataLakeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _charge(self, label: str, usd: float) -> None:
        if self._meter is not None and usd:
            self._meter(label, usd)

    def _request(self, method: str, path: str, **kwargs: Any) -> dict:
        body, _ = self._request_with_id(method, path, **kwargs)
        return body

    def _request_with_id(self, method: str, path: str, **kwargs: Any) -> tuple[dict, str]:
        """One call, with Retry-After-driven retries on the retryable statuses.

        Returns the body and the request id. On success the id arrives only in
        the `X-Request-ID` header — the body carries it on errors — and search
        needs it to know which scores are comparable with which.
        """
        last: errors.DataLakeError | None = None
        for attempt in range(self._max_retries + 1):
            response = self._client.request(method, path, **kwargs)
            if response.is_success:
                request_id = response.headers.get("X-Request-ID", "")
                if not response.content:
                    return {}, request_id
                body = response.json()
                return (body if isinstance(body, dict) else {"data": body}), request_id

            payload = _safe_json(response)
            error = errors.from_response(
                response.status_code, payload, f"HTTP {response.status_code} from {path}"
            )
            if not error.request_id:
                error.request_id = response.headers.get("X-Request-ID", "")

            if response.status_code not in RETRY_STATUSES or attempt == self._max_retries:
                raise error

            delay = error.retry_after
            if delay is None:
                header = response.headers.get("Retry-After")
                delay = float(header) if header and header.isdigit() else 2.0 * (attempt + 1)
            last = error
            self._sleep(min(delay, 30.0))

        raise last or errors.DataLakeError(f"{method} {path} failed")

    # ------------------------------------------------------------- collections

    def list_collections(self, *, limit: int = 100) -> list[Collection]:
        body = self._request("GET", "/collections", params={"limit": limit})
        return [Collection.model_validate(c) for c in body.get("collections", [])]

    def get_collection(self, collection_id: str) -> Collection:
        return Collection.model_validate(self._request("GET", f"/collections/{collection_id}"))

    def create_collection(self, name: str, *, face_recognition_enabled: bool = False) -> Collection:
        body = self._request(
            "POST",
            "/collections",
            json={"name": name, "face_recognition_enabled": face_recognition_enabled},
        )
        return Collection.model_validate(body)

    # ------------------------------------------------------------------ videos

    def list_videos(
        self,
        collection_id: str,
        *,
        status: str | None = None,
        limit: int = 50,
        cursor: str | None = None,
    ) -> tuple[list[Video], str | None]:
        params: dict[str, Any] = {"collection_id": collection_id, "limit": limit}
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        body = self._request("GET", "/videos", params=params)
        videos = [Video.model_validate(v) for v in body.get("videos", [])]
        return videos, body.get("next_cursor")

    def get_video(self, video_id: str) -> Video:
        return Video.model_validate(self._request("GET", f"/videos/{video_id}"))

    def upload_video_url(
        self,
        collection_id: str,
        source_url: str,
        *,
        metadata: dict | None = None,
        captured_at: str | None = None,
        fps: float | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Index a video from a **public direct link** (not a YouTube/TikTok page)."""
        payload: dict[str, Any] = {"collection_id": collection_id, "source_url": source_url}
        if metadata:
            payload["metadata"] = metadata
        if captured_at:
            payload["captured_at"] = captured_at
        if fps is not None:
            payload["fps"] = fps
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key
        return self._request("POST", "/videos", json=payload)

    def upload_video_file(
        self,
        collection_id: str,
        path: str | Path,
        *,
        metadata: dict | None = None,
        captured_at: str | None = None,
        fps: float | None = None,
        idempotency_key: str | None = None,
    ) -> dict:
        """Index a local file: a `json` part plus a `file` part, multipart."""
        import json as _json

        file_path = Path(path)
        payload: dict[str, Any] = {"collection_id": collection_id}
        payload["metadata"] = {"title": file_path.name, **(metadata or {})}
        if captured_at:
            payload["captured_at"] = captured_at
        if fps is not None:
            payload["fps"] = fps
        if idempotency_key:
            payload["idempotency_key"] = idempotency_key

        with file_path.open("rb") as handle:
            return self._request(
                "POST",
                "/videos",
                data={"json": _json.dumps(payload)},
                files={"file": (file_path.name, handle, "video/mp4")},
            )

    # -------------------------------------------------------------- operations

    def get_operation(self, operation_id: str) -> Operation:
        return Operation.model_validate(self._request("GET", f"/operations/{operation_id}"))

    def balance_usd(self) -> float:
        return float(self._request("GET", "/usage/balance").get("balance_usd", 0.0))

    # ------------------------------------------------------------------ search

    def search(
        self,
        collection_id: str,
        query: str,
        *,
        targets: Iterable[str] = ("caption", "transcription"),
        mode: str = "semantic",
        top_k: int = 20,
        filter: dict | None = None,  # noqa: A002 - mirrors the API field name
        rerank: bool = False,
        group_by: str = "moment",
        cursor: str | None = None,
    ) -> SearchPage:
        targets = [t for t in targets if t in SEARCH_TARGETS]
        if not targets:
            raise errors.InvalidArgument(f"targets must be a non-empty subset of {SEARCH_TARGETS}")
        payload: dict[str, Any] = {
            "collection_id": collection_id,
            "query": query,
            "mode": mode,
            "targets": targets,
            "top_k": top_k,
            "group_by": group_by,
        }
        if filter:
            payload["filter"] = filter
        if rerank:
            payload["rerank"] = True
        if cursor:
            if mode == "hybrid":
                raise errors.InvalidArgument(
                    "hybrid mode does not paginate — raise top_k instead of passing a cursor"
                )
            payload["cursor"] = cursor

        body, header_request_id = self._request_with_id("POST", "/search", json=payload)
        self._charge("search", PRICE_SEARCH * (RERANK_MULTIPLIER if rerank else 1.0))

        request_id = str(body.get("request_id") or header_request_id)
        hits = []
        for raw in body.get("results", []):
            hit = SearchHit.model_validate(raw)
            hit.request_id = request_id
            hits.append(hit)
        return SearchPage(
            results=hits,
            next_cursor=body.get("next_cursor"),
            hint=str(body.get("hint") or ""),
            request_id=request_id,
        )

    # ----------------------------------------------------------------- moments

    def get_moment(
        self, ref: str, *, expand: Iterable[str] = ("caption", "transcription")
    ) -> Moment:
        expand = [e for e in expand if e in MOMENT_EXPANDS]
        body = self._request(
            "GET", f"/moments/{ref}", params={"expand": ",".join(expand)} if expand else None
        )
        cost = PRICE_MOMENT + (PRICE_MOMENT_CLIP if "clip" in expand else 0.0)
        self._charge("get_moment", cost)
        return Moment.model_validate(body)

    def get_transcription(
        self, video_id: str, *, start: float | None = None, end: float | None = None
    ) -> list[Segment]:
        body = self._request("GET", f"/videos/{video_id}/transcription", params=_window(start, end))
        self._charge("transcription", PRICE_DERIVED_READ)
        return [Segment.model_validate(s) for s in body.get("segments", body.get("data", []))]

    def get_caption(
        self, video_id: str, *, start: float | None = None, end: float | None = None
    ) -> list[Segment]:
        body = self._request("GET", f"/videos/{video_id}/caption", params=_window(start, end))
        self._charge("caption", PRICE_DERIVED_READ)
        return [Segment.model_validate(s) for s in body.get("segments", body.get("data", []))]

    def get_summary(self, video_id: str) -> str:
        body = self._request("GET", f"/videos/{video_id}/summary")
        self._charge("summary", PRICE_DERIVED_READ)
        return str(body.get("summary", ""))

    def get_title(self, video_id: str) -> str:
        body = self._request("GET", f"/videos/{video_id}/title")
        self._charge("title", PRICE_DERIVED_READ)
        return str(body.get("title", ""))

    def get_speakers(self, video_id: str) -> list[Speaker]:
        body = self._request("GET", f"/videos/{video_id}/speakers")
        self._charge("speakers", PRICE_DERIVED_READ)
        return [Speaker.model_validate(s) for s in body.get("speakers", body.get("data", []))]

    def get_frame_url(self, video_id: str, t: float) -> str:
        body = self._request("GET", f"/videos/{video_id}/frame", params={"t": t})
        self._charge("frame", PRICE_DERIVED_READ)
        return str(body.get("url", ""))

    def get_clip_url(self, video_id: str, start: float, end: float) -> str:
        body = self._request("GET", f"/videos/{video_id}/clip", params={"start": start, "end": end})
        self._charge("clip", PRICE_DERIVED_READ)
        return str(body.get("url", body.get("clip_url", "")))


def _window(start: float | None, end: float | None) -> dict[str, Any] | None:
    params = {k: v for k, v in (("start", start), ("end", end)) if v is not None}
    return params or None


def _safe_json(response: httpx.Response) -> dict:
    try:
        body = response.json()
    except ValueError:
        return {}
    return body if isinstance(body, dict) else {}
