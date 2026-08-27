"""Demo mode.

`VPI_DEMO=1` swaps the HTTP transport for a fixture one and changes nothing
else — the real client, tools, ledger and gate all run. That means the demo
exercises the same code path as a live key, so a bug that only shows up in demo
mode is a real bug.
"""

from __future__ import annotations

import json
from typing import Any

import httpx

DEMO_COLLECTION = "col_demo00000000000000000000"
_V1 = "vid_demo0000000000000000kitchen"
_V2 = "vid_demo0000000000000000standup"

VIDEOS: dict[str, dict[str, Any]] = {
    _V1: {
        "id": _V1,
        "collection_id": DEMO_COLLECTION,
        "status": "ready",
        "duration": 900.0,
        "captured_at": "2026-08-24T02:15:00Z",
        "metadata": {"title": "kitchen-prep.mp4", "tags": ["home"]},
        "summary": "A long stretch of kitchen work: sanitising a bucket, mixing detergent, "
        "then prepping vegetables while talking to someone off camera.",
        "title": "Kitchen prep and sanitiser mix",
        "transcription": [
            {
                "start": 597.1,
                "end": 601.3,
                "text": "to put some of that detergent in my sanitizer here so fortunately I have",
            },
            {
                "start": 601.0,
                "end": 606.5,
                "text": "Craig here who's helping me, his hands are clean so he's the one measuring",
            },
            {
                "start": 606.5,
                "end": 612.0,
                "text": "we're going for two hundred parts per million, that's the target",
            },
        ],
        "caption": [
            {
                "start": 577.0,
                "end": 602.0,
                "text": "A white bucket sits in a stainless steel sink, partly filled with murky water. A person in blue gloves sprays it down.",
            },
            {
                "start": 602.0,
                "end": 615.0,
                "text": "Two pairs of hands work over the sink; one holds a measuring cup of pale liquid.",
            },
        ],
    },
    _V2: {
        "id": _V2,
        "collection_id": DEMO_COLLECTION,
        "status": "ready",
        "duration": 420.0,
        "captured_at": "2026-08-25T09:02:00Z",
        "metadata": {"title": "standup-monday.mp4", "tags": ["work"]},
        "summary": "A short team stand-up over video call about a pricing decision and who "
        "owns the follow-up.",
        "title": "Monday stand-up — pricing follow-up",
        "transcription": [
            {
                "start": 42.0,
                "end": 48.5,
                "text": "so the tiered model is what we're taking into the call on Thursday",
                "speaker_id": None,
            },
            {
                "start": 48.5,
                "end": 55.0,
                "text": "I'll own the deck, but I need the usage numbers from you by Wednesday morning",
                "speaker_id": None,
            },
        ],
        "caption": [
            {
                "start": 40.0,
                "end": 60.0,
                "text": "A video-call grid with four participants; a slide showing three pricing tiers is shared.",
            },
        ],
    },
}

_SEARCH_INDEX = [
    (
        _V1,
        "transcription",
        597.1,
        601.3,
        "to put some of that detergent in my sanitizer here so fortunately I have",
    ),
    (_V1, "transcription", 601.0, 606.5, "Craig here who's helping me, his hands are clean"),
    (_V1, "caption", 577.0, 602.0, "A white bucket sits in a stainless steel sink"),
    (_V1, "transcription", 606.5, 612.0, "we're going for two hundred parts per million"),
    (
        _V2,
        "transcription",
        42.0,
        48.5,
        "the tiered model is what we're taking into the call on Thursday",
    ),
    (
        _V2,
        "caption",
        40.0,
        60.0,
        "A video-call grid with four participants; a pricing slide is shared",
    ),
    (
        _V2,
        "transcription",
        48.5,
        55.0,
        "I'll own the deck, but I need the usage numbers by Wednesday",
    ),
]


_BASELINE_SCORE = 0.2


def _score(query: str, text: str) -> float:
    """Crude word overlap. Enough to make the demo respond to the question asked."""
    words = {w.strip(".,!?").lower() for w in query.split() if len(w) > 3}
    if not words:
        return 0.1
    hit = {w.strip(".,!?").lower() for w in text.split()}
    return round(0.2 + 0.6 * len(words & hit) / len(words), 4)


def _json(payload: Any, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload, headers={"X-Request-ID": "req_demo"})


def handler(request: httpx.Request) -> httpx.Response:  # noqa: C901 - a router is a router
    path = request.url.path.replace("/datalake/v1", "", 1)
    params = dict(request.url.params)

    if path == "/collections":
        return _json(
            {
                "collections": [
                    {
                        "id": DEMO_COLLECTION,
                        "name": "vpi-demo",
                        "video_count": len(VIDEOS),
                        "face_recognition_enabled": False,
                        "enabled_detectors": [],
                    }
                ]
            }
        )

    if path == "/usage/balance":
        return _json({"currency": "USD", "balance_usd": 0.0, "as_of": "2026-08-27T00:00:00Z"})

    if path == "/videos":
        return _json(
            {
                "videos": [
                    {
                        k: v
                        for k, v in video.items()
                        if k not in ("summary", "title", "transcription", "caption")
                    }
                    for video in VIDEOS.values()
                ],
                "next_cursor": None,
            }
        )

    if path == "/search":
        body = json.loads(request.content or b"{}")
        query = str(body.get("query", ""))
        targets = set(body.get("targets") or ["caption", "transcription"])
        top_k = int(body.get("top_k") or 12)
        results = [
            {
                "ref": f"{vid}@{start:.1f}-{end:.1f}",
                "video_id": vid,
                "target": target,
                "score": _score(query, text),
                "start": start,
                "end": end,
                "snippet": text,
                "thumbnail_url": "",
            }
            for vid, target, start, end, text in _SEARCH_INDEX
            if target in targets
        ]
        results.sort(key=lambda r: r["score"], reverse=True)
        # Like real semantic search, any overlap comes back and the agent judges it;
        # a query that overlaps nothing returns nothing, plus a hint.
        results = [r for r in results if r["score"] > _BASELINE_SCORE][:top_k]
        return _json(
            {
                "results": results,
                "next_cursor": None,
                "request_id": "req_demo_search",
                "hint": ""
                if results
                else "No indexed moment matches those words. This demo collection holds one "
                "kitchen-prep video and one stand-up about pricing.",
            }
        )

    if path.startswith("/moments/"):
        ref = path.removeprefix("/moments/")
        video_id, _, span = ref.partition("@")
        video = VIDEOS.get(video_id)
        if video is None:
            return _json(
                {"error": {"code": "not_found", "message": f"no such video {video_id}"}}, 404
            )
        start, end = (0.0, float(video["duration"]))
        if span and "-" in span:
            start, end = (float(x) for x in span.split("-", 1))
        expand = set((params.get("expand") or "").split(","))
        payload: dict[str, Any] = {"ref": ref, "video_id": video_id, "start": start, "end": end}
        if "transcription" in expand:
            payload["transcription"] = _overlap(video["transcription"], start, end)
        if "caption" in expand:
            payload["caption"] = _overlap(video["caption"], start, end)
        if "clip" in expand:
            payload["clip_url"] = ""
        if "speakers" in expand:
            payload["speakers"] = [{"label": "SPEAKER_00", "segments": 3, "speaker_id": None}]
        return _json(payload)

    for key in ("transcription", "caption"):
        if path.endswith(f"/{key}"):
            video_id = path.removeprefix("/videos/").removesuffix(f"/{key}")
            video = VIDEOS.get(video_id)
            if video is None:
                return _json({"error": {"code": "not_found", "message": video_id}}, 404)
            start = float(params["start"]) if "start" in params else 0.0
            end = float(params["end"]) if "end" in params else float(video["duration"])
            return _json({"segments": _overlap(video[key], start, end)})

    for key in ("summary", "title"):
        if path.endswith(f"/{key}"):
            video_id = path.removeprefix("/videos/").removesuffix(f"/{key}")
            video = VIDEOS.get(video_id)
            if video is None:
                return _json({"error": {"code": "not_found", "message": video_id}}, 404)
            return _json({key: video[key]})

    if path.endswith("/speakers"):
        return _json({"speakers": [{"label": "SPEAKER_00", "segments": 3, "speaker_id": None}]})

    if path.startswith("/videos/"):
        video_id = path.removeprefix("/videos/")
        video = VIDEOS.get(video_id)
        if video is None:
            return _json({"error": {"code": "not_found", "message": video_id}}, 404)
        return _json(video)

    return _json(
        {"error": {"code": "not_found", "message": f"demo mode does not serve {path}"}}, 404
    )


def _overlap(segments: list[dict], start: float, end: float) -> list[dict]:
    return [s for s in segments if s["end"] > start and s["start"] < end]


def demo_transport() -> httpx.MockTransport:
    return httpx.MockTransport(handler)
