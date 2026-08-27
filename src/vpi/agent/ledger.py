"""The evidence ledger.

Every fact the agent is allowed to state has to live here first. Each entry gets
a short id (`E1`, `E2`, …) that the model cites and the grounding gate checks.

One rule is enforced structurally rather than left to the prompt: **scores from
different search requests are not comparable.** The API computes `score` as
cosine, ts_rank, RRF or a sigmoid depending on which path served the query, so
ranking a hit from request A above one from request B is meaningless. Entries
carry their `request_id` and `comparable_with` refuses the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vpi.datalake.models import Moment, SearchHit, Segment


@dataclass
class Evidence:
    eid: str
    ref: str
    video_id: str
    start: float
    end: float
    text: str
    source: str
    target: str = ""
    score: float = 0.0
    request_id: str = ""
    thumbnail_url: str = ""
    clip_url: str = ""

    @property
    def span(self) -> str:
        return f"{self.start:.1f}-{self.end:.1f}s"

    def render(self) -> str:
        head = f"[{self.eid}] {self.ref} ({self.span}"
        if self.target:
            head += f", {self.target}"
        head += ")"
        body = self.text.strip() or "(no text)"
        return f"{head}\n    {body}"


def _dedupe_key(video_id: str, start: float, end: float, text: str) -> tuple:
    return (video_id, round(start, 1), round(end, 1), text.strip()[:120])


class EvidenceLedger:
    """Append-only, deduplicated, citable."""

    def __init__(self) -> None:
        self._items: list[Evidence] = []
        self._seen: set[tuple] = set()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(self._items)

    @property
    def items(self) -> list[Evidence]:
        return list(self._items)

    def get(self, eid: str) -> Evidence | None:
        return next((e for e in self._items if e.eid == eid), None)

    @property
    def ids(self) -> set[str]:
        return {e.eid for e in self._items}

    def _next_id(self) -> str:
        return f"E{len(self._items) + 1}"

    def _add(self, evidence: Evidence) -> Evidence | None:
        key = _dedupe_key(evidence.video_id, evidence.start, evidence.end, evidence.text)
        if key in self._seen:
            return None
        self._seen.add(key)
        self._items.append(evidence)
        return evidence

    def add_hit(self, hit: SearchHit, *, source: str = "search_moments") -> Evidence | None:
        return self._add(
            Evidence(
                eid=self._next_id(),
                ref=hit.ref,
                video_id=hit.video_id,
                start=hit.start,
                end=hit.end,
                text=hit.snippet,
                source=source,
                target=hit.target,
                score=hit.score,
                request_id=hit.request_id,
                thumbnail_url=hit.thumbnail_url,
            )
        )

    def add_segments(
        self,
        video_id: str,
        segments: list[Segment],
        *,
        target: str,
        source: str,
        clip_url: str = "",
    ) -> list[Evidence]:
        added = []
        for seg in segments:
            if not seg.text.strip():
                continue
            evidence = self._add(
                Evidence(
                    eid=self._next_id(),
                    ref=f"{video_id}@{seg.start:.1f}-{seg.end:.1f}",
                    video_id=video_id,
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    source=source,
                    target=target,
                    clip_url=clip_url,
                )
            )
            if evidence:
                added.append(evidence)
        return added

    def add_moment(self, moment: Moment, *, source: str = "get_moment") -> list[Evidence]:
        added: list[Evidence] = []
        added += self.add_segments(
            moment.video_id,
            moment.transcription,
            target="transcription",
            source=source,
            clip_url=moment.clip_url,
        )
        added += self.add_segments(
            moment.video_id,
            moment.caption,
            target="caption",
            source=source,
            clip_url=moment.clip_url,
        )
        return added

    def add_text(self, video_id: str, text: str, *, target: str, source: str) -> Evidence | None:
        return self._add(
            Evidence(
                eid=self._next_id(),
                ref=video_id,
                video_id=video_id,
                start=0.0,
                end=0.0,
                text=text,
                source=source,
                target=target,
            )
        )

    def drop(self, eids: set[str]) -> None:
        """Remove entries a relevance check rejected. Ids of survivors do not shift."""
        self._items = [e for e in self._items if e.eid not in eids]

    def comparable_with(self, a: Evidence, b: Evidence) -> bool:
        """Two scores may only be compared inside one search request."""
        return bool(a.request_id) and a.request_id == b.request_id

    def render(self, *, limit: int | None = None) -> str:
        items = self._items if limit is None else self._items[:limit]
        if not items:
            return "(no evidence gathered yet)"
        return "\n".join(e.render() for e in items)

    def cited(self, eids: set[str]) -> list[Evidence]:
        return [e for e in self._items if e.eid in eids]


@dataclass
class Citation:
    """What the UI needs to show one piece of evidence."""

    eid: str
    ref: str
    video_id: str
    start: float
    end: float
    text: str
    thumbnail_url: str = ""
    clip_url: str = ""
    tags: list[str] = field(default_factory=list)

    @classmethod
    def of(cls, evidence: Evidence) -> Citation:
        return cls(
            eid=evidence.eid,
            ref=evidence.ref,
            video_id=evidence.video_id,
            start=evidence.start,
            end=evidence.end,
            text=evidence.text,
            thumbnail_url=evidence.thumbnail_url,
            clip_url=evidence.clip_url,
            tags=[t for t in (evidence.target, evidence.source) if t],
        )
