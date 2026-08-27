"""The datalake tools.

The descriptions here are load-bearing. They carry the retrieval discipline the
agent has to follow — search wide then expand narrow, one coherent query per
search, never compare scores across searches, feed an empty result's `hint`
back into the next attempt. A tool description is the only place that guidance
survives being lifted into someone else's harness.
"""

from __future__ import annotations

from vpi.agent.ledger import EvidenceLedger
from vpi.datalake.client import SEARCH_TARGETS, DataLakeClient
from vpi.datalake.errors import DataLakeError, VideoNotReady
from vpi.tools.registry import Tool, ToolOutcome

MAX_SNIPPET = 400


def build_datalake_tools(
    client: DataLakeClient, ledger: EvidenceLedger, collection_id: str
) -> list[Tool]:
    def _search(
        query: str,
        targets: list[str] | None = None,
        mode: str = "semantic",
        top_k: int = 12,
        rerank: bool = False,
        filter: dict | None = None,  # noqa: A002 - mirrors the API field name
        group_by: str = "moment",
    ) -> ToolOutcome:
        try:
            page = client.search(
                collection_id,
                query,
                targets=targets or ["caption", "transcription"],
                mode=mode,
                top_k=top_k,
                rerank=rerank,
                filter=filter,
                group_by=group_by,
            )
        except DataLakeError as exc:
            return ToolOutcome(f"Search failed: {exc}", is_error=True)

        if not page.results:
            hint = page.hint or "No hint returned."
            return ToolOutcome(
                f"No moments matched. The API's hint: {hint}\n"
                "Try a different phrasing, a wider time range, or another target "
                "(transcription for spoken words, caption for what is visible)."
            )

        added = [e for e in (ledger.add_hit(h) for h in page.results) if e]
        lines = [
            f"{len(page.results)} moments (request {page.request_id or 'n/a'}; "
            "scores are only comparable inside this one request):"
        ]
        for evidence in added:
            lines.append(
                f"[{evidence.eid}] {evidence.ref} {evidence.target} score={evidence.score:.3f}\n"
                f"    {evidence.text[:MAX_SNIPPET]}"
            )
        if not added:
            lines.append("(all hits were already in the evidence list)")
        lines.append(
            "Snippets are truncated. Expand the promising refs with get_moment before "
            "answering from them."
        )
        return ToolOutcome("\n".join(lines), evidence=added)

    def _get_moment(ref: str, expand: list[str] | None = None) -> ToolOutcome:
        expand = expand or ["caption", "transcription", "clip"]
        try:
            moment = client.get_moment(ref, expand=expand)
        except VideoNotReady as exc:
            return ToolOutcome(
                f"That video is still indexing ({exc}). Retry in a moment or pick another.",
                is_error=True,
            )
        except DataLakeError as exc:
            return ToolOutcome(f"get_moment failed: {exc}", is_error=True)

        added = ledger.add_moment(moment)
        if not added:
            return ToolOutcome(
                f"{ref} expanded but produced no new text (already in the evidence list, "
                "or the slice has no caption/transcription)."
            )
        body = "\n".join(e.render() for e in added)
        note = ""
        if moment.clip_url:
            note = "\nA playable clip is attached to these entries for the user interface."
        return ToolOutcome(
            f"{ref} expanded into {len(added)} evidence entries:\n{body}{note}", evidence=added
        )

    def _get_transcription(
        video_id: str, start: float | None = None, end: float | None = None
    ) -> ToolOutcome:
        try:
            segments = client.get_transcription(video_id, start=start, end=end)
        except DataLakeError as exc:
            return ToolOutcome(f"transcription failed: {exc}", is_error=True)
        added = ledger.add_segments(
            video_id, segments, target="transcription", source="get_transcription"
        )
        if not added:
            return ToolOutcome(f"No speech transcribed for {video_id} in that window.")
        return ToolOutcome(
            f"{len(added)} transcription segments for {video_id}:\n"
            + "\n".join(e.render() for e in added),
            evidence=added,
        )

    def _get_caption(
        video_id: str, start: float | None = None, end: float | None = None
    ) -> ToolOutcome:
        try:
            segments = client.get_caption(video_id, start=start, end=end)
        except DataLakeError as exc:
            return ToolOutcome(f"caption failed: {exc}", is_error=True)
        added = ledger.add_segments(video_id, segments, target="caption", source="get_caption")
        if not added:
            return ToolOutcome(f"No visual captions for {video_id} in that window.")
        return ToolOutcome(
            f"{len(added)} caption segments for {video_id}:\n"
            + "\n".join(e.render() for e in added),
            evidence=added,
        )

    def _get_summary(video_id: str) -> ToolOutcome:
        try:
            summary = client.get_summary(video_id)
            title = client.get_title(video_id)
        except DataLakeError as exc:
            return ToolOutcome(f"summary failed: {exc}", is_error=True)
        if not summary and not title:
            return ToolOutcome(f"No summary or title generated for {video_id} yet.")
        evidence = ledger.add_text(
            video_id, f"{title}\n{summary}".strip(), target="summary", source="get_summary"
        )
        return ToolOutcome(
            f"{video_id} — {title}\n{summary}",
            evidence=[evidence] if evidence else [],
        )

    def _list_videos(status: str | None = None, limit: int = 20) -> ToolOutcome:
        try:
            videos, _ = client.list_videos(collection_id, status=status, limit=limit)
        except DataLakeError as exc:
            return ToolOutcome(f"list_videos failed: {exc}", is_error=True)
        if not videos:
            return ToolOutcome("This collection has no videos matching that filter.")
        lines = [
            f"- {v.id} [{v.status}] {v.title}"
            + (f" captured {v.captured_at.isoformat()}" if v.captured_at else "")
            for v in videos
        ]
        return ToolOutcome(f"{len(videos)} videos in {collection_id}:\n" + "\n".join(lines))

    def _get_speakers(video_id: str) -> ToolOutcome:
        try:
            speakers = client.get_speakers(video_id)
        except DataLakeError as exc:
            return ToolOutcome(f"speakers failed: {exc}", is_error=True)
        if not speakers:
            return ToolOutcome(f"No diarised speakers for {video_id}.")
        lines = [
            f"- {s.label} ({s.segments} segments, id={s.speaker_id or 'unnamed'})" for s in speakers
        ]
        return ToolOutcome(
            f"{len(speakers)} speakers in {video_id}. Labels are diarisation only — they are "
            "not names, so do not claim who they are:\n" + "\n".join(lines)
        )

    return [
        Tool(
            name="search_moments",
            description=(
                "Semantic / keyword / hybrid search over the collection. Returns moments "
                "(`vid_x@start-end`) with truncated snippets. Targets: "
                f"{', '.join(SEARCH_TARGETS)} "
                "— use `transcription` for what was said, `caption` or `frame_embedding` for "
                "what was visible, `summary`/`title` to find which video something is in. "
                "Write ONE short natural-language query per call; stacking keywords makes "
                "results worse. Set rerank=true when the question is precise and you want the "
                "best page ordering (it costs 3× a plain search). `filter` takes the API DSL, "
                'e.g. {"captured_at": {"gte": "...", "lt": "..."}} or '
                '{"video_ids": ["vid_x"]}. Scores are NOT comparable between calls. In '
                "hybrid mode there is no pagination — raise top_k instead."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "One short natural-language query."},
                    "targets": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(SEARCH_TARGETS)},
                        "description": "What to search. Default caption + transcription.",
                    },
                    "mode": {"type": "string", "enum": ["semantic", "keyword", "hybrid"]},
                    "top_k": {"type": "integer", "description": "1-200, default 12."},
                    "rerank": {"type": "boolean", "description": "Cross-encoder rerank, costs 3×."},
                    "filter": {
                        "type": "object",
                        "description": (
                            "Filter DSL: video_ids, tags, time, captured_at, "
                            "location, speaker_id, event_type."
                        ),
                    },
                    "group_by": {"type": "string", "enum": ["moment", "video"]},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
            run=_search,
        ),
        Tool(
            name="get_moment",
            description=(
                "Expand one moment ref into full evidence — caption, transcription, and a "
                "playable clip. This is how you turn a search snippet into something you may "
                "quote. Always expand before answering from a hit."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "`vid_x@start-end`, or `vid_x` for a whole video.",
                    },
                    "expand": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["caption", "transcription", "frame", "clip", "speakers"],
                        },
                    },
                },
                "required": ["ref"],
                "additionalProperties": False,
            },
            run=_get_moment,
        ),
        Tool(
            name="get_transcription",
            description=(
                "Every spoken word in a video, optionally windowed. Use when the question is "
                "about a conversation and you need more than the matched slice."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                },
                "required": ["video_id"],
                "additionalProperties": False,
            },
            run=_get_transcription,
        ),
        Tool(
            name="get_caption",
            description="Visual captions — what is on screen — for a video, optionally windowed.",
            input_schema={
                "type": "object",
                "properties": {
                    "video_id": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                },
                "required": ["video_id"],
                "additionalProperties": False,
            },
            run=_get_caption,
        ),
        Tool(
            name="get_summary",
            description=(
                "The AI title and summary of a whole video. Cheap orientation — use it to "
                "decide which video to dig into, not as evidence for a specific claim."
            ),
            input_schema={
                "type": "object",
                "properties": {"video_id": {"type": "string"}},
                "required": ["video_id"],
                "additionalProperties": False,
            },
            run=_get_summary,
        ),
        Tool(
            name="list_videos",
            description=(
                "List videos in the collection with their indexing status. Use when the user "
                "asks what is in their memory, or when a search returns nothing and you need "
                "to check whether anything is indexed at all."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "status": {"type": "string", "enum": ["processing", "ready", "failed"]},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
            run=_list_videos,
        ),
        Tool(
            name="get_speakers",
            description=(
                "Diarised speakers for a video. Labels like SPEAKER_00 are voice clusters, "
                "not identities — never turn a label into a person's name."
            ),
            input_schema={
                "type": "object",
                "properties": {"video_id": {"type": "string"}},
                "required": ["video_id"],
                "additionalProperties": False,
            },
            run=_get_speakers,
        ),
    ]
