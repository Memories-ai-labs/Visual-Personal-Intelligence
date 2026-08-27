# Visual-Personal-Intelligence — design

**Date:** 2026-08-27
**Status:** approved, v0.1 implemented

## Origin

The `luci-memories` skill on ClawHub is a CLI wrapper: 16 query types over a
personal-memory HTTP surface, with its retrieval know-how written as prose in a
`SKILL.md` that only survives inside a Claude harness. The goal here was to turn
that into an open-source project: a personal AI chat plus a ReAct agent, built on
the video datalake, published under `Memories-ai-labs`.

## Decision: build on the DataLake API

Two candidate backends were evaluated against the live API.

| | Luci memory v2 (`mavi-backend/api/v2/luci-memory`) | **DataLake v1 (`api.memories.ai/datalake/v1`)** |
|---|---|---|
| Personal content endpoints | 10 live | ~41, none deprecated |
| Portrait (traits/events/relationships/speeches/facts) | **all 7 return `400 deprecated`** | n/a — people modelled as persons/entities/speakers |
| Citation format | assemble by hand from ids + timestamps | native moment ref `vid_x@start-end` |
| Evidence a user can inspect | `bucket` + `blob` only, no signed URL | expiring signed frame, thumbnail and **playable clip** |
| Rerank | none | cross-encoder flag (billed ×3) |
| Filtering | location, time | DSL: video_ids, tags, time, captured_at, location, speaker_id, event_type |
| Onboarding a stranger | needs an existing consumer account | `POST /videos` — bring your own footage |

The datalake won on every axis that matters for an open-source project. It is
also the company's platform surface, so a reference agent on it demonstrates the
product rather than a consumer app's private API.

## Decisions taken

1. **Deliverable:** one Python package with three entry points — terminal chat,
   FastAPI + SSE server, React chat UI. Same shape as `Internet2EgoExo` (uv,
   `src/` layout, `tests/`, `eval/`, MIT) so the two repos read as one org.
2. **LLM:** Claude by default (`claude-opus-5`, adaptive thinking, effort in
   `output_config`); any OpenAI-compatible endpoint as a documented fallback.
3. **Agent core:** ReAct loop + evidence ledger + relevance pass + grounding
   gate. Rejected: plan-and-execute (worse at follow-ups, more code) and thin
   tool-calling chat (reproduces exactly the failure modes the original
   `SKILL.md` warned about).
4. **Ingest included.** Without it a stranger cannot get past step one.
5. **People:** speakers yes (free with a moment expansion), persons/faces no —
   face recognition must be enabled at collection creation and defaulting an
   open-source tool to building a face library is the wrong posture.
6. **Images:** signed URLs, refreshed on demand by the server. No byte proxying
   and no cached links.
7. **Repo:** private first, flipped public after review.

## Architecture

See `docs/ARCHITECTURE.md` for the diagram, module boundaries, and the table of
API behaviours the implementation compensates for.

## Out of scope for v0.1

Persons/faces · safety events · live streams · image-to-frame search · an MCP
layer (one already ships with the datalake) · multi-user auth · a database ·
hosted deployment.

## How it is verified

* 90 unit tests, no network: fixture transports for both the datalake and the
  Anthropic SDK. The Claude tests assert the request body — adaptive thinking,
  `output_config.effort`, tool results returned in a single message, raw
  assistant content replayed verbatim.
* Demo mode (`VPI_DEMO=1`) swaps only the transport, so the demo path exercises
  the real client, tools, ledger and gate.
* A live smoke run against a real collection confirmed search → expand →
  grounded answer with a real moment ref, and surfaced one bug (request id is a
  header, not a body field) that fixtures had hidden.
* `eval/` scores grounding: half the query set is questions the corpus cannot
  answer, and those rows pass only on a refusal.
