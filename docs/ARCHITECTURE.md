# Architecture

## The shape of it

```
question
  │
  ├─ resolve_timeframe ──────────── local phrase → explicit UTC bounds
  │
  ├─ search_moments ─────────────── one short query, targets by question type
  │       │
  │       └─→ EvidenceLedger        every hit gets an id (E1, E2, …)
  │
  ├─ relevance pass ─────────────── drops what the vector search dragged in
  │
  ├─ get_moment / get_transcription  snippet → full text + playable clip
  │       │
  │       └─→ EvidenceLedger
  │
  ├─ model drafts an answer
  │
  └─ grounding gate ─────────────── uncited sentence → one repair → else cut
          │
          └─→ answer + citations + cost
```

Three of those boxes are the reason this is not a wrapper.

**The ledger** is the only thing the answer is allowed to be about. Entries are
deduplicated on `(video, span, text)`, so the same moment arriving from a search
and again from an expansion does not become two facts.

**The relevance pass** runs *after* the tool round and *before* the model drafts,
so a dropped entry can never end up cited. It fails open: an unparseable
judgement keeps everything, because silently pruning to nothing turns a good
answer into a false "not found". It also refuses to drop every entry at once —
one judgement call should not empty the ledger.

**The gate** re-reads the draft. A sentence passes if it cites an id that exists.
Two exemptions: an admission that nothing was found, and a question back to the
user. Everything else uncited triggers one repair request; a second failure gets
cut. A citation of an id that is *not* in the ledger is treated as worse than no
citation — it is a fabricated source, so it is stripped and the sentence dropped.

## Module boundaries

| Module | Knows about | Does not know about |
|---|---|---|
| `datalake/` | HTTP, retries, prices, models | agents, prompts, LLMs |
| `tools/` | the datalake client, the ledger | which model is running, providers |
| `agent/` | tools, the ledger, an LLM interface | HTTP, any SDK |
| `llm/` | one provider each | the datalake, the ledger |
| `server/`, `chat/` | a `Session` | everything else |

`agent/` depends on `llm/base.py` types only, so swapping Claude for an
OpenAI-compatible endpoint touches one file and no logic.

## API behaviour worth knowing

Learned from the docs and from probing the live API — several of these are not
things the endpoint list tells you.

| Behaviour | What we do |
|---|---|
| `request_id` is returned in the `X-Request-ID` **header** on success, and in the body only on errors | Read the header; stamp it on every search hit so score comparability can be enforced |
| `score` scales differ by path (cosine / ts_rank / RRF / sigmoid) | Never compare across requests; `EvidenceLedger.comparable_with` refuses |
| `hybrid` mode does not paginate | Passing a cursor in hybrid mode raises rather than silently returning page one |
| Empty results carry a human-readable `hint` | Fed back to the model verbatim, as the docs suggest |
| Signed URLs expire: search thumbnail 15 min, clip 5 h, source 24 h | Nothing is cached; `/api/media` mints a fresh link at play time |
| Ingest returns `202` + an Operation; `progress.percent` is display only | Only `done` ends the wait; a non-null `error` is failure, including partial |
| `409 video_not_ready` carries `Retry-After` | Honoured by the client's retry path, then surfaced as `VideoNotReady` |
| Auth header is the raw key — no `Bearer` prefix | Set once in the client |
| `POST /videos` file mode is multipart with a `json` part plus a `file` part | `upload_video_file` builds exactly that |
| Platform page links (YouTube, TikTok) fail with `source_unresolvable` | Documented in the CLI; URL mode needs a direct media link |

## Costs, and why they are in the loop

Retrieval here is metered per call, so an agent that searches five times because
it stacked keywords costs five times as much as one that searched once well. The
cost ledger makes that visible per turn rather than at the end of the month:
search $0.008 (×3 with `rerank`), `get_moment` $0.008 (+$0.005 with a clip),
derived reads $0.001, indexing $0.04/minute.

LLM tokens are priced from a small table of known models. An unknown model
reports tokens with no dollar figure — a wrong number would be worse.

## What is not here yet

Face-recognised people (`/persons`), safety events, live stream sessions, and
image-to-frame search all exist in the datalake and are not wired up. Speakers
are read but deliberately never named: diarisation gives voice clusters, and
turning `SPEAKER_00` into a person is the kind of confident error this project
exists to prevent.
