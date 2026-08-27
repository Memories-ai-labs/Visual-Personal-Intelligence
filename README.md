# Visual-Personal-Intelligence

**Chat with your own video memory — and see the evidence for every sentence.**

A grounded ReAct agent over the [Memories.ai](https://memories.ai) video
datalake. You ask a question in your own words; the agent searches your indexed
recordings, pulls the full transcript and captions for what it found, throws
away what turned out to be irrelevant, and answers with a citation on every
claim. Each citation is a moment — `vid_x@601.0-606.5` — that you can play.

[![License: MIT](https://img.shields.io/badge/licence-MIT-0E0E10.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-0E0E10.svg)
![Status: early](https://img.shields.io/badge/status-early-6B6B70.svg)

## Why this exists

Video search returns clips. That is not the same as answering a question about
your own life, and the gap between the two is where these systems usually lie to
you: a plausible sentence with no recording behind it.

This project closes that gap mechanically rather than by asking a model to
behave. Everything retrieved goes into an evidence ledger with a short id. A
relevance pass prunes what the vector search dragged in by accident. Then a
**grounding gate** reads the drafted answer and cuts every sentence that does
not cite a real ledger entry. If nothing survives, you are told nothing was
found — which is a useful answer, and an honest one.

## What a question looks like

```
you › what did we decide about pricing, and who owns the follow-up?

  → resolve_timeframe(phrase=this week)
  ← 'this week' in Asia/Shanghai is 2026-08-23T16:00:00+00:00 to 2026-08-30T16:00:00+00:00 (UTC).
  → search_moments(query=pricing decision and who owns the follow-up, targets=['transcription', 'caption'])
  ← 3 moments (request req_demo_search; scores are only comparable inside this one request)  (+3 evidence)
  → get_moment(ref=vid_demo…standup@42.0-55.0, expand=['transcription', 'caption', 'clip'])
  ← vid_demo…standup@42.0-55.0 expanded into 3 evidence entries  (+3 evidence)
  · dropped 1 irrelevant entry: a slide description, not the decision

The tiered model is what is going into Thursday's call [E2]. You own the deck,
and you need the usage numbers by Wednesday morning [E3].

  [E2] vid_demo…standup 42.0-48.5s — the tiered model is what we're taking into the call on Thursday
  [E3] vid_demo…standup 48.5-55.0s — I'll own the deck, but I need the usage numbers by Wednesday

  $0.0210 total — $0.0210 datalake (1×get_moment, 1×search), 0 in / 0 out tokens
```

That is a real run against the bundled demo corpus, so you can reproduce it with
no key and no account. Two honest notes about it: the model was scripted for the
capture (everything else — tools, ledger, pruning, gate, costs — is the real
code path), so the token counts read zero, and a real model will word the answer
differently. What will not differ is the shape: an answer that cites, or an
answer that says it could not find anything.

## Try it with no key and no account

```bash
git clone https://github.com/Memories-ai-labs/Visual-Personal-Intelligence
cd Visual-Personal-Intelligence
uv venv && uv pip install -e ".[dev]"
export VPI_DEMO=1 ANTHROPIC_API_KEY=sk-ant-...
.venv/bin/vpi chat
```

Demo mode swaps the HTTP transport for bundled fixtures and changes nothing
else — the real client, tools, ledger and gate all run, so what you see is how
it behaves.

## Use it on your own videos

```bash
cp .env.example .env          # add MEMORIES_API_KEY from console.memories.ai
vpi ingest ~/Videos --name my-memory
vpi chat
```

`vpi ingest` accepts files, directories, or public direct URLs (a YouTube page
link will not work — the API needs a direct media URL). Indexing is asynchronous
and billed per minute; the command waits for `preprocess → index → derive` and
only reports a video ready when the operation says `done`.

| Command | What it does |
|---|---|
| `vpi collections` | List collections, video counts, account balance |
| `vpi ingest <paths…>` | Index videos, wait for them to become searchable |
| `vpi chat` | Terminal chat (`/evidence`, `/cost`, `/exit`) |
| `vpi ask "…"` | One question, then exit |
| `vpi serve` | Web chat UI + streaming API on localhost |

## How the loop works

| Stage | What happens | Why it is a separate stage |
|---|---|---|
| **Timeframe** | `resolve_timeframe` converts "last week" into explicit UTC bounds | The datalake filters in UTC and you speak local time; a model doing this in its head silently shifts whole days |
| **Locate** | `search_moments` — one short query, targets chosen by question type | `transcription` for what was said, `caption`/`frame_embedding` for what was visible, `summary`/`title` to find which video |
| **Verify** | An LLM pass drops evidence that is clearly about something else | Rerank *orders* a page; it does not remove near-misses |
| **Expand** | `get_moment` turns a truncated snippet into full text plus a playable clip | A search snippet is not evidence and must never be quoted |
| **Gate** | Uncited sentences trigger one repair pass, then get cut | The only mechanism that makes "no hallucination" true rather than requested |

## What it refuses to do

**Compare scores across searches.** The API computes `score` as cosine, ts_rank,
RRF or a sigmoid depending on which path served the query. Entries carry their
`request_id` and the ledger will not compare across them.

**Turn a speaker label into a person.** `SPEAKER_00` is a voice cluster from
diarisation, not an identity.

**Cache a signed URL.** Frames, clips and thumbnails expire (15 min to 5 h). The
browser asks for a fresh link at play time instead of storing one.

**Trust `progress.percent`.** When polling an ingest operation only `done` is
truth, and a non-null `error` means failure — including partial failure.

## Cost

Every turn prints what it spent. Datalake prices are per call: search $0.008
(×3 with `rerank`), `get_moment` $0.008, derived reads $0.001, indexing $0.05
per minute of video — the [published table](https://docs.memories.ai/datalake/pricing)
is the authority, and `src/vpi/datalake/client.py` is where these live if they
drift. LLM tokens are priced per model; an unrecognised model reports tokens and
no dollar figure rather than a wrong one.

## Privacy

This reads a record of your life, so: your key stays in `.env` and never leaves
your machine, the server binds to localhost and has no auth because it has no
multi-user model to protect, there is no telemetry, and nothing is written
anywhere except the collection you chose. `bucket`/`blob` internals never reach
the browser.

## Layout

```
src/vpi/
  datalake/   typed client for api.memories.ai/datalake/v1 — retries, errors, models
  ingest/     upload + operation polling
  tools/      the tools the model may call; their descriptions carry the discipline
  agent/      ReAct loop · evidence ledger · relevance pass · grounding gate · cost
  llm/        Claude by default, any OpenAI-compatible endpoint as fallback
  chat/       terminal chat
  server/     FastAPI: streaming chat, signed-URL refresh
ui/           React chat — streamed answer, citation cards, inline clips
eval/         golden questions scored on grounding, not vibes
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `MEMORIES_API_KEY` | — | DataLake key from [console.memories.ai](https://console.memories.ai) |
| `VPI_COLLECTION_ID` | auto | Which collection to chat with (required if you have several) |
| `VPI_LLM_MODEL` | `claude-opus-5` | Any current Claude model, or a model your endpoint serves |
| `VPI_LLM_BASE_URL` | — | Set to use an OpenAI-compatible endpoint instead of Claude |
| `VPI_TIMEZONE` | `UTC` | Your IANA timezone, for local↔UTC conversion |
| `VPI_MAX_STEPS` | `12` | Tool-calling steps before the loop gives up |
| `VPI_DEMO` | `0` | `1` runs on bundled fixtures, no key needed |

## Status

Early. The personal-memory half of the Memories.ai stack — face-recognised
people, safety events, live streams — is not wired up here yet; this v0.1 is
content retrieval done properly. Issues and PRs welcome.

MIT licensed.
