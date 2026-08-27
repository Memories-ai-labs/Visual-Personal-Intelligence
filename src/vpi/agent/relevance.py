"""The relevance pass.

Search returns what is near in vector space, which is not the same as what the
user asked about. The API can rerank a page, but reranking orders hits — it does
not throw away the ones that are simply about something else. This pass does
that, once, before the answer is written.

It fails open: if the check cannot be parsed, everything is kept. Dropping
evidence on a parse error would silently turn a good answer into "I couldn't
find it", and the grounding gate still stands between the model and a claim.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from vpi.agent.ledger import EvidenceLedger
from vpi.llm.base import LLMBackend, Turn, Usage

PROMPT = """\
A user asked: {question}

Below are retrieved moments from their video memory. Some are relevant; vector \
search also returns near-misses about something else entirely.

For each entry decide whether it could support part of an answer to the question. \
Keep anything plausibly useful — you are removing the clearly-unrelated, not \
picking favourites.

{entries}

Reply with JSON only: {{"drop": ["E2", "E7"], "reason": "one short line"}}
Use an empty list if everything is relevant.
"""

MAX_TEXT = 300


@dataclass
class RelevanceResult:
    dropped: set[str] = field(default_factory=set)
    reason: str = ""
    usage: Usage = field(default_factory=Usage)
    failed: bool = False


def verify(
    backend: LLMBackend,
    question: str,
    ledger: EvidenceLedger,
    *,
    min_items: int = 4,
) -> RelevanceResult:
    """Drop clearly-irrelevant evidence from the ledger in place."""
    items = ledger.items
    if len(items) < min_items:
        return RelevanceResult()

    entries = "\n".join(
        f"[{e.eid}] {e.target or 'moment'} {e.ref}: {e.text[:MAX_TEXT]}" for e in items
    )
    prompt = PROMPT.format(question=question, entries=entries)

    try:
        response = backend.complete(
            system="You judge retrieval relevance. You reply with JSON and nothing else.",
            turns=[Turn(role="user", text=prompt)],
            max_tokens=2000,
        )
    except Exception:  # noqa: BLE001 - a failed check must not fail the turn
        return RelevanceResult(failed=True)

    parsed = _parse(response.text)
    if parsed is None:
        return RelevanceResult(usage=response.usage, failed=True)

    dropped = {eid for eid in parsed.get("drop", []) if eid in ledger.ids}
    # Never drop everything on the strength of one judgement call.
    if dropped and len(dropped) == len(items):
        return RelevanceResult(
            usage=response.usage,
            reason="relevance pass wanted to drop every entry; kept them all",
            failed=True,
        )
    ledger.drop(dropped)
    return RelevanceResult(
        dropped=dropped, reason=str(parsed.get("reason", "")), usage=response.usage
    )


def _parse(text: str) -> dict | None:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict) or not isinstance(parsed.get("drop", []), list):
        return None
    return parsed
