#!/usr/bin/env python3
"""Score the agent on grounding rather than on how good the prose sounds.

    python eval/run_eval.py                  # demo fixtures
    python eval/run_eval.py --collection col_x --queries eval/mine.json

Four things are measured:

* **answered correctly** — the expected words appear, cited to the right video.
* **refused correctly** — a question the corpus cannot answer gets a refusal.
  This is the half that catches hallucination, and it is weighted equally.
* **citation validity** — every id in the answer exists in the ledger. A single
  invented id fails the row outright.
* **cost** — dollars and tool calls per question.

Needs an LLM key. The datalake half runs on fixtures unless you pass
--collection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from vpi.agent import grounding
from vpi.config import Settings
from vpi.session import build_session


@dataclass
class Row:
    id: str
    question: str
    expect: str
    passed: bool
    reason: str
    answer: str
    citations: list[str]
    invalid_citations: list[str]
    steps: int
    api_usd: float
    seconds: float


def refused(text: str) -> bool:
    """A refusal is a sentence the gate would let through with no citation."""
    return grounding.is_exempt(text.strip().split("\n")[0]) or text == grounding.NO_EVIDENCE_ANSWER


def score(query: dict, answer, elapsed: float, api_usd: float) -> Row:
    text = answer.text
    cited = [c.eid for c in answer.citations]
    invalid = sorted(grounding.citations_in(text) - set(cited))

    if invalid:
        return _row(
            query,
            False,
            f"cited ids that are not in the ledger: {invalid}",
            text,
            cited,
            invalid,
            answer,
            elapsed,
            api_usd,
        )

    if query["expect"] == "not_found":
        if refused(text) and not cited:
            return _row(
                query, True, "refused, as it should", text, cited, invalid, answer, elapsed, api_usd
            )
        return _row(
            query,
            False,
            "answered a question the corpus cannot support",
            text,
            cited,
            invalid,
            answer,
            elapsed,
            api_usd,
        )

    missing = [w for w in query.get("must_mention", []) if w.lower() not in text.lower()]
    if missing:
        return _row(
            query,
            False,
            f"missing expected content: {missing}",
            text,
            cited,
            invalid,
            answer,
            elapsed,
            api_usd,
        )
    if not cited:
        return _row(
            query,
            False,
            "answered with no citation",
            text,
            cited,
            invalid,
            answer,
            elapsed,
            api_usd,
        )

    wanted = query.get("must_cite_video")
    if wanted and not any(c.video_id == wanted for c in answer.citations):
        return _row(
            query,
            False,
            f"cited the wrong video (wanted {wanted})",
            text,
            cited,
            invalid,
            answer,
            elapsed,
            api_usd,
        )

    return _row(
        query, True, "answered and grounded", text, cited, invalid, answer, elapsed, api_usd
    )


def _row(query, passed, reason, text, cited, invalid, answer, elapsed, api_usd) -> Row:
    return Row(
        id=query["id"],
        question=query["question"],
        expect=query["expect"],
        passed=passed,
        reason=reason,
        answer=text,
        citations=cited,
        invalid_citations=invalid,
        steps=answer.steps,
        api_usd=round(api_usd, 4),
        seconds=round(elapsed, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="eval/queries.json")
    parser.add_argument("--collection", default="")
    parser.add_argument("--out", default="eval/results")
    args = parser.parse_args()

    spec = json.loads(Path(args.queries).read_text())
    if not args.collection:
        os.environ["VPI_DEMO"] = "1"

    rows: list[Row] = []
    for query in spec["queries"]:
        session = build_session(Settings(), collection_id=args.collection or None)
        started = time.monotonic()
        try:
            answer = session.agent.answer(query["question"])
        finally:
            elapsed = time.monotonic() - started
            api_usd = session.cost.api_usd
            session.close()
        row = score(query, answer, elapsed, api_usd)
        rows.append(row)
        mark = "PASS" if row.passed else "FAIL"
        print(f"[{mark}] {row.id}: {row.reason}")

    passed = sum(r.passed for r in rows)
    answered = [r for r in rows if r.expect == "answer"]
    refusals = [r for r in rows if r.expect == "not_found"]
    total_usd = sum(r.api_usd for r in rows)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "last.jsonl").write_text("\n".join(json.dumps(asdict(r)) for r in rows))
    card = [
        "# vpi grounding scorecard",
        "",
        f"- corpus: {args.collection or 'demo fixtures'}",
        f"- overall: **{passed}/{len(rows)}**",
        f"- answered correctly: {sum(r.passed for r in answered)}/{len(answered)}",
        f"- refused correctly: {sum(r.passed for r in refusals)}/{len(refusals)}",
        f"- invented citations: {sum(bool(r.invalid_citations) for r in rows)}",
        f"- datalake spend: ${total_usd:.4f}",
        "",
        "| id | expect | result | why |",
        "|---|---|---|---|",
        *[f"| {r.id} | {r.expect} | {'pass' if r.passed else 'fail'} | {r.reason} |" for r in rows],
    ]
    (out / "last.md").write_text("\n".join(card) + "\n")
    print(f"\n{passed}/{len(rows)} · ${total_usd:.4f} · wrote {out}/last.md")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
