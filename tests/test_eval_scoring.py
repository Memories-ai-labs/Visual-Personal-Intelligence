"""The scorer has to be strict in the right direction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from vpi.agent.ledger import Citation
from vpi.agent.loop import Answer

spec = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).resolve().parents[1] / "eval" / "run_eval.py"
)
run_eval = importlib.util.module_from_spec(spec)
# Registering first is required: @dataclass resolves its module from sys.modules.
sys.modules["run_eval"] = run_eval
spec.loader.exec_module(run_eval)


def make_answer(text: str, citations: list[Citation] | None = None) -> Answer:
    from vpi.agent.cost import CostLedger

    return Answer(text=text, citations=citations or [], grounded=True, cost=CostLedger(), steps=2)


def citation(eid: str, video_id: str = "vid_a") -> Citation:
    return Citation(
        eid=eid, ref=f"{video_id}@1.0-2.0", video_id=video_id, start=1.0, end=2.0, text="t"
    )


def test_grounded_answer_passes():
    query = {
        "id": "q",
        "question": "?",
        "expect": "answer",
        "must_mention": ["200"],
        "must_cite_video": "vid_a",
    }
    row = run_eval.score(
        query, make_answer("Target was 200 ppm [E1].", [citation("E1")]), 1.0, 0.01
    )
    assert row.passed


def test_answer_without_citation_fails():
    query = {"id": "q", "question": "?", "expect": "answer", "must_mention": ["200"]}
    row = run_eval.score(query, make_answer("Target was 200 ppm."), 1.0, 0.01)
    assert not row.passed
    assert "no citation" in row.reason


def test_invented_citation_fails_even_when_content_is_right():
    query = {"id": "q", "question": "?", "expect": "answer", "must_mention": ["200"]}
    row = run_eval.score(
        query, make_answer("Target was 200 ppm [E7].", [citation("E1")]), 1.0, 0.01
    )
    assert not row.passed
    assert row.invalid_citations == ["E7"]


def test_citing_the_wrong_video_fails():
    query = {
        "id": "q",
        "question": "?",
        "expect": "answer",
        "must_mention": ["200"],
        "must_cite_video": "vid_b",
    }
    row = run_eval.score(query, make_answer("200 ppm [E1].", [citation("E1", "vid_a")]), 1.0, 0.01)
    assert not row.passed
    assert "wrong video" in row.reason


def test_refusal_passes_when_refusal_expected():
    query = {"id": "q", "question": "?", "expect": "not_found"}
    row = run_eval.score(query, make_answer("I couldn't find anything about that."), 1.0, 0.0)
    assert row.passed


def test_confident_answer_to_an_unanswerable_question_fails():
    query = {"id": "q", "question": "?", "expect": "not_found"}
    row = run_eval.score(
        query, make_answer("You were in Barcelona [E1].", [citation("E1")]), 1.0, 0.01
    )
    assert not row.passed
    assert "cannot support" in row.reason
