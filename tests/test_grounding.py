"""The gate is the honesty mechanism, so it gets the most tests."""

from __future__ import annotations

from vpi.agent import grounding


def test_sentence_with_valid_citation_survives():
    result = grounding.enforce("You mixed sanitiser to 200 ppm [E4].", {"E4"})
    assert result.text == "You mixed sanitiser to 200 ppm [E4]."
    assert result.cited_ids == {"E4"}


def test_uncited_sentence_is_cut():
    text = "You mixed sanitiser [E4]. You then drove to the airport."
    result = grounding.enforce(text, {"E4"})
    assert "airport" not in result.text
    assert "[E4]" in result.text
    assert result.grounded is False


def test_fabricated_id_is_stripped_and_sentence_cut():
    result = grounding.enforce("You flew to Tokyo [E9].", {"E1"})
    assert "E9" not in result.text
    assert result.text == grounding.NO_EVIDENCE_ANSWER
    assert result.invalid_ids == {"E9"}


def test_no_evidence_statement_needs_no_citation():
    text = "I couldn't find anything about that in your videos."
    result = grounding.enforce(text, set())
    assert result.text == text


def test_chinese_no_evidence_statement_is_exempt():
    result = grounding.enforce("视频里没有找到相关内容。", set())
    assert "没有找到" in result.text


def test_question_back_to_user_is_exempt():
    result = grounding.enforce("Which week did you mean?", set())
    assert result.text == "Which week did you mean?"


def test_everything_unsupported_falls_back_to_not_found():
    result = grounding.enforce("You had lunch with Ana. It was raining.", {"E1"})
    assert result.text == grounding.NO_EVIDENCE_ANSWER


def test_check_reports_without_mutating():
    text = "A [E1]. B."
    result = grounding.check(text, {"E1"})
    assert result.text == text
    assert result.dropped == ["B."]
    assert result.needs_repair


def test_repair_prompt_names_the_unsupported_sentences():
    result = grounding.check("A [E1]. The meeting ran late.", {"E1"})
    prompt = grounding.repair_prompt(result)
    assert "The meeting ran late." in prompt
    assert "Delete any claim you cannot cite" in prompt


def test_repair_prompt_calls_out_invented_ids():
    result = grounding.check("Something happened [E42].", {"E1"})
    assert "E42" in grounding.repair_prompt(result)


def test_multiple_citations_in_one_sentence():
    result = grounding.enforce("Two things happened [E1][E2].", {"E1", "E2"})
    assert result.cited_ids == {"E1", "E2"}
    assert result.grounded
