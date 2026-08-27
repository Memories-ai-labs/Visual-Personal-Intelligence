"""The loop: tool calls, the repair pass, the relevance pass, giving up."""

from __future__ import annotations

from tests.conftest import ScriptedBackend
from vpi.agent.loop import Agent
from vpi.llm.base import LLMResponse, ToolCall
from vpi.tools import build_registry


def make_agent(responses, demo_client, ledger, **kwargs) -> Agent:
    from zoneinfo import ZoneInfo

    from vpi.demo import DEMO_COLLECTION

    registry = build_registry(demo_client, ledger, DEMO_COLLECTION, ZoneInfo("Asia/Shanghai"))
    return Agent(
        ScriptedBackend(responses),
        registry,
        ledger,
        timezone="Asia/Shanghai",
        verify_relevance=kwargs.pop("verify_relevance", False),
        **kwargs,
    )


def test_search_then_answer_with_a_citation(demo_client, ledger):
    agent = make_agent(
        [
            LLMResponse(
                text="",
                tool_calls=[ToolCall("t1", "search_moments", {"query": "sanitizer detergent mix"})],
            ),
            LLMResponse(text="You were mixing sanitiser to a target concentration [E1]."),
        ],
        demo_client,
        ledger,
    )
    answer = agent.answer("what was I saying about the sanitizer?")
    assert "[E1]" in answer.text
    assert answer.grounded
    assert answer.citations and answer.citations[0].eid == "E1"


def test_unsupported_sentence_triggers_one_repair_then_is_accepted(demo_client, ledger):
    agent = make_agent(
        [
            LLMResponse(
                text="", tool_calls=[ToolCall("t1", "search_moments", {"query": "sanitizer mix"})]
            ),
            LLMResponse(text="You mixed sanitiser [E1]. Afterwards you drove to the airport."),
            LLMResponse(text="You mixed sanitiser [E1]."),
        ],
        demo_client,
        ledger,
    )
    events = list(agent.ask("what happened?"))
    warnings = [e for e in events if e.kind == "warning"]
    answers = [e for e in events if e.kind == "answer"]
    assert any("unsupported" in w.text for w in warnings)
    assert "airport" not in answers[-1].text


def test_a_second_unsupported_answer_is_cut_not_repaired_again(demo_client, ledger):
    agent = make_agent(
        [
            LLMResponse(
                text="", tool_calls=[ToolCall("t1", "search_moments", {"query": "sanitizer"})]
            ),
            LLMResponse(text="You mixed sanitiser [E1]. You flew to Tokyo."),
            LLMResponse(text="You mixed sanitiser [E1]. You definitely flew to Tokyo."),
        ],
        demo_client,
        ledger,
    )
    answer = agent.answer("what happened?")
    assert "Tokyo" not in answer.text
    assert "[E1]" in answer.text
    assert answer.grounded is False


def test_fabricated_citation_never_reaches_the_user(demo_client, ledger):
    agent = make_agent(
        [
            LLMResponse(
                text="", tool_calls=[ToolCall("t1", "search_moments", {"query": "sanitizer"})]
            ),
            LLMResponse(text="You met Ana for lunch [E99]."),
            LLMResponse(text="You met Ana for lunch [E99]."),
        ],
        demo_client,
        ledger,
    )
    answer = agent.answer("did I have lunch with Ana?")
    assert "E99" not in answer.text
    assert "Ana" not in answer.text


def test_answer_with_no_tool_calls_and_no_evidence_says_not_found(demo_client, ledger):
    agent = make_agent(
        [LLMResponse(text="You had a great week."), LLMResponse(text="You had a great week.")],
        demo_client,
        ledger,
    )
    answer = agent.answer("how was my week?")
    assert "couldn't find" in answer.text
    assert answer.citations == []


def test_relevance_pass_drops_an_entry_before_answering(demo_client, ledger):
    agent = make_agent(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        "t1", "search_moments", {"query": "tiered pricing model Thursday call deck"}
                    )
                ],
            ),
            LLMResponse(text='{"drop": ["E2"], "reason": "a slide, not the decision"}'),
            LLMResponse(text="You were taking the tiered model into Thursday's call [E1]."),
        ],
        demo_client,
        ledger,
        verify_relevance=True,
        relevance_min_items=1,
    )
    events = list(agent.ask("what did we decide about pricing?"))
    notes = [e for e in events if e.kind == "note"]
    assert any("dropped 1 irrelevant" in n.text for n in notes)
    assert ledger.get("E2") is None


def test_relevance_pass_refuses_to_drop_everything(demo_client, ledger):
    from vpi.agent import relevance
    from vpi.datalake.models import SearchHit

    for i in range(4):
        ledger.add_hit(
            SearchHit(
                ref=f"vid_a@{i}-{i + 1}", video_id="vid_a", start=i, end=i + 1, snippet=f"t{i}"
            )
        )
    backend = ScriptedBackend([LLMResponse(text='{"drop": ["E1","E2","E3","E4"]}')])
    result = relevance.verify(backend, "anything?", ledger)
    assert result.failed
    assert len(ledger) == 4


def test_unparseable_relevance_reply_keeps_everything(demo_client, ledger):
    from vpi.agent import relevance
    from vpi.datalake.models import SearchHit

    for i in range(4):
        ledger.add_hit(
            SearchHit(
                ref=f"vid_a@{i}-{i + 1}", video_id="vid_a", start=i, end=i + 1, snippet=f"t{i}"
            )
        )
    result = relevance.verify(ScriptedBackend([LLMResponse(text="sure, looks fine")]), "q", ledger)
    assert result.failed
    assert len(ledger) == 4


def test_loop_gives_up_after_max_steps(demo_client, ledger):
    responses = [
        LLMResponse(text="", tool_calls=[ToolCall(f"t{i}", "search_moments", {"query": "again"})])
        for i in range(5)
    ]
    agent = make_agent(responses, demo_client, ledger, max_steps=3)
    events = list(agent.ask("loop forever?"))
    assert any("stopped after 3 steps" in e.text for e in events if e.kind == "warning")
    assert events[-1].kind == "done"


def test_evidence_is_offered_to_the_model_in_the_system_prompt(demo_client, ledger):
    agent = make_agent(
        [
            LLMResponse(
                text="", tool_calls=[ToolCall("t1", "search_moments", {"query": "sanitizer"})]
            ),
            LLMResponse(text="Mixing sanitiser [E1]."),
        ],
        demo_client,
        ledger,
    )
    agent.answer("what about the sanitizer?")
    second_system = agent.backend.calls[1]["system"]
    assert "Evidence gathered so far" in second_system
    assert "[E1]" in second_system


def test_tool_results_for_one_turn_go_back_as_a_single_turn(demo_client, ledger):
    agent = make_agent(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall("t1", "search_moments", {"query": "sanitizer"}),
                    ToolCall("t2", "list_videos", {}),
                ],
            ),
            LLMResponse(text="Mixing sanitiser [E1]."),
        ],
        demo_client,
        ledger,
    )
    agent.answer("what is in my memory?")
    tool_turns = [t for t in agent.turns if t.role == "tool"]
    assert len(tool_turns) == 1
    assert len(tool_turns[0].tool_results) == 2


def test_cost_is_reported_on_every_turn(demo_client, ledger, cost):
    agent = make_agent(
        [
            LLMResponse(
                text="", tool_calls=[ToolCall("t1", "search_moments", {"query": "sanitizer"})]
            ),
            LLMResponse(text="Mixing sanitiser [E1]."),
        ],
        demo_client,
        ledger,
        cost=cost,
    )
    events = list(agent.ask("what about the sanitizer?"))
    cost_events = [e for e in events if e.kind == "cost"]
    assert cost_events
    assert cost.calls["search"] == 1
    assert cost.api_usd > 0


def test_relevance_tokens_are_priced_like_any_other_call(demo_client, ledger, cost):
    """A turn that runs the relevance pass must still report a dollar total."""
    agent = make_agent(
        [
            LLMResponse(
                text="",
                tool_calls=[
                    ToolCall(
                        "t1", "search_moments", {"query": "tiered pricing model Thursday call deck"}
                    )
                ],
            ),
            LLMResponse(text='{"drop": []}'),
            LLMResponse(text="You were taking the tiered model into Thursday's call [E1]."),
        ],
        demo_client,
        ledger,
        cost=cost,
        verify_relevance=True,
        relevance_min_items=1,
    )
    agent.answer("what did we decide about pricing?")
    assert cost.total_usd is not None
    assert "model price unknown" not in cost.summary()
