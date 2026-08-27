"""What we actually put on the wire to Claude.

No key needed: the SDK's transport is mocked, and the assertions are about the
request body. These are the details that are wrong by default — `budget_tokens`
instead of adaptive thinking, effort at the top level instead of inside
`output_config`, tool results split across messages.

Note the transport comes from `httpx2`: the anthropic 1.x SDK is built on it,
and an `httpx` object is rejected at request time.
"""

from __future__ import annotations

import json

import httpx2
import pytest

from vpi.llm.anthropic_backend import AnthropicBackend
from vpi.llm.base import ToolCall, ToolResult, ToolSpec, Turn, Usage

TOOL = ToolSpec(
    name="search_moments",
    description="search",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
)


def backend_with(handler) -> AnthropicBackend:
    import anthropic

    backend = AnthropicBackend("claude-opus-5", api_key="sk-ant-test")
    backend._client = anthropic.Anthropic(
        api_key="sk-ant-test",
        http_client=httpx2.Client(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    return backend


def reply(content: list[dict], stop_reason: str = "end_turn") -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5",
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 120, "output_tokens": 34},
    }


@pytest.fixture
def captured() -> dict:
    return {}


def test_request_uses_adaptive_thinking_and_effort_not_budget_tokens(captured):
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=reply([{"type": "text", "text": "hi"}]))

    backend_with(handler).complete(system="s", turns=[Turn(role="user", text="hello")])

    body = captured["body"]
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "high"}
    assert "budget_tokens" not in json.dumps(body)
    assert body["model"] == "claude-opus-5"


def test_tools_are_sent_in_anthropic_shape(captured):
    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=reply([{"type": "text", "text": "hi"}]))

    backend_with(handler).complete(system="s", turns=[Turn(role="user", text="q")], tools=[TOOL])

    tool = captured["body"]["tools"][0]
    assert tool["name"] == "search_moments"
    assert tool["input_schema"]["type"] == "object"


def test_tool_use_reply_is_parsed_into_tool_calls():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json=reply(
                [
                    {"type": "text", "text": "looking"},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "search_moments",
                        "input": {"query": "sanitizer"},
                    },
                ],
                stop_reason="tool_use",
            ),
        )

    response = backend_with(handler).complete(system="s", turns=[Turn(role="user", text="q")])
    assert response.text == "looking"
    assert response.tool_calls == [ToolCall("toolu_1", "search_moments", {"query": "sanitizer"})]
    assert response.stop_reason == "tool_use"
    assert response.usage == Usage(120, 34)


def test_all_tool_results_go_back_in_one_user_message(captured):
    """Splitting them teaches the model to stop calling tools in parallel."""

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=reply([{"type": "text", "text": "done"}]))

    turns = [
        Turn(role="user", text="q"),
        Turn(
            role="assistant",
            tool_calls=[ToolCall("t1", "search_moments", {}), ToolCall("t2", "list_videos", {})],
        ),
        Turn(
            role="tool",
            tool_results=[
                ToolResult("t1", "four hits"),
                ToolResult("t2", "two videos", is_error=True),
            ],
        ),
    ]
    backend_with(handler).complete(system="s", turns=turns)

    messages = captured["body"]["messages"]
    tool_messages = [
        m
        for m in messages
        if isinstance(m["content"], list) and m["content"][0].get("type") == "tool_result"
    ]
    assert len(tool_messages) == 1
    blocks = tool_messages[0]["content"]
    assert [b["tool_use_id"] for b in blocks] == ["t1", "t2"]
    assert blocks[1]["is_error"] is True
    assert "is_error" not in blocks[0]


def test_assistant_raw_content_is_replayed_verbatim(captured):
    """Claude requires thinking blocks to come back unchanged."""
    raw = [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "partial"},
    ]

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json=reply([{"type": "text", "text": "ok"}]))

    backend_with(handler).complete(
        system="s",
        turns=[Turn(role="user", text="q"), Turn(role="assistant", text="partial", raw=raw)],
    )
    assert captured["body"]["messages"][1]["content"] == raw


def test_known_model_is_priced_and_unknown_is_not():
    priced = AnthropicBackend("claude-opus-5", api_key="sk-ant-test")
    assert priced.price(Usage(1_000_000, 1_000_000)) == pytest.approx(30.0)
    unknown = AnthropicBackend("some-local-model", api_key="sk-ant-test")
    assert unknown.price(Usage(1000, 1000)) is None
