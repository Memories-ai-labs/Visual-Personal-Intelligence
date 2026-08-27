"""Claude backend — the default.

Adaptive thinking is on and `effort` carries the depth knob; `budget_tokens` is
rejected by current models and is deliberately absent. Assistant content is
replayed from `Turn.raw` so thinking blocks return unchanged.
"""

from __future__ import annotations

from typing import Any

import anthropic

from vpi.llm.base import LLMResponse, ToolCall, ToolSpec, Turn, Usage

# Input / output USD per 1M tokens, for the cost ledger. Unknown models simply
# report tokens with no dollar figure rather than a wrong one.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-fable-5": (10.0, 50.0),
}


class AnthropicBackend:
    def __init__(self, model: str, *, api_key: str = "", effort: str = "high") -> None:
        self.model = model
        self.effort = effort
        self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

    def complete(
        self,
        *,
        system: str,
        turns: list[Turn],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": _to_messages(turns),
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": self.effort},
        }
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                for t in tools
            ]

        response = self._client.messages.create(**kwargs)

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input)))

        return LLMResponse(
            text="\n".join(text_parts).strip(),
            tool_calls=calls,
            usage=Usage(response.usage.input_tokens, response.usage.output_tokens),
            raw=response.content,
            stop_reason=response.stop_reason or "",
        )

    def price(self, usage: Usage) -> float | None:
        prices = MODEL_PRICES.get(self.model)
        if prices is None:
            return None
        per_in, per_out = prices
        return (usage.input_tokens * per_in + usage.output_tokens * per_out) / 1_000_000


def _to_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    """Turns → Anthropic messages.

    All tool results for one assistant turn go into a *single* user message;
    splitting them teaches the model to stop calling tools in parallel.
    """
    messages: list[dict[str, Any]] = []
    for turn in turns:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.text})
        elif turn.role == "assistant":
            if turn.raw is not None:
                messages.append({"role": "assistant", "content": turn.raw})
                continue
            content: list[dict[str, Any]] = []
            if turn.text:
                content.append({"type": "text", "text": turn.text})
            for call in turn.tool_calls:
                content.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            messages.append({"role": "assistant", "content": content})
        else:
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": r.call_id,
                            "content": r.content,
                            **({"is_error": True} if r.is_error else {}),
                        }
                        for r in turn.tool_results
                    ],
                }
            )
    return messages
