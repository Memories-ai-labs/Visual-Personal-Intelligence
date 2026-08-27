"""OpenAI-compatible backend — the bring-your-own-endpoint path.

Anything that speaks the chat-completions shape works here: OpenRouter, vLLM,
LM Studio, Ollama. Tool-calling fidelity varies by model; the grounding gate is
what keeps a weaker model honest.
"""

from __future__ import annotations

import json
from typing import Any

from openai import OpenAI

from vpi.llm.base import LLMResponse, ToolCall, ToolSpec, Turn, Usage


class OpenAICompatBackend:
    def __init__(self, model: str, *, base_url: str, api_key: str = "not-needed") -> None:
        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

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
            "messages": [{"role": "system", "content": system}, *_to_messages(turns)],
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in tools
            ]

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        calls = [
            ToolCall(id=c.id, name=c.function.name, arguments=_loads(c.function.arguments))
            for c in (choice.message.tool_calls or [])
        ]
        usage = response.usage
        return LLMResponse(
            text=(choice.message.content or "").strip(),
            tool_calls=calls,
            usage=Usage(
                getattr(usage, "prompt_tokens", 0) or 0,
                getattr(usage, "completion_tokens", 0) or 0,
            ),
            stop_reason=choice.finish_reason or "",
        )

    def price(self, usage: Usage) -> float | None:  # noqa: ARG002 - unknown third-party pricing
        return None


def _loads(raw: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _to_messages(turns: list[Turn]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for turn in turns:
        if turn.role == "user":
            messages.append({"role": "user", "content": turn.text})
        elif turn.role == "assistant":
            message: dict[str, Any] = {"role": "assistant", "content": turn.text or None}
            if turn.tool_calls:
                message["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in turn.tool_calls
                ]
            messages.append(message)
        else:
            # One message per result here — the chat-completions shape requires it.
            messages.extend(
                {"role": "tool", "tool_call_id": r.call_id, "content": r.content}
                for r in turn.tool_results
            )
    return messages
