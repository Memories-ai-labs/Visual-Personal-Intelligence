"""The ReAct loop.

Think → act → observe, with three additions that matter more than the loop
itself: everything retrieved lands in an evidence ledger, a relevance pass
prunes it once before the answer is written, and a grounding gate refuses to
show the user a sentence that nothing supports.

The loop emits events rather than returning a string, so the terminal chat and
the web UI can show the same run as it happens.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from vpi.agent import grounding, relevance
from vpi.agent.cost import CostLedger
from vpi.agent.ledger import Citation, EvidenceLedger
from vpi.agent.prompts import evidence_block, system_prompt
from vpi.llm.base import LLMBackend, ToolResult, Turn
from vpi.tools.registry import ToolRegistry

EventKind = Literal[
    "tool_call", "tool_result", "note", "answer", "citations", "cost", "warning", "done"
]


@dataclass
class AgentEvent:
    kind: EventKind
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Answer:
    text: str
    citations: list[Citation]
    grounded: bool
    cost: CostLedger
    steps: int


class Agent:
    def __init__(
        self,
        backend: LLMBackend,
        registry: ToolRegistry,
        ledger: EvidenceLedger,
        *,
        timezone: str = "UTC",
        max_steps: int = 12,
        cost: CostLedger | None = None,
        verify_relevance: bool = True,
        relevance_min_items: int = 4,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.ledger = ledger
        self.timezone = timezone
        self.max_steps = max_steps
        self.cost = cost or CostLedger()
        self.verify_relevance = verify_relevance
        # Below this many entries the pass costs more than it saves.
        self.relevance_min_items = relevance_min_items
        self._turns: list[Turn] = []

    # ------------------------------------------------------------------ helpers

    @property
    def turns(self) -> list[Turn]:
        return list(self._turns)

    def _charge(self, usage: Any) -> None:
        """Record token spend, priced by the backend when it knows its own rates."""
        price = getattr(self.backend, "price", None)
        self.cost.charge_llm(usage, price(usage) if price else None)

    def _system(self) -> str:
        prompt = system_prompt(self.timezone)
        if len(self.ledger):
            prompt += "\n\n" + evidence_block(self.ledger.render())
        return prompt

    def _prune(self, question: str) -> Iterator[AgentEvent]:
        """Drop evidence the vector search dragged in by accident."""
        result = relevance.verify(
            self.backend, question, self.ledger, min_items=self.relevance_min_items
        )
        # Priced like any other call — charging it as unpriced made every turn
        # that ran this pass report "model price unknown".
        self._charge(result.usage)
        if result.dropped:
            yield AgentEvent(
                "note",
                f"dropped {len(result.dropped)} irrelevant "
                f"{'entry' if len(result.dropped) == 1 else 'entries'}"
                + (f": {result.reason}" if result.reason else ""),
                {"dropped": sorted(result.dropped)},
            )
        if result.failed:
            yield AgentEvent("warning", "relevance check unavailable; kept all evidence")

    # --------------------------------------------------------------------- ask

    def ask(self, question: str) -> Iterator[AgentEvent]:
        """Run one question to a grounded answer, emitting progress as it goes."""
        self._turns.append(Turn(role="user", text=question))
        repaired = False
        verified = False

        for step in range(1, self.max_steps + 1):
            response = self.backend.complete(
                system=self._system(),
                turns=self._turns,
                tools=self.registry.specs(),
            )
            self._charge(response.usage)

            if response.tool_calls:
                self._turns.append(
                    Turn(
                        role="assistant",
                        text=response.text,
                        tool_calls=response.tool_calls,
                        raw=response.raw,
                    )
                )
                results: list[ToolResult] = []
                for call in response.tool_calls:
                    yield AgentEvent(
                        "tool_call",
                        f"{call.name}({_brief(call.arguments)})",
                        {"name": call.name, "arguments": call.arguments},
                    )
                    outcome = self.registry.call(call.name, call.arguments)
                    results.append(ToolResult(call.id, outcome.text, is_error=outcome.is_error))
                    yield AgentEvent(
                        "tool_result",
                        outcome.text,
                        {
                            "name": call.name,
                            "is_error": outcome.is_error,
                            "evidence": [c.eid for c in outcome.evidence],
                        },
                    )
                # All results for one assistant turn go back in a single turn.
                self._turns.append(Turn(role="tool", tool_results=results))

                # Prune before the model drafts anything, so it never cites an
                # entry we were about to throw away.
                if (
                    self.verify_relevance
                    and not verified
                    and len(self.ledger) >= self.relevance_min_items
                ):
                    verified = True
                    yield from self._prune(question)
                continue

            gate = grounding.check(response.text, self.ledger.ids)
            if gate.needs_repair and not repaired:
                repaired = True
                yield AgentEvent(
                    "warning",
                    f"{len(gate.dropped)} unsupported "
                    f"{'sentence' if len(gate.dropped) == 1 else 'sentences'}; "
                    "asking for citations",
                    {"unsupported": gate.dropped, "invalid_ids": sorted(gate.invalid_ids)},
                )
                self._turns.append(Turn(role="assistant", text=response.text, raw=response.raw))
                self._turns.append(Turn(role="user", text=grounding.repair_prompt(gate)))
                continue

            final = grounding.enforce(response.text, self.ledger.ids)
            self._turns.append(Turn(role="assistant", text=final.text, raw=response.raw))
            citations = [Citation.of(e) for e in self.ledger.cited(final.cited_ids)]

            yield AgentEvent("answer", final.text, {"grounded": final.grounded})
            if citations:
                yield AgentEvent(
                    "citations",
                    f"{len(citations)} cited moments",
                    {"citations": [c.__dict__ for c in citations]},
                )
            yield AgentEvent("cost", self.cost.summary(), {"api_usd": self.cost.api_usd})
            yield AgentEvent(
                "done",
                "",
                {
                    "answer": final.text,
                    "grounded": final.grounded,
                    "steps": step,
                    "citations": [c.__dict__ for c in citations],
                },
            )
            return

        yield AgentEvent(
            "warning",
            f"stopped after {self.max_steps} steps without settling on an answer",
        )
        yield AgentEvent("answer", grounding.NO_EVIDENCE_ANSWER, {"grounded": False})
        yield AgentEvent("cost", self.cost.summary(), {"api_usd": self.cost.api_usd})
        yield AgentEvent("done", "", {"answer": grounding.NO_EVIDENCE_ANSWER, "grounded": False})

    def answer(self, question: str) -> Answer:
        """Run a question and return only the outcome — for tests and scripts."""
        text = grounding.NO_EVIDENCE_ANSWER
        citations: list[Citation] = []
        grounded = False
        steps = 0
        for event in self.ask(question):
            if event.kind == "done":
                text = event.data.get("answer", text)
                grounded = bool(event.data.get("grounded"))
                steps = int(event.data.get("steps", 0))
                citations = [Citation(**c) for c in event.data.get("citations", [])]
        return Answer(
            text=text, citations=citations, grounded=grounded, cost=self.cost, steps=steps
        )


def _brief(arguments: dict[str, Any], limit: int = 90) -> str:
    parts = []
    for key, value in arguments.items():
        rendered = str(value)
        if len(rendered) > limit:
            rendered = rendered[:limit] + "…"
        parts.append(f"{key}={rendered}")
    return ", ".join(parts)
