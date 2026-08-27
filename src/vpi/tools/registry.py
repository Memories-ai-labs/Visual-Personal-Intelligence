"""Tool registry.

A tool is a name, a description the model reads, a JSON schema, and a callable
that returns a `ToolOutcome`. Nothing provider-specific lives here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vpi.agent.ledger import Evidence
from vpi.llm.base import ToolSpec


@dataclass
class ToolOutcome:
    """What a tool hands back: text for the model, evidence for the ledger."""

    text: str
    evidence: list[Evidence] = field(default_factory=list)
    is_error: bool = False


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    run: Callable[..., ToolOutcome]

    def spec(self) -> ToolSpec:
        return ToolSpec(self.name, self.description, self.input_schema)


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[ToolSpec]:
        return [t.spec() for t in self._tools.values()]

    def call(self, name: str, arguments: dict[str, Any]) -> ToolOutcome:
        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome(
                f"No such tool: {name}. Available: {', '.join(self.names)}", is_error=True
            )
        try:
            return tool.run(**arguments)
        except TypeError as exc:
            return ToolOutcome(f"Bad arguments for {name}: {exc}", is_error=True)
