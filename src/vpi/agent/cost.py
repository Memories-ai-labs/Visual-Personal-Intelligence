"""What a turn cost, in dollars and in calls.

Two ledgers on purpose: datalake calls are priced per call from a published
table, LLM tokens are priced per model. A model we have no price for reports
tokens and no dollars — an invented number is worse than a missing one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from vpi.llm.base import Usage


@dataclass
class CostLedger:
    api_usd: float = 0.0
    calls: Counter = field(default_factory=Counter)
    usage: Usage = field(default_factory=Usage)
    llm_usd: float | None = 0.0

    def charge_api(self, label: str, usd: float) -> None:
        self.calls[label] += 1
        self.api_usd += usd

    def charge_llm(self, usage: Usage, usd: float | None) -> None:
        self.usage = self.usage + usage
        if usd is None:
            self.llm_usd = None
        elif self.llm_usd is not None:
            self.llm_usd += usd

    @property
    def total_usd(self) -> float | None:
        if self.llm_usd is None:
            return None
        return self.api_usd + self.llm_usd

    def summary(self) -> str:
        call_note = ", ".join(f"{n}×{k}" for k, n in sorted(self.calls.items())) or "no api calls"
        tokens = f"{self.usage.input_tokens:,} in / {self.usage.output_tokens:,} out"
        if self.total_usd is None:
            return (
                f"${self.api_usd:.4f} datalake ({call_note}) · {tokens} tokens "
                "(model price unknown)"
            )
        return (
            f"${self.total_usd:.4f} total — ${self.api_usd:.4f} datalake "
            f"({call_note}), {tokens} tokens"
        )
