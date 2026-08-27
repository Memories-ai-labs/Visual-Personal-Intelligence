from __future__ import annotations

import httpx
import pytest

from vpi.agent.cost import CostLedger
from vpi.agent.ledger import EvidenceLedger
from vpi.config import Settings
from vpi.datalake.client import DataLakeClient
from vpi.demo import DEMO_COLLECTION, demo_transport
from vpi.llm.base import LLMResponse, ToolSpec, Turn


class ScriptedBackend:
    """A model whose every reply is decided by the test."""

    model = "test-model"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def complete(
        self,
        *,
        system: str,
        turns: list[Turn],
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 16000,
    ) -> LLMResponse:
        self.calls.append({"system": system, "turns": list(turns), "tools": tools})
        if not self.responses:
            return LLMResponse(text="I couldn't find anything about that.")
        return self.responses.pop(0)

    def price(self, usage) -> float | None:  # noqa: ANN001
        return 0.0


@pytest.fixture
def settings() -> Settings:
    return Settings(
        MEMORIES_API_KEY="sk-mai-test",
        VPI_COLLECTION_ID=DEMO_COLLECTION,
        VPI_TIMEZONE="Asia/Shanghai",
        VPI_DEMO=True,
    )


@pytest.fixture
def cost() -> CostLedger:
    return CostLedger()


@pytest.fixture
def demo_client(settings: Settings, cost: CostLedger) -> DataLakeClient:
    client = DataLakeClient(settings, transport=demo_transport(), meter=cost.charge_api)
    yield client
    client.close()


@pytest.fixture
def ledger() -> EvidenceLedger:
    return EvidenceLedger()


def make_client(handler, **kwargs) -> DataLakeClient:
    """A client whose transport is a bare handler function."""
    settings = Settings(MEMORIES_API_KEY="sk-mai-test")
    return DataLakeClient(settings, transport=httpx.MockTransport(handler), **kwargs)
