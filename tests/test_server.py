"""The HTTP surface: health, streaming, session reset, media refresh."""

from __future__ import annotations

import json
from zoneinfo import ZoneInfo

import pytest
from fastapi.testclient import TestClient

from tests.conftest import ScriptedBackend
from vpi.agent.cost import CostLedger
from vpi.agent.ledger import EvidenceLedger
from vpi.agent.loop import Agent
from vpi.config import Settings
from vpi.datalake.client import DataLakeClient
from vpi.demo import DEMO_COLLECTION, demo_transport
from vpi.llm.base import LLMResponse, ToolCall
from vpi.server import app as server_module
from vpi.session import Session
from vpi.tools import build_registry


@pytest.fixture
def client(monkeypatch) -> TestClient:
    settings = Settings(MEMORIES_API_KEY="sk-mai-test", VPI_DEMO=True, VPI_TIMEZONE="UTC")
    monkeypatch.setattr(server_module, "get_settings", lambda: settings)

    def build() -> Session:
        cost = CostLedger()
        dl = DataLakeClient(settings, transport=demo_transport(), meter=cost.charge_api)
        ledger = EvidenceLedger()
        agent = Agent(
            ScriptedBackend(
                [
                    LLMResponse(
                        text="",
                        tool_calls=[
                            ToolCall("t1", "search_moments", {"query": "detergent sanitizer"})
                        ],
                    ),
                    LLMResponse(text="You were mixing sanitiser [E1]."),
                ]
            ),
            build_registry(dl, ledger, DEMO_COLLECTION, ZoneInfo("UTC")),
            ledger,
            cost=cost,
            verify_relevance=False,
        )
        return Session(
            agent=agent,
            client=dl,
            ledger=ledger,
            cost=cost,
            collection_id=DEMO_COLLECTION,
            settings=settings,
        )

    monkeypatch.setattr(server_module, "build_session", build)
    server_module.store.sessions.clear()
    return TestClient(server_module.app)


def test_health_reports_mode_and_model(client: TestClient):
    body = client.get("/api/health").json()
    assert body["ok"] is True
    assert body["demo"] is True
    assert body["model"] == "claude-opus-5"


def test_empty_question_is_rejected(client: TestClient):
    assert client.post("/api/chat", json={"question": "  "}).status_code == 400


def test_chat_streams_events_ending_in_done(client: TestClient):
    response = client.post("/api/chat", json={"question": "what about the sanitizer?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    events = [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "tool_call"
    assert "answer" in kinds
    assert kinds[-1] == "done"

    answer = next(e for e in events if e["kind"] == "answer")
    assert "[E1]" in answer["text"]
    citations = next(e for e in events if e["kind"] == "citations")
    assert citations["citations"][0]["eid"] == "E1"


def test_cost_event_is_streamed(client: TestClient):
    response = client.post("/api/chat", json={"question": "sanitizer?"})
    assert any('"kind": "cost"' in line for line in response.text.splitlines())


def test_session_can_be_reset(client: TestClient):
    client.post("/api/chat", json={"question": "sanitizer?", "session_id": "s1"})
    assert client.delete("/api/session/s1").json() == {"dropped": True}
    assert client.delete("/api/session/s1").json() == {"dropped": False}


def test_media_endpoint_rejects_an_unusable_kind(client: TestClient):
    assert client.get("/api/media", params={"ref": "vid_a", "kind": "nonsense"}).status_code == 400
