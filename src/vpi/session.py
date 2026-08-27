"""Wiring.

One place that knows how the pieces fit together, so the CLI, the server and the
tests all build the same agent.
"""

from __future__ import annotations

from dataclasses import dataclass

from vpi.agent.cost import CostLedger
from vpi.agent.ledger import EvidenceLedger
from vpi.agent.loop import Agent
from vpi.config import Settings, get_settings
from vpi.datalake.client import DataLakeClient
from vpi.llm import build_backend
from vpi.llm.base import LLMBackend
from vpi.tools import build_registry


class MissingCollection(RuntimeError):
    """No collection to chat with, and we will not guess one."""


@dataclass
class Session:
    """One conversation: its own ledger, its own cost, one collection."""

    agent: Agent
    client: DataLakeClient
    ledger: EvidenceLedger
    cost: CostLedger
    collection_id: str
    settings: Settings

    def close(self) -> None:
        self.client.close()


def build_client(settings: Settings | None = None, *, meter=None) -> DataLakeClient:
    settings = settings or get_settings()
    if settings.demo:
        from vpi.demo import demo_transport

        return DataLakeClient(settings, transport=demo_transport(), meter=meter)
    return DataLakeClient(settings, meter=meter)


def resolve_collection(client: DataLakeClient, settings: Settings) -> str:
    """Explicit config wins; a single collection is unambiguous; otherwise ask."""
    if settings.demo:
        from vpi.demo import DEMO_COLLECTION

        return DEMO_COLLECTION
    if settings.collection_id:
        return settings.collection_id

    collections = client.list_collections()
    if len(collections) == 1:
        return collections[0].id
    if not collections:
        raise MissingCollection(
            "This account has no collections. Run `vpi ingest <path-or-url>` to index "
            "something first, or set VPI_DEMO=1 to try the bundled fixtures."
        )
    listing = "\n".join(f"  {c.id}  {c.name} ({c.video_count} videos)" for c in collections)
    raise MissingCollection(
        "Several collections exist — set VPI_COLLECTION_ID to the one you want to chat "
        f"with:\n{listing}"
    )


def build_session(
    settings: Settings | None = None,
    *,
    backend: LLMBackend | None = None,
    collection_id: str | None = None,
    verify_relevance: bool = True,
) -> Session:
    settings = settings or get_settings()
    cost = CostLedger()
    client = build_client(settings, meter=cost.charge_api)
    try:
        collection_id = collection_id or resolve_collection(client, settings)
    except Exception:
        client.close()
        raise

    ledger = EvidenceLedger()
    registry = build_registry(client, ledger, collection_id, settings.tz)
    agent = Agent(
        backend or build_backend(settings),
        registry,
        ledger,
        timezone=settings.timezone,
        max_steps=settings.max_steps,
        cost=cost,
        verify_relevance=verify_relevance,
    )
    return Session(
        agent=agent,
        client=client,
        ledger=ledger,
        cost=cost,
        collection_id=collection_id,
        settings=settings,
    )
