"""Tools the agent may call."""

from __future__ import annotations

from zoneinfo import ZoneInfo

from vpi.agent.ledger import EvidenceLedger
from vpi.datalake.client import DataLakeClient
from vpi.tools.datalake_tools import build_datalake_tools
from vpi.tools.registry import Tool, ToolOutcome, ToolRegistry
from vpi.tools.time_tools import build_time_tool, resolve_timeframe


def build_registry(
    client: DataLakeClient, ledger: EvidenceLedger, collection_id: str, tz: ZoneInfo
) -> ToolRegistry:
    return ToolRegistry([*build_datalake_tools(client, ledger, collection_id), build_time_tool(tz)])


__all__ = [
    "Tool",
    "ToolOutcome",
    "ToolRegistry",
    "build_datalake_tools",
    "build_registry",
    "build_time_tool",
    "resolve_timeframe",
]
