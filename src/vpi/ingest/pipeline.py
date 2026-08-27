"""Ingest: get videos into a collection and wait for them to be usable.

The one rule the API documentation is emphatic about: when polling an operation,
**only `done` is truth**. `progress.percent` is for display, and a non-null
`error` means failure — including partial failure. This module treats both
accordingly.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from vpi.datalake.client import PRICE_INDEX_PER_MINUTE, DataLakeClient
from vpi.datalake.errors import DataLakeError

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi"}


@dataclass
class IngestItem:
    source: str
    video_id: str = ""
    operation: str = ""
    done: bool = False
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.done and not self.error


def expand_sources(sources: Iterable[str]) -> list[str]:
    """Turn paths, directories and URLs into a flat list of things to upload."""
    out: list[str] = []
    for source in sources:
        if source.startswith(("http://", "https://")):
            out.append(source)
            continue
        path = Path(source).expanduser()
        if path.is_dir():
            out.extend(str(p) for p in sorted(path.iterdir()) if p.suffix.lower() in VIDEO_SUFFIXES)
        elif path.is_file():
            out.append(str(path))
        else:
            raise FileNotFoundError(f"{source} is neither a file, a directory, nor a URL")
    return out


def submit(client: DataLakeClient, collection_id: str, source: str) -> IngestItem:
    item = IngestItem(source=source)
    try:
        if source.startswith(("http://", "https://")):
            body = client.upload_video_url(collection_id, source)
        else:
            body = client.upload_video_file(collection_id, source)
    except DataLakeError as exc:
        item.error = str(exc)
        item.done = True
        return item
    item.video_id = str(body.get("video_id", ""))
    item.operation = str(body.get("operation", ""))
    return item


def wait(
    client: DataLakeClient,
    items: list[IngestItem],
    *,
    timeout: float = 3600.0,
    interval: float = 5.0,
    sleep: Callable[[float], None] = time.sleep,
    now: Callable[[], float] = time.monotonic,
) -> Iterator[IngestItem]:
    """Yield each item as it finishes. Only `done` decides that."""
    pending = {i.operation: i for i in items if i.operation and not i.done}
    for item in items:
        if item.done:
            yield item

    deadline = now() + timeout
    while pending:
        if now() > deadline:
            for item in pending.values():
                item.error = f"still indexing after {timeout:.0f}s (operation {item.operation})"
                item.done = True
                yield item
            return

        sleep(interval)
        for operation_id, item in list(pending.items()):
            try:
                operation = client.get_operation(operation_id)
            except DataLakeError as exc:
                item.error = str(exc)
                item.done = True
                pending.pop(operation_id)
                yield item
                continue

            if not operation.done:
                continue
            item.done = True
            if operation.failed:
                item.error = str(operation.error)
            pending.pop(operation_id)
            yield item


def estimate_cost_usd(minutes: float) -> float:
    return minutes * PRICE_INDEX_PER_MINUTE
