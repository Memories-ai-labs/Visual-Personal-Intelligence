"""Ingest: only `done` decides, and a non-null error means failure."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from tests.conftest import make_client
from vpi.ingest import expand_sources, submit, wait


def test_directory_expands_to_video_files_only(tmp_path: Path):
    (tmp_path / "a.mp4").write_bytes(b"x")
    (tmp_path / "b.mov").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("no")
    found = expand_sources([str(tmp_path)])
    assert [Path(f).name for f in found] == ["a.mp4", "b.mov"]


def test_urls_pass_through_untouched():
    assert expand_sources(["https://example.com/a.mp4"]) == ["https://example.com/a.mp4"]


def test_missing_path_is_reported():
    with pytest.raises(FileNotFoundError):
        expand_sources(["/no/such/place.mp4"])


def test_submit_returns_the_operation_handle():
    with make_client(
        lambda r: httpx.Response(200, json={"video_id": "vid_a", "operation": "op_1"})
    ) as client:
        item = submit(client, "col_x", "https://example.com/a.mp4")
    assert (item.video_id, item.operation, item.done) == ("vid_a", "op_1", False)


def test_submit_failure_is_terminal_not_pending():
    with make_client(
        lambda r: httpx.Response(
            415, json={"error": {"code": "unsupported_media", "message": "not a video"}}
        )
    ) as client:
        item = submit(client, "col_x", "https://example.com/a.txt")
    assert item.done and not item.ok
    assert "not a video" in item.error


def test_percent_progress_does_not_mean_finished():
    """A 100%-progress operation that is not `done` is still running."""
    states = [
        {"operation": "op_1", "done": False, "progress": {"percent": 100}},
        {"operation": "op_1", "done": True, "progress": {"percent": 100}, "error": None},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=states.pop(0) if len(states) > 1 else states[0])

    with make_client(handler) as client:
        item = submit(client, "col_x", "https://example.com/a.mp4")
        finished = list(wait(client, [item], sleep=lambda _: None, interval=0))
    assert len(finished) == 1 and finished[0].ok


def test_operation_error_marks_failure_even_when_done():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "operation": "op_1",
                "done": True,
                "error": {"code": "index_failed", "message": "codec"},
            },
        )

    with make_client(handler) as client:
        item = submit(client, "col_x", "https://example.com/a.mp4")
        finished = list(wait(client, [item], sleep=lambda _: None, interval=0))
    assert not finished[0].ok
    assert "codec" in finished[0].error


def test_timeout_stops_waiting_and_says_so():
    clock = {"t": 0.0}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"operation": "op_1", "done": False})

    def tick(_seconds: float) -> None:
        clock["t"] += 10.0

    with make_client(handler) as client:
        item = submit(client, "col_x", "https://example.com/a.mp4")
        finished = list(
            wait(client, [item], timeout=20.0, interval=10.0, sleep=tick, now=lambda: clock["t"])
        )
    assert not finished[0].ok
    assert "still indexing" in finished[0].error


def test_submit_uses_the_json_plus_file_multipart_shape(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00")
    seen: dict[str, bytes] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = request.content
        seen["type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"video_id": "vid_a", "operation": "op_1"})

    with make_client(handler) as client:
        submit(client, "col_x", str(video))

    assert "multipart/form-data" in seen["type"]
    assert b'name="json"' in seen["body"]
    assert b'name="file"' in seen["body"]
    assert b"clip.mp4" in seen["body"]
