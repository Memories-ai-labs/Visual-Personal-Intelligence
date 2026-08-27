"""Client behaviour: auth, error mapping, retries, cost metering."""

from __future__ import annotations

import httpx
import pytest

from tests.conftest import make_client
from vpi.datalake import errors
from vpi.datalake.client import DataLakeClient


def test_key_goes_in_authorization_without_bearer_prefix():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"collections": []})

    with make_client(handler) as client:
        client.list_collections()
    assert seen["auth"] == "sk-mai-test"


@pytest.mark.parametrize(
    ("status", "message", "expected"),
    [
        (401, "unauthorized", errors.AuthError),
        (403, "out of scope", errors.PermissionDenied),
        (404, "no such video", errors.NotFound),
        (400, "targets must be a non-empty array", errors.InvalidArgument),
        (400, "query/traits interface is deprecated", errors.EndpointDeprecated),
    ],
)
def test_status_codes_map_to_typed_errors(status, message, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"code": "x", "message": message, "request_id": "req_1"}}
        )

    with make_client(handler) as client, pytest.raises(expected) as caught:
        client.get_video("vid_x")
    assert caught.value.request_id == "req_1"


def test_retry_after_is_honoured_then_the_call_succeeds():
    slept: list[float] = []
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(
                429,
                json={
                    "error": {"code": "rate_limited", "message": "slow down", "retry_after": 1.5}
                },
            )
        return httpx.Response(200, json={"id": "vid_x", "status": "ready"})

    client = make_client(handler, sleep=slept.append)
    with client:
        video = client.get_video("vid_x")
    assert video.id == "vid_x"
    assert slept == [1.5]


def test_video_not_ready_is_its_own_error_after_retries_run_out():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            409,
            json={
                "error": {
                    "code": "video_not_ready",
                    "message": "still indexing",
                    "retry_after": 0.1,
                }
            },
        )

    client = make_client(handler, sleep=lambda _: None, max_retries=1)
    with client, pytest.raises(errors.VideoNotReady):
        client.get_moment("vid_x@0-1")


def test_server_errors_are_retried_and_then_raised():
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(503, json={"error": {"code": "unavailable", "message": "nope"}})

    client = make_client(handler, sleep=lambda _: None, max_retries=2)
    with client, pytest.raises(errors.ServerError):
        client.get_video("vid_x")
    assert calls["n"] == 3


def test_missing_key_is_a_clear_error(monkeypatch):
    from vpi.config import Settings

    with pytest.raises(errors.AuthError, match="MEMORIES_API_KEY"):
        DataLakeClient(Settings(MEMORIES_API_KEY=""))


def test_search_requires_known_targets():
    with (
        make_client(lambda r: httpx.Response(200, json={})) as client,
        pytest.raises(errors.InvalidArgument, match="targets"),
    ):
        client.search("col_x", "q", targets=["nonsense"])


def test_hybrid_mode_refuses_a_cursor():
    """Hybrid does not paginate — asking for a page is a bug worth surfacing."""
    with (
        make_client(lambda r: httpx.Response(200, json={})) as client,
        pytest.raises(errors.InvalidArgument, match="hybrid"),
    ):
        client.search("col_x", "q", mode="hybrid", cursor="abc")


def test_search_stamps_request_id_on_every_hit():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "request_id": "req_search_1",
                "results": [
                    {
                        "ref": "vid_a@1.0-2.0",
                        "video_id": "vid_a",
                        "score": 0.5,
                        "start": 1.0,
                        "end": 2.0,
                        "snippet": "x",
                    }
                ],
            },
        )

    with make_client(handler) as client:
        page = client.search("col_x", "q")
    assert page.request_id == "req_search_1"
    assert page.results[0].request_id == "req_search_1"


def test_cost_meter_charges_per_call_and_triples_for_rerank():
    charged: list[tuple[str, float]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [], "request_id": "r"})

    with make_client(handler, meter=lambda label, usd: charged.append((label, usd))) as client:
        client.search("col_x", "q")
        client.search("col_x", "q", rerank=True)

    assert charged[0] == ("search", 0.008)
    assert charged[1][1] == pytest.approx(0.024)


def test_clip_expansion_costs_more_than_a_plain_moment():
    charged: list[tuple[str, float]] = []

    with make_client(
        lambda r: httpx.Response(200, json={"ref": "vid_a@0-1", "video_id": "vid_a"}),
        meter=lambda label, usd: charged.append((label, usd)),
    ) as client:
        client.get_moment("vid_a@0-1", expand=["caption"])
        client.get_moment("vid_a@0-1", expand=["caption", "clip"])

    assert charged[0][1] == pytest.approx(0.008)
    assert charged[1][1] == pytest.approx(0.013)


def test_request_id_falls_back_to_the_header_on_success():
    """Live responses carry the id only in X-Request-ID; the body has it on errors."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "ref": "vid_a@1.0-2.0",
                        "video_id": "vid_a",
                        "score": 0.5,
                        "start": 1.0,
                        "end": 2.0,
                        "snippet": "x",
                    }
                ]
            },
            headers={"X-Request-ID": "req_from_header"},
        )

    with make_client(handler) as client:
        page = client.search("col_x", "q")
    assert page.request_id == "req_from_header"
    assert page.results[0].request_id == "req_from_header"
