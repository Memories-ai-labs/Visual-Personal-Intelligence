"""Demo mode runs the real client, so it is worth testing as a real path."""

from __future__ import annotations

from vpi.demo import DEMO_COLLECTION


def test_search_then_expand_yields_full_text(demo_client, ledger):
    page = demo_client.search(DEMO_COLLECTION, "detergent sanitizer", targets=["transcription"])
    assert page.results
    moment = demo_client.get_moment(page.results[0].ref, expand=["transcription", "caption"])
    added = ledger.add_moment(moment)
    assert added
    assert any("sanitizer" in e.text or "hands" in e.text for e in added)


def test_query_matching_nothing_returns_a_usable_hint(demo_client):
    page = demo_client.search(DEMO_COLLECTION, "snowboarding halfpipe competition")
    assert page.results == []
    assert "demo collection" in page.hint


def test_videos_are_listed_with_status(demo_client):
    videos, _ = demo_client.list_videos(DEMO_COLLECTION)
    assert len(videos) == 2
    assert all(v.is_ready for v in videos)
    assert {v.title for v in videos} == {"kitchen-prep.mp4", "standup-monday.mp4"}


def test_summary_and_title_are_available(demo_client):
    videos, _ = demo_client.list_videos(DEMO_COLLECTION)
    summary = demo_client.get_summary(videos[0].id)
    assert summary
    assert demo_client.get_title(videos[0].id)


def test_transcription_window_filters_segments(demo_client):
    videos, _ = demo_client.list_videos(DEMO_COLLECTION)
    kitchen = next(v for v in videos if v.title.startswith("kitchen"))
    everything = demo_client.get_transcription(kitchen.id)
    window = demo_client.get_transcription(kitchen.id, start=605.0, end=615.0)
    assert len(window) < len(everything)
    assert all(s.end > 605.0 for s in window)


def test_unknown_video_is_a_not_found(demo_client):
    import pytest

    from vpi.datalake.errors import NotFound

    with pytest.raises(NotFound):
        demo_client.get_video("vid_nope")


def test_demo_costs_are_still_metered(demo_client, cost):
    demo_client.search(DEMO_COLLECTION, "sanitizer")
    assert cost.api_usd > 0
    assert cost.calls["search"] == 1
