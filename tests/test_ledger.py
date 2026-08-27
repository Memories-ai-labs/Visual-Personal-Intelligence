from __future__ import annotations

from vpi.datalake.models import Moment, SearchHit, Segment


def test_hits_get_sequential_ids(ledger):
    for i in range(3):
        ledger.add_hit(
            SearchHit(
                ref=f"vid_a@{i}.0-{i + 1}.0",
                video_id="vid_a",
                start=i,
                end=i + 1,
                snippet=f"text {i}",
            )
        )
    assert [e.eid for e in ledger] == ["E1", "E2", "E3"]


def test_identical_spans_are_deduped(ledger):
    hit = SearchHit(ref="vid_a@1.0-2.0", video_id="vid_a", start=1.0, end=2.0, snippet="same")
    assert ledger.add_hit(hit) is not None
    assert ledger.add_hit(hit) is None
    assert len(ledger) == 1


def test_moment_expansion_adds_both_streams(ledger):
    moment = Moment(
        ref="vid_a@0.0-5.0",
        video_id="vid_a",
        start=0.0,
        end=5.0,
        transcription=[Segment(start=0.0, end=2.0, text="spoken")],
        caption=[Segment(start=0.0, end=2.0, text="visible")],
        clip_url="https://signed/clip.mp4",
    )
    added = ledger.add_moment(moment)
    assert {e.target for e in added} == {"transcription", "caption"}
    assert all(e.clip_url for e in added)


def test_empty_segments_are_skipped(ledger):
    added = ledger.add_segments(
        "vid_a", [Segment(start=0, end=1, text="   ")], target="caption", source="t"
    )
    assert added == []


def test_scores_are_not_comparable_across_requests(ledger):
    a = ledger.add_hit(
        SearchHit(ref="vid_a@0-1", video_id="vid_a", snippet="a", score=0.9, request_id="req_1")
    )
    b = ledger.add_hit(
        SearchHit(ref="vid_b@0-1", video_id="vid_b", snippet="b", score=0.4, request_id="req_2")
    )
    c = ledger.add_hit(
        SearchHit(ref="vid_c@0-1", video_id="vid_c", snippet="c", score=0.5, request_id="req_1")
    )
    assert ledger.comparable_with(a, c) is True
    assert ledger.comparable_with(a, b) is False


def test_drop_removes_without_renumbering_survivors(ledger):
    for i in range(3):
        ledger.add_hit(
            SearchHit(
                ref=f"vid_a@{i}-{i + 1}", video_id="vid_a", start=i, end=i + 1, snippet=f"t{i}"
            )
        )
    ledger.drop({"E2"})
    assert [e.eid for e in ledger] == ["E1", "E3"]
    assert ledger.get("E2") is None


def test_render_is_citable(ledger):
    ledger.add_hit(
        SearchHit(
            ref="vid_a@1.0-2.0",
            video_id="vid_a",
            start=1.0,
            end=2.0,
            snippet="hello",
            target="transcription",
        )
    )
    rendered = ledger.render()
    assert "[E1]" in rendered and "vid_a@1.0-2.0" in rendered and "hello" in rendered
