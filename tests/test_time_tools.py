"""Local-to-UTC conversion. Getting this wrong shifts a whole day of memories."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from vpi.tools.time_tools import build_time_tool, resolve_timeframe

SHANGHAI = ZoneInfo("Asia/Shanghai")
NEW_YORK = ZoneInfo("America/New_York")
# 2026-08-27 10:00 Shanghai
NOW = datetime(2026, 8, 27, 2, 0, tzinfo=UTC)


def test_yesterday_in_utc8_starts_the_previous_afternoon_in_utc():
    start, end = resolve_timeframe("yesterday", SHANGHAI, now=NOW)
    assert start == datetime(2026, 8, 25, 16, 0, tzinfo=UTC)
    assert end == datetime(2026, 8, 26, 16, 0, tzinfo=UTC)


def test_today_is_a_full_local_day():
    start, end = resolve_timeframe("today", SHANGHAI, now=NOW)
    assert (end - start).total_seconds() == 86400
    assert start == datetime(2026, 8, 26, 16, 0, tzinfo=UTC)


def test_same_phrase_differs_by_timezone():
    shanghai = resolve_timeframe("today", SHANGHAI, now=NOW)
    new_york = resolve_timeframe("today", NEW_YORK, now=NOW)
    assert shanghai != new_york


def test_last_week_is_seven_days():
    start, end = resolve_timeframe("last week", SHANGHAI, now=NOW)
    assert (end - start).days == 7
    assert start < datetime(2026, 8, 24, tzinfo=UTC)


def test_named_weekday_looks_backwards():
    start, end = resolve_timeframe("last wednesday", SHANGHAI, now=NOW)
    assert (end - start).total_seconds() == 86400
    assert start == datetime(2026, 8, 25, 16, 0, tzinfo=UTC)


def test_iso_date_is_taken_literally():
    start, _ = resolve_timeframe("2026-08-14", SHANGHAI, now=NOW)
    assert start == datetime(2026, 8, 13, 16, 0, tzinfo=UTC)


def test_chinese_phrases_work():
    assert resolve_timeframe("昨天", SHANGHAI, now=NOW) == resolve_timeframe(
        "yesterday", SHANGHAI, now=NOW
    )


def test_unparseable_phrase_returns_none_rather_than_guessing():
    assert resolve_timeframe("around the time of the thing", SHANGHAI, now=NOW) is None


def test_tool_reports_failure_instead_of_inventing_a_range():
    outcome = build_time_tool(SHANGHAI).run(phrase="whenever")
    assert outcome.is_error
    assert "Could not read" in outcome.text


def test_tool_output_names_utc_and_the_filter_field():
    outcome = build_time_tool(SHANGHAI).run(phrase="today")
    assert "UTC" in outcome.text
    assert "captured_at" in outcome.text
