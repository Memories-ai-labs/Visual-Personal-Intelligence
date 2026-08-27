"""Timeframe resolution.

The datalake stores and filters on UTC ISO-8601. People say "yesterday" and
"last Wednesday" in their own timezone. Getting this wrong silently shifts a
whole day's worth of memories, so it is a tool the model must call rather than
arithmetic it is trusted to do in its head.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

from vpi.tools.registry import Tool, ToolOutcome

_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}


def resolve_timeframe(
    phrase: str, tz: ZoneInfo, *, now: datetime | None = None
) -> tuple[datetime, datetime] | None:
    """Turn a phrase into a UTC [start, end) range, or None if unrecognised."""
    now = (now or datetime.now(UTC)).astimezone(tz)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    text = phrase.strip().lower()

    def utc(start: datetime, end: datetime) -> tuple[datetime, datetime]:
        return start.astimezone(UTC), end.astimezone(UTC)

    if text in ("today", "今天"):
        return utc(today, today + timedelta(days=1))
    if text in ("yesterday", "昨天"):
        return utc(today - timedelta(days=1), today)
    if text in ("this week", "本周", "这周"):
        start = today - timedelta(days=today.weekday())
        return utc(start, start + timedelta(days=7))
    if text in ("last week", "上周", "上个星期"):
        start = today - timedelta(days=today.weekday() + 7)
        return utc(start, start + timedelta(days=7))
    if text in ("this month", "本月"):
        start = today.replace(day=1)
        end = (start + timedelta(days=31)).replace(day=1)
        return utc(start, end)

    for unit, days in (("day", 1), ("week", 7), ("month", 30), ("year", 365)):
        for pattern in (f"last {unit}s", f"past {unit}s", f"{unit}s"):
            prefix = text.removesuffix(pattern).strip()
            if pattern in text and prefix.replace("last", "").replace("past", "").strip().isdigit():
                count = int(prefix.replace("last", "").replace("past", "").strip())
                return utc(today - timedelta(days=count * days), today + timedelta(days=1))

    for name, index in _WEEKDAYS.items():
        if name in text:
            delta = (today.weekday() - index) % 7 or 7
            start = today - timedelta(days=delta)
            return utc(start, start + timedelta(days=1))

    try:
        day = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=tz)
    except ValueError:
        return None
    return utc(day, day + timedelta(days=1))


def build_time_tool(tz: ZoneInfo) -> Tool:
    def run(phrase: str) -> ToolOutcome:
        window = resolve_timeframe(phrase, tz)
        if window is None:
            return ToolOutcome(
                f"Could not read {phrase!r} as a date or range. Ask the user for explicit "
                "dates, or pass an ISO date like 2026-08-14.",
                is_error=True,
            )
        start, end = window
        return ToolOutcome(
            f"{phrase!r} in {tz.key} is {start.isoformat()} to {end.isoformat()} (UTC).\n"
            "Use these two values as captured_at.gte / captured_at.lt in a search filter."
        )

    return Tool(
        name="resolve_timeframe",
        description=(
            "Convert a relative date phrase ('yesterday', 'last week', 'last Wednesday', "
            "'2026-08-14') into an explicit UTC range for use in a search filter. The "
            "datalake filters on UTC and the user speaks local time — always call this "
            "instead of computing dates yourself."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "phrase": {
                    "type": "string",
                    "description": "The date phrase exactly as the user said it.",
                }
            },
            "required": ["phrase"],
            "additionalProperties": False,
        },
        run=run,
    )
