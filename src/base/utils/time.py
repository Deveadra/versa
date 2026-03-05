# src/base/utils/time.py

from __future__ import annotations

from datetime import UTC, datetime

TS_UTC_FORMAT = "%Y-%m-%d %H:%M:%S"


def utc_now() -> datetime:
    """UTC-aware current time."""
    return datetime.now(UTC)


def utc_now_naive() -> datetime:
    """
    UTC time with tzinfo removed.

    Use this ONLY when a library specifically expects naive datetimes (e.g., some date parsers).
    """
    return utc_now().replace(tzinfo=None)


def utc_iso(dt: datetime | None = None) -> str:
    """
    ISO 8601 string in UTC.
    If dt is None, uses utc_now().
    """
    d = dt or utc_now()
    # Ensure UTC-aware output
    d = ensure_utc(d)
    return d.isoformat()


def utc_compact_stamp(dt: datetime | None = None) -> str:
    """Compact UTC timestamp safe for filenames."""
    d = ensure_utc(dt) if dt else utc_now()
    return d.strftime("%Y%m%d_%H%M%S")


def ensure_utc(dt: datetime) -> datetime:
    """
    Normalize a datetime to UTC-aware.
    - naive -> assume UTC
    - aware -> convert to UTC
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def parse_iso_utc(ts: str) -> datetime:
    """
    Parse ISO timestamps and guarantee UTC-aware datetime.
    Accepts naive ISO strings (assumed UTC) and aware ISO strings.
    """
    s = (ts or "").strip()
    if not s:
        raise ValueError("empty timestamp")
    # normalize trailing Z if present
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return ensure_utc(dt)


def parse_utc_ts(ts: str, fmt: str = TS_UTC_FORMAT) -> datetime:
    """
    Parse timestamps stored without tzinfo but semantically UTC (e.g. "%Y-%m-%d %H:%M:%S").
    """
    dt = datetime.strptime(ts, fmt)  # naive
    return dt.replace(tzinfo=UTC)
