from __future__ import annotations

from datetime import datetime
from typing import Any


TIME_FORMAT_12H = "12h"
TIME_FORMAT_24H = "24h"
TIME_FORMAT_CHOICES = (TIME_FORMAT_24H, TIME_FORMAT_12H)

_time_format = TIME_FORMAT_24H


def normalize_time_format(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"12", "12h", "12-hour", "12 hour", "ampm", "am/pm"}:
        return TIME_FORMAT_12H
    return TIME_FORMAT_24H


def set_time_format(value: Any) -> str:
    global _time_format
    _time_format = normalize_time_format(value)
    return _time_format


def current_time_format() -> str:
    return _time_format


def datetime_pattern(*, seconds: bool = False) -> str:
    if _time_format == TIME_FORMAT_12H:
        return "%Y-%m-%d %I:%M:%S %p" if seconds else "%Y-%m-%d %I:%M %p"
    return "%Y-%m-%d %H:%M:%S" if seconds else "%Y-%m-%d %H:%M"


def time_pattern(*, seconds: bool = False) -> str:
    if _time_format == TIME_FORMAT_12H:
        return "%I:%M:%S %p" if seconds else "%I:%M %p"
    return "%H:%M:%S" if seconds else "%H:%M"


def format_timestamp(value: Any, *, seconds: bool = False, empty: str = "") -> str:
    if value in (None, "", 0, 0.0):
        return empty
    try:
        return datetime.fromtimestamp(float(value)).strftime(datetime_pattern(seconds=seconds))
    except Exception:
        return str(value)


def format_clock_timestamp(value: Any, *, seconds: bool = False, empty: str = "") -> str:
    if value in (None, "", 0, 0.0):
        return empty
    try:
        return datetime.fromtimestamp(float(value)).strftime(time_pattern(seconds=seconds))
    except Exception:
        return str(value)


def format_datetime_text(value: Any, *, seconds: bool = False, empty: str = "") -> str:
    text = str(value or "").strip()
    if not text:
        return empty
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone()
        return parsed.strftime(datetime_pattern(seconds=seconds))
    except Exception:
        return text


def format_duration_seconds(
    value: Any,
    *,
    empty: str = "",
    max_parts: int = 2,
) -> str:
    if value in (None, ""):
        return empty
    try:
        total = max(0, int(round(float(value))))
    except Exception:
        return empty

    units = (
        ("y", 365 * 24 * 60 * 60),
        ("d", 24 * 60 * 60),
        ("h", 60 * 60),
        ("m", 60),
        ("s", 1),
    )
    if total == 0:
        return "0s"
    parts: list[str] = []
    remaining = total
    for suffix, size in units:
        amount, remaining = divmod(remaining, size)
        if amount <= 0 and not parts:
            continue
        if amount > 0:
            parts.append(f"{amount}{suffix}")
        if len(parts) >= max(1, int(max_parts)):
            break
    return " ".join(parts) if parts else "0s"
