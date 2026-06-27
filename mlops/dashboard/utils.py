from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def human_bytes(value: int) -> str:
    size = float(max(0, int(value)))
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit = units[0]
    for unit in units:
        if size < 1024 or unit == units[-1]:
            break
        size /= 1024
    if unit == "B":
        return f"{int(size)} {unit}"
    return f"{size:.2f} {unit}"


@dataclass(frozen=True)
class FileStat:
    path: str
    exists: bool
    is_file: bool
    is_dir: bool
    size_bytes: int


def stat_path(path: Path) -> FileStat:
    try:
        resolved = path.resolve()
    except Exception:
        resolved = path
    try:
        exists = resolved.exists()
    except Exception:
        exists = False
    try:
        is_file = resolved.is_file()
    except Exception:
        is_file = False
    try:
        is_dir = resolved.is_dir()
    except Exception:
        is_dir = False
    try:
        size = resolved.stat().st_size if (exists and is_file) else 0
    except Exception:
        size = 0
    return FileStat(
        path=str(resolved),
        exists=exists,
        is_file=is_file,
        is_dir=is_dir,
        size_bytes=int(size),
    )


def stable_jsonable(value: Any) -> Any:
    """Best-effort convert objects to JSON-able types for Streamlit rendering."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): stable_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [stable_jsonable(v) for v in value]
    return str(value)

