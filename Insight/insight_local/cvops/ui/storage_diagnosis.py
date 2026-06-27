from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any


_SPACE_ERROR_PATTERNS = (
    re.compile(r"no space left", re.IGNORECASE),
    re.compile(r"out of (?:disk )?space", re.IGNORECASE),
    re.compile(r"disk(?: quota)? exceeded", re.IGNORECASE),
    re.compile(r"errno\s*28", re.IGNORECASE),
    re.compile(r"mac.*out of space", re.IGNORECASE),
)


def looks_like_storage_error(message: str) -> bool:
    text = str(message or "")
    return any(pattern.search(text) for pattern in _SPACE_ERROR_PATTERNS)


def _fmt_gib(value: float) -> str:
    if value >= 100:
        return f"{value:.0f} GiB"
    if value >= 10:
        return f"{value:.1f} GiB"
    return f"{value:.2f} GiB"


def _disk_record(path: Path, label: str) -> dict[str, Any] | None:
    try:
        usage = shutil.disk_usage(path)
    except Exception:
        return None
    gib = 1024 ** 3
    used = usage.total - usage.free
    return {
        "label": label,
        "path": str(path),
        "total_gib": usage.total / gib,
        "used_gib": used / gib,
        "free_gib": usage.free / gib,
        "used_pct": (used / usage.total) * 100.0 if usage.total else 0.0,
    }


def _du_kib(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        proc = subprocess.run(
            ["du", "-sk", str(path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    first = (proc.stdout or "").strip().split(maxsplit=1)
    if not first:
        return None
    try:
        return int(first[0])
    except Exception:
        return None


def build_storage_diagnosis(
    *,
    message: str = "",
    asset_root: str = "",
    extra_paths: list[str] | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    home = Path.home()
    paths: list[tuple[str, Path]] = [
        ("user cache", home / ".cache"),
        ("app caches", home / "Library" / "Caches"),
        ("developer data", home / "Library" / "Developer"),
        ("application support", home / "Library" / "Application Support"),
        ("mac temp", Path(os.environ.get("TMPDIR") or "/private/tmp")),
        ("private tmp", Path("/private/tmp")),
    ]
    if asset_root:
        paths.insert(0, ("training asset root", Path(asset_root).expanduser()))
    for raw in extra_paths or []:
        if raw:
            paths.append(("training cache", Path(raw).expanduser()))

    seen: set[str] = set()
    cache_entries: list[dict[str, Any]] = []
    for label, path in paths:
        try:
            resolved = str(path.expanduser())
        except Exception:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        kib = _du_kib(Path(resolved))
        if kib is None:
            continue
        cache_entries.append(
            {
                "label": label,
                "path": resolved,
                "size_gib": kib / (1024 ** 2),
            }
        )
    cache_entries.sort(key=lambda item: float(item.get("size_gib") or 0), reverse=True)

    disk_paths: list[tuple[str, Path]] = [
        ("system data", Path.home()),
        ("cwd", Path.cwd()),
    ]
    if asset_root:
        disk_paths.append(("training asset root", Path(asset_root).expanduser()))
    disk_seen: set[str] = set()
    disks: list[dict[str, Any]] = []
    for label, path in disk_paths:
        rec = _disk_record(path, label)
        if not rec:
            continue
        key = f"{rec.get('total_gib'):.2f}:{rec.get('free_gib'):.2f}:{rec.get('path')}"
        if key in disk_seen:
            continue
        disk_seen.add(key)
        disks.append(rec)

    return {
        "kind": "storage_pressure",
        "trigger": str(message or ""),
        "disks": disks,
        "cache_entries": cache_entries[: max(1, int(limit))],
    }


def format_storage_diagnosis(diagnosis: dict[str, Any]) -> str:
    lines: list[str] = []
    disks = diagnosis.get("disks") if isinstance(diagnosis, dict) else []
    if isinstance(disks, list) and disks:
        lines.append("Drive pressure:")
        for disk in disks:
            if not isinstance(disk, dict):
                continue
            lines.append(
                "  "
                f"{disk.get('label', 'drive')}: "
                f"{_fmt_gib(float(disk.get('free_gib') or 0))} free / "
                f"{_fmt_gib(float(disk.get('total_gib') or 0))} total "
                f"({float(disk.get('used_pct') or 0):.0f}% used)  "
                f"{disk.get('path', '')}"
            )
    entries = diagnosis.get("cache_entries") if isinstance(diagnosis, dict) else []
    if isinstance(entries, list) and entries:
        if lines:
            lines.append("")
        lines.append("Largest cache/system folders:")
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            lines.append(
                "  "
                f"{_fmt_gib(float(entry.get('size_gib') or 0)):>8}  "
                f"{entry.get('label', 'cache')}: {entry.get('path', '')}"
            )
    if not lines:
        return "No local storage diagnosis was available."
    return "\n".join(lines)
