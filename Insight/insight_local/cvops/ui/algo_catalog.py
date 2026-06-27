from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices

from ...config import ROOT_DIR


def _read_doc_first_line(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    # Try to extract module docstring first line.
    stripped = text.lstrip()
    if not stripped.startswith(("'''", '"""')):
        return ""
    quote = stripped[:3]
    end = stripped.find(quote, 3)
    if end < 0:
        return ""
    doc = stripped[3:end].strip()
    first = doc.splitlines()[0].strip() if doc else ""
    return first


def list_algo_files() -> list[dict[str, Any]]:
    """Return catalog entries for known algo cell scripts.

    Each entry: {"name": str, "path": str, "doc": str}
    Path is repo-relative POSIX when possible.
    """
    root = Path(ROOT_DIR).resolve()
    out: list[dict[str, Any]] = []

    candidates: list[Path] = []
    # Repo-root editable template(s).
    for name in ("basic_cnn_for_editing.py",):
        p = root / name
        if p.is_file():
            candidates.append(p)
    # mlops algos folder.
    algos = root / "mlops" / "algos"
    if algos.is_dir():
        for p in sorted(algos.glob("*.py"), key=lambda x: x.name.lower()):
            if p.name.startswith("_") or p.name == "__init__.py":
                continue
            candidates.append(p)

    seen: set[str] = set()
    for p in candidates:
        try:
            rel = str(p.resolve().relative_to(root)).replace("\\", "/")
        except Exception:
            rel = str(p.resolve().as_posix())
        if rel in seen:
            continue
        seen.add(rel)
        out.append(
            {
                "name": p.stem,
                "path": rel,
                "doc": _read_doc_first_line(p),
            }
        )
    return out


def reveal_in_finder(path_value: str) -> None:
    """Reveal a file in the platform file manager (Finder on macOS)."""
    root = Path(ROOT_DIR).resolve()
    p = Path(str(path_value or "").strip())
    if not p:
        return
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    if sys.platform == "darwin":
        try:
            subprocess.Popen(["open", "-R", str(p)])
            return
        except Exception:
            pass
    # Fallback: open containing folder.
    try:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(p.parent)))
    except Exception:
        pass

