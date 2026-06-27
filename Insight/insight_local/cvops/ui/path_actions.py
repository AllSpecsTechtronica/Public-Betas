from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices


def reveal_in_file_manager(path_value: str) -> None:
    target = Path(str(path_value or "").strip()).expanduser()
    if not str(target):
        raise ValueError("path is required")
    if not target.exists():
        raise FileNotFoundError(str(target))

    if sys.platform == "darwin":
        if target.is_dir():
            subprocess.Popen(["open", str(target)])
        else:
            subprocess.Popen(["open", "-R", str(target)])
        return

    if sys.platform.startswith("win"):
        if target.is_dir():
            subprocess.Popen(["explorer", str(target)])
        else:
            subprocess.Popen(["explorer", "/select,", str(target)])
        return

    open_target = target if target.is_dir() else target.parent
    if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(open_target))):
        raise RuntimeError(f"failed to open {open_target}")
