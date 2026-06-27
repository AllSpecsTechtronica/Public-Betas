from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QWidget

from .cvops_theme import repolish


class StatusPill(QLabel):
    def __init__(self, status: str = "empty", parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setMinimumWidth(96)
        self.setObjectName("statusPill")
        self.set_status(status)

    def set_status(self, status: str) -> None:
        key = (status or "empty").lower()
        self.setProperty("status", key)
        repolish(self)
        self.setText(key.replace("_", " ").upper())
