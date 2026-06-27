from __future__ import annotations

from PyQt6.QtCore import QObject, pyqtSignal


class SessionUiAdapter(QObject):
    """Qt bridge for payloads emitted from non-UI runtime threads."""

    payload_ready = pyqtSignal(dict)

    def emit_payload(self, payload: dict) -> None:
        self.payload_ready.emit(payload)

    def emit_many(self, payloads: list[dict]) -> None:
        for payload in payloads:
            self.payload_ready.emit(payload)
