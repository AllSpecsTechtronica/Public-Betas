"""cell_progress_panel.py — Colab-style cell execution display.

Shows backbone cells running sequentially, each with:
  - Status indicator:  [>] running  [OK] done  [ERR] error  [--] pending/skipped
  - Cell name + elapsed time
  - Collapsible output text area (auto-expands when content arrives)

Styled in Solarized Dark with green/cyan/red accents consistent with the rest
of cvops UI.
"""
from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .cvops_theme import repolish
_STATUS_LABELS = {
    "running":  "[>]",
    "done":     "[OK]",
    "error":    "[ERR]",
    "skipped":  "[--]",
    "pending":  "[--]",
}


class _CellCard(QFrame):
    """Single cell card — status icon, name, elapsed, and output text."""

    _OUTPUT_MIN_H = 40
    _OUTPUT_MAX_H = 220

    def __init__(self, index: int, cell_name: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._index = index
        self._cell_name = cell_name
        self._status = "pending"
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("cellCard")
        self.setProperty("cellStatus", "pending")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(3)

        header = QHBoxLayout()
        header.setSpacing(6)

        self._icon = QLabel("[--]")
        self._icon.setObjectName("cellIcon")
        self._icon.setFixedWidth(36)
        header.addWidget(self._icon)

        self._name_lbl = QLabel(cell_name)
        self._name_lbl.setObjectName("cellName")
        header.addWidget(self._name_lbl, stretch=1)

        self._elapsed_lbl = QLabel("")
        self._elapsed_lbl.setObjectName("cellElapsed")
        self._elapsed_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._elapsed_lbl)

        outer.addLayout(header)

        self._output = QTextEdit()
        self._output.setObjectName("cellOutput")
        self._output.setReadOnly(True)
        self._output.setVisible(False)
        self._output.setMinimumHeight(self._OUTPUT_MIN_H)
        self._output.setMaximumHeight(self._OUTPUT_MAX_H)
        self._output.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        outer.addWidget(self._output)

    def update_status(
        self,
        status: str,
        output: str = "",
        elapsed_ms: float = 0,
    ) -> None:
        self._status = status
        icon_text = _STATUS_LABELS.get(status, "[--]")

        self._icon.setText(icon_text)
        self.setProperty("cellStatus", status)
        self._icon.setProperty("cellStatus", status)
        repolish(self)
        repolish(self._icon)

        if elapsed_ms > 0:
            if elapsed_ms >= 1000:
                self._elapsed_lbl.setText(f"{elapsed_ms / 1000:.1f}s")
            else:
                self._elapsed_lbl.setText(f"{elapsed_ms:.0f}ms")
        else:
            self._elapsed_lbl.setText("" if status == "pending" else "...")

        if output:
            self._output.setPlainText(output)
            self._output.setVisible(True)
        elif status in ("running",):
            self._output.setVisible(True)
            if not self._output.toPlainText():
                self._output.setPlainText("running...")


class CellProgressPanel(QWidget):
    """Scrollable list of cell cards updated by WebSocket cell_progress events."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._current_job_id = ""
        self._cards: dict[int, _CellCard] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._header = QLabel("No job selected.")
        self._header.setObjectName("cellHeader")
        layout.addWidget(self._header)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        self._card_container = QWidget()
        self._card_layout = QVBoxLayout(self._card_container)
        self._card_layout.setContentsMargins(0, 0, 0, 0)
        self._card_layout.setSpacing(4)
        self._card_layout.addStretch(1)
        self._scroll.setWidget(self._card_container)
        layout.addWidget(self._scroll, stretch=1)

    def reset(self, job_id: str) -> None:
        """Clear all cards and start fresh for a new job."""
        self._current_job_id = job_id
        self._cards.clear()
        # Remove all widgets except the trailing stretch.
        while self._card_layout.count() > 1:
            item = self._card_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()
        self._header.setText(f"[CELLS]  job {job_id}")

    def apply_cell_event(self, payload: dict[str, Any]) -> None:
        """Process a cell_progress WebSocket event."""
        job_id = str(payload.get("job_id") or "")
        if job_id != self._current_job_id:
            self.reset(job_id)

        index = int(payload.get("cell_index") or 0)
        cell_name = str(payload.get("cell_name") or f"Cell {index}")
        status = str(payload.get("cell_status") or "pending")
        output = str(payload.get("output") or "")
        elapsed_ms = float(payload.get("elapsed_ms") or 0)

        if index not in self._cards:
            card = _CellCard(index, cell_name)
            # Insert before the stretch item.
            insert_pos = self._card_layout.count() - 1
            self._card_layout.insertWidget(insert_pos, card)
            self._cards[index] = card

        self._cards[index].update_status(status, output, elapsed_ms)
        # Scroll to the active card.
        if status == "running":
            self._scroll.ensureWidgetVisible(self._cards[index])
