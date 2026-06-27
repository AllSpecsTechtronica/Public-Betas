from __future__ import annotations

import re
import os
from collections import deque
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QTextCharFormat, QColor, QTextCursor
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QScrollArea,
)

from ...ui.theme import current_color_scheme, is_aurora_family_scheme, text_qcolor, theme_hex
from .cvops_theme import cvops_qcolor, repolish
from .training_graph import TrainingGraphWidget

_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_MAX_LINES = 2000


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text or "")


class TrainingConsoleWidget(QFrame):
    """Read-only monospace console that surfaces Ultralytics/training log lines."""

    stop_requested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, *, terminal_only: bool = False) -> None:
        super().__init__(parent)
        self._terminal_only = bool(terminal_only)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 5, 6, 6)
        outer.setSpacing(5)

        head = QHBoxLayout()
        title = QLabel("Training Console")
        title.setProperty("isTitle", True)
        head.addWidget(title, stretch=0)
        head.addStretch(1)
        self._autoscroll = QCheckBox("Auto-scroll")
        self._autoscroll.setChecked(True)
        head.addWidget(self._autoscroll)
        copy_btn = QPushButton("Copy All")
        copy_btn.clicked.connect(self._copy_all)
        head.addWidget(copy_btn)
        self._stop_btn = QPushButton("Stop & Save")
        self._stop_btn.clicked.connect(self._on_stop_clicked)
        self._stop_btn.setEnabled(False)
        head.addWidget(self._stop_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.reset)
        head.addWidget(clear_btn)
        outer.addLayout(head)

        self._view = QTextEdit()
        self._view.setObjectName("logView")
        self._view.setReadOnly(True)
        font = QFont("JetBrains Mono")
        if not font.exactMatch():
            font = QFont("IBM Plex Mono")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(11)
        self._view.setFont(font)
        self._view.setMinimumHeight(96)
        outer.addWidget(self._view)

        self._buffer: deque[tuple[str, str]] = deque(maxlen=_MAX_LINES)
        self._graph: Optional[TrainingGraphWidget] = None
        self._health: Optional[QLabel] = None
        self._health_info: Optional[QTextEdit] = None
        self._metrics_text: Optional[QTextEdit] = None

        if not self._terminal_only:
            graph_metrics_row = QHBoxLayout()
            graph_metrics_row.setSpacing(8)
            graph_metrics_row.setContentsMargins(0, 0, 0, 0)

            self._graph = TrainingGraphWidget()
            self._graph.setMinimumHeight(110)
            graph_metrics_row.addWidget(self._graph, stretch=2)

            health_panel_scroll = QScrollArea()
            health_panel_scroll.setWidgetResizable(True)
            health_panel_scroll.setMinimumWidth(220)
            health_panel_scroll.setMinimumHeight(110)
            health_panel_container = QWidget()
            health_panel_layout = QVBoxLayout(health_panel_container)
            health_panel_layout.setContentsMargins(8, 8, 8, 8)
            health_panel_layout.setSpacing(6)

            health_title = QLabel("MODEL HEALTH")
            health_title.setStyleSheet("font-weight: 700; font-size: 11px;")
            health_panel_layout.addWidget(health_title)

            self._health = QLabel("HEALTH: IDLE")
            self._health.setObjectName("trainingHealthBadge")
            self._health.setProperty("healthState", "idle")
            self._health.setStyleSheet("font-weight: 600; font-size: 12px;")
            health_panel_layout.addWidget(self._health)

            mono_font = QFont("JetBrains Mono", 9)
            if not mono_font.exactMatch():
                mono_font = QFont("IBM Plex Mono", 9)
            mono_font.setStyleHint(QFont.StyleHint.Monospace)

            self._health_info = QTextEdit()
            self._health_info.setObjectName("healthInfoPanel")
            self._health_info.setReadOnly(True)
            self._health_info.setFont(mono_font)
            self._health_info.setMinimumHeight(90)
            self._health_info.setPlainText("No training data yet.")
            health_panel_layout.addWidget(self._health_info, stretch=1)
            health_panel_container.setLayout(health_panel_layout)
            health_panel_scroll.setWidget(health_panel_container)
            graph_metrics_row.addWidget(health_panel_scroll, stretch=0)

            metrics_scroll = QScrollArea()
            metrics_scroll.setWidgetResizable(True)
            metrics_scroll.setMinimumWidth(220)
            metrics_scroll.setMinimumHeight(110)
            metrics_container = QWidget()
            metrics_layout = QVBoxLayout(metrics_container)
            metrics_layout.setContentsMargins(8, 8, 8, 8)
            metrics_layout.setSpacing(6)

            metrics_title = QLabel("LIVE METRICS")
            metrics_title.setStyleSheet("font-weight: 700; font-size: 11px;")
            metrics_layout.addWidget(metrics_title)

            self._metrics_text = QTextEdit()
            self._metrics_text.setObjectName("metricsPanel")
            self._metrics_text.setReadOnly(True)
            self._metrics_text.setFont(mono_font)
            self._metrics_text.setMinimumHeight(100)
            metrics_layout.addWidget(self._metrics_text)
            metrics_container.setLayout(metrics_layout)
            metrics_scroll.setWidget(metrics_container)
            graph_metrics_row.addWidget(metrics_scroll, stretch=0)

            outer.addLayout(graph_metrics_row)

    def reset(self) -> None:
        self._buffer.clear()
        self._view.clear()
        if self._graph is not None:
            self._graph.clear()
        if self._metrics_text is not None:
            self._metrics_text.setPlainText("No live metrics yet.")
        if self._health_info is not None:
            self._health_info.setPlainText("No training data yet.")
        self._stop_btn.setEnabled(False)
        self.set_health("idle")

    def set_training_active(self, active: bool) -> None:
        """Enable/disable the Stop & Save button based on training state."""
        self._stop_btn.setEnabled(bool(active))

    def append_line(self, line: str, stream: str = "stdout") -> None:
        clean = _strip_ansi(str(line or "")).rstrip("\r\n")
        if not clean and stream != "stderr":
            return
        evicted = len(self._buffer) == self._buffer.maxlen
        self._buffer.append((clean, stream))
        if evicted:
            # Re-render full buffer to drop the oldest line cheaply enough.
            self._rerender()
        else:
            self._write_line(clean, stream)
        if self._autoscroll.isChecked():
            sb = self._view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def set_lines(self, lines: list[tuple[str, str]]) -> None:
        self._buffer.clear()
        for line, stream in lines[-_MAX_LINES:]:
            self._buffer.append((line, stream))
        self._rerender()

    def set_points(self, points: list[dict]) -> None:
        if self._graph is not None:
            self._graph.set_points(points)

    def set_metrics_text(self, text: str) -> None:
        """Update the scrollable metrics panel with formatted training metrics."""
        if self._metrics_text is None:
            return
        self._metrics_text.setPlainText(str(text or "No live metrics yet."))
        sb = self._metrics_text.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def set_health(self, state: str) -> None:
        key = str(state or "idle").strip().lower()
        palette = {
            "healthy": "HEALTH: HEALTHY",
            "watch": "HEALTH: WATCH",
            "at_risk": "HEALTH: AT RISK",
            "completed": "HEALTH: COMPLETED",
            "failed": "HEALTH: FAILED",
            "cancelled": "HEALTH: CANCELLED",
            "starting": "HEALTH: STARTING",
            "idle": "HEALTH: IDLE",
        }
        if self._health is None:
            return
        self._health.setText(palette.get(key, palette["idle"]))
        self._health.setProperty("healthState", key)
        repolish(self._health)

    def set_health_info(self, info_text: str) -> None:
        """Update the health info panel with detailed status."""
        if self._health_info is not None:
            self._health_info.setPlainText(str(info_text or "N/A"))

    def focus_for_training(self) -> None:
        self._view.setFocus(Qt.FocusReason.OtherFocusReason)
        if self._autoscroll.isChecked():
            sb = self._view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _rerender(self) -> None:
        self._view.clear()
        for line, stream in self._buffer:
            self._write_line(line, stream)
        if self._autoscroll.isChecked():
            sb = self._view.verticalScrollBar()
            sb.setValue(sb.maximum())

    def _write_line(self, line: str, stream: str) -> None:
        cursor = self._view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        scheme = current_color_scheme()
        aurora_family = is_aurora_family_scheme(scheme)
        japandi = str(os.environ.get("INSIGHT_CVOPS_JAPANDI", "0")).strip().lower() not in {"0", "false", "off"}
        red_black_default = "0" if scheme in ("marathon", "wear_marathon") or aurora_family else "1"
        red_black = (not japandi) and str(os.environ.get("INSIGHT_CVOPS_RED_BLACK", red_black_default)).strip().lower() not in {"0", "false", "off"}
        if stream == "stderr":
            if scheme in ("marathon", "wear_marathon") or aurora_family:
                fmt.setForeground(cvops_qcolor("accent_alert"))
            else:
                fmt.setForeground(QColor(theme_hex("privacy_warn")))
        elif japandi:
            fmt.setForeground(text_qcolor(0.94))
        elif red_black:
            fmt.setForeground(text_qcolor(0.94))
        elif scheme in ("marathon", "wear_marathon") or aurora_family:
            fmt.setForeground(cvops_qcolor("text_signal"))
        else:
            fmt.setForeground(text_qcolor(0.86))
        cursor.insertText(line + "\n", fmt)
        self._view.setTextCursor(cursor)

    def _on_stop_clicked(self) -> None:
        """Handle Stop & Save button click."""
        self._stop_btn.setEnabled(False)
        self.stop_requested.emit()

    def _copy_all(self) -> None:
        text = "\n".join(line for line, _ in self._buffer)
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)
