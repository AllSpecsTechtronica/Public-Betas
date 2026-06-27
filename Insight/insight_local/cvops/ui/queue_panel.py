from __future__ import annotations

import time
from typing import Any, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...ui.theme import current_color_scheme, is_aurora_family_scheme

from .time_format import format_duration_seconds, format_timestamp
from .cvops_theme import cvops_qcolor


COLUMNS = ["Job ID", "Type", "Scenario", "State", "Source", "Created", "Finished", "Elapsed"]


def _state_background(state: str):
    key = str(state or "").strip().lower()
    if key == "error":
        return cvops_qcolor("accent_alert", 34)
    if is_aurora_family_scheme(current_color_scheme()) and key in {"failed", "failure", "canceled", "cancelled"}:
        return cvops_qcolor("accent_alert", 34)
    if key == "running":
        return cvops_qcolor("accent_active", 30)
    if key == "done":
        return cvops_qcolor("accent_select", 30)
    if key == "queued":
        return cvops_qcolor("accent_select", 18)
    return cvops_qcolor("bg_panel", 0)


def _fmt_ts(value: Any) -> str:
    return format_timestamp(value, seconds=True, empty="")


def _fmt_elapsed(created_at: Any, started_at: Any, finished_at: Any, state: str) -> str:
    try:
        created = float(created_at)
    except Exception:
        created = 0.0
    try:
        start = float(started_at)
    except Exception:
        start = 0.0
    if created <= 0 and start <= 0:
        return ""
    state_key = str(state or "").strip().lower()
    now = time.time()
    if finished_at not in (None, "", 0, 0.0):
        try:
            end = float(finished_at)
        except Exception:
            end = now
    elif state_key in {"running", "queued"}:
        end = now
    else:
        end = start if start > 0 else created

    if start <= 0:
        return f"queued {format_duration_seconds(max(0.0, end - created))}"

    queue_wait = max(0.0, start - created) if created > 0 else 0.0
    run_time = max(0.0, end - start)
    run_text = format_duration_seconds(run_time)
    if queue_wait >= 1:
        return f"wait {format_duration_seconds(queue_wait)} | run {run_text}"
    return f"run {run_text}"


class QueuePanel(QWidget):
    """Live queue table. Seeds from GET /jobs, updates from ws job_status events."""

    jobSelected = pyqtSignal(str)
    cancelRequested = pyqtSignal(str)
    retryRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._row_by_id: dict[str, int] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        self._cancel_btn = QPushButton("Cancel job")
        self._cancel_btn.clicked.connect(self._emit_cancel)
        actions.addWidget(self._cancel_btn)
        self._retry_btn = QPushButton("Retry failed job")
        self._retry_btn.clicked.connect(self._emit_retry)
        actions.addWidget(self._retry_btn)
        actions.addStretch(1)
        layout.addLayout(actions)

        self._table = QTableWidget(0, len(COLUMNS))
        self._table.setHorizontalHeaderLabels(COLUMNS)
        self._table.setStyleSheet("QHeaderView::section { font-weight: 600; }")
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self._table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self._table, stretch=1)

    def seed_jobs(self, jobs: list[dict[str, Any]]) -> None:
        self._row_by_id.clear()
        self._table.setRowCount(0)
        # Most recent first assumed from server; we insert at top to preserve that.
        for job in jobs:
            self.upsert_job(job, prepend=False)
        self._table.resizeColumnsToContents()

    def upsert_job(self, job: dict[str, Any], prepend: bool = True) -> None:
        job_id = str(job.get("job_id") or "")
        if not job_id:
            return
        row = self._row_by_id.get(job_id)
        if row is None:
            row = 0 if prepend else self._table.rowCount()
            self._table.insertRow(row)
            if prepend:
                # Shift existing ids down by 1
                self._row_by_id = {k: (v + 1) for k, v in self._row_by_id.items()}
            self._row_by_id[job_id] = row
        state = str(job.get("state", ""))
        values = [
            job_id,
            str(job.get("job_type", "")),
            str(job.get("scenario", "")),
            state,
            str(job.get("source", "")),
            _fmt_ts(job.get("created_at")),
            _fmt_ts(job.get("finished_at")),
            _fmt_elapsed(job.get("created_at"), job.get("started_at"), job.get("finished_at"), state),
        ]
        brush = QBrush(_state_background(state))
        text_brush = QBrush(cvops_qcolor("text_signal"))
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            item.setBackground(brush)
            item.setForeground(text_brush)
            self._table.setItem(row, col, item)

    def refresh_theme_styles(self) -> None:
        text_brush = QBrush(cvops_qcolor("text_signal"))
        for row in range(self._table.rowCount()):
            state_item = self._table.item(row, 3)
            brush = QBrush(_state_background(state_item.text() if state_item is not None else ""))
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item is None:
                    continue
                item.setBackground(brush)
                item.setForeground(text_brush)

    def selected_job_id(self) -> str:
        items = self._table.selectedItems()
        if not items:
            return ""
        row = items[0].row()
        for job_id, r in self._row_by_id.items():
            if r == row:
                return job_id
        return ""

    def select_job(self, job_id: str) -> None:
        row = self._row_by_id.get(job_id)
        if row is None:
            return
        item = self._table.item(row, 0)
        if item is not None:
            self._table.setCurrentItem(item)

    def _emit_cancel(self) -> None:
        job_id = self.selected_job_id()
        if job_id:
            self.cancelRequested.emit(job_id)

    def _emit_retry(self) -> None:
        job_id = self.selected_job_id()
        if job_id:
            self.retryRequested.emit(job_id)

    def _on_selection(self) -> None:
        job_id = self.selected_job_id()
        if job_id:
            self.jobSelected.emit(job_id)
