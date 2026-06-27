from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QBrush
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .path_actions import reveal_in_file_manager
from .time_format import format_datetime_text, format_duration_seconds
from .cvops_theme import cvops_qcolor


def _status_background(status: str):
    key = str(status or "").strip().lower()
    if key == "error":
        return cvops_qcolor("accent_alert", 34)
    if key == "ready":
        return cvops_qcolor("accent_select", 32)
    if key == "trained":
        return cvops_qcolor("accent_select", 24)
    if key == "metrics_only":
        return cvops_qcolor("accent_active", 22)
    if key == "partial":
        return cvops_qcolor("accent_active", 14)
    if key in {"canceled", "cancelled", "interrupted"}:
        return cvops_qcolor("accent_warn", 28)
    return cvops_qcolor("bg_panel", 0)


def _text_or_na(value: object) -> str:
    if value is None:
        return "N/A"
    text = str(value).strip()
    return text if text else "N/A"


def _bool_or_na(value: object, *, applicable: bool = True) -> str:
    if not applicable or value in (None, ""):
        return "N/A"
    return "Yes" if bool(value) else "No"


def _metric_text(entry: dict[str, Any]) -> str:
    map50 = entry.get("map50")
    task = str(entry.get("task") or "")
    val_metric = entry.get("val_metric")
    try:
        if map50 not in (None, ""):
            return f"mAP50 {float(map50):.4f}"
    except Exception:
        return _text_or_na(map50)
    if val_metric not in (None, ""):
        try:
            val_s = f"{float(val_metric):.4f}"
        except Exception:
            val_s = str(val_metric)
        if task == "regression":
            return f"val_mae {val_s}"
        if task == "classification":
            return f"val_acc {val_s}"
        return _text_or_na(val_s)
    return "N/A"


def _duration_text(entry: dict[str, Any]) -> str:
    return format_duration_seconds(entry.get("training_duration_seconds"), empty="N/A")


class ScenarioHistoryPanel(QFrame):
    runSelected = pyqtSignal(object)

    def __init__(
        self,
        *,
        http_get: Callable[[str], dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._http_get = http_get
        self._scenario = ""
        self._all_entries: list[dict[str, Any]] = []
        self._entries: list[dict[str, Any]] = []
        self._search_value = ""

        self.setFrameShape(QFrame.Shape.NoFrame)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(5)

        head = QHBoxLayout()
        title = QLabel("Model History")
        title.setProperty("isTitle", True)
        title.setStyleSheet("font-size: 10px; font-weight: 600; border: none;")
        head.addWidget(title, stretch=0)
        head.addStretch(1)
        self._reload_btn = QPushButton("Reload History")
        self._reload_btn.clicked.connect(lambda: self.reload())
        head.addWidget(self._reload_btn)
        outer.addLayout(head)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.setSpacing(5)
        filter_lbl = QLabel("Search")
        filter_lbl.setStyleSheet("font-size: 9px;")
        filter_row.addWidget(filter_lbl)
        self._search_input = QLineEdit()
        self._search_input.setPlaceholderText("Version, date, status, or model name")
        self._search_input.textChanged.connect(self._on_search_changed)
        filter_row.addWidget(self._search_input, stretch=1)
        outer.addLayout(filter_row)

        self._status = QLabel("No scenario selected.")
        self._status.setStyleSheet("font-size: 10px;")
        self._status.setWordWrap(True)
        outer.addWidget(self._status)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            ["Version", "Status", "Metric", "Trained", "Duration", "Verified", "Base Model", "Artifacts"]
        )
        self._table.setStyleSheet("QHeaderView::section { font-weight: 600; }")
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        outer.addWidget(self._table, stretch=1)

        self._detail = QLabel("")
        self._detail.setWordWrap(True)
        self._detail.setStyleSheet("font-size: 10px;")
        outer.addWidget(self._detail)

        actions = QHBoxLayout()
        self._reveal_run_btn = QPushButton("Reveal Run")
        self._reveal_run_btn.clicked.connect(self._reveal_run)
        actions.addWidget(self._reveal_run_btn)
        self._reveal_weights_btn = QPushButton("Reveal Weights")
        self._reveal_weights_btn.clicked.connect(self._reveal_weights)
        actions.addWidget(self._reveal_weights_btn)
        self._reveal_data_btn = QPushButton("Reveal Data YAML")
        self._reveal_data_btn.clicked.connect(self._reveal_data_yaml)
        actions.addWidget(self._reveal_data_btn)
        actions.addStretch(1)
        outer.addLayout(actions)

        self._set_action_state(None)

    def clear(self) -> None:
        self._scenario = ""
        self._all_entries = []
        self._entries = []
        self._search_input.blockSignals(True)
        self._search_input.clear()
        self._search_input.blockSignals(False)
        self._search_value = ""
        self._status.setText("No scenario selected.")
        self._detail.clear()
        self._table.setRowCount(0)
        self._set_action_state(None)
        self.runSelected.emit(None)

    def load_scenario(self, scenario: str, *, preferred_version: str = "") -> None:
        scenario = str(scenario or "").strip()
        if not scenario:
            self.clear()
            return
        self._scenario = scenario
        self.reload(preferred_version=preferred_version)

    def reload(self, *, preferred_version: str = "") -> None:
        if not self._scenario:
            self.clear()
            return
        current = self.current_entry()
        current_version = str(current.get("version") or "") if isinstance(current, dict) else ""
        try:
            payload = self._http_get(f"/scenarios/{self._scenario}/history")
        except Exception as exc:
            self._all_entries = []
            self._entries = []
            self._table.setRowCount(0)
            self._detail.setText("")
            self._status.setText(f"Unable to load history: {exc}")
            self._set_action_state(None)
            self.runSelected.emit(None)
            return

        runs = payload.get("runs") if isinstance(payload, dict) else []
        self._all_entries = [dict(item) for item in (runs or []) if isinstance(item, dict)]
        self._apply_filters(preferred_version=preferred_version or current_version)

    def current_entry(self) -> Optional[dict[str, Any]]:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._entries):
            return None
        return dict(self._entries[row])

    @staticmethod
    def _entry_sort_key(entry: dict[str, Any]) -> tuple[int, str, str]:
        raw_version = entry.get("version_number")
        try:
            version_number = int(raw_version)
        except Exception:
            version_number = -1
        trained_at = str(entry.get("trained_at") or "")
        version = str(entry.get("version") or "")
        return (version_number, trained_at, version)

    def _apply_filters(self, *, preferred_version: str = "") -> None:
        entries = sorted(self._all_entries, key=self._entry_sort_key, reverse=True)
        needle = self._search_value.strip().lower()
        if needle:
            filtered: list[dict[str, Any]] = []
            for entry in entries:
                base_model = str(entry.get("base_model") or "")
                haystack = " ".join(
                    [
                        str(entry.get("version") or ""),
                        str(entry.get("status") or ""),
                        str(entry.get("trained_at") or ""),
                        Path(base_model).name if base_model else "",
                        str(entry.get("run_dir") or ""),
                    ]
                ).lower()
                if needle in haystack:
                    filtered.append(entry)
            entries = filtered
        self._entries = entries
        total_count = len(self._all_entries)
        visible_count = len(self._entries)
        if total_count == 0:
            self._status.setText(f"{self._scenario}: no model versions recorded yet.")
        elif visible_count == 0:
            self._status.setText(
                f"{self._scenario}: 0 of {total_count} model version(s) match '{self._search_value}'. Sorted newest -> oldest."
            )
        elif visible_count == total_count:
            self._status.setText(
                f"{self._scenario}: {visible_count} model version(s) since inception. Sorted newest -> oldest."
            )
        else:
            self._status.setText(
                f"{self._scenario}: showing {visible_count} of {total_count} model version(s). Sorted newest -> oldest."
            )
        self._render_table()
        self._select_preferred(preferred_version=preferred_version)

    def _render_table(self) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(len(self._entries))
        text_brush = QBrush(cvops_qcolor("text_signal"))
        for row, entry in enumerate(self._entries):
            status = str(entry.get("status") or "")
            brush = QBrush(_status_background(status))
            applicable_verified = status.strip().lower() not in {
                "partial",
                "error",
                "empty",
                "canceled",
                "cancelled",
                "interrupted",
            }
            base_model = Path(str(entry.get("base_model") or "")).name if entry.get("base_model") else ""
            values = [
                _text_or_na(entry.get("version")),
                _text_or_na(status),
                _metric_text(entry),
                format_datetime_text(entry.get("trained_at"), seconds=True, empty="N/A"),
                _duration_text(entry),
                _bool_or_na(entry.get("verified"), applicable=applicable_verified),
                _text_or_na(base_model),
                _text_or_na(entry.get("artifact_count")) if entry.get("artifact_count") is not None else "N/A",
            ]
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setBackground(brush)
                item.setForeground(text_brush)
                self._table.setItem(row, col, item)
        self._table.blockSignals(False)

    def refresh_theme_styles(self) -> None:
        text_brush = QBrush(cvops_qcolor("text_signal"))
        for row in range(self._table.rowCount()):
            status_item = self._table.item(row, 1)
            brush = QBrush(_status_background(status_item.text() if status_item is not None else ""))
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item is None:
                    continue
                item.setBackground(brush)
                item.setForeground(text_brush)

    def _select_preferred(self, *, preferred_version: str = "") -> None:
        target = str(preferred_version or "").strip()
        if not target:
            latest = next((idx for idx, entry in enumerate(self._entries) if entry.get("is_latest")), -1)
            target_row = latest if latest >= 0 else (len(self._entries) - 1)
        else:
            target_row = next(
                (idx for idx, entry in enumerate(self._entries) if str(entry.get("version") or "") == target),
                -1,
            )
            if target_row < 0:
                target_row = len(self._entries) - 1
        if target_row >= 0:
            # If the preferred row is already selected (common when switching scenarios),
            # Qt may not re-emit itemSelectionChanged; ensure downstream panels refresh.
            prev_row = int(self._table.currentRow())
            self._table.selectRow(target_row)
            if prev_row == target_row:
                self._on_selection_changed()
            return
        if self._entries:
            self._detail.setText("Select a model version to inspect final results.")
        else:
            self._detail.setText("No model versions match the current search.")
        self._set_action_state(None)
        self.runSelected.emit(None)

    def _on_selection_changed(self) -> None:
        entry = self.current_entry()
        self._set_action_state(entry)
        if not entry:
            self._detail.setText("Select a model version to inspect final results.")
            self.runSelected.emit(None)
            return
        base_model = str(entry.get("base_model") or "")
        status = str(entry.get("status") or "")
        applicable_verified = status.strip().lower() not in {
            "partial",
            "error",
            "empty",
            "canceled",
            "cancelled",
            "interrupted",
        }
        metric_line = _metric_text(entry)
        duration_line = _duration_text(entry)
        detail_lines = [
            f"Selected {_text_or_na(entry.get('version'))}  |  status={_text_or_na(status)}  |  artifacts={_text_or_na(entry.get('artifact_count')) if entry.get('artifact_count') is not None else 'N/A'}",
            f"trained_at: {format_datetime_text(entry.get('trained_at'), seconds=True, empty='N/A')}  |  duration: {duration_line}  |  metric: {metric_line}  |  verified: {_bool_or_na(entry.get('verified'), applicable=applicable_verified)}",
            f"base_model: {_text_or_na(Path(base_model).name if base_model else '')}",
            f"run_dir: {_text_or_na(entry.get('run_dir'))}",
        ]
        error_text = str(entry.get("error") or "").strip()
        if error_text:
            detail_lines.append(f"stop_reason: {error_text}")
        self._detail.setText("\n".join(detail_lines))
        self.runSelected.emit(dict(entry))

    def _set_action_state(self, entry: Optional[dict[str, Any]]) -> None:
        run_dir = str(entry.get("run_dir") or "") if entry else ""
        weights = str(entry.get("weights") or "") if entry else ""
        data_yaml = str(entry.get("data_yaml") or "") if entry else ""
        self._reveal_run_btn.setEnabled(bool(run_dir))
        self._reveal_weights_btn.setEnabled(bool(weights))
        self._reveal_data_btn.setEnabled(bool(data_yaml))

    def _reveal_selected_path(self, key: str) -> None:
        entry = self.current_entry()
        if not entry:
            return
        path_value = str(entry.get(key) or "")
        if not path_value:
            return
        try:
            reveal_in_file_manager(path_value)
        except Exception as exc:
            self._status.setText(f"Reveal failed: {exc}")

    def _reveal_run(self) -> None:
        self._reveal_selected_path("run_dir")

    def _reveal_weights(self) -> None:
        self._reveal_selected_path("weights")

    def _reveal_data_yaml(self) -> None:
        self._reveal_selected_path("data_yaml")

    def _on_search_changed(self, value: str) -> None:
        self._search_value = str(value or "")
        current = self.current_entry()
        current_version = str(current.get("version") or "") if isinstance(current, dict) else ""
        self._apply_filters(preferred_version=current_version)
