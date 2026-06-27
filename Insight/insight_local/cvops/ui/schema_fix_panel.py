from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SchemaFixPanel(QWidget):
    """Inline (non-modal) UI for selecting label_col and feature_cols for tabular CSV training."""

    revealRequested = pyqtSignal(str)  # dataset_csv
    applyRequested = pyqtSignal(object, bool)  # patch, rerun

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scenario = ""
        self._dataset_csv = ""
        self._attempted_label_col = ""
        self._columns: list[str] = []
        self._suggested: list[str] = []

        self._build_ui()

    # ---- public API ----

    def set_context(
        self,
        *,
        scenario: str,
        dataset_csv: str,
        attempted_label_col: str,
        columns: list[str],
        suggested_label_cols: list[str],
        current_label_col: str = "",
        current_feature_cols: Optional[list[str]] = None,
    ) -> None:
        self._scenario = str(scenario or "")
        self._dataset_csv = str(dataset_csv or "")
        self._attempted_label_col = str(attempted_label_col or "")
        self._columns = [str(c) for c in (columns or []) if str(c).strip()]
        self._suggested = [str(c) for c in (suggested_label_cols or []) if str(c).strip()]

        self._meta.setText(
            f"scenario: {self._scenario or '—'}    |    dataset_csv: {self._dataset_csv or '—'}"
            + (f"    |    configured label_col: {current_label_col}" if current_label_col else "")
        )

        self._label_combo.blockSignals(True)
        self._label_combo.clear()
        for c in self._columns:
            self._label_combo.addItem(c, c)
        self._label_combo.blockSignals(False)

        default = str(current_label_col or "").strip()
        if not default:
            for cand in self._suggested:
                if cand in self._columns:
                    default = cand
                    break
        if not default and self._columns:
            default = self._columns[-1]
        if default:
            idx = self._label_combo.findData(default)
            if idx >= 0:
                self._label_combo.setCurrentIndex(idx)

        # Features list
        self._feat_list.blockSignals(True)
        self._feat_list.clear()
        for c in self._columns:
            it = QListWidgetItem(c)
            it.setData(Qt.ItemDataRole.UserRole, c)
            self._feat_list.addItem(it)
        self._feat_list.blockSignals(False)

        desired = set(str(c).strip() for c in (current_feature_cols or []) if str(c).strip())
        if desired:
            for i in range(self._feat_list.count()):
                it = self._feat_list.item(i)
                if it is None:
                    continue
                col = str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "")
                it.setSelected(col in desired)
        else:
            self.select_all_features()

        self._status.setText("")
        self._set_enabled(bool(self._columns))

    def dataset_csv(self) -> str:
        return self._dataset_csv

    def chosen_label_col(self) -> str:
        return str(self._label_combo.currentData() or self._label_combo.currentText() or "").strip()

    def chosen_feature_cols(self) -> list[str]:
        label = self.chosen_label_col()
        selected = []
        for it in self._feat_list.selectedItems():
            selected.append(str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "").strip())
        return [c for c in selected if c and c != label]

    def set_status(self, text: str) -> None:
        self._status.setText(str(text or ""))

    def select_all_features(self) -> None:
        for i in range(self._feat_list.count()):
            it = self._feat_list.item(i)
            if it is not None:
                it.setSelected(True)

    def select_none_features(self) -> None:
        self._feat_list.clearSelection()

    def set_buttons_enabled(self, enabled: bool) -> None:
        for b in (self._btn_reveal, self._btn_apply, self._btn_apply_rerun):
            b.setEnabled(bool(enabled))

    # ---- UI + callbacks ----

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        title = QLabel("Labeling (tabular ML)")
        title.setStyleSheet("font-weight: 700;")
        root.addWidget(title)

        info = QLabel(
            "Pick the label (target) column, and optionally choose feature columns.\n"
            "If you are unsure, set the label to the column you want to predict (often the last column)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.8);")
        root.addWidget(info)

        self._meta = QLabel("scenario: —    |    dataset_csv: —")
        self._meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._meta.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        root.addWidget(self._meta)

        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label column:"))
        self._label_combo = QComboBox()
        self._label_combo.setMinimumWidth(260)
        label_row.addWidget(self._label_combo, stretch=1)
        root.addLayout(label_row)

        feat_title = QLabel("Feature columns (multi-select):")
        feat_title.setStyleSheet("font-weight: 600;")
        root.addWidget(feat_title)

        feat_row = QHBoxLayout()
        self._feat_list = QListWidget()
        self._feat_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._feat_list.setMinimumHeight(160)
        feat_row.addWidget(self._feat_list, stretch=1)

        btns = QVBoxLayout()
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self.select_all_features)
        self._select_none_btn = QPushButton("Select None")
        self._select_none_btn.clicked.connect(self.select_none_features)
        btns.addWidget(self._select_all_btn)
        btns.addWidget(self._select_none_btn)
        btns.addStretch(1)
        feat_row.addLayout(btns)
        root.addLayout(feat_row)

        bottom = QHBoxLayout()
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: rgba(239,68,68,0.9);")
        bottom.addWidget(self._status, stretch=1)

        self._btn_reveal = QPushButton("Reveal CSV")
        self._btn_reveal.clicked.connect(self._on_reveal)
        self._btn_apply = QPushButton("Apply Only")
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_apply_rerun = QPushButton("Apply & Re-run Training")
        self._btn_apply_rerun.setDefault(True)
        self._btn_apply_rerun.clicked.connect(self._on_apply_rerun)
        bottom.addWidget(self._btn_reveal)
        bottom.addWidget(self._btn_apply)
        bottom.addWidget(self._btn_apply_rerun)
        root.addLayout(bottom)

    def _set_enabled(self, enabled: bool) -> None:
        self._label_combo.setEnabled(enabled)
        self._feat_list.setEnabled(enabled)
        self._select_all_btn.setEnabled(enabled)
        self._select_none_btn.setEnabled(enabled)
        self.set_buttons_enabled(enabled)

    def _make_patch(self) -> dict:
        label = self.chosen_label_col()
        feat_cols = self.chosen_feature_cols()
        patch: dict = {}
        if label:
            patch["label_col"] = label
        if feat_cols:
            patch["feature_cols"] = feat_cols
        return patch

    def _on_reveal(self) -> None:
        self.revealRequested.emit(self._dataset_csv)

    def _on_apply(self) -> None:
        self.applyRequested.emit(self._make_patch(), False)

    def _on_apply_rerun(self) -> None:
        self.applyRequested.emit(self._make_patch(), True)
