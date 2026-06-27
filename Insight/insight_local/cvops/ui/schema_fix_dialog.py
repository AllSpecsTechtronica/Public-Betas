from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SchemaFixDialog(QDialog):
    """Dialog to resolve CSV schema issues (label_col + feature_cols selection)."""

    def __init__(
        self,
        *,
        scenario: str,
        dataset_csv: str,
        attempted_label_col: str,
        columns: list[str],
        suggested_label_cols: list[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Fix CSV schema")
        self.setModal(True)
        self.resize(720, 520)

        self._scenario = str(scenario or "")
        self._dataset_csv = str(dataset_csv or "")
        self._attempted_label_col = str(attempted_label_col or "")
        self._columns = [str(c) for c in (columns or []) if str(c)]
        self._suggested = [str(c) for c in (suggested_label_cols or []) if str(c)]

        self._action = "cancel"  # cancel | reveal | apply | apply_rerun
        self._chosen_label_col = ""
        self._chosen_feature_cols: list[str] = []

        self._build_ui()

    def action(self) -> str:
        return self._action

    def chosen_label_col(self) -> str:
        return self._chosen_label_col

    def chosen_feature_cols(self) -> list[str]:
        return list(self._chosen_feature_cols)

    # ---- UI ----

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        info = QLabel(
            "Select the label (target) column, and optionally choose which columns to use as features.\n"
            "Tip: leave features as Select All unless you want to exclude IDs/text columns."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        meta = QLabel(
            f"scenario: {self._scenario or '—'}\n"
            f"dataset_csv: {self._dataset_csv or '—'}\n"
            f"attempted label_col: {self._attempted_label_col or '—'}"
        )
        meta.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        meta.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.7);")
        root.addWidget(meta)

        # Label chooser
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label column:"))
        self._label_combo = QComboBox()
        self._label_combo.setMinimumWidth(280)
        for c in self._columns:
            self._label_combo.addItem(c, c)
        label_row.addWidget(self._label_combo, stretch=1)
        root.addLayout(label_row)

        default = ""
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

        # Feature chooser (multi-select with Select All)
        feat_title = QLabel("Feature columns (multi-select):")
        feat_title.setStyleSheet("font-weight: 600;")
        root.addWidget(feat_title)

        feat_row = QHBoxLayout()
        self._feat_list = QListWidget()
        self._feat_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._feat_list.setMinimumHeight(260)
        feat_row.addWidget(self._feat_list, stretch=1)

        btns = QVBoxLayout()
        self._select_all_btn = QPushButton("Select All")
        self._select_all_btn.clicked.connect(self._select_all_features)
        self._select_none_btn = QPushButton("Select None")
        self._select_none_btn.clicked.connect(self._select_none_features)
        btns.addWidget(self._select_all_btn)
        btns.addWidget(self._select_none_btn)
        btns.addStretch(1)
        feat_row.addLayout(btns)
        root.addLayout(feat_row)

        for c in self._columns:
            it = QListWidgetItem(c)
            it.setData(Qt.ItemDataRole.UserRole, c)
            self._feat_list.addItem(it)

        # Default: select all features (we'll exclude label on accept).
        self._select_all_features()

        bottom = QHBoxLayout()
        bottom.addStretch(1)
        self._btn_reveal = QPushButton("Reveal CSV")
        self._btn_reveal.clicked.connect(self._on_reveal)
        self._btn_apply = QPushButton("Apply Only")
        self._btn_apply.clicked.connect(self._on_apply)
        self._btn_apply_rerun = QPushButton("Apply & Re-run Training")
        self._btn_apply_rerun.setDefault(True)
        self._btn_apply_rerun.clicked.connect(self._on_apply_rerun)
        self._btn_cancel = QPushButton("Cancel")
        self._btn_cancel.clicked.connect(self.reject)
        bottom.addWidget(self._btn_reveal)
        bottom.addWidget(self._btn_apply)
        bottom.addWidget(self._btn_apply_rerun)
        bottom.addWidget(self._btn_cancel)
        root.addLayout(bottom)

    def _select_all_features(self) -> None:
        for i in range(self._feat_list.count()):
            it = self._feat_list.item(i)
            if it is not None:
                it.setSelected(True)

    def _select_none_features(self) -> None:
        self._feat_list.clearSelection()

    # ---- actions ----

    def _capture(self) -> None:
        label = str(self._label_combo.currentData() or self._label_combo.currentText() or "").strip()
        self._chosen_label_col = label
        selected = []
        for it in self._feat_list.selectedItems():
            selected.append(str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "").strip())
        # Remove label from features if selected.
        self._chosen_feature_cols = [c for c in selected if c and c != label]

    def _on_reveal(self) -> None:
        self._capture()
        self._action = "reveal"
        self.accept()

    def _on_apply(self) -> None:
        self._capture()
        self._action = "apply"
        self.accept()

    def _on_apply_rerun(self) -> None:
        self._capture()
        self._action = "apply_rerun"
        self.accept()

