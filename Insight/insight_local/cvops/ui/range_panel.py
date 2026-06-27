from __future__ import annotations

import json
import urllib.parse
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .dropdown_pane_stack import DropdownPaneStack
from .time_format import format_timestamp


HttpCall = Callable[..., Any]


def _fmt_ts(ts: Optional[float]) -> str:
    return format_timestamp(ts, empty="-")


def _short(s: str, n: int = 14) -> str:
    s = str(s or "")
    if len(s) <= n:
        return s
    return s[: n - 1] + "\u2026"


class _NewRangeDialog(QDialog):
    def __init__(self, *, sectors: list[dict[str, Any]], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Range")
        self.resize(440, 220)
        form = QFormLayout()

        self._name = QLineEdit()
        form.addRow("Name:", self._name)

        self._sector = QComboBox()
        for s in sectors:
            path = str(s.get("path") or s.get("sector_path") or "/")
            sid = str(s.get("sector_id") or "")
            if not sid:
                continue
            self._sector.addItem(f"{path}  [{sid}]", userData=(sid, path))
        if self._sector.count() == 0:
            self._sector.addItem("/  [root]", userData=("root", "/"))
        form.addRow("Sector:", self._sector)

        self._mode = QComboBox()
        self._mode.addItems(["single", "compare", "lineage_sweep"])
        form.addRow("Mode:", self._mode)

        self._desc = QLineEdit()
        self._desc.setPlaceholderText("Optional description")
        form.addRow("Description:", self._desc)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def payload(self) -> Optional[dict[str, Any]]:
        name = self._name.text().strip()
        if not name:
            return None
        sector_data = self._sector.currentData() or ("root", "/")
        return {
            "name": name,
            "sector_id": sector_data[0],
            "sector_path": sector_data[1],
            "mode": self._mode.currentText(),
            "description": self._desc.text().strip(),
        }


class _AttachSubjectDialog(QDialog):
    def __init__(self, *, snapshots: list[dict[str, Any]], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Attach Snapshot as Subject")
        self.resize(480, 160)
        form = QFormLayout()

        self._snapshot = QComboBox()
        for sn in snapshots:
            snid = str(sn.get("snapshot_id") or "")
            if not snid:
                continue
            label = f"{snid[:18]}  {sn.get('model_type','')}  ({_fmt_ts(sn.get('created_at'))})"
            self._snapshot.addItem(label, userData=snid)
        if self._snapshot.count() == 0:
            self._snapshot.addItem("(no snapshots registered yet)", userData="")
            self._snapshot.setEnabled(False)
        form.addRow("Snapshot:", self._snapshot)

        self._label = QLineEdit()
        self._label.setPlaceholderText("Display label (defaults to snapshot id)")
        form.addRow("Label:", self._label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def payload(self) -> Optional[dict[str, Any]]:
        snap_id = str(self._snapshot.currentData() or "").strip()
        if not snap_id:
            return None
        return {
            "snapshot_id": snap_id,
            "label": self._label.text().strip(),
        }


class _SealGoldenSetDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Seal Golden Set")
        self.resize(560, 420)
        form = QFormLayout()

        self._name = QLineEdit()
        form.addRow("Name:", self._name)

        self._storage_uri = QLineEdit()
        self._storage_uri.setPlaceholderText("Path or URI to the sealed dataset")
        form.addRow("Storage URI:", self._storage_uri)

        self._row_count = QLineEdit("0")
        form.addRow("Row count:", self._row_count)

        self._sha256 = QLineEdit()
        self._sha256.setPlaceholderText("SHA-256 of the sealed content")
        form.addRow("Content SHA-256:", self._sha256)

        self._split_spec = QPlainTextEdit()
        self._split_spec.setPlaceholderText('{"split":"holdout","strategy":"...","details":{...}}')
        self._split_spec.setMaximumHeight(100)
        form.addRow("Split spec (JSON):", self._split_spec)

        self._desc = QLineEdit()
        form.addRow("Description:", self._desc)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def payload(self) -> Optional[dict[str, Any]]:
        name = self._name.text().strip()
        uri = self._storage_uri.text().strip()
        sha = self._sha256.text().strip()
        if not name or not uri or not sha:
            return None
        try:
            row_count = int((self._row_count.text() or "0").strip() or "0")
        except ValueError:
            return {"__error__": "row_count must be an integer"}
        raw_spec = (self._split_spec.toPlainText() or "").strip()
        if raw_spec:
            try:
                split_spec = json.loads(raw_spec)
                if not isinstance(split_spec, dict):
                    return {"__error__": "split_spec must be a JSON object"}
            except Exception as exc:
                return {"__error__": f"split_spec JSON invalid: {exc}"}
        else:
            split_spec = {}
        return {
            "name": name,
            "storage_uri": uri,
            "row_count": row_count,
            "content_sha256": sha,
            "split_spec": split_spec,
            "description": self._desc.text().strip(),
        }


class _AddDriftDialog(QDialog):
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Drift Scenario")
        self.resize(460, 260)
        form = QFormLayout()

        self._name = QLineEdit()
        form.addRow("Name:", self._name)

        self._kind = QComboBox()
        self._kind.addItems(["label_noise", "covariate_shift", "corruption", "imbalance"])
        form.addRow("Kind:", self._kind)

        self._params = QPlainTextEdit()
        self._params.setPlaceholderText('{"p": 0.1, "sigma": 0.25}')
        self._params.setMaximumHeight(120)
        form.addRow("Params (JSON):", self._params)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def payload(self) -> Optional[dict[str, Any]]:
        name = self._name.text().strip()
        if not name:
            return None
        raw = (self._params.toPlainText() or "").strip()
        if raw:
            try:
                params = json.loads(raw)
                if not isinstance(params, dict):
                    return {"__error__": "params must be a JSON object"}
            except Exception as exc:
                return {"__error__": f"params JSON invalid: {exc}"}
        else:
            params = {}
        return {
            "name": name,
            "kind": self._kind.currentText(),
            "params": params,
        }


class _AddGateDialog(QDialog):
    def __init__(
        self,
        *,
        golden_sets: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Regression Gate")
        self.resize(500, 280)
        form = QFormLayout()

        self._metric = QLineEdit()
        self._metric.setPlaceholderText("e.g. mAP, accuracy, f1")
        form.addRow("Metric:", self._metric)

        self._threshold_type = QComboBox()
        self._threshold_type.addItems(["absolute", "delta_from_baseline", "delta_from_prev"])
        form.addRow("Threshold type:", self._threshold_type)

        self._threshold_value = QLineEdit("0.0")
        form.addRow("Threshold value:", self._threshold_value)

        self._golden = QComboBox()
        self._golden.addItem("(any)", userData="")
        for g in golden_sets:
            self._golden.addItem(
                f"{g.get('name','?')}  [{_short(str(g.get('golden_id','')), 12)}]",
                userData=str(g.get("golden_id") or ""),
            )
        form.addRow("Golden set:", self._golden)

        self._baseline = QComboBox()
        self._baseline.addItem("(none)", userData="")
        for sn in snapshots:
            snid = str(sn.get("snapshot_id") or "")
            if not snid:
                continue
            self._baseline.addItem(f"{snid[:18]} {sn.get('model_type','')}", userData=snid)
        form.addRow("Baseline snapshot:", self._baseline)

        self._action = QComboBox()
        self._action.addItems(["warn", "block"])
        form.addRow("Action:", self._action)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def payload(self) -> Optional[dict[str, Any]]:
        metric = self._metric.text().strip()
        if not metric:
            return None
        try:
            value = float(self._threshold_value.text().strip() or "0")
        except ValueError:
            return {"__error__": "threshold_value must be a number"}
        return {
            "metric": metric,
            "threshold_type": self._threshold_type.currentText(),
            "threshold_value": value,
            "golden_id": str(self._golden.currentData() or "") or None,
            "baseline_snapshot_id": str(self._baseline.currentData() or "") or None,
            "action": self._action.currentText(),
        }


class _RecordEvalDialog(QDialog):
    def __init__(
        self,
        *,
        subjects: list[dict[str, Any]],
        golden_sets: list[dict[str, Any]],
        drifts: list[dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Record Evaluation")
        self.resize(560, 380)
        form = QFormLayout()

        self._subject = QComboBox()
        for s in subjects:
            snid = str(s.get("snapshot_id") or "")
            if not snid:
                continue
            label = f"{s.get('label') or snid[:18]}  [{_short(snid, 14)}]"
            self._subject.addItem(label, userData=snid)
        if self._subject.count() == 0:
            self._subject.addItem("(no subjects attached)", userData="")
            self._subject.setEnabled(False)
        form.addRow("Subject:", self._subject)

        self._golden = QComboBox()
        for g in golden_sets:
            gid = str(g.get("golden_id") or "")
            if not gid:
                continue
            self._golden.addItem(f"{g.get('name','?')}  [{_short(gid, 12)}]", userData=gid)
        if self._golden.count() == 0:
            self._golden.addItem("(no golden sets)", userData="")
            self._golden.setEnabled(False)
        form.addRow("Golden set:", self._golden)

        self._drift = QComboBox()
        self._drift.addItem("(none)", userData="")
        for d in drifts:
            did = str(d.get("drift_id") or "")
            if not did:
                continue
            self._drift.addItem(f"{d.get('name','?')} / {d.get('kind','?')}", userData=did)
        form.addRow("Drift scenario:", self._drift)

        self._metrics = QPlainTextEdit()
        self._metrics.setPlaceholderText('{"mAP": 0.712, "accuracy": 0.89}')
        self._metrics.setMaximumHeight(120)
        form.addRow("Metrics (JSON):", self._metrics)

        self._predictions_uri = QLineEdit()
        self._predictions_uri.setPlaceholderText("Optional: path to predictions artifact")
        form.addRow("Predictions URI:", self._predictions_uri)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def payload(self) -> Optional[dict[str, Any]]:
        sid = str(self._subject.currentData() or "").strip()
        gid = str(self._golden.currentData() or "").strip()
        if not sid or not gid:
            return None
        raw = (self._metrics.toPlainText() or "").strip()
        if not raw:
            return {"__error__": "metrics JSON is required"}
        try:
            metrics = json.loads(raw)
            if not isinstance(metrics, dict):
                return {"__error__": "metrics must be a JSON object"}
        except Exception as exc:
            return {"__error__": f"metrics JSON invalid: {exc}"}
        drift_id = str(self._drift.currentData() or "").strip() or None
        return {
            "snapshot_id": sid,
            "golden_id": gid,
            "metrics": metrics,
            "drift_id": drift_id,
            "predictions_uri": self._predictions_uri.text().strip(),
        }


class TestRangePanel(QWidget):
    """Range catalog panel.

    Range list on the left, tabbed detail on the right (Subjects, Golden Sets,
    Drifts, Evaluations, Gates).
    """

    errorRaised = pyqtSignal(str)

    def __init__(
        self,
        *,
        http_get: HttpCall,
        http_post: HttpCall,
        http_delete: HttpCall,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._http_get = http_get
        self._http_post = http_post
        self._http_delete = http_delete
        self._selected_id: Optional[str] = None
        self._current: Optional[dict[str, Any]] = None
        self._current_evals: list[dict[str, Any]] = []
        self._sectors_cache: list[dict[str, Any]] = []
        self._datasets_cache: list[dict[str, Any]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # -- Header toolbar --
        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("Range Catalog")
        f = QFont()
        f.setBold(True)
        title.setFont(f)
        header.addWidget(title)
        header.addStretch(1)

        self._sector_filter = QComboBox()
        self._sector_filter.addItem("All sectors", userData="")
        self._sector_filter.currentIndexChanged.connect(lambda _i: self._reload_ranges())
        header.addWidget(QLabel("Sector:"))
        header.addWidget(self._sector_filter)

        new_btn = QPushButton("[NEW RANGE]")
        new_btn.clicked.connect(self._on_new_range)
        header.addWidget(new_btn)

        refresh_btn = QPushButton("[REFRESH]")
        refresh_btn.clicked.connect(self._reload_all)
        header.addWidget(refresh_btn)
        root.addLayout(header)

        # -- Splitter --
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(True)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        self._list.setMinimumWidth(260)
        splitter.addWidget(self._list)

        detail = QWidget()
        dl = QVBoxLayout(detail)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(6)

        self._detail_header = QLabel("Select a range.")
        self._detail_header.setWordWrap(True)
        self._detail_header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        dl.addWidget(self._detail_header)

        top_actions = QHBoxLayout()
        top_actions.setSpacing(6)
        self._delete_range_btn = QPushButton("[DELETE RANGE]")
        self._delete_range_btn.clicked.connect(self._on_delete_range)
        self._delete_range_btn.setEnabled(False)
        top_actions.addStretch(1)
        top_actions.addWidget(self._delete_range_btn)
        dl.addLayout(top_actions)

        self._tabs = DropdownPaneStack()
        self._tabs.addTab(self._build_subjects_tab(), "Subjects")
        self._tabs.addTab(self._build_golden_tab(), "Golden Sets")
        self._tabs.addTab(self._build_drift_tab(), "Drifts")
        self._tabs.addTab(self._build_eval_tab(), "Evaluations")
        self._tabs.addTab(self._build_gates_tab(), "Gates")
        dl.addWidget(self._tabs, stretch=1)

        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 700])
        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------

    def _build_subjects_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(4, 4, 4, 4)
        l.setSpacing(4)

        bar = QHBoxLayout()
        self._attach_btn = QPushButton("[ATTACH SNAPSHOT]")
        self._attach_btn.clicked.connect(self._on_attach_subject)
        self._attach_btn.setEnabled(False)
        bar.addWidget(self._attach_btn)
        self._detach_btn = QPushButton("[DETACH]")
        self._detach_btn.clicked.connect(self._on_detach_subject)
        self._detach_btn.setEnabled(False)
        bar.addWidget(self._detach_btn)
        bar.addStretch(1)
        l.addLayout(bar)

        self._subjects_table = QTableWidget(0, 3)
        self._subjects_table.setHorizontalHeaderLabels(["Snapshot", "Label", "Added"])
        self._subjects_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._subjects_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._subjects_table.setAlternatingRowColors(True)
        self._subjects_table.verticalHeader().setVisible(False)
        self._subjects_table.horizontalHeader().setStretchLastSection(True)
        l.addWidget(self._subjects_table, stretch=1)
        return w

    def _build_golden_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(4, 4, 4, 4)
        l.setSpacing(4)

        bar = QHBoxLayout()
        self._seal_btn = QPushButton("[SEAL GOLDEN SET]")
        self._seal_btn.clicked.connect(self._on_seal_golden)
        self._seal_btn.setEnabled(False)
        bar.addWidget(self._seal_btn)
        self._seal_dataset_btn = QPushButton("[SEAL SELECTED DATASET]")
        self._seal_dataset_btn.clicked.connect(self._on_seal_selected_dataset)
        self._seal_dataset_btn.setEnabled(False)
        bar.addWidget(self._seal_dataset_btn)
        self._delete_golden_btn = QPushButton("[DELETE]")
        self._delete_golden_btn.clicked.connect(self._on_delete_golden)
        self._delete_golden_btn.setEnabled(False)
        bar.addWidget(self._delete_golden_btn)
        dataset_refresh = QPushButton("[REFRESH DATASETS]")
        dataset_refresh.clicked.connect(self._reload_datasets)
        bar.addWidget(dataset_refresh)
        bar.addStretch(1)
        l.addLayout(bar)

        self._dataset_status = QLabel("No datasets loaded.")
        self._dataset_status.setProperty("muted", True)
        l.addWidget(self._dataset_status)

        self._dataset_table = QTableWidget(0, 5)
        self._dataset_table.setHorizontalHeaderLabels(["Dataset", "Type", "Items", "Storage", "SHA-256"])
        self._dataset_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._dataset_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._dataset_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._dataset_table.setAlternatingRowColors(True)
        self._dataset_table.verticalHeader().setVisible(False)
        dh = self._dataset_table.horizontalHeader()
        dh.setStretchLastSection(True)
        dh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        dh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        dh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        l.addWidget(self._dataset_table, stretch=1)

        self._golden_table = QTableWidget(0, 5)
        self._golden_table.setHorizontalHeaderLabels([
            "ID", "Name", "Rows", "SHA-256", "Sealed",
        ])
        self._golden_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._golden_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._golden_table.setAlternatingRowColors(True)
        self._golden_table.verticalHeader().setVisible(False)
        self._golden_table.horizontalHeader().setStretchLastSection(True)
        l.addWidget(self._golden_table, stretch=1)
        return w

    def _build_drift_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(4, 4, 4, 4)
        l.setSpacing(4)

        bar = QHBoxLayout()
        self._add_drift_btn = QPushButton("[ADD DRIFT]")
        self._add_drift_btn.clicked.connect(self._on_add_drift)
        self._add_drift_btn.setEnabled(False)
        bar.addWidget(self._add_drift_btn)
        self._delete_drift_btn = QPushButton("[DELETE]")
        self._delete_drift_btn.clicked.connect(self._on_delete_drift)
        self._delete_drift_btn.setEnabled(False)
        bar.addWidget(self._delete_drift_btn)
        bar.addStretch(1)
        l.addLayout(bar)

        self._drift_table = QTableWidget(0, 4)
        self._drift_table.setHorizontalHeaderLabels(["ID", "Name", "Kind", "Params"])
        self._drift_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._drift_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._drift_table.setAlternatingRowColors(True)
        self._drift_table.verticalHeader().setVisible(False)
        self._drift_table.horizontalHeader().setStretchLastSection(True)
        l.addWidget(self._drift_table, stretch=1)
        return w

    def _build_eval_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(4, 4, 4, 4)
        l.setSpacing(4)

        bar = QHBoxLayout()
        self._record_eval_btn = QPushButton("[RECORD EVAL]")
        self._record_eval_btn.clicked.connect(self._on_record_eval)
        self._record_eval_btn.setEnabled(False)
        bar.addWidget(self._record_eval_btn)
        self._delete_eval_btn = QPushButton("[DELETE]")
        self._delete_eval_btn.clicked.connect(self._on_delete_eval)
        self._delete_eval_btn.setEnabled(False)
        bar.addWidget(self._delete_eval_btn)
        bar.addStretch(1)
        l.addLayout(bar)

        self._eval_table = QTableWidget(0, 6)
        self._eval_table.setHorizontalHeaderLabels([
            "Eval ID", "Snapshot", "Golden", "Drift", "Metrics", "Ran",
        ])
        self._eval_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._eval_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._eval_table.setAlternatingRowColors(True)
        self._eval_table.verticalHeader().setVisible(False)
        hh = self._eval_table.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        l.addWidget(self._eval_table, stretch=1)
        return w

    def _build_gates_tab(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(4, 4, 4, 4)
        l.setSpacing(4)

        bar = QHBoxLayout()
        self._add_gate_btn = QPushButton("[ADD GATE]")
        self._add_gate_btn.clicked.connect(self._on_add_gate)
        self._add_gate_btn.setEnabled(False)
        bar.addWidget(self._add_gate_btn)
        self._delete_gate_btn = QPushButton("[DELETE]")
        self._delete_gate_btn.clicked.connect(self._on_delete_gate)
        self._delete_gate_btn.setEnabled(False)
        bar.addWidget(self._delete_gate_btn)
        bar.addStretch(1)
        l.addLayout(bar)

        self._gates_table = QTableWidget(0, 6)
        self._gates_table.setHorizontalHeaderLabels([
            "ID", "Metric", "Type", "Value", "Baseline", "Action",
        ])
        self._gates_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._gates_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._gates_table.setAlternatingRowColors(True)
        self._gates_table.verticalHeader().setVisible(False)
        self._gates_table.horizontalHeader().setStretchLastSection(True)
        l.addWidget(self._gates_table, stretch=1)
        return w

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def reload(self) -> None:
        self._reload_all()

    def select_range(self, range_id: str) -> None:
        """Public hook used by the Ecosystem quick-nav router."""
        rid = str(range_id or "").strip()
        if not rid:
            return
        for i in range(self._list.count()):
            it = self._list.item(i)
            if str(it.data(Qt.ItemDataRole.UserRole) or "") == rid:
                self._list.setCurrentRow(i)
                return
        try:
            self._reload_all()
        except Exception:
            return
        for i in range(self._list.count()):
            it = self._list.item(i)
            if str(it.data(Qt.ItemDataRole.UserRole) or "") == rid:
                self._list.setCurrentRow(i)
                return

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _reload_all(self) -> None:
        self._reload_sectors()
        self._reload_datasets()
        self._reload_ranges()

    def _reload_datasets(self) -> None:
        try:
            resp = self._http_get("/database")
        except Exception as exc:
            self.errorRaised.emit(f"datasets: {exc}")
            return
        if not isinstance(resp, dict):
            return
        names = [str(n) for n in (resp.get("datasets") or []) if str(n).strip()]
        out: list[dict[str, Any]] = []
        for name in names:
            try:
                detail = self._http_get(f"/database/{urllib.parse.quote(name, safe='')}")
            except Exception:
                detail = {"slug": name, "format": "", "category": "", "count": 0}
            if isinstance(detail, dict):
                detail.setdefault("name", name)
                detail.setdefault("slug", name)
                out.append(detail)
        for entry in resp.get("tabular_datasets") or []:
            if not isinstance(entry, dict):
                continue
            path = str(entry.get("path") or "")
            name = str(entry.get("filename") or entry.get("name") or path)
            if not name:
                continue
            out.append({
                "name": name,
                "slug": name,
                "path": path,
                "format": entry.get("format") or "csv",
                "category": entry.get("category") or "tabular",
                "count": 1,
                "content_sha256": self._metadata_sha(entry),
            })
        self._datasets_cache = out
        self._render_datasets()

    @staticmethod
    def _metadata_sha(payload: dict[str, Any]) -> str:
        import hashlib
        raw = json.dumps(payload or {}, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _render_datasets(self) -> None:
        self._dataset_table.setRowCount(0)
        for ds in self._datasets_cache:
            r = self._dataset_table.rowCount()
            self._dataset_table.insertRow(r)
            values = [
                str(ds.get("name") or ds.get("slug") or ""),
                str(ds.get("format") or ds.get("category") or ""),
                str(ds.get("count") or 0),
                str(ds.get("path") or ds.get("slug") or ""),
                _short(str(ds.get("content_sha256") or ""), 14),
            ]
            for c, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(str(value))
                self._dataset_table.setItem(r, c, item)
        self._dataset_status.setText(f"{len(self._datasets_cache)} dataset{'s' if len(self._datasets_cache) != 1 else ''} available for golden sets")

    def _selected_dataset(self) -> Optional[dict[str, Any]]:
        rows = self._dataset_table.selectionModel().selectedRows()
        if not rows:
            return None
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._datasets_cache):
            return None
        return self._datasets_cache[idx]

    def _reload_sectors(self) -> None:
        current = self._sector_filter.currentData() or ""
        try:
            resp = self._http_get("/sectors")
        except Exception as exc:
            self.errorRaised.emit(f"sectors: {exc}")
            return
        sectors = []
        if isinstance(resp, dict):
            sectors = resp.get("items") or resp.get("sectors") or []
        elif isinstance(resp, list):
            sectors = resp
        self._sectors_cache = list(sectors) if isinstance(sectors, list) else []

        self._sector_filter.blockSignals(True)
        self._sector_filter.clear()
        self._sector_filter.addItem("All sectors", userData="")
        for s in self._sectors_cache:
            path = str(s.get("path") or s.get("sector_path") or "/")
            self._sector_filter.addItem(path, userData=path)
        idx = self._sector_filter.findData(current)
        if idx >= 0:
            self._sector_filter.setCurrentIndex(idx)
        self._sector_filter.blockSignals(False)

    def _reload_ranges(self) -> None:
        sp = self._sector_filter.currentData() or ""
        qs = f"?sector_path={sp}" if sp else ""
        try:
            resp = self._http_get(f"/ranges{qs}")
        except Exception as exc:
            self.errorRaised.emit(f"ranges: {exc}")
            return
        items = []
        if isinstance(resp, dict):
            items = resp.get("items") or []
        self._list.clear()
        for r in items:
            rid = str(r.get("range_id") or "")
            name = str(r.get("name") or "(unnamed)")
            mode = str(r.get("mode") or "single")
            sector_path = str(r.get("sector_path") or "/")
            last = _fmt_ts(r.get("last_run_at"))
            label = f"{name}\n  {sector_path}  \u00B7  {mode}  \u00B7  last-run={last}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, rid)
            self._list.addItem(item)
        if self._selected_id:
            for i in range(self._list.count()):
                it = self._list.item(i)
                if it.data(Qt.ItemDataRole.UserRole) == self._selected_id:
                    self._list.setCurrentRow(i)
                    break

    def _on_selection_changed(self) -> None:
        items = self._list.selectedItems()
        if not items:
            self._clear_detail()
            return
        rid = items[0].data(Qt.ItemDataRole.UserRole)
        if not rid:
            self._clear_detail()
            return
        self._selected_id = str(rid)
        try:
            resp = self._http_get(f"/ranges/{rid}")
        except Exception as exc:
            self.errorRaised.emit(f"range detail: {exc}")
            return
        if not isinstance(resp, dict):
            self._clear_detail()
            return
        self._current = resp
        self._render_detail(resp)
        self._reload_evals()

    def _reload_evals(self) -> None:
        if not self._selected_id:
            self._current_evals = []
            self._eval_table.setRowCount(0)
            return
        try:
            resp = self._http_get(f"/ranges/{self._selected_id}/evaluations")
        except Exception as exc:
            self.errorRaised.emit(f"evaluations: {exc}")
            return
        items = []
        if isinstance(resp, dict):
            items = resp.get("items") or []
        self._current_evals = list(items)
        self._render_evals()

    def _clear_detail(self) -> None:
        self._selected_id = None
        self._current = None
        self._current_evals = []
        self._detail_header.setText("Select a range.")
        for tbl in (
            self._subjects_table, self._golden_table,
            self._drift_table, self._eval_table, self._gates_table,
        ):
            tbl.setRowCount(0)
        for b in (
            self._delete_range_btn, self._attach_btn, self._detach_btn,
            self._seal_btn, self._delete_golden_btn,
            self._add_drift_btn, self._delete_drift_btn,
            self._record_eval_btn, self._delete_eval_btn,
            self._add_gate_btn, self._delete_gate_btn,
        ):
            b.setEnabled(False)
        self._seal_dataset_btn.setEnabled(False)

    def _render_detail(self, detail: dict[str, Any]) -> None:
        rng = detail.get("range") or {}
        subjects = detail.get("subjects") or []
        goldens = detail.get("golden_sets") or []
        drifts = detail.get("drifts") or []
        gates = detail.get("gates") or []

        parts = [
            f"<b>{rng.get('name','(unnamed)')}</b>",
            f"sector: {rng.get('sector_path','/')}",
            f"mode: {rng.get('mode','single')}",
            f"subjects: {len(subjects)}",
            f"goldens: {len(goldens)}",
            f"drifts: {len(drifts)}",
            f"gates: {len(gates)}",
            f"last-run: {_fmt_ts(rng.get('last_run_at'))}",
        ]
        desc = str(rng.get("description") or "").strip()
        if desc:
            parts.append(f"<i>{desc}</i>")
        self._detail_header.setText("  \u00B7  ".join(parts))

        # Subjects
        self._subjects_table.setRowCount(0)
        for s in subjects:
            r = self._subjects_table.rowCount()
            self._subjects_table.insertRow(r)
            for c, v in enumerate([
                _short(str(s.get("snapshot_id", "")), 22),
                str(s.get("label") or ""),
                _fmt_ts(s.get("added_at")),
            ]):
                self._subjects_table.setItem(r, c, QTableWidgetItem(v))

        # Golden sets
        self._golden_table.setRowCount(0)
        for g in goldens:
            r = self._golden_table.rowCount()
            self._golden_table.insertRow(r)
            for c, v in enumerate([
                _short(str(g.get("golden_id", "")), 18),
                str(g.get("name") or ""),
                str(g.get("row_count", 0)),
                _short(str(g.get("content_sha256", "")), 12),
                _fmt_ts(g.get("sealed_at")),
            ]):
                self._golden_table.setItem(r, c, QTableWidgetItem(v))

        # Drifts
        self._drift_table.setRowCount(0)
        for d in drifts:
            r = self._drift_table.rowCount()
            self._drift_table.insertRow(r)
            params = d.get("params") or {}
            params_s = json.dumps(params, separators=(",", ":")) if params else ""
            for c, v in enumerate([
                _short(str(d.get("drift_id", "")), 18),
                str(d.get("name") or ""),
                str(d.get("kind") or ""),
                params_s,
            ]):
                self._drift_table.setItem(r, c, QTableWidgetItem(v))

        # Gates
        self._gates_table.setRowCount(0)
        for g in gates:
            r = self._gates_table.rowCount()
            self._gates_table.insertRow(r)
            for c, v in enumerate([
                _short(str(g.get("gate_id", "")), 18),
                str(g.get("metric") or ""),
                str(g.get("threshold_type") or ""),
                f"{g.get('threshold_value', 0.0)}",
                _short(str(g.get("baseline_snapshot_id") or ""), 14),
                str(g.get("action") or ""),
            ]):
                self._gates_table.setItem(r, c, QTableWidgetItem(v))

        self._delete_range_btn.setEnabled(True)
        self._attach_btn.setEnabled(True)
        self._detach_btn.setEnabled(True)
        self._seal_btn.setEnabled(True)
        self._seal_dataset_btn.setEnabled(True)
        self._delete_golden_btn.setEnabled(True)
        self._add_drift_btn.setEnabled(True)
        self._delete_drift_btn.setEnabled(True)
        self._record_eval_btn.setEnabled(bool(subjects and goldens))
        self._delete_eval_btn.setEnabled(True)
        self._add_gate_btn.setEnabled(True)
        self._delete_gate_btn.setEnabled(True)

    def _render_evals(self) -> None:
        self._eval_table.setRowCount(0)
        for e in self._current_evals:
            r = self._eval_table.rowCount()
            self._eval_table.insertRow(r)
            metrics = e.get("metrics") or {}
            metrics_s = ", ".join(f"{k}={v}" for k, v in metrics.items())
            for c, v in enumerate([
                _short(str(e.get("eval_id", "")), 18),
                _short(str(e.get("snapshot_id", "")), 18),
                _short(str(e.get("golden_id", "")), 14),
                _short(str(e.get("drift_id") or ""), 12),
                metrics_s,
                _fmt_ts(e.get("ran_at")),
            ]):
                self._eval_table.setItem(r, c, QTableWidgetItem(v))

    # ------------------------------------------------------------------
    # Actions - range lifecycle
    # ------------------------------------------------------------------

    def _on_new_range(self) -> None:
        dlg = _NewRangeDialog(sectors=self._sectors_cache, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            QMessageBox.warning(self, "New Range", "Name is required.")
            return
        try:
            self._http_post("/ranges", payload)
        except Exception as exc:
            self.errorRaised.emit(f"create range: {exc}")
            return
        self._reload_ranges()

    def _on_delete_range(self) -> None:
        if not self._selected_id:
            return
        confirm = QMessageBox.question(
            self, "Delete Range",
            "Delete this range and all its subjects / golden sets / evaluations?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._http_delete(f"/ranges/{self._selected_id}")
        except Exception as exc:
            self.errorRaised.emit(f"delete range: {exc}")
            return
        self._selected_id = None
        self._clear_detail()
        self._reload_ranges()

    # ------------------------------------------------------------------
    # Actions - subjects
    # ------------------------------------------------------------------

    def _fetch_snapshots(self) -> list[dict[str, Any]]:
        try:
            resp = self._http_get("/snapshots?limit=500")
        except Exception as exc:
            self.errorRaised.emit(f"snapshots: {exc}")
            return []
        if isinstance(resp, dict):
            return list(resp.get("items") or [])
        return []

    def _current_subjects(self) -> list[dict[str, Any]]:
        if not self._current:
            return []
        return list(self._current.get("subjects") or [])

    def _current_goldens(self) -> list[dict[str, Any]]:
        if not self._current:
            return []
        return list(self._current.get("golden_sets") or [])

    def _current_drifts(self) -> list[dict[str, Any]]:
        if not self._current:
            return []
        return list(self._current.get("drifts") or [])

    def _on_attach_subject(self) -> None:
        if not self._selected_id:
            return
        dlg = _AttachSubjectDialog(snapshots=self._fetch_snapshots(), parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            QMessageBox.warning(self, "Attach Subject", "Snapshot is required.")
            return
        try:
            self._http_post(f"/ranges/{self._selected_id}/subjects", payload)
        except Exception as exc:
            self.errorRaised.emit(f"attach subject: {exc}")
            return
        self._on_selection_changed()

    def _on_detach_subject(self) -> None:
        if not self._selected_id:
            return
        rows = self._subjects_table.selectionModel().selectedRows()
        if not rows:
            return
        subjects = self._current_subjects()
        row = rows[0].row()
        if row >= len(subjects):
            return
        snap = subjects[row]
        snap_id = str(snap.get("snapshot_id") or "")
        if not snap_id:
            return
        try:
            self._http_delete(f"/ranges/{self._selected_id}/subjects/{snap_id}")
        except Exception as exc:
            self.errorRaised.emit(f"detach subject: {exc}")
            return
        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Actions - golden sets
    # ------------------------------------------------------------------

    def _on_seal_golden(self) -> None:
        if not self._selected_id:
            return
        dlg = _SealGoldenSetDialog(parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            QMessageBox.warning(self, "Seal Golden Set", "Name, URI and SHA-256 are required.")
            return
        if "__error__" in payload:
            QMessageBox.warning(self, "Seal Golden Set", payload["__error__"])
            return
        try:
            self._http_post(f"/ranges/{self._selected_id}/golden_sets", payload)
        except Exception as exc:
            self.errorRaised.emit(f"seal golden set: {exc}")
            return
        self._on_selection_changed()

    def _on_seal_selected_dataset(self) -> None:
        if not self._selected_id:
            return
        ds = self._selected_dataset()
        if not ds:
            QMessageBox.warning(self, "Seal Dataset", "Select a dataset from the Dataset Library table first.")
            return
        name = str(ds.get("name") or ds.get("slug") or "dataset").strip()
        storage_uri = str(ds.get("path") or ds.get("slug") or name).strip()
        sha = str(ds.get("content_sha256") or "").strip()
        if len(sha) != 64:
            sha = self._metadata_sha(ds)
        try:
            row_count = int(ds.get("count") or 0)
        except Exception:
            row_count = 0
        split_spec = {
            "source": "cvops_dataset_library",
            "dataset": name,
            "format": str(ds.get("format") or ""),
            "category": str(ds.get("category") or ""),
            "split_counts": ds.get("split_counts") if isinstance(ds.get("split_counts"), dict) else {},
        }
        payload = {
            "name": name,
            "storage_uri": storage_uri,
            "row_count": row_count,
            "content_sha256": sha,
            "split_spec": split_spec,
            "description": f"Golden set sealed from CV Ops dataset library: {name}",
        }
        try:
            self._http_post(f"/ranges/{self._selected_id}/golden_sets", payload)
        except Exception as exc:
            self.errorRaised.emit(f"seal dataset: {exc}")
            return
        self._on_selection_changed()

    def _on_delete_golden(self) -> None:
        if not self._selected_id:
            return
        rows = self._golden_table.selectionModel().selectedRows()
        if not rows:
            return
        goldens = self._current_goldens()
        row = rows[0].row()
        if row >= len(goldens):
            return
        gid = str(goldens[row].get("golden_id") or "")
        if not gid:
            return
        try:
            self._http_delete(f"/ranges/{self._selected_id}/golden_sets/{gid}")
        except Exception as exc:
            self.errorRaised.emit(f"delete golden: {exc}")
            return
        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Actions - drifts
    # ------------------------------------------------------------------

    def _on_add_drift(self) -> None:
        if not self._selected_id:
            return
        dlg = _AddDriftDialog(parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            QMessageBox.warning(self, "Add Drift", "Name is required.")
            return
        if "__error__" in payload:
            QMessageBox.warning(self, "Add Drift", payload["__error__"])
            return
        try:
            self._http_post(f"/ranges/{self._selected_id}/drifts", payload)
        except Exception as exc:
            self.errorRaised.emit(f"add drift: {exc}")
            return
        self._on_selection_changed()

    def _on_delete_drift(self) -> None:
        if not self._selected_id:
            return
        rows = self._drift_table.selectionModel().selectedRows()
        if not rows:
            return
        drifts = self._current_drifts()
        row = rows[0].row()
        if row >= len(drifts):
            return
        did = str(drifts[row].get("drift_id") or "")
        if not did:
            return
        try:
            self._http_delete(f"/ranges/{self._selected_id}/drifts/{did}")
        except Exception as exc:
            self.errorRaised.emit(f"delete drift: {exc}")
            return
        self._on_selection_changed()

    # ------------------------------------------------------------------
    # Actions - evaluations
    # ------------------------------------------------------------------

    def _on_record_eval(self) -> None:
        if not self._selected_id:
            return
        dlg = _RecordEvalDialog(
            subjects=self._current_subjects(),
            golden_sets=self._current_goldens(),
            drifts=self._current_drifts(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            QMessageBox.warning(self, "Record Eval", "Subject, golden set and metrics are required.")
            return
        if "__error__" in payload:
            QMessageBox.warning(self, "Record Eval", payload["__error__"])
            return
        try:
            self._http_post(f"/ranges/{self._selected_id}/evaluations", payload)
        except Exception as exc:
            self.errorRaised.emit(f"record eval: {exc}")
            return
        self._reload_evals()

    def _on_delete_eval(self) -> None:
        if not self._selected_id:
            return
        rows = self._eval_table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        if row >= len(self._current_evals):
            return
        eid = str(self._current_evals[row].get("eval_id") or "")
        if not eid:
            return
        try:
            self._http_delete(f"/ranges/{self._selected_id}/evaluations/{eid}")
        except Exception as exc:
            self.errorRaised.emit(f"delete eval: {exc}")
            return
        self._reload_evals()

    # ------------------------------------------------------------------
    # Actions - gates
    # ------------------------------------------------------------------

    def _on_add_gate(self) -> None:
        if not self._selected_id:
            return
        dlg = _AddGateDialog(
            golden_sets=self._current_goldens(),
            snapshots=self._fetch_snapshots(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            QMessageBox.warning(self, "Add Gate", "Metric is required.")
            return
        if "__error__" in payload:
            QMessageBox.warning(self, "Add Gate", payload["__error__"])
            return
        try:
            self._http_post(f"/ranges/{self._selected_id}/gates", payload)
        except Exception as exc:
            self.errorRaised.emit(f"add gate: {exc}")
            return
        self._on_selection_changed()

    def _on_delete_gate(self) -> None:
        if not self._selected_id:
            return
        rows = self._gates_table.selectionModel().selectedRows()
        if not rows:
            return
        current_gates = list((self._current or {}).get("gates") or [])
        row = rows[0].row()
        if row >= len(current_gates):
            return
        gid = str(current_gates[row].get("gate_id") or "")
        if not gid:
            return
        try:
            self._http_delete(f"/ranges/{self._selected_id}/gates/{gid}")
        except Exception as exc:
            self.errorRaised.emit(f"delete gate: {exc}")
            return
        self._on_selection_changed()
