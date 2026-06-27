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
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .selectable_panel import SelectablePanel
from .time_format import format_timestamp


HttpCall = Callable[..., Any]

_REGISTRY_LINEAGE_PREFIX = "registry:"


def _is_registry_lineage(lineage_id: Any) -> bool:
    return str(lineage_id or "").startswith(_REGISTRY_LINEAGE_PREFIX)


def _fmt_ts(ts: Optional[float]) -> str:
    return format_timestamp(ts, empty="-")


def _short(s: str, n: int = 14) -> str:
    s = str(s or "")
    if len(s) <= n:
        return s
    return s[: n - 1] + "\u2026"


class _NewLineageDialog(QDialog):
    """Inline dialog for creating a new lineage."""

    def __init__(
        self,
        *,
        sectors: list[dict[str, Any]],
        snapshots: list[dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Lineage")
        self.resize(500, 260)

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
        form.addRow("Base snapshot:", self._snapshot)

        self._strategy = QComboBox()
        self._strategy.addItems(["head_only", "lora", "replay_mixed", "full"])
        form.addRow("Update strategy:", self._strategy)

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
        snap_id = str(self._snapshot.currentData() or "").strip()
        if not snap_id:
            return None
        return {
            "name": name,
            "sector_id": sector_data[0],
            "sector_path": sector_data[1],
            "base_snapshot_id": snap_id,
            "update_strategy": self._strategy.currentText(),
            "description": self._desc.text().strip(),
        }


class _AddDropDialog(QDialog):
    """Power-user dialog to manually append a drop by JSON payload."""

    def __init__(
        self,
        *,
        snapshots: list[dict[str, Any]],
        dataset: Optional[dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Drop")
        self.resize(600, 420)

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
        form.addRow("New snapshot:", self._snapshot)

        self._sample_count = QLineEdit("0")
        form.addRow("Sample count:", self._sample_count)

        self._notes = QLineEdit()
        form.addRow("Notes:", self._notes)

        self._source = QPlainTextEdit()
        self._source.setPlaceholderText('{"kind": "user_drop", "dataset": "..."}')
        self._source.setMaximumHeight(90)
        if dataset:
            self._source.setPlainText(json.dumps({
                "kind": "dataset",
                "dataset": dataset.get("name") or dataset.get("slug") or dataset.get("path") or "",
                "format": dataset.get("format") or "",
                "category": dataset.get("category") or "",
                "storage_uri": dataset.get("path") or dataset.get("storage_uri") or "",
            }, indent=2, ensure_ascii=True))
        form.addRow("Source JSON:", self._source)

        self._training_delta = QPlainTextEdit()
        self._training_delta.setPlaceholderText('{"epochs": 1, "lr": 1e-4}')
        self._training_delta.setMaximumHeight(90)
        form.addRow("Training delta JSON:", self._training_delta)

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

        def _json_or_empty(raw: str, default: Any) -> Any:
            raw = (raw or "").strip()
            if not raw:
                return default
            try:
                return json.loads(raw)
            except Exception:
                return None

        source = _json_or_empty(self._source.toPlainText(), {})
        if source is None:
            return {"__error__": "source JSON is invalid"}
        training_delta = _json_or_empty(self._training_delta.toPlainText(), {})
        if training_delta is None:
            return {"__error__": "training_delta JSON is invalid"}
        try:
            sample_count = int((self._sample_count.text() or "0").strip() or "0")
        except ValueError:
            return {"__error__": "sample_count must be an integer"}

        return {
            "snapshot_id": snap_id,
            "source": dict(source) if isinstance(source, dict) else {},
            "training_delta": dict(training_delta) if isinstance(training_delta, dict) else {},
            "sample_count": sample_count,
            "notes": self._notes.text().strip(),
        }


class LineageCatalogPanel(SelectablePanel, QWidget):
    """Continuous Learning catalog panel.

    Lineages on the left, detail + drops timeline on the right.
    """

    panel_entity_type = "lineage"

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
        self._datasets_cache: list[dict[str, Any]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # -- Header toolbar --
        header = QHBoxLayout()
        header.setSpacing(6)
        title = QLabel("Continuous Learning Catalog")
        f = QFont()
        f.setBold(True)
        title.setFont(f)
        header.addWidget(title)
        header.addStretch(1)

        self._sector_filter = QComboBox()
        self._sector_filter.addItem("All sectors", userData="")
        self._sector_filter.currentIndexChanged.connect(lambda _i: self._reload_lineages())
        header.addWidget(QLabel("Sector:"))
        header.addWidget(self._sector_filter)

        self._state_filter = QComboBox()
        self._state_filter.addItem("Any state", userData="")
        for st in ("active", "frozen", "archived"):
            self._state_filter.addItem(st, userData=st)
        self._state_filter.currentIndexChanged.connect(lambda _i: self._reload_lineages())
        header.addWidget(self._state_filter)

        new_btn = QPushButton("[NEW LINEAGE]")
        new_btn.clicked.connect(self._on_new_lineage)
        header.addWidget(new_btn)

        refresh_btn = QPushButton("[REFRESH]")
        refresh_btn.clicked.connect(self._reload_all)
        header.addWidget(refresh_btn)
        root.addLayout(header)

        dataset_bar = QHBoxLayout()
        dataset_bar.setSpacing(6)
        dataset_title = QLabel("Dataset Library")
        dataset_title.setProperty("isTitle", True)
        dataset_bar.addWidget(dataset_title)
        self._dataset_status = QLabel("No datasets loaded.")
        self._dataset_status.setProperty("muted", True)
        dataset_bar.addWidget(self._dataset_status, stretch=1)
        dataset_refresh = QPushButton("[REFRESH DATASETS]")
        dataset_refresh.clicked.connect(self._reload_datasets)
        dataset_bar.addWidget(dataset_refresh)
        root.addLayout(dataset_bar)

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
        root.addWidget(self._dataset_table, stretch=0)

        # -- Splitter: list | detail --
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

        self._detail_header = QLabel("Select a lineage.")
        self._detail_header.setWordWrap(True)
        self._detail_header.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        dl.addWidget(self._detail_header)

        action_row = QHBoxLayout()
        action_row.setSpacing(6)
        self._add_drop_btn = QPushButton("[ADD DROP]")
        self._add_drop_btn.clicked.connect(self._on_add_drop)
        self._fork_btn = QPushButton("[FORK]")
        self._fork_btn.clicked.connect(self._on_fork)
        self._freeze_btn = QPushButton("[FREEZE]")
        self._freeze_btn.clicked.connect(lambda: self._set_state("frozen"))
        self._activate_btn = QPushButton("[ACTIVATE]")
        self._activate_btn.clicked.connect(lambda: self._set_state("active"))
        self._archive_btn = QPushButton("[ARCHIVE]")
        self._archive_btn.clicked.connect(lambda: self._set_state("archived"))
        self._delete_btn = QPushButton("[DELETE]")
        self._delete_btn.clicked.connect(self._on_delete)
        for b in (
            self._add_drop_btn, self._fork_btn, self._freeze_btn,
            self._activate_btn, self._archive_btn, self._delete_btn,
        ):
            b.setEnabled(False)
            action_row.addWidget(b)
        action_row.addStretch(1)
        dl.addLayout(action_row)

        self._drops_table = QTableWidget(0, 7)
        self._drops_table.setHorizontalHeaderLabels([
            "#", "Snapshot", "Samples", "SHA256", "Duration", "Created", "Notes",
        ])
        self._drops_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._drops_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._drops_table.setAlternatingRowColors(True)
        self._drops_table.verticalHeader().setVisible(False)
        hh = self._drops_table.horizontalHeader()
        hh.setStretchLastSection(True)
        hh.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)
        dl.addWidget(self._drops_table, stretch=1)

        prov_lbl = QLabel("W3C PROV (lineage closure)")
        prov_lbl.setProperty("isTitle", True)
        dl.addWidget(prov_lbl)
        self._prov_json = QPlainTextEdit()
        self._prov_json.setReadOnly(True)
        self._prov_json.setPlaceholderText("Select a non-registry lineage to load PROV-JSON.")
        self._prov_json.setMinimumHeight(140)
        dl.addWidget(self._prov_json, stretch=0)

        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 700])
        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    # Public entrypoints
    # ------------------------------------------------------------------

    def reload(self) -> None:
        self._reload_all()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _reload_all(self) -> None:
        self._reload_sectors()
        self._reload_datasets()
        self._reload_lineages()

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
        self._dataset_status.setText(f"{len(self._datasets_cache)} dataset{'s' if len(self._datasets_cache) != 1 else ''} available for drops")

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

    def _reload_lineages(self) -> None:
        params = []
        sp = self._sector_filter.currentData() or ""
        if sp:
            params.append(f"sector_path={sp}")
        st = self._state_filter.currentData() or ""
        if st:
            params.append(f"state={st}")
        qs = ("?" + "&".join(params)) if params else ""
        try:
            resp = self._http_get(f"/lineages{qs}")
        except Exception as exc:
            self.errorRaised.emit(f"lineages: {exc}")
            return
        items = []
        if isinstance(resp, dict):
            items = resp.get("items") or []
        self._list.clear()
        for ln in items:
            lid = str(ln.get("lineage_id") or "")
            name = str(ln.get("name") or "(unnamed)")
            state = str(ln.get("state") or "active")
            sector_path = str(ln.get("sector_path") or "/")
            head = str(ln.get("head_snapshot_id") or "")
            origin = "[REGISTRY] " if _is_registry_lineage(lid) else ""
            label = f"{origin}{name}\n  {sector_path}  \u00B7  {state}  \u00B7  head={_short(head, 16)}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, lid)
            self._list.addItem(item)
        # Re-select previously-selected
        if self._selected_id:
            for i in range(self._list.count()):
                it = self._list.item(i)
                if it.data(Qt.ItemDataRole.UserRole) == self._selected_id:
                    self._list.setCurrentRow(i)
                    break

    def select_lineage(self, lineage_id: str) -> None:
        """Public hook used by the Ecosystem quick-nav router."""
        lid = str(lineage_id or "").strip()
        if not lid:
            return
        for i in range(self._list.count()):
            it = self._list.item(i)
            if str(it.data(Qt.ItemDataRole.UserRole) or "") == lid:
                self._list.setCurrentRow(i)
                return
        # Not currently in the list — reload (filters may have hidden it),
        # then try once more.
        try:
            self._reload_lineages()
        except Exception:
            return
        for i in range(self._list.count()):
            it = self._list.item(i)
            if str(it.data(Qt.ItemDataRole.UserRole) or "") == lid:
                self._list.setCurrentRow(i)
                return

    def select_lineage_for_scenario(self, scenario: str) -> None:
        """Auto-select the lineage that belongs to ``scenario``.

        Prefers a user-defined lineage in the LineageStore whose name matches
        the scenario (case-insensitive). Falls back to the synthetic registry
        lineage ``registry:<scenario>``.
        """
        name = str(scenario or "").strip()
        if not name:
            return
        target_registry = f"{_REGISTRY_LINEAGE_PREFIX}{name}"
        wanted_lower = name.lower()

        def _try_select() -> bool:
            registry_idx: Optional[int] = None
            name_match_idx: Optional[int] = None
            for i in range(self._list.count()):
                it = self._list.item(i)
                lid = str(it.data(Qt.ItemDataRole.UserRole) or "")
                if lid == target_registry:
                    registry_idx = i
                    continue
                if _is_registry_lineage(lid):
                    continue
                # Match LineageStore lineage by exact-name (case-insensitive).
                # We compare against the first line of the label, which is the
                # raw lineage name with no decoration.
                first_line = str(it.text() or "").split("\n", 1)[0].strip().lower()
                if first_line == wanted_lower:
                    name_match_idx = i
            target = name_match_idx if name_match_idx is not None else registry_idx
            if target is not None:
                self._list.setCurrentRow(target)
                return True
            return False

        if _try_select():
            return
        try:
            self._reload_lineages()
        except Exception:
            return
        _try_select()

    def _on_selection_changed(self) -> None:
        items = self._list.selectedItems()
        if not items:
            self._clear_detail()
            return
        lid = items[0].data(Qt.ItemDataRole.UserRole)
        if not lid:
            self._clear_detail()
            return
        self._selected_id = str(lid)
        self.emit_entity_selected("lineage", self._selected_id)
        try:
            resp = self._http_get(f"/lineages/{urllib.parse.quote(str(lid), safe='')}")
        except Exception as exc:
            self.errorRaised.emit(f"lineage detail: {exc}")
            return
        if not isinstance(resp, dict):
            self._clear_detail()
            return
        self._current = resp
        self._render_detail(resp)

    def _clear_detail(self) -> None:
        self._selected_id = None
        self._current = None
        self._detail_header.setText("Select a lineage.")
        self._drops_table.setRowCount(0)
        self._prov_json.clear()
        for b in (
            self._add_drop_btn, self._fork_btn, self._freeze_btn,
            self._activate_btn, self._archive_btn, self._delete_btn,
        ):
            b.setEnabled(False)

    def _render_detail(self, detail: dict[str, Any]) -> None:
        lineage = detail.get("lineage") or {}
        drops = detail.get("drops") or []
        state = str(lineage.get("state") or "active")

        parts = [
            f"<b>{lineage.get('name','(unnamed)')}</b>",
            f"sector: {lineage.get('sector_path','/')}",
            f"state: {state}",
            f"strategy: {lineage.get('update_strategy','head_only')}",
            f"base: {_short(str(lineage.get('base_snapshot_id','')), 18)}",
            f"head: {_short(str(lineage.get('head_snapshot_id','')), 18)}",
            f"drops: {len(drops)}",
        ]
        desc = str(lineage.get("description") or "").strip()
        if desc:
            parts.append(f"<i>{desc}</i>")
        self._detail_header.setText("  \u00B7  ".join(parts))

        self._drops_table.setRowCount(0)
        for d in drops:
            r = self._drops_table.rowCount()
            self._drops_table.insertRow(r)
            values = [
                str(d.get("drop_index", "")),
                _short(str(d.get("snapshot_id", "")), 18),
                str(d.get("sample_count", 0)),
                _short(str(d.get("data_sha256", "")), 10),
                f"{int(d.get('duration_ms', 0))} ms",
                _fmt_ts(d.get("finished_at") or d.get("started_at")),
                str(d.get("notes") or ""),
            ]
            for c, v in enumerate(values):
                self._drops_table.setItem(r, c, QTableWidgetItem(v))

        is_registry = _is_registry_lineage(lineage.get("lineage_id"))
        if is_registry:
            self._add_drop_btn.setEnabled(False)
            self._fork_btn.setEnabled(False)
            self._freeze_btn.setEnabled(False)
            self._activate_btn.setEnabled(False)
            self._archive_btn.setEnabled(False)
            self._delete_btn.setEnabled(False)
            self._prov_json.setPlainText(
                "Registry-derived lineages have no persisted W3C PROV overlay in cvops."
            )
        else:
            self._add_drop_btn.setEnabled(state == "active")
            self._fork_btn.setEnabled(True)
            self._freeze_btn.setEnabled(state == "active")
            self._activate_btn.setEnabled(state != "active")
            self._archive_btn.setEnabled(state != "archived")
            self._delete_btn.setEnabled(True)
            lid_q = urllib.parse.quote(str(lineage.get("lineage_id") or ""), safe="")
            try:
                prov = self._http_get(f"/lineages/{lid_q}/provenance")
                if isinstance(prov, dict) and isinstance(prov.get("prov"), dict):
                    raw = json.dumps(prov.get("prov"), indent=2, ensure_ascii=True)
                    if len(raw) > 120_000:
                        raw = raw[:120_000] + "\n... [truncated]"
                    self._prov_json.setPlainText(raw)
                else:
                    self._prov_json.setPlainText(json.dumps(prov, indent=2, ensure_ascii=True)[:120_000])
            except Exception as exc:
                self._prov_json.setPlainText(f"(could not load provenance: {exc})")

    # ------------------------------------------------------------------
    # Actions
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

    def _on_new_lineage(self) -> None:
        dlg = _NewLineageDialog(
            sectors=getattr(self, "_sectors_cache", []),
            snapshots=self._fetch_snapshots(),
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            QMessageBox.warning(self, "New Lineage", "Name and base snapshot are required.")
            return
        try:
            self._http_post("/lineages", payload)
        except Exception as exc:
            self.errorRaised.emit(f"create lineage: {exc}")
            return
        self._reload_lineages()

    def _on_add_drop(self) -> None:
        if not self._selected_id:
            return
        dlg = _AddDropDialog(snapshots=self._fetch_snapshots(), dataset=self._selected_dataset(), parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            QMessageBox.warning(self, "Add Drop", "Snapshot is required.")
            return
        if "__error__" in payload:
            QMessageBox.warning(self, "Add Drop", payload["__error__"])
            return
        try:
            self._http_post(f"/lineages/{self._selected_id}/drops", payload)
        except Exception as exc:
            self.errorRaised.emit(f"add drop: {exc}")
            return
        self._on_selection_changed()

    def _on_fork(self) -> None:
        if not self._selected_id or not self._current:
            return
        drops = (self._current.get("drops") or [])
        default_idx = drops[-1].get("drop_index", 0) if drops else 0
        idx, ok = QInputDialog.getInt(
            self, "Fork Lineage",
            "Fork at drop_index:",
            value=int(default_idx), min=0, max=max(0, len(drops) - 1),
        )
        if not ok:
            return
        name, ok = QInputDialog.getText(self, "Fork Lineage", "New lineage name:")
        if not ok or not name.strip():
            return
        try:
            self._http_post(
                f"/lineages/{self._selected_id}/fork",
                {"at_drop_index": int(idx), "new_name": name.strip()},
            )
        except Exception as exc:
            self.errorRaised.emit(f"fork: {exc}")
            return
        self._reload_lineages()

    def _set_state(self, state: str) -> None:
        if not self._selected_id:
            return
        try:
            self._http_post(f"/lineages/{self._selected_id}/state", {"state": state})
        except Exception as exc:
            self.errorRaised.emit(f"set state: {exc}")
            return
        self._reload_lineages()
        self._on_selection_changed()

    def _on_delete(self) -> None:
        if not self._selected_id:
            return
        confirm = QMessageBox.question(
            self, "Delete Lineage",
            "Delete this lineage? Drops metadata will be removed. Snapshots remain in the snapshot store.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._http_delete(f"/lineages/{self._selected_id}")
        except Exception as exc:
            self.errorRaised.emit(f"delete: {exc}")
            return
        self._selected_id = None
        self._clear_detail()
        self._reload_lineages()
