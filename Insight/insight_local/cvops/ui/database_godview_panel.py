"""Database god-view panel — unified selector across every CV Ops data store.

Discovers and presents:
  * Operational SQLite stores under ``state/insight_local/`` (jobs, catalog,
    snapshots, lineages, ranges, forecasting signals).
  * The gallery SQLite at ``gallery/gallery.db``.
  * Image dataset directories under ``database/`` and ``mlops/database/``
    (any folder that looks like a YOLO/COCO-style dataset).
  * Tabular CSV drops under ``mlops/datasets/`` (single ``.csv`` files and
    folders whose root contains CSV tabular data).
  * JSON registries under ``mlops/`` (scenario registry, model registry,
    dataset registry entries).

The panel is mostly a god's-eye selector; dataset creation is exposed as a small
entry action for the database library.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

from PyQt6.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPolygonF
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .selectable_panel import SelectablePanel

try:
    from mlops.pipeline import registry as _mlops_registry
except Exception:  # pragma: no cover - import may fail without repo on sys.path
    _mlops_registry = None  # type: ignore[assignment]


_PREVIEW_ROW_LIMIT = 100
_PREVIEW_PAGE_SIZE = 100
_DATASET_HINTS: tuple[str, ...] = ("images", "labels", "train", "valid", "val", "test")
_REGISTRY_HINTS: tuple[str, ...] = (
    "registry.json",
    "model_registry.json",
)


class _ProvenanceGraphWidget(QWidget):
    """Compact in-panel graph renderer for provenance-focused god's-eye view."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._nodes: list[dict[str, Any]] = []
        self._edges: list[dict[str, Any]] = []
        self._focus_ids: set[str] = set()
        self._empty_text: str = "No provenance nodes available."
        self.setMinimumHeight(220)

    def set_graph(
        self,
        *,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        focus_ids: set[str],
        empty_text: str = "",
    ) -> None:
        self._nodes = list(nodes or [])
        self._edges = list(edges or [])
        self._focus_ids = {str(v) for v in (focus_ids or set()) if str(v)}
        if empty_text:
            self._empty_text = str(empty_text)
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(8, 8, -8, -8)
        p.fillRect(rect, QColor("#12161d"))
        p.setPen(QPen(QColor("#2b3442"), 1.0))
        p.drawRect(rect)

        if not self._nodes:
            p.setPen(QPen(QColor("#9aa8bd"), 1.0))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._empty_text)
            return

        positions = self._layout_positions(rect)
        node_types = {str(n.get("id") or ""): str(n.get("type") or "") for n in self._nodes}
        show_edge_labels = len(self._edges) <= 18

        for edge in self._edges:
            src = str(edge.get("source") or "")
            dst = str(edge.get("target") or "")
            if src not in positions or dst not in positions:
                continue
            etype = str(edge.get("type") or "")
            self._draw_arrow(
                p,
                start=positions[src],
                end=positions[dst],
                src_radius=13 if src in self._focus_ids else 10,
                dst_radius=13 if dst in self._focus_ids else 10,
                color=self._edge_color(etype),
                label=etype if show_edge_labels else "",
            )

        for node in self._nodes:
            nid = str(node.get("id") or "")
            label = str(node.get("label") or nid)
            ntype = node_types.get(nid, "")
            if nid not in positions:
                continue
            x, y = positions[nid]
            focused = nid in self._focus_ids
            radius = 13 if focused else 10
            fill = self._node_color(ntype)
            p.setBrush(fill)
            p.setPen(QPen(QColor("#f2f5fb") if focused else QColor("#1e2633"), 2.0 if focused else 1.0))
            p.drawEllipse(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
            p.setPen(QPen(QColor("#d7e2f2"), 1.0))
            p.drawText(int(x + radius + 4), int(y + 4), label[:40])

    def _draw_arrow(
        self,
        p: QPainter,
        *,
        start: tuple[float, float],
        end: tuple[float, float],
        src_radius: int,
        dst_radius: int,
        color: QColor,
        label: str,
    ) -> None:
        sx, sy = start
        dx, dy = end
        vx = dx - sx
        vy = dy - sy
        dist = math.hypot(vx, vy)
        if dist < 1e-3:
            return
        ux, uy = vx / dist, vy / dist
        # Trim the line so it stops at the rims of the two nodes.
        sx2 = sx + ux * src_radius
        sy2 = sy + uy * src_radius
        ex2 = dx - ux * (dst_radius + 2)
        ey2 = dy - uy * (dst_radius + 2)

        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(sx2, sy2), QPointF(ex2, ey2))

        # Arrowhead: a small triangle at the destination end.
        head_len = 9.0
        head_half = 4.5
        perp_x, perp_y = -uy, ux
        tip = QPointF(ex2, ey2)
        base_x = ex2 - ux * head_len
        base_y = ey2 - uy * head_len
        left = QPointF(base_x + perp_x * head_half, base_y + perp_y * head_half)
        right = QPointF(base_x - perp_x * head_half, base_y - perp_y * head_half)
        head = QPolygonF([tip, left, right])
        p.setBrush(QBrush(color))
        p.setPen(QPen(color, 1.0))
        p.drawPolygon(head)

        if label:
            mid_x = (sx2 + ex2) / 2.0
            mid_y = (sy2 + ey2) / 2.0
            # Nudge label off the line along its normal so it doesn't overlap.
            offset = 8.0
            tx = mid_x + perp_x * offset
            ty = mid_y + perp_y * offset
            p.setPen(QPen(QColor("#9aa8bd"), 1.0))
            p.drawText(int(tx), int(ty), label[:20])

    def _edge_color(self, etype: str) -> QColor:
        colors = {
            "has_head":      QColor("#ff9f59"),
            "derived_from":  QColor("#79b6ff"),
            "branched_from": QColor("#b78cff"),
            "belongs_to":    QColor("#57d6c8"),
            "governed_by":   QColor("#f5d76e"),
            "trains_on":     QColor("#4fa3ff"),
            "produces":      QColor("#7be39d"),
            "evaluates":     QColor("#ff7a90"),
        }
        return colors.get(etype, QColor("#6b7c95"))

    def _node_color(self, ntype: str) -> QColor:
        colors = {
            "dataset": QColor("#4fa3ff"),
            "database": QColor("#4fa3ff"),
            "dataset_snapshot": QColor("#57d6c8"),
            "model_snapshot": QColor("#79b6ff"),
            "lineage": QColor("#ff9f59"),
        }
        return colors.get(ntype, QColor("#8ea2bd"))

    def _layout_positions(self, rect) -> dict[str, tuple[float, float]]:
        cx = rect.center().x()
        cy = rect.center().y()
        width = max(80, rect.width())
        height = max(80, rect.height())
        rings = {
            "dataset": 0.18,
            "database": 0.18,
            "dataset_snapshot": 0.36,
            "model_snapshot": 0.56,
            "lineage": 0.76,
        }
        grouped: dict[str, list[dict[str, Any]]] = {}
        for node in self._nodes:
            t = str(node.get("type") or "other")
            grouped.setdefault(t, []).append(node)
        pos: dict[str, tuple[float, float]] = {}
        for ntype, nodes in grouped.items():
            radius_scale = rings.get(ntype, 0.86)
            rx = (width * radius_scale) / 2.0
            ry = (height * radius_scale) / 2.0
            total = max(1, len(nodes))
            for idx, node in enumerate(nodes):
                nid = str(node.get("id") or "")
                theta = (2.0 * math.pi * idx / total) - math.pi / 2.0
                jitter = (idx % 3) * 0.08
                x = cx + math.cos(theta + jitter) * rx
                y = cy + math.sin(theta + jitter) * ry
                pos[nid] = (x, y)
        return pos


@dataclass
class _DbEntry:
    """A discovered database/dataset/registry source."""

    key: str
    name: str
    kind: str           # "sqlite" | "dataset" | "registry" | "scrape" | "lineage"
    group: str          # display group on the left tree
    path: Path
    size_bytes: int = 0
    modified: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


def _human_bytes(n: float) -> str:
    if n <= 0:
        return "0 B"
    units = ("B", "KB", "MB", "GB", "TB")
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    return f"{n:.1f} {units[idx]}" if idx > 0 else f"{int(n)} {units[idx]}"


def _dir_size(path: Path, *, max_files: int = 20000) -> tuple[int, int]:
    """Return (total bytes, file count). Cheap walk with a hard cap."""
    total = 0
    count = 0
    try:
        for sub in path.rglob("*"):
            if not sub.is_file():
                continue
            try:
                total += sub.stat().st_size
            except OSError:
                continue
            count += 1
            if count >= max_files:
                break
    except OSError:
        pass
    return total, count


def _looks_like_dataset(path: Path) -> bool:
    if not path.is_dir():
        return False
    children = {p.name.lower() for p in path.iterdir() if p.is_dir()}
    if children & set(_DATASET_HINTS):
        return True
    return (path / "data.yaml").is_file()


def _count_split(images_root: Path) -> dict[str, int]:
    splits: dict[str, int] = {}
    if not images_root.is_dir():
        return splits
    for split_dir in images_root.iterdir():
        if not split_dir.is_dir():
            continue
        n = sum(1 for p in split_dir.iterdir() if p.is_file())
        splits[split_dir.name] = n
    return splits


class DatabaseGodViewPanel(SelectablePanel, QWidget):
    """Unified browser of every CV Ops data store."""

    panel_entity_type = "database"
    errorRaised = pyqtSignal(str)
    scenario_focused = pyqtSignal(dict)

    def __init__(
        self,
        project_root: Path,
        http_get: Optional[Callable[[str], dict[str, Any]]] = None,
        http_post: Optional[Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
        *,
        selector_only: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("databaseGodViewPanel")
        self._selector_only = bool(selector_only)
        self._root = Path(project_root).resolve()
        self._http_get = http_get
        self._http_post = http_post
        self._entries: dict[str, _DbEntry] = {}
        self._scenarios: list[dict[str, Any]] = []
        self._preview_page: int = 0
        self._preview_all_rows: list[list[str]] = []
        self._preview_all_headers: list[str] = []
        self._provenance_mode_enabled: bool = False
        self._current_entry: Optional[_DbEntry] = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("[DB] Database — God's Eye View")
        title_font = QFont()
        title_font.setBold(True)
        title_font.setPointSize(13)
        title.setFont(title_font)
        header.addWidget(title)
        header.addStretch(1)
        self._summary = QLabel("Scanning...")
        self._summary.setObjectName("databaseGodSummary")
        header.addWidget(self._summary)
        self._refresh_btn = QPushButton("[REFRESH]")
        self._refresh_btn.setObjectName("databaseGodRefresh")
        self._refresh_btn.clicked.connect(self.refresh)
        header.addWidget(self._refresh_btn)
        self._new_entry_btn = QPushButton("New Entry")
        self._new_entry_btn.setToolTip("Create an empty YOLO dataset folder under database/.")
        self._new_entry_btn.clicked.connect(self._create_empty_dataset_entry)
        self._new_entry_btn.setVisible(bool(self._http_post) and not self._selector_only)
        header.addWidget(self._new_entry_btn)
        self._provenance_btn = QPushButton("Provenance")
        self._provenance_btn.setCheckable(True)
        self._provenance_btn.toggled.connect(self.set_provenance_enabled)
        header.addWidget(self._provenance_btn)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(True)
        outer.addWidget(splitter, stretch=1)

        # Left: universal search over source/size tree
        left_wrap = QWidget()
        left_layout = QVBoxLayout(left_wrap)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        search_row = QHBoxLayout()
        search_row.setSpacing(8)
        self._tree_search = QLineEdit()
        self._tree_search.setObjectName("databaseGodTreeSearch")
        self._tree_search.setPlaceholderText(
            "Search sources — name, path, group, type, size, tables…"
        )
        self._tree_search.setClearButtonEnabled(True)
        self._tree_search.textChanged.connect(self._on_tree_search_changed)
        search_row.addWidget(self._tree_search, stretch=1)
        left_layout.addLayout(search_row)

        self._tree = QTreeWidget()
        self._tree.setObjectName("databaseGodTree")
        self._tree.setHeaderLabels(["Source", "Size"])
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tree.itemSelectionChanged.connect(self._on_selection)
        self._tree.header().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        left_layout.addWidget(self._tree, stretch=1)
        splitter.addWidget(left_wrap)

        # Right: details + provenance splitter
        right_split = QSplitter(Qt.Orientation.Horizontal)
        right_split.setChildrenCollapsible(False)
        right_split.setHandleWidth(3)
        self._detail_splitter = right_split

        right = QWidget()
        self._detail_surface = right
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        self._detail_header = QLabel("Select a source on the left.")
        self._detail_header.setObjectName("databaseGodDetailHeader")
        self._detail_header.setWordWrap(True)
        right_layout.addWidget(self._detail_header)

        self._meta_label = QLabel("")
        self._meta_label.setObjectName("databaseGodMeta")
        self._meta_label.setWordWrap(True)
        self._meta_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        right_layout.addWidget(self._meta_label)

        # Tables list (sqlite only)
        self._tables = QTableWidget(0, 2)
        self._tables.setObjectName("databaseGodTables")
        self._tables.setHorizontalHeaderLabels(["Table", "Rows"])
        self._tables.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._tables.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tables.verticalHeader().setVisible(False)
        self._tables.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._tables.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._tables.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._tables.itemSelectionChanged.connect(self._on_table_selected)
        self._tables.setMaximumHeight(220)
        right_layout.addWidget(self._tables)

        # Generic preview / contents area
        self._preview_label = QLabel("")
        self._preview_label.setObjectName("databaseGodPreviewLabel")
        right_layout.addWidget(self._preview_label)

        page_nav = QHBoxLayout()
        page_nav.setSpacing(6)
        self._prev_page_btn = QPushButton("Prev page")
        self._prev_page_btn.setMinimumHeight(28)
        self._prev_page_btn.setEnabled(False)
        self._prev_page_btn.clicked.connect(self._on_prev_page)
        page_nav.addWidget(self._prev_page_btn)
        self._page_info_label = QLabel("")
        self._page_info_label.setObjectName("databaseGodPageInfo")
        page_nav.addWidget(self._page_info_label)
        self._next_page_btn = QPushButton("Next page")
        self._next_page_btn.setMinimumHeight(28)
        self._next_page_btn.setEnabled(False)
        self._next_page_btn.clicked.connect(self._on_next_page)
        page_nav.addWidget(self._next_page_btn)
        page_nav.addStretch()
        right_layout.addLayout(page_nav)

        self._preview_table = QTableWidget(0, 0)
        self._preview_table.setObjectName("databaseGodPreviewTable")
        self._preview_table.verticalHeader().setVisible(False)
        self._preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._preview_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        right_layout.addWidget(self._preview_table, stretch=1)

        self._preview_text = QTextEdit()
        self._preview_text.setObjectName("databaseGodPreviewText")
        self._preview_text.setReadOnly(True)
        self._preview_text.setVisible(False)
        right_layout.addWidget(self._preview_text, stretch=1)

        right_split.addWidget(right)

        provenance_wrap = QFrame()
        self._provenance_wrap = provenance_wrap
        prov_layout = QVBoxLayout(provenance_wrap)
        prov_layout.setContentsMargins(6, 0, 0, 0)
        prov_layout.setSpacing(6)
        self._provenance_header = QLabel("Provenance (God's Eye)")
        prov_layout.addWidget(self._provenance_header)
        self._provenance_status = QLabel("Enable Provenance and select an asset.")
        self._provenance_status.setWordWrap(True)
        prov_layout.addWidget(self._provenance_status)
        self._provenance_graph = _ProvenanceGraphWidget()
        prov_layout.addWidget(self._provenance_graph, stretch=1)
        self._provenance_lineage_label = QLabel("Model Lineage")
        prov_layout.addWidget(self._provenance_lineage_label)
        self._provenance_lineage_text = QTextEdit()
        self._provenance_lineage_text.setReadOnly(True)
        self._provenance_lineage_text.setMinimumHeight(120)
        self._provenance_lineage_text.setPlainText("No model lineage resolved yet.")
        prov_layout.addWidget(self._provenance_lineage_text, stretch=0)
        right_split.addWidget(provenance_wrap)
        provenance_wrap.setVisible(False)

        splitter.addWidget(right_split)
        splitter.setCollapsible(0, True)
        splitter.setCollapsible(1, True)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([320, 760])

        if self._selector_only:
            def _relax_horizontal(widget: QWidget) -> None:
                widget.setMinimumWidth(0)
                pol = widget.sizePolicy()
                pol.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
                widget.setSizePolicy(pol)

            _relax_horizontal(self)
            _relax_horizontal(splitter)
            _relax_horizontal(left_wrap)
            _relax_horizontal(self._tree_search)
            _relax_horizontal(self._tree)
            for child in self.findChildren(QWidget):
                _relax_horizontal(child)
            self._provenance_btn.setVisible(False)
            right_split.hide()
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 0)
            splitter.setSizes([1, 0])

        self.refresh()

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().minimumSizeHint()
        if not self._selector_only:
            return hint
        return QSize(0, max(0, hint.height()))

    def sizeHint(self) -> QSize:  # type: ignore[override]
        hint = super().sizeHint()
        if not self._selector_only:
            return hint
        return QSize(max(220, hint.width()), hint.height())

    # ------------------------------------------------------------------ scan

    def refresh(self) -> None:
        try:
            self._entries = self._discover_all()
        except Exception as exc:  # pragma: no cover - defensive
            self.errorRaised.emit(f"Database scan failed: {exc}")
            self._entries = {}
        self._populate_tree()
        if self._current_entry is not None:
            self._update_provenance_for_entry(self._current_entry)

    def _create_empty_dataset_entry(self) -> None:
        if self._http_post is None:
            return
        name, ok = QInputDialog.getText(
            self,
            "New Database Entry",
            "Dataset folder name:",
            text="new_dataset",
        )
        name = str(name or "").strip()
        if not ok or not name:
            return
        try:
            payload = self._http_post(
                "/database/create_yolo_template",
                {"name": name, "classes": ["object"], "unique": True},
            )
        except Exception as exc:
            msg = f"New database entry failed: {exc}"
            self.errorRaised.emit(msg)
            QMessageBox.warning(self, "New Database Entry", msg)
            return
        slug = str((payload or {}).get("slug") or name).strip()
        self.refresh()
        QMessageBox.information(self, "New Database Entry", f"Created dataset '{slug}'.")

    def apply_scenarios(self, scenarios: list[dict[str, Any]]) -> None:
        """Refresh the Scenarios tree group with the latest scenario payloads.

        Driven by ``/scenarios`` so the god-view stays in sync with the catalog
        without scanning disk itself. Only fields already in the payload are
        used to enumerate related DB resources.
        """
        self._scenarios = [dict(s) for s in (scenarios or []) if isinstance(s, dict)]
        self._populate_tree()

    def select_scrape_job(self, slug: str) -> bool:
        """Locate the scrape-job entry whose folder name matches ``slug``.

        Used by the Collect mode to keep god-view context in sync with the
        scrape panel's active job. Returns True if a matching node was found
        and selected.
        """
        name = str(slug or "").strip()
        if not name:
            return False
        target_key: Optional[str] = None
        for entry in self._entries.values():
            if entry.kind == "scrape" and entry.name == name:
                target_key = entry.key
                break
        if target_key is None:
            # Tree may not be populated yet; trigger a scan once and retry.
            self.refresh()
            for entry in self._entries.values():
                if entry.kind == "scrape" and entry.name == name:
                    target_key = entry.key
                    break
        if target_key is None:
            return False
        for gi in range(self._tree.topLevelItemCount()):
            group_item = self._tree.topLevelItem(gi)
            for ci in range(group_item.childCount()):
                leaf = group_item.child(ci)
                if leaf.data(0, Qt.ItemDataRole.UserRole) == target_key:
                    self._tree.setCurrentItem(leaf)
                    self._tree.scrollToItem(leaf)
                    return True
        return False

    def set_provenance_enabled(self, enabled: bool) -> None:
        self._provenance_mode_enabled = bool(enabled)
        self._provenance_wrap.setVisible(self._provenance_mode_enabled)
        if self._provenance_mode_enabled:
            self._detail_splitter.setSizes([600, 420])
            if self._current_entry is None:
                self._auto_select_default_entry()
            if self._current_entry is not None:
                self._update_provenance_for_entry(self._current_entry)
            else:
                self._provenance_status.setText("Select an asset to inspect lineage/provenance.")
                self._provenance_lineage_text.setPlainText("No model lineage resolved yet.")
        else:
            self._detail_splitter.setSizes([1, 0])

    def _auto_select_default_entry(self) -> None:
        """Pick a sensible first asset so provenance has something to show.

        Preference: first dataset (most users care about dataset lineage),
        then first lineage, then first leaf in the tree.
        """
        preferred_kinds = ("dataset", "lineage", "sqlite", "registry", "scrape")
        by_kind: dict[str, QTreeWidgetItem] = {}
        for gi in range(self._tree.topLevelItemCount()):
            group_item = self._tree.topLevelItem(gi)
            for ci in range(group_item.childCount()):
                leaf = group_item.child(ci)
                key = leaf.data(0, Qt.ItemDataRole.UserRole)
                entry = self._entries.get(str(key) if key else "")
                if entry is None:
                    continue
                by_kind.setdefault(entry.kind, leaf)
        for kind in preferred_kinds:
            leaf = by_kind.get(kind)
            if leaf is not None:
                self._tree.setCurrentItem(leaf)
                self._tree.scrollToItem(leaf)
                return

    def _discover_all(self) -> dict[str, _DbEntry]:
        entries: dict[str, _DbEntry] = {}
        for entry in self._discover_sqlite():
            entries[entry.key] = entry
        for entry in self._discover_datasets():
            entries[entry.key] = entry
        for entry in self._discover_registries():
            entries[entry.key] = entry
        for entry in self._discover_scrape_jobs():
            entries[entry.key] = entry
        for entry in self._discover_lineages():
            entries[entry.key] = entry
        return entries

    def _discover_sqlite(self) -> list[_DbEntry]:
        out: list[_DbEntry] = []
        state_dir = self._root / "state"
        gallery_db = self._root / "gallery" / "gallery.db"

        candidates: list[tuple[str, Path]] = []
        if state_dir.is_dir():
            for db_path in state_dir.rglob("*.db"):
                if db_path.suffix != ".db":
                    continue
                if db_path.name == "lineages.db":
                    # Surfaced via the dedicated "Lineages" group instead.
                    continue
                rel = db_path.relative_to(self._root)
                top = rel.parts[1] if len(rel.parts) > 1 else "state"
                group = f"Operational [{top}]" if top != state_dir.name else "Operational"
                candidates.append((group, db_path))
        if gallery_db.is_file():
            candidates.append(("Gallery", gallery_db))

        for group, path in candidates:
            try:
                stat = path.stat()
            except OSError:
                continue
            key = f"sqlite::{path}"
            entry = _DbEntry(
                key=key,
                name=path.name,
                kind="sqlite",
                group=group,
                path=path,
                size_bytes=stat.st_size,
                modified=stat.st_mtime,
            )
            try:
                entry.extra["tables"] = self._sqlite_tables(path)
            except Exception as exc:
                entry.extra["tables"] = []
                entry.extra["error"] = str(exc)
            out.append(entry)
        out.sort(key=lambda e: (e.group, e.name))
        return out

    def _discover_datasets(self) -> list[_DbEntry]:
        roots = [self._root / "database", self._root / "mlops" / "database"]
        out: list[_DbEntry] = []
        for root in roots:
            if not root.is_dir():
                continue
            group = f"Datasets [{root.relative_to(self._root)}]"
            for child in sorted(root.iterdir()):
                if not child.is_dir():
                    continue
                if not _looks_like_dataset(child):
                    continue
                size, files = _dir_size(child)
                key = f"dataset::{child}"
                entry = _DbEntry(
                    key=key,
                    name=child.name,
                    kind="dataset",
                    group=group,
                    path=child,
                    size_bytes=size,
                    modified=child.stat().st_mtime if child.exists() else 0.0,
                )
                entry.extra["file_count"] = files
                entry.extra["slug"] = child.name
                entry.extra["splits"] = _count_split(child / "images") or _count_split(child)
                entry.extra["has_yaml"] = (child / "data.yaml").is_file()
                out.append(entry)

        tabular_root = self._root / "mlops" / "datasets"
        if tabular_root.is_dir():
            t_group = "Datasets [mlops/datasets]"
            seen_paths = {e.path.resolve() for e in out}
            for child in sorted(tabular_root.iterdir(), key=lambda x: x.name.lower()):
                if child.name.startswith("."):
                    continue
                try:
                    c_res = child.resolve()
                except OSError:
                    continue
                if c_res in seen_paths:
                    continue
                if child.is_file() and child.suffix.lower() == ".csv":
                    try:
                        st = child.stat()
                    except OSError:
                        continue
                    slug = child.stem
                    key = f"dataset::{c_res}"
                    entry = _DbEntry(
                        key=key,
                        name=slug,
                        kind="dataset",
                        group=t_group,
                        path=child,
                        size_bytes=st.st_size,
                        modified=st.st_mtime,
                    )
                    entry.extra["slug"] = slug
                    entry.extra["file_count"] = 1
                    entry.extra["splits"] = {}
                    entry.extra["has_yaml"] = False
                    entry.extra["tabular_csv_file"] = True
                    out.append(entry)
                    seen_paths.add(c_res)
                elif child.is_dir() and _mlops_registry is not None:
                    try:
                        fmt = _mlops_registry.detect_library_dataset_format(child)
                    except Exception:
                        fmt = ""
                    if fmt != _mlops_registry.LIBRARY_DATASET_FORMAT_CSV:
                        continue
                    size, files = _dir_size(child)
                    key = f"dataset::{c_res}"
                    entry = _DbEntry(
                        key=key,
                        name=child.name,
                        kind="dataset",
                        group=t_group,
                        path=child,
                        size_bytes=size,
                        modified=child.stat().st_mtime if child.exists() else 0.0,
                    )
                    entry.extra["slug"] = child.name
                    entry.extra["file_count"] = files
                    entry.extra["splits"] = _count_split(child / "images") or _count_split(child)
                    entry.extra["has_yaml"] = (child / "data.yaml").is_file()
                    entry.extra["tabular_csv_bundle"] = True
                    out.append(entry)
                    seen_paths.add(c_res)
        return out

    def _discover_registries(self) -> list[_DbEntry]:
        out: list[_DbEntry] = []
        mlops = self._root / "mlops"
        if not mlops.is_dir():
            return out
        # Top-level registry JSONs.
        for hint in _REGISTRY_HINTS:
            path = mlops / hint
            if path.is_file():
                out.append(self._registry_entry(path, group="Registries"))
        # Per-dataset registry entries.
        reg_dir = mlops / "dataset_registry"
        if reg_dir.is_dir():
            for child in sorted(reg_dir.iterdir()):
                if child.is_file() and child.suffix == ".json":
                    out.append(self._registry_entry(child, group="Registries [dataset_registry]"))
                elif child.is_dir():
                    manifest = child / "manifest.json"
                    if manifest.is_file():
                        out.append(self._registry_entry(manifest, group="Registries [dataset_registry]", display_name=child.name))
        return out

    def _discover_scrape_jobs(self) -> list[_DbEntry]:
        """Find every scrape job folder (has scrap.json) in database/."""
        out: list[_DbEntry] = []
        db_root = self._root / "database"
        if not db_root.is_dir():
            return out
        for child in sorted(db_root.iterdir()):
            if not child.is_dir():
                continue
            scrap_json = child / "scrap.json"
            if not scrap_json.is_file():
                continue
            size, files = _dir_size(child)
            raw_cnt = sum(1 for p in (child / "raw").iterdir() if p.is_file()) if (child / "raw").exists() else 0
            staged_cnt = sum(1 for p in (child / "staged").iterdir() if p.is_file()) if (child / "staged").exists() else 0
            try:
                job_data = json.loads(scrap_json.read_text(encoding="utf-8"))
            except Exception:
                job_data = {}
            entry = _DbEntry(
                key=f"scrape::{child}",
                name=child.name,
                kind="scrape",
                group="Scrape Jobs",
                path=child,
                size_bytes=size,
                modified=scrap_json.stat().st_mtime,
            )
            entry.extra["slug"] = child.name
            entry.extra["file_count"] = files
            entry.extra["raw_count"] = raw_cnt
            entry.extra["staged_count"] = staged_cnt
            entry.extra["job_data"] = job_data
            out.append(entry)
        return out

    def _discover_lineages(self) -> list[_DbEntry]:
        """Fetch lineages via the service and emit one entry per lineage.

        Each lineage (LineageStore or model-registry synthetic) becomes its
        own selectable source so the user can browse drops without digging
        into a raw sqlite preview.
        """
        out: list[_DbEntry] = []
        if self._http_get is None:
            return out
        try:
            payload = self._http_get("/lineages")
        except Exception as exc:
            self.errorRaised.emit(f"lineage discovery: {exc}")
            return out
        if not isinstance(payload, dict):
            return out
        items = payload.get("items") if isinstance(payload.get("items"), list) else []
        # The on-disk store; used so size/path stays useful even for synthetic entries.
        store_path = self._root / "state" / "insight_local" / "cvops" / "lineages.db"
        registry_path = self._root / "mlops" / "model_registry.json"
        try:
            store_stat = store_path.stat() if store_path.is_file() else None
        except OSError:
            store_stat = None
        try:
            registry_stat = registry_path.stat() if registry_path.is_file() else None
        except OSError:
            registry_stat = None

        for raw in items:
            if not isinstance(raw, dict):
                continue
            lid = str(raw.get("lineage_id") or "").strip()
            if not lid:
                continue
            is_registry = lid.startswith("registry:")
            base_path = registry_path if is_registry else store_path
            base_stat = registry_stat if is_registry else store_stat
            group = "Lineages [model registry]" if is_registry else "Lineages [continuous learning]"
            name = str(raw.get("name") or lid)
            entry = _DbEntry(
                key=f"lineage::{lid}",
                name=name,
                kind="lineage",
                group=group,
                path=base_path,
                size_bytes=int(base_stat.st_size) if base_stat else 0,
                modified=float(base_stat.st_mtime) if base_stat else float(raw.get("updated_at") or 0.0),
            )
            entry.extra["lineage_id"] = lid
            entry.extra["lineage"] = raw
            entry.extra["is_registry"] = is_registry
            out.append(entry)
        out.sort(key=lambda e: (e.group, e.name.lower()))
        return out

    def _registry_entry(self, path: Path, *, group: str, display_name: Optional[str] = None) -> _DbEntry:
        stat = path.stat()
        entry = _DbEntry(
            key=f"registry::{path}",
            name=display_name or path.name,
            kind="registry",
            group=group,
            path=path,
            size_bytes=stat.st_size,
            modified=stat.st_mtime,
        )
        try:
            with path.open("r", encoding="utf-8") as fh:
                entry.extra["json"] = json.load(fh)
        except Exception as exc:
            entry.extra["json"] = None
            entry.extra["error"] = str(exc)
        return entry

    def _sqlite_tables(self, path: Path) -> list[tuple[str, int]]:
        tables: list[tuple[str, int]] = []
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            )
            names = [row["name"] for row in cur.fetchall()]
            for name in names:
                try:
                    rcur = conn.execute(f'SELECT COUNT(*) AS n FROM "{name}"')
                    n = int(rcur.fetchone()["n"])
                except sqlite3.Error:
                    n = -1
                tables.append((name, n))
        return tables

    def _on_tree_search_changed(self, _text: str) -> None:
        self._populate_tree()

    def _entry_search_blob(self, entry: _DbEntry) -> str:
        """Lowercased single string for substring / token search."""
        parts: list[str] = [
            entry.group,
            entry.name,
            entry.kind,
            entry.key.replace("::", " "),
            _human_bytes(entry.size_bytes),
            str(int(entry.size_bytes)),
        ]
        try:
            parts.append(entry.path.relative_to(self._root).as_posix())
        except ValueError:
            parts.append(entry.path.as_posix())
        if entry.kind == "sqlite":
            for table_name, nrow in entry.extra.get("tables") or []:
                parts.append(str(table_name))
                parts.append(str(nrow))
        elif entry.kind == "scrape":
            jd = entry.extra.get("job_data") if isinstance(entry.extra.get("job_data"), dict) else {}
            parts.append(str(jd.get("topic") or ""))
            parts.append(str(jd.get("state") or ""))
            parts.append(str(jd.get("last_scrape_query") or ""))
            for c in jd.get("classes") or []:
                parts.append(str(c))
            parts.append(str(entry.extra.get("raw_count") or ""))
            parts.append(str(entry.extra.get("staged_count") or ""))
        elif entry.kind == "dataset":
            if entry.extra.get("tabular_csv_file"):
                parts.append("tabular csv file")
            if entry.extra.get("tabular_csv_bundle"):
                parts.append("tabular csv folder")
        blob = " ".join(p.strip() for p in parts if str(p).strip())
        return blob.lower()

    def _entry_matches_search(self, entry: _DbEntry, query: str) -> bool:
        q = query.strip().lower()
        if not q:
            return True
        blob = self._entry_search_blob(entry)
        for token in q.split():
            if not token:
                continue
            if token not in blob:
                return False
        return True

    # ------------------------------------------------------------------ tree

    def _populate_tree(self) -> None:
        current_key = None
        sel = self._tree.selectedItems()
        if sel:
            dk = sel[0].data(0, Qt.ItemDataRole.UserRole)
            if dk:
                current_key = str(dk)

        query = self._tree_search.text().strip()
        total_all = len(self._entries)

        filtered: list[_DbEntry] = [
            e for e in self._entries.values() if self._entry_matches_search(e, query)
        ]
        filtered.sort(key=lambda e: (e.group, e.name))

        self._tree.clear()
        groups: dict[str, QTreeWidgetItem] = {}
        visible_bytes = 0
        kind_counts: dict[str, int] = {"sqlite": 0, "dataset": 0, "registry": 0, "scrape": 0, "lineage": 0}

        for entry in filtered:
            parent = groups.get(entry.group)
            if parent is None:
                parent = QTreeWidgetItem([entry.group, ""])
                parent.setFirstColumnSpanned(False)
                parent.setFlags(parent.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                self._tree.addTopLevelItem(parent)
                groups[entry.group] = parent
            label = entry.name
            if entry.kind == "sqlite":
                tables = entry.extra.get("tables") or []
                label = f"[SQL] {entry.name}  ({len(tables)} tables)"
            elif entry.kind == "dataset":
                files = int(entry.extra.get("file_count", 0) or 0)
                if entry.extra.get("tabular_csv_file"):
                    label = f"[CSV] {entry.name}  (tabular file)"
                elif entry.extra.get("tabular_csv_bundle"):
                    label = f"[CSV] {entry.name}  ({files} files in folder)"
                else:
                    label = f"[DS] {entry.name}  ({files} files)"
            elif entry.kind == "registry":
                label = f"[REG] {entry.name}"
            elif entry.kind == "scrape":
                raw_cnt = entry.extra.get("raw_count", 0)
                staged_cnt = entry.extra.get("staged_count", 0)
                label = f"[SCRAPE] {entry.name}  (raw {raw_cnt} · staged {staged_cnt})"
            elif entry.kind == "lineage":
                lin = entry.extra.get("lineage") or {}
                state = str(lin.get("state") or "active")
                label = f"[LIN] {entry.name}  ({state})"
            item = QTreeWidgetItem([label, _human_bytes(entry.size_bytes)])
            item.setData(0, Qt.ItemDataRole.UserRole, entry.key)
            parent.addChild(item)
            visible_bytes += entry.size_bytes
            kind_counts[entry.kind] = kind_counts.get(entry.kind, 0) + 1

        # Scenarios live at the top of the tree as a synthetic group so users
        # see them first and can drill into the DB resources each one touches.
        self._insert_scenarios_group(query)

        self._tree.expandAll()

        base_summary = (
            f"{len(self._scenarios)} scenarios  |  "
            f"{kind_counts['sqlite']} sqlite  |  "
            f"{kind_counts['dataset']} datasets  |  "
            f"{kind_counts['registry']} registries  |  "
            f"{kind_counts['scrape']} scrape jobs  |  "
            f"{kind_counts['lineage']} lineages  |  "
            f"{_human_bytes(visible_bytes)} total"
        )
        if query:
            base_summary += f"  ·  Showing {len(filtered)} of {total_all} matching search"
        self._summary.setText(base_summary)

        if current_key:
            found = False
            for gi in range(self._tree.topLevelItemCount()):
                group_item = self._tree.topLevelItem(gi)
                for ci in range(group_item.childCount()):
                    leaf = group_item.child(ci)
                    key = leaf.data(0, Qt.ItemDataRole.UserRole)
                    if key == current_key:
                        self._tree.setCurrentItem(leaf)
                        self._tree.scrollToItem(leaf)
                        found = True
                        break
                if found:
                    break

    def _scenario_matches_search(self, scen: dict[str, Any], query: str) -> bool:
        q = query.strip().lower()
        if not q:
            return True
        blob_parts = [
            str(scen.get("name") or ""),
            str(scen.get("display_name") or ""),
            str(scen.get("description") or ""),
            str(scen.get("dataset") or ""),
            str(scen.get("backbone_type") or ""),
            str(scen.get("status") or ""),
            " ".join(str(c) for c in (scen.get("classes") or [])),
        ]
        latest = scen.get("latest_run") or {}
        if isinstance(latest, dict):
            blob_parts.extend(
                [
                    str(latest.get("run_dir") or ""),
                    str(latest.get("weights") or ""),
                    str(latest.get("final_model_path") or ""),
                ]
            )
        blob = " ".join(blob_parts).lower()
        return all(tok in blob for tok in q.split() if tok)

    def _insert_scenarios_group(self, query: str) -> None:
        if not self._scenarios:
            return
        visible = [s for s in self._scenarios if self._scenario_matches_search(s, query)]
        if not visible:
            return
        group = QTreeWidgetItem(["Scenarios", ""])
        group.setFirstColumnSpanned(False)
        group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        # Insert above all other top-level groups.
        self._tree.insertTopLevelItem(0, group)
        for scen in sorted(visible, key=lambda s: str(s.get("name") or "")):
            name = str(scen.get("name") or "")
            if not name:
                continue
            status = str(scen.get("status") or "")
            label = f"[SCEN] {name}"
            if status:
                label += f"  ({status})"
            parent = QTreeWidgetItem([label, ""])
            parent.setData(0, Qt.ItemDataRole.UserRole, f"scenario::{name}")
            group.addChild(parent)
            self._attach_scenario_children(parent, scen)

    def _attach_scenario_children(self, parent: QTreeWidgetItem, scen: dict[str, Any]) -> None:
        name = str(scen.get("name") or "")
        rows: list[tuple[str, str, str]] = []  # (label, value, kind)
        dataset = str(scen.get("dataset") or "").strip()
        if dataset:
            rows.append(("Dataset", dataset, "dataset"))
        backbone = str(scen.get("backbone_type") or "").strip()
        if backbone:
            rows.append(("Backbone", backbone, "info"))
        ds_count = scen.get("dataset_count")
        if isinstance(ds_count, (int, float)) and int(ds_count) > 0:
            rows.append(("Samples", str(int(ds_count)), "info"))
        classes = scen.get("classes") or []
        if isinstance(classes, list) and classes:
            preview = ", ".join(str(c) for c in classes[:6])
            if len(classes) > 6:
                preview += f", +{len(classes) - 6} more"
            rows.append((f"Classes ({len(classes)})", preview, "info"))
        base_model = str(scen.get("base_model") or "").strip()
        base_model_resolved = str(scen.get("base_model_resolved") or "").strip()
        if base_model_resolved or base_model:
            rows.append(("Base model", base_model_resolved or base_model, "model"))
        latest = scen.get("latest_run") or {}
        if isinstance(latest, dict):
            run_dir = str(latest.get("run_dir") or "").strip()
            if run_dir:
                version = str(latest.get("version") or "")
                label = f"Latest run{f' v{version}' if version else ''}"
                rows.append((label, run_dir, "run"))
            weights = str(latest.get("weights") or "").strip()
            if weights:
                rows.append(("Weights", weights, "model"))
            map50 = latest.get("map50")
            if isinstance(map50, (int, float)):
                rows.append(("mAP50", f"{float(map50):.3f}", "info"))
        history_count = scen.get("history_count")
        if isinstance(history_count, int) and history_count > 0:
            rows.append(("Run history", f"{history_count} runs", "info"))
        if bool(scen.get("verified")):
            rows.append(("Verified", "yes", "info"))
        if not rows:
            placeholder = QTreeWidgetItem(["(no resources yet)", ""])
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            parent.addChild(placeholder)
            return
        for label, value, kind in rows:
            child = QTreeWidgetItem([f"{label}: {value}", ""])
            child.setData(0, Qt.ItemDataRole.UserRole, f"scenario::{name}::{label.lower()}")
            child.setToolTip(0, value)
            if kind == "info":
                # Pure metadata — not a navigable resource.
                child.setFlags(child.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            parent.addChild(child)

    # --------------------------------------------------------------- details

    def _on_selection(self) -> None:
        items = self._tree.selectedItems()
        if not items:
            return
        key = items[0].data(0, Qt.ItemDataRole.UserRole)
        if not key:
            return
        key_str = str(key)
        if key_str.startswith("scenario::"):
            parts = key_str.split("::")
            name = parts[1] if len(parts) >= 2 else ""
            scen = next((s for s in self._scenarios if str(s.get("name") or "") == name), None)
            if isinstance(scen, dict):
                self.scenario_focused.emit(dict(scen))
            self._show_scenario_from_key(key_str)
            return
        entry = self._entries.get(key_str)
        if entry is None:
            return
        self._current_entry = entry
        self.emit_entity_selected("database", entry.path.as_posix())
        self._show_entry(entry)
        self._update_provenance_for_entry(entry)

    def _show_scenario_from_key(self, key: str) -> None:
        # Strip the trailing ::<child> if present so we always resolve the scenario.
        parts = key.split("::", 2)
        if len(parts) < 2:
            return
        name = parts[1]
        scen = next((s for s in self._scenarios if str(s.get("name") or "") == name), None)
        if scen is None:
            return
        self._render_scenario_detail(scen)

    def _render_scenario_detail(self, scen: dict[str, Any]) -> None:
        name = str(scen.get("name") or "")
        status = str(scen.get("status") or "-")
        display = str(scen.get("display_name") or name)
        self._detail_header.setText(f"[SCEN] {display}  ({status})")
        meta_lines = [f"name: {name}"]
        if display and display != name:
            meta_lines.append(f"display: {display}")
        desc = str(scen.get("description") or "").strip()
        if desc:
            meta_lines.append(f"description: {desc}")
        meta_lines.append(f"status: {status}")
        backbone = str(scen.get("backbone_type") or "").strip()
        if backbone:
            meta_lines.append(f"backbone: {backbone}")
        ds = str(scen.get("dataset") or "").strip()
        if ds:
            meta_lines.append(f"dataset: {ds}")
        ds_count = scen.get("dataset_count")
        if isinstance(ds_count, (int, float)):
            meta_lines.append(f"sample count: {int(ds_count)}")
        classes = scen.get("classes") or []
        if isinstance(classes, list) and classes:
            meta_lines.append(f"classes ({len(classes)}): {', '.join(str(c) for c in classes)}")
        latest = scen.get("latest_run") or {}
        if isinstance(latest, dict) and latest:
            ver = latest.get("version")
            if ver:
                meta_lines.append(f"latest run version: {ver}")
            map50 = latest.get("map50")
            if isinstance(map50, (int, float)):
                meta_lines.append(f"latest mAP50: {float(map50):.3f}")
            run_dir = str(latest.get("run_dir") or "").strip()
            if run_dir:
                meta_lines.append(f"run dir: {run_dir}")
            weights = str(latest.get("weights") or "").strip()
            if weights:
                meta_lines.append(f"weights: {weights}")
        base_model_resolved = str(scen.get("base_model_resolved") or "").strip()
        if base_model_resolved:
            meta_lines.append(f"base model: {base_model_resolved}")
        if bool(scen.get("verified")):
            meta_lines.append("verified: yes")
        self._meta_label.setText("\n".join(meta_lines))
        # Tables/preview don't apply to scenarios — clear them.
        self._tables.setRowCount(0)
        self._preview_label.setText("")
        self._preview_all_rows = []
        self._preview_all_headers = []
        self._preview_page = 0
        self._prev_page_btn.setEnabled(False)
        self._next_page_btn.setEnabled(False)
        self._current_entry = None

    def _show_entry(self, entry: _DbEntry) -> None:
        self._detail_header.setText(f"[{entry.kind.upper()}] {entry.name}")
        modified = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.modified)) if entry.modified else "-"
        try:
            rel = entry.path.relative_to(self._root).as_posix()
        except ValueError:
            rel = entry.path.as_posix()
        meta_lines = [
            f"path: {rel}",
            f"size: {_human_bytes(entry.size_bytes)}",
            f"modified: {modified}",
            f"group: {entry.group}",
        ]
        if entry.extra.get("error"):
            meta_lines.append(f"error: {entry.extra['error']}")
        self._meta_label.setText("\n".join(meta_lines))

        # Reset preview surfaces.
        self._tables.setRowCount(0)
        self._preview_table.setRowCount(0)
        self._preview_table.setColumnCount(0)
        self._preview_text.clear()
        self._preview_text.setVisible(False)
        self._preview_table.setVisible(False)
        self._preview_label.setText("")
        self._preview_page = 0
        self._preview_all_rows = []
        self._preview_all_headers = []
        self._prev_page_btn.setEnabled(False)
        self._next_page_btn.setEnabled(False)
        self._page_info_label.setText("")

        if entry.kind == "sqlite":
            self._render_sqlite(entry)
        elif entry.kind == "dataset":
            self._render_dataset(entry)
        elif entry.kind == "registry":
            self._render_registry(entry)
        elif entry.kind == "scrape":
            self._render_scrape(entry)
        elif entry.kind == "lineage":
            self._render_lineage(entry)

    def _update_provenance_for_entry(self, entry: _DbEntry) -> None:
        if not self._provenance_mode_enabled:
            return
        graph = {}
        if self._http_get is not None:
            try:
                payload = self._http_get("/ontology/graph")
                graph = payload if isinstance(payload, dict) else {}
            except Exception as exc:
                self._provenance_status.setText(f"Provenance graph fetch failed: {exc}")
                self._provenance_graph.set_graph(nodes=[], edges=[], focus_ids=set(), empty_text="Failed to load ontology graph.")
                self._provenance_lineage_text.setPlainText("Model lineage unavailable: ontology graph fetch failed.")
                return

        nodes_raw = graph.get("nodes") if isinstance(graph.get("nodes"), list) else []
        edges_raw = graph.get("edges") if isinstance(graph.get("edges"), list) else []
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        # /ontology/graph returns flat {id,type,label,meta}; tolerate the
        # Cytoscape {data: {...}} envelope too in case callers swap endpoints.
        for node in nodes_raw:
            if not isinstance(node, dict):
                continue
            src = node.get("data") if isinstance(node.get("data"), dict) else node
            nid = str(src.get("id") or "")
            if not nid:
                continue
            nodes.append({
                "id": nid,
                "type": str(src.get("type") or ""),
                "label": str(src.get("label") or nid),
            })
        for edge in edges_raw:
            if not isinstance(edge, dict):
                continue
            src_obj = edge.get("data") if isinstance(edge.get("data"), dict) else edge
            s = str(src_obj.get("source") or "")
            t = str(src_obj.get("target") or "")
            if not s or not t:
                continue
            edges.append({"source": s, "target": t, "type": str(src_obj.get("type") or "")})

        filtered_nodes, filtered_edges, focus_ids, msg = self._build_provenance_slice(entry, nodes, edges)
        self._provenance_status.setText(msg)
        self._provenance_graph.set_graph(
            nodes=filtered_nodes,
            edges=filtered_edges,
            focus_ids=focus_ids,
            empty_text="No eligible provenance nodes in ontology graph.",
        )
        self._render_model_lineage_details(filtered_nodes, filtered_edges, focus_ids)

    def _build_provenance_slice(
        self,
        entry: _DbEntry,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], str]:
        node_map = {str(n.get("id") or ""): n for n in nodes if str(n.get("id") or "")}
        adj: dict[str, set[str]] = {}
        for e in edges:
            s = str(e.get("source") or "")
            t = str(e.get("target") or "")
            if not s or not t:
                continue
            adj.setdefault(s, set()).add(t)
            adj.setdefault(t, set()).add(s)

        candidates = self._entry_candidate_tokens(entry)
        lineage_id = self._resolve_lineage_id(candidates)
        matched: set[str] = set()
        if lineage_id:
            lid = f"lineage:{lineage_id}" if not str(lineage_id).startswith("lineage:") else str(lineage_id)
            if lid in node_map:
                matched.add(lid)

        for node in nodes:
            nid = str(node.get("id") or "")
            ntype = str(node.get("type") or "")
            label = str(node.get("label") or "").lower()
            tail = nid.split(":", 1)[-1].lower()
            if ntype not in {"dataset", "database", "dataset_snapshot"}:
                continue
            if any(tok and (tok in label or tok == tail or tail.endswith(f"/{tok}")) for tok in candidates):
                matched.add(nid)

        focus_ids = set(matched)
        keep: set[str] = set()
        for src in matched:
            keep.add(src)
            for neigh in adj.get(src, set()):
                ntype = str(node_map.get(neigh, {}).get("type") or "")
                if ntype in {"lineage", "dataset_snapshot", "model_snapshot", "dataset", "database"}:
                    keep.add(neigh)
                    for neigh2 in adj.get(neigh, set()):
                        ntype2 = str(node_map.get(neigh2, {}).get("type") or "")
                        if ntype2 in {"lineage", "dataset_snapshot", "model_snapshot", "dataset", "database"}:
                            keep.add(neigh2)

        if not keep:
            # Fallback global provenance slice.
            keep = {
                nid for nid, n in node_map.items()
                if str(n.get("type") or "") in {"lineage", "model_snapshot", "dataset_snapshot"}
            }
            focus_ids = set()
            msg = "No direct lineage match for selected asset. Showing global provenance slice."
        else:
            msg = "Showing provenance focused on selected asset."

        out_nodes = [node_map[nid] for nid in keep if nid in node_map]
        out_edges = [
            e for e in edges
            if str(e.get("source") or "") in keep and str(e.get("target") or "") in keep
        ]
        return out_nodes, out_edges, focus_ids, msg

    def _render_model_lineage_details(
        self,
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
        focus_ids: set[str],
    ) -> None:
        node_map = {str(n.get("id") or ""): n for n in nodes if str(n.get("id") or "")}
        lineage_nodes = [n for n in nodes if str(n.get("type") or "") == "lineage"]
        model_nodes = [n for n in nodes if str(n.get("type") or "") == "model_snapshot"]
        ds_nodes = [n for n in nodes if str(n.get("type") or "") in {"dataset", "database", "dataset_snapshot"}]
        if not lineage_nodes and not model_nodes:
            self._provenance_lineage_text.setPlainText("No model lineage found for this dataset selection.")
            return

        incoming_to: dict[str, list[str]] = {}
        outgoing_from: dict[str, list[str]] = {}
        for e in edges:
            src = str(e.get("source") or "")
            dst = str(e.get("target") or "")
            if src and dst:
                outgoing_from.setdefault(src, []).append(dst)
                incoming_to.setdefault(dst, []).append(src)

        lines: list[str] = []
        if ds_nodes:
            lines.append("Dataset focus:")
            for n in ds_nodes[:4]:
                nid = str(n.get("id") or "")
                mark = "*" if nid in focus_ids else "-"
                lines.append(f"  {mark} {n.get('label') or nid}")

        if lineage_nodes:
            lines.append("")
            lines.append("Lineages:")
            for lin in lineage_nodes[:10]:
                lid = str(lin.get("id") or "")
                lines.append(f"  - {lin.get('label') or lid}")
                attached_models = []
                for nid in incoming_to.get(lid, []) + outgoing_from.get(lid, []):
                    node = node_map.get(nid) or {}
                    if str(node.get("type") or "") == "model_snapshot":
                        attached_models.append(str(node.get("label") or nid))
                if attached_models:
                    for model_name in sorted(set(attached_models))[:8]:
                        lines.append(f"      model: {model_name}")

        if model_nodes:
            lines.append("")
            lines.append("Model snapshots:")
            for m in model_nodes[:12]:
                mid = str(m.get("id") or "")
                lines.append(f"  - {m.get('label') or mid}")

        self._provenance_lineage_text.setPlainText("\n".join(lines) if lines else "No model lineage found for this dataset selection.")

    def _entry_candidate_tokens(self, entry: _DbEntry) -> list[str]:
        raw: list[str] = [entry.name, entry.path.name, entry.path.stem]
        slug = str(entry.extra.get("slug") or "").strip()
        if slug:
            raw.append(slug)
        if entry.kind == "lineage":
            lid = str(entry.extra.get("lineage_id") or "").strip()
            if lid:
                raw.append(lid.replace("registry:", ""))
                raw.append(lid)
        tokens: list[str] = []
        for part in raw:
            for tok in re.split(r"[^a-zA-Z0-9_]+", str(part).strip().lower()):
                tok = tok.strip()
                if tok and tok not in tokens:
                    tokens.append(tok)
        return tokens

    def _resolve_lineage_id(self, candidate_tokens: list[str]) -> str:
        if self._http_get is None:
            return ""
        try:
            payload = self._http_get("/lineages")
        except Exception:
            return ""
        items = payload.get("items") if isinstance(payload, dict) and isinstance(payload.get("items"), list) else []
        if not items:
            return ""

        # 1) exact registry lineage id.
        for tok in candidate_tokens:
            target = f"registry:{tok}"
            for row in items:
                if not isinstance(row, dict):
                    continue
                lid = str(row.get("lineage_id") or "")
                if lid == target:
                    return lid

        # 2) exact lineage name match.
        for tok in candidate_tokens:
            for row in items:
                if not isinstance(row, dict):
                    continue
                nm = str(row.get("name") or "").strip().lower()
                if nm and nm == tok:
                    return str(row.get("lineage_id") or "")

        # 3) lineage name contains token.
        for tok in candidate_tokens:
            for row in items:
                if not isinstance(row, dict):
                    continue
                nm = str(row.get("name") or "").strip().lower()
                if nm and tok and tok in nm:
                    return str(row.get("lineage_id") or "")
        return ""

    def _render_sqlite(self, entry: _DbEntry) -> None:
        tables: list[tuple[str, int]] = entry.extra.get("tables") or []
        self._tables.setVisible(True)
        self._tables.setRowCount(len(tables))
        for row, (name, count) in enumerate(tables):
            n_item = QTableWidgetItem(name)
            r_item = QTableWidgetItem(str(count) if count >= 0 else "-")
            self._tables.setItem(row, 0, n_item)
            self._tables.setItem(row, 1, r_item)
        self._preview_label.setText("Select a table to preview rows.")
        self._preview_table.setVisible(True)

    def _render_dataset(self, entry: _DbEntry) -> None:
        self._tables.setVisible(False)
        payload = self._load_dataset_payload(entry)
        if payload:
            self._render_dataset_payload(entry, payload)
            return

        splits: dict[str, int] = entry.extra.get("splits") or {}
        files: int = entry.extra.get("file_count", 0)
        has_yaml: bool = bool(entry.extra.get("has_yaml"))
        lines = [f"files (capped scan): {files}", f"data.yaml: {'yes' if has_yaml else 'no'}"]
        if splits:
            lines.append("splits:")
            for name in sorted(splits):
                lines.append(f"  - {name}: {splits[name]}")
        fallback_rows = self._dataset_fallback_rows(entry)
        if fallback_rows:
            lines.append("")
            lines.append(f"sample contents ({len(fallback_rows)} row{'s' if len(fallback_rows) != 1 else ''}):")
            lines.extend(f"  - {rel}  |  {_human_bytes(size)}" for rel, size in fallback_rows)
        if has_yaml:
            try:
                yaml_text = (entry.path / "data.yaml").read_text(encoding="utf-8")
                lines.append("")
                lines.append("data.yaml:")
                lines.append(yaml_text.strip())
            except OSError:
                pass
        self._preview_text.setPlainText("\n".join(lines))
        self._preview_text.setVisible(True)
        self._preview_label.setText("Dataset overview")

    def _load_dataset_payload(self, entry: _DbEntry) -> dict[str, Any]:
        if self._http_get is None:
            return {}
        slug = str(entry.extra.get("slug") or entry.name or "").strip()
        if not slug:
            return {}
        try:
            payload = self._http_get(f"/database/{quote(slug, safe='')}")
        except Exception as exc:
            entry.extra["preview_error"] = str(exc)
            return {}
        return payload if isinstance(payload, dict) else {}

    def _render_dataset_payload(self, entry: _DbEntry, payload: dict[str, Any]) -> None:
        fmt = str(payload.get("format") or "").strip()
        count = int(payload.get("count") or 0)
        splits = payload.get("split_counts") if isinstance(payload.get("split_counts"), dict) else {}
        classes = [str(c).strip() for c in (payload.get("classes") or []) if str(c).strip()]
        sha = str(payload.get("content_sha256") or "").strip()

        meta_lines = [self._meta_label.text()]
        if fmt:
            meta_lines.append(f"format: {fmt}")
        if count:
            meta_lines.append(f"items: {count}")
        if splits:
            parts = [f"{name} {int(value or 0)}" for name, value in sorted(splits.items(), key=lambda kv: str(kv[0]).lower())]
            if parts:
                meta_lines.append("splits: " + "  |  ".join(parts))
        if classes:
            shown = ", ".join(classes[:8])
            suffix = "..." if len(classes) > 8 else ""
            meta_lines.append(f"classes: {shown}{suffix}")
        if sha:
            meta_lines.append(f"content sha256: {sha}")
        if entry.extra.get("preview_error"):
            meta_lines.append(f"preview fetch: {entry.extra['preview_error']}")
        self._meta_label.setText("\n".join(line for line in meta_lines if line))

        headers, rows, total_rows = self._dataset_preview_rows(payload)
        if headers and rows:
            self._fill_preview_table(headers, rows)
            shown = len(rows)
            kind_label = "contents"
            if fmt == "csv_tabular":
                kind_label = "CSV files"
            elif fmt == "audiofolder_classification":
                kind_label = "audio assets"
            elif fmt == "face_csv":
                kind_label = "face images"
            else:
                kind_label = "dataset contents"
            self._preview_label.setText(f"{kind_label} — {total_rows} row{'s' if total_rows != 1 else ''}")
            self._preview_table.setVisible(True)
        yaml_path = (entry.path / "data.yaml") if entry.path.is_dir() else None
        if yaml_path is not None and yaml_path.is_file():
            try:
                yaml_text = yaml_path.read_text(encoding="utf-8")
            except OSError:
                yaml_text = ""
            if yaml_text.strip():
                self._preview_text.setPlainText(yaml_text.strip())
                self._preview_text.setVisible(True)
        elif not headers or not rows:
            self._preview_text.setPlainText("No dataset contents available.")
            self._preview_text.setVisible(True)

    def _dataset_preview_rows(self, payload: dict[str, Any]) -> tuple[list[str], list[list[str]], int]:
        fmt = str(payload.get("format") or "").strip().lower()
        if fmt == "csv_tabular":
            rows = [dict(row) for row in (payload.get("csv_files") or []) if isinstance(row, dict)]
            table_rows = [
                [
                    str(row.get("name") or row.get("filename") or row.get("path") or ""),
                    _human_bytes(float(row.get("size") or row.get("size_bytes") or 0)),
                    str(row.get("path") or ""),
                ]
                for row in rows
            ]
            return ["File", "Size", "Path"], table_rows, len(rows)
        if fmt == "audiofolder_classification":
            rows = [dict(row) for row in (payload.get("audio_files") or []) if isinstance(row, dict)]
            table_rows = [
                [
                    str(row.get("name") or row.get("relative_path") or row.get("path") or ""),
                    str(row.get("classification_label") or ""),
                    _human_bytes(float(row.get("size") or 0)),
                    str(row.get("relative_path") or row.get("path") or ""),
                ]
                for row in rows
            ]
            return ["Asset", "Class", "Size", "Relative Path"], table_rows, len(rows)

        rows = [dict(row) for row in (payload.get("images") or []) if isinstance(row, dict)]
        table_rows = [
            [
                str(row.get("display_name") or row.get("name") or row.get("relative_path") or ""),
                str(row.get("split") or ""),
                "yes" if bool(row.get("has_label")) else "missing",
                _human_bytes(float(row.get("size") or 0)),
                str(row.get("relative_path") or ""),
            ]
            for row in rows
        ]
        return ["Name", "Split", "Labeled", "Size", "Relative Path"], table_rows, len(rows)

    def _fill_preview_table(self, headers: list[str], rows: list[list[str]]) -> None:
        """Store all rows and render the current page."""
        self._preview_all_headers = headers
        self._preview_all_rows = rows
        self._preview_page = 0
        self._render_preview_page()

    def _render_preview_page(self) -> None:
        headers = self._preview_all_headers
        all_rows = self._preview_all_rows
        total = len(all_rows)
        page_size = _PREVIEW_PAGE_SIZE
        start = self._preview_page * page_size
        end = min(start + page_size, total)
        page_rows = all_rows[start:end]
        total_pages = max(1, (total + page_size - 1) // page_size)

        self._preview_table.clear()
        self._preview_table.setColumnCount(len(headers))
        self._preview_table.setHorizontalHeaderLabels(headers)
        self._preview_table.setRowCount(len(page_rows))
        for r, row in enumerate(page_rows):
            for c, value in enumerate(row):
                item = QTableWidgetItem(value)
                if headers[c].lower() in {"path", "relative path"}:
                    item.setToolTip(value)
                self._preview_table.setItem(r, c, item)
        header = self._preview_table.horizontalHeader()
        for idx, name in enumerate(headers):
            resize_mode = QHeaderView.ResizeMode.ResizeToContents
            if idx == 0 or "path" in name.lower():
                resize_mode = QHeaderView.ResizeMode.Stretch
            header.setSectionResizeMode(idx, resize_mode)

        self._page_info_label.setText(
            f"Page {self._preview_page + 1} of {total_pages}  ({start + 1}–{end} of {total})"
            if total else ""
        )
        self._prev_page_btn.setEnabled(self._preview_page > 0)
        self._next_page_btn.setEnabled(end < total)

    def _on_prev_page(self) -> None:
        if self._preview_page > 0:
            self._preview_page -= 1
            self._render_preview_page()

    def _on_next_page(self) -> None:
        total = len(self._preview_all_rows)
        if (self._preview_page + 1) * _PREVIEW_PAGE_SIZE < total:
            self._preview_page += 1
            self._render_preview_page()

    def _render_scrape(self, entry: _DbEntry) -> None:
        """Show scrape job meta and pageable file list of raw/ + staged/ images."""
        self._tables.setVisible(False)
        job_data = entry.extra.get("job_data") or {}
        raw_cnt = entry.extra.get("raw_count", 0)
        staged_cnt = entry.extra.get("staged_count", 0)
        label_cnt = len(dict(job_data.get("labels") or {}))
        classes = [str(c) for c in (job_data.get("classes") or []) if str(c).strip()]

        meta_lines = [self._meta_label.text()]
        meta_lines.append(f"topic: {job_data.get('topic') or entry.name}")
        meta_lines.append(f"state: {job_data.get('state') or '—'}")
        meta_lines.append(f"raw images: {raw_cnt}  ·  staged images: {staged_cnt}  ·  labeled: {label_cnt}")
        if classes:
            meta_lines.append(f"classes: {', '.join(classes)}")
        if job_data.get("last_scrape_query"):
            meta_lines.append(f"last query: {job_data['last_scrape_query']}")
        self._meta_label.setText("\n".join(l for l in meta_lines if l))

        # Build full file table: raw/ then staged/.
        suffixes = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tiff", ".tif"}
        rows: list[list[str]] = []
        labels_map: dict[str, str] = {}
        for fname in (job_data.get("labels") or {}):
            labels_map[str(fname)] = "yes"

        for folder, src in (("raw", "raw/"), ("staged", "staged/")):
            folder_path = entry.path / folder
            if not folder_path.exists():
                continue
            for p in sorted(folder_path.iterdir()):
                if not p.is_file() or p.suffix.lower() not in suffixes:
                    continue
                try:
                    size = p.stat().st_size
                except OSError:
                    size = 0
                labeled = labels_map.get(p.name, "—") if folder == "staged" else "—"
                rows.append([p.name, src, labeled, _human_bytes(size), p.as_posix()])

        if rows:
            self._fill_preview_table(["Name", "Source", "Labeled", "Size", "Path"], rows)
            self._preview_label.setText(f"Images — {len(rows)} file(s) (raw {raw_cnt} + staged {staged_cnt})")
            self._preview_table.setVisible(True)
        else:
            self._preview_text.setPlainText("No image files found in raw/ or staged/.")
            self._preview_text.setVisible(True)
            self._preview_label.setText("No images yet")

    def _render_lineage(self, entry: _DbEntry) -> None:
        """Show lineage metadata and drops timeline for a single lineage."""
        self._tables.setVisible(False)
        lineage = entry.extra.get("lineage") or {}
        lid = str(entry.extra.get("lineage_id") or "").strip()
        is_registry = bool(entry.extra.get("is_registry"))

        drops: list[dict[str, Any]] = []
        if lid and self._http_get is not None:
            try:
                detail = self._http_get(f"/lineages/{quote(lid, safe='')}")
            except Exception as exc:
                entry.extra["lineage_error"] = str(exc)
                detail = {}
            if isinstance(detail, dict):
                lin_detail = detail.get("lineage")
                if isinstance(lin_detail, dict):
                    lineage = lin_detail
                raw_drops = detail.get("drops")
                if isinstance(raw_drops, list):
                    drops = [d for d in raw_drops if isinstance(d, dict)]

        meta_lines = [self._meta_label.text()]
        meta_lines.append(f"lineage id: {lid}")
        meta_lines.append(f"source: {'model registry (read-only)' if is_registry else 'continuous learning store'}")
        meta_lines.append(f"name: {lineage.get('name','')}")
        meta_lines.append(f"sector: {lineage.get('sector_path','/')}")
        meta_lines.append(f"state: {lineage.get('state','active')}  ·  strategy: {lineage.get('update_strategy','head_only')}")
        meta_lines.append(f"base snap: {lineage.get('base_snapshot_id','')}")
        meta_lines.append(f"head snap: {lineage.get('head_snapshot_id','')}")
        meta_lines.append(f"drops: {len(drops)}")
        desc = str(lineage.get("description") or "").strip()
        if desc:
            meta_lines.append(f"description: {desc}")
        if entry.extra.get("lineage_error"):
            meta_lines.append(f"detail fetch: {entry.extra['lineage_error']}")
        self._meta_label.setText("\n".join(l for l in meta_lines if l))

        if drops:
            headers = ["#", "Snapshot", "Samples", "SHA256/Manifest", "Duration", "Created", "Notes"]
            rows: list[list[str]] = []
            for d in drops:
                finished = d.get("finished_at") or d.get("started_at")
                ts_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(float(finished))) if finished else "-"
                dur_ms = int(d.get("duration_ms") or 0)
                rows.append([
                    str(d.get("drop_index", "")),
                    str(d.get("snapshot_id", "")),
                    str(d.get("sample_count", 0)),
                    str(d.get("data_sha256") or ""),
                    f"{dur_ms} ms" if dur_ms else "-",
                    ts_text,
                    str(d.get("notes") or ""),
                ])
            self._fill_preview_table(headers, rows)
            self._preview_label.setText(f"Drops — {len(rows)} row{'s' if len(rows) != 1 else ''}")
            self._preview_table.setVisible(True)
        else:
            self._preview_text.setPlainText("This lineage has no drops yet.")
            self._preview_text.setVisible(True)
            self._preview_label.setText("Drops")

    def _dataset_fallback_rows(self, entry: _DbEntry) -> list[tuple[str, int]]:
        rows: list[tuple[str, int]] = []
        if entry.path.is_file():
            try:
                sz = int(entry.path.stat().st_size)
            except OSError:
                sz = 0
            return [(entry.path.name, sz)]
        roots = []
        images_root = entry.path / "images"
        if images_root.is_dir():
            roots.append(images_root)
        roots.append(entry.path)
        seen: set[str] = set()
        for root in roots:
            try:
                iterator = root.rglob("*")
            except OSError:
                continue
            for path in iterator:
                if not path.is_file():
                    continue
                try:
                    rel = path.relative_to(entry.path).as_posix()
                except ValueError:
                    rel = path.name
                if rel in seen:
                    continue
                seen.add(rel)
                try:
                    size = int(path.stat().st_size)
                except OSError:
                    size = 0
                rows.append((rel, size))
                if len(rows) >= min(_PREVIEW_ROW_LIMIT, 25):
                    return rows
        return rows

    def _render_registry(self, entry: _DbEntry) -> None:
        self._tables.setVisible(False)
        data = entry.extra.get("json")
        if data is None:
            self._preview_text.setPlainText(entry.extra.get("error") or "(unable to read registry)")
            self._preview_text.setVisible(True)
            return
        # If the registry is a list of records or has a single list member, show as table.
        rows = self._registry_rows(data)
        if rows:
            cols = sorted({k for row in rows for k in row.keys()})
            self._preview_table.setColumnCount(len(cols))
            self._preview_table.setHorizontalHeaderLabels(cols)
            self._preview_table.setRowCount(len(rows))
            for r, row in enumerate(rows):
                for c, key in enumerate(cols):
                    val = row.get(key, "")
                    if isinstance(val, (dict, list)):
                        val = json.dumps(val, ensure_ascii=False)
                    self._preview_table.setItem(r, c, QTableWidgetItem(str(val)))
            self._preview_table.setVisible(True)
            self._preview_label.setText(f"{len(rows)} entries")
        else:
            self._preview_text.setPlainText(json.dumps(data, indent=2, ensure_ascii=False))
            self._preview_text.setVisible(True)
            self._preview_label.setText("Registry contents")

    @staticmethod
    def _registry_rows(data: Any) -> list[dict[str, Any]]:
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data  # type: ignore[return-value]
        if isinstance(data, dict):
            for value in data.values():
                if isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
                    return value  # type: ignore[return-value]
        return []

    # ---------------------------------------------------------- table preview

    def _on_table_selected(self) -> None:
        items = self._tables.selectedItems()
        if not items:
            return
        items_in_row = [it for it in items if it.column() == 0]
        if not items_in_row:
            return
        table_name = items_in_row[0].text()
        # Resolve currently-selected sqlite entry.
        sel = self._tree.selectedItems()
        if not sel:
            return
        key = sel[0].data(0, Qt.ItemDataRole.UserRole)
        entry = self._entries.get(key) if key else None
        if entry is None or entry.kind != "sqlite":
            return
        try:
            cols, rows = self._sqlite_preview(entry.path, table_name)
        except Exception as exc:
            self._preview_text.setPlainText(f"error reading {table_name}: {exc}")
            self._preview_text.setVisible(True)
            self._preview_table.setVisible(False)
            return

        str_rows = [
            [("" if v is None else (str(v)[:500] + "..." if len(str(v)) > 500 else str(v))) for v in row]
            for row in rows
        ]
        self._fill_preview_table(cols, str_rows)
        self._preview_table.setVisible(True)
        self._preview_text.setVisible(False)
        self._preview_label.setText(f"{table_name} — {len(rows)} row{'s' if len(rows) != 1 else ''}")

    @staticmethod
    def _sqlite_preview(path: Path, table: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            cur = conn.execute(f'SELECT * FROM "{table}"')
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchall()
        return cols, rows
