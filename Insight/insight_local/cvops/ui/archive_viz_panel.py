from __future__ import annotations

import html
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


# Structure-tree item data roles. UserRole holds the owning object_id (existing
# convention); these two carry the Phase 3 edit target so action buttons know
# what is selected without re-deriving it from the payload.
_ROLE_KIND = int(Qt.ItemDataRole.UserRole) + 1
_ROLE_TARGET_ID = int(Qt.ItemDataRole.UserRole) + 2


def _fmt_duration(seconds: float) -> str:
    s = max(0.0, seconds)
    days = int(s // 86400)
    s -= days * 86400
    hours = int(s // 3600)
    s -= hours * 3600
    minutes = int(s // 60)
    secs = int(s % 60)
    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _fmt_ts(value: Any) -> str:
    try:
        token = float(value or 0.0)
    except Exception:
        return ""
    if token <= 0:
        return ""
    try:
        from datetime import datetime

        return datetime.fromtimestamp(token).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ""


def _fmt_bytes(value: Any) -> str:
    try:
        amount = int(value or 0)
    except Exception:
        return "0 B"
    if amount >= 1 << 30:
        return f"{amount / float(1 << 30):.1f} GB"
    if amount >= 1 << 20:
        return f"{amount / float(1 << 20):.1f} MB"
    if amount >= 1 << 10:
        return f"{amount / float(1 << 10):.0f} KB"
    return f"{amount} B"


def _badge_text(missing_count: int, edited_count: int) -> str:
    parts: list[str] = []
    if edited_count > 0:
        parts.append(f"E{edited_count}")
    if missing_count > 0:
        parts.append(f"!{missing_count}")
    return " ".join(parts)


def _count_summary(items: Any, *, limit: int = 4, empty_label: str = "n/a") -> str:
    if not isinstance(items, list) or not items:
        return empty_label
    parts: list[str] = []
    for item in items[:limit]:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or item.get("value") or "").strip() or empty_label
        parts.append(f"{label} ({int(item.get('count') or 0)})")
    return ", ".join(parts) if parts else empty_label


def _phase_goal_lines(payload: dict[str, Any], current_phase: str) -> str:
    goals = payload.get("phase_goals") if isinstance(payload, dict) else []
    if not isinstance(goals, list) or not goals:
        return "No phase-line goals available."
    lines = []
    for item in goals:
        if not isinstance(item, dict):
            continue
        phase = str(item.get("phase") or "")
        prefix = ">>" if phase == current_phase else "  "
        label = str(item.get("label") or phase)
        summary = str(item.get("summary") or "")
        lines.append(f"{prefix} {label}")
        if summary:
            lines.append(f"   {summary}")
    return "\n".join(lines).strip()


class _RoleEditorDialog(QDialog):
    def __init__(self, object_row: dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Set File Roles")
        self.resize(720, 340)
        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["File", "Path", "Role", "Ordinal"])
        self._table.verticalHeader().setVisible(False)

        outer = QVBoxLayout(self)
        title = QLabel(str(object_row.get("title") or "Object"))
        title.setStyleSheet("font-weight: 600;")
        outer.addWidget(title)
        outer.addWidget(self._table, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        for ref in object_row.get("files") or []:
            if not isinstance(ref, dict):
                continue
            row = self._table.rowCount()
            self._table.insertRow(row)
            file_id = str(ref.get("file_id") or "")
            file_meta = ref.get("file") if isinstance(ref.get("file"), dict) else {}
            rel_path = str(file_meta.get("relative_path") or file_id)
            self._table.setItem(row, 0, QTableWidgetItem(file_id))
            self._table.setItem(row, 1, QTableWidgetItem(rel_path))

            role_combo = QComboBox()
            for role in ("front", "back", "page", "detail", "recording", "transcript", "component"):
                role_combo.addItem(role, role)
            idx = max(0, role_combo.findData(str(ref.get("role") or "component")))
            role_combo.setCurrentIndex(idx)
            self._table.setCellWidget(row, 2, role_combo)

            ordinal = QSpinBox()
            ordinal.setRange(0, 9999)
            ordinal.setValue(int(ref.get("ordinal") or row + 1))
            self._table.setCellWidget(row, 3, ordinal)

        self._table.resizeColumnsToContents()

    def file_roles(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for row in range(self._table.rowCount()):
            file_id_item = self._table.item(row, 0)
            role_combo = self._table.cellWidget(row, 2)
            ordinal_spin = self._table.cellWidget(row, 3)
            if file_id_item is None or not isinstance(role_combo, QComboBox) or not isinstance(ordinal_spin, QSpinBox):
                continue
            rows.append(
                {
                    "file_id": str(file_id_item.text() or "").strip(),
                    "role": str(role_combo.currentData() or "component"),
                    "ordinal": int(ordinal_spin.value()),
                }
            )
        return rows


class _PinDateDialog(QDialog):
    """Phase 3 anchor resolution: pin an explicit earliest/laRange."""

    def __init__(self, anchor_row: dict[str, Any], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pin / Resolve Date")
        self.resize(420, 200)
        outer = QVBoxLayout(self)
        caption = QLabel(str(anchor_row.get("raw_expression") or anchor_row.get("anchor_id") or "anchor"))
        caption.setStyleSheet("font-weight: 600;")
        outer.addWidget(caption)
        form = QFormLayout()
        self._earliest = QLineEdit(str(anchor_row.get("earliest") or ""))
        self._earliest.setPlaceholderText("YYYY-MM-DD")
        self._latest = QLineEdit(str(anchor_row.get("latest") or ""))
        self._latest.setPlaceholderText("YYYY-MM-DD")
        self._note = QLineEdit("")
        self._note.setPlaceholderText("Optional reviewer note")
        form.addRow("Earliest", self._earliest)
        form.addRow("Latest", self._latest)
        form.addRow("Note", self._note)
        outer.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def values(self) -> dict[str, str]:
        earliest = str(self._earliest.text() or "").strip()
        latest = str(self._latest.text() or "").strip()
        return {
            "earliest": earliest,
            "latest": latest or earliest,
            "note": str(self._note.text() or "").strip(),
        }


class _EntityMergeDialog(QDialog):
    """Phase 3 entity review: pick other entities to merge into (or keep separate from) a primary entity."""

    def __init__(
        self,
        *,
        primary: dict[str, Any],
        candidates: list[dict[str, Any]],
        reject: bool,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Reject Entity Merge" if reject else "Merge Entities")
        self.resize(480, 420)
        outer = QVBoxLayout(self)
        verb = "Keep separate from" if reject else "Merge into"
        caption = QLabel(f"{verb}: {str(primary.get('canonical_name') or primary.get('entity_id') or 'entity')}")
        caption.setStyleSheet("font-weight: 600;")
        outer.addWidget(caption)
        outer.addWidget(QLabel("Select the other entities:"))
        self._list = QListWidget(self)
        self._list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        for cand in candidates:
            label = f"{str(cand.get('canonical_name') or '')}  [{str(cand.get('entity_type') or 'unknown')}]"
            item = QListWidgetItem(label.strip())
            item.setData(Qt.ItemDataRole.UserRole, str(cand.get("entity_id") or ""))
            self._list.addItem(item)
        outer.addWidget(self._list, stretch=1)
        self._canonical = QLineEdit("")
        if not reject:
            form = QFormLayout()
            self._canonical.setPlaceholderText("Optional merged canonical name")
            form.addRow("Canonical name", self._canonical)
            outer.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def other_entity_ids(self) -> list[str]:
        out: list[str] = []
        for item in self._list.selectedItems():
            entity_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
            if entity_id:
                out.append(entity_id)
        return out

    def canonical_name(self) -> str:
        return str(self._canonical.text() or "").strip()


class ArchiveVizPanel(QWidget):
    """Native archive workbench for CV Ops, centered on Phase 0 assembly review."""

    def __init__(
        self,
        *,
        http_get: Optional[Callable[[str], dict[str, Any]]] = None,
        http_post: Optional[Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]] = None,
        get_import_progress: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._http_get = http_get
        self._http_post = http_post
        self._get_import_progress = get_import_progress
        self._scenario_name = ""
        self._current_corpus_id = ""
        self._current_dataset_version_id = ""
        self._current_snapshot_id = ""
        self._pending_corpus_id = ""
        self._pending_dataset_version_id = ""
        self._pending_snapshot_id = ""
        self._current_snapshot_phase = ""
        self._poll_timer: Optional[QTimer] = None
        self._active_correlation_id: str = ""
        self._poll_line_cursor: int = 0
        self._job_poll_timer: Optional[QTimer] = None
        self._pending_job_id = ""
        self._phase0_payload: dict[str, Any] = {}
        self._phase2_payload: dict[str, Any] = {}
        self._phase3_payload: dict[str, Any] = {}
        self._phase4_payload: dict[str, Any] = {}
        self._phase5_payload: dict[str, Any] = {}
        self._phase5_parent_snapshot_id = ""
        self._timeline_payload: dict[str, Any] = {}
        self._version_payload: dict[str, Any] = {}
        self._object_detail: dict[str, Any] = {}
        self._current_object_id = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)
        self._import_folder_btn = QPushButton("Import Folder…")
        self._import_folder_btn.clicked.connect(self._import_folder)
        top.addWidget(self._import_folder_btn)
        self._import_files_btn = QPushButton("Import Files…")
        self._import_files_btn.clicked.connect(self._import_files)
        top.addWidget(self._import_files_btn)
        top.addSpacing(10)
        self._phase_combo = QComboBox()
        self._phase_combo.addItem("Pipeline (0-5)", "archive_pipeline")
        for idx in range(6):
            self._phase_combo.addItem(f"Phase {idx}", f"archive_phase{idx}")
        self._phase_combo.addItem("Reconcile", "archive_reconcile")
        top.addWidget(self._phase_combo)
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(self._run_selected_phase)
        top.addWidget(self._run_btn)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.clicked.connect(self.refresh)
        top.addWidget(self._refresh_btn)
        top.addStretch(1)
        self._status = QLabel("Archival mode")
        self._status.setStyleSheet("font-size: 10px; color: rgba(147,161,161,0.85);")
        top.addWidget(self._status)
        outer.addLayout(top)

        self._progress_frame = QFrame()
        self._progress_frame.setFrameShape(QFrame.Shape.NoFrame)
        self._progress_frame.setStyleSheet(
            "QFrame { background: #050d14; border: 1px solid rgba(64,120,160,0.4); border-radius: 3px; }"
        )
        prog_layout = QVBoxLayout(self._progress_frame)
        prog_layout.setContentsMargins(0, 0, 0, 0)
        prog_layout.setSpacing(0)
        self._progress_headline = QLabel("Import Terminal")
        self._progress_headline.setStyleSheet(
            "font-family: 'JetBrains Mono'; font-size: 9px; font-weight: bold; color: rgba(100,210,255,0.9);"
            "padding: 6px 8px; background: rgba(20,35,50,0.95);"
        )
        prog_layout.addWidget(self._progress_headline)
        self._progress_log = QPlainTextEdit()
        self._progress_log.setReadOnly(True)
        self._progress_log.setMinimumHeight(180)
        self._progress_log.setStyleSheet(
            "QPlainTextEdit { font-family: 'JetBrains Mono', monospace; font-size: 9px; color: #9dd49d; background: #050d14; border: none; padding: 6px 8px; }"
        )
        prog_layout.addWidget(self._progress_log, stretch=1)
        self._progress_frame.setVisible(False)
        outer.addWidget(self._progress_frame)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(2)
        split.addWidget(self._build_left_rail())
        split.addWidget(self._build_center_workspace())
        split.addWidget(self._build_right_inspector())
        split.setSizes([280, 740, 420])

        # Host the assembly workbench and a Semantic Carve surface as tabs so the
        # same CLIP carve (folder of images -> ImageFolder dataset) is reachable
        # from the archive page, not only Collect & Edit.
        self._page_tabs = QTabWidget()
        workbench = QWidget()
        wb_layout = QVBoxLayout(workbench)
        wb_layout.setContentsMargins(0, 0, 0, 0)
        wb_layout.addWidget(split)
        self._page_tabs.addTab(workbench, "Workbench")

        self._carve_panel = None
        if self._http_get is not None and self._http_post is not None:
            from .semantic_carve_panel import SemanticCarvePanel

            self._carve_panel = SemanticCarvePanel(
                http_get=self._http_get, http_post=self._http_post
            )
            self._page_tabs.addTab(self._carve_panel, "Semantic Carve")
        outer.addWidget(self._page_tabs, stretch=1)

        self.refresh()

    def _build_left_rail(self) -> QWidget:
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(self._section_label("Corpus"))
        self._corpus_combo = QComboBox()
        self._corpus_combo.currentIndexChanged.connect(self._on_corpus_changed)
        layout.addWidget(self._corpus_combo)

        layout.addWidget(self._section_label("Dataset Version"))
        self._version_combo = QComboBox()
        self._version_combo.currentIndexChanged.connect(self._on_version_changed)
        layout.addWidget(self._version_combo)

        layout.addWidget(self._section_label("Snapshot"))
        self._snapshot_combo = QComboBox()
        self._snapshot_combo.currentIndexChanged.connect(self._on_snapshot_changed)
        layout.addWidget(self._snapshot_combo)

        layout.addWidget(self._section_label("Filters"))
        self._query = QLineEdit()
        self._query.setPlaceholderText("Search title/entity/method/type/era")
        self._query.returnPressed.connect(self._reload_current_payloads)
        layout.addWidget(self._query)
        filter_grid = QGridLayout()
        filter_grid.setContentsMargins(0, 0, 0, 0)
        filter_grid.setHorizontalSpacing(6)
        filter_grid.setVerticalSpacing(4)
        self._unresolved_only = QCheckBox("Unresolved")
        self._unresolved_only.toggled.connect(self._reload_timeline)
        filter_grid.addWidget(self._unresolved_only, 0, 0)
        self._missing_only = QCheckBox("Missing Files")
        self._missing_only.toggled.connect(self._rerender_all)
        filter_grid.addWidget(self._missing_only, 0, 1)
        self._edited_only = QCheckBox("Edited Facts")
        self._edited_only.toggled.connect(self._rerender_all)
        filter_grid.addWidget(self._edited_only, 1, 0)
        self._media_combo = QComboBox()
        self._media_combo.addItem("All Media", "all")
        self._media_combo.addItem("Image", "image")
        self._media_combo.addItem("Document", "document")
        self._media_combo.addItem("Audio", "audio")
        self._media_combo.currentIndexChanged.connect(self._rerender_all)
        filter_grid.addWidget(self._media_combo, 1, 1)
        self._object_type_combo = QComboBox()
        self._object_type_combo.addItem("All Types", "all")
        for value in (
            "photograph",
            "newspaper_page",
            "newspaper_article",
            "map_sheet",
            "document",
            "correspondence",
            "audio_recording",
            "unknown",
        ):
            self._object_type_combo.addItem(value.replace("_", " ").title(), value)
        self._object_type_combo.currentIndexChanged.connect(self._rerender_all)
        filter_grid.addWidget(self._object_type_combo, 2, 0)
        self._era_combo = QComboBox()
        self._era_combo.addItem("All Eras", "all")
        for value in ("pre-1900", "1900-1920", "1920-1940", "1940-1960", "1960-1980", "1980-present", ""):
            label = "Unspecified Era" if not value else value
            self._era_combo.addItem(label, value if value else "__empty__")
        self._era_combo.currentIndexChanged.connect(self._rerender_all)
        filter_grid.addWidget(self._era_combo, 2, 1)
        self._complexity_combo = QComboBox()
        self._complexity_combo.addItem("All Complexity", "all")
        self._complexity_combo.addItem("Single", "single")
        self._complexity_combo.addItem("Multi", "multi")
        self._complexity_combo.addItem("Unknown", "unknown")
        self._complexity_combo.currentIndexChanged.connect(self._rerender_all)
        filter_grid.addWidget(self._complexity_combo, 3, 0, 1, 2)
        layout.addLayout(filter_grid)

        layout.addWidget(self._section_label("Current Version"))
        self._summary = QLabel("No archive loaded")
        self._summary.setWordWrap(True)
        self._summary.setStyleSheet("font-size: 10px; color: rgba(168,180,172,0.95);")
        layout.addWidget(self._summary)

        layout.addWidget(self._section_label("Corpus Health"))
        self._health_view = QTextBrowser()
        self._health_view.setMinimumHeight(130)
        self._health_view.setOpenExternalLinks(False)
        layout.addWidget(self._health_view)

        layout.addWidget(self._section_label("Review Buckets"))
        self._review_view = QTextBrowser()
        self._review_view.setMinimumHeight(120)
        self._review_view.setOpenExternalLinks(False)
        layout.addWidget(self._review_view)

        layout.addWidget(self._section_label("Phase Summary"))
        self._classification_view = QTextBrowser()
        self._classification_view.setMinimumHeight(150)
        self._classification_view.setOpenExternalLinks(False)
        layout.addWidget(self._classification_view)

        layout.addWidget(self._section_label("Full Phase Line"))
        self._phase_goals_view = QPlainTextEdit()
        self._phase_goals_view.setReadOnly(True)
        self._phase_goals_view.setMinimumHeight(220)
        layout.addWidget(self._phase_goals_view, stretch=1)
        return left

    def _build_center_workspace(self) -> QWidget:
        center = QWidget()
        layout = QVBoxLayout(center)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(self._section_label("Workspace"))
        self._view_combo = QComboBox()
        self._view_combo.addItem("Assembly Review", "assembly")
        self._view_combo.addItem("Classification", "classification")
        self._view_combo.addItem("Extraction", "extraction")
        self._view_combo.addItem("Structure Review", "structure")
        self._view_combo.addItem("Visual", "visual")
        self._view_combo.addItem("Cross-Reference", "xref")
        self._view_combo.addItem("Timeline", "timeline")
        self._view_combo.currentIndexChanged.connect(self._on_view_changed)
        controls.addWidget(self._view_combo)
        controls.addStretch(1)

        self._merge_btn = QPushButton("Merge Selected")
        self._merge_btn.clicked.connect(self._merge_selected_objects)
        controls.addWidget(self._merge_btn)
        self._split_btn = QPushButton("Split Object")
        self._split_btn.clicked.connect(self._split_selected_object)
        controls.addWidget(self._split_btn)
        self._roles_btn = QPushButton("Set Roles")
        self._roles_btn.clicked.connect(self._edit_selected_roles)
        controls.addWidget(self._roles_btn)

        self._pin_date_btn = QPushButton("Pin / Resolve Date")
        self._pin_date_btn.clicked.connect(self._pin_selected_anchor)
        controls.addWidget(self._pin_date_btn)
        self._merge_entity_btn = QPushButton("Merge Entities")
        self._merge_entity_btn.clicked.connect(self._merge_selected_entity)
        controls.addWidget(self._merge_entity_btn)
        self._reject_merge_btn = QPushButton("Reject Merge")
        self._reject_merge_btn.clicked.connect(self._reject_selected_entity)
        controls.addWidget(self._reject_merge_btn)
        layout.addLayout(controls)

        self._center_stack = QStackedWidget()
        layout.addWidget(self._center_stack, stretch=1)

        assembly_page = QWidget()
        assembly_layout = QVBoxLayout(assembly_page)
        assembly_layout.setContentsMargins(0, 0, 0, 0)
        assembly_layout.setSpacing(6)
        self._assembly_status = QLabel("Phase 0 assembly review is ready.")
        self._assembly_status.setStyleSheet("font-size: 10px; color: rgba(168,180,172,0.95);")
        assembly_layout.addWidget(self._assembly_status)
        self._assembly_tree = QTreeWidget()
        self._assembly_tree.setHeaderLabels(["Title", "Method", "Review", "Files", "Missing"])
        self._assembly_tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self._assembly_tree.itemSelectionChanged.connect(self._on_assembly_selection_changed)
        self._assembly_tree.itemDoubleClicked.connect(self._open_object_from_tree)
        assembly_layout.addWidget(self._assembly_tree, stretch=1)
        self._center_stack.addWidget(assembly_page)

        classification_page = QWidget()
        classification_layout = QVBoxLayout(classification_page)
        classification_layout.setContentsMargins(0, 0, 0, 0)
        classification_layout.setSpacing(6)
        self._classification_status = QLabel("Phase 1 classification will appear here once a Phase 1 snapshot exists.")
        self._classification_status.setStyleSheet("font-size: 10px; color: rgba(168,180,172,0.95);")
        classification_layout.addWidget(self._classification_status)
        self._classification_tree = QTreeWidget()
        self._classification_tree.setHeaderLabels(["Title", "Type", "Era", "Complexity", "Routes"])
        self._classification_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._classification_tree.itemSelectionChanged.connect(self._on_classification_selection_changed)
        self._classification_tree.itemDoubleClicked.connect(self._open_object_from_tree)
        classification_layout.addWidget(self._classification_tree, stretch=1)
        self._center_stack.addWidget(classification_page)

        extraction_page = QWidget()
        extraction_layout = QVBoxLayout(extraction_page)
        extraction_layout.setContentsMargins(0, 0, 0, 0)
        extraction_layout.setSpacing(6)
        self._extraction_status = QLabel("Phase 2 extraction review appears here once a Phase 2 snapshot exists.")
        self._extraction_status.setStyleSheet("font-size: 10px; color: rgba(168,180,172,0.95);")
        extraction_layout.addWidget(self._extraction_status)
        self._extraction_tree = QTreeWidget()
        self._extraction_tree.setHeaderLabels(["Title", "Status", "Providers", "Blocks", "Signal"])
        self._extraction_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._extraction_tree.itemSelectionChanged.connect(self._on_extraction_selection_changed)
        self._extraction_tree.itemDoubleClicked.connect(self._open_object_from_tree)
        extraction_layout.addWidget(self._extraction_tree, stretch=1)
        self._center_stack.addWidget(extraction_page)

        structure_page = QWidget()
        structure_layout = QVBoxLayout(structure_page)
        structure_layout.setContentsMargins(0, 0, 0, 0)
        structure_layout.setSpacing(6)
        self._structure_status = QLabel("Phase 3 structured extraction review appears here once a Phase 3 snapshot exists.")
        self._structure_status.setStyleSheet("font-size: 10px; color: rgba(168,180,172,0.95);")
        structure_layout.addWidget(self._structure_status)
        self._structure_tree = QTreeWidget()
        self._structure_tree.setHeaderLabels(["Item", "Kind", "Object", "Confidence", "State"])
        self._structure_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._structure_tree.itemSelectionChanged.connect(self._on_structure_selection_changed)
        self._structure_tree.itemDoubleClicked.connect(self._open_object_from_tree)
        structure_layout.addWidget(self._structure_tree, stretch=1)
        self._center_stack.addWidget(structure_page)

        timeline_page = QWidget()
        timeline_layout = QVBoxLayout(timeline_page)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(6)
        self._timeline_status = QLabel("Timeline / holding pen")
        self._timeline_status.setStyleSheet("font-size: 10px; color: rgba(168,180,172,0.95);")
        timeline_layout.addWidget(self._timeline_status)
        self._timeline_tree = QTreeWidget()
        self._timeline_tree.setHeaderLabels(["When", "Title", "Type", "Flags"])
        self._timeline_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._timeline_tree.itemSelectionChanged.connect(self._on_timeline_selection_changed)
        self._timeline_tree.itemDoubleClicked.connect(self._open_object_from_tree)
        timeline_layout.addWidget(self._timeline_tree, stretch=1)
        self._density_label = QLabel("")
        self._density_label.setWordWrap(True)
        self._density_label.setStyleSheet("font-size: 10px; color: rgba(147,161,161,0.82);")
        timeline_layout.addWidget(self._density_label)
        self._center_stack.addWidget(timeline_page)

        visual_page = QWidget()
        visual_layout = QVBoxLayout(visual_page)
        visual_layout.setContentsMargins(0, 0, 0, 0)
        visual_layout.setSpacing(6)
        self._visual_status = QLabel("Phase 4 visual review appears here once a Phase 4 snapshot exists.")
        self._visual_status.setStyleSheet("font-size: 10px; color: rgba(168,180,172,0.95);")
        visual_layout.addWidget(self._visual_status)
        self._visual_tree = QTreeWidget()
        self._visual_tree.setHeaderLabels(["Item", "Scene", "Era", "Mean Luma", "Detail"])
        self._visual_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._visual_tree.itemSelectionChanged.connect(self._on_visual_selection_changed)
        self._visual_tree.itemDoubleClicked.connect(self._open_object_from_tree)
        visual_layout.addWidget(self._visual_tree, stretch=1)
        self._center_stack.addWidget(visual_page)

        xref_page = QWidget()
        xref_layout = QVBoxLayout(xref_page)
        xref_layout.setContentsMargins(0, 0, 0, 0)
        xref_layout.setSpacing(6)
        self._xref_status = QLabel("Phase 5 cross-reference proposals appear here once a Phase 5 snapshot exists.")
        self._xref_status.setStyleSheet("font-size: 10px; color: rgba(168,180,172,0.95);")
        xref_layout.addWidget(self._xref_status)
        xref_controls = QHBoxLayout()
        xref_controls.setSpacing(6)
        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.clicked.connect(lambda: self._decide_selected_proposal("confirm"))
        xref_controls.addWidget(self._confirm_btn)
        self._reject_btn = QPushButton("Reject")
        self._reject_btn.clicked.connect(lambda: self._decide_selected_proposal("reject"))
        xref_controls.addWidget(self._reject_btn)
        self._defer_btn = QPushButton("Defer")
        self._defer_btn.clicked.connect(lambda: self._decide_selected_proposal("defer"))
        xref_controls.addWidget(self._defer_btn)
        xref_controls.addStretch(1)
        self._apply_decisions_btn = QPushButton("Apply Decisions (Rerun Phase 5)")
        self._apply_decisions_btn.clicked.connect(self._rerun_phase5_apply)
        xref_controls.addWidget(self._apply_decisions_btn)
        xref_layout.addLayout(xref_controls)
        self._xref_tree = QTreeWidget()
        self._xref_tree.setHeaderLabels(["Item", "Type", "Confidence", "Status", "Detail"])
        self._xref_tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self._xref_tree.itemSelectionChanged.connect(self._on_xref_selection_changed)
        self._xref_tree.itemDoubleClicked.connect(self._open_object_from_tree)
        xref_layout.addWidget(self._xref_tree, stretch=1)
        self._center_stack.addWidget(xref_page)
        return center

    def _build_right_inspector(self) -> QWidget:
        right = QWidget()
        layout = QVBoxLayout(right)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(self._section_label("Object Inspector"))
        self._object_head = QLabel("Select an object")
        self._object_head.setWordWrap(True)
        self._object_head.setStyleSheet("font-size: 13px; font-weight: 600;")
        layout.addWidget(self._object_head)
        self._object_meta = QLabel("")
        self._object_meta.setWordWrap(True)
        self._object_meta.setStyleSheet("font-size: 10px; color: rgba(147,161,161,0.85);")
        layout.addWidget(self._object_meta)

        layout.addWidget(self._section_label("Preview"))
        self._preview_caption = QLabel("No preview loaded")
        self._preview_caption.setWordWrap(True)
        self._preview_caption.setStyleSheet("font-size: 10px; color: rgba(147,161,161,0.85);")
        layout.addWidget(self._preview_caption)
        self._preview_stack = QStackedWidget()
        self._preview_info = QLabel("Select an object to inspect previewable source material.")
        self._preview_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_info.setWordWrap(True)
        self._preview_info.setMinimumHeight(160)
        self._preview_stack.addWidget(self._preview_info)
        self._preview_image = QLabel("")
        self._preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview_image.setMinimumHeight(160)
        self._preview_image.setStyleSheet("background: rgba(7,14,18,0.75); border: 1px solid rgba(64,120,160,0.25);")
        self._preview_stack.addWidget(self._preview_image)
        self._preview_text = QTextBrowser()
        self._preview_text.setMinimumHeight(160)
        self._preview_stack.addWidget(self._preview_text)
        layout.addWidget(self._preview_stack)

        self._inspector_tabs = QTabWidget()
        layout.addWidget(self._inspector_tabs, stretch=1)

        self._provenance_view = QTextBrowser()
        self._inspector_tabs.addTab(self._provenance_view, "Provenance")

        self._files_tree = QTreeWidget()
        self._files_tree.setHeaderLabels(["Role", "Status", "Current Path", "Original Path", "Size"])
        self._inspector_tabs.addTab(self._files_tree, "Files")

        self._assertions_tree = QTreeWidget()
        self._assertions_tree.setHeaderLabels(["Field", "Current", "Confidence", "Source"])
        self._inspector_tabs.addTab(self._assertions_tree, "Assertions")

        self._structure_detail_tree = QTreeWidget()
        self._structure_detail_tree.setHeaderLabels(["Kind", "Value", "Confidence", "State"])
        self._inspector_tabs.addTab(self._structure_detail_tree, "Structure")

        self._related_tree = QTreeWidget()
        self._related_tree.setHeaderLabels(["Relation", "Object", "When / Meta"])
        self._inspector_tabs.addTab(self._related_tree, "Related")
        return right

    def _section_label(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-family: 'JetBrains Mono'; font-size: 10px; color: rgba(147,161,161,0.82);")
        return label

    def set_archive_context(
        self,
        corpus_id: str = "",
        dataset_version_id: str = "",
        snapshot_id: str = "",
        scenario: str = "",
    ) -> None:
        self._scenario_name = str(scenario or "").strip()
        self._pending_corpus_id = str(corpus_id or "").strip()
        self._pending_dataset_version_id = str(dataset_version_id or "").strip()
        self._pending_snapshot_id = str(snapshot_id or "").strip()
        self.refresh()

    def clear(self) -> None:
        self._scenario_name = ""
        self._current_corpus_id = ""
        self._current_dataset_version_id = ""
        self._current_snapshot_id = ""
        self._pending_corpus_id = ""
        self._pending_dataset_version_id = ""
        self._pending_snapshot_id = ""
        self._current_snapshot_phase = ""
        self._phase0_payload = {}
        self._phase2_payload = {}
        self._phase3_payload = {}
        self._timeline_payload = {}
        self._version_payload = {}
        self._object_detail = {}
        self._current_object_id = ""
        self._corpus_combo.clear()
        self._version_combo.clear()
        self._snapshot_combo.clear()
        self._assembly_tree.clear()
        self._classification_tree.clear()
        self._extraction_tree.clear()
        self._structure_tree.clear()
        self._timeline_tree.clear()
        self._clear_detail()
        self._summary.setText("No archive loaded")
        self._health_view.setPlainText("")
        self._review_view.setPlainText("")
        self._classification_view.setPlainText("")
        self._phase_goals_view.setPlainText("")
        self._set_status("Archival mode")

    def refresh(self) -> None:
        if self._http_get is None:
            self._set_status("Archive API unavailable")
            return
        try:
            payload = self._http_get("/archives")
        except Exception as exc:
            self._set_status(f"Archive load failed: {exc}")
            return
        corpora = payload.get("corpora") if isinstance(payload, dict) else []
        if not isinstance(corpora, list):
            corpora = []
        self._corpus_combo.blockSignals(True)
        self._corpus_combo.clear()
        for item in corpora:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("slug") or "Corpus")
            slug = str(item.get("slug") or "")
            label = f"{name}  [{slug}]" if slug else name
            self._corpus_combo.addItem(label, str(item.get("corpus_id") or ""))
        self._corpus_combo.blockSignals(False)
        if not corpora:
            self.clear()
            return
        target = self._pending_corpus_id or self._current_corpus_id or str(corpora[0].get("corpus_id") or "")
        self._select_combo_data(self._corpus_combo, target)
        self._load_selected_corpus()

    def _set_status(self, text: str) -> None:
        self._status.setText(str(text or ""))

    def _import_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import Archival Folder", str(Path.home()))
        if folder:
            self._submit_import([folder])

    def _import_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Import Archival Files", str(Path.home()))
        if files:
            self._submit_import(files)

    def _submit_import(self, paths: list[str]) -> None:
        if self._http_post is None:
            self._set_status("Archive API unavailable")
            return
        self._import_folder_btn.setEnabled(False)
        self._import_files_btn.setEnabled(False)
        corr_id = uuid.uuid4().hex
        self._active_correlation_id = corr_id
        self._poll_line_cursor = 0
        self._progress_log.clear()
        self._progress_headline.setText("Import Terminal  —  waiting for server…")
        self._progress_frame.setVisible(True)
        self._set_status(f"Importing {len(paths)} path(s)…")
        self._start_progress_poll(corr_id)

        body = {
            "source_paths": [str(Path(p).expanduser()) for p in paths],
            "name": Path(paths[0]).name if paths else "Archive Corpus",
            "scenario": self._scenario_name,
            "correlation_id": corr_id,
        }
        http_post = self._http_post
        n_paths = len(paths)

        def _run() -> None:
            try:
                result = http_post("/archives/import", body)
                QTimer.singleShot(0, lambda: self._on_import_done(result, None, n_paths, corr_id))
            except Exception as exc:
                QTimer.singleShot(0, lambda: self._on_import_done(None, str(exc), n_paths, corr_id))

        threading.Thread(target=_run, daemon=True).start()

    def _start_progress_poll(self, corr_id: str) -> None:
        self._stop_progress_poll()
        timer = QTimer(self)
        timer.setInterval(600)
        timer.timeout.connect(lambda: self._do_progress_poll(corr_id))
        timer.start()
        self._poll_timer = timer

    def _stop_progress_poll(self) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None

    def _do_progress_poll(self, corr_id: str) -> None:
        if corr_id != self._active_correlation_id:
            self._stop_progress_poll()
            return
        if self._get_import_progress is not None:
            try:
                prog = self._get_import_progress(corr_id)
                if isinstance(prog, dict):
                    self._update_progress_display(prog)
            except Exception as exc:
                self._append_log_line(f"[POLL] Progress read error: {exc}")
            return
        if self._http_get is None:
            return
        http_get = self._http_get

        def _fetch() -> None:
            try:
                prog = http_get(f"/archives/import_progress/{corr_id}")
                QTimer.singleShot(0, lambda: self._update_progress_display(prog))
            except Exception as exc:
                QTimer.singleShot(0, lambda: self._append_log_line(f"[POLL] Server unreachable: {exc}"))

        threading.Thread(target=_fetch, daemon=True).start()

    def _append_log_line(self, line: str) -> None:
        lines = self._progress_log.toPlainText().splitlines()
        if lines and lines[-1] == line:
            return
        self._progress_log.appendPlainText(line)
        sb = self._progress_log.verticalScrollBar()
        if sb is not None:
            sb.setValue(sb.maximum())

    def _update_progress_display(self, prog: dict[str, Any]) -> None:
        phase = str(prog.get("phase") or "")
        current = int(prog.get("current") or 0)
        total = int(prog.get("total") or 0)
        elapsed = prog.get("elapsed_seconds")
        eta = prog.get("eta_seconds")
        tag_map = {
            "starting": "STARTING",
            "scanning": "SCANNING",
            "copying": "COPYING",
            "hashing": "HASHING",
            "indexing": "INDEXING",
            "done": "DONE",
            "complete": "COMPLETE",
            "error": "ERROR",
            "unknown": "WAITING",
        }
        tag = tag_map.get(phase, phase.upper())
        parts = [f"[{tag}]"]
        if total > 0:
            parts.append(f"{current}/{total}")
        if elapsed is not None:
            parts.append(f"{_fmt_duration(float(elapsed))} elapsed")
        if eta is not None and float(eta or 0.0) > 0 and phase not in {"done", "complete", "error"}:
            parts.append(f"ETA {_fmt_duration(float(eta))}")
        self._progress_headline.setText("Import Terminal  —  " + "  |  ".join(parts))
        lines = prog.get("lines")
        if isinstance(lines, list) and len(lines) > self._poll_line_cursor:
            new_lines = lines[self._poll_line_cursor :]
            self._poll_line_cursor = len(lines)
            for line in new_lines:
                self._progress_log.appendPlainText(str(line))
            sb = self._progress_log.verticalScrollBar()
            if sb is not None:
                sb.setValue(sb.maximum())

    def _on_import_done(self, payload: Optional[dict[str, Any]], error: Optional[str], n_paths: int, corr_id: str) -> None:
        del corr_id
        self._stop_progress_poll()
        self._active_correlation_id = ""
        self._import_folder_btn.setEnabled(True)
        self._import_files_btn.setEnabled(True)
        if error is not None:
            self._progress_headline.setText("Import Terminal  —  [FAILED]")
            self._append_log_line(f"[ERROR]  {error}")
            self._set_status(f"Import failed: {error}")
            return
        file_count = int((payload or {}).get("file_count") or 0)
        self._progress_headline.setText(f"Import Terminal  —  [COMPLETE]  {file_count} file(s)")
        self._pending_corpus_id = str(((payload or {}).get("corpus") or {}).get("corpus_id") or "")
        self._pending_dataset_version_id = str((payload or {}).get("dataset_version_id") or "")
        self._pending_snapshot_id = ""
        self._set_status(f"Imported {n_paths} path(s)")
        self.refresh()

    def _run_selected_phase(self) -> None:
        if self._http_post is None:
            self._set_status("Archive API unavailable")
            return
        if not self._current_corpus_id or not self._current_dataset_version_id:
            self._set_status("Select a corpus and dataset version first")
            return
        phase = str(self._phase_combo.currentData() or "archive_pipeline")
        try:
            payload = self._http_post(
                f"/archives/{self._current_corpus_id}/jobs",
                {
                    "dataset_version_id": self._current_dataset_version_id,
                    "phase": phase,
                    "parent_snapshot_id": self._current_snapshot_id,
                    "scenario": self._scenario_name,
                    "write_run_artifacts": True,
                },
            )
        except Exception as exc:
            self._set_status(f"Job submit failed: {exc}")
            return
        job_id = str((payload or {}).get("job_id") or "")
        self._set_status(f"Queued {phase} as {job_id}")
        self._start_job_poll(job_id)

    def _start_job_poll(self, job_id: str) -> None:
        self._pending_job_id = str(job_id or "").strip()
        if not self._pending_job_id:
            return
        if self._job_poll_timer is not None:
            self._job_poll_timer.stop()
        self._job_poll_timer = QTimer(self)
        self._job_poll_timer.setInterval(1500)
        self._job_poll_timer.timeout.connect(self._poll_job_state)
        self._job_poll_timer.start()

    def _poll_job_state(self) -> None:
        if not self._pending_job_id or self._http_get is None:
            if self._job_poll_timer is not None:
                self._job_poll_timer.stop()
            return
        try:
            payload = self._http_get(f"/archives/jobs/{self._pending_job_id}")
        except Exception:
            return
        state = str(payload.get("state") or "")
        if state not in {"done", "error"}:
            return
        if self._job_poll_timer is not None:
            self._job_poll_timer.stop()
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        if isinstance(result, dict):
            snap = str(result.get("snapshot_id") or "")
            if snap:
                self._pending_snapshot_id = snap
        self._pending_job_id = ""
        self.refresh()

    def _on_corpus_changed(self) -> None:
        self._load_selected_corpus()

    def _on_version_changed(self) -> None:
        self._load_selected_version()

    def _on_snapshot_changed(self) -> None:
        self._current_snapshot_id = str(self._snapshot_combo.currentData() or "")
        self._reload_current_payloads()

    def _load_selected_corpus(self) -> None:
        corpus_id = str(self._corpus_combo.currentData() or "")
        if not corpus_id or self._http_get is None:
            return
        self._current_corpus_id = corpus_id
        try:
            payload = self._http_get(f"/archives/{corpus_id}")
        except Exception as exc:
            self._set_status(f"Corpus load failed: {exc}")
            return
        versions = payload.get("versions") if isinstance(payload, dict) else []
        if not isinstance(versions, list):
            versions = []
        self._version_combo.blockSignals(True)
        self._version_combo.clear()
        for item in versions:
            if not isinstance(item, dict):
                continue
            label = (
                f"{str(item.get('label') or '')}  "
                f"({int(item.get('file_count') or 0):,} files, {int(item.get('snapshot_count') or 0)} snapshots)"
            )
            self._version_combo.addItem(label.strip(), str(item.get("dataset_version_id") or ""))
        self._version_combo.blockSignals(False)
        if versions:
            target = self._pending_dataset_version_id or self._current_dataset_version_id or str(versions[0].get("dataset_version_id") or "")
            self._select_combo_data(self._version_combo, target)
            self._load_selected_version()
        else:
            self._version_combo.clear()
            self._snapshot_combo.clear()
            self._phase0_payload = {}
            self._phase2_payload = {}
            self._phase3_payload = {}
            self._timeline_payload = {}
            self._clear_detail()
            self._summary.setText("Corpus imported but no dataset versions found")

    def _load_selected_version(self) -> None:
        dataset_version_id = str(self._version_combo.currentData() or "")
        if not dataset_version_id or self._http_get is None or not self._current_corpus_id:
            return
        self._current_dataset_version_id = dataset_version_id
        try:
            payload = self._http_get(f"/archives/{self._current_corpus_id}/versions/{dataset_version_id}")
        except Exception as exc:
            self._set_status(f"Version load failed: {exc}")
            return
        self._version_payload = dict(payload or {})
        snapshots = payload.get("snapshots") if isinstance(payload, dict) else []
        if not isinstance(snapshots, list):
            snapshots = []
        self._snapshot_combo.blockSignals(True)
        self._snapshot_combo.clear()
        for item in snapshots:
            if not isinstance(item, dict):
                continue
            label = f"{str(item.get('label') or item.get('phase') or 'Snapshot')}  [{str(item.get('phase') or '')}]"
            self._snapshot_combo.addItem(label.strip(), str(item.get("snapshot_id") or ""))
        self._snapshot_combo.blockSignals(False)
        self._summary.setText(
            f"{str(payload.get('label') or '')} | "
            f"{int(payload.get('file_count') or 0):,} files | "
            f"{int(payload.get('processable_count') or 0):,} processable | "
            f"{len(snapshots)} snapshots"
        )
        if snapshots:
            target = self._pending_snapshot_id or self._current_snapshot_id or str(snapshots[0].get("snapshot_id") or "")
            self._select_combo_data(self._snapshot_combo, target)
            self._current_snapshot_id = str(self._snapshot_combo.currentData() or "")
        else:
            self._current_snapshot_id = ""
        self._reload_current_payloads()

    def _reload_current_payloads(self) -> None:
        self._load_phase0_review()
        self._reload_timeline()
        self._reload_phase2_review()
        self._reload_phase3_review()
        self._reload_phase4_review()
        self._reload_phase5_review()

    def _reload_phase2_review(self) -> None:
        self._phase2_payload = {}
        self._extraction_tree.clear()
        if self._http_get is None or not self._current_snapshot_id:
            self._extraction_status.setText("No snapshot selected. Run Phase 2 to review extraction output.")
            return
        if self._current_snapshot_phase != "archive_phase2":
            self._extraction_status.setText("Extraction becomes primary at Phase 2. Select a Phase 2 snapshot to inspect it here.")
            return
        query = str(self._query.text() or "").strip()
        try:
            payload = self._http_get(f"/archives/snapshots/{self._current_snapshot_id}/phase2_review?q={query}")
        except Exception as exc:
            self._set_status(f"Phase 2 review failed: {exc}")
            return
        self._phase2_payload = dict(payload or {})
        self._render_phase2_review()
        self._render_left_metrics()

    def _reload_phase3_review(self) -> None:
        self._phase3_payload = {}
        self._structure_tree.clear()
        if self._http_get is None or not self._current_snapshot_id:
            self._structure_status.setText("No snapshot selected. Run Phase 3 to review structured extraction output.")
            return
        if self._current_snapshot_phase != "archive_phase3":
            self._structure_status.setText("Structure review becomes primary at Phase 3. Select a Phase 3 snapshot to inspect it here.")
            return
        query = str(self._query.text() or "").strip()
        try:
            payload = self._http_get(f"/archives/snapshots/{self._current_snapshot_id}/phase3_review?q={query}")
        except Exception as exc:
            self._set_status(f"Phase 3 review failed: {exc}")
            return
        self._phase3_payload = dict(payload or {})
        self._render_phase3_review()
        self._render_left_metrics()

    def _reload_phase4_review(self) -> None:
        self._phase4_payload = {}
        self._visual_tree.clear()
        if self._http_get is None or not self._current_snapshot_id:
            self._visual_status.setText("No snapshot selected. Run Phase 4 to review visual extraction.")
            return
        if self._current_snapshot_phase != "archive_phase4":
            self._visual_status.setText("Visual review becomes primary at Phase 4. Select a Phase 4 snapshot to inspect it here.")
            return
        query = str(self._query.text() or "").strip()
        try:
            payload = self._http_get(f"/archives/snapshots/{self._current_snapshot_id}/phase4_review?q={query}")
        except Exception as exc:
            self._set_status(f"Phase 4 review failed: {exc}")
            return
        self._phase4_payload = dict(payload or {})
        self._render_phase4_review()
        self._render_left_metrics()

    def _reload_phase5_review(self) -> None:
        self._phase5_payload = {}
        self._phase5_parent_snapshot_id = ""
        self._xref_tree.clear()
        if self._http_get is None or not self._current_snapshot_id:
            self._xref_status.setText("No snapshot selected. Run Phase 5 to review cross-reference proposals.")
            self._update_xref_actions()
            return
        if self._current_snapshot_phase != "archive_phase5":
            self._xref_status.setText("Cross-reference becomes primary at Phase 5. Select a Phase 5 snapshot to review proposals here.")
            self._update_xref_actions()
            return
        query = str(self._query.text() or "").strip()
        try:
            payload = self._http_get(f"/archives/snapshots/{self._current_snapshot_id}/phase5_review?q={query}")
        except Exception as exc:
            self._set_status(f"Phase 5 review failed: {exc}")
            return
        self._phase5_payload = dict(payload or {})
        snap = self._phase5_payload.get("snapshot") if isinstance(self._phase5_payload, dict) else {}
        self._phase5_parent_snapshot_id = str((snap or {}).get("parent_snapshot_id") or "")
        self._render_phase5_review()
        self._render_left_metrics()

    def _load_phase0_review(self) -> None:
        self._phase0_payload = {}
        self._assembly_tree.clear()
        if self._http_get is None or not self._current_corpus_id or not self._current_dataset_version_id:
            return
        query = str(self._query.text() or "").strip()
        try:
            payload = self._http_get(
                f"/archives/{self._current_corpus_id}/versions/{self._current_dataset_version_id}/phase0_review?q={query}"
            )
        except Exception as exc:
            self._set_status(f"Phase 0 review failed: {exc}")
            return
        self._phase0_payload = dict(payload or {})
        self._render_phase0_review()
        current_phase = self._current_snapshot_phase or "archive_phase0"
        self._phase_goals_view.setPlainText(_phase_goal_lines(self._phase0_payload, current_phase))
        self._render_left_metrics()

    def _reload_timeline(self) -> None:
        self._timeline_payload = {}
        self._classification_tree.clear()
        self._structure_tree.clear()
        self._visual_tree.clear()
        self._xref_tree.clear()
        self._timeline_tree.clear()
        self._density_label.setText("")
        if self._http_get is None or not self._current_snapshot_id:
            self._timeline_status.setText("No snapshot selected. Use Phase 0 review or run a phase job.")
            self._classification_status.setText("No snapshot selected. Run Phase 1 to review classification.")
            self._current_snapshot_phase = ""
            self._choose_default_center_view()
            self._render_left_metrics()
            return
        query = str(self._query.text() or "").strip()
        unresolved = 1 if self._unresolved_only.isChecked() else 0
        try:
            payload = self._http_get(
                f"/archives/snapshots/{self._current_snapshot_id}/timeline?q={query}&unresolved_only={unresolved}"
            )
        except Exception as exc:
            self._set_status(f"Timeline load failed: {exc}")
            return
        self._timeline_payload = dict(payload or {})
        self._current_snapshot_phase = str(self._timeline_payload.get("phase") or "")
        self._render_classification()
        self._render_timeline()
        self._choose_default_center_view()
        source_payload = self._timeline_payload if self._timeline_payload else self._phase0_payload
        self._phase_goals_view.setPlainText(_phase_goal_lines(source_payload, self._current_snapshot_phase))
        self._render_left_metrics()

    def _choose_default_center_view(self) -> None:
        if self._current_snapshot_phase in {"", "archive_phase0"}:
            want = "assembly"
        elif self._current_snapshot_phase == "archive_phase1":
            want = "classification"
        elif self._current_snapshot_phase == "archive_phase2":
            want = "extraction"
        elif self._current_snapshot_phase == "archive_phase3":
            want = "structure"
        elif self._current_snapshot_phase == "archive_phase4":
            want = "visual"
        elif self._current_snapshot_phase == "archive_phase5":
            want = "xref"
        else:
            want = "timeline"
        self._select_combo_data(self._view_combo, want)
        self._on_view_changed()

    def _on_view_changed(self) -> None:
        mode = str(self._view_combo.currentData() or "assembly")
        index_map = {"assembly": 0, "classification": 1, "extraction": 2, "structure": 3, "timeline": 4, "visual": 5, "xref": 6}
        self._center_stack.setCurrentIndex(index_map.get(mode, 0))
        phase0_active = mode == "assembly"
        self._merge_btn.setEnabled(phase0_active)
        self._split_btn.setEnabled(phase0_active)
        self._roles_btn.setEnabled(phase0_active)
        self._update_structure_actions()
        self._update_xref_actions()

    def _filtered_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        media = str(self._media_combo.currentData() or "all")
        object_type = str(self._object_type_combo.currentData() or "all")
        era_bucket = str(self._era_combo.currentData() or "all")
        complexity = str(self._complexity_combo.currentData() or "all")
        missing_only = self._missing_only.isChecked()
        edited_only = self._edited_only.isChecked()
        filtered: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            if media != "all" and str(row.get("media_family") or "") != media:
                continue
            row_type = str((row.get("classification") or {}).get("object_type") or row.get("object_type") or "unknown")
            row_era = str((row.get("classification") or {}).get("era_bucket") or row.get("era_bucket") or "")
            row_complexity = str(
                (row.get("classification") or {}).get("content_complexity") or row.get("content_complexity") or "unknown"
            )
            if object_type != "all" and row_type != object_type:
                continue
            if era_bucket != "all":
                if era_bucket == "__empty__":
                    if row_era:
                        continue
                elif row_era != era_bucket:
                    continue
            if complexity != "all" and row_complexity != complexity:
                continue
            if missing_only and int(row.get("missing_file_count") or 0) <= 0:
                continue
            if edited_only and int(row.get("edited_assertion_count") or 0) <= 0:
                continue
            filtered.append(row)
        return filtered

    def _render_left_metrics(self) -> None:
        phase0_summary = self._phase0_payload.get("summary") if isinstance(self._phase0_payload, dict) else {}
        if not isinstance(phase0_summary, dict):
            phase0_summary = {}
        timeline_summary = self._timeline_payload.get("summary") if isinstance(self._timeline_payload, dict) else {}
        if not isinstance(timeline_summary, dict):
            timeline_summary = {}
        health_lines = [
            f"<b>Objects</b>: {int(phase0_summary.get('object_count') or timeline_summary.get('object_count') or 0):,}",
            f"<b>Files</b>: {int(phase0_summary.get('file_count') or self._version_payload.get('file_count') or 0):,}",
            f"<b>Processable</b>: {int(phase0_summary.get('processable_count') or self._version_payload.get('processable_count') or 0):,}",
            f"<b>Missing Files</b>: {int(timeline_summary.get('missing_file_count') or phase0_summary.get('missing_file_count') or 0):,}",
            f"<b>Edited Facts</b>: {int(timeline_summary.get('edited_assertion_count') or 0):,}",
            f"<b>Anchors</b>: {int(timeline_summary.get('anchor_count') or 0):,}",
            f"<b>Entities</b>: {int(timeline_summary.get('entity_count') or 0):,}",
        ]
        self._health_view.setHtml("<br>".join(health_lines))

        review_lines = [
            f"<b>High confidence</b>: {int(phase0_summary.get('high_confidence_count') or 0):,}",
            f"<b>Needs review</b>: {int(phase0_summary.get('needs_review_count') or 0):,}",
            f"<b>Retained</b>: {int(phase0_summary.get('retained_count') or 0):,}",
            f"<b>Overrides</b>: {int(phase0_summary.get('override_count') or 0):,}",
            f"<b>On timeline</b>: {int(timeline_summary.get('timeline_count') or 0):,}",
            f"<b>Holding pen</b>: {int(timeline_summary.get('holding_pen_count') or 0):,}",
        ]
        if self._current_snapshot_phase == "archive_phase2" and isinstance(self._phase2_payload, dict):
            phase2_summary = self._phase2_payload.get("summary") or {}
            if isinstance(phase2_summary, dict):
                review_lines.extend(
                    [
                        f"<b>Extracted</b>: {int(phase2_summary.get('extracted_count') or 0):,}",
                        f"<b>Capability Unavailable</b>: {int(phase2_summary.get('capability_unavailable_count') or 0):,}",
                        f"<b>Failed</b>: {int(phase2_summary.get('failed_count') or 0):,}",
                        f"<b>Text Blocks</b>: {int(phase2_summary.get('text_block_count') or 0):,}",
                    ]
                )
        if self._current_snapshot_phase == "archive_phase3" and isinstance(self._phase3_payload, dict):
            phase3_summary = self._phase3_payload.get("summary") or {}
            if isinstance(phase3_summary, dict):
                review_lines.extend(
                    [
                        f"<b>Structured Objects</b>: {int(phase3_summary.get('structured_object_count') or 0):,}",
                        f"<b>Anchors</b>: {int(phase3_summary.get('anchor_count') or 0):,}",
                        f"<b>Unresolved Anchors</b>: {int(phase3_summary.get('unresolved_anchor_count') or 0):,}",
                        f"<b>Mentions</b>: {int(phase3_summary.get('mention_count') or 0):,}",
                        f"<b>Relationships</b>: {int(phase3_summary.get('relationship_count') or 0):,}",
                    ]
                )
        self._review_view.setHtml("<br>".join(review_lines))

        classification_summary = {}
        if self._current_snapshot_phase == "archive_phase2" and isinstance(self._phase2_payload, dict):
            classification_summary = self._phase2_payload.get("classification_summary") or {}
        elif self._current_snapshot_phase == "archive_phase3" and isinstance(self._phase3_payload, dict):
            classification_summary = self._phase3_payload.get("classification_summary") or {}
        elif self._current_snapshot_phase not in {"", "archive_phase0"} and isinstance(self._timeline_payload, dict):
            classification_summary = self._timeline_payload.get("classification_summary") or {}
        if not isinstance(classification_summary, dict) or not classification_summary:
            self._classification_view.setHtml("Run <b>Phase 1</b> to populate classification summaries and routing.")
            return
        classification_lines = [
            f"<b>Classified</b>: {int(classification_summary.get('classified_count') or 0):,}",
            f"<b>Types</b>: {_count_summary(classification_summary.get('object_types'), empty_label='unknown')}",
            f"<b>Eras</b>: {_count_summary(classification_summary.get('era_buckets'), empty_label='unspecified')}",
            f"<b>Complexity</b>: {_count_summary(classification_summary.get('content_complexities'), empty_label='unknown')}",
            f"<b>Routes</b>: {_count_summary(classification_summary.get('routes'), empty_label='not routed')}",
        ]
        if self._current_snapshot_phase == "archive_phase2":
            summary = self._phase2_payload.get("summary") if isinstance(self._phase2_payload, dict) else {}
            if isinstance(summary, dict) and summary:
                classification_lines.extend(
                    [
                        f"<b>Extracted</b>: {int(summary.get('extracted_count') or 0):,}",
                        f"<b>Audio</b>: {int(summary.get('audio_transcription_count') or 0):,} | "
                        f"<b>Printed OCR</b>: {int(summary.get('printed_ocr_count') or 0):,} | "
                        f"<b>Handwriting</b>: {int(summary.get('handwriting_ocr_count') or 0):,}",
                    ]
                )
        if self._current_snapshot_phase == "archive_phase3":
            summary = self._phase3_payload.get("summary") if isinstance(self._phase3_payload, dict) else {}
            if isinstance(summary, dict) and summary:
                classification_lines.extend(
                    [
                        f"<b>Anchors</b>: {int(summary.get('anchor_count') or 0):,} | "
                        f"<b>Resolved</b>: {int(summary.get('resolved_anchor_count') or 0):,} | "
                        f"<b>Open</b>: {int(summary.get('unresolved_anchor_count') or 0):,}",
                        f"<b>Entities</b>: {int(summary.get('entity_count') or 0):,} | "
                        f"<b>Mentions</b>: {int(summary.get('mention_count') or 0):,} | "
                        f"<b>Relations</b>: {int(summary.get('relationship_count') or 0):,}",
                    ]
                )
        self._classification_view.setHtml("<br>".join(classification_lines))

    def _render_phase0_review(self) -> None:
        self._assembly_tree.clear()
        payload = self._phase0_payload
        rows = self._filtered_rows(list(payload.get("objects") or []))
        if not rows:
            self._assembly_status.setText("No Phase 0 objects match the current filters.")
            return
        self._assembly_status.setText(
            f"{len(rows):,} assembly objects | "
            f"{int((payload.get('summary') or {}).get('needs_review_count') or 0):,} need review"
        )
        buckets: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in rows:
            bucket = str(row.get("review_bucket") or "needs_review")
            method = str(row.get("assembly_method") or "unknown")
            buckets.setdefault(bucket, {}).setdefault(method, []).append(row)
        for bucket, method_map in sorted(buckets.items(), key=lambda item: item[0]):
            bucket_item = QTreeWidgetItem(
                [bucket.replace("_", " ").title(), "", "", str(sum(len(v) for v in method_map.values())), ""]
            )
            bucket_item.setFlags(bucket_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._assembly_tree.addTopLevelItem(bucket_item)
            for method, items in sorted(method_map.items(), key=lambda item: item[0]):
                method_item = QTreeWidgetItem(
                    ["", method, "", str(sum(int(item.get("file_count") or 0) for item in items)), ""]
                )
                method_item.setFlags(method_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                bucket_item.addChild(method_item)
                for row in sorted(items, key=lambda item: str(item.get("title") or "").lower()):
                    missing = int(row.get("missing_file_count") or 0)
                    edited = int(row.get("edited_assertion_count") or 0)
                    title = f"{_badge_text(missing, edited)}  {str(row.get('title') or 'Object')}".strip()
                    item = QTreeWidgetItem(
                        [
                            title,
                            method,
                            bucket.replace("_", " "),
                            str(int(row.get("file_count") or 0)),
                            str(missing),
                        ]
                    )
                    item.setData(0, Qt.ItemDataRole.UserRole, str(row.get("object_id") or ""))
                    method_item.addChild(item)
            bucket_item.setExpanded(True)
        self._assembly_tree.expandToDepth(1)
        self._assembly_tree.resizeColumnToContents(0)

    def _render_classification(self) -> None:
        self._classification_tree.clear()
        payload = self._timeline_payload
        rows = self._filtered_rows(list(payload.get("items") or []) + list(payload.get("holding_pen") or []))
        if not rows:
            if self._current_snapshot_phase == "archive_phase1":
                self._classification_status.setText("No classified objects match the current filters.")
            else:
                self._classification_status.setText("Classification becomes primary at Phase 1. Select a Phase 1+ snapshot to inspect it here.")
            return
        summary = payload.get("classification_summary") if isinstance(payload, dict) else {}
        type_count = len((summary or {}).get("object_types") or []) if isinstance(summary, dict) else 0
        era_count = len((summary or {}).get("era_buckets") or []) if isinstance(summary, dict) else 0
        self._classification_status.setText(
            f"{len(rows):,} classified objects | {type_count} types | {era_count} era buckets"
        )
        grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in rows:
            classification = dict(row.get("classification") or {})
            object_type = str(classification.get("object_type") or row.get("object_type") or "unknown")
            era_bucket = str(classification.get("era_bucket") or row.get("era_bucket") or "") or "unspecified"
            grouped.setdefault(object_type, {}).setdefault(era_bucket, []).append(row)
        for object_type, era_map in sorted(grouped.items(), key=lambda item: item[0]):
            root = QTreeWidgetItem([object_type.replace("_", " ").title(), "", "", "", ""])
            root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._classification_tree.addTopLevelItem(root)
            for era_bucket, items in sorted(era_map.items(), key=lambda item: item[0]):
                group = QTreeWidgetItem(["", object_type.replace("_", " "), era_bucket, "", str(len(items))])
                group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                root.addChild(group)
                for row in sorted(items, key=lambda item: str(item.get("title") or "").lower()):
                    classification = dict(row.get("classification") or {})
                    routes = ", ".join(classification.get("routes") or row.get("routing_labels") or [])
                    missing = int(row.get("missing_file_count") or 0)
                    edited = int(row.get("edited_assertion_count") or 0)
                    title = f"{_badge_text(missing, edited)}  {str(row.get('title') or 'Object')}".strip()
                    child = QTreeWidgetItem(
                        [
                            title,
                            str(classification.get("object_type") or row.get("object_type") or ""),
                            str(classification.get("era_bucket") or row.get("era_bucket") or ""),
                            str(classification.get("content_complexity") or row.get("content_complexity") or ""),
                            routes,
                        ]
                    )
                    child.setData(0, Qt.ItemDataRole.UserRole, str(row.get("object_id") or ""))
                    group.addChild(child)
                group.setExpanded(True)
            root.setExpanded(True)
        self._classification_tree.expandToDepth(1)
        self._classification_tree.resizeColumnToContents(0)

    def _render_phase2_review(self) -> None:
        self._extraction_tree.clear()
        payload = self._phase2_payload
        rows = self._filtered_rows(list(payload.get("objects") or []))
        if not rows:
            self._extraction_status.setText("No extraction rows match the current filters.")
            return
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        self._extraction_status.setText(
            f"{len(rows):,} extraction objects | "
            f"{int((summary or {}).get('extracted_count') or 0):,} extracted | "
            f"{int((summary or {}).get('capability_unavailable_count') or 0):,} capability unavailable"
        )
        groups: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for row in rows:
            extraction_summary = dict(row.get("extraction_summary") or {})
            status = str(extraction_summary.get("status") or "extraction_ready")
            provider_summary = str(row.get("provider_summary") or "none")
            groups.setdefault(status, {}).setdefault(provider_summary, []).append(row)
        for status, provider_map in sorted(groups.items(), key=lambda item: item[0]):
            root = QTreeWidgetItem([status.replace("_", " ").title(), "", "", str(sum(len(items) for items in provider_map.values())), ""])
            root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._extraction_tree.addTopLevelItem(root)
            for provider_summary, items in sorted(provider_map.items(), key=lambda item: item[0]):
                branch = QTreeWidgetItem(["", status.replace("_", " "), provider_summary, str(sum(int(item.get("text_block_count") or 0) for item in items)), ""])
                branch.setFlags(branch.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                root.addChild(branch)
                for row in sorted(items, key=lambda item: str(item.get("title") or "").lower()):
                    missing = int(row.get("missing_file_count") or 0)
                    edited = int(row.get("edited_assertion_count") or 0)
                    title = f"{_badge_text(missing, edited)}  {str(row.get('title') or 'Object')}".strip()
                    signal_kind = "audio" if bool(row.get("has_audio")) else ("image" if str(row.get("media_family") or "") == "image" else "text")
                    child = QTreeWidgetItem(
                        [
                            title,
                            status.replace("_", " "),
                            provider_summary,
                            str(int(row.get("text_block_count") or 0)),
                            signal_kind,
                        ]
                    )
                    child.setData(0, Qt.ItemDataRole.UserRole, str(row.get("object_id") or ""))
                    branch.addChild(child)
                branch.setExpanded(True)
            root.setExpanded(True)
        self._extraction_tree.expandToDepth(1)
        self._extraction_tree.resizeColumnToContents(0)

    def _render_phase3_review(self) -> None:
        self._structure_tree.clear()
        payload = self._phase3_payload
        rows = self._filtered_rows(list(payload.get("objects") or []))
        if not rows:
            self._structure_status.setText("No structured extraction rows match the current filters.")
            return
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        self._structure_status.setText(
            f"{len(rows):,} structured objects | "
            f"{int((summary or {}).get('anchor_count') or 0):,} anchors | "
            f"{int((summary or {}).get('mention_count') or 0):,} mentions | "
            f"{int((summary or {}).get('relationship_count') or 0):,} relationships"
        )

        object_ids = {str(row.get("object_id") or "") for row in rows}
        anchors = [
            item
            for item in list(payload.get("anchors") or [])
            if isinstance(item, dict) and str(item.get("object_id") or "") in object_ids
        ]
        mentions = [
            item
            for item in list(payload.get("mentions") or [])
            if isinstance(item, dict) and str(item.get("object_id") or "") in object_ids
        ]
        relationships = [
            item
            for item in list(payload.get("relationships") or [])
            if isinstance(item, dict) and str(item.get("object_id") or "") in object_ids
        ]

        object_root = QTreeWidgetItem(["Objects", "object", "", "", str(len(rows))])
        object_root.setFlags(object_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._structure_tree.addTopLevelItem(object_root)
        for row in sorted(rows, key=lambda item: (str(item.get("review_state") or ""), str(item.get("title") or "").lower())):
            missing = int(row.get("missing_file_count") or 0)
            edited = int(row.get("edited_assertion_count") or 0)
            title = f"{_badge_text(missing, edited)}  {str(row.get('title') or 'Object')}".strip()
            child = QTreeWidgetItem(
                [
                    title,
                    str(row.get("object_type") or ""),
                    str(row.get("earliest") or row.get("era_bucket") or ""),
                    str(int(row.get("anchor_count") or 0)),
                    str(row.get("review_state") or ""),
                ]
            )
            child.setData(0, Qt.ItemDataRole.UserRole, str(row.get("object_id") or ""))
            child.setData(0, _ROLE_KIND, "object")
            child.setData(0, _ROLE_TARGET_ID, str(row.get("object_id") or ""))
            object_root.addChild(child)
        object_root.setExpanded(True)

        anchor_root = QTreeWidgetItem(["Temporal Anchors", "anchor", "", "", str(len(anchors))])
        anchor_root.setFlags(anchor_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._structure_tree.addTopLevelItem(anchor_root)
        anchors_by_state: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for anchor in anchors:
            state = str(anchor.get("review_state") or ("resolved" if bool(anchor.get("resolved")) else "needs_resolution"))
            anchor_type = str(anchor.get("type") or "unknown")
            anchors_by_state.setdefault(state, {}).setdefault(anchor_type, []).append(anchor)
        for state, type_map in sorted(anchors_by_state.items(), key=lambda item: item[0]):
            state_item = QTreeWidgetItem([state.replace("_", " ").title(), "anchor", "", "", str(sum(len(v) for v in type_map.values()))])
            state_item.setFlags(state_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            anchor_root.addChild(state_item)
            for anchor_type, items in sorted(type_map.items(), key=lambda item: item[0]):
                type_item = QTreeWidgetItem(["", anchor_type, "", "", str(len(items))])
                type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                state_item.addChild(type_item)
                for anchor in sorted(items, key=lambda item: (str(item.get("title") or "").lower(), str(item.get("raw_expression") or ""))):
                    label = str(anchor.get("raw_expression") or anchor.get("anchor_id") or "anchor")
                    when = str(anchor.get("earliest") or "")
                    latest = str(anchor.get("latest") or "")
                    if latest and latest != when:
                        when = f"{when} -> {latest}" if when else latest
                    child = QTreeWidgetItem(
                        [
                            label,
                            anchor_type,
                            str(anchor.get("title") or ""),
                            f"{float(anchor.get('confidence') or 0.0):.2f}",
                            when or str(anchor.get("resolution_requires") or state),
                        ]
                    )
                    child.setData(0, Qt.ItemDataRole.UserRole, str(anchor.get("object_id") or ""))
                    child.setData(0, _ROLE_KIND, "anchor")
                    child.setData(0, _ROLE_TARGET_ID, str(anchor.get("anchor_id") or ""))
                    type_item.addChild(child)
                type_item.setExpanded(True)
            state_item.setExpanded(True)
        anchor_root.setExpanded(True)

        mention_root = QTreeWidgetItem(["Entity Mentions", "mention", "", "", str(len(mentions))])
        mention_root.setFlags(mention_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._structure_tree.addTopLevelItem(mention_root)
        mentions_by_type: dict[str, list[dict[str, Any]]] = {}
        for mention in mentions:
            mentions_by_type.setdefault(str(mention.get("entity_type") or "unknown"), []).append(mention)
        for entity_type, items in sorted(mentions_by_type.items(), key=lambda item: item[0]):
            type_item = QTreeWidgetItem([entity_type.replace("_", " ").title(), "entity", "", "", str(len(items))])
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            mention_root.addChild(type_item)
            for mention in sorted(items, key=lambda item: (str(item.get("canonical_name") or "").lower(), str(item.get("title") or "").lower())):
                child = QTreeWidgetItem(
                    [
                        str(mention.get("canonical_name") or mention.get("mention_text") or ""),
                        entity_type,
                        str(mention.get("title") or ""),
                        f"{float(mention.get('mention_confidence') or 0.0):.2f}",
                        str(mention.get("mention_text") or ""),
                    ]
                )
                child.setData(0, Qt.ItemDataRole.UserRole, str(mention.get("object_id") or ""))
                child.setData(0, _ROLE_KIND, "entity")
                child.setData(0, _ROLE_TARGET_ID, str(mention.get("entity_id") or ""))
                type_item.addChild(child)
            type_item.setExpanded(True)
        mention_root.setExpanded(True)

        relation_root = QTreeWidgetItem(["Relationships", "relationship", "", "", str(len(relationships))])
        relation_root.setFlags(relation_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._structure_tree.addTopLevelItem(relation_root)
        relations_by_type: dict[str, list[dict[str, Any]]] = {}
        for rel in relationships:
            relations_by_type.setdefault(str(rel.get("relationship_type") or "unknown"), []).append(rel)
        for rel_type, items in sorted(relations_by_type.items(), key=lambda item: item[0]):
            type_item = QTreeWidgetItem([rel_type.replace("_", " ").title(), "relationship", "", "", str(len(items))])
            type_item.setFlags(type_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            relation_root.addChild(type_item)
            for rel in sorted(items, key=lambda item: str(item.get("title") or "").lower()):
                endpoint = str(rel.get("source_entity_name") or "")
                target = str(rel.get("target_entity_name") or "")
                if target:
                    endpoint = f"{endpoint} -> {target}" if endpoint else target
                child = QTreeWidgetItem(
                    [
                        endpoint or str(rel.get("relationship_id") or ""),
                        rel_type,
                        str(rel.get("title") or ""),
                        f"{float(rel.get('confidence') or 0.0):.2f}",
                        "",
                    ]
                )
                child.setData(0, Qt.ItemDataRole.UserRole, str(rel.get("object_id") or ""))
                child.setData(0, _ROLE_KIND, "relationship")
                child.setData(0, _ROLE_TARGET_ID, str(rel.get("relationship_id") or ""))
                type_item.addChild(child)
            type_item.setExpanded(True)
        relation_root.setExpanded(True)

        self._structure_tree.expandToDepth(1)
        self._structure_tree.resizeColumnToContents(0)

    def _render_phase4_review(self) -> None:
        self._visual_tree.clear()
        payload = self._phase4_payload
        objects = self._filtered_rows(list(payload.get("objects") or []))
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        if not objects:
            self._visual_status.setText("No visual extraction rows match the current filters.")
            return
        self._visual_status.setText(
            f"{len(objects):,} visual objects | "
            f"{int((summary or {}).get('scene_class_count') or 0):,} scene classes | "
            f"{int((summary or {}).get('visual_anchor_count') or 0):,} visual-era anchors | "
            f"{int((summary or {}).get('embedded_object_count') or 0):,} embedded"
        )
        visible_ids = {str(item.get("object_id") or "") for item in objects}

        scene_root = QTreeWidgetItem(["Scene Classes", "", "", "", ""])
        scene_root.setFlags(scene_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._visual_tree.addTopLevelItem(scene_root)
        scenes: dict[str, list[dict[str, Any]]] = {}
        for row in objects:
            scenes.setdefault(str(row.get("scene_class") or "unknown"), []).append(row)
        for scene, rows in sorted(scenes.items(), key=lambda item: item[0]):
            group = QTreeWidgetItem([scene.replace("_", " ").title(), scene, "", "", str(len(rows))])
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            scene_root.addChild(group)
            for row in sorted(rows, key=lambda item: str(item.get("title") or "").lower()):
                child = QTreeWidgetItem(
                    [
                        str(row.get("title") or "Object"),
                        scene,
                        str(row.get("era_bucket") or row.get("earliest") or ""),
                        f"{float(row.get('mean_luma') or 0.0):.1f}",
                        str(row.get("object_type") or ""),
                    ]
                )
                child.setData(0, Qt.ItemDataRole.UserRole, str(row.get("object_id") or ""))
                group.addChild(child)
            group.setExpanded(True)
        scene_root.setExpanded(True)

        era_cards = [card for card in (payload.get("era_cards") or []) if isinstance(card, dict)]
        if era_cards:
            era_root = QTreeWidgetItem(["Visual-Era Cards", "", "", "", str(len(era_cards))])
            era_root.setFlags(era_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._visual_tree.addTopLevelItem(era_root)
            for card in era_cards:
                scene_bits = ", ".join(
                    f"{str(item.get('scene_class') or '')}:{int(item.get('count') or 0)}"
                    for item in (card.get("scene_classes") or [])
                )
                era_item = QTreeWidgetItem(
                    [str(card.get("era_bucket") or "undated"), "", "", str(int(card.get("object_count") or 0)), scene_bits]
                )
                era_item.setFlags(era_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                era_root.addChild(era_item)
            era_root.setExpanded(True)

        anchors = [
            anchor
            for anchor in (payload.get("visual_anchors") or [])
            if isinstance(anchor, dict) and str(anchor.get("object_id") or "") in visible_ids
        ]
        if anchors:
            anchor_root = QTreeWidgetItem(["Visual-Estimate Anchors", "", "", "", str(len(anchors))])
            anchor_root.setFlags(anchor_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._visual_tree.addTopLevelItem(anchor_root)
            for anchor in anchors:
                when = str(anchor.get("earliest") or "")
                latest = str(anchor.get("latest") or "")
                if latest and latest != when:
                    when = f"{when} -> {latest}" if when else latest
                child = QTreeWidgetItem(
                    [
                        str(anchor.get("title") or "Object"),
                        str((anchor.get("metadata") or {}).get("scene_class") or "visual_estimate"),
                        when,
                        f"{float(anchor.get('confidence') or 0.0):.2f}",
                        str(anchor.get("raw_expression") or ""),
                    ]
                )
                child.setData(0, Qt.ItemDataRole.UserRole, str(anchor.get("object_id") or ""))
                anchor_root.addChild(child)
            anchor_root.setExpanded(True)

        sims = [
            sim
            for sim in (payload.get("similarities") or [])
            if isinstance(sim, dict) and str(sim.get("object_id") or "") in visible_ids
        ]
        if sims:
            sim_root = QTreeWidgetItem(["Visual Similarity", "", "", "", str(len(sims))])
            sim_root.setFlags(sim_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._visual_tree.addTopLevelItem(sim_root)
            for sim in sorted(sims, key=lambda item: str(item.get("title") or "").lower()):
                parent_item = QTreeWidgetItem(
                    [str(sim.get("title") or "Object"), "", "", "", str(len(sim.get("neighbors") or []))]
                )
                parent_item.setData(0, Qt.ItemDataRole.UserRole, str(sim.get("object_id") or ""))
                sim_root.addChild(parent_item)
                for neighbor in sim.get("neighbors") or []:
                    child = QTreeWidgetItem(
                        [
                            str(neighbor.get("title") or "Object"),
                            "similar",
                            "",
                            f"{float(neighbor.get('score') or 0.0):.3f}",
                            "",
                        ]
                    )
                    child.setData(0, Qt.ItemDataRole.UserRole, str(neighbor.get("object_id") or ""))
                    parent_item.addChild(child)
            sim_root.setExpanded(True)

        self._visual_tree.expandToDepth(1)
        self._visual_tree.resizeColumnToContents(0)

    def _on_visual_selection_changed(self) -> None:
        object_ids = self._selected_object_ids(self._visual_tree)
        if len(object_ids) == 1:
            self._load_object_detail(object_ids[0])
        elif not object_ids:
            self._clear_detail()

    # --- Phase 5 cross-reference review -----------------------------------

    @staticmethod
    def _proposal_label(proposal: dict[str, Any]) -> str:
        meta = dict(proposal.get("metadata") or {})
        pv = dict(proposal.get("proposed_value") or {})
        ptype = str(proposal.get("proposal_type") or "")
        if ptype == "entity_merge":
            return f"{meta.get('subject_name') or proposal.get('subject_id') or '?'}  ~  {meta.get('related_name') or proposal.get('related_id') or '?'}"
        if ptype in ("anchor_resolution", "temporal_propagation"):
            return f"{meta.get('raw_expression') or 'anchor'}  ->  {str(pv.get('earliest') or '')}..{str(pv.get('latest') or '')}"
        if ptype == "relationship":
            return f"{meta.get('source_name') or pv.get('source') or '?'}  <->  {meta.get('target_name') or pv.get('target') or '?'}"
        if ptype == "cluster_membership":
            return f"cluster: {pv.get('cluster_label') or meta.get('cluster_label') or ''}"
        return str(proposal.get("proposal_id") or "proposal")

    def _add_proposal_item(self, parent_item: QTreeWidgetItem, proposal: dict[str, Any]) -> None:
        item = QTreeWidgetItem(
            [
                self._proposal_label(proposal),
                str(proposal.get("proposal_type") or ""),
                f"{float(proposal.get('confidence') or 0.0):.2f}",
                str(proposal.get("status") or ""),
                str(proposal.get("review_bucket") or ""),
            ]
        )
        meta = dict(proposal.get("metadata") or {})
        item.setData(0, Qt.ItemDataRole.UserRole, str(meta.get("object_id") or ""))
        item.setData(0, _ROLE_KIND, "proposal")
        item.setData(0, _ROLE_TARGET_ID, str(proposal.get("proposal_id") or ""))
        parent_item.addChild(item)
        for ev in proposal.get("evidence") or []:
            if not isinstance(ev, dict):
                continue
            item.addChild(
                QTreeWidgetItem(
                    [
                        str(ev.get("description") or ""),
                        str(ev.get("evidence_type") or ""),
                        f"{float(ev.get('weight') or 0.0):.2f}",
                        "",
                        "evidence",
                    ]
                )
            )
        if str(proposal.get("cascade_source_proposal_id") or ""):
            item.addChild(QTreeWidgetItem([f"cascaded from {str(proposal.get('cascade_source_proposal_id'))}", "cascade", "", "", ""]))

    def _render_phase5_review(self) -> None:
        self._xref_tree.clear()
        payload = self._phase5_payload
        summary = payload.get("summary") if isinstance(payload, dict) else {}
        proposals = [p for p in (payload.get("proposals") or []) if isinstance(p, dict)]
        clusters = [c for c in (payload.get("clusters") or []) if isinstance(c, dict)]
        if not proposals and not clusters:
            self._xref_status.setText("No cross-reference proposals or clusters for this snapshot yet.")
            self._update_xref_actions()
            return
        s = summary or {}
        self._xref_status.setText(
            f"{int(s.get('proposal_count') or 0):,} proposals | "
            f"{int(s.get('open_count') or 0):,} open | "
            f"{int(s.get('confirmed_count') or 0):,} confirmed | "
            f"{int(s.get('rejected_count') or 0):,} rejected | "
            f"{int(s.get('auto_suppressed_count') or 0):,} auto-suppressed | "
            f"{int(s.get('cluster_count') or 0):,} clusters"
        )

        buckets = [b for b in (payload.get("review_buckets") or []) if isinstance(b, dict)]
        open_root = QTreeWidgetItem(["Review Buckets (open)", "", "", "", str(sum(int(b.get("count") or 0) for b in buckets))])
        open_root.setFlags(open_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._xref_tree.addTopLevelItem(open_root)
        for bucket in buckets:
            items = [p for p in (bucket.get("proposals") or []) if isinstance(p, dict)]
            bucket_item = QTreeWidgetItem([str(bucket.get("bucket") or "other").replace("_", " ").title(), "", "", "", str(len(items))])
            bucket_item.setFlags(bucket_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            open_root.addChild(bucket_item)
            for proposal in items:
                self._add_proposal_item(bucket_item, proposal)
            bucket_item.setExpanded(True)
        open_root.setExpanded(True)

        decided = [p for p in proposals if str(p.get("status") or "") != "proposed"]
        if decided:
            decided_root = QTreeWidgetItem(["Decided", "", "", "", str(len(decided))])
            decided_root.setFlags(decided_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._xref_tree.addTopLevelItem(decided_root)
            for proposal in sorted(decided, key=lambda item: (str(item.get("status") or ""), str(item.get("proposal_type") or ""))):
                self._add_proposal_item(decided_root, proposal)
            decided_root.setExpanded(False)

        if clusters:
            cluster_root = QTreeWidgetItem(["Semantic Clusters", "", "", "", str(len(clusters))])
            cluster_root.setFlags(cluster_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._xref_tree.addTopLevelItem(cluster_root)
            for cluster in clusters:
                cl_item = QTreeWidgetItem(
                    [str(cluster.get("label") or "cluster"), "cluster", "", "", f"{int(cluster.get('object_count') or 0)} objects"]
                )
                cl_item.setFlags(cl_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
                cluster_root.addChild(cl_item)
                for oid in cluster.get("object_ids") or []:
                    obj_item = QTreeWidgetItem([str(oid), "object", "", "", ""])
                    obj_item.setData(0, Qt.ItemDataRole.UserRole, str(oid))
                    obj_item.setData(0, _ROLE_KIND, "object")
                    cl_item.addChild(obj_item)
            cluster_root.setExpanded(False)

        self._xref_tree.expandToDepth(1)
        self._xref_tree.resizeColumnToContents(0)
        self._update_xref_actions()

    def _selected_xref_item(self) -> Optional[QTreeWidgetItem]:
        items = self._xref_tree.selectedItems()
        return items[0] if items else None

    def _on_xref_selection_changed(self) -> None:
        item = self._selected_xref_item()
        object_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "").strip() if item is not None else ""
        if object_id:
            self._load_object_detail(object_id)
        self._update_xref_actions()

    def _update_xref_actions(self) -> None:
        mode = str(self._view_combo.currentData() or "assembly")
        is_xref = mode == "xref"
        kind = ""
        status = ""
        if is_xref:
            item = self._selected_xref_item()
            if item is not None:
                kind = str(item.data(0, _ROLE_KIND) or "")
                status = str(item.text(3) or "")
        actionable = is_xref and kind == "proposal" and status == "proposed"
        self._confirm_btn.setEnabled(actionable)
        self._reject_btn.setEnabled(actionable)
        self._defer_btn.setEnabled(actionable)
        self._apply_decisions_btn.setEnabled(is_xref and bool(self._current_snapshot_id))

    def _decide_selected_proposal(self, decision: str) -> None:
        item = self._selected_xref_item()
        if item is None or str(item.data(0, _ROLE_KIND) or "") != "proposal" or self._http_post is None:
            self._set_status("Select an open proposal to decide")
            return
        proposal_id = str(item.data(0, _ROLE_TARGET_ID) or "").strip()
        if not proposal_id or not self._current_snapshot_id:
            return
        try:
            self._http_post(
                f"/archives/proposals/{proposal_id}/decision",
                {
                    "snapshot_id": self._current_snapshot_id,
                    "decision": str(decision),
                    "decided_by": "cvops-operator",
                    "reason": "",
                },
            )
        except Exception as exc:
            self._set_status(f"Decision failed: {exc}")
            return
        self._set_status(f"Proposal decision '{decision}' recorded. Use 'Apply Decisions' to re-run Phase 5 and materialize confirmed changes.")
        self._reload_phase5_review()

    def _rerun_phase5_apply(self) -> None:
        if self._http_post is None or not self._current_corpus_id or not self._current_dataset_version_id:
            return
        parent = self._phase5_parent_snapshot_id or self._current_snapshot_id
        if not parent:
            self._set_status("No parent snapshot available for Phase 5 rerun")
            return
        try:
            payload = self._http_post(
                f"/archives/{self._current_corpus_id}/jobs",
                {
                    "dataset_version_id": self._current_dataset_version_id,
                    "phase": "archive_phase5",
                    "parent_snapshot_id": parent,
                    "scenario": self._scenario_name,
                    "write_run_artifacts": True,
                },
            )
        except Exception as exc:
            self._set_status(f"Phase 5 rerun failed: {exc}")
            return
        self._set_status("Applying decisions: re-running Phase 5…")
        self._start_job_poll(str((payload or {}).get("job_id") or ""))

    def _render_timeline(self) -> None:
        self._timeline_tree.clear()
        payload = self._timeline_payload
        items = self._filtered_rows(list(payload.get("items") or []))
        holding = self._filtered_rows(list(payload.get("holding_pen") or []))
        self._timeline_status.setText(
            f"{len(items):,} on timeline | {len(holding):,} in holding pen | phase {self._current_snapshot_phase or 'n/a'}"
        )
        timeline_root = QTreeWidgetItem(["Timeline", "", "", ""])
        timeline_root.setFlags(timeline_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._timeline_tree.addTopLevelItem(timeline_root)
        decades: dict[str, list[dict[str, Any]]] = {}
        for item in items:
            year = str(item.get("earliest") or "")[:4]
            if year.isdigit():
                decade = f"{int(year) // 10 * 10}s"
            else:
                decade = "Unknown"
            decades.setdefault(decade, []).append(item)
        for decade, rows in sorted(decades.items(), key=lambda item: item[0]):
            group = QTreeWidgetItem([decade, "", "", str(len(rows))])
            group.setFlags(group.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            timeline_root.addChild(group)
            for row in sorted(rows, key=lambda item: (str(item.get("earliest") or ""), str(item.get("title") or "").lower())):
                flags = _badge_text(int(row.get("missing_file_count") or 0), int(row.get("edited_assertion_count") or 0))
                item = QTreeWidgetItem(
                    [
                        str(row.get("earliest") or "")[:10],
                        str(row.get("title") or ""),
                        str(row.get("object_type") or ""),
                        flags,
                    ]
                )
                item.setData(0, Qt.ItemDataRole.UserRole, str(row.get("object_id") or ""))
                group.addChild(item)
        timeline_root.setExpanded(True)

        holding_root = QTreeWidgetItem(["Holding Pen", "", "", str(len(holding))])
        holding_root.setFlags(holding_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._timeline_tree.addTopLevelItem(holding_root)
        for row in sorted(holding, key=lambda item: str(item.get("title") or "").lower()):
            flags = _badge_text(int(row.get("missing_file_count") or 0), int(row.get("edited_assertion_count") or 0))
            item = QTreeWidgetItem(
                [
                    "UNDATED",
                    str(row.get("title") or ""),
                    str(row.get("object_type") or ""),
                    flags,
                ]
            )
            item.setData(0, Qt.ItemDataRole.UserRole, str(row.get("object_id") or ""))
            holding_root.addChild(item)
        holding_root.setExpanded(True)

        density = payload.get("density") if isinstance(payload, dict) else []
        if isinstance(density, list) and density:
            parts = [f"{str(item.get('bucket') or '')}:{int(item.get('count') or 0)}" for item in density if isinstance(item, dict)]
            self._density_label.setText("Density  " + " | ".join(parts[:24]))
        else:
            self._density_label.setText("Density  no anchored timeline buckets yet.")
        self._timeline_tree.expandToDepth(1)
        self._timeline_tree.resizeColumnToContents(0)

    def _on_assembly_selection_changed(self) -> None:
        object_ids = self._selected_object_ids(self._assembly_tree)
        if len(object_ids) == 1:
            self._load_object_detail(object_ids[0])
        elif not object_ids:
            self._clear_detail()

    def _on_timeline_selection_changed(self) -> None:
        object_ids = self._selected_object_ids(self._timeline_tree)
        if len(object_ids) == 1:
            self._load_object_detail(object_ids[0])
        elif not object_ids:
            self._clear_detail()

    def _on_classification_selection_changed(self) -> None:
        object_ids = self._selected_object_ids(self._classification_tree)
        if len(object_ids) == 1:
            self._load_object_detail(object_ids[0])
        elif not object_ids:
            self._clear_detail()

    def _on_extraction_selection_changed(self) -> None:
        object_ids = self._selected_object_ids(self._extraction_tree)
        if len(object_ids) == 1:
            self._load_object_detail(object_ids[0])
        elif not object_ids:
            self._clear_detail()

    def _on_structure_selection_changed(self) -> None:
        object_ids = self._selected_object_ids(self._structure_tree)
        if len(object_ids) == 1:
            self._load_object_detail(object_ids[0])
        elif not object_ids:
            self._clear_detail()
        self._update_structure_actions()

    def _open_object_from_tree(self, item: QTreeWidgetItem, _column: int) -> None:
        object_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "")
        if object_id:
            self._load_object_detail(object_id)

    def _selected_object_ids(self, tree: QTreeWidget) -> list[str]:
        out: list[str] = []
        for item in tree.selectedItems():
            object_id = str(item.data(0, Qt.ItemDataRole.UserRole) or "").strip()
            if object_id:
                out.append(object_id)
        return out

    def _phase0_objects_by_id(self) -> dict[str, dict[str, Any]]:
        return {
            str(item.get("object_id") or ""): item
            for item in (self._phase0_payload.get("objects") or [])
            if isinstance(item, dict)
        }

    def _load_object_detail(self, object_id: str) -> None:
        self._current_object_id = str(object_id or "").strip()
        if not self._current_object_id:
            self._clear_detail()
            return
        if self._http_get is None:
            return
        phase0_snapshot_id = str(self._phase0_payload.get("snapshot_id") or "")
        target_snapshot_id = self._current_snapshot_id or phase0_snapshot_id
        if not target_snapshot_id:
            row = self._phase0_objects_by_id().get(self._current_object_id)
            if row is None:
                self._clear_detail()
                return
            self._object_detail = self._local_phase0_detail(row)
            self._render_detail()
            return
        try:
            payload = self._http_get(f"/archives/snapshots/{target_snapshot_id}/objects/{self._current_object_id}")
        except Exception as exc:
            self._set_status(f"Object load failed: {exc}")
            row = self._phase0_objects_by_id().get(self._current_object_id)
            if row is None:
                return
            self._object_detail = self._local_phase0_detail(row)
        else:
            self._object_detail = dict(payload or {})
        self._render_detail()

    def _local_phase0_detail(self, row: dict[str, Any]) -> dict[str, Any]:
        title_meta = dict((row.get("metadata") or {}).get("title_provenance") or {})
        raw_title = str(title_meta.get("raw") or row.get("title") or "")
        assertion = {
            "assertion_id": f"assert-title-{str(row.get('object_id') or '')}",
            "field": "title",
            "raw_extraction": raw_title,
            "current_value": str(row.get("title") or ""),
            "current_confidence": 1.0,
            "extraction_model": str(row.get("assembly_method") or "assembly"),
            "source_file": str(next((ref.get("file_id") for ref in (row.get("files") or []) if isinstance(ref, dict)), "") or ""),
            "source_type": "assembly",
            "raw_region": {},
            "edits": [],
        }
        return {
            "object": row,
            "anchors": [],
            "mentions": [],
            "relationships": [],
            "clusters": [],
            "assertions": [assertion],
            "preview": {"kind": "none", "label": "No preview available", "path": ""},
            "extraction_summary": {},
            "text_blocks": [],
            "segmentation": {},
            "related": {"by_entity": [], "by_cluster": [], "by_classification": []},
            "health": {
                "file_count": int(row.get("file_count") or len(row.get("files") or [])),
                "missing_file_count": int(row.get("missing_file_count") or 0),
                "assertion_count": 1,
                "edited_assertion_count": 0,
                "resolved_anchor_count": 0,
            },
        }

    def _render_detail(self) -> None:
        detail = self._object_detail
        obj = detail.get("object") if isinstance(detail.get("object"), dict) else {}
        if not obj:
            self._clear_detail()
            return
        health = detail.get("health") if isinstance(detail.get("health"), dict) else {}
        self._object_head.setText(str(obj.get("title") or "Object"))
        self._object_meta.setText(
            f"{str(obj.get('object_type') or '')} | "
            f"{str(obj.get('assembly_method') or '')} "
            f"({float(obj.get('assembly_confidence') or 0.0):.2f}) | "
            f"{str(obj.get('era_bucket') or 'no era bucket')} | "
            f"{str(obj.get('content_complexity') or 'unknown complexity')} | "
            f"{int(health.get('missing_file_count') or 0)} missing file(s)"
        )
        self._render_preview(detail)
        self._render_provenance_tab(detail)
        self._render_files_tab(detail)
        self._render_assertions_tab(detail)
        self._render_structure_tab(detail)
        self._render_related_tab(detail)

    def _render_preview(self, detail: dict[str, Any]) -> None:
        preview = detail.get("preview") if isinstance(detail.get("preview"), dict) else {}
        kind = str(preview.get("kind") or "none")
        label = str(preview.get("label") or "No preview available")
        self._preview_caption.setText(label)
        self._preview_image.clear()
        self._preview_text.clear()
        if kind == "image":
            path = str(preview.get("path") or "")
            pixmap = QPixmap(path) if path else QPixmap()
            if not pixmap.isNull():
                self._preview_image.setPixmap(
                    pixmap.scaled(
                        360,
                        220,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self._preview_stack.setCurrentIndex(1)
                return
            self._preview_info.setText(f"Image preview unavailable.\n{path or label}")
            self._preview_stack.setCurrentIndex(0)
            return
        if kind == "pdf_page":
            page_index = int(preview.get("page_index") or 0)
            path = str(preview.get("path") or "")
            try:
                import fitz  # type: ignore[import-not-found]

                doc = fitz.open(path)
                page = doc.load_page(page_index)
                pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
                png = pix.tobytes("png")
                doc.close()
                pixmap = QPixmap()
                pixmap.loadFromData(png, "PNG")
            except Exception:
                pixmap = QPixmap()
            if not pixmap.isNull():
                self._preview_image.setPixmap(
                    pixmap.scaled(
                        360,
                        220,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                self._preview_stack.setCurrentIndex(1)
                return
            self._preview_text.setHtml(
                "<b>PDF preview</b><br>"
                f"Path: <code>{html.escape(path)}</code><br>"
                f"Page: {page_index + 1}"
            )
            self._preview_stack.setCurrentIndex(2)
            return
        if kind == "audio":
            lines = ["<b>Audio transcript segments</b>"]
            for segment in preview.get("segments") or []:
                if not isinstance(segment, dict):
                    continue
                start_sec = float(segment.get("start_sec") or 0.0)
                end_sec = float(segment.get("end_sec") or 0.0)
                text = html.escape(str(segment.get("text") or ""))
                lines.append(f"[{start_sec:.1f}s - {end_sec:.1f}s] {text}")
            if len(lines) == 1:
                lines.append("No transcript segments available.")
            self._preview_text.setHtml("<br>".join(lines))
            self._preview_stack.setCurrentIndex(2)
            return
        self._preview_info.setText(label or "No preview available.")
        self._preview_stack.setCurrentIndex(0)

    def _render_provenance_tab(self, detail: dict[str, Any]) -> None:
        obj = detail.get("object") if isinstance(detail.get("object"), dict) else {}
        assertions = detail.get("assertions") if isinstance(detail.get("assertions"), list) else []
        anchors = detail.get("anchors") if isinstance(detail.get("anchors"), list) else []
        mentions = detail.get("mentions") if isinstance(detail.get("mentions"), list) else []
        clusters = detail.get("clusters") if isinstance(detail.get("clusters"), list) else []
        extraction_summary = detail.get("extraction_summary") if isinstance(detail.get("extraction_summary"), dict) else {}
        title_assert = next((item for item in assertions if isinstance(item, dict) and str(item.get("field") or "") == "title"), None)
        lines = [
            f"<b>Object ID</b>: {html.escape(str(obj.get('object_id') or ''))}",
            f"<b>Type</b>: {html.escape(str(obj.get('object_type') or ''))}",
            f"<b>Assembly</b>: {html.escape(str(obj.get('assembly_method') or ''))} ({float(obj.get('assembly_confidence') or 0.0):.2f})",
            f"<b>Status</b>: {html.escape(str(obj.get('status') or ''))}",
        ]
        classification = dict((obj.get("metadata") or {}).get("classification") or {})
        if not classification:
            classification = {
                "object_type": str(obj.get("object_type") or ""),
                "era_bucket": str(obj.get("era_bucket") or ""),
                "content_complexity": str(obj.get("content_complexity") or ""),
                "routes": [],
            }
        if title_assert is not None:
            lines.append(
                f"<b>Title Provenance</b>: RAW <code>{html.escape(str(title_assert.get('raw_extraction') or ''))}</code>"
                f" -> NOW <code>{html.escape(str(title_assert.get('current_value') or ''))}</code>"
            )
        lines.append(
            f"<b>Phase 1 Classification</b>: "
            f"type={html.escape(str(classification.get('object_type') or obj.get('object_type') or 'unknown'))}, "
            f"era={html.escape(str(classification.get('era_bucket') or obj.get('era_bucket') or 'unspecified'))}, "
            f"complexity={html.escape(str(classification.get('content_complexity') or obj.get('content_complexity') or 'unknown'))}"
        )
        routes = [str(route) for route in (classification.get("routes") or []) if str(route or "").strip()]
        if routes:
            lines.append(f"<b>Routing</b>: {html.escape(', '.join(routes))}")
        if extraction_summary:
            lines.append(
                f"<b>Phase 2 Extraction</b>: "
                f"status={html.escape(str(extraction_summary.get('status') or 'unknown'))}, "
                f"providers={html.escape(str(extraction_summary.get('provider_summary') or 'none'))}, "
                f"blocks={int(extraction_summary.get('text_block_count') or 0)}"
            )
            capability_counts = dict(extraction_summary.get("capability_counts") or {})
            if capability_counts:
                lines.append(
                    f"<b>Capabilities</b>: "
                    + html.escape(", ".join(f"{key}:{int(value or 0)}" for key, value in sorted(capability_counts.items())))
                )
        earliest = str(obj.get("earliest") or "")
        latest = str(obj.get("latest") or "")
        if earliest or latest:
            lines.append(f"<b>Temporal</b>: {html.escape(earliest or '--')} -> {html.escape(latest or '--')}")
        if str(obj.get("unresolved_reason") or ""):
            lines.append(f"<b>Unresolved</b>: {html.escape(str(obj.get('unresolved_reason') or ''))}")
        lines.append("<b>Lineage</b>: RAW FILE -> ASSEMBLY -> EXTRACTION -> RESOLUTION")
        if anchors:
            lines.append("<br><b>Anchors</b>")
            for anchor in anchors:
                if not isinstance(anchor, dict):
                    continue
                lines.append(
                    f"&nbsp;&nbsp;• {html.escape(str(anchor.get('type') or ''))}: "
                    f"{html.escape(str(anchor.get('raw_expression') or ''))} "
                    f"-> {html.escape(str(anchor.get('earliest') or ''))} / {html.escape(str(anchor.get('latest') or ''))}"
                )
        if mentions:
            lines.append("<br><b>Entity Mentions</b>")
            for mention in mentions[:20]:
                if not isinstance(mention, dict):
                    continue
                lines.append(f"&nbsp;&nbsp;• {html.escape(str(mention.get('mention_text') or mention.get('text_span') or ''))}")
        if clusters:
            lines.append("<br><b>Clusters</b>")
            for cluster in clusters[:12]:
                if not isinstance(cluster, dict):
                    continue
                lines.append(f"&nbsp;&nbsp;• {html.escape(str(cluster.get('label') or ''))}")
        self._provenance_view.setHtml("<br>".join(lines))

    def _render_files_tab(self, detail: dict[str, Any]) -> None:
        self._files_tree.clear()
        obj = detail.get("object") if isinstance(detail.get("object"), dict) else {}
        preview = detail.get("preview") if isinstance(detail.get("preview"), dict) else {}
        for ref in obj.get("files") or []:
            if not isinstance(ref, dict):
                continue
            file_meta = ref.get("file") if isinstance(ref.get("file"), dict) else {}
            exists = bool(file_meta.get("exists"))
            status = "MISSING" if not exists else ("MOVED" if bool(file_meta.get("moved")) else "OK")
            if str(preview.get("source_file_id") or "") == str(ref.get("file_id") or ""):
                status = f"{status} PREVIEW"
            current_path = str(file_meta.get("current_path") or file_meta.get("stored_path") or "")
            item = QTreeWidgetItem(
                [
                    str(ref.get("role") or ""),
                    status,
                    current_path,
                    str(file_meta.get("original_path") or ""),
                    _fmt_bytes(file_meta.get("size_bytes")),
                ]
            )
            item.setToolTip(0, str(file_meta.get("checksum_sha256") or ""))
            if not exists:
                item.setToolTip(1, f"Missing since {_fmt_ts(file_meta.get('lost_detected_at'))}")
            self._files_tree.addTopLevelItem(item)
        self._files_tree.resizeColumnToContents(0)

    def _render_assertions_tab(self, detail: dict[str, Any]) -> None:
        self._assertions_tree.clear()
        assertions = detail.get("assertions") if isinstance(detail.get("assertions"), list) else []
        for assertion in assertions:
            if not isinstance(assertion, dict):
                continue
            root = QTreeWidgetItem(
                [
                    str(assertion.get("field") or ""),
                    str(assertion.get("current_value") or ""),
                    f"{float(assertion.get('current_confidence') or 0.0):.2f}",
                    str(assertion.get("source_type") or assertion.get("extraction_model") or ""),
                ]
            )
            raw_value = str(assertion.get("raw_extraction") or "")
            root.addChild(QTreeWidgetItem(["RAW", raw_value, "", ""]))
            root.addChild(QTreeWidgetItem(["MODEL", str(assertion.get("extraction_model") or ""), "", ""]))
            run_text = str(assertion.get("extraction_run_id") or assertion.get("extraction_run") or "")
            ts_text = _fmt_ts(assertion.get("extraction_timestamp") or assertion.get("extraction_ts"))
            root.addChild(QTreeWidgetItem(["RUN", run_text, ts_text, ""]))
            raw_region = assertion.get("raw_region") if isinstance(assertion.get("raw_region"), dict) else {}
            if raw_region:
                root.addChild(QTreeWidgetItem(["REGION", str(raw_region), "", ""]))
            metadata = assertion.get("metadata") if isinstance(assertion.get("metadata"), dict) else {}
            capability = str(metadata.get("capability") or "")
            if capability:
                root.addChild(QTreeWidgetItem(["CAPABILITY", capability, "", ""]))
            if metadata.get("page_index") is not None:
                root.addChild(QTreeWidgetItem(["PAGE", str(metadata.get("page_index")), "", ""]))
            segment_id = str(metadata.get("segment_id") or "")
            if segment_id:
                root.addChild(QTreeWidgetItem(["SEGMENT", segment_id, "", ""]))
            for edit in assertion.get("edits") or []:
                if not isinstance(edit, dict):
                    continue
                child = QTreeWidgetItem(
                    [
                        f"EDIT {str(edit.get('editor') or '')}",
                        str(edit.get("new_value") or ""),
                        _fmt_ts(edit.get("created_at")),
                        str(edit.get("reason") or ""),
                    ]
                )
                child.addChild(QTreeWidgetItem(["PREVIOUS", str(edit.get("previous_value") or edit.get("previous") or ""), "", ""]))
                root.addChild(child)
            self._assertions_tree.addTopLevelItem(root)
        self._assertions_tree.expandToDepth(0)
        self._assertions_tree.resizeColumnToContents(0)

    def _render_structure_tab(self, detail: dict[str, Any]) -> None:
        self._structure_detail_tree.clear()
        anchors = detail.get("anchors") if isinstance(detail.get("anchors"), list) else []
        mentions = detail.get("mentions") if isinstance(detail.get("mentions"), list) else []
        relationships = detail.get("relationships") if isinstance(detail.get("relationships"), list) else []

        anchor_root = QTreeWidgetItem(["Temporal Anchors", "", "", str(len(anchors))])
        anchor_root.setFlags(anchor_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._structure_detail_tree.addTopLevelItem(anchor_root)
        for anchor in anchors:
            if not isinstance(anchor, dict):
                continue
            when = str(anchor.get("earliest") or "")
            latest = str(anchor.get("latest") or "")
            if latest and latest != when:
                when = f"{when} -> {latest}" if when else latest
            state = "resolved" if bool(anchor.get("resolved")) else str(anchor.get("resolution_requires") or "needs resolution")
            item = QTreeWidgetItem(
                [
                    str(anchor.get("type") or "anchor"),
                    str(anchor.get("raw_expression") or when or anchor.get("anchor_id") or ""),
                    f"{float(anchor.get('confidence') or 0.0):.2f}",
                    state,
                ]
            )
            if when:
                item.addChild(QTreeWidgetItem(["range", when, "", ""]))
            if str(anchor.get("source") or ""):
                item.addChild(QTreeWidgetItem(["source", str(anchor.get("source") or ""), "", ""]))
            anchor_root.addChild(item)
        anchor_root.setExpanded(True)

        mention_root = QTreeWidgetItem(["Entity Mentions", "", "", str(len(mentions))])
        mention_root.setFlags(mention_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._structure_detail_tree.addTopLevelItem(mention_root)
        for mention in mentions:
            if not isinstance(mention, dict):
                continue
            item = QTreeWidgetItem(
                [
                    "mention",
                    str(mention.get("mention_text") or mention.get("text_span") or ""),
                    f"{float(mention.get('mention_confidence') or 0.0):.2f}",
                    str(mention.get("entity_id") or ""),
                ]
            )
            mention_root.addChild(item)
        mention_root.setExpanded(True)

        relationship_root = QTreeWidgetItem(["Relationships", "", "", str(len(relationships))])
        relationship_root.setFlags(relationship_root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self._structure_detail_tree.addTopLevelItem(relationship_root)
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            endpoint = str(rel.get("source_entity_id") or "")
            target = str(rel.get("target_entity_id") or "")
            if target:
                endpoint = f"{endpoint} -> {target}" if endpoint else target
            item = QTreeWidgetItem(
                [
                    str(rel.get("relationship_type") or "relationship"),
                    endpoint,
                    f"{float(rel.get('confidence') or 0.0):.2f}",
                    "",
                ]
            )
            relationship_root.addChild(item)
        relationship_root.setExpanded(True)
        self._structure_detail_tree.expandToDepth(0)
        self._structure_detail_tree.resizeColumnToContents(0)

    def _render_related_tab(self, detail: dict[str, Any]) -> None:
        self._related_tree.clear()
        related = detail.get("related") if isinstance(detail.get("related"), dict) else {}
        for label, key in (("By Entity", "by_entity"), ("By Cluster", "by_cluster"), ("By Classification", "by_classification")):
            rows = related.get(key) if isinstance(related.get(key), list) else []
            root = QTreeWidgetItem([label, "", ""])
            root.setFlags(root.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self._related_tree.addTopLevelItem(root)
            for row in rows:
                if not isinstance(row, dict):
                    continue
                child = QTreeWidgetItem(
                    [
                        label,
                        str(row.get("title") or ""),
                        str(row.get("earliest") or row.get("era_bucket") or ""),
                    ]
                )
                child.setData(0, Qt.ItemDataRole.UserRole, str(row.get("object_id") or ""))
                root.addChild(child)
            root.setExpanded(True)
        self._related_tree.resizeColumnToContents(0)

    def _clear_detail(self) -> None:
        self._object_detail = {}
        self._current_object_id = ""
        self._object_head.setText("Select an object")
        self._object_meta.setText("")
        self._preview_caption.setText("No preview loaded")
        self._preview_info.setText("Select an object to inspect previewable source material.")
        self._preview_image.clear()
        self._preview_text.clear()
        self._preview_stack.setCurrentIndex(0)
        self._provenance_view.setPlainText("")
        self._files_tree.clear()
        self._assertions_tree.clear()
        self._structure_detail_tree.clear()
        self._related_tree.clear()

    def _rerender_all(self) -> None:
        self._render_phase0_review()
        if self._timeline_payload:
            self._render_classification()
        if self._phase2_payload:
            self._render_phase2_review()
        if self._phase3_payload:
            self._render_phase3_review()
        if self._phase4_payload:
            self._render_phase4_review()
        if self._phase5_payload:
            self._render_phase5_review()
        if self._timeline_payload:
            self._render_timeline()
        self._render_left_metrics()

    def _selected_phase0_objects(self) -> list[dict[str, Any]]:
        lookup = self._phase0_objects_by_id()
        return [lookup[object_id] for object_id in self._selected_object_ids(self._assembly_tree) if object_id in lookup]

    def _merge_selected_objects(self) -> None:
        rows = self._selected_phase0_objects()
        if len(rows) < 2 or self._http_post is None:
            self._set_status("Select at least two Phase 0 objects to merge")
            return
        file_ids = [str(ref.get("file_id") or "") for row in rows for ref in (row.get("files") or []) if isinstance(ref, dict)]
        title = "Manual Merge: " + " + ".join(str(row.get("title") or "") for row in rows[:3])
        anchor = rows[0]
        try:
            self._http_post(
                f"/archives/objects/{str(anchor.get('object_id') or '')}/assembly_override",
                {
                    "corpus_id": self._current_corpus_id,
                    "dataset_version_id": self._current_dataset_version_id,
                    "scope_key": f"merge:{uuid.uuid4().hex[:8]}",
                    "action": "merge_files",
                    "payload": {
                        "file_ids": file_ids,
                        "title": title,
                        "object_type": str(anchor.get("object_type") or "unknown"),
                        "media_family": str(anchor.get("media_family") or ""),
                        "group_key": str(anchor.get("object_key") or "manual-merge"),
                    },
                },
            )
        except Exception as exc:
            self._set_status(f"Merge override failed: {exc}")
            return
        self._set_status("Merge override saved. Re-running Phase 0…")
        self._rerun_phase0_after_override()

    def _split_selected_object(self) -> None:
        rows = self._selected_phase0_objects()
        if len(rows) != 1 or self._http_post is None:
            self._set_status("Select exactly one Phase 0 object to split")
            return
        row = rows[0]
        file_ids = [str(ref.get("file_id") or "") for ref in (row.get("files") or []) if isinstance(ref, dict)]
        try:
            self._http_post(
                f"/archives/objects/{str(row.get('object_id') or '')}/assembly_override",
                {
                    "corpus_id": self._current_corpus_id,
                    "dataset_version_id": self._current_dataset_version_id,
                    "scope_key": f"split:{str(row.get('object_id') or '')}",
                    "action": "split_object",
                    "payload": {"file_ids": file_ids},
                },
            )
        except Exception as exc:
            self._set_status(f"Split override failed: {exc}")
            return
        self._set_status("Split override saved. Re-running Phase 0…")
        self._rerun_phase0_after_override()

    def _edit_selected_roles(self) -> None:
        rows = self._selected_phase0_objects()
        if len(rows) != 1 or self._http_post is None:
            self._set_status("Select exactly one Phase 0 object to edit file roles")
            return
        dlg = _RoleEditorDialog(rows[0], self)
        try:
            if dlg.exec() != int(QDialog.DialogCode.Accepted):
                return
            file_roles = dlg.file_roles()
        finally:
            dlg.deleteLater()
        try:
            self._http_post(
                f"/archives/objects/{str(rows[0].get('object_id') or '')}/assembly_override",
                {
                    "corpus_id": self._current_corpus_id,
                    "dataset_version_id": self._current_dataset_version_id,
                    "scope_key": f"roles:{str(rows[0].get('object_id') or '')}",
                    "action": "set_roles",
                    "payload": {"file_roles": file_roles},
                },
            )
        except Exception as exc:
            self._set_status(f"Role override failed: {exc}")
            return
        self._set_status("Role override saved. Re-running Phase 0…")
        self._rerun_phase0_after_override()

    def _rerun_phase0_after_override(self) -> None:
        if self._http_post is None or not self._current_corpus_id or not self._current_dataset_version_id:
            return
        try:
            payload = self._http_post(
                f"/archives/{self._current_corpus_id}/jobs",
                {
                    "dataset_version_id": self._current_dataset_version_id,
                    "phase": "archive_phase0",
                    "parent_snapshot_id": "",
                    "scenario": self._scenario_name,
                    "write_run_artifacts": True,
                },
            )
        except Exception as exc:
            self._set_status(f"Phase 0 rerun failed: {exc}")
            return
        self._start_job_poll(str((payload or {}).get("job_id") or ""))

    # --- Phase 3 structured-extraction edits -------------------------------

    def _selected_structure_item(self) -> Optional[QTreeWidgetItem]:
        items = self._structure_tree.selectedItems()
        return items[0] if items else None

    def _update_structure_actions(self) -> None:
        mode = str(self._view_combo.currentData() or "assembly")
        is_structure = mode == "structure"
        kind = ""
        if is_structure:
            item = self._selected_structure_item()
            if item is not None:
                kind = str(item.data(0, _ROLE_KIND) or "")
        self._pin_date_btn.setEnabled(is_structure and kind == "anchor")
        self._merge_entity_btn.setEnabled(is_structure and kind == "entity")
        self._reject_merge_btn.setEnabled(is_structure and kind == "entity")

    def _structure_anchor_by_id(self, anchor_id: str) -> dict[str, Any]:
        for anchor in self._phase3_payload.get("anchors") or []:
            if isinstance(anchor, dict) and str(anchor.get("anchor_id") or "") == anchor_id:
                return anchor
        return {}

    def _structure_entities(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for mention in self._phase3_payload.get("mentions") or []:
            if not isinstance(mention, dict):
                continue
            entity_id = str(mention.get("entity_id") or "").strip()
            if not entity_id or entity_id in out:
                continue
            out[entity_id] = {
                "entity_id": entity_id,
                "canonical_name": str(mention.get("canonical_name") or mention.get("mention_text") or entity_id),
                "entity_type": str(mention.get("entity_type") or "unknown"),
            }
        return out

    def _pin_selected_anchor(self) -> None:
        item = self._selected_structure_item()
        if item is None or str(item.data(0, _ROLE_KIND) or "") != "anchor" or self._http_post is None:
            self._set_status("Select a temporal anchor to pin a date")
            return
        anchor_id = str(item.data(0, _ROLE_TARGET_ID) or "").strip()
        if not anchor_id or not self._current_snapshot_id:
            return
        dlg = _PinDateDialog(self._structure_anchor_by_id(anchor_id), self)
        try:
            if dlg.exec() != int(QDialog.DialogCode.Accepted):
                return
            values = dlg.values()
        finally:
            dlg.deleteLater()
        if not values.get("earliest"):
            self._set_status("Pin date requires at least an earliest date")
            return
        try:
            self._http_post(
                f"/archives/anchors/{anchor_id}/resolve_override",
                {
                    "corpus_id": self._current_corpus_id,
                    "snapshot_id": self._current_snapshot_id,
                    "earliest": values["earliest"],
                    "latest": values["latest"],
                    "note": values["note"],
                },
            )
        except Exception as exc:
            self._set_status(f"Anchor pin failed: {exc}")
            return
        self._set_status("Date pinned. Re-running Phase 5 resolution…")
        self._rerun_phase5_after_override()

    def _merge_selected_entity(self) -> None:
        self._entity_merge_action(reject=False)

    def _reject_selected_entity(self) -> None:
        self._entity_merge_action(reject=True)

    def _entity_merge_action(self, *, reject: bool) -> None:
        item = self._selected_structure_item()
        if item is None or str(item.data(0, _ROLE_KIND) or "") != "entity" or self._http_post is None:
            self._set_status("Select an entity mention to review merges")
            return
        entity_id = str(item.data(0, _ROLE_TARGET_ID) or "").strip()
        if not entity_id or not self._current_snapshot_id:
            return
        entities = self._structure_entities()
        primary = entities.get(entity_id, {"entity_id": entity_id})
        candidates = [
            value
            for key, value in sorted(entities.items(), key=lambda kv: str(kv[1].get("canonical_name") or "").lower())
            if key != entity_id
        ]
        if not candidates:
            self._set_status("No other entities available to review")
            return
        dlg = _EntityMergeDialog(primary=primary, candidates=candidates, reject=reject, parent=self)
        try:
            if dlg.exec() != int(QDialog.DialogCode.Accepted):
                return
            other_ids = dlg.other_entity_ids()
            canonical = dlg.canonical_name()
        finally:
            dlg.deleteLater()
        if not other_ids:
            self._set_status("Select at least one other entity")
            return
        try:
            self._http_post(
                f"/archives/entities/{entity_id}/merge_override",
                {
                    "corpus_id": self._current_corpus_id,
                    "snapshot_id": self._current_snapshot_id,
                    "other_entity_ids": other_ids,
                    "canonical_name": canonical,
                    "reject": bool(reject),
                },
            )
        except Exception as exc:
            self._set_status(f"Entity merge override failed: {exc}")
            return
        verb = "Merge rejection" if reject else "Merge"
        self._set_status(f"{verb} saved. Re-running Phase 5 resolution…")
        self._rerun_phase5_after_override()

    def _rerun_phase5_after_override(self) -> None:
        if self._http_post is None or not self._current_corpus_id or not self._current_dataset_version_id:
            return
        if not self._current_snapshot_id:
            self._set_status("No snapshot selected for Phase 5 rerun")
            return
        try:
            payload = self._http_post(
                f"/archives/{self._current_corpus_id}/jobs",
                {
                    "dataset_version_id": self._current_dataset_version_id,
                    "phase": "archive_phase5",
                    "parent_snapshot_id": self._current_snapshot_id,
                    "scenario": self._scenario_name,
                    "write_run_artifacts": True,
                },
            )
        except Exception as exc:
            self._set_status(f"Phase 5 rerun failed: {exc}")
            return
        self._start_job_poll(str((payload or {}).get("job_id") or ""))

    @staticmethod
    def _select_combo_data(combo: QComboBox, value: str) -> None:
        want = str(value or "").strip()
        for idx in range(combo.count()):
            if str(combo.itemData(idx) or "") == want:
                combo.setCurrentIndex(idx)
                return
        if combo.count() > 0:
            combo.setCurrentIndex(0)
