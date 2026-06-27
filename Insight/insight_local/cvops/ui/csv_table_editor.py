from __future__ import annotations

import csv
import math
import os
import re
import random
from collections import Counter, defaultdict
from itertools import product as _iproduct
from pathlib import Path
from typing import Any, Optional

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .cvops_theme import cvops_color
from .time_format import format_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mono_font(size: int = 10) -> QFont:
    font = QFont("JetBrains Mono", size)
    if not font.exactMatch():
        font = QFont("IBM Plex Mono", size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def _fmt_size(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.2f} GB"


def _detect_col_type(values: list[str]) -> str:
    non_empty = [v for v in values if v.strip()]
    if not non_empty:
        return "empty"
    numeric = sum(1 for v in non_empty if _is_numeric(v))
    if numeric / len(non_empty) >= 0.90:
        return "numeric"
    if all(v.lower() in ("true", "false", "yes", "no", "0", "1") for v in non_empty):
        return "bool"
    return "string"


def _is_numeric(v: str) -> bool:
    try:
        float(v.strip().replace(",", ""))
        return True
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Folder Browser
# ---------------------------------------------------------------------------

class _FolderBrowserWidget(QWidget):
    """Folder view: file-type aggregation table + filterable / sortable file list."""

    fileOpenRequested = pyqtSignal(object)   # Path
    fileMergeRequested = pyqtSignal(object)  # list[Path]

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._folder: Optional[Path] = None
        self._all_files: list[dict] = []
        self._active_ext_filter: Optional[str] = None
        self._recursive = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # -- Path bar --
        path_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Folder path…")
        self._path_edit.setFont(_mono_font(10))
        self._path_edit.returnPressed.connect(self._on_path_entered)
        path_row.addWidget(self._path_edit, stretch=1)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_folder)
        path_row.addWidget(browse_btn)
        self._recursive_chk = QCheckBox("Recursive")
        self._recursive_chk.stateChanged.connect(lambda _: self._refresh())
        path_row.addWidget(self._recursive_chk)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        path_row.addWidget(refresh_btn)
        layout.addLayout(path_row)

        # -- Summary --
        self._summary_label = QLabel("No folder loaded.")
        self._summary_label.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.7);")
        layout.addWidget(self._summary_label)

        # -- Splitter: type table | file list --
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # Left: file-type aggregation
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        ll.addWidget(QLabel("File Types  (click to filter):"))
        self._type_table = QTableWidget(0, 4)
        self._type_table.setHorizontalHeaderLabels(["Ext", "Count", "Total Size", "%"])
        self._type_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._type_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._type_table.setAlternatingRowColors(True)
        self._type_table.setFont(_mono_font(9))
        self._type_table.setSortingEnabled(True)
        try:
            h = self._type_table.horizontalHeader()
            h.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            h.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            h.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            h.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
            self._type_table.verticalHeader().setVisible(False)
        except Exception:
            pass
        self._type_table.itemSelectionChanged.connect(self._on_type_filter_changed)
        ll.addWidget(self._type_table, stretch=1)
        show_all_btn = QPushButton("Clear Type Filter")
        show_all_btn.clicked.connect(self._clear_ext_filter)
        ll.addWidget(show_all_btn)
        splitter.addWidget(left)
        splitter.setStretchFactor(0, 1)

        # Right: file list
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        file_filter_row = QHBoxLayout()
        file_filter_row.addWidget(QLabel("Search:"))
        self._file_search = QLineEdit()
        self._file_search.setPlaceholderText("Filter by filename…")
        self._file_search.textChanged.connect(self._apply_file_filter)
        file_filter_row.addWidget(self._file_search, stretch=1)
        self._csv_only_chk = QCheckBox("CSV/TSV only")
        self._csv_only_chk.stateChanged.connect(lambda _: self._apply_file_filter())
        file_filter_row.addWidget(self._csv_only_chk)
        rl.addLayout(file_filter_row)

        self._file_table = QTableWidget(0, 5)
        self._file_table.setHorizontalHeaderLabels(["Name", "Ext", "Size", "Modified", "Path"])
        self._file_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._file_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._file_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._file_table.setAlternatingRowColors(True)
        self._file_table.setFont(_mono_font(9))
        self._file_table.setSortingEnabled(True)
        try:
            fh = self._file_table.horizontalHeader()
            fh.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            fh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            fh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
            fh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
            fh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
            self._file_table.verticalHeader().setVisible(False)
        except Exception:
            pass
        self._file_table.doubleClicked.connect(self._on_file_double_clicked)
        self._file_table.itemSelectionChanged.connect(self._on_file_selection_changed)
        rl.addWidget(self._file_table, stretch=1)

        self._sel_label = QLabel("No selection.")
        self._sel_label.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        rl.addWidget(self._sel_label)

        action_row = QHBoxLayout()
        self._open_btn = QPushButton("Open Selected in Editor")
        self._open_btn.setToolTip("Open the selected file in the CSV editor tab.")
        self._open_btn.clicked.connect(self._open_selected)
        self._open_btn.setEnabled(False)
        action_row.addWidget(self._open_btn)
        self._merge_btn = QPushButton("Merge & Open Selected CSVs")
        self._merge_btn.setToolTip(
            "Stack multiple selected CSV files vertically (union of headers) and open in editor."
        )
        self._merge_btn.clicked.connect(self._merge_selected)
        self._merge_btn.setEnabled(False)
        action_row.addWidget(self._merge_btn)
        rl.addLayout(action_row)

        splitter.addWidget(right)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([220, 700])

    # -- Public --

    def load_folder(self, folder: Path) -> None:
        self._folder = folder.resolve()
        self._path_edit.setText(str(self._folder))
        self._scan()

    # -- Internal --

    def _on_path_entered(self) -> None:
        p = Path(self._path_edit.text().strip())
        if p.is_dir():
            self._folder = p.resolve()
            self._scan()
        else:
            QMessageBox.warning(self, "Invalid Path", f"Not a directory:\n{p}")

    def _browse_folder(self) -> None:
        start = str(self._folder) if self._folder else ""
        d = QFileDialog.getExistingDirectory(self, "Select Folder", start)
        if d:
            self._folder = Path(d).resolve()
            self._path_edit.setText(str(self._folder))
            self._scan()

    def _refresh(self) -> None:
        self._recursive = self._recursive_chk.isChecked()
        if self._folder:
            self._scan()

    def _scan(self) -> None:
        if not self._folder or not self._folder.is_dir():
            return
        self._recursive = self._recursive_chk.isChecked()
        self._all_files = []
        try:
            iterator = self._folder.rglob("*") if self._recursive else self._folder.iterdir()
            for entry in iterator:
                if not entry.is_file():
                    continue
                try:
                    st = entry.stat()
                    ext = entry.suffix.lower() or "(none)"
                    mtime = format_timestamp(st.st_mtime)
                    self._all_files.append({
                        "path": entry,
                        "name": entry.name,
                        "ext": ext,
                        "size": st.st_size,
                        "mtime": mtime,
                        "rel": str(entry.relative_to(self._folder)),
                    })
                except Exception:
                    pass
        except Exception as exc:
            QMessageBox.warning(self, "Scan Error", str(exc))
            return

        total = len(self._all_files)
        total_size = sum(f["size"] for f in self._all_files)
        n_types = len({f["ext"] for f in self._all_files})
        mode = "recursive" if self._recursive else "flat"
        self._summary_label.setText(
            f"{total:,} file(s)   |   {_fmt_size(total_size)}   |   "
            f"{n_types} type(s)   |   [{mode}]"
        )
        self._rebuild_type_table()
        self._active_ext_filter = None
        self._apply_file_filter()

    def _rebuild_type_table(self) -> None:
        by_ext: dict[str, dict] = defaultdict(lambda: {"count": 0, "size": 0})
        for f in self._all_files:
            by_ext[f["ext"]]["count"] += 1
            by_ext[f["ext"]]["size"] += f["size"]
        total = len(self._all_files) or 1

        self._type_table.setSortingEnabled(False)
        rows = sorted(by_ext.items(), key=lambda kv: kv[1]["count"], reverse=True)
        self._type_table.setRowCount(len(rows))
        for r, (ext, info) in enumerate(rows):
            self._type_table.setItem(r, 0, QTableWidgetItem(ext))
            cnt = QTableWidgetItem(str(info["count"]))
            cnt.setData(Qt.ItemDataRole.UserRole, info["count"])
            self._type_table.setItem(r, 1, cnt)
            self._type_table.setItem(r, 2, QTableWidgetItem(_fmt_size(info["size"])))
            self._type_table.setItem(r, 3, QTableWidgetItem(f"{100.0 * info['count'] / total:.1f}%"))
        self._type_table.setSortingEnabled(True)

    def _on_type_filter_changed(self) -> None:
        sel = self._type_table.selectionModel().selectedRows()
        if sel:
            it = self._type_table.item(sel[0].row(), 0)
            self._active_ext_filter = it.text() if it else None
        else:
            self._active_ext_filter = None
        self._apply_file_filter()

    def _clear_ext_filter(self) -> None:
        self._type_table.clearSelection()
        self._active_ext_filter = None
        self._apply_file_filter()

    def _apply_file_filter(self) -> None:
        search = self._file_search.text().strip().lower()
        csv_only = self._csv_only_chk.isChecked()
        visible = [
            f for f in self._all_files
            if (not csv_only or f["ext"] in (".csv", ".tsv"))
            and (not self._active_ext_filter or f["ext"] == self._active_ext_filter)
            and (not search or search in f["name"].lower())
        ]

        self._file_table.setSortingEnabled(False)
        self._file_table.setRowCount(len(visible))
        for r, f in enumerate(visible):
            name_it = QTableWidgetItem(f["name"])
            name_it.setData(Qt.ItemDataRole.UserRole, str(f["path"]))
            self._file_table.setItem(r, 0, name_it)
            self._file_table.setItem(r, 1, QTableWidgetItem(f["ext"]))
            sz_it = QTableWidgetItem(_fmt_size(f["size"]))
            sz_it.setData(Qt.ItemDataRole.UserRole, f["size"])
            self._file_table.setItem(r, 2, sz_it)
            self._file_table.setItem(r, 3, QTableWidgetItem(f["mtime"]))
            self._file_table.setItem(r, 4, QTableWidgetItem(f.get("rel", "")))
        self._file_table.setSortingEnabled(True)
        self._on_file_selection_changed()

    def _on_file_selection_changed(self) -> None:
        sel_rows = self._file_table.selectionModel().selectedRows()
        n = len(sel_rows)
        csv_count = sum(
            1 for idx in sel_rows
            if (self._file_table.item(idx.row(), 1) or QTableWidgetItem("")).text()
            in (".csv", ".tsv")
        )
        self._sel_label.setText(f"{n} selected   |   {csv_count} CSV/TSV")
        self._open_btn.setEnabled(n == 1)
        self._merge_btn.setEnabled(csv_count >= 2)

    def _on_file_double_clicked(self, index) -> None:
        it = self._file_table.item(index.row(), 0)
        if it:
            self.fileOpenRequested.emit(Path(str(it.data(Qt.ItemDataRole.UserRole))))

    def _open_selected(self) -> None:
        sel = self._file_table.selectionModel().selectedRows()
        if len(sel) == 1:
            it = self._file_table.item(sel[0].row(), 0)
            if it:
                self.fileOpenRequested.emit(Path(str(it.data(Qt.ItemDataRole.UserRole))))

    def _merge_selected(self) -> None:
        sel = self._file_table.selectionModel().selectedRows()
        paths = []
        for idx in sel:
            ext_it = self._file_table.item(idx.row(), 1)
            if ext_it and ext_it.text() in (".csv", ".tsv"):
                name_it = self._file_table.item(idx.row(), 0)
                if name_it:
                    paths.append(Path(str(name_it.data(Qt.ItemDataRole.UserRole))))
        if len(paths) >= 2:
            self.fileMergeRequested.emit(paths)


# ---------------------------------------------------------------------------
# CSV Editor
# ---------------------------------------------------------------------------

class _CsvEditorWidget(QWidget):
    """Full-featured CSV table editor with sort, filter, column stats, find & replace."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._path: Optional[Path] = None
        self._headers: list[str] = []
        self._all_rows: list[list[str]] = []
        self._filtered_indices: list[int] = []
        self._sort_col: int = -1
        self._sort_asc: bool = True
        self._dirty = False
        self._fully_loaded = False
        self.max_rows: int = 50_000
        self._find_pos: tuple[int, int] = (0, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # -- Toolbar --
        tb = QHBoxLayout()
        self._file_label = QLabel("No file loaded.")
        self._file_label.setFont(_mono_font(9))
        self._file_label.setStyleSheet("color: rgba(133,153,0,0.7);")
        self._file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        tb.addWidget(self._file_label, stretch=1)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self.reload)
        tb.addWidget(reload_btn)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._save)
        self._save_btn.setEnabled(False)
        tb.addWidget(self._save_btn)
        save_as_btn = QPushButton("Save As…")
        save_as_btn.clicked.connect(self._save_as)
        tb.addWidget(save_as_btn)
        layout.addLayout(tb)

        # -- Stats bar --
        self._stats_label = QLabel("")
        self._stats_label.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        layout.addWidget(self._stats_label)

        # -- Ops row --
        ops = QHBoxLayout()
        ops.addWidget(QLabel("Filter rows:"))
        self._row_filter = QLineEdit()
        self._row_filter.setPlaceholderText("search across all cells to hide non-matching rows…")
        self._row_filter.textChanged.connect(self._apply_row_filter)
        ops.addWidget(self._row_filter, stretch=1)
        ops.addWidget(QLabel("  "))

        for label, tip, slot in (
            ("+ Row",   "Append an empty row.",                  self._add_row),
            ("- Row(s)", "Delete selected row(s).",              self._delete_selected_rows),
            ("+ Col",   "Append a new column.",                  self._add_column),
            ("- Col(s)", "Delete selected column(s).",           self._delete_selected_cols),
        ):
            btn = QPushButton(label)
            btn.setToolTip(tip)
            btn.clicked.connect(slot)
            ops.addWidget(btn)

        self._fr_toggle = QPushButton("Find & Replace")
        self._fr_toggle.setCheckable(True)
        self._fr_toggle.toggled.connect(self._toggle_find_replace)
        ops.addWidget(self._fr_toggle)

        col_stats_btn = QPushButton("Col Stats")
        col_stats_btn.setToolTip("Show statistics for the focused column.")
        col_stats_btn.clicked.connect(self._show_focused_col_stats)
        ops.addWidget(col_stats_btn)
        layout.addLayout(ops)

        # -- Find & Replace bar --
        self._fr_bar = QWidget()
        fr_l = QHBoxLayout(self._fr_bar)
        fr_l.setContentsMargins(0, 0, 0, 0)
        fr_l.addWidget(QLabel("Find:"))
        self._find_edit = QLineEdit()
        fr_l.addWidget(self._find_edit, stretch=1)
        fr_l.addWidget(QLabel("Replace:"))
        self._replace_edit = QLineEdit()
        fr_l.addWidget(self._replace_edit, stretch=1)
        self._fr_case = QCheckBox("Case-sensitive")
        fr_l.addWidget(self._fr_case)
        self._fr_whole = QCheckBox("Whole cell")
        fr_l.addWidget(self._fr_whole)
        for label, slot in (
            ("Find Next",    self._find_next),
            ("Replace",      self._replace_current),
            ("Replace All",  self._replace_all),
        ):
            b = QPushButton(label)
            b.clicked.connect(slot)
            fr_l.addWidget(b)
        self._fr_bar.setVisible(False)
        layout.addWidget(self._fr_bar)

        # -- Main table --
        self._table = QTableWidget(0, 0)
        self._table.setFont(_mono_font(10))
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._table.itemChanged.connect(self._on_item_changed)
        hdr = self._table.horizontalHeader()
        hdr.sectionClicked.connect(self._on_header_clicked)
        hdr.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        hdr.customContextMenuRequested.connect(self._col_header_menu)
        self._table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._table.customContextMenuRequested.connect(self._row_context_menu)
        try:
            hdr.setStretchLastSection(True)
        except Exception:
            pass
        layout.addWidget(self._table, stretch=1)

        # -- Status --
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        layout.addWidget(self._status)

    # -- Public API --

    def load(self, path: Path) -> None:
        self._path = path.resolve()
        self._file_label.setText(str(self._path))
        self.reload()

    def load_merged(self, paths: list[Path]) -> None:
        """Vertically stack multiple CSV files (union of columns)."""
        merged_headers: list[str] = []
        merged_rows: list[list[str]] = []
        for p in paths:
            rows, _ = self._read_csv(p)
            if not rows:
                continue
            h = rows[0]
            for col in h:
                if col not in merged_headers:
                    merged_headers.append(col)
            col_map = {col: i for i, col in enumerate(h)}
            for row in rows[1:]:
                aligned = [row[col_map[c]] if c in col_map and col_map[c] < len(row) else "" for c in merged_headers]
                merged_rows.append(aligned)

        if not merged_headers:
            self._status.setText("[ERROR] Merge failed: no readable CSV files.")
            return

        self._path = None
        self._file_label.setText(f"[MERGED] {len(paths)} files — {len(merged_rows):,} total rows")
        self._headers = merged_headers
        self._all_rows = merged_rows
        self._fully_loaded = True
        self._dirty = False
        self._sort_col = -1
        self._sort_asc = True
        self._row_filter.blockSignals(True)
        self._row_filter.clear()
        self._row_filter.blockSignals(False)
        self._apply_row_filter()
        self._update_stats()
        self._status.setText(
            f"Merged {len(paths)} file(s) — {len(merged_rows):,} rows, {len(merged_headers)} columns."
        )

    def reload(self) -> None:
        if not self._path:
            return
        if not self._path.exists() or not self._path.is_file():
            self._status.setText("[ERROR] File not found.")
            return
        rows, too_large = self._read_csv(self._path)
        if not rows:
            return
        self._fully_loaded = not too_large
        self._headers = rows[0] if rows else []
        self._all_rows = rows[1:] if len(rows) > 1 else []
        self._dirty = False
        self._sort_col = -1
        self._sort_asc = True
        self._row_filter.blockSignals(True)
        self._row_filter.clear()
        self._row_filter.blockSignals(False)
        self._apply_row_filter()
        self._update_stats()
        hint = f" [WARNING: truncated to {self.max_rows:,} rows]" if too_large else ""
        self._status.setText(
            f"Loaded {len(self._all_rows):,} row(s), {len(self._headers)} column(s){hint}."
        )

    # -- Data I/O --

    def _read_csv(self, path: Path) -> tuple[list[list[str]], bool]:
        rows: list[list[str]] = []
        too_large = False
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i >= self.max_rows:
                        too_large = True
                        break
                    rows.append([str(c) for c in row])
        except Exception as exc:
            self._status.setText(f"[ERROR] Read failed: {exc}")
        return rows, too_large

    # -- Table rendering --

    def _rebuild_table(self) -> None:
        self._table.blockSignals(True)
        self._table.setColumnCount(len(self._headers))
        self._table.setRowCount(len(self._filtered_indices))

        labels = list(self._headers)
        if 0 <= self._sort_col < len(labels):
            labels[self._sort_col] += " [v]" if self._sort_asc else " [^]"
        if labels:
            self._table.setHorizontalHeaderLabels(labels)

        for tbl_row, data_idx in enumerate(self._filtered_indices):
            row = self._all_rows[data_idx]
            for c in range(len(self._headers)):
                val = row[c] if c < len(row) else ""
                it = QTableWidgetItem(val)
                it.setData(Qt.ItemDataRole.UserRole, data_idx)
                self._table.setItem(tbl_row, c, it)

        self._table.blockSignals(False)
        try:
            self._table.resizeColumnsToContents()
            for c in range(self._table.columnCount()):
                if self._table.columnWidth(c) > 300:
                    self._table.setColumnWidth(c, 300)
        except Exception:
            pass

    def _apply_row_filter(self) -> None:
        needle = self._row_filter.text().strip().lower()
        self._filtered_indices = [
            i for i, row in enumerate(self._all_rows)
            if not needle or any(needle in cell.lower() for cell in row)
        ]
        self._find_pos = (0, 0)
        self._rebuild_table()
        self._update_stats()

    def _update_stats(self) -> None:
        total = len(self._all_rows)
        shown = len(self._filtered_indices)
        cols = len(self._headers)
        nulls = sum(
            1
            for i in self._filtered_indices
            for cell in self._all_rows[i]
            if not cell.strip()
        )
        parts = [f"{shown:,}/{total:,} rows", f"{cols} cols"]
        if nulls:
            parts.append(f"{nulls:,} empty cells")
        if self._dirty:
            parts.append("[UNSAVED]")
        if 0 <= self._sort_col < cols:
            d = "ASC" if self._sort_asc else "DESC"
            parts.append(f"sorted: '{self._headers[self._sort_col]}' {d}")
        self._stats_label.setText("   |   ".join(parts))

    # -- Public data access for visualization --

    def get_data_snapshot(self) -> tuple[list[str], list[list[str]]]:
        """Return a snapshot of (headers, all_rows) for the visualizer."""
        return list(self._headers), [list(r) for r in self._all_rows]

    # -- Cell change tracking --

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        data_idx = item.data(Qt.ItemDataRole.UserRole)
        if data_idx is None:
            return
        col = item.column()
        try:
            data_idx = int(data_idx)
            row = self._all_rows[data_idx]
            while len(row) <= col:
                row.append("")
            row[col] = item.text()
        except (IndexError, TypeError, ValueError):
            pass
        self._dirty = True
        self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
        self._update_stats()

    # -- Sorting --

    def _on_header_clicked(self, col: int) -> None:
        if col == self._sort_col:
            self._sort_asc = not self._sort_asc
        else:
            self._sort_col = col
            self._sort_asc = True
        self._sort_data()
        self._apply_row_filter()

    def _sort_data(self) -> None:
        c = self._sort_col
        if c < 0 or c >= len(self._headers):
            return

        def key(row: list[str]) -> tuple:
            val = row[c] if c < len(row) else ""
            try:
                return (0, float(val.strip().replace(",", "")))
            except (ValueError, AttributeError):
                return (1, val.lower())

        self._all_rows.sort(key=key, reverse=not self._sort_asc)

    # -- Row / Column operations --

    def _add_row(self) -> None:
        self._all_rows.append([""] * len(self._headers))
        self._dirty = True
        self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
        self._apply_row_filter()
        self._table.scrollToBottom()
        self._status.setText("Row appended.")

    def _delete_selected_rows(self) -> None:
        sel = {idx.row() for idx in self._table.selectionModel().selectedRows()}
        if not sel:
            return
        data_set = {self._filtered_indices[r] for r in sel if r < len(self._filtered_indices)}
        self._all_rows = [row for i, row in enumerate(self._all_rows) if i not in data_set]
        self._dirty = True
        self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
        self._apply_row_filter()
        self._status.setText(f"Deleted {len(data_set)} row(s).")

    def _add_column(self) -> None:
        name, ok = QInputDialog.getText(self, "Add Column", "Column name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if name in self._headers:
            QMessageBox.warning(self, "Duplicate Column", f"Column '{name}' already exists.")
            return
        self._headers.append(name)
        for row in self._all_rows:
            while len(row) < len(self._headers):
                row.append("")
        self._dirty = True
        self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
        self._rebuild_table()
        self._update_stats()

    def _delete_selected_cols(self) -> None:
        cols = sorted({idx.column() for idx in self._table.selectedIndexes()}, reverse=True)
        if not cols:
            return
        if QMessageBox.warning(
            self, "Delete Columns",
            f"Delete {len(cols)} column(s)? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        for c in cols:
            if 0 <= c < len(self._headers):
                del self._headers[c]
                for row in self._all_rows:
                    if c < len(row):
                        del row[c]
        self._dirty = True
        self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
        self._rebuild_table()
        self._update_stats()

    def _rename_column(self, col: int) -> None:
        if not (0 <= col < len(self._headers)):
            return
        name, ok = QInputDialog.getText(self, "Rename Column", "New name:", text=self._headers[col])
        if ok and name.strip() and name.strip() != self._headers[col]:
            self._headers[col] = name.strip()
            self._dirty = True
            self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
            self._rebuild_table()

    # -- Column stats --

    def _show_focused_col_stats(self) -> None:
        col = self._table.currentColumn()
        if col < 0:
            self._status.setText("Click a cell first to select a column.")
            return
        self._show_column_stats(col)

    def _show_column_stats(self, col: int) -> None:
        if not (0 <= col < len(self._headers)):
            return
        name = self._headers[col]
        values = [
            (self._all_rows[i][col] if col < len(self._all_rows[i]) else "")
            for i in self._filtered_indices
        ]
        n = len(values)
        empty = sum(1 for v in values if not v.strip())
        stripped = [v.strip() for v in values]
        unique = len(set(stripped))
        col_type = _detect_col_type(values)

        lines = [
            f"Column:       {name}",
            f"Type hint:    {col_type}",
            f"Total rows:   {n:,}",
            f"Non-empty:    {n - empty:,}",
            f"Empty/null:   {empty:,}  ({100.0 * empty / n:.1f}%" if n else "Empty/null:   0",
            f"Unique vals:  {unique:,}",
        ]
        if col_type == "numeric":
            nums = [float(v.replace(",", "")) for v in values if _is_numeric(v)]
            if nums:
                lines += [
                    f"Min:          {min(nums):.6g}",
                    f"Max:          {max(nums):.6g}",
                    f"Mean:         {sum(nums) / len(nums):.6g}",
                    f"Median:       {sorted(nums)[len(nums) // 2]:.6g}",
                ]
        if unique <= 30:
            ctr = Counter(v for v in stripped if v)
            lines.append("\nTop values:")
            for val, cnt in ctr.most_common(10):
                lines.append(f"  {val!r}: {cnt:,}")

        QMessageBox.information(self, f"[STATS] {name}", "\n".join(lines))

    # -- Context menus --

    def _col_header_menu(self, pos) -> None:
        col = self._table.horizontalHeader().logicalIndexAt(pos)
        if not (0 <= col < len(self._headers)):
            return
        menu = QMenu(self)
        acts = {
            "rename":     menu.addAction(f"Rename '{self._headers[col]}'…"),
            "sort_asc":   menu.addAction("Sort Ascending"),
            "sort_desc":  menu.addAction("Sort Descending"),
        }
        menu.addSeparator()
        acts["stats"] = menu.addAction("Column Stats")
        menu.addSeparator()
        acts["delete"] = menu.addAction(f"Delete Column '{self._headers[col]}'")

        act = menu.exec(self._table.horizontalHeader().mapToGlobal(pos))
        if act == acts["rename"]:
            self._rename_column(col)
        elif act == acts["sort_asc"]:
            self._sort_col = col; self._sort_asc = True
            self._sort_data(); self._apply_row_filter()
        elif act == acts["sort_desc"]:
            self._sort_col = col; self._sort_asc = False
            self._sort_data(); self._apply_row_filter()
        elif act == acts["stats"]:
            self._show_column_stats(col)
        elif act == acts["delete"]:
            del self._headers[col]
            for row in self._all_rows:
                if col < len(row):
                    del row[col]
            self._dirty = True
            self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
            self._rebuild_table(); self._update_stats()

    def _row_context_menu(self, pos) -> None:
        menu = QMenu(self)
        acts = {
            "ins_above": menu.addAction("Insert Row Above"),
            "ins_below": menu.addAction("Insert Row Below"),
        }
        menu.addSeparator()
        acts["del"] = menu.addAction("Delete Selected Row(s)")
        act = menu.exec(self._table.mapToGlobal(pos))
        tbl_row = self._table.rowAt(pos.y())
        if act == acts["ins_above"]:
            self._insert_row_at(tbl_row)
        elif act == acts["ins_below"]:
            self._insert_row_at(tbl_row + 1)
        elif act == acts["del"]:
            self._delete_selected_rows()

    def _insert_row_at(self, tbl_row: int) -> None:
        tbl_row = max(0, tbl_row)
        if tbl_row < len(self._filtered_indices):
            data_idx = self._filtered_indices[tbl_row]
        else:
            data_idx = len(self._all_rows)
        self._all_rows.insert(data_idx, [""] * len(self._headers))
        self._dirty = True
        self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
        self._apply_row_filter()

    # -- Find & Replace --

    def _toggle_find_replace(self, checked: bool) -> None:
        self._fr_bar.setVisible(checked)

    def _find_next(self) -> None:
        needle = self._find_edit.text()
        if not needle:
            return
        case = self._fr_case.isChecked()
        whole = self._fr_whole.isChecked()
        r_count = self._table.rowCount()
        c_count = self._table.columnCount()
        positions = [(r, c) for r in range(r_count) for c in range(c_count)]
        sr, sc = self._find_pos
        # Start searching after current position
        start = next(
            (i for i, (r, c) in enumerate(positions) if r > sr or (r == sr and c > sc)),
            0,
        )
        ordered = positions[start:] + positions[:start]
        for r, c in ordered:
            it = self._table.item(r, c)
            if not it:
                continue
            val, n = (it.text(), needle) if case else (it.text().lower(), needle.lower())
            if (val == n) if whole else (n in val):
                self._table.setCurrentCell(r, c)
                self._table.scrollTo(self._table.currentIndex())
                self._find_pos = (r, c)
                return
        self._status.setText(f"'{needle}' not found.")

    def _replace_current(self) -> None:
        it = self._table.currentItem()
        if not it:
            return
        self._do_replace_item(it, self._find_edit.text(), self._replace_edit.text())

    def _replace_all(self) -> None:
        needle = self._find_edit.text()
        replacement = self._replace_edit.text()
        if not needle:
            return
        count = 0
        self._table.blockSignals(True)
        for r in range(self._table.rowCount()):
            for c in range(self._table.columnCount()):
                it = self._table.item(r, c)
                if it and self._do_replace_item(it, needle, replacement, commit=True):
                    count += 1
        self._table.blockSignals(False)
        if count:
            self._dirty = True
            self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
            self._update_stats()
        self._status.setText(f"Replaced {count:,} cell(s).")

    def _do_replace_item(
        self, it: QTableWidgetItem, needle: str, replacement: str, *, commit: bool = False
    ) -> bool:
        if not needle:
            return False
        case = self._fr_case.isChecked()
        whole = self._fr_whole.isChecked()
        val = it.text()
        test_v = val if case else val.lower()
        test_n = needle if case else needle.lower()
        if whole:
            if test_v != test_n:
                return False
            new_val = replacement
        else:
            if test_n not in test_v:
                return False
            flags = 0 if case else re.IGNORECASE
            new_val = re.sub(re.escape(needle), replacement, val, flags=flags)

        it.setText(new_val)
        data_idx = it.data(Qt.ItemDataRole.UserRole)
        if data_idx is not None:
            try:
                col = it.column()
                row = self._all_rows[int(data_idx)]
                while len(row) <= col:
                    row.append("")
                row[col] = new_val
            except (IndexError, TypeError, ValueError):
                pass
        if not commit:
            self._dirty = True
            self._save_btn.setEnabled(bool(self._fully_loaded and self._path))
        return True

    # -- Save --

    def _save(self) -> None:
        if not self._path:
            self._save_as()
            return
        if not (self._fully_loaded and self._dirty):
            return
        if QMessageBox.warning(
            self, "Save CSV", f"Overwrite file?\n\n{self._path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        ) != QMessageBox.StandardButton.Yes:
            return
        self._write_to(self._path)

    def _save_as(self) -> None:
        start = str(self._path) if self._path else ""
        dest, _ = QFileDialog.getSaveFileName(self, "Save CSV As", start, "CSV Files (*.csv)")
        if not dest:
            return
        self._path = Path(dest)
        self._file_label.setText(str(self._path))
        self._fully_loaded = True
        self._write_to(self._path)

    def _write_to(self, path: Path) -> None:
        try:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self._headers)
                for row in self._all_rows:
                    writer.writerow(row)
        except Exception as exc:
            self._status.setText(f"[ERROR] Save failed: {exc}")
            return
        self._dirty = False
        self._save_btn.setEnabled(False)
        self._update_stats()
        self._status.setText(f"Saved {path.name}  ({len(self._all_rows):,} rows).")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

_CHART_TYPES = [
    "Investigate Deck",
    "Scatter",
    "Line",
    "Bar",
    "Histogram",
    "Box Plot",
    "Correlation Heatmap",
    "Missing Data",
    "Summary Table",
    "Pie",
]

_AGGREGATIONS = ["Count", "Mean", "Sum", "Median", "Min", "Max"]

def _plot_colors() -> list[str]:
    return [
        cvops_color("accent_active"),
        cvops_color("accent_select"),
        cvops_color("text_signal"),
        cvops_color("accent_warn"),
        cvops_color("text_iron"),
        cvops_color("accent_alert"),
        cvops_color("line_med"),
        cvops_color("text_bright"),
    ]


def _fig_bg() -> str:
    return cvops_color("bg_void")


def _axes_bg() -> str:
    return cvops_color("bg_panel")


def _text_clr() -> str:
    return cvops_color("text_signal")


def _grid_clr() -> str:
    return cvops_color("line_med")


def _apply_dark_axes(ax) -> None:
    """Apply solarized-dark styling to a matplotlib Axes."""
    ax.set_facecolor(_axes_bg())
    for spine in ax.spines.values():
        spine.set_color(_grid_clr())
    ax.tick_params(colors=_text_clr(), labelsize=8)
    ax.xaxis.label.set_color(_text_clr())
    ax.yaxis.label.set_color(_text_clr())
    ax.title.set_color(_text_clr())
    ax.grid(True, color=_grid_clr(), linewidth=0.5, linestyle="--", alpha=0.6)


class _VisualizationWidget(QWidget):
    """Data visualization panel: 7 chart types, dark theme, matplotlib canvas."""
    cellSpaceToggleRequested = pyqtSignal(bool)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("csvInvestigatorWidget")
        self._headers: list[str] = []
        self._all_rows: list[list[str]] = []
        self._source_name: str = ""
        self._last_chart_summary: str = ""
        self._profile: dict[str, Any] = {}
        self._deck_template_order: list[str] = ["distribution", "correlation", "scatter", "categorical", "missing", "stats"]
        self._deck_active_templates: list[str] = list(self._deck_template_order)
        self._cell_space_open: bool = False

        self.setMinimumHeight(460)
        self.setStyleSheet(
            """
            QWidget#csvInvestigatorWidget {
              background: #050807;
            }
            QWidget#csvInvestigatorWidget QScrollArea {
              border: 1px solid rgba(90,104,98,0.62);
              background: rgba(18,24,21,0.88);
            }
            QWidget#csvInvestigatorWidget QLabel {
              color: rgba(232,237,233,0.9);
            }
            QWidget#csvInvestigatorWidget QComboBox,
            QWidget#csvInvestigatorWidget QListWidget,
            QWidget#csvInvestigatorWidget QSpinBox {
              background: rgba(10,14,12,0.95);
              border: 1px solid rgba(90,104,98,0.62);
              color: rgba(232,237,233,0.9);
              padding: 2px 4px;
            }
            QWidget#csvInvestigatorWidget QComboBox:hover,
            QWidget#csvInvestigatorWidget QListWidget:hover,
            QWidget#csvInvestigatorWidget QSpinBox:hover {
              border-color: rgba(126,140,132,0.82);
            }
            QWidget#csvInvestigatorWidget QPushButton {
              background: rgba(24,32,28,0.94);
              border: 1px solid rgba(90,104,98,0.62);
              color: rgba(232,237,233,0.88);
              padding: 3px 8px;
            }
            QWidget#csvInvestigatorWidget QPushButton:hover {
              background: rgba(34,42,38,0.95);
              border-color: rgba(126,140,132,0.82);
            }
            QWidget#csvInvestigatorWidget QPushButton:checked {
              background: rgba(122,232,96,0.13);
              color: rgba(122,232,96,0.96);
              border-color: rgba(122,232,96,0.52);
            }
            """
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(6)

        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setChildrenCollapsible(False)
        main_splitter.setHandleWidth(4)
        outer.addWidget(main_splitter, stretch=1)

        # ---- LEFT: flat inline form ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(240)
        scroll.setMaximumWidth(320)
        scroll.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        ctrl_container = QWidget()
        ctrl_l = QVBoxLayout(ctrl_container)
        ctrl_l.setContentsMargins(8, 8, 8, 8)
        ctrl_l.setSpacing(6)
        scroll.setWidget(ctrl_container)
        self._ctrl_scroll = scroll
        main_splitter.addWidget(scroll)

        template_title = QLabel("Templates")
        template_title.setFont(_mono_font(9))
        template_title.setStyleSheet("color: rgba(147,161,161,0.82); letter-spacing: 0.8px; font-weight: 700; text-transform: uppercase;")
        ctrl_l.addWidget(template_title)

        template_grid = QHBoxLayout()
        template_grid.setContentsMargins(0, 0, 0, 0)
        template_grid.setSpacing(4)
        self._template_buttons: dict[str, QPushButton] = {}
        for label, key in (
            ("Deck", "investigate"),
            ("Dist", "distribution"),
            ("Corr", "correlation"),
            ("Scatter", "scatter"),
            ("Cat", "categorical"),
            ("Missing", "missing"),
            ("Stats", "stats"),
        ):
            btn = QPushButton(label)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(f"Switch to {label} investigator template.")
            btn.setCheckable(True)
            btn.setStyleSheet(
                "QPushButton { padding: 3px 7px; min-height: 16px; font-size: 9px; "
                "border-radius: 2px; font-weight: 650; }"
            )
            if key == "investigate":
                btn.clicked.connect(lambda _checked=False: self._set_chart_type("Investigate Deck"))
            else:
                btn.clicked.connect(lambda _checked=False, template=key: self._toggle_deck_template(template))
            self._template_buttons[key] = btn
            template_grid.addWidget(btn)
        ctrl_l.addLayout(template_grid)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(8)
        form.setVerticalSpacing(4)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)

        self._x_label = QLabel("X")
        self._x_combo = QComboBox()
        form.addRow(self._x_label, self._x_combo)

        self._y_label = QLabel("Y")
        self._y_combo = QComboBox()
        form.addRow(self._y_label, self._y_combo)

        self._y_multi_label = QLabel("Y cols")
        self._y_multi_label.setToolTip("Ctrl+click to select multiple numeric columns.")
        self._y_list = QListWidget()
        self._y_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._y_list.setMinimumHeight(90)
        self._y_list.setMaximumHeight(150)
        form.addRow(self._y_multi_label, self._y_list)

        self._color_label = QLabel("Color")
        self._color_combo = QComboBox()
        form.addRow(self._color_label, self._color_combo)

        self._agg_label = QLabel("Agg")
        self._agg_combo = QComboBox()
        for a in _AGGREGATIONS:
            self._agg_combo.addItem(a)
        form.addRow(self._agg_label, self._agg_combo)

        self._bins_label = QLabel("Bins")
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(2, 500)
        self._bins_spin.setValue(30)
        form.addRow(self._bins_label, self._bins_spin)

        self._top_n_label = QLabel("Top N")
        self._top_n_spin = QSpinBox()
        self._top_n_spin.setRange(2, 200)
        self._top_n_spin.setValue(20)
        form.addRow(self._top_n_label, self._top_n_spin)

        self._sample_label = QLabel("Sample")
        self._sample_spin = QSpinBox()
        self._sample_spin.setRange(100, 200_000)
        self._sample_spin.setValue(10_000)
        self._sample_spin.setSingleStep(1000)
        form.addRow(self._sample_label, self._sample_spin)

        ctrl_l.addLayout(form)

        ctrl_l.addSpacing(2)
        self._col_summary = QLabel("")
        self._col_summary.setFont(_mono_font(9))
        self._col_summary.setWordWrap(True)
        self._col_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._col_summary.setStyleSheet(
            "color: rgba(168,180,172,0.88); padding: 6px 8px; "
            "border: 1px solid rgba(90,104,98,0.62); border-radius: 3px; "
            "background: rgba(10,14,12,0.88);"
        )
        self._col_summary.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self._col_summary.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        ctrl_l.addWidget(self._col_summary)

        self._quality_summary = QLabel("")
        self._quality_summary.setFont(_mono_font(9))
        self._quality_summary.setWordWrap(True)
        self._quality_summary.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._quality_summary.setStyleSheet(
            "color: rgba(232,237,233,0.92); padding: 7px 8px; "
            "border: 1px solid rgba(122,232,96,0.35); border-radius: 3px; "
            "background: rgba(122,232,96,0.08);"
        )
        ctrl_l.addWidget(self._quality_summary)

        self._cell_space_btn = QPushButton("CELL SPACE")
        self._cell_space_btn.setCheckable(True)
        self._cell_space_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cell_space_btn.setToolTip("Toggle Cell Space panel.")
        self._cell_space_btn.clicked.connect(self._on_cell_space_btn_toggled)
        ctrl_l.addWidget(self._cell_space_btn)

        ctrl_l.addStretch(1)

        # ---- RIGHT: canvas area ----
        canvas_container = QWidget()
        canvas_container.setMinimumWidth(480)
        canvas_l = QVBoxLayout(canvas_container)
        canvas_l.setContentsMargins(8, 6, 8, 6)
        canvas_l.setSpacing(6)
        main_splitter.addWidget(canvas_container)
        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([280, 1060])

        # -- Header bar: context text + export --
        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        self._context_label = QLabel("No dataset loaded")
        self._context_label.setFont(_mono_font(9))
        self._context_label.setStyleSheet(
            "color: rgba(168,180,172,0.92); padding: 3px 4px;"
        )
        self._context_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        header_row.addWidget(self._context_label, stretch=1)

        self._export_btn = QToolButton()
        self._export_btn.setText("Export…")
        self._export_btn.setToolTip("Export the current figure as PNG / SVG / PDF.")
        self._export_btn.setAutoRaise(True)
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._export)
        header_row.addWidget(self._export_btn)

        canvas_l.addLayout(header_row)

        self._profile_strip = QHBoxLayout()
        self._profile_strip.setContentsMargins(0, 0, 0, 0)
        self._profile_strip.setSpacing(6)
        self._profile_cards: dict[str, QLabel] = {}
        for key, label in (
            ("rows", "ROWS"),
            ("cols", "COLS"),
            ("num", "NUM"),
            ("cat", "CAT"),
            ("missing", "MISS"),
            ("health", "HEALTH"),
        ):
            card = QLabel(f"{label}\n-")
            card.setFont(_mono_font(9))
            card.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            card.setStyleSheet(
                "QLabel { color: rgba(232,237,233,0.92); padding: 5px 8px; "
                "border: 1px solid rgba(90,104,98,0.62); border-radius: 3px; "
                "background: rgba(10,14,12,0.90); min-width: 72px; }"
            )
            self._profile_cards[key] = card
            self._profile_strip.addWidget(card)
        self._profile_strip.addStretch(1)
        canvas_l.addLayout(self._profile_strip)

        # -- Chart type chip row --
        chip_row = QHBoxLayout()
        chip_row.setContentsMargins(0, 0, 0, 0)
        chip_row.setSpacing(4)
        self._chart_group = QButtonGroup(self)
        self._chart_group.setExclusive(True)
        self._chart_buttons: dict[str, QPushButton] = {}
        _chip_qss = (
            "QPushButton { padding: 3px 10px; min-height: 16px; font-size: 10px; "
            "letter-spacing: 0.5px; border-radius: 2px; font-weight: 600; }"
        )
        for i, ct in enumerate(_CHART_TYPES):
            btn = QPushButton(ct)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setStyleSheet(_chip_qss)
            self._chart_group.addButton(btn, i)
            chip_row.addWidget(btn)
            self._chart_buttons[ct] = btn
        chip_row.addStretch(1)
        canvas_l.addLayout(chip_row)

        # -- Placeholder (empty state only) --
        self._placeholder = QLabel(
            "[DATASET INVESTIGATOR]\n\n"
            "Load a CSV to begin.\n"
            "Investigate Deck mirrors the in-app explorer style."
        )
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.setStyleSheet(
            "color: rgba(147,161,161,0.38); font-size: 13px; "
            "letter-spacing: 0.4px; line-height: 1.6em;"
        )
        canvas_l.addWidget(self._placeholder, stretch=1)

        # -- Matplotlib figure --
        self._fig = Figure(figsize=(11, 7), facecolor=_fig_bg(), constrained_layout=True)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setVisible(False)
        self._toolbar = NavigationToolbar2QT(self._canvas, canvas_container)
        self._toolbar.setVisible(False)
        canvas_l.addWidget(self._toolbar)
        canvas_l.addWidget(self._canvas, stretch=1)

        # -- Debounced auto-plot --
        self._plot_timer = QTimer(self)
        self._plot_timer.setSingleShot(True)
        self._plot_timer.setInterval(180)
        self._plot_timer.timeout.connect(self._plot)

        # Default chart selection
        self._chart_buttons[_CHART_TYPES[0]].setChecked(True)
        if "investigate" in self._template_buttons:
            self._template_buttons["investigate"].setChecked(True)
        for key in self._deck_template_order:
            if key in self._template_buttons:
                self._template_buttons[key].setChecked(True)

        # Wiring — auto-plot on any relevant change
        self._chart_group.idToggled.connect(self._on_chart_type_toggled)
        self._x_combo.currentTextChanged.connect(self._on_selection_changed)
        self._y_combo.currentTextChanged.connect(self._on_selection_changed)
        self._color_combo.currentTextChanged.connect(self._schedule_plot)
        self._y_list.itemSelectionChanged.connect(self._schedule_plot)
        self._agg_combo.currentTextChanged.connect(self._schedule_plot)
        self._bins_spin.valueChanged.connect(self._schedule_plot)
        self._top_n_spin.valueChanged.connect(self._schedule_plot)
        self._sample_spin.valueChanged.connect(self._schedule_plot)

        # Initial visibility
        self._apply_chart_type(self._current_chart_type())
        self._update_context_label()
        self._update_column_summary()
        self._update_profile_view()

    # -- Public --

    def refresh_from(
        self,
        headers: list[str],
        rows: list[list[str]],
        source: str = "",
    ) -> None:
        self._headers = list(headers)
        self._all_rows = [list(r) for r in rows]
        self._source_name = (source or "").strip()
        self._profile = self._compute_profile()
        self._update_combos()
        self._update_context_label()
        self._update_column_summary()
        self._update_profile_view()
        if headers:
            self._schedule_plot()
        else:
            self._show_placeholder()

    # -- Control wiring --

    def _update_combos(self) -> None:
        cols = list(self._headers)
        numeric_cols = [c for c in cols if self._col_is_numeric(c)]

        def _refill(combo: QComboBox, items: list[str], *, blank: bool = False) -> None:
            prev = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            if blank:
                combo.addItem("(none)", "")
            for it in items:
                combo.addItem(it)
            idx = combo.findText(prev)
            combo.setCurrentIndex(max(0, idx))
            combo.blockSignals(False)

        _refill(self._x_combo, cols)
        _refill(self._y_combo, numeric_cols if numeric_cols else cols)
        _refill(self._color_combo, cols, blank=True)

        self._y_list.clear()
        for c in numeric_cols if numeric_cols else cols:
            self._y_list.addItem(QListWidgetItem(c))
        if self._y_list.count():
            self._y_list.item(0).setSelected(True)

    def _compute_profile(self) -> dict[str, Any]:
        if not self._headers:
            return {}
        rows = len(self._all_rows)
        columns: list[dict[str, Any]] = []
        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        missing_total = 0
        for col in self._headers:
            vals = self._col_values(col)
            col_type = _detect_col_type(vals[:500])
            missing = sum(1 for v in vals if not str(v).strip())
            unique = len(set(str(v).strip() for v in vals if str(v).strip()))
            missing_total += missing
            item: dict[str, Any] = {
                "name": col,
                "type": col_type,
                "missing": missing,
                "missing_pct": (missing / rows * 100.0) if rows else 0.0,
                "unique": unique,
            }
            if col_type == "numeric":
                nums = self._col_numeric(col)
                numeric_cols.append(col)
                if nums:
                    item.update({
                        "min": min(nums),
                        "max": max(nums),
                        "mean": sum(nums) / len(nums),
                        "median": sorted(nums)[len(nums) // 2],
                    })
            else:
                categorical_cols.append(col)
                item["top"] = Counter(str(v).strip() for v in vals if str(v).strip()).most_common(3)
            columns.append(item)
        cells = max(1, rows * len(self._headers))
        missing_pct = missing_total / cells * 100.0
        duplicate_count = 0
        try:
            duplicate_count = rows - len({tuple(r) for r in self._all_rows})
        except Exception:
            duplicate_count = 0
        health = max(0.0, 1.0 - (missing_pct / 100.0) - min(0.25, duplicate_count / max(1, rows) * 2.0))
        return {
            "rows": rows,
            "cols": len(self._headers),
            "numeric_cols": numeric_cols,
            "categorical_cols": categorical_cols,
            "columns": columns,
            "missing_pct": missing_pct,
            "duplicates": duplicate_count,
            "health": health,
        }

    def _update_profile_view(self) -> None:
        profile = self._profile or {}
        if not profile:
            for key, card in self._profile_cards.items():
                labels = {"rows": "ROWS", "cols": "COLS", "num": "NUM", "cat": "CAT", "missing": "MISS", "health": "HEALTH"}
                card.setText(f"{labels.get(key, key.upper())}\n-")
            self._quality_summary.setText("Load a CSV to see dataset health, missingness, and feature recommendations.")
            return

        numeric = len(profile.get("numeric_cols") or [])
        categorical = len(profile.get("categorical_cols") or [])
        missing_pct = float(profile.get("missing_pct") or 0.0)
        health = float(profile.get("health") or 0.0)
        values = {
            "rows": f"ROWS\n{int(profile.get('rows') or 0):,}",
            "cols": f"COLS\n{int(profile.get('cols') or 0):,}",
            "num": f"NUM\n{numeric}",
            "cat": f"CAT\n{categorical}",
            "missing": f"MISS\n{missing_pct:.1f}%",
            "health": f"HEALTH\n{health * 100:.0f}%",
        }
        for key, text in values.items():
            self._profile_cards[key].setText(text)

        issues: list[str] = []
        if missing_pct > 10:
            issues.append(f"missing data {missing_pct:.1f}%")
        dupes = int(profile.get("duplicates") or 0)
        if dupes:
            issues.append(f"{dupes:,} duplicate rows")
        if numeric < 2:
            issues.append("scatter/correlation need at least 2 numeric columns")
        strongest = self._strongest_correlations(limit=1)
        if strongest:
            a, b, corr = strongest[0]
            issues.append(f"strongest relation {a} / {b}: r={corr:.2f}")
        self._quality_summary.setText(" | ".join(issues) if issues else "Dataset profile is clean enough for exploratory visualization.")

    def _set_chart_type(self, chart_type: str) -> None:
        btn = self._chart_buttons.get(chart_type)
        if btn is not None:
            btn.setChecked(True)

    def _set_combo_text(self, combo: QComboBox, text: str) -> None:
        idx = combo.findText(text)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def _select_y_columns(self, columns: list[str]) -> None:
        wanted = set(columns)
        self._y_list.blockSignals(True)
        for i in range(self._y_list.count()):
            item = self._y_list.item(i)
            item.setSelected(item.text() in wanted)
        self._y_list.blockSignals(False)

    def _apply_template(self, template: str) -> None:
        if not self._headers:
            return
        numeric_cols = list((self._profile or {}).get("numeric_cols") or [c for c in self._headers if self._col_is_numeric(c)])
        categorical_cols = list((self._profile or {}).get("categorical_cols") or [c for c in self._headers if not self._col_is_numeric(c)])
        first_num = numeric_cols[0] if numeric_cols else (self._headers[0] if self._headers else "")
        second_num = numeric_cols[1] if len(numeric_cols) > 1 else first_num
        first_cat = categorical_cols[0] if categorical_cols else (self._headers[0] if self._headers else "")

        if template == "investigate":
            self._set_chart_type("Investigate Deck")
        elif template == "distribution":
            self._set_chart_type("Histogram")
            self._set_combo_text(self._y_combo, first_num)
        elif template == "correlation":
            self._set_chart_type("Correlation Heatmap")
        elif template == "scatter":
            self._set_chart_type("Scatter")
            self._set_combo_text(self._x_combo, first_num)
            self._set_combo_text(self._y_combo, second_num)
        elif template == "categorical":
            self._set_chart_type("Bar")
            self._set_combo_text(self._x_combo, first_cat)
            self._agg_combo.setCurrentText("Count")
        elif template == "missing":
            self._set_chart_type("Missing Data")
        elif template == "stats":
            self._set_chart_type("Summary Table")
        self._update_column_summary()
        self._schedule_plot()

    def _toggle_deck_template(self, template: str) -> None:
        if template not in self._deck_template_order:
            return
        btn = self._template_buttons.get(template)
        checked = bool(btn.isChecked()) if btn is not None else False
        active = list(self._deck_active_templates)
        if checked and template not in active:
            active.append(template)
        elif not checked and template in active:
            active = [item for item in active if item != template]
        if not active:
            active = [template]
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(True)
                btn.blockSignals(False)
        # preserve stable ordering for deterministic grid placement
        self._deck_active_templates = [item for item in self._deck_template_order if item in active]
        self._set_chart_type("Investigate Deck")
        self._update_column_summary()
        self._schedule_plot()

    def set_deck_templates(self, templates: list[str]) -> None:
        wanted = [item for item in templates if item in self._deck_template_order]
        if not wanted:
            wanted = list(self._deck_template_order)
        self._deck_active_templates = [item for item in self._deck_template_order if item in wanted]
        for key in self._deck_template_order:
            btn = self._template_buttons.get(key)
            if btn is None:
                continue
            btn.blockSignals(True)
            btn.setChecked(key in self._deck_active_templates)
            btn.blockSignals(False)
        self._set_chart_type("Investigate Deck")
        self._schedule_plot()

    def set_feature_focus(self, feature_name: str) -> None:
        if not feature_name:
            return
        self._set_combo_text(self._y_combo, feature_name)
        if self._current_chart_type() == "Investigate Deck":
            self._schedule_plot()

    def _on_cell_space_btn_toggled(self, checked: bool) -> None:
        self._cell_space_open = bool(checked)
        self.cellSpaceToggleRequested.emit(self._cell_space_open)

    def set_cell_space_checked(self, enabled: bool) -> None:
        value = bool(enabled)
        self._cell_space_open = value
        self._cell_space_btn.blockSignals(True)
        self._cell_space_btn.setChecked(value)
        self._cell_space_btn.blockSignals(False)

    def set_controls_visible(self, visible: bool) -> None:
        try:
            self._ctrl_scroll.setVisible(bool(visible))
        except Exception:
            pass

    def _current_chart_type(self) -> str:
        for ct, btn in self._chart_buttons.items():
            if btn.isChecked():
                return ct
        return _CHART_TYPES[0]

    def _on_chart_type_toggled(self, id: int, checked: bool) -> None:
        if not checked:
            return
        self._apply_chart_type(_CHART_TYPES[id])
        self._update_column_summary()
        self._update_context_label()
        self._schedule_plot()

    def _on_selection_changed(self, _text: str = "") -> None:
        self._update_column_summary()
        self._schedule_plot()

    def _apply_chart_type(self, ctype: str) -> None:
        is_deck      = ctype == "Investigate Deck"
        is_scatter   = ctype == "Scatter"
        is_line      = ctype == "Line"
        is_bar       = ctype == "Bar"
        is_hist      = ctype == "Histogram"
        is_box       = ctype == "Box Plot"
        is_heatmap   = ctype == "Correlation Heatmap"
        is_missing   = ctype == "Missing Data"
        is_stats     = ctype == "Summary Table"
        is_pie       = ctype == "Pie"

        needs_x        = not is_deck and not is_hist and not is_heatmap and not is_missing and not is_stats
        needs_y_single = is_scatter or is_bar or is_box or is_pie or is_hist
        needs_y_multi  = is_line
        needs_color    = is_scatter
        needs_agg      = is_bar or is_box
        needs_bins     = is_hist
        needs_top_n    = is_bar or is_pie
        needs_sample   = is_deck or is_scatter or is_line or is_heatmap

        self._x_label.setVisible(needs_x)
        self._x_combo.setVisible(needs_x)

        self._y_label.setVisible(needs_y_single and not needs_y_multi)
        self._y_combo.setVisible(needs_y_single and not needs_y_multi)

        self._y_multi_label.setVisible(needs_y_multi)
        self._y_list.setVisible(needs_y_multi)

        self._color_label.setVisible(needs_color)
        self._color_combo.setVisible(needs_color)

        self._agg_label.setVisible(needs_agg)
        self._agg_combo.setVisible(needs_agg)

        self._bins_label.setVisible(needs_bins)
        self._bins_spin.setVisible(needs_bins)

        self._top_n_label.setVisible(needs_top_n)
        self._top_n_spin.setVisible(needs_top_n)

        self._sample_label.setVisible(needs_sample)
        self._sample_spin.setVisible(needs_sample)

        # Context label for Y varies by chart
        if is_hist:
            self._y_label.setText("Col")
        elif is_box:
            self._y_label.setText("Y num")
        else:
            self._y_label.setText("Y")

        # Box plot reuses X as group-by
        if is_box:
            self._x_label.setVisible(True)
            self._x_combo.setVisible(True)

    # -- Data helpers --

    def _col_values(self, col_name: str) -> list[str]:
        idx = self._headers.index(col_name) if col_name in self._headers else -1
        if idx < 0:
            return []
        return [row[idx] if idx < len(row) else "" for row in self._all_rows]

    def _col_numeric(self, col_name: str) -> list[float]:
        return [float(v.strip().replace(",", "")) for v in self._col_values(col_name) if _is_numeric(v)]

    def _col_is_numeric(self, col_name: str) -> bool:
        sample = self._col_values(col_name)[:200]
        if not sample:
            return False
        return _detect_col_type(sample) == "numeric"

    def _sample_rows(self, limit: int) -> list[list[str]]:
        rows = self._all_rows
        if len(rows) > limit:
            rows = random.sample(rows, limit)
        return rows

    def _col_values_from(self, rows: list[list[str]], col_name: str) -> list[str]:
        idx = self._headers.index(col_name) if col_name in self._headers else -1
        if idx < 0:
            return []
        return [row[idx] if idx < len(row) else "" for row in rows]

    def _strongest_correlations(self, *, limit: int = 12) -> list[tuple[str, str, float]]:
        numeric_cols = [c for c in self._headers if self._col_is_numeric(c)]
        if len(numeric_cols) < 2:
            return []
        rows = self._sample_rows(min(self._sample_spin.value(), 20_000))
        pairs: list[tuple[str, str, float]] = []
        arrays: dict[str, np.ndarray] = {}
        for col in numeric_cols[:40]:
            ci = self._headers.index(col)
            vals = [
                float(row[ci].replace(",", "")) if ci < len(row) and _is_numeric(row[ci]) else float("nan")
                for row in rows
            ]
            arrays[col] = np.array(vals, dtype=float)
        for i, a_name in enumerate(numeric_cols[:40]):
            for b_name in numeric_cols[i + 1:40]:
                a, b = arrays[a_name], arrays[b_name]
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() < 3:
                    continue
                try:
                    corr = float(np.corrcoef(a[mask], b[mask])[0, 1])
                except Exception:
                    continue
                if corr == corr:
                    pairs.append((a_name, b_name, corr))
        pairs.sort(key=lambda p: abs(p[2]), reverse=True)
        return pairs[:limit]

    # -- Plot dispatch --

    def _schedule_plot(self) -> None:
        if not self._headers:
            return
        self._plot_timer.start()

    def _show_placeholder(self) -> None:
        self._placeholder.setVisible(True)
        self._canvas.setVisible(False)
        self._toolbar.setVisible(False)
        self._export_btn.setEnabled(False)

    def _plot(self) -> None:
        if not self._headers:
            self._show_placeholder()
            return
        ctype = self._current_chart_type()
        self._fig.clear()
        self._fig.set_facecolor(_fig_bg())
        try:
            {
                "Investigate Deck":    self._plot_investigate_deck,
                "Scatter":             self._plot_scatter,
                "Line":                self._plot_line,
                "Bar":                 self._plot_bar,
                "Histogram":           self._plot_histogram,
                "Box Plot":            self._plot_box,
                "Correlation Heatmap": self._plot_heatmap,
                "Missing Data":        self._plot_missing_data,
                "Summary Table":       self._plot_summary_table,
                "Pie":                 self._plot_pie,
            }[ctype]()
        except Exception as exc:
            self._fig.clear()
            ax = self._fig.add_subplot(111)
            _apply_dark_axes(ax)
            ax.text(
                0.5, 0.5, f"[ERROR]\n{exc}",
                transform=ax.transAxes, ha="center", va="center",
                color=cvops_color("accent_alert"), fontsize=10, wrap=True,
            )
            self._last_chart_summary = f"[error] {exc}"
        try:
            self._fig.tight_layout(pad=1.8)
        except Exception:
            pass
        self._canvas.draw()
        self._placeholder.setVisible(False)
        self._canvas.setVisible(True)
        self._toolbar.setVisible(True)
        self._export_btn.setEnabled(True)
        self._update_context_label()

    # -- Individual chart implementations --

    def _plot_investigate_deck(self) -> None:
        profile = self._profile or {}
        numeric_cols = list(profile.get("numeric_cols") or [c for c in self._headers if self._col_is_numeric(c)])
        if not numeric_cols:
            raise ValueError("Investigate Deck needs at least one numeric column.")

        focus_name = self._y_combo.currentText().strip()
        primary = focus_name if focus_name in numeric_cols else numeric_cols[0]
        secondary = numeric_cols[1] if len(numeric_cols) > 1 else primary
        sample_limit = self._sample_spin.value()
        rows = self._sample_rows(sample_limit)
        categorical_cols = list(profile.get("categorical_cols") or [c for c in self._headers if not self._col_is_numeric(c)])

        active = [item for item in self._deck_active_templates if item in self._deck_template_order]
        if not active:
            active = ["distribution"]

        # compact grid that auto-reflows by active template count
        n = len(active)
        ncols = 1 if n <= 1 else 2 if n <= 4 else 3
        nrows = int(math.ceil(n / float(ncols)))

        def _ax(position: int):
            return self._fig.add_subplot(nrows, ncols, position + 1)

        for idx, template in enumerate(active):
            ax = _ax(idx)
            _apply_dark_axes(ax)
            if template == "distribution":
                vals = self._col_numeric(primary)
                if vals:
                    ax.hist(vals, bins=min(44, max(12, self._bins_spin.value())), color=_plot_colors()[0], alpha=0.8, edgecolor=_axes_bg(), linewidth=0.4)
                    mean = float(np.mean(vals))
                    std = float(np.std(vals))
                    ax.text(
                        0.99,
                        1.01,
                        f"m={mean:.3g}  s={std:.3g}",
                        transform=ax.transAxes,
                        ha="right",
                        va="bottom",
                        color=_text_clr(),
                        fontsize=8,
                    )
                else:
                    ax.text(0.5, 0.5, "No numeric values", transform=ax.transAxes, ha="center", va="center", color=_text_clr())
                ax.set_title(f"Distribution  [{primary}]")
            elif template == "correlation":
                strongest = self._strongest_correlations(limit=10)
                if strongest:
                    pairs = [f"{a[:12]}/{b[:12]}" for a, b, _corr in strongest]
                    corr_vals = [corr for _a, _b, corr in strongest]
                    colors = [cvops_color("accent_select") if v >= 0 else cvops_color("accent_alert") for v in corr_vals]
                    ax.barh(range(len(pairs)), corr_vals, color=colors, alpha=0.8)
                    ax.set_yticks(range(len(pairs)))
                    ax.set_yticklabels(pairs, fontsize=7)
                    ax.set_xlim(-1.0, 1.0)
                    ax.invert_yaxis()
                    ax.set_xlabel("r")
                else:
                    ax.text(0.5, 0.5, "No significant correlations", transform=ax.transAxes, ha="center", va="center", color=_text_clr())
                ax.set_title("Correlation")
            elif template == "scatter":
                xi = self._headers.index(primary)
                yi = self._headers.index(secondary)
                xs: list[float] = []
                ys: list[float] = []
                for row in rows:
                    xv = row[xi] if xi < len(row) else ""
                    yv = row[yi] if yi < len(row) else ""
                    if _is_numeric(xv) and _is_numeric(yv):
                        xs.append(float(xv.replace(",", "")))
                        ys.append(float(yv.replace(",", "")))
                if xs and ys:
                    ax.scatter(xs, ys, color=cvops_color("accent_select"), alpha=0.45, s=14)
                else:
                    ax.text(0.5, 0.5, "No paired numeric points", transform=ax.transAxes, ha="center", va="center", color=_text_clr())
                ax.set_xlabel(primary)
                ax.set_ylabel(secondary)
                ax.set_title("Scatter")
            elif template == "categorical":
                cat_col = categorical_cols[0] if categorical_cols else ""
                if cat_col:
                    ctr = Counter(v for v in self._col_values(cat_col) if str(v).strip())
                    top = ctr.most_common(max(3, min(10, self._top_n_spin.value())))
                    labels = [k for k, _ in top]
                    vals = [v for _, v in top]
                    if labels:
                        ax.bar(range(len(labels)), vals, color=cvops_color("accent_active"), alpha=0.72)
                        ax.set_xticks(range(len(labels)))
                        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
                    else:
                        ax.text(0.5, 0.5, "No category values", transform=ax.transAxes, ha="center", va="center", color=_text_clr())
                    ax.set_title(f"Categorical  [{cat_col}]")
                else:
                    ax.text(0.5, 0.5, "No categorical columns", transform=ax.transAxes, ha="center", va="center", color=_text_clr())
                    ax.set_title("Categorical")
            elif template == "missing":
                columns = list(profile.get("columns") or [])
                ranked = sorted(columns, key=lambda c: float(c.get("missing_pct") or 0.0), reverse=True)
                ranked = [c for c in ranked if float(c.get("missing_pct") or 0.0) > 0.0][:10]
                if ranked:
                    labels = [str(c.get("name") or "")[:14] for c in ranked]
                    missing_vals = [float(c.get("missing_pct") or 0.0) for c in ranked]
                    colors = [
                        cvops_color("accent_alert") if v > 50 else cvops_color("accent_warn") if v > 10 else cvops_color("accent_active")
                        for v in missing_vals
                    ]
                    ax.barh(range(len(labels)), missing_vals, color=colors, alpha=0.82)
                    ax.set_yticks(range(len(labels)))
                    ax.set_yticklabels(labels, fontsize=7)
                    ax.invert_yaxis()
                    ax.set_xlabel("Missing %")
                else:
                    ax.text(0.5, 0.5, "No missing data", transform=ax.transAxes, ha="center", va="center", color=cvops_color("accent_select"))
                ax.set_title("Missing")
            elif template == "stats":
                cols = list(profile.get("columns") or [])
                subset = cols[:9]
                rows_text = []
                for col in subset:
                    rows_text.append([
                        str(col.get("name") or "")[:14],
                        f"{float(col.get('mean')):.3g}" if col.get("mean") is not None else "-",
                        f"{float(col.get('missing_pct') or 0.0):.1f}%",
                    ])
                ax.set_axis_off()
                if rows_text:
                    tbl = ax.table(
                        cellText=rows_text,
                        colLabels=["Feature", "Mean", "Miss%"],
                        loc="center",
                        cellLoc="left",
                        colLoc="left",
                    )
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(7)
                    tbl.scale(1.0, 1.2)
                    for (r, _c), cell in tbl.get_celld().items():
                        cell.set_edgecolor(_grid_clr())
                        cell.set_linewidth(0.35)
                        if r == 0:
                            cell.set_facecolor(cvops_color("bg_panel_alt"))
                            cell.get_text().set_color(cvops_color("text_signal"))
                            cell.get_text().set_weight("bold")
                        else:
                            cell.set_facecolor(_axes_bg())
                            cell.get_text().set_color(_text_clr())
                else:
                    ax.text(0.5, 0.5, "No stats available", transform=ax.transAxes, ha="center", va="center", color=_text_clr())
                ax.set_title("Summary Table", color=_text_clr(), pad=10)
            else:
                ax.text(0.5, 0.5, "Template unavailable", transform=ax.transAxes, ha="center", va="center", color=_text_clr())
                ax.set_title(str(template))

        sampled = f" (sampled {sample_limit:,})" if len(self._all_rows) > sample_limit else ""
        self._last_chart_summary = f"investigate deck · {len(active)} template(s) · {len(self._all_rows):,} rows{sampled}"

    def _plot_scatter(self) -> None:
        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()
        color_col = self._color_combo.currentData() or self._color_combo.currentText()
        if color_col == "(none)" or not color_col:
            color_col = None
        if not x_col or not y_col:
            raise ValueError("Select X and Y columns.")

        limit = self._sample_spin.value()
        rows = self._sample_rows(limit)
        xi = self._headers.index(x_col)
        yi = self._headers.index(y_col)

        pairs = [
            (row[xi] if xi < len(row) else "", row[yi] if yi < len(row) else "")
            for row in rows
        ]
        x_vals, y_vals = [], []
        for xv, yv in pairs:
            if _is_numeric(xv) and _is_numeric(yv):
                x_vals.append(float(xv.replace(",", "")))
                y_vals.append(float(yv.replace(",", "")))

        ax = self._fig.add_subplot(111)
        _apply_dark_axes(ax)

        if color_col and color_col in self._headers:
            ci = self._headers.index(color_col)
            cat_vals = [row[ci] if ci < len(row) else "" for row in rows]
            # Only use pairs where numeric
            valid = [(x, y, c) for (xv, yv), x, y, c in zip(pairs, x_vals, y_vals, cat_vals[:len(x_vals)])]
            # Rebuild matched lists
            x_vals2, y_vals2, c_vals = [], [], []
            xi2, yi2, ci2 = self._headers.index(x_col), self._headers.index(y_col), self._headers.index(color_col)
            for row in rows:
                xv = row[xi2] if xi2 < len(row) else ""
                yv = row[yi2] if yi2 < len(row) else ""
                cv = row[ci2] if ci2 < len(row) else ""
                if _is_numeric(xv) and _is_numeric(yv):
                    x_vals2.append(float(xv.replace(",", "")))
                    y_vals2.append(float(yv.replace(",", "")))
                    c_vals.append(cv)
            categories = sorted(set(c_vals))
            colors = _plot_colors()
            color_map = {cat: colors[i % len(colors)] for i, cat in enumerate(categories)}
            for cat in categories:
                xs = [x for x, c in zip(x_vals2, c_vals) if c == cat]
                ys = [y for y, c in zip(y_vals2, c_vals) if c == cat]
                ax.scatter(xs, ys, label=cat, color=color_map[cat], alpha=0.65, s=18)
            ax.legend(fontsize=7, labelcolor=_text_clr(), facecolor=_axes_bg(), edgecolor=_grid_clr())
        else:
            ax.scatter(x_vals, y_vals, color=_plot_colors()[0], alpha=0.55, s=18)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"Scatter: {y_col} vs {x_col}")
        sampled = f"  (sampled {limit:,})" if len(self._all_rows) > limit else ""
        self._last_chart_summary = f"scatter · {len(x_vals):,} valid points{sampled}"

    def _plot_line(self) -> None:
        x_col = self._x_combo.currentText()
        y_items = self._y_list.selectedItems()
        if not y_items:
            raise ValueError("Select at least one Y column in the list.")
        y_cols = [it.text() for it in y_items]

        limit = self._sample_spin.value()
        rows = self._sample_rows(limit)

        ax = self._fig.add_subplot(111)
        _apply_dark_axes(ax)

        xi = self._headers.index(x_col) if x_col in self._headers else -1
        x_raw = [row[xi] if xi >= 0 and xi < len(row) else str(i) for i, row in enumerate(rows)]

        # Try numeric x for proper ordering
        x_numeric = all(_is_numeric(v) for v in x_raw if v)
        if x_numeric:
            x_plot = [float(v.replace(",", "")) if _is_numeric(v) else float("nan") for v in x_raw]
            sort_order = sorted(range(len(x_plot)), key=lambda i: (x_plot[i] != x_plot[i], x_plot[i]))
        else:
            x_plot = x_raw
            sort_order = list(range(len(x_plot)))

        x_sorted = [x_plot[i] for i in sort_order]

        for idx, y_col in enumerate(y_cols):
            yi = self._headers.index(y_col) if y_col in self._headers else -1
            y_raw = [rows[i][yi] if yi >= 0 and yi < len(rows[i]) else "" for i in sort_order]
            y_vals = [float(v.replace(",", "")) if _is_numeric(v) else float("nan") for v in y_raw]
            colors = _plot_colors()
            ax.plot(x_sorted, y_vals, label=y_col, color=colors[idx % len(colors)], linewidth=1.2)

        if len(y_cols) > 1:
            ax.legend(fontsize=7, labelcolor=_text_clr(), facecolor=_axes_bg(), edgecolor=_grid_clr())
        ax.set_xlabel(x_col)
        ax.set_ylabel(", ".join(y_cols) if len(y_cols) == 1 else "Value")
        ax.set_title(f"Line: {', '.join(y_cols)} over {x_col}")
        self._last_chart_summary = f"line · {len(rows):,} rows · {len(y_cols)} series"

    def _plot_bar(self) -> None:
        x_col = self._x_combo.currentText()
        y_col = self._y_combo.currentText()
        agg   = self._agg_combo.currentText()
        top_n = self._top_n_spin.value()
        if not x_col:
            raise ValueError("Select an X column.")

        x_vals = self._col_values(x_col)
        y_vals = self._col_values(y_col) if y_col else []

        # Aggregate
        groups: dict[str, list[float]] = defaultdict(list)
        for i, xv in enumerate(x_vals):
            if not xv.strip():
                continue
            if agg != "Count" and i < len(y_vals) and _is_numeric(y_vals[i]):
                groups[xv].append(float(y_vals[i].replace(",", "")))
            elif agg == "Count":
                groups[xv].append(1.0)

        def _aggregate(vals: list[float]) -> float:
            if not vals:
                return 0.0
            if agg == "Count":   return float(len(vals))
            if agg == "Sum":     return float(sum(vals))
            if agg == "Mean":    return float(sum(vals) / len(vals))
            if agg == "Median":  sv = sorted(vals); return sv[len(sv) // 2]
            if agg == "Min":     return min(vals)
            if agg == "Max":     return max(vals)
            return 0.0

        aggregated = {k: _aggregate(v) for k, v in groups.items()}
        sorted_items = sorted(aggregated.items(), key=lambda kv: -kv[1])[:top_n]
        if not sorted_items:
            raise ValueError("No data after aggregation.")

        labels, heights = zip(*sorted_items)
        colors = [_plot_colors()[i % len(_plot_colors())] for i in range(len(labels))]

        ax = self._fig.add_subplot(111)
        _apply_dark_axes(ax)
        bars = ax.bar(range(len(labels)), heights, color=colors, edgecolor=_axes_bg(), linewidth=0.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_xlabel(x_col)
        ylabel = f"{agg}({y_col})" if agg != "Count" else "Count"
        ax.set_ylabel(ylabel)
        ax.set_title(f"Bar: {ylabel} by {x_col}")
        self._last_chart_summary = f"bar · {len(aggregated):,} categories · top {len(sorted_items)}"

    def _plot_histogram(self) -> None:
        col = self._y_combo.currentText()
        if not col:
            raise ValueError("Select a column.")
        bins = self._bins_spin.value()
        vals = self._col_numeric(col)
        if not vals:
            raise ValueError(f"No numeric values in column '{col}'.")

        ax = self._fig.add_subplot(111)
        _apply_dark_axes(ax)
        colors = _plot_colors()
        n, edges, patches = ax.hist(vals, bins=bins, color=colors[0], edgecolor=_axes_bg(), linewidth=0.4, alpha=0.85)
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.set_title(f"Histogram: {col}  ({len(vals):,} values, {bins} bins)")

        # Overlay a KDE if scipy available
        try:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(vals)
            xs = np.linspace(min(vals), max(vals), 300)
            ax2 = ax.twinx()
            ax2.plot(xs, kde(xs), color=colors[1], linewidth=1.5, label="KDE")
            ax2.set_ylabel("Density", color=colors[1])
            ax2.tick_params(colors=colors[1], labelsize=8)
            ax2.set_facecolor(_axes_bg())
            for spine in ax2.spines.values():
                spine.set_color(_grid_clr())
            ax2.grid(False)
        except ImportError:
            pass

        self._last_chart_summary = f"histogram · {len(vals):,} values · {bins} bins"

    def _plot_box(self) -> None:
        y_col  = self._y_combo.currentText()
        x_col  = self._x_combo.currentText()   # used as group-by
        if not y_col:
            raise ValueError("Select a Y column.")

        y_vals = self._col_values(y_col)
        x_vals = self._col_values(x_col) if x_col and x_col != y_col else None

        ax = self._fig.add_subplot(111)
        _apply_dark_axes(ax)

        if x_vals:
            groups: dict[str, list[float]] = defaultdict(list)
            for xv, yv in zip(x_vals, y_vals):
                if _is_numeric(yv) and xv.strip():
                    groups[xv].append(float(yv.replace(",", "")))
            sorted_keys = sorted(groups.keys(), key=lambda k: -len(groups[k]))[:30]
            data   = [groups[k] for k in sorted_keys]
            labels = list(sorted_keys)
        else:
            numeric = [float(v.replace(",", "")) for v in y_vals if _is_numeric(v)]
            if not numeric:
                raise ValueError(f"No numeric values in '{y_col}'.")
            data   = [numeric]
            labels = [y_col]

        if not any(data):
            raise ValueError("No numeric data to plot.")

        bp = ax.boxplot(
            data, labels=labels, patch_artist=True,
            medianprops=dict(color=_plot_colors()[2], linewidth=2),
            whiskerprops=dict(color=_text_clr()),
            capprops=dict(color=_text_clr()),
            flierprops=dict(marker="o", color=_plot_colors()[3], alpha=0.4, markersize=3),
        )
        colors = _plot_colors()
        for patch, color in zip(bp["boxes"], (colors[i % len(colors)] for i in range(len(data)))):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        if len(labels) > 6:
            ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(y_col)
        group_title = f" grouped by {x_col}" if x_vals else ""
        ax.set_title(f"Box Plot: {y_col}{group_title}")
        self._last_chart_summary = f"box · {len(labels)} group(s)"

    def _plot_heatmap(self) -> None:
        numeric_cols = [c for c in self._headers if self._col_is_numeric(c)]
        if len(numeric_cols) < 2:
            raise ValueError("Need at least 2 numeric columns for a correlation heatmap.")
        limit = self._sample_spin.value()
        rows = self._sample_rows(limit)

        # Build matrix
        col_indices = [self._headers.index(c) for c in numeric_cols]
        arrays = []
        for ci in col_indices:
            arrays.append([
                float(row[ci].replace(",", "")) if ci < len(row) and _is_numeric(row[ci]) else float("nan")
                for row in rows
            ])
        mat = np.array(arrays, dtype=float)   # shape (n_cols, n_rows)

        # Correlation (ignoring NaN pairs column-wise)
        n = len(numeric_cols)
        corr = np.full((n, n), float("nan"))
        for i in range(n):
            for j in range(n):
                a, b = mat[i], mat[j]
                mask = ~(np.isnan(a) | np.isnan(b))
                if mask.sum() >= 2:
                    try:
                        corr[i, j] = float(np.corrcoef(a[mask], b[mask])[0, 1])
                    except Exception:
                        pass

        ax = self._fig.add_subplot(111)
        _apply_dark_axes(ax)
        im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        cbar = self._fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.ax.tick_params(colors=_text_clr(), labelsize=7)
        cbar.ax.yaxis.label.set_color(_text_clr())

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        fontsize = max(5, min(9, int(120 / n)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right", fontsize=fontsize)
        ax.set_yticklabels(numeric_cols, fontsize=fontsize)

        # Annotate cells (only if not too many)
        if n <= 20:
            for i, j in _iproduct(range(n), range(n)):
                v = corr[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=max(5, fontsize - 1),
                            color=cvops_color("text_bright") if abs(v) > 0.6 else _text_clr())

        ax.set_title(f"Correlation Heatmap ({n} numeric columns)")
        self._last_chart_summary = f"heatmap · {n} numeric cols · {len(rows):,} rows"

    def _plot_missing_data(self) -> None:
        columns = list((self._profile or {}).get("columns") or [])
        rows = int((self._profile or {}).get("rows") or len(self._all_rows))
        ranked = sorted(columns, key=lambda c: float(c.get("missing_pct") or 0.0), reverse=True)
        ranked = [c for c in ranked if float(c.get("missing_pct") or 0.0) > 0.0][:30]

        ax = self._fig.add_subplot(111)
        _apply_dark_axes(ax)
        if not ranked:
            ax.text(
                0.5, 0.5, "No missing values detected",
                transform=ax.transAxes, ha="center", va="center",
                color=cvops_color("accent_active"), fontsize=11,
            )
            ax.set_axis_off()
            self._last_chart_summary = f"missing · 0 missing cells · {rows:,} rows"
            return

        labels = [str(c.get("name") or "")[:28] for c in ranked]
        values = [float(c.get("missing_pct") or 0.0) for c in ranked]
        colors = [
            cvops_color("accent_alert") if v > 50 else cvops_color("accent_warn") if v > 10 else cvops_color("accent_select")
            for v in values
        ]
        ax.barh(range(len(labels)), values, color=colors, alpha=0.78)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlim(0, max(100.0, max(values) * 1.08))
        ax.set_xlabel("Missing %")
        ax.set_title("Missing Data by Feature")
        self._last_chart_summary = f"missing · {len(ranked)} affected columns · {rows:,} rows"

    def _plot_summary_table(self) -> None:
        columns = list((self._profile or {}).get("columns") or [])
        if not columns:
            raise ValueError("No feature profile available.")
        ranked = columns[:18]
        headers = ["Feature", "Type", "Unique", "Missing", "Mean", "Min", "Max"]
        table_rows: list[list[str]] = []
        for col in ranked:
            missing_pct = float(col.get("missing_pct") or 0.0)
            table_rows.append([
                str(col.get("name") or "")[:28],
                str(col.get("type") or ""),
                f"{int(col.get('unique') or 0):,}",
                f"{missing_pct:.1f}%",
                f"{float(col.get('mean')):.4g}" if col.get("mean") is not None else "-",
                f"{float(col.get('min')):.4g}" if col.get("min") is not None else "-",
                f"{float(col.get('max')):.4g}" if col.get("max") is not None else "-",
            ])

        ax = self._fig.add_subplot(111)
        ax.set_facecolor(_fig_bg())
        ax.set_axis_off()
        table = ax.table(
            cellText=table_rows,
            colLabels=headers,
            loc="center",
            cellLoc="left",
            colLoc="left",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.35)
        for (row, _col), cell in table.get_celld().items():
            cell.set_edgecolor(_grid_clr())
            cell.set_linewidth(0.35)
            if row == 0:
                cell.set_facecolor(cvops_color("bg_panel_alt"))
                cell.get_text().set_color(cvops_color("text_signal"))
                cell.get_text().set_weight("bold")
            else:
                cell.set_facecolor(_axes_bg())
                cell.get_text().set_color(_text_clr())
        ax.set_title("Feature Summary", color=_text_clr(), pad=12)
        self._last_chart_summary = f"stats · {len(columns)} profiled columns"

    def _plot_pie(self) -> None:
        x_col = self._x_combo.currentText()
        top_n = self._top_n_spin.value()
        if not x_col:
            raise ValueError("Select an X column.")

        vals = self._col_values(x_col)
        ctr  = Counter(v for v in vals if v.strip())
        if not ctr:
            raise ValueError(f"No values in column '{x_col}'.")

        top = ctr.most_common(top_n)
        others = sum(v for _, v in ctr.most_common()[top_n:])

        labels = [k for k, _ in top]
        sizes  = [v for _, v in top]
        if others:
            labels.append("(other)")
            sizes.append(others)

        plot_colors = _plot_colors()
        colors = [plot_colors[i % len(plot_colors)] for i in range(len(labels))]

        ax = self._fig.add_subplot(111)
        ax.set_facecolor(_fig_bg())
        ax.pie(
            sizes, labels=labels, colors=colors, autopct="%1.1f%%",
            pctdistance=0.82, startangle=90,
            textprops={"color": _text_clr(), "fontsize": 8},
            wedgeprops={"edgecolor": _fig_bg(), "linewidth": 1},
        )
        ax.set_title(f"Pie: {x_col}  (top {len(top)} of {len(ctr):,} categories)", color=_text_clr())
        self._last_chart_summary = f"pie · {len(top)} slices · {len(ctr):,} unique values"

    # -- Quick stats --

    # -- Header + summary --

    def _update_context_label(self) -> None:
        if not self._headers:
            self._context_label.setText("No dataset loaded")
            return
        rows = len(self._all_rows)
        cols = len(self._headers)
        src = self._source_name or "in-memory data"
        chart = self._last_chart_summary or self._current_chart_type().lower()
        self._context_label.setText(
            f"{src}  ·  {rows:,} rows  ·  {cols} cols  ·  {chart}"
        )

    def _update_column_summary(self) -> None:
        if not self._headers:
            self._col_summary.setText(
                "No dataset loaded.\nLoad a CSV to see column types and stats here."
            )
            return

        lines: list[str] = []
        seen: set[str] = set()
        targets: list[tuple[str, str]] = []
        for label, combo in (("x", self._x_combo), ("y", self._y_combo)):
            if not combo.isVisible():
                continue
            col = combo.currentText().strip()
            if not col or col in seen:
                continue
            seen.add(col)
            targets.append((label, col))

        for label, col in targets:
            lines.extend(self._column_brief(label, col))

        if not targets:
            lines.extend(self._dataset_brief())

        self._col_summary.setText("\n".join(lines))

    def _column_brief(self, label: str, col: str) -> list[str]:
        vals = self._col_values(col)
        if not vals:
            return [f"{label}  {col}", "    (no data)"]
        col_type = _detect_col_type(vals[:500])
        n = len(vals)
        empty = sum(1 for v in vals if not v.strip())
        unique = len(set(v.strip() for v in vals if v.strip()))
        out = [
            f"{label}  {col}",
            f"    {col_type} · {n:,} rows · {unique:,} unique · {empty:,} empty",
        ]
        if col_type == "numeric":
            nums = self._col_numeric(col)
            if nums:
                out.append(
                    f"    min {min(nums):.4g} · max {max(nums):.4g} · "
                    f"mean {sum(nums)/len(nums):.4g}"
                )
        elif unique and unique <= 8:
            top = Counter(v.strip() for v in vals if v.strip()).most_common(4)
            out.append("    top: " + ", ".join(f"{k} ({c})" for k, c in top))
        return out

    def _dataset_brief(self) -> list[str]:
        type_counts: Counter = Counter()
        for col in self._headers:
            type_counts[_detect_col_type(self._col_values(col)[:200])] += 1
        parts = [f"{c} {t}" for t, c in type_counts.most_common()]
        return [
            f"{len(self._headers)} columns · {len(self._all_rows):,} rows",
            "    " + ", ".join(parts) if parts else "    (no columns)",
        ]

    def refresh_theme_styles(self) -> None:
        self._fig.set_facecolor(_fig_bg())
        if self._headers and self._canvas.isVisible():
            self._plot()
        else:
            self._canvas.draw_idle()

    # -- Export --

    def _export(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Figure", "", "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)"
        )
        if not dest:
            return
        try:
            self._fig.savefig(dest, facecolor=_fig_bg(), bbox_inches="tight", dpi=150)
            self._last_chart_summary = f"saved {Path(dest).name}"
            self._update_context_label()
        except Exception as exc:
            QMessageBox.warning(self, "Export Failed", str(exc))


# ---------------------------------------------------------------------------
# Dialog (public entry point — backward-compatible with old signature)
# ---------------------------------------------------------------------------

class CsvTableEditorDialog(QDialog):
    """Comprehensive CSV dataset editor with folder browser and full table editing.

    Args:
        csv_path:    Open a specific CSV file directly in the editor tab.
        open_folder: Pre-load a folder in the browser tab.
        max_rows:    Row cap before the file is considered too large to save.
    """

    def __init__(
        self,
        *,
        csv_path: Optional[Path] = None,
        open_folder: Optional[Path] = None,
        parent: Optional[QWidget] = None,
        max_rows: int = 50_000,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("CSV Dataset Editor")
        self.resize(1340, 840)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, stretch=1)

        self._browser = _FolderBrowserWidget()
        self._editor  = _CsvEditorWidget()
        self._viz     = _VisualizationWidget()
        self._editor.max_rows = int(max_rows)

        self._tabs.addTab(self._browser, "[FOLDER]")
        self._tabs.addTab(self._editor,  "[EDITOR]")
        self._tabs.addTab(self._viz,     "[VISUALIZE]")

        self._browser.fileOpenRequested.connect(self._on_open_file)
        self._browser.fileMergeRequested.connect(self._on_merge_files)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        if csv_path:
            p = Path(csv_path).resolve()
            self._editor.load(p)
            self._tabs.setCurrentWidget(self._editor)
            if p.parent.is_dir():
                self._browser.load_folder(p.parent)
            self._set_context_window_title("Editor")
        elif open_folder:
            self._browser.load_folder(Path(open_folder))

    def show_visualization(self) -> None:
        """Switch to the visualization workspace and refresh it from the editor data."""
        if self._tabs.currentWidget() is self._viz:
            self._refresh_visualization()
        else:
            self._tabs.setCurrentWidget(self._viz)
        self._set_context_window_title("Visualizer")

    # -- Slots --

    def _on_open_file(self, path: object) -> None:
        self._editor.load(Path(str(path)))
        self._tabs.setCurrentWidget(self._editor)
        self._set_context_window_title("Editor")

    def _on_merge_files(self, paths: object) -> None:
        self._editor.load_merged([Path(str(p)) for p in paths])  # type: ignore[arg-type]
        self._tabs.setCurrentWidget(self._editor)
        self._set_context_window_title("Editor")

    def _on_tab_changed(self, index: int) -> None:
        if self._tabs.widget(index) is self._viz:
            self._refresh_visualization()
            self._set_context_window_title("Visualizer")
        elif self._tabs.widget(index) is self._editor:
            self._set_context_window_title("Editor")
        else:
            self._set_context_window_title("Browser")

    def _refresh_visualization(self) -> None:
        headers, rows = self._editor.get_data_snapshot()
        src = ""
        path = getattr(self._editor, "_path", None)
        if path is not None:
            try:
                src = Path(path).name
            except Exception:
                src = ""
        self._viz.refresh_from(headers, rows, source=src)

    def _set_context_window_title(self, role: str) -> None:
        label = str(role or "Editor").strip() or "Editor"
        source = ""
        path = getattr(self._editor, "_path", None)
        if path is not None:
            try:
                source = Path(path).name
            except Exception:
                source = ""
        if source:
            self.setWindowTitle(f"CSV Dataset {label} - {source}")
        else:
            self.setWindowTitle(f"CSV Dataset {label}")


# Public alias — importable by other panels without using the private name.
CsvVisualizationWidget = _VisualizationWidget
