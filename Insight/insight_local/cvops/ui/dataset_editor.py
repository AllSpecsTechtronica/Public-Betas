from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QInputDialog,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class _EvenDatasetWorker(QThread):
    """Background worker to balance train/val and class folders via augmentation."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, base_url: str, slug: str, folders: Optional[list[str]] = None) -> None:
        super().__init__()
        self.base_url = base_url
        self.slug = slug
        self.folders = [str(f or "").strip("/") for f in (folders or []) if str(f or "").strip("/")]
        self._http_timeout = 600.0

    def run(self) -> None:
        try:
            enc_slug = urllib.parse.quote(self.slug, safe="")
            url = self.base_url.rstrip("/") + f"/database/{enc_slug}/even_dataset"
            data = json.dumps({"folders": self.folders}).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(url, data=data, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=self._http_timeout) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            self.finished.emit(payload)
        except Exception as exc:
            self.error.emit(str(exc))


class _AutoAugmentWorker(QThread):
    """Background worker for auto augmentation with progress reporting."""
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(
        self,
        base_url: str,
        slug: str,
        goal: int,
        folders: Optional[list[str]] = None,
        *,
        val_frac: float = 0.2,
        ensure_val: bool = True,
    ) -> None:
        super().__init__()
        self.base_url = base_url
        self.slug = slug
        self.goal = goal
        self.folders = [str(f or "").strip("/") for f in (folders or []) if str(f or "").strip("/")]
        self.val_frac = val_frac
        self.ensure_val = ensure_val
        self._http_timeout = 600.0

    def run(self) -> None:
        try:
            enc_slug = urllib.parse.quote(self.slug, safe="")
            url = self.base_url.rstrip("/") + f"/database/{enc_slug}/auto_augment"
            data = json.dumps(
                {
                    "target_total": self.goal,
                    "folders": self.folders,
                    "val_frac": self.val_frac,
                    "ensure_val": self.ensure_val,
                }
            ).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            req = urllib.request.Request(url, data=data, method="POST", headers=headers)
            with urllib.request.urlopen(req, timeout=self._http_timeout) as resp:
                raw = resp.read().decode("utf-8")
            payload = json.loads(raw) if raw else {}
            self.finished.emit(payload)
        except Exception as exc:
            self.error.emit(str(exc))


def _mono_font(size: int = 11) -> QFont:
    font = QFont("JetBrains Mono", size)
    if not font.exactMatch():
        font = QFont("IBM Plex Mono", size)
    font.setStyleHint(QFont.StyleHint.Monospace)
    return font


def _human_bytes(n: int) -> str:
    try:
        val = float(int(n))
    except Exception:
        return ""
    if val <= 0:
        return ""
    units = ["B", "KB", "MB", "GB", "TB"]
    u = 0
    while val >= 1024.0 and u < len(units) - 1:
        val /= 1024.0
        u += 1
    if u == 0:
        return f"{int(val)} {units[u]}"
    return f"{val:.2f} {units[u]}"


def _image_folder_from_relpath(rel_path: str) -> str:
    rel = str(rel_path or "").strip().replace("\\", "/").strip("/")
    parts = [p for p in rel.split("/") if p]
    if len(parts) >= 3 and parts[0].lower() == "images":
        folder = "/".join(parts[2:-1])
    else:
        folder = "/".join(parts[:-1])
    return folder or "(root)"


def _entry_subfolder_key(entry: dict[str, Any]) -> str:
    folder = _image_folder_from_relpath(_entry_relpath(entry))
    return "" if folder == "(root)" else folder.lower()


def _normalize_folder_scope(folder_scope: Optional[set[str]]) -> Optional[set[str]]:
    if folder_scope is None:
        return None
    return {"" if f == "(root)" else f for f in folder_scope}


def _estimate_even_dataset_adds(
    entries: list[dict[str, Any]],
    folder_scope: Optional[set[str]] = None,
) -> tuple[int, int, dict[str, int]]:
    """Return target per bucket, total augmentations needed, and counts before."""
    scope = _normalize_folder_scope(folder_scope)
    scoped = [
        e for e in entries
        if str(e.get("split") or "").lower() in {"train", "val"}
        and (scope is None or _entry_subfolder_key(e) in scope)
    ]
    buckets: dict[tuple[str, str], int] = {}
    for entry in scoped:
        split_name = str(entry.get("split") or "").strip().lower()
        folder_name = _entry_subfolder_key(entry)
        key = (split_name, folder_name)
        buckets[key] = buckets.get(key, 0) + 1
    folder_names = sorted({folder_name for _split, folder_name in buckets})
    if not folder_names:
        return 0, 0, {}
    full_counts: dict[str, int] = {}
    for split_name in ("train", "val"):
        for folder_name in folder_names:
            count = buckets.get((split_name, folder_name), 0)
            label = f"{split_name}/{folder_name or '(root)'}"
            full_counts[label] = count
    target = max(full_counts.values(), default=0)
    total_add = sum(max(0, target - count) for count in full_counts.values())
    return target, total_add, full_counts


def _entry_relpath(entry: dict[str, Any]) -> str:
    return str(entry.get("relative_path") or entry.get("name") or "").strip()


def _folder_name_for_ext(ext: str) -> str:
    e = str(ext or "").strip().lower()
    if not e or e == "(none)":
        return "none"
    if e.startswith("."):
        e = e[1:]
    out = "".join(c for c in e if c.isalnum() or c in ("-", "_"))[:32]
    return out or "unknown"


class FolderInventoryDialog(QDialog):
    def __init__(
        self,
        *,
        base_url: str,
        dataset_slug: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = str(base_url or "").rstrip("/")
        self._slug = str(dataset_slug or "").strip()
        self._enc_slug = urllib.parse.quote(self._slug, safe="")

        self.setWindowTitle(f"Folder Inventory — {self._slug}")
        self.resize(980, 620)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header = QHBoxLayout()
        title = QLabel(f"Dataset: {self._slug}")
        title.setStyleSheet("font-weight: 700;")
        header.addWidget(title, stretch=1)
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        header.addWidget(self._status)
        self._reload_btn = QPushButton("Refresh")
        self._reload_btn.clicked.connect(self.reload)
        header.addWidget(self._reload_btn)
        outer.addLayout(header)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Subfolder (relative):"))
        self._rel = QLineEdit()
        self._rel.setPlaceholderText("e.g. images/train  (blank = dataset root)")
        controls.addWidget(self._rel, stretch=1)
        self._include_hidden = QCheckBox("Include hidden")
        self._include_hidden.setChecked(False)
        controls.addWidget(self._include_hidden)
        self._preserve_tree = QCheckBox("Preserve paths on move")
        self._preserve_tree.setChecked(True)
        controls.addWidget(self._preserve_tree)
        outer.addLayout(controls)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Ext", "MIME", "Count", "Total size", "Examples"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        try:
            hdr = self._table.horizontalHeader()
            hdr.setStretchLastSection(True)
            hdr.setSectionResizeMode(0, hdr.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(1, hdr.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(2, hdr.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(3, hdr.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(4, hdr.ResizeMode.Stretch)
        except Exception:
            pass
        self._table.verticalHeader().setVisible(False)
        outer.addWidget(self._table, stretch=1)

        actions = QHBoxLayout()
        self._move_btn = QPushButton("Move selected types…")
        self._move_btn.clicked.connect(self._move_selected_types)
        actions.addWidget(self._move_btn)
        self._delete_btn = QPushButton("Delete selected types…")
        self._delete_btn.clicked.connect(self._delete_selected_types)
        actions.addWidget(self._delete_btn)
        actions.addStretch(1)
        outer.addLayout(actions)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.reload()

    def _http_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        url = self._base_url + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _selected_ext_rows(self) -> list[tuple[str, int]]:
        rows = sorted({idx.row() for idx in self._table.selectionModel().selectedRows()})
        out: list[tuple[str, int]] = []
        for r in rows:
            it_ext = self._table.item(r, 0)
            it_count = self._table.item(r, 2)
            if it_ext is None:
                continue
            ext = str(it_ext.text() or "").strip()
            if not ext:
                continue
            try:
                count = int(str(it_count.text() or "0").strip())
            except Exception:
                count = 0
            out.append((ext, count))
        return out

    def reload(self) -> None:
        if not self._slug:
            return
        rel = str(self._rel.text() or "").strip()
        include_hidden = 1 if bool(self._include_hidden.isChecked()) else 0
        q = f"?rel={urllib.parse.quote(rel, safe='')}&include_hidden={include_hidden}"
        try:
            payload = self._http_json("GET", f"/database/{self._enc_slug}/inventory{q}")
        except Exception as exc:
            self._status.setText(f"Load failed: {exc}")
            return

        types = payload.get("types") or []
        if not isinstance(types, list):
            types = []

        self._table.setRowCount(0)
        self._table.setRowCount(len(types))
        for r, t in enumerate(types):
            if not isinstance(t, dict):
                continue
            ext = str(t.get("ext") or "")
            mime = str(t.get("mime") or "")
            count = int(t.get("count") or 0)
            size_b = int(t.get("bytes") or 0)
            ex = t.get("examples") or []
            examples = ", ".join(str(x) for x in ex[:3]) if isinstance(ex, list) else ""

            self._table.setItem(r, 0, QTableWidgetItem(ext))
            self._table.setItem(r, 1, QTableWidgetItem(mime))
            self._table.setItem(r, 2, QTableWidgetItem(str(count)))
            self._table.setItem(r, 3, QTableWidgetItem(_human_bytes(size_b)))
            it4 = QTableWidgetItem(examples)
            it4.setFont(_mono_font(9))
            self._table.setItem(r, 4, it4)

        total_files = int(payload.get("total_files") or 0)
        total_dirs = int(payload.get("total_dirs") or 0)
        total_bytes = int(payload.get("total_bytes") or 0)
        elapsed_ms = int(payload.get("elapsed_ms") or 0)
        truncated = bool(payload.get("truncated"))
        suffix = " (truncated)" if truncated else ""
        self._status.setText(
            f"{total_files} files / {total_dirs} dirs  |  {_human_bytes(total_bytes)}  |  {elapsed_ms} ms{suffix}"
        )

    def _move_selected_types(self) -> None:
        selected = self._selected_ext_rows()
        if not selected:
            return
        total = sum(c for _e, c in selected)
        base_dest, ok = QInputDialog.getText(
            self,
            "Move Types",
            "Destination subfolder (relative to dataset root):",
            text="_quarantine",
        )
        if not ok:
            return
        base_dest = str(base_dest or "").strip().strip("/")
        if not base_dest:
            QMessageBox.warning(self, "Invalid Destination", "Destination subfolder is required.")
            return
        preserve_tree = bool(self._preserve_tree.isChecked())
        rel = str(self._rel.text() or "").strip()
        include_hidden = bool(self._include_hidden.isChecked())

        confirm = QMessageBox.warning(
            self,
            "Move Types",
            f"Move {total} files across {len(selected)} type(s) into '{base_dest}/…'?\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        moved_total = 0
        errors: list[str] = []
        for ext, _count in selected:
            dest = f"{base_dest}/{_folder_name_for_ext(ext)}"
            try:
                payload = self._http_json(
                    "POST",
                    f"/database/{self._enc_slug}/inventory/move_by_ext",
                    {
                        "ext": ext,
                        "dest_relative_dir": dest,
                        "relative_dir": rel,
                        "include_hidden": include_hidden,
                        "preserve_tree": preserve_tree,
                        "dry_run": False,
                    },
                )
                moved_total += int(payload.get("moved") or 0)
                errs = payload.get("errors") or []
                if isinstance(errs, list):
                    errors.extend(str(e) for e in errs[:3])
            except Exception as exc:
                errors.append(f"{ext}: {exc}")

        if errors:
            QMessageBox.warning(self, "Move Errors", errors[0])
        self._status.setText(f"Moved {moved_total}.")
        self.reload()

    def _delete_selected_types(self) -> None:
        selected = self._selected_ext_rows()
        if not selected:
            return
        total = sum(c for _e, c in selected)
        rel = str(self._rel.text() or "").strip()
        include_hidden = bool(self._include_hidden.isChecked())

        confirm = QMessageBox.warning(
            self,
            "Delete Types",
            f"Delete {total} files across {len(selected)} type(s) in '{rel or '.'}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        deleted_total = 0
        errors: list[str] = []
        for ext, _count in selected:
            try:
                payload = self._http_json(
                    "POST",
                    f"/database/{self._enc_slug}/inventory/delete_by_ext",
                    {"ext": ext, "relative_dir": rel, "include_hidden": include_hidden, "dry_run": False},
                )
                deleted_total += int(payload.get("deleted") or 0)
                errs = payload.get("errors") or []
                if isinstance(errs, list):
                    errors.extend(str(e) for e in errs[:3])
            except Exception as exc:
                errors.append(f"{ext}: {exc}")

        if errors:
            QMessageBox.warning(self, "Delete Errors", errors[0])
        self._status.setText(f"Deleted {deleted_total}.")
        self.reload()


class DatasetEditorDialog(QDialog):
    def __init__(
        self,
        *,
        base_url: str,
        dataset_slug: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = str(base_url or "").rstrip("/")
        self._slug = str(dataset_slug or "").strip()
        self._enc_slug = urllib.parse.quote(self._slug, safe="")
        self._entries: list[dict[str, Any]] = []
        self._classes: list[str] = []
        self._aug_poll_timer: Optional[QTimer] = None
        self._auto_aug_start_time: float = 0.0
        self._auto_aug_goal_val: int = 0
        self._auto_aug_current_total: int = 0
        self._aug_worker_thread: Optional[QThread] = None
        self._aug_worker_kind: str = ""

        self.setWindowTitle(f"Dataset Editor — {self._slug}")
        self.resize(1040, 680)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header = QHBoxLayout()
        self._title_label = QLabel(f"Dataset: {self._slug}" if self._slug else "Dataset: [NONE SELECTED]")
        self._title_label.setStyleSheet("font-weight: 700;")
        header.addWidget(self._title_label, stretch=1)
        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        header.addWidget(self._status)
        self._reload_btn = QPushButton("Reload")
        self._reload_btn.clicked.connect(self.reload)
        header.addWidget(self._reload_btn)
        self._inventory_btn = QPushButton("Inventory")
        self._inventory_btn.clicked.connect(self._open_inventory)
        header.addWidget(self._inventory_btn)
        outer.addLayout(header)

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter = QLineEdit()
        self._filter.setPlaceholderText("type to filter by filename / split / path…")
        self._filter.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter, stretch=1)
        self._selected_label = QLabel("")
        self._selected_label.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        filter_row.addWidget(self._selected_label)
        outer.addLayout(filter_row)

        body = QHBoxLayout()
        body.setSpacing(10)
        outer.addLayout(body, stretch=1)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Item", "Split", "Label", "Size", "Rel path"])
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setSelectionMode(QTableWidget.SelectionMode.ExtendedSelection)
        self._table.itemSelectionChanged.connect(self._sync_selection_status)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        try:
            hdr = self._table.horizontalHeader()
            hdr.setStretchLastSection(True)
            hdr.setSectionResizeMode(0, hdr.ResizeMode.Stretch)
            hdr.setSectionResizeMode(1, hdr.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(2, hdr.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(3, hdr.ResizeMode.ResizeToContents)
            hdr.setSectionResizeMode(4, hdr.ResizeMode.ResizeToContents)
        except Exception:
            pass
        self._table.verticalHeader().setVisible(False)
        body.addWidget(self._table, stretch=1)

        right = QVBoxLayout()
        right.setSpacing(8)
        body.addLayout(right)

        sel_box = QGroupBox("Selection")
        sel_layout = QVBoxLayout(sel_box)
        sel_layout.setContentsMargins(10, 10, 10, 10)
        sel_layout.setSpacing(6)

        move_row = QHBoxLayout()
        move_row.addWidget(QLabel("Move to split:"))
        self._move_split = QComboBox()
        self._move_split.addItem("train", "train")
        self._move_split.addItem("val", "val")
        move_row.addWidget(self._move_split, stretch=1)
        self._move_btn = QPushButton("Move")
        self._move_btn.clicked.connect(self._move_selected)
        move_row.addWidget(self._move_btn)
        sel_layout.addLayout(move_row)

        self._clear_labels_btn = QPushButton("Clear labels (selected)")
        self._clear_labels_btn.clicked.connect(self._clear_labels_selected)
        sel_layout.addWidget(self._clear_labels_btn)

        aug_box = QGroupBox("Augment train -> val")
        aug_layout = QVBoxLayout(aug_box)
        aug_layout.setContentsMargins(8, 8, 8, 8)
        aug_layout.setSpacing(6)

        aug_row1 = QHBoxLayout()
        aug_row1.addWidget(QLabel("Copies"))
        self._aug_copies = QSpinBox()
        self._aug_copies.setRange(1, 50)
        self._aug_copies.setValue(1)
        aug_row1.addWidget(self._aug_copies)
        self._aug_balance = QCheckBox("Balance val to train")
        self._aug_balance.setChecked(True)
        aug_row1.addWidget(self._aug_balance, stretch=1)
        aug_layout.addLayout(aug_row1)

        aug_row2 = QHBoxLayout()
        aug_row2.addWidget(QLabel("Scale %"))
        self._aug_scale = QSpinBox()
        self._aug_scale.setRange(10, 300)
        self._aug_scale.setValue(100)
        aug_row2.addWidget(self._aug_scale)
        aug_row2.addWidget(QLabel("Angle"))
        self._aug_angle = QDoubleSpinBox()
        self._aug_angle.setRange(-180.0, 180.0)
        self._aug_angle.setDecimals(1)
        self._aug_angle.setSingleStep(5.0)
        self._aug_angle.setValue(0.0)
        aug_row2.addWidget(self._aug_angle)
        aug_layout.addLayout(aug_row2)

        aug_row3 = QHBoxLayout()
        aug_row3.addWidget(QLabel("Quality"))
        self._aug_quality = QSpinBox()
        self._aug_quality.setRange(1, 100)
        self._aug_quality.setValue(90)
        aug_row3.addWidget(self._aug_quality)
        self._aug_gray = QCheckBox("Grayscale")
        aug_row3.addWidget(self._aug_gray, stretch=1)
        aug_layout.addLayout(aug_row3)

        self._aug_btn = QPushButton("Copy augmented selection to val")
        self._aug_btn.clicked.connect(self._copy_augmented_selected_to_val)
        aug_layout.addWidget(self._aug_btn)

        seed_val_row = QHBoxLayout()
        seed_val_row.addWidget(QLabel("Val fraction"))
        self._seed_val_frac = QDoubleSpinBox()
        self._seed_val_frac.setRange(0.05, 0.50)
        self._seed_val_frac.setSingleStep(0.05)
        self._seed_val_frac.setDecimals(2)
        self._seed_val_frac.setValue(0.20)
        seed_val_row.addWidget(self._seed_val_frac)
        self._seed_val_btn = QPushButton("Seed val split from train")
        self._seed_val_btn.clicked.connect(self._seed_val_from_train)
        self._seed_val_btn.setVisible(False)
        seed_val_row.addWidget(self._seed_val_btn, stretch=1)
        aug_layout.addLayout(seed_val_row)

        aug_hint = QLabel(
            "Select train images first. Labels are copied; rotated images get approximate rotated boxes."
        )
        aug_hint.setWordWrap(True)
        aug_hint.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        aug_layout.addWidget(aug_hint)
        sel_layout.addWidget(aug_box)

        auto_aug_box = QGroupBox("Auto augment")
        auto_aug_layout = QVBoxLayout(auto_aug_box)
        auto_aug_layout.setContentsMargins(8, 8, 8, 8)
        auto_aug_layout.setSpacing(6)

        auto_goal_row = QHBoxLayout()
        auto_goal_row.addWidget(QLabel("Goal total"))
        self._auto_aug_goal = QSpinBox()
        self._auto_aug_goal.setRange(1, 1_000_000)
        self._auto_aug_goal.setValue(1000)
        auto_goal_row.addWidget(self._auto_aug_goal, stretch=1)
        auto_aug_layout.addLayout(auto_goal_row)

        self._auto_aug_count = QLabel("Current train+val: 0")
        self._auto_aug_count.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        auto_aug_layout.addWidget(self._auto_aug_count)

        auto_val_row = QHBoxLayout()
        self._auto_aug_ensure_val = QCheckBox("Create val split if missing")
        self._auto_aug_ensure_val.setChecked(True)
        auto_val_row.addWidget(self._auto_aug_ensure_val)
        auto_val_row.addWidget(QLabel("Val %"))
        self._auto_aug_val_frac = QDoubleSpinBox()
        self._auto_aug_val_frac.setRange(0.05, 0.50)
        self._auto_aug_val_frac.setSingleStep(0.05)
        self._auto_aug_val_frac.setDecimals(2)
        self._auto_aug_val_frac.setValue(0.20)
        auto_val_row.addWidget(self._auto_aug_val_frac)
        auto_aug_layout.addLayout(auto_val_row)

        self._auto_aug_all_folders = QCheckBox("All folders")
        self._auto_aug_all_folders.setChecked(True)
        self._auto_aug_all_folders.toggled.connect(self._sync_auto_aug_folder_controls)
        auto_aug_layout.addWidget(self._auto_aug_all_folders)

        self._auto_aug_folder_list = QListWidget()
        self._auto_aug_folder_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self._auto_aug_folder_list.setMaximumHeight(110)
        self._auto_aug_folder_list.setVisible(False)
        auto_aug_layout.addWidget(self._auto_aug_folder_list)

        self._auto_aug_progress = QProgressBar()
        self._auto_aug_progress.setRange(0, 100)
        self._auto_aug_progress.setVisible(False)
        self._auto_aug_progress.setTextVisible(True)
        self._auto_aug_progress.setFormat("%p% · %v / %m images")
        self._auto_aug_progress.setStyleSheet(
            "QProgressBar { border: 1px solid rgba(10,143,168,0.3); border-radius: 2px; background: rgba(10,143,168,0.05); }"
            "QProgressBar::chunk { background-color: #0A8FA8; }"
        )
        auto_aug_layout.addWidget(self._auto_aug_progress)

        self._auto_aug_eta = QLabel("")
        self._auto_aug_eta.setStyleSheet("font-size: 10px; color: #0A8FA8;")
        auto_aug_layout.addWidget(self._auto_aug_eta)

        self._auto_aug_btn = QPushButton("Auto augment to total")
        self._auto_aug_btn.clicked.connect(self._auto_augment_to_total)
        auto_aug_layout.addWidget(self._auto_aug_btn)

        self._even_dataset_btn = QPushButton("Even dataset")
        self._even_dataset_btn.clicked.connect(self._even_dataset)
        auto_aug_layout.addWidget(self._even_dataset_btn)

        auto_aug_hint = QLabel(
            "Randomizes scale, angle, image quality, grayscale, and BGR channel shuffle. "
            "Preserves train/val ratio when both splits exist; otherwise seeds val from train. "
            "Even dataset augments until every train/val folder has the same image count."
        )
        auto_aug_hint.setWordWrap(True)
        auto_aug_hint.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        auto_aug_layout.addWidget(auto_aug_hint)
        sel_layout.addWidget(auto_aug_box)

        apply_row = QHBoxLayout()
        apply_row.addWidget(QLabel("Apply full-image box class_id:"))
        self._apply_class_id = QSpinBox()
        self._apply_class_id.setRange(0, 9999)
        apply_row.addWidget(self._apply_class_id)
        self._apply_only_missing = QCheckBox("Only missing")
        self._apply_only_missing.setChecked(True)
        apply_row.addWidget(self._apply_only_missing)
        self._apply_replace = QCheckBox("Replace")
        self._apply_replace.setChecked(True)
        apply_row.addWidget(self._apply_replace)
        self._apply_btn = QPushButton("Apply")
        self._apply_btn.clicked.connect(self._apply_full_image_selected)
        apply_row.addWidget(self._apply_btn)
        sel_layout.addLayout(apply_row)

        self._delete_btn = QPushButton("Delete selected")
        self._delete_btn.clicked.connect(self._delete_selected)
        sel_layout.addWidget(self._delete_btn)

        right.addWidget(sel_box)

        classes_box = QGroupBox("Classes (classes.txt)")
        classes_layout = QVBoxLayout(classes_box)
        classes_layout.setContentsMargins(10, 10, 10, 10)
        classes_layout.setSpacing(6)
        self._classes_editor = QTextEdit()
        self._classes_editor.setFont(_mono_font(11))
        self._classes_editor.setPlaceholderText("one class name per line")
        classes_layout.addWidget(self._classes_editor, stretch=1)

        classes_btn_row = QHBoxLayout()
        self._save_classes_btn = QPushButton("Save classes")
        self._save_classes_btn.clicked.connect(self._save_classes_only)
        classes_btn_row.addWidget(self._save_classes_btn)
        self._drop_unmapped = QCheckBox("Drop unmapped boxes on remap")
        self._drop_unmapped.setChecked(False)
        classes_btn_row.addWidget(self._drop_unmapped, stretch=1)
        self._save_remap_btn = QPushButton("Save + Remap labels")
        self._save_remap_btn.clicked.connect(self._save_and_remap)
        classes_btn_row.addWidget(self._save_remap_btn)
        classes_layout.addLayout(classes_btn_row)

        hint = QLabel(
            "Remap uses class-name matching (case-insensitive):\n"
            "old class IDs are updated to the new index of the same name."
        )
        hint.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        classes_layout.addWidget(hint)

        right.addWidget(classes_box, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self.reload()

    def set_dataset_slug(self, slug: str) -> None:
        """Switch the editor to a different dataset library in place."""
        new_slug = str(slug or "").strip()
        if new_slug == self._slug:
            return
        # Stop any in-flight auto-augment worker on the previous dataset before
        # swapping context so its results do not land on the new slug.
        try:
            if self._aug_worker_thread is not None and self._aug_worker_thread.isRunning():
                self._aug_worker_thread.requestInterruption()
        except Exception:
            pass
        if self._aug_poll_timer is not None:
            try:
                self._aug_poll_timer.stop()
            except Exception:
                pass
        self._slug = new_slug
        self._enc_slug = urllib.parse.quote(self._slug, safe="")
        self.setWindowTitle(f"Dataset Editor — {self._slug}" if self._slug else "Dataset Editor")
        self._title_label.setText(f"Dataset: {self._slug}" if self._slug else "Dataset: [NONE SELECTED]")
        self._auto_aug_progress.setVisible(False)
        self._auto_aug_eta.setText("")
        try:
            self._filter.clear()
        except Exception:
            pass
        if self._slug:
            self.reload()
        else:
            self._entries = []
            self._table.setRowCount(0)
            self._classes = []
            try:
                self._classes_editor.clear()
            except Exception:
                pass
            self._set_status("")

    def _open_inventory(self) -> None:
        dlg = FolderInventoryDialog(base_url=self._base_url, dataset_slug=self._slug, parent=self)
        dlg.exec()

    def _http_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        url = self._base_url + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        with urllib.request.urlopen(req, timeout=60.0) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _selected_relative_paths(self) -> list[str]:
        rows = sorted({idx.row() for idx in self._table.selectionModel().selectedRows()})
        out: list[str] = []
        for r in rows:
            it = self._table.item(r, 0)
            if it is None:
                continue
            rel = str(it.data(Qt.ItemDataRole.UserRole) or "")
            if rel:
                out.append(rel)
        return out

    def _set_status(self, text: str) -> None:
        self._status.setText(str(text or ""))

    def _apply_filter(self) -> None:
        needle = str(self._filter.text() or "").strip().lower()
        for r in range(self._table.rowCount()):
            hide = False
            if needle:
                row_text = " ".join(
                    str(self._table.item(r, c).text() if self._table.item(r, c) is not None else "")
                    for c in range(self._table.columnCount())
                ).lower()
                hide = needle not in row_text
            self._table.setRowHidden(r, hide)
        self._sync_selection_status()

    def _sync_selection_status(self) -> None:
        selected = len(self._selected_relative_paths())
        visible = sum(1 for r in range(self._table.rowCount()) if not self._table.isRowHidden(r))
        total = self._table.rowCount()
        msg = f"{selected} selected" if selected else "no selection"
        self._selected_label.setText(f"{msg}   |   {visible}/{total} visible")

    def reload(self) -> None:
        if not self._slug:
            return
        try:
            payload = self._http_json("GET", f"/database/{self._enc_slug}")
        except Exception as exc:
            self._set_status(f"Load failed: {exc}")
            return

        fmt = str(payload.get("format") or "")
        if fmt and fmt != "yolo_detection":
            self._set_status(f"Dataset format: {fmt} (editor focuses on YOLO detection)")
        else:
            self._set_status("")

        self._entries = [e for e in (payload.get("images") or []) if isinstance(e, dict)]
        raw_classes = payload.get("classes") if isinstance(payload, dict) else []
        self._classes = [str(c).strip() for c in (raw_classes or []) if str(c).strip()]
        self._classes_editor.setPlainText("\n".join(self._classes).strip())

        train_count = sum(1 for e in self._entries if str(e.get("split") or "").lower() == "train")
        val_count = sum(1 for e in self._entries if str(e.get("split") or "").lower() == "val")
        current_total = train_count + val_count
        self._auto_aug_count.setText(f"Current train+val: {current_total} ({train_count} train / {val_count} val)")
        self._seed_val_btn.setVisible(val_count <= 0 and train_count > 0)
        if int(self._auto_aug_goal.value()) < max(1, current_total):
            self._auto_aug_goal.setValue(max(1, current_total))
        self._refresh_auto_aug_folder_list()

        self._table.setRowCount(0)
        self._table.setRowCount(len(self._entries))
        for r, e in enumerate(self._entries):
            disp = str(e.get("display_name") or e.get("name") or "")
            split = str(e.get("split") or "")
            has_label = bool(e.get("has_label"))
            rel_path = str(e.get("relative_path") or "")
            size = int(e.get("size") or 0)
            size_h = f"{size / 1024.0:.1f} KB" if size else ""

            it0 = QTableWidgetItem(disp)
            it0.setData(Qt.ItemDataRole.UserRole, rel_path)
            self._table.setItem(r, 0, it0)
            self._table.setItem(r, 1, QTableWidgetItem(split))
            self._table.setItem(r, 2, QTableWidgetItem("yes" if has_label else "missing"))
            self._table.setItem(r, 3, QTableWidgetItem(size_h))
            it4 = QTableWidgetItem(rel_path)
            it4.setFont(_mono_font(10))
            self._table.setItem(r, 4, it4)

        self._apply_filter()

    def _refresh_auto_aug_folder_list(self) -> None:
        if not hasattr(self, "_auto_aug_folder_list"):
            return
        previous = {
            str(self._auto_aug_folder_list.item(i).data(Qt.ItemDataRole.UserRole) or "")
            for i in range(self._auto_aug_folder_list.count())
            if self._auto_aug_folder_list.item(i).checkState() == Qt.CheckState.Checked
        }
        folders: dict[str, int] = {}
        for e in self._entries:
            folder = _image_folder_from_relpath(_entry_relpath(e))
            folders[folder] = folders.get(folder, 0) + 1
        self._auto_aug_folder_list.clear()
        for folder, count in sorted(folders.items(), key=lambda kv: kv[0].lower()):
            item = QListWidgetItem(f"{folder} ({count})")
            item.setData(Qt.ItemDataRole.UserRole, folder)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked if not previous or folder in previous else Qt.CheckState.Unchecked)
            self._auto_aug_folder_list.addItem(item)
        self._sync_auto_aug_folder_controls()

    def _sync_auto_aug_folder_controls(self) -> None:
        if hasattr(self, "_auto_aug_folder_list"):
            self._auto_aug_folder_list.setVisible(not self._auto_aug_all_folders.isChecked())

    def _selected_auto_aug_folders(self) -> list[str]:
        if not hasattr(self, "_auto_aug_folder_list") or self._auto_aug_all_folders.isChecked():
            return []
        out: list[str] = []
        for i in range(self._auto_aug_folder_list.count()):
            item = self._auto_aug_folder_list.item(i)
            if item is not None and item.checkState() == Qt.CheckState.Checked:
                out.append(str(item.data(Qt.ItemDataRole.UserRole) or ""))
        return out

    def _confirm_many(self, title: str, text: str, *, count: int) -> bool:
        if count <= 0:
            return False
        if count <= 1:
            return True
        resp = QMessageBox.warning(
            self,
            title,
            f"{text}\n\nItems: {count}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return resp == QMessageBox.StandardButton.Yes

    def _move_selected(self) -> None:
        rels = self._selected_relative_paths()
        if not rels:
            return
        target = str(self._move_split.currentData() or "train")
        if not self._confirm_many("Move Items", f"Move selected items to '{target}'?", count=len(rels)):
            return
        try:
            self._set_status("Moving…")
            payload = self._http_json(
                "POST",
                f"/database/{self._enc_slug}/move_to_split",
                {"relative_paths": rels, "target_split": target},
            )
            moved = int(payload.get("moved") or 0)
            errors = payload.get("errors") or []
            self._set_status(f"Moved {moved}.")
            if errors:
                QMessageBox.warning(self, "Move Errors", str(errors[0]))
        except Exception as exc:
            self._set_status(f"Move failed: {exc}")
        self.reload()

    def _copy_augmented_selected_to_val(self) -> None:
        rels = self._selected_relative_paths()
        if not rels:
            return
        selected_entries = [e for e in self._entries if str(e.get("relative_path") or "") in set(rels)]
        train_count = sum(1 for e in selected_entries if str(e.get("split") or "").lower() == "train")
        if train_count <= 0:
            QMessageBox.information(
                self,
                "Augment to Val",
                "Select one or more train images before copying augmented samples into val.",
            )
            return
        balance = bool(self._aug_balance.isChecked())
        copies = int(self._aug_copies.value())
        if balance:
            prompt = "Copy augmented train samples into val until val is balanced toward train?"
        else:
            prompt = f"Copy {copies} augmented val sample(s) for each selected train image?"
        if not self._confirm_many("Augment to Val", prompt, count=train_count):
            return
        try:
            self._set_status("Augmenting to val…")
            payload = self._http_json(
                "POST",
                f"/database/{self._enc_slug}/copy_augmented_to_split",
                {
                    "relative_paths": [
                        str(e.get("relative_path") or "")
                        for e in selected_entries
                        if str(e.get("split") or "").lower() == "train"
                    ],
                    "target_split": "val",
                    "copies_per_image": copies,
                    "balance_to_train": balance,
                    "scale_pct": int(self._aug_scale.value()),
                    "angle_deg": float(self._aug_angle.value()),
                    "jpeg_quality": int(self._aug_quality.value()),
                    "grayscale": bool(self._aug_gray.isChecked()),
                    "suffix": "aug",
                },
            )
            copied = int(payload.get("copied") or 0)
            skipped = int(payload.get("skipped") or 0)
            errors = payload.get("errors") or []
            self._set_status(f"Augmented {copied} into val (skipped {skipped}).")
            if errors:
                QMessageBox.warning(self, "Augment Errors", str(errors[0]))
        except Exception as exc:
            self._set_status(f"Augment failed: {exc}")
        self.reload()

    def _auto_augment_to_total(self) -> None:
        selected_folders = self._selected_auto_aug_folders()
        folder_scope = set(selected_folders)
        scoped_entries = [
            e for e in self._entries
            if str(e.get("split") or "").lower() in {"train", "val"}
            and (not folder_scope or _image_folder_from_relpath(_entry_relpath(e)) in folder_scope)
        ]
        if (not self._auto_aug_all_folders.isChecked()) and not selected_folders:
            QMessageBox.information(
                self,
                "Auto Augment",
                "Select one or more folders, or choose All folders.",
            )
            return
        train_count = sum(1 for e in scoped_entries if str(e.get("split") or "").lower() == "train")
        val_count = sum(1 for e in scoped_entries if str(e.get("split") or "").lower() == "val")
        current_total = train_count + val_count
        if current_total <= 0:
            QMessageBox.information(
                self,
                "Auto Augment",
                "This dataset needs train or val images before auto augmentation can run.",
            )
            return
        goal = int(self._auto_aug_goal.value())
        if goal <= current_total:
            QMessageBox.information(
                self,
                "Auto Augment",
                f"Goal total is already met. Current train+val total is {current_total}.",
            )
            return
        add_count = goal - current_total
        avg_img_size = 0
        sizes = [int(e.get("size") or 0) for e in scoped_entries if int(e.get("size") or 0) > 0]
        if sizes:
            avg_img_size = int(sum(sizes) / max(1, len(sizes)))
        estimated_add = avg_img_size * add_count
        dataset_current = sum(int(e.get("size") or 0) for e in self._entries)
        dataset_estimated = dataset_current + estimated_add
        scope_label = "all folders" if not selected_folders else ", ".join(selected_folders[:8])
        if selected_folders and len(selected_folders) > 8:
            scope_label += f" (+{len(selected_folders) - 8} more)"
        val_note = ""
        if val_count <= 0 and self._auto_aug_ensure_val.isChecked():
            val_pct = int(float(self._auto_aug_val_frac.value()) * 100)
            val_note = (
                f"\n\nNo val split yet — about {val_pct}% of new images will be written to images/val."
            )
        prompt = (
            f"Create {add_count} randomized augmented image(s) to reach {goal} train+val images in scope.\n\n"
            f"Folders: {scope_label}\n"
            f"This augmentation will add an additional: {_human_bytes(estimated_add) or 'unknown'} "
            f"to dataset: {_human_bytes(dataset_estimated) or 'unknown'}"
            f"{val_note}"
        )
        resp = QMessageBox.warning(
            self,
            "Auto Augment Storage Estimate",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return

        self._auto_aug_current_total = current_total
        self._start_auto_augment_worker(
            goal,
            selected_folders,
            val_frac=float(self._auto_aug_val_frac.value()),
            ensure_val=bool(self._auto_aug_ensure_val.isChecked()),
        )

    def _seed_val_from_train(self) -> None:
        train_count = sum(1 for e in self._entries if str(e.get("split") or "").lower() == "train")
        val_count = sum(1 for e in self._entries if str(e.get("split") or "").lower() == "val")
        if train_count <= 0:
            QMessageBox.information(
                self,
                "Seed Val Split",
                "This dataset has no train images to sample from.",
            )
            return
        if val_count > 0:
            QMessageBox.information(
                self,
                "Seed Val Split",
                "A validation split already exists.",
            )
            return
        val_frac = float(self._seed_val_frac.value())
        val_add = max(1, int(round(train_count * val_frac)))
        goal = train_count + val_add
        prompt = (
            f"Create {val_add} augmented validation image(s) ({int(val_frac * 100)}% of train) "
            f"under images/val and labels/val?"
        )
        if not self._confirm_many("Seed Val Split", prompt, count=val_add):
            return
        self._auto_aug_current_total = train_count
        self._start_auto_augment_worker(
            goal,
            self._selected_auto_aug_folders(),
            val_frac=val_frac,
            ensure_val=True,
        )

    def _set_aug_controls_busy(self, busy: bool) -> None:
        self._auto_aug_btn.setEnabled(not busy)
        self._even_dataset_btn.setEnabled(not busy)
        self._seed_val_btn.setEnabled(not busy)

    def _even_dataset(self) -> None:
        selected_folders = self._selected_auto_aug_folders()
        folder_scope: Optional[set[str]] = None
        if not self._auto_aug_all_folders.isChecked():
            if not selected_folders:
                QMessageBox.information(
                    self,
                    "Even Dataset",
                    "Select one or more folders, or choose All folders.",
                )
                return
            folder_scope = set(selected_folders)
        target, total_add, before_counts = _estimate_even_dataset_adds(self._entries, folder_scope)
        if not before_counts:
            QMessageBox.information(
                self,
                "Even Dataset",
                "This dataset needs train or val images before it can be evened.",
            )
            return
        if total_add <= 0:
            QMessageBox.information(
                self,
                "Even Dataset",
                f"All folders already have {target} images each.",
            )
            return
        summary_lines = [f"  {name}: {count}" for name, count in sorted(before_counts.items())]
        if len(summary_lines) > 12:
            summary_lines = summary_lines[:12] + [f"  … (+{len(before_counts) - 12} more)"]
        prompt = (
            f"Create {total_add} augmented image(s) so every train/val folder reaches {target} images.\n\n"
            f"Current counts:\n" + "\n".join(summary_lines)
        )
        if not self._confirm_many("Even Dataset", prompt, count=total_add):
            return
        self._auto_aug_current_total = 0
        self._auto_aug_goal_val = total_add
        self._set_status("Evening dataset…")
        self._auto_aug_start_time = time.time()
        self._auto_aug_progress.setRange(0, max(1, total_add))
        self._auto_aug_progress.setValue(0)
        self._auto_aug_progress.setVisible(True)
        self._auto_aug_eta.setText("")
        self._set_aug_controls_busy(True)
        self._aug_worker_kind = "even"
        self._aug_worker_thread = _EvenDatasetWorker(self._base_url, self._slug, selected_folders)
        self._aug_worker_thread.finished.connect(self._on_aug_worker_finished)
        self._aug_worker_thread.error.connect(self._on_aug_worker_error)
        self._aug_worker_thread.start()
        if self._aug_poll_timer is None:
            self._aug_poll_timer = QTimer()
            self._aug_poll_timer.timeout.connect(self._update_auto_augment_progress)
            self._aug_poll_timer.start(200)

    def _start_auto_augment_worker(
        self,
        goal: int,
        selected_folders: list[str],
        *,
        val_frac: float,
        ensure_val: bool,
    ) -> None:
        add_count = max(0, goal - int(self._auto_aug_current_total or 0))
        self._set_status("Auto augmenting dataset…")
        self._auto_aug_start_time = time.time()
        self._auto_aug_goal_val = goal
        self._auto_aug_progress.setRange(0, max(1, add_count))
        self._auto_aug_progress.setValue(0)
        self._auto_aug_progress.setVisible(True)
        self._auto_aug_eta.setText("")
        self._set_aug_controls_busy(True)
        self._aug_worker_kind = "auto"

        self._aug_worker_thread = _AutoAugmentWorker(
            self._base_url,
            self._slug,
            goal,
            selected_folders,
            val_frac=val_frac,
            ensure_val=ensure_val,
        )
        self._aug_worker_thread.finished.connect(self._on_aug_worker_finished)
        self._aug_worker_thread.error.connect(self._on_aug_worker_error)
        self._aug_worker_thread.start()

        if self._aug_poll_timer is None:
            self._aug_poll_timer = QTimer()
            self._aug_poll_timer.timeout.connect(self._update_auto_augment_progress)
            self._aug_poll_timer.start(200)

    def _update_auto_augment_progress(self) -> None:
        """Update progress bar and ETA while augmentation is running."""
        if self._aug_worker_thread is None or not self._aug_worker_thread.isRunning():
            return

        elapsed_sec = time.time() - self._auto_aug_start_time
        add_count = self._auto_aug_goal_val - self._auto_aug_current_total

        progress = min(int(elapsed_sec * 10), add_count)
        progress_pct = int((progress / max(1, add_count)) * 100)

        eta_remaining = max(1.0, (add_count - progress) / max(1, progress + 1))
        eta_min = int(eta_remaining / 60)
        eta_sec = int(eta_remaining % 60)
        if eta_min > 0:
            self._auto_aug_eta.setText(f"ETA: {eta_min}m {eta_sec}s")
        else:
            self._auto_aug_eta.setText(f"ETA: {eta_sec}s")

        self._auto_aug_progress.setMaximum(add_count)
        self._auto_aug_progress.setValue(progress)
        self._auto_aug_progress.setFormat(f"{progress_pct}% · {self._auto_aug_current_total + progress} / {self._auto_aug_goal_val} images")

    def _on_aug_worker_finished(self, payload: dict[str, Any]) -> None:
        """Handle completion of background augmentation workers."""
        if self._aug_poll_timer is not None:
            self._aug_poll_timer.stop()

        copied = int(payload.get("copied") or 0)
        skipped = int(payload.get("skipped") or 0)
        errors = payload.get("errors") or []
        additions = payload.get("additions_by_split") or {}
        train_added = int(additions.get("train") or 0)
        val_added = int(additions.get("val") or 0)

        self._auto_aug_progress.setVisible(False)
        self._auto_aug_eta.setText("")
        self._set_aug_controls_busy(False)
        layout_note = ""
        if bool(payload.get("val_layout_updated")):
            layout_note = " Updated images/val, labels/val, and data.yaml."
        if self._aug_worker_kind == "even":
            target = int(payload.get("target_per_bucket") or 0)
            self._set_status(
                f"Evened dataset: {copied} augmented ({train_added} train / {val_added} val, "
                f"target {target} per folder, skipped {skipped}).{layout_note}"
            )
            title = "Even Dataset Errors"
        else:
            self._set_status(
                f"Auto augmented {copied} ({train_added} train / {val_added} val, skipped {skipped})."
                f"{layout_note}"
            )
            title = "Auto Augment Errors"
        if errors:
            QMessageBox.warning(self, title, str(errors[0]))
        self._aug_worker_kind = ""
        self.reload()

    def _on_aug_worker_error(self, error_msg: str) -> None:
        """Handle error during background augmentation workers."""
        if self._aug_poll_timer is not None:
            self._aug_poll_timer.stop()

        self._auto_aug_progress.setVisible(False)
        self._auto_aug_eta.setText("")
        self._set_aug_controls_busy(False)
        label = "Even dataset" if self._aug_worker_kind == "even" else "Auto augment"
        self._set_status(f"{label} failed: {error_msg}")
        self._aug_worker_kind = ""

    def _clear_labels_selected(self) -> None:
        rels = self._selected_relative_paths()
        if not rels:
            return
        if not self._confirm_many("Clear Labels", "Clear label text for selected items?", count=len(rels)):
            return
        try:
            self._set_status("Clearing labels…")
            payload = self._http_json(
                "POST",
                f"/database/{self._enc_slug}/labels/clear_to_paths",
                {"relative_paths": rels},
            )
            cleared = int(payload.get("cleared") or 0)
            errors = payload.get("errors") or []
            self._set_status(f"Cleared {cleared}.")
            if errors:
                QMessageBox.warning(self, "Clear Errors", str(errors[0]))
        except Exception as exc:
            self._set_status(f"Clear failed: {exc}")
        self.reload()

    def _apply_full_image_selected(self) -> None:
        rels = self._selected_relative_paths()
        if not rels:
            return
        class_id = int(self._apply_class_id.value())
        only_missing = bool(self._apply_only_missing.isChecked())
        replace = bool(self._apply_replace.isChecked())
        if not self._confirm_many(
            "Bulk Apply Labels",
            f"Apply a full-image box for class_id {class_id} to selected items?",
            count=len(rels),
        ):
            return
        try:
            self._set_status("Applying…")
            payload = self._http_json(
                "POST",
                f"/database/{self._enc_slug}/labels/bulk_apply_to_paths",
                {
                    "relative_paths": rels,
                    "class_id": class_id,
                    "geometry": "full_image",
                    "center_w": 0.5,
                    "center_h": 0.5,
                    "only_missing": only_missing,
                    "replace": replace,
                },
            )
            applied = int(payload.get("applied") or 0)
            skipped = int(payload.get("skipped") or 0)
            errors = payload.get("errors") or []
            self._set_status(f"Applied {applied} (skipped {skipped}).")
            if errors:
                QMessageBox.warning(self, "Apply Errors", str(errors[0]))
        except Exception as exc:
            self._set_status(f"Apply failed: {exc}")
        self.reload()

    def _delete_selected(self) -> None:
        rels = self._selected_relative_paths()
        if not rels:
            return
        if not self._confirm_many("Delete Items", "Delete selected items (image + label)?", count=len(rels)):
            return
        deleted = 0
        errors: list[str] = []
        self._set_status("Deleting…")
        for rel in rels:
            try:
                encoded = urllib.parse.quote(rel, safe="")
                self._http_json("DELETE", f"/database/{self._enc_slug}/{encoded}")
                deleted += 1
            except Exception as exc:
                errors.append(f"{rel}: {exc}")
                if len(errors) >= 5:
                    break
        self._set_status(f"Deleted {deleted}.")
        if errors:
            QMessageBox.warning(self, "Delete Errors", errors[0])
        self.reload()

    def _parsed_classes_editor(self) -> list[str]:
        raw = self._classes_editor.toPlainText()
        out: list[str] = []
        seen: set[str] = set()
        for ln in (raw or "").splitlines():
            name = ln.strip()
            if not name:
                continue
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(name)
            if len(out) >= 500:
                break
        return out

    def _save_classes_only(self) -> None:
        new_classes = self._parsed_classes_editor()
        if not new_classes:
            QMessageBox.warning(self, "Invalid Classes", "Class list is empty.")
            return
        try:
            self._set_status("Saving classes…")
            self._http_json("PUT", f"/database/{self._enc_slug}/classes", {"classes": new_classes})
            self._classes = list(new_classes)
            self._set_status(f"Saved {len(new_classes)} classes.")
        except Exception as exc:
            self._set_status(f"Save failed: {exc}")
            return
        self.reload()

    def _save_and_remap(self) -> None:
        old_classes = list(self._classes)
        new_classes = self._parsed_classes_editor()
        if not new_classes:
            QMessageBox.warning(self, "Invalid Classes", "Class list is empty.")
            return
        drop_unmapped = bool(self._drop_unmapped.isChecked())
        confirm = QMessageBox.warning(
            self,
            "Save + Remap",
            "This will save classes.txt and rewrite label files to match the new class order.\n\n"
            "Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        try:
            self._set_status("Saving classes…")
            self._http_json("PUT", f"/database/{self._enc_slug}/classes", {"classes": new_classes})
            self._set_status("Remapping labels…")
            payload = self._http_json(
                "POST",
                f"/database/{self._enc_slug}/labels/remap_by_name",
                {
                    "old_classes": old_classes,
                    "new_classes": new_classes,
                    "drop_unmapped": drop_unmapped,
                },
            )
            touched = int(payload.get("files_touched") or 0)
            changed = int(payload.get("files_changed") or 0)
            self._classes = list(new_classes)
            self._set_status(f"Remapped labels ({changed}/{touched} files changed).")
        except Exception as exc:
            self._set_status(f"Remap failed: {exc}")
            return
        self.reload()
