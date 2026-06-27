"""Pre-index dialog: choose where to save frames that match detection filters."""

from __future__ import annotations

import re
import time
import urllib.parse
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

HttpGet = Callable[[str], dict[str, Any]]
HttpPost = Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]


def _sanitize_subfolder(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(name or "").strip()).strip("._-")
    return s[:80] if s else "frames"


def _default_subfolder(video_stem: str) -> str:
    ts = time.strftime("%Y%m%d_%H%M%S")
    return _sanitize_subfolder(f"from_{video_stem}_{ts}")


def _compute_database_export_dir(
    *,
    dataset_path: str,
    fmt: str,
    split: str,
    subfolder: str,
) -> Path:
    base = Path(dataset_path).expanduser().resolve()
    sub = _sanitize_subfolder(subfolder)
    if fmt == "yolo_detection":
        return base / "images" / split / sub
    return base / "imports" / "video_index" / sub


class VideoIndexExportDialog(QDialog):
    """Returns export directory via :meth:`export_dir` when user accepts."""

    def __init__(
        self,
        *,
        video_path: Path,
        http_get: Optional[HttpGet] = None,
        http_post: Optional[HttpPost] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Video index — save frames")
        self.resize(520, 420)
        self._http_get = http_get
        self._http_post = http_post
        self._video_path = Path(video_path)
        self._export_dir: str = ""
        self._dataset_meta: dict[str, Any] = {}

        stem = self._video_path.stem
        default_sub = _default_subfolder(stem)

        outer = QVBoxLayout(self)
        outer.setSpacing(10)

        intro = QLabel(
            "Frames where the model finds detections (after your category and label filters) "
            "can be written as JPEGs. Put them under a database folder so you can label them "
            "in the Database tab and wire the folder into a scenario for training. "
            "Use New YOLO dataset to scaffold an empty YOLO layout under database/ (images, labels, data.yaml)."
        )
        intro.setWordWrap(True)
        outer.addWidget(intro)

        self._enable_export = QRadioButton("Export matching frames to disk")
        self._enable_export.setChecked(True)
        self._skip_export = QRadioButton("Index only (timeline / overlay, no files)")
        outer.addWidget(self._enable_export)
        outer.addWidget(self._skip_export)

        self._dest_box = QGroupBox("Export destination")
        dest_lay = QVBoxLayout(self._dest_box)

        row_mode = QHBoxLayout()
        self._mode_db = QRadioButton("Inside a database folder")
        self._mode_custom = QRadioButton("Custom folder")
        self._mode_db.setChecked(True)
        row_mode.addWidget(self._mode_db)
        row_mode.addWidget(self._mode_custom)
        row_mode.addStretch(1)
        dest_lay.addLayout(row_mode)

        self._stack = QStackedWidget()
        # Page 0: database
        db_page = QWidget()
        db_form = QFormLayout(db_page)
        self._slug_combo = QComboBox()
        self._slug_combo.setMinimumWidth(260)
        self._slug_combo.currentIndexChanged.connect(self._on_slug_changed)
        db_form.addRow("Dataset:", self._slug_combo)

        self._split_combo = QComboBox()
        for sp in ("train", "val", "test"):
            self._split_combo.addItem(sp, sp)
        db_form.addRow("Split:", self._split_combo)
        self._split_combo.currentIndexChanged.connect(self._refresh_computed_path)

        self._subfolder_edit = QLineEdit(default_sub)
        self._subfolder_edit.setPlaceholderText("subfolder name under images/… or imports/…")
        self._subfolder_edit.textChanged.connect(self._refresh_computed_path)
        db_form.addRow("Subfolder:", self._subfolder_edit)

        self._format_lbl = QLabel("")
        self._format_lbl.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.75);")
        self._format_lbl.setWordWrap(True)
        db_form.addRow("Layout:", self._format_lbl)

        self._computed_path = QLabel("")
        self._computed_path.setWordWrap(True)
        self._computed_path.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        db_form.addRow("Full path:", self._computed_path)

        dataset_actions = QHBoxLayout()
        refresh_btn = QPushButton("Reload list")
        refresh_btn.clicked.connect(self._load_dataset_names)
        dataset_actions.addWidget(refresh_btn)
        self._new_yolo_btn = QPushButton("New YOLO dataset…")
        self._new_yolo_btn.setToolTip(
            "Create database/<slug>/ with images/{train,val,test}, matching labels/, classes.txt, and data.yaml."
        )
        self._new_yolo_btn.clicked.connect(self._on_create_yolo_template)
        dataset_actions.addWidget(self._new_yolo_btn)
        dataset_actions.addStretch(1)
        db_form.addRow("", dataset_actions)

        self._stack.addWidget(db_page)

        # Page 1: custom
        custom_page = QWidget()
        custom_v = QVBoxLayout(custom_page)
        custom_row = QHBoxLayout()
        self._custom_path = QLineEdit()
        self._custom_path.setPlaceholderText("Absolute path to a writable folder")
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_custom)
        custom_row.addWidget(self._custom_path)
        custom_row.addWidget(browse_btn)
        custom_v.addLayout(custom_row)
        self._stack.addWidget(custom_page)

        dest_lay.addWidget(self._stack)

        self._mode_db.toggled.connect(self._on_mode_toggled)
        self._mode_custom.toggled.connect(self._on_mode_toggled)
        self._on_mode_toggled()

        outer.addWidget(self._dest_box)

        self._enable_export.toggled.connect(self._sync_enabled)
        self._skip_export.toggled.connect(self._sync_enabled)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

        self._load_dataset_names()
        self._refresh_computed_path()
        self._sync_enabled()

    def _sync_enabled(self) -> None:
        on = self._enable_export.isChecked()
        self._dest_box.setEnabled(on)

    def _on_mode_toggled(self) -> None:
        self._stack.setCurrentIndex(0 if self._mode_db.isChecked() else 1)

    def _browse_custom(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select folder for exported frames")
        if path:
            self._custom_path.setText(path)

    def _on_create_yolo_template(self) -> None:
        if self._http_post is None:
            QMessageBox.warning(
                self,
                "Video index",
                "Cannot create a dataset: CV Ops HTTP API is not available from this panel.",
            )
            return
        default_name = _sanitize_subfolder(f"vid_{self._video_path.stem}_{time.strftime('%Y%m%d')}")
        if not default_name or default_name == "frames":
            default_name = f"vid_index_{time.strftime('%Y%m%d_%H%M%S')}"

        wiz = QDialog(self)
        wiz.setWindowTitle("New YOLO dataset (template)")
        wiz.resize(440, 200)
        form = QFormLayout(wiz)
        name_edit = QLineEdit(default_name)
        name_edit.setToolTip("Folder name under database/. A numeric suffix is added if the name already exists.")
        form.addRow("Dataset name:", name_edit)
        classes_edit = QLineEdit("object")
        classes_edit.setPlaceholderText("comma-separated, e.g. person, car, helmet")
        classes_edit.setToolTip("Class names for classes.txt and data.yaml. Default single class: object.")
        form.addRow("Classes:", classes_edit)
        hint = QLabel(
            "Creates database/<slug>/ with images and labels for train, val, and test splits, plus "
            "classes.txt and data.yaml — the same layout as scrape emit and training expects."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.72);")
        form.addRow(hint)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(wiz.accept)
        buttons.rejected.connect(wiz.reject)
        form.addRow(buttons)

        if wiz.exec() != QDialog.DialogCode.Accepted:
            return
        raw_name = name_edit.text().strip()
        if not raw_name:
            QMessageBox.warning(self, "Video index", "Dataset name is required.")
            return
        class_parts = [c.strip() for c in classes_edit.text().split(",") if c.strip()]
        body: dict[str, Any] = {"name": raw_name, "unique": True}
        if class_parts:
            body["classes"] = class_parts
        try:
            payload = self._http_post("/database/create_yolo_template", body)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Video index",
                f"Could not create dataset template:\n{exc}",
            )
            return
        slug = str((payload or {}).get("slug") or "").strip()
        path = str((payload or {}).get("path") or "").strip()
        if not slug:
            QMessageBox.critical(self, "Video index", "Server did not return a dataset slug.")
            return
        self._load_dataset_names()
        idx = self._slug_combo.findData(slug)
        if idx >= 0:
            self._slug_combo.setCurrentIndex(idx)
        self._on_slug_changed()
        QMessageBox.information(
            self,
            "Video index",
            f"Created YOLO dataset `{slug}`.\n{path}\n\nSelect split and subfolder, then OK to run the index.",
        )

    def _load_dataset_names(self) -> None:
        self._slug_combo.blockSignals(True)
        self._slug_combo.clear()
        self._slug_combo.addItem("(select dataset)", "")
        if self._http_get is None:
            self._slug_combo.addItem("(HTTP unavailable — use Custom folder)", "")
            self._slug_combo.blockSignals(False)
            self._format_lbl.setText("Connect CV Ops or choose a custom export path.")
            return
        try:
            payload = self._http_get("/database")
            names = payload.get("datasets") if isinstance(payload, dict) else []
            categories = payload.get("categories") if isinstance(payload, dict) else {}
            if not isinstance(names, list):
                names = []
            for name in sorted(str(n) for n in names if n):
                cat = str(categories.get(name, "image") if isinstance(categories, dict) else "image")
                if cat != "image":
                    continue
                self._slug_combo.addItem(name, name)
        except Exception as exc:
            self._format_lbl.setText(f"Could not list databases: {exc}")
        self._slug_combo.blockSignals(False)
        self._on_slug_changed()

    def _on_slug_changed(self) -> None:
        slug = str(self._slug_combo.currentData() or "").strip()
        self._dataset_meta = {}
        if not slug or self._http_get is None:
            self._format_lbl.setText("Select an image dataset under database/.")
            self._refresh_computed_path()
            return
        try:
            enc = urllib.parse.quote(slug, safe="")
            payload = self._http_get(f"/database/{enc}")
            if isinstance(payload, dict):
                self._dataset_meta = payload
        except Exception as exc:
            self._format_lbl.setText(f"Dataset info failed: {exc}")
            self._refresh_computed_path()
            return
        fmt = str(self._dataset_meta.get("format") or "unknown")
        path = str(self._dataset_meta.get("path") or "")
        if fmt == "yolo_detection":
            self._format_lbl.setText(
                "YOLO layout: files go under images/<split>/<subfolder>/ — add matching labels under labels/<split>/<subfolder>/ in the Database tab."
            )
        else:
            self._format_lbl.setText(
                f"Format `{fmt}`: files go under imports/video_index/<subfolder>/ at `{path}` so they stay separate from class folders until you organize them."
            )
        self._refresh_computed_path()

    def _refresh_computed_path(self) -> None:
        if self._mode_custom.isChecked():
            self._computed_path.setText(self._custom_path.text().strip())
            return
        slug = str(self._slug_combo.currentData() or "").strip()
        path = str(self._dataset_meta.get("path") or "")
        fmt = str(self._dataset_meta.get("format") or "")
        if not slug or not path:
            self._computed_path.setText("")
            return
        split = str(self._split_combo.currentData() or "train")
        sub = self._subfolder_edit.text().strip() or "frames"
        try:
            resolved = _compute_database_export_dir(
                dataset_path=path,
                fmt=fmt,
                split=split,
                subfolder=sub,
            )
            self._computed_path.setText(str(resolved))
        except Exception as exc:
            self._computed_path.setText(f"(invalid path: {exc})")

    def _on_accept(self) -> None:
        if self._skip_export.isChecked():
            self._export_dir = ""
            self.accept()
            return
        if self._mode_custom.isChecked():
            raw = self._custom_path.text().strip()
            if not raw:
                self._computed_path.setText("Choose a folder.")
                return
            p = Path(raw).expanduser()
            try:
                p.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self._computed_path.setText(f"Cannot create folder: {exc}")
                return
            self._export_dir = str(p.resolve())
            self.accept()
            return
        slug = str(self._slug_combo.currentData() or "").strip()
        if not slug:
            self._format_lbl.setText("Select a dataset, or switch to Custom folder.")
            return
        path = str(self._dataset_meta.get("path") or "")
        fmt = str(self._dataset_meta.get("format") or "")
        if not path:
            self._format_lbl.setText("Dataset path unknown — pick another dataset.")
            return
        split = str(self._split_combo.currentData() or "train")
        sub = self._subfolder_edit.text().strip() or "frames"
        try:
            dest = _compute_database_export_dir(
                dataset_path=path,
                fmt=fmt,
                split=split,
                subfolder=sub,
            )
            dest.mkdir(parents=True, exist_ok=True)
            self._export_dir = str(dest.resolve())
        except Exception as exc:
            self._format_lbl.setText(f"Cannot create export folder: {exc}")
            return
        self.accept()

    @property
    def export_dir(self) -> str:
        """Absolute directory for JPEG exports, or empty when indexing without export."""
        return self._export_dir

    @property
    def export_enabled(self) -> bool:
        return self._enable_export.isChecked() and bool(self._export_dir)
