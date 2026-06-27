"""Semantic Carve panel for Collect & Edit.

≤5-click flow to turn a folder of images into a trainable ImageFolder dataset:

  1. Browse a source folder
  2. Index  (CLIP-embeds the folder in the background)
  3. type a query + drag the threshold (live thumbnail preview)
  4. name the dataset + class label
  5. Create dataset  -> mlops/dataset_registry/<slug> (imagefolder_classification)

Talks to the carve endpoints: POST /carve/index, GET /carve/index_progress/{id},
POST /carve/preview, POST /carve/create.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, QSize, QTimer, pyqtSignal
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


class SemanticCarvePanel(QFrame):
    errorRaised = pyqtSignal(str)
    statusChanged = pyqtSignal(str)
    datasetCreated = pyqtSignal(str)  # slug

    def __init__(
        self,
        *,
        http_get: Callable[[str], dict[str, Any]],
        http_post: Callable[[str, Optional[dict[str, Any]]], dict[str, Any]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("semanticCarvePanel")
        self._http_get = http_get
        self._http_post = http_post
        self._index_cid = ""
        self._indexed = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        guide = QLabel(
            "Semantic Carve — point at a folder of images, describe what you want "
            "(\"schools\", \"orchards\", \"main street\"), and carve a labeled "
            "ImageFolder dataset by meaning. No manual labeling."
        )
        guide.setWordWrap(True)
        guide.setObjectName("stageInfo")
        root.addWidget(guide)

        # 1 + 2: folder + index
        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Source image folder…")
        src_row.addWidget(self._folder_edit, stretch=1)
        self._browse_btn = QPushButton("Browse")
        self._browse_btn.clicked.connect(self._on_browse)
        src_row.addWidget(self._browse_btn)
        self._index_btn = QPushButton("Index")
        self._index_btn.setProperty("isPrimary", True)
        self._index_btn.setToolTip("CLIP-embed the folder so you can carve by meaning")
        self._index_btn.clicked.connect(self._on_index)
        src_row.addWidget(self._index_btn)
        root.addLayout(src_row)

        self._index_progress = QProgressBar()
        self._index_progress.setVisible(False)
        root.addWidget(self._index_progress)
        self._index_status = QLabel("")
        self._index_status.setObjectName("stageInfo")
        root.addWidget(self._index_status)

        # 3: query + threshold
        q_row = QHBoxLayout()
        q_row.setSpacing(6)
        q_row.addWidget(QLabel("Query:"))
        self._query_edit = QLineEdit()
        self._query_edit.setPlaceholderText("what to carve, e.g. schools")
        self._query_edit.setEnabled(False)
        self._query_edit.textChanged.connect(self._schedule_preview)
        q_row.addWidget(self._query_edit, stretch=1)
        root.addLayout(q_row)

        thr_row = QHBoxLayout()
        thr_row.setSpacing(6)
        thr_row.addWidget(QLabel("Threshold:"))
        self._threshold = QSlider(Qt.Orientation.Horizontal)
        self._threshold.setRange(10, 40)   # 0.10 .. 0.40
        self._threshold.setValue(22)
        self._threshold.setEnabled(False)
        self._threshold.valueChanged.connect(self._on_threshold)
        thr_row.addWidget(self._threshold, stretch=1)
        self._threshold_label = QLabel("0.22")
        self._threshold_label.setMinimumWidth(36)
        thr_row.addWidget(self._threshold_label)
        root.addLayout(thr_row)

        self._counts_label = QLabel("")
        self._counts_label.setObjectName("stageInfo")
        root.addWidget(self._counts_label)

        self._preview = QListWidget()
        self._preview.setViewMode(QListWidget.ViewMode.IconMode)
        self._preview.setIconSize(QSize(96, 96))
        self._preview.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._preview.setMovement(QListWidget.Movement.Static)
        self._preview.setMinimumHeight(220)
        root.addWidget(self._preview, stretch=1)

        # 4 + 5: name + label + create
        out_row = QHBoxLayout()
        out_row.setSpacing(6)
        self._slug_edit = QLineEdit()
        self._slug_edit.setPlaceholderText("dataset name (slug)")
        out_row.addWidget(self._slug_edit, stretch=1)
        self._class_edit = QLineEdit()
        self._class_edit.setPlaceholderText("class label")
        out_row.addWidget(self._class_edit, stretch=1)
        self._create_btn = QPushButton("Create dataset")
        self._create_btn.setProperty("isPrimary", True)
        self._create_btn.setEnabled(False)
        self._create_btn.clicked.connect(self._on_create)
        out_row.addWidget(self._create_btn)
        root.addLayout(out_row)

        # Auto-fill slug/class from the query for the fast path.
        self._query_edit.textChanged.connect(self._autofill_names)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(450)
        self._preview_timer.timeout.connect(self._run_preview)

        self._index_timer = QTimer(self)
        self._index_timer.setInterval(700)
        self._index_timer.timeout.connect(self._poll_index)

    # ----------------------------------------------------------------- inputs
    def _on_browse(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose source image folder")
        if folder:
            self._folder_edit.setText(folder)

    def _autofill_names(self, text: str) -> None:
        q = str(text or "").strip().lower().replace(" ", "_")
        if q and not self._class_edit.text().strip():
            self._class_edit.setText(q)
        if q and not self._slug_edit.text().strip():
            self._slug_edit.setText(f"carve_{q}")

    def _on_threshold(self, value: int) -> None:
        self._threshold_label.setText(f"{value / 100:.2f}")
        self._schedule_preview()

    def _threshold_value(self) -> float:
        return self._threshold.value() / 100.0

    # ------------------------------------------------------------------ index
    def _on_index(self) -> None:
        folder = self._folder_edit.text().strip()
        if not folder:
            self.errorRaised.emit("Choose a source folder first.")
            return
        try:
            resp = self._http_post("/carve/index", {"folder": folder}) or {}
        except Exception as exc:
            self.errorRaised.emit(f"Index failed: {exc}")
            return
        self._index_cid = str(resp.get("correlation_id") or "")
        if not self._index_cid:
            self.errorRaised.emit("Index did not start.")
            return
        self._indexed = False
        self._index_btn.setEnabled(False)
        self._index_progress.setVisible(True)
        self._index_progress.setRange(0, 0)  # busy until counts arrive
        self._index_status.setText("Loading model / embedding…")
        self._index_timer.start()

    def _poll_index(self) -> None:
        if not self._index_cid:
            self._index_timer.stop()
            return
        try:
            st = self._http_get(f"/carve/index_progress/{self._index_cid}") or {}
        except Exception as exc:
            self._index_timer.stop()
            self.errorRaised.emit(f"Index progress failed: {exc}")
            return
        phase = str(st.get("phase") or "")
        cur, tot = int(st.get("current") or 0), int(st.get("total") or 0)
        if phase == "embedding" and tot:
            self._index_progress.setRange(0, tot)
            self._index_progress.setValue(cur)
            self._index_status.setText(f"Embedding {cur}/{tot}…")
        elif phase == "loading_model":
            self._index_status.setText("Loading CLIP model…")
        if phase == "error":
            self._index_timer.stop()
            self._index_progress.setVisible(False)
            self._index_btn.setEnabled(True)
            self.errorRaised.emit(f"Index error: {st.get('error')}")
            return
        if st.get("ready"):
            self._index_timer.stop()
            self._index_progress.setVisible(False)
            self._index_btn.setEnabled(True)
            self._indexed = True
            n = int(st.get("indexed") or 0)
            self._index_status.setText(f"Indexed {n} images — type a query to carve.")
            self._query_edit.setEnabled(True)
            self._threshold.setEnabled(True)
            self._schedule_preview()

    # ---------------------------------------------------------------- preview
    def _schedule_preview(self) -> None:
        if self._indexed and self._query_edit.text().strip():
            self._preview_timer.start()

    def _run_preview(self) -> None:
        query = self._query_edit.text().strip()
        if not (self._indexed and query):
            return
        try:
            r = self._http_post("/carve/preview", {
                "query": query, "threshold": self._threshold_value(), "sample": 24,
            }) or {}
        except Exception as exc:
            self.errorRaised.emit(f"Preview failed: {exc}")
            return
        pos = int(r.get("positive_count") or 0)
        neg = int(r.get("negative_count") or 0)
        self._counts_label.setText(
            f"{pos} match (>= {self._threshold_value():.2f}) · {neg} negatives · "
            f"score max {r.get('score_max')} mean {r.get('score_mean')}"
        )
        self._create_btn.setEnabled(pos > 0)
        self._populate_preview(r.get("sample") or [])

    def _populate_preview(self, sample: list[dict[str, Any]]) -> None:
        self._preview.clear()
        thr = self._threshold_value()
        for entry in sample:
            path = str(entry.get("path") or "")
            score = float(entry.get("score") or 0.0)
            item = QListWidgetItem(f"{score:.2f}")
            pm = QPixmap(path)
            if not pm.isNull():
                item.setIcon(QIcon(pm.scaled(
                    96, 96, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )))
            item.setToolTip(f"{Path(path).name}\nscore {score:.3f}")
            # Dim items below threshold so the cut is visible.
            if score < thr:
                item.setForeground(Qt.GlobalColor.gray)
            self._preview.addItem(item)

    # ----------------------------------------------------------------- create
    def _on_create(self) -> None:
        slug = self._slug_edit.text().strip()
        cls = self._class_edit.text().strip()
        query = self._query_edit.text().strip()
        if not (slug and cls and query):
            self.errorRaised.emit("Need a dataset name, class label, and query.")
            return
        self._create_btn.setEnabled(False)
        try:
            res = self._http_post("/carve/create", {
                "slug": slug, "class_name": cls, "query": query,
                "threshold": self._threshold_value(),
            }) or {}
        except Exception as exc:
            self._create_btn.setEnabled(True)
            self.errorRaised.emit(f"Create failed: {exc}")
            return
        counts = res.get("counts") or {}
        self.statusChanged.emit(
            f"Created dataset '{slug}' ({counts}) at {res.get('dataset_path')}"
        )
        self.datasetCreated.emit(slug)
        self._create_btn.setEnabled(True)
