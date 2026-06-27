from __future__ import annotations

import base64
import json
import time
import urllib.parse
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import (
    QDragEnterEvent,
    QDropEvent,
    QIcon,
    QImage,
    QKeySequence,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .cvops_theme import repolish
from .dataset_panel import _multipart_upload
from ..detection_backends import is_supported_video_test_model
from ..paths import CVOPS_STATE_DIR
from .test_range_subroutine import (
    SubroutineSession,
    collect_video_test_models,
    qimage_to_bgr_ndarray,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_REGISTRY_SENTINEL = "[REGISTRY]"
_DROP_PROMPT = "Drop or paste an image, or choose a file"
_RECENT_MAX = 5


class _DropZone(QFrame):
    fileDropped = pyqtSignal(str)
    imageDropped = pyqtSignal(QImage)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Compact: a fixed-ish working height leaves room for the recent strip
        # below instead of the zone swallowing all the panel's vertical space.
        self.setMinimumHeight(120)
        self.setMaximumHeight(240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setObjectName("dropZone")
        self.setProperty("state", "idle")
        self._preview_pixmap = QPixmap()
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(6)
        layout.addStretch(1)
        self._preview = QLabel()
        self._preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._preview.setMinimumSize(140, 90)
        self._preview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._preview.setVisible(False)
        layout.addWidget(self._preview, stretch=1)
        self._label = QLabel(_DROP_PROMPT)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)
        self._caption = QLabel("")
        self._caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._caption.setWordWrap(True)
        self._caption.setProperty("muted", True)
        self._caption.setVisible(False)
        layout.addWidget(self._caption)
        layout.addStretch(1)

    def set_text(self, text: str) -> None:
        self._label.setText(text)

    def show_preview(self, path: str, caption: str = "") -> None:
        pix = QPixmap(str(path))
        if pix.isNull():
            self._preview_pixmap = QPixmap()
            self._preview.clear()
            self._preview.setVisible(False)
            self._label.setVisible(True)
            self._caption.setVisible(False)
            return
        self._preview_pixmap = pix
        self._render_preview()
        self._preview.setVisible(True)
        self._label.setVisible(False)
        self._caption.setText(str(caption or Path(path).name))
        self._caption.setVisible(True)

    def clear_preview(self, prompt: str = _DROP_PROMPT) -> None:
        self._preview_pixmap = QPixmap()
        self._preview.clear()
        self._preview.setVisible(False)
        self._label.setText(prompt)
        self._label.setVisible(True)
        self._caption.clear()
        self._caption.setVisible(False)

    def set_state(self, state: str) -> None:
        self.setProperty("state", str(state or "idle"))
        repolish(self)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._render_preview()

    def _render_preview(self) -> None:
        if self._preview_pixmap.isNull():
            return
        scaled = self._preview_pixmap.scaled(
            max(120, self._preview.width()),
            max(100, self._preview.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._preview.setPixmap(scaled)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is not None:
            if md.hasUrls():
                for url in md.urls():
                    if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_SUFFIXES:
                        self.set_state("dragover")
                        event.acceptProposedAction()
                        return
            # Image dragged straight from a browser/app carries pixel data, not a file.
            if md.hasImage():
                self.set_state("dragover")
                event.acceptProposedAction()
                return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self.set_state("idle")
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is None:
            return
        if md.hasUrls():
            for url in md.urls():
                if url.isLocalFile():
                    path = url.toLocalFile()
                    if Path(path).suffix.lower() in IMAGE_SUFFIXES:
                        self.set_state("ready")
                        self.fileDropped.emit(path)
                        event.acceptProposedAction()
                        return
        if md.hasImage():
            image = QImage(md.imageData())
            if not image.isNull():
                self.set_state("ready")
                self.imageDropped.emit(image)
                event.acceptProposedAction()
                return


class _RecentStrip(QWidget):
    """Thin horizontal strip of recently used images for one-click re-testing.

    Fixed, shallow height; the thumbnail row scrolls horizontally when more
    images are present than fit, so it never grows into a tall block."""

    picked = pyqtSignal(str)
    _THUMB_W = 76
    _THUMB_H = 48

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 2, 0, 0)
        outer.setSpacing(2)
        header = QLabel("Recent")
        header.setProperty("muted", True)
        repolish(header)
        outer.addWidget(header)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("recentStripScroll")
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Just tall enough for one row of thumbnails plus the horizontal scrollbar.
        self._scroll.setFixedHeight(self._THUMB_H + 16)
        self._row_host = QWidget()
        self._row = QHBoxLayout(self._row_host)
        self._row.setContentsMargins(0, 0, 0, 0)
        self._row.setSpacing(6)
        self._empty = QLabel("Recently used images appear here.")
        self._empty.setProperty("muted", True)
        repolish(self._empty)
        self._row.addWidget(self._empty)
        self._row.addStretch(1)
        self._scroll.setWidget(self._row_host)
        outer.addWidget(self._scroll)
        self._buttons: list[QToolButton] = []

    def set_items(self, paths: list[str]) -> None:
        for btn in self._buttons:
            self._row.removeWidget(btn)
            btn.deleteLater()
        self._buttons.clear()
        valid = [p for p in paths if Path(p).is_file()][:_RECENT_MAX]
        self._empty.setVisible(not valid)
        for i, path in enumerate(valid):
            btn = QToolButton()
            btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            btn.setFixedSize(self._THUMB_W, self._THUMB_H)
            btn.setIconSize(QSize(self._THUMB_W - 6, self._THUMB_H - 6))
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(Path(path).name)
            pix = QPixmap(path)
            if not pix.isNull():
                btn.setIcon(QIcon(pix))
            btn.clicked.connect(lambda _checked=False, p=path: self.picked.emit(p))
            self._row.insertWidget(i, btn)
            self._buttons.append(btn)


class SubmitPanel(QWidget):
    """Drag-and-drop + file picker. POSTs base64 image to /jobs."""

    jobSubmitted = pyqtSignal(dict)
    submissionFailed = pyqtSignal(str)
    registryModelsChanged = pyqtSignal()
    registryResultReady = pyqtSignal(dict)  # local inference result, no job_id needed

    def __init__(
        self,
        base_url: str,
        scenarios_provider: Callable[[], list[dict[str, Any]]],
        http_get: Optional[Callable[[str], dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._scenarios_provider = scenarios_provider
        self._http_get = http_get
        self._pending_path: str = ""
        self._scenario_enabled_check: Optional[Callable[[str], bool]] = None
        self._selected_versions: dict[str, str] = {}
        self._selected_models: dict[tuple[str, str], str] = {}
        self._version_entries: dict[str, dict[str, Any]] = {}
        self._version_fetch_error = ""
        self._config_fallback_available = False
        self._cached_backbone_type = ""
        self._registry_session = SubroutineSession(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("Scenario:"))
        self._combo = QComboBox()
        self._combo.addItem(_REGISTRY_SENTINEL)
        self._combo.currentTextChanged.connect(self._on_scenario_changed)
        row.addWidget(self._combo, stretch=1)
        layout.addLayout(row)

        # -- Registry mode UI (shown only when [REGISTRY] is selected) --
        self._registry_box = QFrame()
        reg_layout = QVBoxLayout(self._registry_box)
        reg_layout.setContentsMargins(0, 2, 0, 2)
        reg_layout.setSpacing(4)

        reg_model_row = QHBoxLayout()
        reg_model_row.addWidget(QLabel("Model:"))
        self._registry_model_combo = QComboBox()
        self._registry_model_combo.currentIndexChanged.connect(self._on_registry_model_changed)
        reg_model_row.addWidget(self._registry_model_combo, stretch=1)
        self._registry_upload_weights_btn = QPushButton("[+ WEIGHTS]")
        self._registry_upload_weights_btn.setToolTip(
            "Browse for a weights file (.pt, .onnx, .mlpackage, …) and add it immediately."
        )
        self._registry_upload_weights_btn.clicked.connect(self._on_registry_browse_weights)
        reg_model_row.addWidget(self._registry_upload_weights_btn)
        reg_layout.addLayout(reg_model_row)

        self._registry_run_btn = QPushButton("[RUN REGISTRY MODEL]")
        self._registry_run_btn.clicked.connect(self._on_registry_run)
        reg_layout.addWidget(self._registry_run_btn)
        self._registry_box.setVisible(False)
        layout.addWidget(self._registry_box)

        self._scenario_rows = QWidget()
        sr_layout = QVBoxLayout(self._scenario_rows)
        sr_layout.setContentsMargins(0, 0, 0, 0)
        sr_layout.setSpacing(6)

        version_row = QHBoxLayout()
        version_row.addWidget(QLabel("Version:"))
        self._version_combo = QComboBox()
        self._version_combo.currentIndexChanged.connect(self._on_version_changed)
        version_row.addWidget(self._version_combo, stretch=1)
        sr_layout.addLayout(version_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self._model_combo, stretch=1)
        sr_layout.addLayout(model_row)

        upload_row = QHBoxLayout()
        upload_row.addWidget(QLabel("Registry id (optional):"))
        self._registry_upload_id = QLineEdit()
        self._registry_upload_id.setPlaceholderText("auto if empty")
        self._registry_upload_id.setToolTip(
            "Optional stable id for this upload (letters, digits, _, -). "
            "Weights are stored and registered so Range and training can reference them."
        )
        upload_row.addWidget(self._registry_upload_id, stretch=1)
        self._registry_upload_btn = QPushButton("Upload weights to registry…")
        self._registry_upload_btn.clicked.connect(self._on_registry_upload_weights)
        upload_row.addWidget(self._registry_upload_btn)
        sr_layout.addLayout(upload_row)

        layout.addWidget(self._scenario_rows)

        self._config_box = QGroupBox("Config (per-submit)")
        cfg_outer = QVBoxLayout(self._config_box)
        cfg_outer.setContentsMargins(10, 8, 10, 8)
        cfg_outer.setSpacing(6)

        self._yolo_row = QWidget()
        yolo_layout = QHBoxLayout(self._yolo_row)
        yolo_layout.setContentsMargins(0, 0, 0, 0)
        yolo_layout.setSpacing(8)
        yolo_layout.addWidget(QLabel("YOLO conf"))
        self._yolo_conf = QDoubleSpinBox()
        self._yolo_conf.setRange(0.0, 1.0)
        self._yolo_conf.setSingleStep(0.05)
        self._yolo_conf.setDecimals(2)
        self._yolo_conf.setValue(0.25)
        yolo_layout.addWidget(self._yolo_conf)
        yolo_layout.addWidget(QLabel("IOU"))
        self._yolo_iou = QDoubleSpinBox()
        self._yolo_iou.setRange(0.0, 1.0)
        self._yolo_iou.setSingleStep(0.05)
        self._yolo_iou.setDecimals(2)
        self._yolo_iou.setValue(0.70)
        yolo_layout.addWidget(self._yolo_iou)
        yolo_layout.addWidget(QLabel("max_det"))
        self._yolo_max_det = QSpinBox()
        self._yolo_max_det.setRange(1, 3000)
        self._yolo_max_det.setValue(300)
        yolo_layout.addWidget(self._yolo_max_det)
        yolo_layout.addStretch(1)
        cfg_outer.addWidget(self._yolo_row)

        self._face_row = QWidget()
        face_layout = QHBoxLayout(self._face_row)
        face_layout.setContentsMargins(0, 0, 0, 0)
        face_layout.setSpacing(8)
        face_layout.addWidget(QLabel("Face threshold"))
        self._face_threshold = QDoubleSpinBox()
        self._face_threshold.setRange(0.0, 1.0)
        self._face_threshold.setSingleStep(0.02)
        self._face_threshold.setDecimals(3)
        self._face_threshold.setValue(0.72)
        face_layout.addWidget(self._face_threshold)
        face_layout.addWidget(QLabel("margin"))
        self._face_margin = QDoubleSpinBox()
        self._face_margin.setRange(0.0, 0.2)
        self._face_margin.setSingleStep(0.005)
        self._face_margin.setDecimals(3)
        self._face_margin.setValue(0.045)
        face_layout.addWidget(self._face_margin)
        face_layout.addWidget(QLabel("top_k"))
        self._face_top_k = QSpinBox()
        self._face_top_k.setRange(1, 25)
        self._face_top_k.setValue(5)
        face_layout.addWidget(self._face_top_k)
        face_layout.addStretch(1)
        cfg_outer.addWidget(self._face_row)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        self._reset_cfg = QPushButton("Reset")
        self._reset_cfg.clicked.connect(self._reset_config)
        reset_row.addWidget(self._reset_cfg)
        cfg_outer.addLayout(reset_row)

        layout.addWidget(self._config_box)

        # Buttons sit ABOVE the drop zone so the zone can claim all remaining
        # height below them instead of being squeezed between controls.
        btn_row = QHBoxLayout()
        self._choose = QPushButton("Choose File...")
        self._choose.clicked.connect(self._pick_file)
        self._paste = QPushButton("Paste")
        self._paste.setToolTip("Paste an image from the clipboard (Cmd/Ctrl+V).")
        self._paste.clicked.connect(self._paste_from_clipboard)
        self._submit = QPushButton("Submit")
        self._submit.clicked.connect(self._submit_job)
        self._submit.setEnabled(False)
        btn_row.addWidget(self._choose)
        btn_row.addWidget(self._paste)
        btn_row.addStretch(1)
        btn_row.addWidget(self._submit)
        layout.addLayout(btn_row)

        self._drop = _DropZone()
        self._drop.fileDropped.connect(self._on_file_selected)
        self._drop.imageDropped.connect(self._on_image_object)
        # Capped height (set on the zone) keeps it compact; stretch lets it claim
        # up to that cap, and the trailing stretch below absorbs the remainder.
        layout.addWidget(self._drop, stretch=1)

        self._recent_strip = _RecentStrip()
        self._recent_strip.picked.connect(self._on_file_selected)
        layout.addWidget(self._recent_strip)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        layout.addWidget(self._status)
        layout.addStretch(1)

        paste_sc = QShortcut(QKeySequence(QKeySequence.StandardKey.Paste), self)
        paste_sc.activated.connect(self._paste_from_clipboard)
        self._recent_paths: list[str] = []
        self._load_recents()
        self._config_box.setVisible(False)
        self._yolo_row.setVisible(False)
        self._face_row.setVisible(False)
        self._sync_mode_widgets()
        self.refresh_scenarios()

    def set_ready_check(self, fn: Optional[Callable[[str], bool]]) -> None:
        """Install a predicate that decides whether a scenario is submit-ready."""
        self._scenario_enabled_check = fn
        self._update_submit_enabled()

    def refresh_scenarios(self) -> None:
        try:
            items = self._scenarios_provider() or []
        except Exception:
            items = []
        current = self._combo.currentText()
        current_version = self.current_version()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItem(_REGISTRY_SENTINEL)  # always first
        for item in items:
            name = str(item.get("name") or "")
            if not name:
                continue
            self._combo.addItem(name)
        if current:
            idx = self._combo.findText(current)
            if idx >= 0:
                self._combo.setCurrentIndex(idx)
        self._combo.blockSignals(False)
        if self._sync_mode_widgets():
            self._refresh_registry_models()
            self._update_submit_enabled()
            return
        self._refresh_versions(preferred_version=current_version)
        self._refresh_model_choices()
        self._cached_backbone_type = ""
        self._sync_config_visibility()

    def select_scenario(self, name: str) -> bool:
        """Select ``name`` in the scenario combo if present (refreshing first).

        Returns True when the scenario was found and selected. Used by the
        Collect & Edit import flow to drop the user straight onto a freshly
        created scenario, ready to train a model.
        """
        target = str(name or "").strip()
        if not target:
            return False
        self.refresh_scenarios()
        idx = self._combo.findText(target)
        if idx < 0:
            return False
        self._combo.setCurrentIndex(idx)
        return True

    def current_scenario(self) -> str:
        return self._combo.currentText().strip()

    def current_version(self) -> str:
        return str(self._version_combo.currentData() or "").strip()

    def current_model_artifact(self) -> str:
        return str(self._model_combo.currentData() or "").strip()

    def _pick_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if path:
            self._on_file_selected(path)

    def _on_file_selected(self, path: str) -> None:
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            self._status.setText(f"Invalid file: {path}")
            self._drop.clear_preview()
            self._drop.set_state("error")
            return
        if file_path.suffix.lower() not in IMAGE_SUFFIXES:
            self._status.setText(f"Unsupported file type: {file_path.suffix}")
            self._drop.clear_preview()
            self._drop.set_state("error")
            return
        self._pending_path = str(file_path)
        self._drop.show_preview(str(file_path), caption=f"[READY] {file_path.name}")
        self._drop.set_state("ready")
        self._remember_recent(str(file_path))
        self._update_submit_enabled()

    # ---- Recent images + clipboard paste ----

    def _recent_index_path(self) -> Path:
        return Path(CVOPS_STATE_DIR) / "range_recent.json"

    def _paste_cache_dir(self) -> Path:
        d = Path(CVOPS_STATE_DIR) / "range_pastes"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _load_recents(self) -> None:
        paths: list[str] = []
        try:
            raw = self._recent_index_path().read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                paths = [str(p) for p in data if isinstance(p, (str, Path))]
        except Exception:
            paths = []
        self._recent_paths = [p for p in paths if Path(p).is_file()][:_RECENT_MAX]
        self._recent_strip.set_items(self._recent_paths)

    def _save_recents(self) -> None:
        try:
            self._recent_index_path().write_text(
                json.dumps(self._recent_paths[:_RECENT_MAX]), encoding="utf-8"
            )
        except Exception:
            pass

    def _remember_recent(self, path: str) -> None:
        norm = str(Path(path))
        self._recent_paths = [norm] + [p for p in self._recent_paths if p != norm]
        self._recent_paths = self._recent_paths[:_RECENT_MAX]
        self._save_recents()
        self._recent_strip.set_items(self._recent_paths)

    def _on_image_object(self, image: QImage) -> None:
        """Persist a pasted/dropped image (which has no source file) then select it."""
        if image is None or image.isNull():
            self._status.setText("Clipboard has no usable image.")
            return
        out = self._paste_cache_dir() / f"paste_{int(time.time() * 1000)}.png"
        if not image.save(str(out), "PNG"):
            self._status.setText("Failed to save pasted image.")
            return
        self._on_file_selected(str(out))

    def _paste_from_clipboard(self) -> None:
        cb = QApplication.clipboard()
        if cb is None:
            return
        md = cb.mimeData()
        if md is not None and md.hasImage():
            self._on_image_object(QImage(md.imageData()))
            return
        if md is not None and md.hasUrls():
            for url in md.urls():
                if url.isLocalFile() and Path(url.toLocalFile()).suffix.lower() in IMAGE_SUFFIXES:
                    self._on_file_selected(url.toLocalFile())
                    return
        img = cb.image()
        if img is not None and not img.isNull():
            self._on_image_object(img)
            return
        self._status.setText("No image found in clipboard.")

    def _is_registry_mode(self) -> bool:
        return self._combo.currentText() == _REGISTRY_SENTINEL

    def _sync_mode_widgets(self) -> bool:
        is_reg = self._is_registry_mode()
        self._registry_box.setVisible(is_reg)
        self._scenario_rows.setVisible(not is_reg)
        self._config_box.setVisible(not is_reg)
        return is_reg

    def _on_scenario_changed(self, _text: str) -> None:
        is_reg = self._sync_mode_widgets()
        if is_reg:
            self._refresh_registry_models()
            self._update_submit_enabled()
            return
        self._refresh_versions()
        self._sync_config_visibility()

    def _on_registry_model_changed(self, _index: int) -> None:
        self._update_submit_enabled()

    def _load_scenario_backbone_type(self, scenario: str) -> str:
        if not scenario:
            return ""
        if scenario == self.current_scenario() and self._cached_backbone_type:
            return self._cached_backbone_type
        getter = self._http_get or self._get_json
        try:
            payload = getter(f"/scenarios/{urllib.parse.quote(scenario, safe='')}/status")
        except Exception:
            return ""
        btype = str(payload.get("backbone_type") or "")
        if scenario == self.current_scenario():
            self._cached_backbone_type = btype
        return btype

    def _sync_config_visibility(self) -> None:
        scen = self.current_scenario()
        btype = self._load_scenario_backbone_type(scen).strip().lower()
        show_yolo = btype in {"", "yolo_detection"}
        show_face = btype == "face_recognition"
        disable_submit = btype == "llm_fine_tuning"
        self._config_box.setVisible(bool(scen) and not disable_submit)
        self._yolo_row.setVisible(show_yolo)
        self._face_row.setVisible(show_face)

    def _reset_config(self) -> None:
        self._yolo_conf.setValue(0.25)
        self._yolo_iou.setValue(0.70)
        self._yolo_max_det.setValue(300)
        self._face_threshold.setValue(0.72)
        self._face_margin.setValue(0.045)
        self._face_top_k.setValue(5)

    def _on_version_changed(self, _index: int) -> None:
        scenario = self.current_scenario()
        version = self.current_version()
        if scenario:
            self._selected_versions[scenario] = version
        self._refresh_model_choices()
        self._update_submit_enabled()

    def _on_model_changed(self, _index: int) -> None:
        scenario = self.current_scenario()
        version = self.current_version()
        if scenario:
            self._selected_models[(scenario, version)] = self.current_model_artifact()
        self._update_submit_enabled()

    def _on_registry_upload_weights(self) -> None:
        scen = self.current_scenario()
        if not scen:
            self._status.setText("Select a scenario before uploading.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select model weights",
            "",
            "Model weights (*.pt *.onnx *.engine);;All files (*.*)",
        )
        if not path:
            return
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix not in {".pt", ".onnx", ".engine"}:
            self._status.setText(f"Unsupported extension {suffix} (use .pt, .onnx, or .engine).")
            return
        upload_id = self._registry_upload_id.text().strip()
        fields: dict[str, str] = {"scenario": scen}
        if upload_id:
            fields["run_version"] = upload_id
        url = self._base_url.rstrip("/") + "/models/upload"
        try:
            out = _multipart_upload(url, fields=fields, files={"file": p}, timeout=300.0)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.submissionFailed.emit(f"HTTP {exc.code}: {detail}")
            self._status.setText(f"Upload failed: HTTP {exc.code}")
            return
        except Exception as exc:
            self.submissionFailed.emit(str(exc))
            self._status.setText(f"Upload failed: {exc}")
            return
        ref = str((out or {}).get("model_ref") or "").strip()
        self._registry_upload_id.clear()
        self._refresh_model_choices()
        if ref:
            idx = self._model_combo.findData(ref)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._status.setText(f"Registered {ref}" if ref else "Weights registered.")
        self._update_submit_enabled()
        self.registryModelsChanged.emit()

    def _refresh_versions(self, preferred_version: str = "") -> None:
        scenario = self.current_scenario()
        runs = self._load_scenario_history(scenario)
        remembered = preferred_version or self._selected_versions.get(scenario, "")
        self._version_entries = {}
        self._config_fallback_available = False

        self._version_combo.blockSignals(True)
        self._version_combo.clear()
        for run in runs:
            version = str(run.get("version") or "").strip()
            if not version:
                continue
            self._version_entries[version] = run
            self._version_combo.addItem(self._format_version_label(run), version)

        if self._version_entries:
            target_version = remembered if remembered in self._version_entries else self._pick_default_version(runs)
            if target_version:
                index = self._version_combo.findData(target_version)
                if index >= 0:
                    self._version_combo.setCurrentIndex(index)
            if self._version_combo.currentIndex() < 0 and self._version_combo.count():
                self._version_combo.setCurrentIndex(0)
        elif scenario and self._is_config_fallback_ready(scenario):
            self._config_fallback_available = True
            self._version_combo.addItem("Configured Weights", "")
            self._version_combo.setCurrentIndex(0)

        self._version_combo.blockSignals(False)
        if scenario:
            self._selected_versions[scenario] = self.current_version()
        self._refresh_model_choices()
        self._update_submit_enabled()

    def _refresh_model_choices(self) -> None:
        scenario = self.current_scenario()
        version = self.current_version()
        selected_key = (scenario, version)
        remembered = self._selected_models.get(selected_key, "")
        run = self._selected_version_entry()

        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        self._model_combo.addItem("Default weights", "")

        if run is not None:
            final_file = str(run.get("final_model_file") or "").strip()
            final_name = str(run.get("final_model_name") or "").strip()
            if final_file:
                label = final_name or final_file
                self._model_combo.addItem(f"Custom: {label}", final_file)

        if self._http_get is not None and scenario:
            try:
                payload = self._http_get("/models")
                models = payload.get("models") if isinstance(payload, dict) else []
                if isinstance(models, list):
                    prefix = f"{scenario}:"
                    for m in models:
                        if not isinstance(m, dict):
                            continue
                        val = str(m.get("value") or "").strip()
                        if not val.startswith(prefix):
                            continue
                        if self._model_combo.findData(val) >= 0:
                            continue
                        tail = val[len(prefix) :]
                        self._model_combo.addItem(f"Registry: {tail}", val)
            except Exception:
                pass

        if remembered:
            idx = self._model_combo.findData(remembered)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.setEnabled(self._model_combo.count() > 1)
        self._model_combo.blockSignals(False)
        if scenario:
            self._selected_models[selected_key] = self.current_model_artifact()

    def _load_scenario_history(self, scenario: str) -> list[dict[str, Any]]:
        self._version_fetch_error = ""
        if not scenario:
            return []
        getter = self._http_get or self._get_json
        try:
            payload = getter(f"/scenarios/{urllib.parse.quote(scenario, safe='')}/history")
        except Exception as exc:
            self._version_fetch_error = str(exc)
            return []
        runs = payload.get("runs") if isinstance(payload, dict) else []
        if not isinstance(runs, list):
            return []
        return sorted(
            [run for run in runs if isinstance(run, dict)],
            key=lambda run: (
                int(run.get("version_number") or -1),
                str(run.get("trained_at") or ""),
                str(run.get("version") or ""),
            ),
            reverse=True,
        )

    def _pick_default_version(self, runs: list[dict[str, Any]]) -> str:
        ready_run = next((run for run in runs if bool(run.get("weights_ready"))), None)
        if ready_run is not None:
            return str(ready_run.get("version") or "")
        if runs:
            return str(runs[0].get("version") or "")
        return ""

    @staticmethod
    def _format_version_label(run: dict[str, Any]) -> str:
        version = str(run.get("version") or "unknown")
        status = str(run.get("status") or "unknown")
        map50 = str(run.get("map50") or "").strip()
        task = str(run.get("task") or "").strip()
        val_metric = str(run.get("val_metric") or "").strip()
        parts = [version, status]
        if map50:
            parts.append(f"mAP50 {map50}")
        elif val_metric:
            if task == "regression":
                parts.append(f"val_mae {val_metric}")
            elif task == "classification":
                parts.append(f"val_acc {val_metric}")
            else:
                parts.append(val_metric)
        if bool(run.get("verified")):
            parts.append("verified")
        return "  |  ".join(parts)

    def _is_config_fallback_ready(self, scenario: str) -> bool:
        if self._scenario_enabled_check is None or not scenario:
            return False
        try:
            return bool(self._scenario_enabled_check(scenario))
        except Exception:
            return False

    def _selected_version_entry(self) -> Optional[dict[str, Any]]:
        version = self.current_version()
        if not version:
            return None
        return self._version_entries.get(version)

    def _update_submit_enabled(self) -> None:
        if self._is_registry_mode():
            has_file = bool(self._pending_path)
            has_model = bool(self._registry_model_combo.currentData() or
                             self._registry_model_combo.currentText().strip())
            ready = has_file and has_model
            self._submit.setEnabled(ready)
            self._registry_run_btn.setEnabled(ready)
            if not has_file:
                self._status.setText("Drop or choose an image to run.")
            elif not has_model:
                self._status.setText("Select or upload a model.")
            else:
                self._status.setText("Ready — click [RUN REGISTRY MODEL].")
            return
        scen = self.current_scenario()
        btype = self._load_scenario_backbone_type(scen).strip().lower()
        if btype == "llm_fine_tuning":
            self._submit.setEnabled(False)
            self._status.setText("LLM fine-tuning scenarios are train-only in CV Ops v1.")
            return
        has_file = bool(self._pending_path)
        selected = self._selected_version_entry()
        meta = self.current_model_artifact()
        registry_ready = bool(scen and meta and meta.startswith(f"{scen}:"))
        if selected is not None:
            ready = bool(selected.get("weights_ready"))
        elif registry_ready:
            ready = True
        elif self._config_fallback_available:
            ready = True
        else:
            ready = self._is_config_fallback_ready(scen)
        self._submit.setEnabled(bool(scen) and has_file and ready)
        if not scen:
            self._status.setText("Select a scenario.")
        elif scen and not ready:
            version = self.current_version()
            if version:
                self._status.setText(f"Version '{version}' for scenario '{scen}' is not ready for inference.")
            elif self._version_fetch_error:
                self._status.setText(f"Version history unavailable: {self._version_fetch_error}")
            else:
                self._status.setText(f"Scenario '{scen}' has no usable version for inference.")
        elif not has_file:
            self._status.setText("Select an image to submit.")
        elif registry_ready:
            model = self.current_model_artifact()
            self._status.setText(f"Ready to run {scen} with registry weights ({model}).")
        elif self.current_version():
            model = self.current_model_artifact()
            suffix = f" ({model})" if model else ""
            self._status.setText(f"Ready to run {scen} on {self.current_version()}{suffix}.")
        elif self._config_fallback_available:
            self._status.setText(f"Ready to run {scen} on configured weights.")
        else:
            self._status.setText("")

    def _refresh_registry_models(self) -> None:
        current = self._registry_model_combo.currentData()
        self._registry_model_combo.blockSignals(True)
        self._registry_model_combo.clear()
        for label, path in collect_video_test_models(http_get=self._http_get):
            self._registry_model_combo.addItem(label, userData=path)
        if self._registry_model_combo.count() == 0:
            self._registry_model_combo.addItem("(no models found)", userData="")
        else:
            idx = self._registry_model_combo.findData(current)
            if idx >= 0:
                self._registry_model_combo.setCurrentIndex(idx)
        self._registry_model_combo.blockSignals(False)
        self._update_submit_enabled()

    def _on_registry_browse_weights(self) -> None:
        exts = "*.pt *.torchscript *.onnx *.engine *.mlmodel *.tflite"
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model Weights",
            "",
            f"Model Weights ({exts} *.mlpackage);;All Files (*.*)",
        )
        if not path_str:
            path_str = QFileDialog.getExistingDirectory(
                self, "Select .mlpackage Bundle (directory)", ""
            )
        if not path_str:
            return
        p = Path(path_str)
        if not is_supported_video_test_model(p):
            self._status.setText(f"Unsupported file: {p.name}")
            return
        key = str(p.resolve())
        idx = self._registry_model_combo.findData(key)
        if idx >= 0:
            self._registry_model_combo.setCurrentIndex(idx)
        else:
            if self._registry_model_combo.count() == 1 and not self._registry_model_combo.itemData(0):
                self._registry_model_combo.clear()
            self._registry_model_combo.addItem(p.name, userData=key)
            self._registry_model_combo.setCurrentIndex(self._registry_model_combo.count() - 1)
        self._status.setText(f"Weights loaded: {p.name}")
        self._update_submit_enabled()

    def _on_registry_run(self) -> None:
        import time
        path = self._pending_path
        if not path:
            self._status.setText("Drop or choose an image first.")
            return
        model_path = str(self._registry_model_combo.currentData() or "").strip()
        if not model_path:
            self._status.setText("Select a model first.")
            return
        try:
            from PyQt6.QtGui import QImage
            img = QImage(path)
            if img.isNull():
                self._status.setText("Could not load image.")
                return
            frame_bgr = qimage_to_bgr_ndarray(img)
        except Exception as exc:
            self._status.setText(f"Image load failed: {exc}")
            return

        self._registry_run_btn.setEnabled(False)
        self._submit.setEnabled(False)
        self._status.setText("Running…")
        _img_path = path
        _model = model_path
        try:
            _source_b64 = base64.b64encode(Path(_img_path).read_bytes()).decode("ascii")
        except Exception:
            _source_b64 = ""
        _start = time.time()

        def _done(detections: list) -> None:
            elapsed_ms = int((time.time() - _start) * 1000)
            import base64
            from PyQt6.QtGui import QImage
            from PyQt6.QtCore import QBuffer, QIODevice
            # build overlay pixmap with boxes drawn
            from .test_range_subroutine import SubroutineControlsWidget  # lazy, avoid circular
            src = QImage(_img_path)
            from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QFontMetrics
            from PyQt6.QtCore import QRectF
            overlay_b64 = ""
            try:
                out = src.convertToFormat(QImage.Format.Format_RGB32)
                painter = QPainter(out)
                colour = QColor("#22d3ee")
                fw, fh = src.width(), src.height()
                for det in detections:
                    try:
                        x1, y1 = float(det["x1"]), float(det["y1"])
                        x2, y2 = float(det["x2"]), float(det["y2"])
                    except Exception:
                        continue
                    painter.setPen(QPen(colour, 2))
                    painter.setBrush(QColor(34, 211, 238, 30))
                    painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                    label = str(det.get("label") or "")
                    conf = det.get("conf", det.get("confidence", det.get("score")))
                    tag = f"{label} {float(conf):.2f}" if label and conf is not None else label
                    if tag:
                        fm = painter.fontMetrics()
                        tw = fm.horizontalAdvance(tag) + 6
                        th = fm.height() + 2
                        tag_y = max(0.0, y1 - th)
                        painter.fillRect(QRectF(x1, tag_y, tw, th), colour)
                        painter.setPen(QPen(QColor("#000000")))
                        painter.drawText(int(x1 + 3), int(tag_y + fm.ascent() + 1), tag)
                painter.end()
                buf = QBuffer()
                buf.open(QIODevice.OpenModeFlag.WriteOnly)
                out.save(buf, "JPEG", 90)
                overlay_b64 = base64.b64encode(bytes(buf.data())).decode("ascii")
            except Exception:
                pass

            result = {
                "job_id": f"registry-{int(_start * 1000)}",
                "scenario": "[REGISTRY]",
                "job_type": "infer",
                "state": "complete",
                "backbone_type": "yolo_detection",
                "weights": _model,
                "elapsed_ms": elapsed_ms,
                "overlay_image": overlay_b64,
                "source_image_b64": _source_b64,
                "source_name": Path(_img_path).name,
                "raw": {
                    "detections": detections,
                    "signal": {"flag": False, "summary": f"{len(detections)} detection(s)"},
                },
                "summary": f"{len(detections)} detection(s) — {elapsed_ms} ms",
            }
            self._update_submit_enabled()
            self._status.setText(result["summary"])
            self.registryResultReady.emit(result)
            self.jobSubmitted.emit(result)

        def _fail(msg: str) -> None:
            self._update_submit_enabled()
            self._status.setText(f"Failed: {msg}")

        self._registry_session.start(
            model_path=model_path,
            device="",
            frame_bgr=frame_bgr,
            on_finished=_done,
            on_failed=_fail,
        )

    def _submit_job(self) -> None:
        if self._is_registry_mode():
            self._on_registry_run()
            return
        scen = self.current_scenario()
        path = self._pending_path
        if not scen or not path:
            return
        try:
            raw = Path(path).read_bytes()
        except Exception as exc:
            self.submissionFailed.emit(f"Failed to read image: {exc}")
            self._status.setText(f"Read error: {exc}")
            return
        payload: dict[str, Any] = {
            "scenario": scen,
            "version": self.current_version(),
            "model_artifact": self.current_model_artifact(),
            "image_b64": base64.b64encode(raw).decode("ascii"),
            "source": "cvops_ui",
        }
        btype = self._load_scenario_backbone_type(scen).strip().lower()
        if btype == "face_recognition":
            payload["backbone_config_override"] = {
                "threshold": float(self._face_threshold.value()),
                "margin_threshold": float(self._face_margin.value()),
                "top_k": int(self._face_top_k.value()),
            }
        else:
            payload["infer_overrides"] = {
                "conf": float(self._yolo_conf.value()),
                "iou": float(self._yolo_iou.value()),
                "max_det": int(self._yolo_max_det.value()),
            }
        try:
            result = self._post_json("/jobs", payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self.submissionFailed.emit(f"HTTP {exc.code}: {detail}")
            self._status.setText(f"Submit failed: HTTP {exc.code}")
            return
        except Exception as exc:
            self.submissionFailed.emit(str(exc))
            self._status.setText(f"Submit failed: {exc}")
            return
        self.jobSubmitted.emit(result)
        self._status.setText(f"Submitted {result.get('job_id', '')}")
        self._update_submit_enabled()

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self._base_url + path
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST", headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _get_json(self, path: str) -> dict[str, Any]:
        url = self._base_url + path
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}
