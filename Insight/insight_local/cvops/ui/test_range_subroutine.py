"""Range SUBROUTINE: ROI crop + on-demand model inference.

Used by Video Test Bench and Quick Test (image) result preview.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QEvent, QObject, QPointF, QRect, QRectF, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGraphicsItem,
    QGraphicsRectItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...ui.theme import text_css, theme_rgba
from ..detection_backends import (
    COCO_SKELETON,
    OnnxInferenceBackend,
    YuNetFaceDetectorBackend,
    _model_type_hint,
    extract_yolo_detections,
    extract_yolo_pose_detections,
    extract_yolo_seg_detections,
    is_supported_video_test_model,
    is_yunet_face_detector_model,
)

HttpCall = Callable[..., Any]

_SUBROUTINE_PALETTE = "#22d3ee"
_MIN_ROI_PX = 12
_REPO_ROOT = Path(__file__).resolve().parents[4]
_MODELS_DIR = _REPO_ROOT / "assets" / "models"
_OCR_MODELS_DIR = _MODELS_DIR / "ocr"
_MODEL_FILE_GLOBS = "*.pt *.torchscript *.onnx *.engine *.mlmodel *.tflite"


def _path_is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def _is_ocr_catalog_path(path: Path) -> bool:
    return _path_is_under(path, _OCR_MODELS_DIR)


def _catalog_model_label(path: Path, label: Optional[str] = None, *, is_ocr: bool = False) -> str:
    base = (label or path.name).strip() or path.name
    if (is_ocr or _is_ocr_catalog_path(path)) and not base.lower().startswith("ocr /"):
        return f"OCR / {base}"
    return base


def _iter_local_catalog_models() -> list[Path]:
    candidates: list[Path] = []
    for root in (_MODELS_DIR, _OCR_MODELS_DIR):
        if not root.exists():
            continue
        try:
            children = sorted(root.iterdir(), key=lambda p: p.name.lower())
        except OSError:
            continue
        candidates.extend(children)
    return candidates


def _unique_catalog_destination(src: Path, dest_dir: Path) -> Path:
    dest = dest_dir / src.name
    try:
        if dest.exists() and dest.resolve() == src.resolve():
            return dest
    except Exception:
        pass
    if not dest.exists():
        return dest
    index = 1
    while True:
        candidate = dest_dir / f"{src.stem}-{index}{src.suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def store_ocr_model_in_catalog(src: Path) -> Path:
    """Copy a supported Range model into the OCR catalog and return its stored path."""
    if not is_supported_video_test_model(src):
        raise ValueError(f"Unsupported file: {src.name}")
    _OCR_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resolved_src = src.expanduser().resolve()
    if _path_is_under(resolved_src, _OCR_MODELS_DIR):
        return resolved_src
    dest = _unique_catalog_destination(resolved_src, _OCR_MODELS_DIR)
    if resolved_src.is_dir():
        shutil.copytree(resolved_src, dest)
    else:
        shutil.copy2(resolved_src, dest)
    if not is_supported_video_test_model(dest):
        raise ValueError(f"Stored file is not a supported Range model: {dest.name}")
    return dest.resolve()


def collect_video_test_models(
    *,
    http_get: Optional[HttpCall] = None,
) -> list[tuple[str, str]]:
    """Return [(label, absolute_path), ...] for models runnable in Range."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    def _add(path: Path, label: Optional[str] = None) -> None:
        key = str(path.resolve())
        if key in seen or not is_supported_video_test_model(path):
            return
        seen.add(key)
        out.append((label or path.name, key))

    for path in _iter_local_catalog_models():
        _add(path, _catalog_model_label(path))

    if http_get is not None:
        try:
            payload = http_get("/models")
        except Exception:
            payload = None
        if isinstance(payload, dict):
            rows = payload.get("models")
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    raw = str(row.get("path") or row.get("weights") or row.get("value") or "").strip()
                    if not raw:
                        continue
                    p = Path(raw).expanduser()
                    if not p.is_absolute():
                        p = (_REPO_ROOT / p).resolve()
                    origin = str(row.get("origin") or "").strip().lower()
                    label = _catalog_model_label(
                        p,
                        str(row.get("name") or row.get("id") or p.name),
                        is_ocr=(origin == "ocr"),
                    )
                    _add(p, label)
    return out


def _model_supports_to(model_path: str) -> bool:
    return Path(model_path).suffix.lower() in {".pt", ".torchscript"}


def _resolve_auto_device() -> str:
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    try:
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def qimage_to_bgr_ndarray(img: QImage):
    """RGB32 QImage -> OpenCV BGR ndarray."""
    import numpy as np  # type: ignore

    if img.format() != QImage.Format.Format_RGB32:
        img = img.convertToFormat(QImage.Format.Format_RGB32)
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(h * w * 4)
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, w, 4))
    return arr[:, :, :3][:, :, ::-1].copy()


def crop_qimage(img: QImage, roi: QRect) -> QImage:
    r = roi.normalized()
    r = r.intersected(QRect(0, 0, img.width(), img.height()))
    if r.width() < 1 or r.height() < 1:
        return QImage()
    return img.copy(r)


def offset_detections(detections: list[dict], ox: int, oy: int) -> list[dict]:
    out: list[dict] = []
    for det in detections or []:
        if not isinstance(det, dict):
            continue
        row = dict(det)
        for k in ("x1", "y1", "x2", "y2"):
            if k in row:
                try:
                    row[k] = float(row[k]) + (ox if k.startswith("x") else oy)
                except Exception:
                    pass
        if "bbox" in row and isinstance(row["bbox"], (list, tuple)) and len(row["bbox"]) >= 4:
            try:
                b = list(row["bbox"])
                b[0] = float(b[0]) + ox
                b[1] = float(b[1]) + oy
                b[2] = float(b[2]) + ox
                b[3] = float(b[3]) + oy
                row["bbox"] = b
            except Exception:
                pass
        out.append(row)
    return out


def pixmap_letterbox_rect(label_w: int, label_h: int, pix: QPixmap) -> QRect:
    """Where the pixmap is drawn inside the label (KeepAspectRatio)."""
    if pix.isNull() or label_w <= 0 or label_h <= 0:
        return QRect(0, 0, 0, 0)
    pw, ph = pix.width(), pix.height()
    scale = min(label_w / pw, label_h / ph)
    dw = int(pw * scale)
    dh = int(ph * scale)
    x = (label_w - dw) // 2
    y = (label_h - dh) // 2
    return QRect(x, y, dw, dh)


def _letterbox_rect(label_w: int, label_h: int, image_w: int, image_h: int) -> QRect:
    if image_w <= 0 or image_h <= 0 or label_w <= 0 or label_h <= 0:
        return QRect()
    scale = min(label_w / image_w, label_h / image_h)
    dw = int(image_w * scale)
    dh = int(image_h * scale)
    x = (label_w - dw) // 2
    y = (label_h - dh) // 2
    return QRect(x, y, dw, dh)


def label_point_to_image(
    point: QPointF,
    label_w: int,
    label_h: int,
    image_w: int,
    image_h: int,
) -> QPointF:
    """Map widget coords to full-resolution image coords."""
    if image_w <= 0 or image_h <= 0:
        return QPointF()
    disp = _letterbox_rect(label_w, label_h, image_w, image_h)
    if disp.width() <= 0 or disp.height() <= 0:
        return QPointF()
    lx = point.x() - disp.x()
    ly = point.y() - disp.y()
    if lx < 0 or ly < 0 or lx > disp.width() or ly > disp.height():
        return QPointF(-1, -1)
    ix = lx * image_w / disp.width()
    iy = ly * image_h / disp.height()
    return QPointF(ix, iy)


class SubroutineInferenceWorker(QObject):
    finished = pyqtSignal(list)
    failed = pyqtSignal(str)

    @pyqtSlot(str, str, object)
    def run(self, model_path: str, device: str, frame_bgr: object) -> None:
        try:
            import numpy as np  # type: ignore
        except Exception as exc:
            self.failed.emit(f"numpy unavailable: {exc}")
            return
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            self.failed.emit("empty crop")
            return
        fh, fw = frame_bgr.shape[:2]
        resolved = (device or "").strip() or _resolve_auto_device()
        suffix = Path(model_path).suffix.lower()
        mtype = _model_type_hint(model_path)
        try:
            if is_yunet_face_detector_model(model_path):
                backend = YuNetFaceDetectorBackend(model_path)
                detections = backend.predict(frame_bgr)
            elif suffix == ".onnx" and not is_yunet_face_detector_model(model_path):
                # ONNX path: use our OnnxInferenceBackend for all types
                backend_onnx = OnnxInferenceBackend(model_path, model_type=mtype)
                detections = backend_onnx.predict(frame_bgr)
            else:
                from ultralytics import YOLO  # type: ignore

                model = YOLO(model_path)
                if _model_supports_to(model_path):
                    model.to(resolved)
                predict_kwargs: dict[str, Any] = {"verbose": False}
                if _model_supports_to(model_path):
                    predict_kwargs["device"] = resolved
                results = model.predict(frame_bgr, **predict_kwargs)
                # Route to the appropriate extractor based on what the model produced
                if mtype == "pose" or any(getattr(r, "keypoints", None) is not None for r in results):
                    detections = extract_yolo_pose_detections(results, fw, fh)
                elif mtype == "seg" or any(getattr(r, "masks", None) is not None for r in results):
                    detections = extract_yolo_seg_detections(results, fw, fh)
                else:
                    detections = extract_yolo_detections(results, fw, fh)
            self.finished.emit(list(detections or []))
        except Exception as exc:
            self.failed.emit(str(exc))


class _PreviewBlinkOverlay(QWidget):
    """Transparent overlay that blinks a gold/blue box over the annotated result label.

    Resizes itself to always cover the parent label. Coords are passed in
    label-display space (i.e. already accounting for letterboxing).
    """

    _TICK_MS = 33
    _CYCLE_MS = 500
    _BLINK_DURATION_MS = 2000
    _GOLD = QColor(255, 200, 0)
    _BLUE = QColor(0, 180, 255)

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setVisible(False)
        self._rect: Optional[QRectF] = None
        self._elapsed = 0
        self._timer = QTimer(self)
        self._timer.setInterval(self._TICK_MS)
        self._timer.timeout.connect(self._tick)

    def start(self, rect: QRectF) -> None:
        self._rect = rect
        self._elapsed = 0
        self.setVisible(True)
        self.raise_()
        self.resize(self.parent().size())  # type: ignore[union-attr]
        self._timer.start()
        self.update()

    def stop(self) -> None:
        self._timer.stop()
        self._rect = None
        self.setVisible(False)

    def _tick(self) -> None:
        self._elapsed += self._TICK_MS
        if self._elapsed >= self._BLINK_DURATION_MS:
            self.stop()
            return
        self.update()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        if self._rect is None or self._rect.isEmpty():
            return
        t = (self._elapsed % self._CYCLE_MS) / self._CYCLE_MS
        alpha_t = 1.0 - abs(t * 2 - 1.0)
        r = int(self._GOLD.red()   + (self._BLUE.red()   - self._GOLD.red())   * alpha_t)
        g = int(self._GOLD.green() + (self._BLUE.green() - self._GOLD.green()) * alpha_t)
        b = int(self._GOLD.blue()  + (self._BLUE.blue()  - self._GOLD.blue())  * alpha_t)
        colour = QColor(r, g, b, 220)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(colour, 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        fill = QColor(r, g, b, int(40 * alpha_t))
        painter.setBrush(fill)
        painter.drawRect(self._rect)
        painter.end()


class ModelCatalogDialog(QDialog):
    """Model picker for the subroutine.

    Supports two modes:
      - Single: exactly one model checked; checking another auto-unchecks.
      - Fusion: 2+ models checked; results are overlaid.

    Also offers a [+ Store OCR Model] button to add OCR weights on the fly.
    """

    MODE_SINGLE = "single"
    MODE_FUSION = "fusion"

    def __init__(
        self,
        models: list[tuple[str, str]],
        *,
        preselected: Optional[set[str]] = None,
        mode: str = MODE_SINGLE,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Configure Models")
        self.setMinimumSize(460, 420)
        pre = preselected or set()
        self._suppress_item_changed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # --- Mode toggle ---
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        mode_lbl = QLabel("Mode")
        mode_lbl.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {text_css(0.84)};")
        mode_row.addWidget(mode_lbl)
        self._single_rb = QRadioButton("Single shot")
        self._single_rb.setToolTip("Pick one model. Each run produces results from that model only.")
        self._fusion_rb = QRadioButton("Fusion")
        self._fusion_rb.setToolTip("Pick two or more models. Every run executes them all and overlays results.")
        self._mode_group = QButtonGroup(self)
        self._mode_group.addButton(self._single_rb)
        self._mode_group.addButton(self._fusion_rb)
        if mode == self.MODE_FUSION:
            self._fusion_rb.setChecked(True)
        else:
            self._single_rb.setChecked(True)
        for rb in (self._single_rb, self._fusion_rb):
            rb.setStyleSheet(f"font-size: 11px; color: {text_css(0.84)};")
            mode_row.addWidget(rb)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        # --- Description ---
        self._desc = QLabel()
        self._desc.setStyleSheet(f"font-size: 10px; color: {text_css(0.6)};")
        self._desc.setWordWrap(True)
        layout.addWidget(self._desc)

        # --- Model list ---
        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for label, path in models:
            self._add_list_item(label, path, checked=(path in pre))
        layout.addWidget(self._list, stretch=1)

        # --- Bottom row: count + browse ---
        bottom_row = QHBoxLayout()
        self._count_lbl = QLabel("0 selected")
        self._count_lbl.setStyleSheet(f"font-size: 10px; color: {text_css(0.6)};")
        bottom_row.addWidget(self._count_lbl)
        bottom_row.addStretch(1)
        self._browse_btn = QPushButton("[+ Store OCR Model]")
        self._browse_btn.setToolTip("Copy supported OCR weights into assets/models/ocr for Range fusion.")
        self._browse_btn.clicked.connect(self._on_browse_weights)
        bottom_row.addWidget(self._browse_btn)
        layout.addLayout(bottom_row)

        # --- OK / Cancel ---
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        layout.addWidget(self._button_box)

        # --- Wiring ---
        self._list.itemChanged.connect(self._on_item_changed)
        self._single_rb.toggled.connect(self._on_mode_changed)
        self._fusion_rb.toggled.connect(self._on_mode_changed)
        self._on_mode_changed()  # initializes desc, OK button label, enforces single-mode constraint

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def _add_list_item(self, label: str, path: str, *, checked: bool) -> None:
        item = QListWidgetItem(label)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setToolTip(path)
        self._list.addItem(item)

    def selected_paths(self) -> list[str]:
        out: list[str] = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                path = str(item.data(Qt.ItemDataRole.UserRole) or "")
                if path:
                    out.append(path)
        return out

    def mode(self) -> str:
        return self.MODE_FUSION if self._fusion_rb.isChecked() else self.MODE_SINGLE

    # ------------------------------------------------------------------
    # Mode + selection handling
    # ------------------------------------------------------------------

    def _on_mode_changed(self) -> None:
        if self._fusion_rb.isChecked():
            self._desc.setText(
                "Fusion mode: pick at least two models. Every subroutine run executes all of "
                "them on the same crop and overlays the results."
            )
            self._button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Save Fusion Set")
        else:
            self._desc.setText(
                "Single-shot mode: pick exactly one model. Each subroutine run produces results "
                "from that model alone."
            )
            self._button_box.button(QDialogButtonBox.StandardButton.Ok).setText("Save Selection")
            # Enforce single-selection invariant on mode switch: keep only the first check
            self._enforce_single_check(initial=True)
        self._refresh_count()

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._suppress_item_changed:
            return
        if self._single_rb.isChecked() and item.checkState() == Qt.CheckState.Checked:
            # Uncheck all other items so exactly one stays checked
            self._suppress_item_changed = True
            try:
                for i in range(self._list.count()):
                    other = self._list.item(i)
                    if other is not item and other.checkState() == Qt.CheckState.Checked:
                        other.setCheckState(Qt.CheckState.Unchecked)
            finally:
                self._suppress_item_changed = False
        self._refresh_count()

    def _enforce_single_check(self, *, initial: bool = False) -> None:
        """In single mode, keep only the first checked item checked."""
        self._suppress_item_changed = True
        try:
            seen_one = False
            for i in range(self._list.count()):
                item = self._list.item(i)
                if item.checkState() == Qt.CheckState.Checked:
                    if seen_one:
                        item.setCheckState(Qt.CheckState.Unchecked)
                    else:
                        seen_one = True
        finally:
            self._suppress_item_changed = False

    def _refresh_count(self) -> None:
        n = len(self.selected_paths())
        if self._fusion_rb.isChecked():
            need = max(0, 2 - n)
            txt = f"{n} selected" + (f" — need {need} more for fusion" if need else "")
        else:
            txt = f"{n} selected" + (" — pick one" if n == 0 else "")
        self._count_lbl.setText(txt)
        # Enable OK only when the rule is satisfied
        ok_btn = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        if self._fusion_rb.isChecked():
            ok_btn.setEnabled(n >= 2)
        else:
            ok_btn.setEnabled(n == 1)

    # ------------------------------------------------------------------
    # Browse weights
    # ------------------------------------------------------------------

    def _on_browse_weights(self) -> None:
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select OCR Model Weights",
            "",
            f"Model Weights ({_MODEL_FILE_GLOBS} *.mlpackage);;All Files (*.*)",
        )
        if not path_str:
            path_str = QFileDialog.getExistingDirectory(
                self,
                "Select .mlpackage Bundle (directory)",
                "",
            )
        if not path_str:
            return
        p = Path(path_str)
        if not is_supported_video_test_model(p):
            self._count_lbl.setText(f"Unsupported file: {p.name}")
            return
        try:
            stored = store_ocr_model_in_catalog(p)
        except Exception as exc:
            self._count_lbl.setText(f"Could not store {p.name}: {exc}")
            return
        key = str(stored.resolve())
        # If already in the list, just check it
        for i in range(self._list.count()):
            item = self._list.item(i)
            if str(item.data(Qt.ItemDataRole.UserRole) or "") == key:
                item.setCheckState(Qt.CheckState.Checked)
                self._list.scrollToItem(item)
                return
        self._add_list_item(_catalog_model_label(stored, stored.name, is_ocr=True), key, checked=True)
        self._list.scrollToBottom()


class SubroutineControlsWidget(QWidget):
    """Full-tab subroutine panel.

    Layout (top to bottom):
      [status bar]
      [annotated result image — with bounding boxes drawn on crop]
      [raw crop image]
      ── splitter boundary ──
      [model picker]
      [run / try-again / clear buttons]
    """

    runRequested = pyqtSignal(str, str)       # model_path, device_id
    dismissed = pyqtSignal()
    highlightRequested = pyqtSignal(int)      # detection index, or -1 to clear
    fusionRequested = pyqtSignal(list)        # list[str] of selected model paths
    reAnalyzeRequested = pyqtSignal()         # fresh-frame recapture + re-run

    def __init__(
        self,
        *,
        http_get: Optional[HttpCall] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._http_get = http_get
        self._raw_crop: Optional[QImage] = None
        self._detections: list[dict] = []
        self._highlighted_row: int = -1
        self._crop_frame_w: int = 0
        self._crop_frame_h: int = 0
        # Catalog state: either single model or fusion set, controlled by the catalog dialog.
        self._catalog_mode: str = ModelCatalogDialog.MODE_SINGLE
        self._catalog_single_path: str = ""
        self._catalog_fusion_paths: list[str] = []
        self._annotated_source_pixmap = QPixmap()
        self._raw_source_pixmap = QPixmap()
        # Cached last-render context for layer-toggle re-renders
        self._last_render_crop: Optional[QImage] = None
        self._last_render_dets: list[dict] = []
        # Streaming-fusion state
        self._stream_active: bool = False
        self._stream_per_model: list[tuple[str, list[dict]]] = []
        self._stream_pending: list[str] = []   # model labels still to run
        self._stream_crop_w: int = 0
        self._stream_crop_h: int = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # -- Layer toggles: boxes, masks, pose --
        toggles_row = QHBoxLayout()
        toggles_row.setContentsMargins(0, 0, 0, 0)
        toggles_row.setSpacing(10)
        toggles_lbl = QLabel("Show")
        toggles_lbl.setStyleSheet(f"font-size: 10px; color: {text_css(0.6)}; border: none;")
        toggles_row.addWidget(toggles_lbl)
        self._show_boxes_cb = QCheckBox("Boxes")
        self._show_boxes_cb.setChecked(True)
        self._show_boxes_cb.setToolTip("Draw bounding boxes for detection / pose results. Face boxes are always shown.")
        self._show_boxes_cb.toggled.connect(self._on_layer_toggled)
        self._show_masks_cb = QCheckBox("Masks")
        self._show_masks_cb.setChecked(True)
        self._show_masks_cb.setToolTip("Draw segmentation polygons.")
        self._show_masks_cb.toggled.connect(self._on_layer_toggled)
        self._show_pose_cb = QCheckBox("Pose")
        self._show_pose_cb.setChecked(True)
        self._show_pose_cb.setToolTip("Draw pose keypoint skeletons.")
        self._show_pose_cb.toggled.connect(self._on_layer_toggled)
        self._stream_cb = QCheckBox("Stream")
        self._stream_cb.setChecked(True)
        self._stream_cb.setToolTip("In fusion mode, paint each model's results as they finish instead of waiting for all.")
        self._show_labels_cb = QCheckBox("Labels")
        self._show_labels_cb.setChecked(True)
        self._show_labels_cb.setToolTip("Draw the class/confidence text tag on each result. Turn off when the tags clutter the view.")
        self._show_labels_cb.toggled.connect(self._on_layer_toggled)
        for cb in (
            self._show_boxes_cb,
            self._show_masks_cb,
            self._show_pose_cb,
            self._stream_cb,
            self._show_labels_cb,
        ):
            cb.setStyleSheet(f"font-size: 10px; color: {text_css(0.84)};")
            toggles_row.addWidget(cb)
        toggles_row.addStretch(1)
        root.addLayout(toggles_row)

        # -- Status bar --
        self._status = QLabel("Draw a region on the video, then run a model on that crop.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"font-size: 10px; color: {text_css(0.84)};")
        root.addWidget(self._status)

        # -- Image area: annotated result top, raw crop below --
        self._img_splitter = QSplitter(Qt.Orientation.Vertical)
        self._img_splitter.setChildrenCollapsible(False)
        self._img_splitter.setHandleWidth(3)
        self._img_splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        self._annotated_host = QFrame()
        self._annotated_host.setStyleSheet(
            f"QFrame {{ background: {theme_rgba('panel', 0.55)}; border: 1px solid {theme_rgba('accent_dark', 0.18)}; }}"
        )
        ah_layout = QVBoxLayout(self._annotated_host)
        ah_layout.setContentsMargins(4, 4, 4, 4)
        ah_lbl = QLabel("Result")
        ah_lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {theme_rgba('accent_dark', 0.9)};"
            " border: none; background: transparent;"
        )
        ah_layout.addWidget(ah_lbl)
        self._annotated_preview = QLabel()
        self._annotated_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._annotated_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._annotated_preview.setMinimumHeight(80)
        self._annotated_preview.setStyleSheet("border: none; background: transparent;")
        self._annotated_preview.setText("No result yet.")
        ah_layout.addWidget(self._annotated_preview, stretch=1)

        # Blink overlay sits on top of the annotated preview label
        self._blink_overlay = _PreviewBlinkOverlay(self._annotated_preview)
        self._annotated_preview.installEventFilter(self)

        self._img_splitter.addWidget(self._annotated_host)

        self._raw_host = QFrame()
        self._raw_host.setStyleSheet(
            f"QFrame {{ background: {theme_rgba('panel', 0.42)}; border: 1px solid {theme_rgba('accent_dark', 0.12)}; }}"
        )
        rh_layout = QVBoxLayout(self._raw_host)
        rh_layout.setContentsMargins(4, 4, 4, 4)
        rh_lbl = QLabel("Raw crop")
        rh_lbl.setStyleSheet(
            f"font-size: 10px; font-weight: 700; color: {text_css(0.6)};"
            " border: none; background: transparent;"
        )
        rh_layout.addWidget(rh_lbl)
        self._raw_preview = QLabel()
        self._raw_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._raw_preview.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._raw_preview.setMinimumHeight(60)
        self._raw_preview.setStyleSheet("border: none; background: transparent;")
        rh_layout.addWidget(self._raw_preview, stretch=1)
        self._img_splitter.addWidget(self._raw_host)

        self._img_splitter.setSizes([220, 120])
        root.addWidget(self._img_splitter)

        # -- Detection table --
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Label", "Conf", "BBox", "Type"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._table.setMaximumHeight(160)
        self._table.setVisible(False)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setToolTip("Click a row to blink its detection box on the video.")
        self._table.cellClicked.connect(self._on_table_row_clicked)
        root.addWidget(self._table)

        # -- Controls: catalog button + action buttons --
        controls_frame = QFrame()
        controls_frame.setStyleSheet(
            f"QFrame {{ border-top: 1px solid {theme_rgba('accent_dark', 0.18)}; background: transparent; }}"
        )
        cf_layout = QVBoxLayout(controls_frame)
        cf_layout.setContentsMargins(0, 8, 0, 0)
        cf_layout.setSpacing(6)

        # Catalog row: one button that opens the model picker dialog
        catalog_row = QHBoxLayout()
        self._catalog_btn = QPushButton("[CONFIGURE MODELS ▸]")
        self._catalog_btn.setToolTip("Open the model catalog — pick single-shot model or fusion set, and add weights.")
        self._catalog_btn.clicked.connect(self._on_catalog_clicked)
        catalog_row.addWidget(self._catalog_btn, stretch=1)
        cf_layout.addLayout(catalog_row)

        # Active model summary line (e.g. "Single: yolov8n.pt" or "Fusion: 3 models")
        self._catalog_summary = QLabel("No model selected — open the catalog to pick one.")
        self._catalog_summary.setStyleSheet(f"font-size: 10px; color: {text_css(0.6)}; border: none;")
        self._catalog_summary.setWordWrap(True)
        cf_layout.addWidget(self._catalog_summary)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("[RUN SUBROUTINE]")
        self._run_btn.clicked.connect(self._emit_run)
        btn_row.addWidget(self._run_btn)
        self._try_again_btn = QPushButton("[TRY AGAIN]")
        self._try_again_btn.setVisible(False)
        self._try_again_btn.clicked.connect(self._emit_run)
        btn_row.addWidget(self._try_again_btn)
        self._dismiss_btn = QPushButton("[CLEAR]")
        self._dismiss_btn.setVisible(False)
        self._dismiss_btn.clicked.connect(self._on_dismiss)
        btn_row.addWidget(self._dismiss_btn)
        self._reanalyze_btn = QPushButton("[RE-ANALYZE]")
        self._reanalyze_btn.setToolTip(
            "Capture a fresh frame at the current playback position using the same ROI and re-run."
        )
        self._reanalyze_btn.setVisible(False)
        self._reanalyze_btn.clicked.connect(self._on_reanalyze)
        btn_row.addWidget(self._reanalyze_btn)
        btn_row.addStretch(1)
        cf_layout.addLayout(btn_row)

        root.addWidget(controls_frame)

        self._refresh_run_mode()
        self._apply_preview_heights(None)

    def refresh_models(self) -> None:
        """Validate cached selection against the current registry.

        Drops any catalog selections whose paths no longer exist.
        """
        available = {path for _, path in collect_video_test_models(http_get=self._http_get)}
        if self._catalog_single_path and self._catalog_single_path not in available:
            self._catalog_single_path = ""
        self._catalog_fusion_paths = [p for p in self._catalog_fusion_paths if p in available]
        self._refresh_run_mode()

    def current_model_path(self) -> str:
        """Return the active single-shot model path, or empty string if none / fusion mode."""
        if self._catalog_mode == ModelCatalogDialog.MODE_SINGLE:
            return self._catalog_single_path
        return ""

    def raw_crop(self) -> Optional[QImage]:
        return self._raw_crop

    def show_raw_crop(self, crop: QImage) -> None:
        self._raw_crop = crop
        self._apply_preview_heights(crop)
        self._show_raw_pixmap(QPixmap.fromImage(crop))

    def open_for_roi(self, raw_crop: Optional[QImage] = None) -> None:
        self._annotated_preview.clear()
        self._annotated_preview.setText("Waiting for result...")
        self._table.setVisible(False)
        self._table.setRowCount(0)
        self._try_again_btn.setVisible(False)
        self._dismiss_btn.setVisible(True)
        self._reanalyze_btn.setVisible(True)
        # Run button enablement is driven by catalog state, not combo presence
        self._refresh_run_mode()

        has_selection = bool(self.has_fusion_set() or self._catalog_single_path)
        if raw_crop is not None and not raw_crop.isNull():
            self._raw_crop = raw_crop
            self._apply_preview_heights(raw_crop)
            self._show_raw_pixmap(QPixmap.fromImage(raw_crop))
            self._status.setText(
                "Region captured. Running..." if has_selection
                else "Region captured. Pick a model and run."
            )
        else:
            self._raw_crop = None
            self._apply_preview_heights(None)
            self._raw_preview.clear()
            self._raw_preview.setText("No frame captured.")
            self._status.setText("Region selected — pick a model and run.")

    def _show_raw_pixmap(self, pix: QPixmap) -> None:
        if pix.isNull():
            return
        self._raw_source_pixmap = pix
        self._set_scaled_preview_pixmap(self._raw_preview, pix, fallback_w=260, fallback_h=120)

    def _set_scaled_preview_pixmap(
        self,
        label: QLabel,
        pix: QPixmap,
        *,
        fallback_w: int,
        fallback_h: int,
    ) -> None:
        if pix.isNull():
            label.setPixmap(QPixmap())
            return
        w = max(1, label.width() or fallback_w)
        h = max(1, label.height() or fallback_h)
        label.setPixmap(
            pix.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def _show_annotated_pixmap(self, pix: QPixmap) -> None:
        self._annotated_source_pixmap = pix
        self._set_scaled_preview_pixmap(self._annotated_preview, pix, fallback_w=260, fallback_h=220)

    def _preview_height_from_crop(self, crop: Optional[QImage]) -> tuple[int, int]:
        if crop is None or crop.isNull():
            return 220, 120
        aspect = max(0.4, min(4.0, crop.height() / max(1, crop.width())))
        available_w = max(240, (self._img_splitter.width() or self.width() or 520) - 12)
        natural_h = int(round(available_w * aspect))
        annotated_h = max(160, min(360, natural_h))
        raw_h = max(110, min(280, int(round(annotated_h * 0.72))))
        return annotated_h, raw_h

    def _apply_preview_heights(self, crop: Optional[QImage]) -> None:
        annotated_h, raw_h = self._preview_height_from_crop(crop)
        chrome_h = 28
        annotated_host_h = annotated_h + chrome_h
        raw_host_h = raw_h + chrome_h
        for widget, height in (
            (self._annotated_preview, annotated_h),
            (self._raw_preview, raw_h),
            (self._annotated_host, annotated_host_h),
            (self._raw_host, raw_host_h),
        ):
            widget.setMinimumHeight(height)
            widget.setMaximumHeight(height)
        total_h = annotated_host_h + raw_host_h + self._img_splitter.handleWidth()
        self._img_splitter.setMinimumHeight(total_h)
        self._img_splitter.setMaximumHeight(total_h)
        self._img_splitter.setSizes([annotated_host_h, raw_host_h])

    def _refresh_preview_scaling(self) -> None:
        crop = self._raw_crop if self._raw_crop is not None and not self._raw_crop.isNull() else self._last_render_crop
        self._apply_preview_heights(crop)
        if not self._raw_source_pixmap.isNull():
            self._set_scaled_preview_pixmap(self._raw_preview, self._raw_source_pixmap, fallback_w=260, fallback_h=120)
        if not self._annotated_source_pixmap.isNull():
            self._set_scaled_preview_pixmap(
                self._annotated_preview,
                self._annotated_source_pixmap,
                fallback_w=260,
                fallback_h=220,
            )

    def show_running(self) -> None:
        self._status.setText("Running subroutine on crop...")
        self._run_btn.setEnabled(False)

    def show_results(
        self,
        *,
        crop_image: QImage,
        detections: list[dict],
        frame_w: int,
        frame_h: int,
    ) -> None:
        self._run_btn.setEnabled(True)
        self._raw_crop = crop_image
        self._crop_frame_w = int(frame_w or crop_image.width())
        self._crop_frame_h = int(frame_h or crop_image.height())
        self._apply_preview_heights(crop_image)
        self._show_raw_pixmap(QPixmap.fromImage(crop_image))

        if not detections:
            self._status.setText("No detections in this region.")
            self._table.setVisible(False)
            self._try_again_btn.setVisible(True)
            self._dismiss_btn.setVisible(True)
            self._reanalyze_btn.setVisible(True)
            nd_pix = self._render_no_detections(crop_image)
            self._show_annotated_pixmap(nd_pix)
            return

        self._status.setText(f"{len(detections)} detection(s) found.")
        self._last_render_crop = crop_image
        self._last_render_dets = list(detections)
        annotated = self._render_annotated(crop_image, detections, frame_w, frame_h)
        self._show_annotated_pixmap(annotated)
        self._fill_table(detections)
        self._table.setVisible(True)
        self._resize_table_to_content()
        self._try_again_btn.setVisible(False)
        self._dismiss_btn.setVisible(True)
        self._reanalyze_btn.setVisible(True)

    def show_error(self, message: str) -> None:
        self._run_btn.setEnabled(True)
        self._status.setText(str(message or "Subroutine failed."))
        self._annotated_source_pixmap = QPixmap()
        self._annotated_preview.setPixmap(QPixmap())
        self._annotated_preview.setText(str(message or "Error."))
        self._try_again_btn.setVisible(True)
        self._dismiss_btn.setVisible(True)
        self._reanalyze_btn.setVisible(True)

    def _resize_table_to_content(self) -> None:
        self._table.resizeRowsToContents()
        header_h = self._table.horizontalHeader().height()
        rows_h = sum(self._table.rowHeight(r) for r in range(self._table.rowCount()))
        self._table.setMaximumHeight(min(160, header_h + rows_h + 4))

    def _on_table_row_clicked(self, row: int, _col: int) -> None:
        if row == self._highlighted_row:
            self._highlighted_row = -1
            self.stop_blink()
            self.highlightRequested.emit(-1)
        else:
            self._highlighted_row = row
            if row < len(self._detections):
                self.blink_detection(self._detections[row])
            self.highlightRequested.emit(row)

    def _fill_table(self, detections: list[dict]) -> None:
        self._detections = list(detections)
        self._highlighted_row = -1
        # Show Type column only when results include pose or seg
        has_typed = any(det.get("type") in ("pose", "seg") for det in detections)
        self._table.setColumnCount(4 if has_typed else 3)
        self._table.setHorizontalHeaderLabels(
            ["Label", "Conf", "BBox", "Type"] if has_typed else ["Label", "Conf", "BBox"]
        )
        self._table.setRowCount(len(detections))
        for row, det in enumerate(detections):
            label = str(det.get("label") or "")
            conf = det.get("conf", det.get("confidence", det.get("score")))
            try:
                conf_s = f"{float(conf):.3f}" if conf is not None else ""
            except Exception:
                conf_s = str(conf)
            x1, y1, x2, y2 = det.get("x1"), det.get("y1"), det.get("x2"), det.get("y2")
            if all(v is not None for v in (x1, y1, x2, y2)):
                try:
                    bbox_s = f"[{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]"
                except Exception:
                    bbox_s = f"[{x1}, {y1}, {x2}, {y2}]"
            else:
                bbox_s = ""
            det_type = str(det.get("type") or "det")
            if det_type == "pose":
                kps = det.get("keypoints") or []
                visible = sum(1 for k in kps if len(k) > 2 and k[2] >= 0.2)
                type_s = f"pose ({visible}/{len(kps)} kp)"
            elif det_type == "seg":
                pts = len(det.get("mask_xy") or [])
                type_s = f"seg ({pts} pts)" if pts else "seg"
            else:
                type_s = "det"
            cells = [label, conf_s, bbox_s]
            if self._table.columnCount() == 4:
                cells.append(type_s)
            for col, val in enumerate(cells):
                self._table.setItem(row, col, QTableWidgetItem(val))

    def _render_no_detections(self, crop: QImage) -> QPixmap:
        pix = QPixmap.fromImage(crop)
        if pix.isNull():
            pix = QPixmap(260, 180)
            pix.fill(QColor("#101010"))
        out = QPixmap(pix.size())
        out.fill(QColor("#101010"))
        painter = QPainter(out)
        painter.drawPixmap(0, 0, pix)
        msg = "No detections in this region."
        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        fm = painter.fontMetrics()
        padding_x, padding_y = 10, 5
        tw = fm.horizontalAdvance(msg) + padding_x * 2
        th = fm.height() + padding_y * 2
        rx = (out.width() - tw) / 2
        ry = (out.height() - th) / 2
        painter.fillRect(QRectF(rx, ry, tw, th), QColor("#cc0000"))
        painter.setPen(QPen(QColor("#ffffff")))
        painter.drawText(QPointF(rx + padding_x, ry + padding_y + fm.ascent()), msg)
        painter.end()
        return out

    def _render_annotated(
        self,
        crop: QImage,
        detections: list[dict],
        frame_w: int,
        frame_h: int,
    ) -> QPixmap:
        pix = QPixmap.fromImage(crop)
        if pix.isNull():
            return pix
        out = QPixmap(pix.size())
        out.fill(QColor("#101010"))
        painter = QPainter(out)
        painter.drawPixmap(0, 0, pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        colour = QColor(_SUBROUTINE_PALETTE)
        fw = max(1, int(frame_w or crop.width()))
        fh = max(1, int(frame_h or crop.height()))
        sx = out.width() / fw
        sy = out.height() / fh
        show_boxes = self._show_boxes_cb.isChecked()
        show_masks = self._show_masks_cb.isChecked()
        show_pose = self._show_pose_cb.isChecked()
        show_labels = self._show_labels_cb.isChecked()
        for det in detections:
            det_type = str(det.get("type") or "det")
            label_lower = str(det.get("label") or "").lower()
            # Face detections are always shown — they have no mask or keypoints,
            # so suppressing their box would hide them entirely.
            is_face = label_lower == "face"
            try:
                x1 = float(det["x1"]) * sx
                y1 = float(det["y1"]) * sy
                x2 = float(det["x2"]) * sx
                y2 = float(det["y2"]) * sy
            except Exception:
                continue

            drew_anything = False
            if det_type == "seg":
                if show_masks:
                    self._render_seg_mask(painter, det, sx, sy, colour)
                    drew_anything = True
            else:
                if show_boxes or is_face:
                    painter.setPen(QPen(colour, 2))
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
                    drew_anything = True

            if det_type == "pose" and show_pose:
                self._render_pose_skeleton(painter, det, sx, sy)
                drew_anything = True

            # Skip the label tag entirely when nothing was drawn for this det
            if not drew_anything:
                continue

            label = str(det.get("label") or "")
            conf = det.get("conf", det.get("confidence", det.get("score")))
            tag = f"{label} {float(conf):.2f}" if label and conf is not None else label
            # In fusion mode, prefix with the model name so provenance is visible
            model_label = str(det.get("model_label") or "")
            if model_label:
                tag = f"[{model_label}] {tag}" if tag else f"[{model_label}]"
            if tag and show_labels:
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(tag) + 6
                th = fm.height() + 2
                tag_y = max(0.0, y1 - th)
                painter.fillRect(QRectF(x1, tag_y, tw, th), colour)
                painter.setPen(QPen(QColor("#000000")))
                painter.drawText(QPointF(x1 + 3, tag_y + fm.ascent() + 1), tag)
        painter.end()
        return out

    @staticmethod
    def _render_pose_skeleton(painter: QPainter, det: dict, sx: float, sy: float) -> None:
        """Draw COCO skeleton lines and joint dots over a pose detection."""
        kps: list[list[float]] = det.get("keypoints") or []
        if not kps:
            return
        # limb lines
        limb_colour = QColor(0, 255, 128, 200)
        painter.setPen(QPen(limb_colour, 2, Qt.PenStyle.SolidLine))
        for a, b in COCO_SKELETON:
            if a >= len(kps) or b >= len(kps):
                continue
            kp_a, kp_b = kps[a], kps[b]
            va = kp_a[2] if len(kp_a) > 2 else 1.0
            vb = kp_b[2] if len(kp_b) > 2 else 1.0
            if va < 0.2 or vb < 0.2:
                continue
            painter.drawLine(
                QPointF(kp_a[0] * sx, kp_a[1] * sy),
                QPointF(kp_b[0] * sx, kp_b[1] * sy),
            )
        # joint dots
        dot_colour = QColor(255, 220, 0, 230)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(dot_colour)
        for kp in kps:
            v = kp[2] if len(kp) > 2 else 1.0
            if v < 0.2:
                continue
            painter.drawEllipse(QPointF(kp[0] * sx, kp[1] * sy), 3.0, 3.0)

    @staticmethod
    def _render_seg_mask(painter: QPainter, det: dict, sx: float, sy: float, colour: QColor) -> None:
        """Draw segmentation polygon and semi-transparent fill."""
        from PyQt6.QtGui import QPolygonF  # local import to avoid top-level overhead

        contour: list[list[float]] = det.get("mask_xy") or []
        if len(contour) >= 3:
            poly = QPolygonF([QPointF(p[0] * sx, p[1] * sy) for p in contour])
            fill = QColor(colour.red(), colour.green(), colour.blue(), 50)
            painter.setBrush(fill)
            painter.setPen(QPen(colour, 1, Qt.PenStyle.SolidLine))
            painter.drawPolygon(poly)
        else:
            # fallback: just draw the bbox when mask data is unavailable
            try:
                x1 = float(det["x1"]) * sx
                y1 = float(det["y1"]) * sy
                x2 = float(det["x2"]) * sx
                y2 = float(det["y2"]) * sy
                painter.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))
            except Exception:
                pass

    def _emit_run(self) -> None:
        """Run whichever mode the catalog is in (single or fusion)."""
        if self._catalog_mode == ModelCatalogDialog.MODE_FUSION:
            if len(self._catalog_fusion_paths) < 2:
                self._status.setText("Open the catalog and pick at least two models for fusion.")
                return
            self.fusionRequested.emit(list(self._catalog_fusion_paths))
            return
        # Single-shot mode
        if not self._catalog_single_path:
            self._status.setText("Open the catalog and pick a model first.")
            return
        self.runRequested.emit(self._catalog_single_path, "")

    def _on_catalog_clicked(self) -> None:
        models = collect_video_test_models(http_get=self._http_get)
        if self._catalog_mode == ModelCatalogDialog.MODE_FUSION:
            pre = set(self._catalog_fusion_paths)
        elif self._catalog_single_path:
            pre = {self._catalog_single_path}
        else:
            pre = set()
        dialog = ModelCatalogDialog(
            models,
            preselected=pre,
            mode=self._catalog_mode,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected = dialog.selected_paths()
        self._catalog_mode = dialog.mode()
        if self._catalog_mode == ModelCatalogDialog.MODE_FUSION:
            self._catalog_fusion_paths = list(selected)
            self._catalog_single_path = ""
        else:
            self._catalog_single_path = selected[0] if selected else ""
            self._catalog_fusion_paths = []
        self._refresh_run_mode()

    def _refresh_run_mode(self) -> None:
        """Update button labels and the active-selection summary."""
        if self._catalog_mode == ModelCatalogDialog.MODE_FUSION:
            n = len(self._catalog_fusion_paths)
            if n >= 2:
                self._run_btn.setText(f"[RUN FUSION ({n})]")
                self._try_again_btn.setText(f"[TRY AGAIN — FUSION ({n})]")
                names = ", ".join(Path(p).stem for p in self._catalog_fusion_paths)
                self._catalog_summary.setText(f"Fusion: {n} models — {names}")
                self._run_btn.setEnabled(True)
            else:
                self._run_btn.setText("[RUN FUSION]")
                self._try_again_btn.setText("[TRY AGAIN]")
                self._catalog_summary.setText("Fusion mode — open the catalog and pick at least two models.")
                self._run_btn.setEnabled(False)
        else:
            self._run_btn.setText("[RUN SUBROUTINE]")
            self._try_again_btn.setText("[TRY AGAIN]")
            if self._catalog_single_path:
                self._catalog_summary.setText(f"Single: {Path(self._catalog_single_path).name}")
                self._run_btn.setEnabled(True)
            else:
                self._catalog_summary.setText("No model selected — open the catalog to pick one.")
                self._run_btn.setEnabled(False)

    def has_fusion_set(self) -> bool:
        return (
            self._catalog_mode == ModelCatalogDialog.MODE_FUSION
            and len(self._catalog_fusion_paths) >= 2
        )

    def fusion_paths(self) -> list[str]:
        return list(self._catalog_fusion_paths)

    def _on_reanalyze(self) -> None:
        """Ask the host panel to take a fresh frame snapshot and re-run inference."""
        self.reAnalyzeRequested.emit()

    def show_fusion_results(
        self,
        *,
        crop_image: QImage,
        per_model: list[tuple[str, list[dict]]],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Render overlapped fusion output and a compact per-model status line.

        per_model: list of (model_label, detections-in-crop-local-coords)
        """
        self._run_btn.setEnabled(True)
        self._raw_crop = crop_image
        self._crop_frame_w = int(frame_w or crop_image.width())
        self._crop_frame_h = int(frame_h or crop_image.height())
        self._apply_preview_heights(crop_image)
        self._show_raw_pixmap(QPixmap.fromImage(crop_image))

        # Compact status line: `pose: 2, seg: 1, face: 1, tiger: not detected`
        parts: list[str] = []
        merged: list[dict] = []
        for label, dets in per_model:
            short = Path(label).stem
            if dets:
                parts.append(f"{short}: {len(dets)}")
            else:
                parts.append(f"{short}: not detected")
            for det in dets:
                row = dict(det)
                row["model_label"] = short
                merged.append(row)
        self._status.setText(" | ".join(parts) if parts else "Fusion: no models ran.")

        if not merged:
            self._table.setVisible(False)
            self._try_again_btn.setVisible(True)
            self._dismiss_btn.setVisible(True)
            self._reanalyze_btn.setVisible(True)
            nd_pix = self._render_no_detections(crop_image)
            self._show_annotated_pixmap(nd_pix)
            return

        self._last_render_crop = crop_image
        self._last_render_dets = merged
        annotated = self._render_annotated(crop_image, merged, frame_w, frame_h)
        self._show_annotated_pixmap(annotated)
        self._fill_table(merged)
        self._table.setVisible(True)
        self._resize_table_to_content()
        self._try_again_btn.setVisible(False)
        self._dismiss_btn.setVisible(True)
        self._reanalyze_btn.setVisible(True)

    # ------------------------------------------------------------------
    # Streaming fusion (incremental rendering as each model finishes)
    # ------------------------------------------------------------------

    def is_stream_enabled(self) -> bool:
        return self._stream_cb.isChecked()

    def begin_fusion_stream(
        self,
        *,
        crop_image: QImage,
        model_labels: list[str],
        frame_w: int,
        frame_h: int,
    ) -> None:
        """Initialize the panel for streaming-fusion output."""
        self._run_btn.setEnabled(False)
        self._raw_crop = crop_image
        self._crop_frame_w = int(frame_w or crop_image.width())
        self._crop_frame_h = int(frame_h or crop_image.height())
        self._stream_crop_w = self._crop_frame_w
        self._stream_crop_h = self._crop_frame_h
        self._apply_preview_heights(crop_image)
        self._show_raw_pixmap(QPixmap.fromImage(crop_image))

        self._stream_active = True
        self._stream_per_model = []
        self._stream_pending = [Path(lbl).stem for lbl in model_labels]

        self._last_render_crop = crop_image
        self._last_render_dets = []
        # Paint the bare crop so the user sees the source image right away
        empty_pix = QPixmap.fromImage(crop_image)
        if not empty_pix.isNull():
            self._show_annotated_pixmap(empty_pix)
        self._fill_table([])
        self._table.setVisible(False)
        self._try_again_btn.setVisible(False)
        self._dismiss_btn.setVisible(True)
        self._reanalyze_btn.setVisible(True)
        self._update_stream_status()

    def append_fusion_model(self, model_label: str, detections: list[dict]) -> None:
        """Add one model's results to the streaming render."""
        if not self._stream_active:
            return
        short = Path(model_label).stem
        self._stream_per_model.append((short, list(detections or [])))
        if short in self._stream_pending:
            self._stream_pending.remove(short)

        # Append to the cumulative merged-detection list
        for det in detections or []:
            row = dict(det)
            row["model_label"] = short
            self._last_render_dets.append(row)

        # Re-render annotated preview from the growing detection set
        if self._last_render_crop is not None:
            annotated = self._render_annotated(
                self._last_render_crop,
                self._last_render_dets,
                self._stream_crop_w,
                self._stream_crop_h,
            )
            self._show_annotated_pixmap(annotated)

        # Update table incrementally
        if self._last_render_dets:
            self._fill_table(self._last_render_dets)
            self._table.setVisible(True)
            self._resize_table_to_content()

        self._update_stream_status()

    def finish_fusion_stream(self) -> None:
        """Finalize the streaming session (no more models incoming)."""
        if not self._stream_active:
            return
        self._stream_active = False
        self._stream_pending = []
        self._run_btn.setEnabled(True)
        # If nothing produced detections, render the no-detections overlay
        if self._last_render_crop is not None and not self._last_render_dets:
            nd_pix = self._render_no_detections(self._last_render_crop)
            self._show_annotated_pixmap(nd_pix)
            self._try_again_btn.setVisible(True)
        self._update_stream_status(final=True)

    def _update_stream_status(self, *, final: bool = False) -> None:
        parts: list[str] = []
        for short, dets in self._stream_per_model:
            parts.append(f"{short}: {len(dets)}" if dets else f"{short}: not detected")
        if not final:
            for short in self._stream_pending:
                parts.append(f"{short}: running…")
        if not parts:
            self._status.setText("Streaming fusion…")
            return
        self._status.setText(" | ".join(parts))

    def eventFilter(self, obj, event) -> bool:  # type: ignore[override]
        if obj is self._annotated_preview and event.type() == QEvent.Type.Resize:
            self._blink_overlay.resize(self._annotated_preview.size())
        return super().eventFilter(obj, event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_preview_scaling()

    def blink_detection(self, det: dict) -> None:
        """Blink the detection box for `det` over the annotated result preview."""
        self._blink_overlay.stop()
        pix = self._annotated_preview.pixmap()
        if pix is None or pix.isNull():
            return
        lw = self._annotated_preview.width()
        lh = self._annotated_preview.height()
        pw, ph = pix.width(), pix.height()
        if pw <= 0 or ph <= 0 or lw <= 0 or lh <= 0:
            return
        # letterbox display rect inside the label
        scale = min(lw / pw, lh / ph)
        dw = pw * scale
        dh = ph * scale
        disp_ox = (lw - dw) / 2
        disp_oy = (lh - dh) / 2
        # crop-local pixel dimensions the annotated pixmap was rendered at
        fw = self._crop_frame_w or pw
        fh = self._crop_frame_h or ph
        sx = dw / fw
        sy = dh / fh
        try:
            x1 = float(det["x1"]) * sx + disp_ox
            y1 = float(det["y1"]) * sy + disp_oy
            x2 = float(det["x2"]) * sx + disp_ox
            y2 = float(det["y2"]) * sy + disp_oy
        except Exception:
            return
        self._blink_overlay.resize(self._annotated_preview.size())
        self._blink_overlay.start(QRectF(x1, y1, x2 - x1, y2 - y1))

    def stop_blink(self) -> None:
        self._blink_overlay.stop()

    def _on_layer_toggled(self, _checked: bool) -> None:
        """Re-render the annotated preview from cached detections — no inference."""
        if self._last_render_crop is None or not self._last_render_dets:
            return
        annotated = self._render_annotated(
            self._last_render_crop,
            self._last_render_dets,
            self._crop_frame_w or self._last_render_crop.width(),
            self._crop_frame_h or self._last_render_crop.height(),
        )
        self._show_annotated_pixmap(annotated)

    def hide_panel(self) -> None:
        self._blink_overlay.stop()
        self._status.setText("Draw a region on the video, then run a model on that crop.")
        self._annotated_preview.setText("No result yet.")
        self._annotated_preview.setPixmap(QPixmap())
        self._raw_preview.setPixmap(QPixmap())
        self._annotated_source_pixmap = QPixmap()
        self._raw_source_pixmap = QPixmap()
        self._raw_crop = None
        self._detections = []
        self._last_render_crop = None
        self._last_render_dets = []
        self._stream_active = False
        self._stream_per_model = []
        self._stream_pending = []
        self._highlighted_row = -1
        self.highlightRequested.emit(-1)
        self._table.setVisible(False)
        self._table.setRowCount(0)
        self._try_again_btn.setVisible(False)
        self._dismiss_btn.setVisible(False)
        self._reanalyze_btn.setVisible(False)
        self._apply_preview_heights(None)

    def _on_dismiss(self) -> None:
        self.hide_panel()
        self.dismissed.emit()


class _RoiDismissItem(QGraphicsItem):
    """Small [X] button pinned to the top-left corner of the ROI rect."""

    _SIZE = 18   # px square for the button
    _PAD  = 3    # inner margin for the X strokes

    def __init__(self, on_click: Callable[[], None]) -> None:
        super().__init__()
        self._on_click = on_click
        self._btn = QRectF()   # the actual button square in scene coords
        self.setZValue(13.0)
        self.setVisible(False)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def set_rect(self, roi: QRectF) -> None:
        self.prepareGeometryChange()
        # Pin to the top-left corner of the ROI, slightly overlapping the border
        s = self._SIZE
        self._btn = QRectF(roi.x() - s / 2, roi.y() - s / 2, s, s)

    def boundingRect(self) -> QRectF:
        return self._btn

    def paint(self, painter, _option, _widget=None) -> None:  # type: ignore[override]
        if self._btn.isEmpty():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Filled background square
        painter.fillRect(self._btn, QColor("#cc0000"))
        # White X strokes
        pen = QPen(QColor("#ffffff"), 2, Qt.PenStyle.SolidLine)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        p = self._PAD
        tl = QPointF(self._btn.x() + p, self._btn.y() + p)
        br = QPointF(self._btn.right() - p, self._btn.bottom() - p)
        tr = QPointF(self._btn.right() - p, self._btn.y() + p)
        bl = QPointF(self._btn.x() + p, self._btn.bottom() - p)
        painter.drawLine(tl, br)
        painter.drawLine(tr, bl)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._on_click()
            event.accept()
        else:
            super().mousePressEvent(event)


class SubroutineBlinkHighlight(QGraphicsItem):
    """Gold/blue pulsing highlight drawn over a single detection box.

    Fades in and out smoothly using a QTimer-driven alpha cycle.
    Auto-stops after _BLINK_DURATION_MS (2 s) unless toggled off sooner.
    """

    _TICK_MS = 33           # ~30 fps
    _CYCLE_MS = 500         # one full gold→blue→gold fade cycle
    _BLINK_DURATION_MS = 2000
    _GOLD = QColor(255, 200, 0)
    _BLUE = QColor(0, 180, 255)

    def __init__(self, video_item: QGraphicsVideoItem) -> None:
        super().__init__()
        self._video_item = video_item
        self._rect: Optional[QRectF] = None
        self._elapsed = 0
        self.setZValue(14.0)
        self.setVisible(False)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

        self._timer = QTimer()
        self._timer.setInterval(self._TICK_MS)
        self._timer.timeout.connect(self._tick)

    def start_detection(self, det: dict) -> None:
        """Scale det coords to video-item display space and begin blinking."""
        size = self._video_item.size()
        pos = self._video_item.pos()
        widget_w = size.width()
        widget_h = size.height()
        if widget_w <= 0 or widget_h <= 0:
            return
        fw = int(det.get("frame_w") or 0)
        fh = int(det.get("frame_h") or 0)
        if fw <= 0 or fh <= 0:
            # crop-local coords — no frame offset, det coords are already in crop pixels
            fw = int(det.get("x2", widget_w) - det.get("x1", 0)) or int(widget_w)
            fh = int(det.get("y2", widget_h) - det.get("y1", 0)) or int(widget_h)
        sx = widget_w / fw
        sy = widget_h / fh
        x1 = float(det.get("x1", 0)) * sx + pos.x()
        y1 = float(det.get("y1", 0)) * sy + pos.y()
        x2 = float(det.get("x2", 0)) * sx + pos.x()
        y2 = float(det.get("y2", 0)) * sy + pos.y()
        self._rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        self._elapsed = 0
        self.setVisible(True)
        self.update()
        self._timer.start()

    def stop(self) -> None:
        self._timer.stop()
        self.setVisible(False)
        self._rect = None

    def boundingRect(self) -> QRectF:
        if self._rect is None:
            return QRectF()
        return self._rect.adjusted(-4, -4, 4, 4)

    def paint(self, painter, _option, _widget=None) -> None:  # type: ignore[override]
        if self._rect is None:
            return
        t = (self._elapsed % self._CYCLE_MS) / self._CYCLE_MS  # 0.0 → 1.0
        # ping-pong: 0→1→0 over the cycle
        alpha_t = 1.0 - abs(t * 2 - 1.0)
        # interpolate gold → blue
        r = int(self._GOLD.red()   + (self._BLUE.red()   - self._GOLD.red())   * alpha_t)
        g = int(self._GOLD.green() + (self._BLUE.green() - self._GOLD.green()) * alpha_t)
        b = int(self._GOLD.blue()  + (self._BLUE.blue()  - self._GOLD.blue())  * alpha_t)
        colour = QColor(r, g, b, 220)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = QPen(colour, 3, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        fill = QColor(r, g, b, int(40 * alpha_t))
        painter.setBrush(fill)
        painter.drawRect(self._rect)

    def _tick(self) -> None:
        self._elapsed += self._TICK_MS
        if self._elapsed >= self._BLINK_DURATION_MS:
            self.stop()
            return
        self.update()


class SubroutineBoxOverlay(QGraphicsItem):
    """Cyan subroutine detection boxes (video stage)."""

    def __init__(self, video_item: QGraphicsVideoItem) -> None:
        super().__init__()
        self._video_item = video_item
        self._boxes: list[dict] = []
        self.setZValue(11.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def sync_to_video(self) -> None:
        self.prepareGeometryChange()
        self.setPos(self._video_item.pos())
        self.update()

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        size = self._video_item.size()
        return QRectF(0.0, 0.0, size.width(), size.height())

    def set_boxes(self, boxes: list[dict]) -> None:
        self._boxes = list(boxes or [])
        self.setVisible(bool(self._boxes))
        self.update()

    def clear(self) -> None:
        self._boxes = []
        self.setVisible(False)
        self.update()

    def paint(self, painter, _option, _widget=None) -> None:  # type: ignore[override]
        if not self._boxes:
            return
        rect = self.boundingRect()
        widget_w = rect.width()
        widget_h = rect.height()
        if widget_w <= 0 or widget_h <= 0:
            return
        colour = QColor(_SUBROUTINE_PALETTE)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for det in self._boxes:
            fw = int(det.get("frame_w") or 0)
            fh = int(det.get("frame_h") or 0)
            if fw <= 0 or fh <= 0:
                continue
            scale_x = widget_w / fw
            scale_y = widget_h / fh
            x1 = float(det["x1"]) * scale_x
            y1 = float(det["y1"]) * scale_y
            x2 = float(det["x2"]) * scale_x
            y2 = float(det["y2"]) * scale_y
            painter.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))


class SubroutineRoiController(QObject):
    """Drag-to-select ROI on a QGraphicsView, maps to video item coords.

    When the drag completes, roiCommitted is emitted and the sidebar panel
    opens automatically — no floating overlay button on the video.
    """

    roiCommitted = pyqtSignal(QRect)  # video-local pixel rect
    cleared = pyqtSignal()

    def __init__(
        self,
        *,
        view: QGraphicsView,
        video_item: QGraphicsVideoItem,
        scene: object,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._view = view
        self._video_item = video_item
        self._active = False
        self._dragging = False
        self._origin = QPointF()
        self._roi_rect: Optional[QRect] = None

        self._rect_item = QGraphicsRectItem()
        pen = QPen(QColor(_SUBROUTINE_PALETTE), 2, Qt.PenStyle.DashLine)
        self._rect_item.setPen(pen)
        self._rect_item.setBrush(QColor(34, 211, 238, 40))
        self._rect_item.setZValue(12.0)
        self._rect_item.setVisible(False)
        scene.addItem(self._rect_item)  # type: ignore[attr-defined]

        self._x_item = _RoiDismissItem(on_click=lambda: self.clear(emit=True))
        scene.addItem(self._x_item)  # type: ignore[attr-defined]

    def set_select_mode(self, enabled: bool) -> None:
        self._active = bool(enabled)
        if not self._active:
            # Only clear (and emit) when deactivating with no committed rect —
            # i.e. the user toggled off without finishing a drag. If a rect is
            # already committed we leave the overlay visible and don't wipe it.
            if self._roi_rect is None:
                self.clear(emit=True)
            else:
                self._dragging = False

    def clear(self, *, emit: bool = True) -> None:
        self._dragging = False
        self._roi_rect = None
        self._rect_item.setVisible(False)
        self._x_item.setVisible(False)
        if emit:
            self.cleared.emit()

    def current_roi(self) -> Optional[QRect]:
        return self._roi_rect

    def sync_layout(self) -> None:
        if self._roi_rect is None:
            return
        self._update_graphics(self._roi_rect)

    def handle_event(self, event) -> bool:
        if not self._active:
            return False
        et = event.type()
        if et == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
            pt = self._map_to_video(event.position())
            if pt is None:
                return False
            self._dragging = True
            self._origin = pt
            self._rect_item.setVisible(True)
            self._update_graphics(QRect(int(pt.x()), int(pt.y()), 0, 0))
            return True
        if et == QEvent.Type.MouseMove and self._dragging:
            pt = self._map_to_video(event.position())
            if pt is None:
                return True
            self._update_graphics(QRect(
                int(min(self._origin.x(), pt.x())),
                int(min(self._origin.y(), pt.y())),
                int(abs(pt.x() - self._origin.x())),
                int(abs(pt.y() - self._origin.y())),
            ))
            return True
        if et == QEvent.Type.MouseButtonRelease and self._dragging:
            self._dragging = False
            pt = self._map_to_video(event.position())
            if pt is None:
                self._rect_item.setVisible(False)
                return True
            rect = QRect(
                int(min(self._origin.x(), pt.x())),
                int(min(self._origin.y(), pt.y())),
                int(abs(pt.x() - self._origin.x())),
                int(abs(pt.y() - self._origin.y())),
            )
            if rect.width() < _MIN_ROI_PX or rect.height() < _MIN_ROI_PX:
                self.clear()
                return True
            self._roi_rect = rect
            self._update_graphics(rect)
            self.roiCommitted.emit(rect)
            return True
        return False

    def _map_to_video(self, view_pos) -> Optional[QPointF]:
        scene_pt = self._view.mapToScene(int(view_pos.x()), int(view_pos.y()))
        local = self._video_item.mapFromScene(scene_pt)
        size = self._video_item.size()
        if local.x() < 0 or local.y() < 0 or local.x() > size.width() or local.y() > size.height():
            return None
        return local

    def _update_graphics(self, rect: QRect) -> None:
        pos = self._video_item.pos()
        rx = float(rect.x()) + pos.x()
        ry = float(rect.y()) + pos.y()
        rw = float(rect.width())
        rh = float(rect.height())
        self._rect_item.setRect(rx, ry, rw, rh)
        # Pass the full ROI rect; the dismiss item pins itself to the top-left corner
        self._x_item.set_rect(QRectF(rx, ry, rw, rh))
        show_x = not self._dragging and rect.width() >= _MIN_ROI_PX and rect.height() >= _MIN_ROI_PX
        self._x_item.setVisible(show_x)


class SubroutineImageOverlay(QWidget):
    """Transparent overlay for ROI drag on a QLabel image preview."""

    roiCommitted = pyqtSignal(QRect)  # full-image pixel rect
    subroutineClicked = pyqtSignal()
    cleared = pyqtSignal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        self.setMouseTracking(True)
        self._active = False
        self._dragging = False
        self._origin = QPointF()
        self._roi: Optional[QRect] = None
        self._image_size = (0, 0)
        self._result_boxes: list[dict] = []

        self._sub_btn = QPushButton("[SUBROUTINE]", self)
        self._sub_btn.clicked.connect(self.subroutineClicked.emit)
        self._sub_btn.setVisible(False)
        self._sub_btn.raise_()

    def set_image_size(self, w: int, h: int) -> None:
        self._image_size = (max(0, int(w)), max(0, int(h)))

    def set_select_mode(self, enabled: bool) -> None:
        self._active = bool(enabled)
        self.setVisible(enabled or self._roi is not None)
        if not self._active and self._roi is None:
            self.clear(emit=True)

    def clear(self, *, emit: bool = True) -> None:
        self._dragging = False
        self._roi = None
        self._result_boxes = []
        self._sub_btn.setVisible(False)
        self.update()
        if emit:
            self.cleared.emit()

    def set_detection_boxes(self, boxes: list[dict]) -> None:
        self._result_boxes = list(boxes or [])
        self.update()

    def current_roi(self) -> Optional[QRect]:
        return self._roi

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        if self._roi is None and not self._dragging:
            return
        painter = QPainter(self)
        colour = QColor(_SUBROUTINE_PALETTE)
        painter.setPen(QPen(colour, 2, Qt.PenStyle.DashLine))
        painter.setBrush(QColor(34, 211, 238, 40))
        disp = self._display_rect()
        if self._dragging and hasattr(self, "_drag_rect"):
            r = self._drag_rect
        elif self._roi is not None:
            r = self._image_rect_to_widget(self._roi)
        else:
            return
        if r is not None and r.width() > 0 and r.height() > 0:
            painter.drawRect(r)
        for det in self._result_boxes:
            try:
                box = QRect(
                    int(det["x1"]),
                    int(det["y1"]),
                    int(det["x2"] - det["x1"]),
                    int(det["y2"] - det["y1"]),
                )
            except Exception:
                continue
            wr = self._image_rect_to_widget(box)
            if wr is not None and wr.width() > 0:
                painter.setPen(QPen(colour, 2, Qt.PenStyle.SolidLine))
                painter.drawRect(wr)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if not self._active or event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        ip = self._widget_to_image(event.position())
        if ip is None or ip.x() < 0:
            return
        self._dragging = True
        self._origin = ip
        self._drag_rect = self._image_rect_to_widget(
            QRect(int(ip.x()), int(ip.y()), 0, 0)
        )
        self._sub_btn.setVisible(False)
        self.update()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        ip = self._widget_to_image(event.position())
        if ip is None:
            return
        rect = QRect(
            int(min(self._origin.x(), ip.x())),
            int(min(self._origin.y(), ip.y())),
            int(abs(ip.x() - self._origin.x())),
            int(abs(ip.y() - self._origin.y())),
        )
        self._drag_rect = self._image_rect_to_widget(rect)
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if not self._dragging:
            super().mouseReleaseEvent(event)
            return
        self._dragging = False
        ip = self._widget_to_image(event.position())
        if ip is None:
            self.clear()
            return
        rect = QRect(
            int(min(self._origin.x(), ip.x())),
            int(min(self._origin.y(), ip.y())),
            int(abs(ip.x() - self._origin.x())),
            int(abs(ip.y() - self._origin.y())),
        )
        if rect.width() < _MIN_ROI_PX or rect.height() < _MIN_ROI_PX:
            self.clear()
            return
        self._roi = rect
        wr = self._image_rect_to_widget(rect)
        if wr is not None:
            self._sub_btn.move(wr.right() + 4, wr.top())
            self._sub_btn.setVisible(True)
        self.roiCommitted.emit(rect)
        self.update()

    def _display_rect(self) -> QRect:
        iw, ih = self._image_size
        return _letterbox_rect(self.width(), self.height(), iw, ih)

    def _widget_to_image(self, pos) -> Optional[QPointF]:
        iw, ih = self._image_size
        if iw <= 0 or ih <= 0:
            return None
        return label_point_to_image(pos, self.width(), self.height(), iw, ih)

    def _image_rect_to_widget(self, rect: QRect) -> Optional[QRect]:
        disp = self._display_rect()
        iw, ih = self._image_size
        if disp.width() <= 0 or ih <= 0:
            return None
        x = disp.x() + int(rect.x() * disp.width() / iw)
        y = disp.y() + int(rect.y() * disp.height() / ih)
        w = max(1, int(rect.width() * disp.width() / iw))
        h = max(1, int(rect.height() * disp.height() / ih))
        return QRect(x, y, w, h)


class SubroutineSession:
    """Runs one-shot inference in a background thread."""

    def __init__(self, parent: QObject) -> None:
        self._thread: Optional[QThread] = None
        self._worker: Optional[SubroutineInferenceWorker] = None
        self._parent = parent

    def start(
        self,
        *,
        model_path: str,
        device: str,
        frame_bgr: object,
        on_finished: Callable[[list], None],
        on_failed: Callable[[str], None],
    ) -> None:
        self.stop()
        thread = QThread()
        worker = SubroutineInferenceWorker()
        worker.moveToThread(thread)
        thread.started.connect(
            lambda: worker.run(model_path, device, frame_bgr)
        )
        worker.finished.connect(on_finished)
        worker.failed.connect(on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        self._thread = thread
        self._worker = worker
        thread.start()

    def stop(self) -> None:
        thread = self._thread
        self._thread = None
        self._worker = None
        if thread is None:
            return
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(2000)
        except RuntimeError:
            # Qt already deleted the underlying C++ object — nothing to do.
            pass

    def start_multi(
        self,
        *,
        model_paths: list[str],
        device: str,
        frame_bgr: object,
        on_each: Callable[[str, list], None],
        on_each_failed: Callable[[str, str], None],
        on_all_done: Callable[[], None],
    ) -> None:
        """Run several models sequentially on the same crop.

        on_each(model_path, detections) fires per success.
        on_each_failed(model_path, error_msg) fires per failure.
        on_all_done() fires once when the queue is exhausted.
        """
        self.stop()
        queue = list(model_paths)
        if not queue:
            on_all_done()
            return

        def _next() -> None:
            if not queue:
                on_all_done()
                return
            path = queue.pop(0)

            def _done(detections: list, _p: str = path) -> None:
                try:
                    on_each(_p, detections)
                finally:
                    _next()

            def _fail(msg: str, _p: str = path) -> None:
                try:
                    on_each_failed(_p, msg)
                finally:
                    _next()

            self.start(
                model_path=path,
                device=device,
                frame_bgr=frame_bgr,
                on_finished=_done,
                on_failed=_fail,
            )

        _next()
