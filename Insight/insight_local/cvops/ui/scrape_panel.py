"""PyQt6 panel for web-scrape dataset building.

Ports the Streamlit ``scrap_panel.render()`` workflow to a native Qt widget:
  1. Create a job (topic + search query + target count).
  2. Background thread scrapes Google Images via Selenium, dedupes, and stages
     the results.
  3. Operator draws boxes on each staged image; after each box, they pick the class
     for that box (dialog). The class name is shown on the canvas and stored in YOLO
     label files on save. Staged-image thumbnails and the annotation canvas share one
     Label tab inside a data card.
  4. Emit a YOLO dataset + scenario profile ready for training.

The Selenium / Chrome import is deferred until the operator actually clicks
"Start Scrape" so the rest of the CV Ops window loads without Chrome.

Threading: scrape runs in a daemon thread.  A 1-second QTimer polls
``scrap.json`` and updates the UI on the main thread.
"""

from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QRect, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFontDatabase,
    QFont,
    QFontMetrics,
    QIcon,
    QPainter,
    QPen,
    QPixmap,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QMenu,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .collapsible_section import CollapsibleSection
from .dropdown_pane_stack import DropdownPaneStack
from .scrape_promote_dialog import ScrapePromoteDatasetDialog
from .cvops_theme import (
    WB_FONT_MONO,
    cvops_color,
    cvops_qcolor,
    cvops_rgba,
    repolish,
    set_cvops_stylesheet,
)

log = logging.getLogger(__name__)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_QA_CACHE_NAME = "scrap_qa.json"
_QA_ALL_FILTER = "__all__"
_QA_NONE_FILTER = "__none__"
_QA_DEFAULT_MODEL = "assets/models/yolov10n.pt"
_QA_DEFAULT_CONF = 0.25
_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".bmp"})
_RAW_THUMB_SIZE = QSize(92, 92)
_GALLERY_THUMB_SIZE = QSize(164, 132)
_LABEL_GALLERY_GRID_SIZE = QSize(120, 124)
_FULL_GALLERY_GRID_SIZE = QSize(190, 188)
_DETECTION_BOX_RED = QColor(255, 0, 0)
_DETECTION_BOX_FILL_RED = QColor(255, 0, 0, 48)
_DETECTION_BOX_HIGHLIGHT_FILL_RED = QColor(255, 0, 0, 72)


def _scrape_ensure_class_index(classes: list[str], class_name: str) -> tuple[list[str], int]:
    """Return (classes', index) for *class_name*, appending to the list if needed."""
    name = str(class_name or "").strip() or "object"
    out = list(classes)
    try:
        return out, out.index(name)
    except ValueError:
        out.append(name)
        return out, len(out) - 1


def _scrape_full_frame_box(class_idx: int) -> list[float]:
    return [float(class_idx), 0.5, 0.5, 1.0, 1.0]


def _scrape_detection_to_yolo_box(det: dict[str, Any], class_idx: int) -> list[float] | None:
    frame_w = float(det.get("frame_w") or 0.0)
    frame_h = float(det.get("frame_h") or 0.0)
    if frame_w <= 0.0 or frame_h <= 0.0:
        return None
    x1 = max(0.0, min(frame_w, float(det.get("x1") or 0.0)))
    y1 = max(0.0, min(frame_h, float(det.get("y1") or 0.0)))
    x2 = max(0.0, min(frame_w, float(det.get("x2") or 0.0)))
    y2 = max(0.0, min(frame_h, float(det.get("y2") or 0.0)))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    bw = max(0.0, right - left)
    bh = max(0.0, bottom - top)
    if bw <= 0.0 or bh <= 0.0:
        return None
    return [
        float(class_idx),
        max(0.0, min(1.0, ((left + right) / 2.0) / frame_w)),
        max(0.0, min(1.0, ((top + bottom) / 2.0) / frame_h)),
        max(0.0, min(1.0, bw / frame_w)),
        max(0.0, min(1.0, bh / frame_h)),
    ]


def _scrape_read_classes_txt(dataset_root: Path) -> list[str]:
    path = dataset_root / "classes.txt"
    if not path.is_file():
        return []
    try:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    except Exception:
        return []


def _scrape_existing_yolo_items(dataset_root: Path) -> tuple[list[str], list[Any]]:
    from mlops.pipeline import registry as reg  # noqa: PLC0415
    from mlops.scrap.emit import LabeledItem  # noqa: PLC0415

    classes = _scrape_read_classes_txt(dataset_root)
    out: list[Any] = []
    images_root = dataset_root / "images"
    if not images_root.is_dir():
        return classes, out
    for img in sorted(p for p in images_root.glob("**/*") if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES):
        label_path = reg.resolve_dataset_label_path(img)
        if label_path is None or not label_path.is_file():
            continue
        boxes: list[tuple[int, float, float, float, float]] = []
        try:
            lines = label_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                boxes.append((int(float(parts[0])), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
            except Exception:
                continue
        if boxes:
            out.append(LabeledItem(image_path=img, boxes=tuple(boxes)))
    return classes, out


def _resolve_repo_path(raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip()).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (_REPO_ROOT / candidate).resolve()


def _resolve_qa_model_reference(model_ref: str) -> Path:
    value = str(model_ref or "").strip()
    if not value:
        value = _QA_DEFAULT_MODEL
    try:
        from mlops.pipeline.registry import resolve_model_reference  # noqa: PLC0415

        return resolve_model_reference(value).resolve()
    except Exception:
        return _resolve_repo_path(value)


def _list_qa_model_refs() -> list[dict[str, str]]:
    try:
        from mlops.pipeline.registry import list_available_models  # noqa: PLC0415
        from ..detection_backends import is_supported_video_test_model  # noqa: PLC0415
    except Exception:
        return []

    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for entry in list_available_models():
        ref = str(entry.get("value") or entry.get("path") or "").strip()
        if not ref or ref in seen:
            continue
        try:
            resolved = _resolve_qa_model_reference(ref)
        except Exception:
            continue
        if not is_supported_video_test_model(resolved):
            continue
        seen.add(ref)
        out.append(
            {
                "ref": ref,
                "path": str(resolved),
                "origin": str(entry.get("origin") or ""),
            }
        )
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


def _qa_signature(images: list[Path]) -> list[str]:
    return [p.name for p in images]


def _qa_entry_for_detections(detections: list[dict[str, Any]], *, error: str = "") -> dict[str, Any]:
    cleaned: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    for det in detections:
        label = str(det.get("label") or "").strip() or "unknown"
        conf = float(det.get("conf") or 0.0)
        cleaned.append(
            {
                "label": label,
                "conf": conf,
                "x1": float(det.get("x1") or 0.0),
                "y1": float(det.get("y1") or 0.0),
                "x2": float(det.get("x2") or 0.0),
                "y2": float(det.get("y2") or 0.0),
                "frame_w": int(det.get("frame_w") or 0),
                "frame_h": int(det.get("frame_h") or 0),
            }
        )
        label_counts[label] += 1
    cleaned.sort(key=lambda det: float(det.get("conf") or 0.0), reverse=True)
    return {
        "detection_count": len(cleaned),
        "label_counts": dict(sorted(label_counts.items())),
        "detections": cleaned,
        "error": str(error or ""),
        "scanned_at": time.time(),
    }


def _qa_entry_refresh_metadata(entry: dict[str, Any], *, resort: bool = True) -> None:
    """Recompute detection_count and label_counts after manual detection edits."""
    dets = list(entry.get("detections") or [])
    label_counts: Counter[str] = Counter()
    for det in dets:
        label = str(det.get("label") or "").strip() or "unknown"
        det["label"] = label
        label_counts[label] += 1
    if resort:
        dets.sort(key=lambda det: float(det.get("conf") or 0.0), reverse=True)
    entry["detections"] = dets
    entry["detection_count"] = len(dets)
    entry["label_counts"] = dict(sorted(label_counts.items()))


def _qa_parse_from_labels(raw: str) -> list[str]:
    """Split comma-separated labels (e.g. ``human, zebra, cat``); strip; drop empties."""
    return [s for s in (chunk.strip() for chunk in str(raw or "").split(",")) if s]


def compute_transform_preview(
    paths: list[Path],
    filter_key: str,
    promoted_class: str,
    promoted_class_idx: int,
    source_job_labels: dict[str, list[list[float]]],
    qa_items: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Mirror _execute_promote_to_dataset without writing files.

    Returns {image_name: entry} where each entry has:
        image_path         — Path to the staged source image
        original_dets      — raw QA detection dicts (label, conf, x1/y1/x2/y2 in pixels)
        original_hand_boxes — hand-drawn YOLO boxes with their original class_ids
        original_source    — "hand_drawn" | "qa_detection" | "full_frame_fallback"
        transformed_boxes  — final [[promoted_class_idx, cx, cy, nw, nh], ...] list
        yolo_txt           — exact .txt content that would be written (YOLO format)
        csv_row            — "filename,class_name" row (CSV/classification format)
    """
    result: dict[str, dict[str, Any]] = {}
    for src in paths:
        if not src.is_file():
            continue
        name = src.name
        new_boxes: list[list[float]] = []
        original_dets: list[dict] = []
        original_hand_boxes: list[list[float]] = []
        original_source = "full_frame_fallback"

        raw = source_job_labels.get(name) if source_job_labels else None
        if raw:
            original_hand_boxes = [list(b) for b in raw if len(b) >= 5]
            for b in original_hand_boxes:
                cx, cy, w, h = float(b[1]), float(b[2]), float(b[3]), float(b[4])
                new_boxes.append([float(promoted_class_idx), cx, cy, w, h])
            original_source = "hand_drawn"
        else:
            entry = qa_items.get(name)
            dets = list(entry.get("detections") or []) if isinstance(entry, dict) else []
            if filter_key not in (_QA_ALL_FILTER, _QA_NONE_FILTER):
                dets = [d for d in dets if str(d.get("label") or "").strip() == str(filter_key)]
            elif filter_key == _QA_NONE_FILTER:
                dets = []
            original_dets = list(dets)
            for det in dets:
                box = _scrape_detection_to_yolo_box(dict(det), promoted_class_idx)
                if box is not None:
                    new_boxes.append(box)
            if new_boxes:
                original_source = "qa_detection"

        if not new_boxes:
            new_boxes = [_scrape_full_frame_box(promoted_class_idx)]
            original_source = "full_frame_fallback"

        yolo_lines = [
            f"{int(b[0])} {float(b[1]):.6f} {float(b[2]):.6f} "
            f"{float(b[3]):.6f} {float(b[4]):.6f}"
            for b in new_boxes
        ]
        result[name] = {
            "image_path": src,
            "original_dets": original_dets,
            "original_hand_boxes": original_hand_boxes,
            "original_source": original_source,
            "transformed_boxes": new_boxes,
            "yolo_txt": "\n".join(yolo_lines) + "\n",
            "csv_row": f"{name},{promoted_class}",
        }
    return result


class _ScrapeProgressCircle(QWidget):
    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        diameter: int = 158,
        tooltip: str = "Raw download progress for the current scrape job.",
        detail: str = "raw 0 / 0",
    ) -> None:
        super().__init__(parent)
        self._diameter = max(72, int(diameter or 158))
        self._value = 0
        self._headline = "0%"
        self._detail = str(detail or "")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setMinimumSize(self._diameter, self._diameter)
        self.setMaximumSize(self._diameter, self._diameter)
        self.setToolTip(str(tooltip or ""))

    def sizeHint(self) -> QSize:
        return QSize(self._diameter, self._diameter)

    def set_progress(self, value: int, *, headline: str = "", detail: str = "") -> None:
        safe_value = max(0, min(100, int(value or 0)))
        next_headline = str(headline or f"{safe_value}%")
        next_detail = str(detail or "")
        if (
            safe_value == self._value
            and next_headline == self._headline
            and next_detail == self._detail
        ):
            return
        self._value = safe_value
        self._headline = next_headline
        self._detail = next_detail
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        side = float(min(self.width(), self.height()) - 10)
        ring_rect = QRectF(
            (self.width() - side) / 2.0,
            (self.height() - side) / 2.0,
            side,
            side,
        )
        ring_width = max(10.0, side * 0.085)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(cvops_qcolor("text_bright", 10))
        painter.drawEllipse(ring_rect)

        painter.setBrush(cvops_qcolor("bg_void", 150))
        inner = ring_rect.adjusted(ring_width + 6, ring_width + 6, -(ring_width + 6), -(ring_width + 6))
        painter.drawEllipse(inner)

        base_pen = QPen(cvops_qcolor("line_light", 80), ring_width)
        base_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(base_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(ring_rect.adjusted(ring_width / 2.0, ring_width / 2.0, -ring_width / 2.0, -ring_width / 2.0), 90 * 16, -360 * 16)

        span = int(round((self._value / 100.0) * 360.0 * 16.0))
        progress_pen = QPen(cvops_qcolor("accent_active"), ring_width)
        progress_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(progress_pen)
        painter.drawArc(
            ring_rect.adjusted(ring_width / 2.0, ring_width / 2.0, -ring_width / 2.0, -ring_width / 2.0),
            90 * 16,
            -span,
        )

        # Center text scaled to diameter — fixed offsets + min(font) were tuned for ~158px
        # widgets and clipped "100%" / detail on smaller QA circles (reading as ".00s").
        margin = max(4.0, min(inner.width(), inner.height()) * 0.07)
        tr = inner.adjusted(margin, margin, -margin, -margin).toRect()
        avail_w = max(1, tr.width())
        avail_h = max(1, tr.height())
        headline = str(self._headline or "").strip()
        detail = str(self._detail or "").strip()
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)

        def _fit_font(*, bold: bool, text: str, start_pt: int, min_pt: int) -> QFont:
            f = QFont(mono)
            f.setBold(bold)
            for pt in range(start_pt, min_pt - 1, -1):
                f.setPointSize(pt)
                if QFontMetrics(f).horizontalAdvance(text) <= avail_w:
                    return f
            f.setPointSize(min_pt)
            return f

        h_start = max(9, min(22, int(self._diameter * 0.11)))
        d_start = max(7, min(13, int(self._diameter * 0.075)))
        h_line_h = 0
        d_line_h = 0
        hf = df = QFont(mono)
        if headline:
            hf = _fit_font(bold=True, text=headline, start_pt=h_start, min_pt=8)
            h_line_h = QFontMetrics(hf).lineSpacing()
        if detail:
            df = _fit_font(bold=False, text=detail, start_pt=d_start, min_pt=7)
            d_line_h = QFontMetrics(df).lineSpacing()
        gap = 2 if headline and detail else 0
        block_h = h_line_h + gap + d_line_h
        y0 = tr.y() + max(0, (avail_h - block_h) // 2)
        line_flags = (
            Qt.AlignmentFlag.AlignHCenter
            | Qt.AlignmentFlag.AlignVCenter
            | Qt.TextFlag.TextSingleLine
        )

        if headline:
            painter.setFont(hf)
            fm = QFontMetrics(hf)
            show_h = fm.elidedText(headline, Qt.TextElideMode.ElideRight, avail_w)
            painter.setPen(cvops_qcolor("text_bright"))
            painter.drawText(QRect(tr.x(), y0, avail_w, h_line_h), line_flags, show_h)
            y0 += h_line_h + gap
        if detail:
            painter.setFont(df)
            fm = QFontMetrics(df)
            show_d = fm.elidedText(detail, Qt.TextElideMode.ElideRight, avail_w)
            painter.setPen(cvops_qcolor("text_iron"))
            painter.drawText(QRect(tr.x(), y0, avail_w, d_line_h), line_flags, show_d)


class _MiniScrapeSpinner(QWidget):
    """Small indeterminate arc for the scrape job list (runs while a worker is active)."""

    _SIDE = 18

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._angle = 0
        self.setFixedSize(self._SIDE, self._SIDE)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._timer = QTimer(self)
        self._timer.setInterval(55)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    def _tick(self) -> None:
        self._angle = (self._angle + 42) % 360
        self.update()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        super().closeEvent(event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        margin = 2.0
        side = float(self._SIDE) - margin * 2
        rect = QRectF(margin, margin, side, side)
        pen_w = max(2.0, side * 0.18)
        track = QPen(cvops_qcolor("line_light", 72), pen_w)
        track.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(track)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(rect.adjusted(pen_w / 2, pen_w / 2, -pen_w / 2, -pen_w / 2), 0, 360 * 16)
        accent = QPen(cvops_qcolor("accent_active"), pen_w)
        accent.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(accent)
        start = (self._angle - 90) * 16
        span = -110 * 16
        painter.drawArc(rect.adjusted(pen_w / 2, pen_w / 2, -pen_w / 2, -pen_w / 2), int(start), int(span))


class _ScrapeJobListRow(QWidget):
    """Job slug with a trailing status spinner/chip.

    The row widget stays mouse-transparent so the parent QListWidget continues
    to own selection and keyboard focus behavior.
    """

    def __init__(
        self,
        slug: str,
        *,
        working: bool,
        badge: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("scrapeJobListRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setProperty("selected", False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self._name = QLabel(slug)
        self._name.setObjectName("scrapeJobNameRow")
        self._name.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        repolish(self._name)
        layout.addWidget(self._name, stretch=1)

        self._trail = QWidget()
        self._trail.setObjectName("scrapeJobTrail")
        self._trail.setFixedWidth(56)
        self._trail.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        trail_l = QVBoxLayout(self._trail)
        trail_l.setContentsMargins(0, 0, 0, 0)
        trail_l.setSpacing(0)
        self._badge: Optional[QWidget] = None
        if working:
            self._badge = _MiniScrapeSpinner(self._trail)
            trail_l.addWidget(self._badge, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        elif badge:
            chip = QLabel(badge)
            chip.setObjectName("scrapeJobBadge")
            chip.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            chip.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if badge == "Failed":
                chip.setProperty("tone", "error")
            elif badge == "Paused":
                chip.setProperty("tone", "paused")
            repolish(chip)
            self._badge = chip
            trail_l.addWidget(chip, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(self._trail, 0, Qt.AlignmentFlag.AlignVCenter)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", bool(selected))
        repolish(self)
        repolish(self._name)
        repolish(self._trail)
        if self._badge is not None:
            repolish(self._badge)


def _scrape_panel_stylesheet() -> str:
    """Flat chrome scoped to Scrape panel — avoids nested opsCell / global side-rule spam."""
    a = cvops_color("accent_active")
    hair = cvops_color("line_light")
    surface = cvops_color("bg_graphite")
    text_iron = cvops_color("text_iron")
    text_signal = cvops_color("text_signal")
    text_bright = cvops_color("text_bright")
    selection_fill = cvops_rgba("selection_active", 0.88)
    selection_fill_strong = cvops_rgba("selection_active", 0.96)
    selection_edge = cvops_rgba("selection_edge", 0.92)
    selection_text = cvops_color("selection_text")
    # Keep job selection aligned with the app-wide matte violet selection token.
    job_selected_fill = selection_fill
    job_selected_row_fill = selection_fill_strong
    job_selected_edge = selection_edge
    return f"""
QWidget#scrapePanel {{
    border: none;
    background: transparent;
}}
/* Sidebar & sections: no rimmed boxes */
QWidget#scrapePanel QWidget#scrapeSidebar,
QWidget#scrapePanel QWidget#scrapeSection {{
    border: none;
    background: transparent;
}}
QWidget#scrapePanel QScrollArea#scrapeQaScroll,
QWidget#scrapePanel QWidget#scrapeQaViewport,
QWidget#scrapePanel QWidget#scrapeQaBody {{
    border: none;
    background: transparent;
}}
QWidget#scrapePanel QWidget#scrapeNavStrip {{
    border: none;
    border-bottom: 1px solid {hair};
    background: transparent;
    margin-bottom: 4px;
    padding-bottom: 4px;
}}
QWidget#scrapePanel QToolButton {{
    border: none;
    background: transparent;
    color: {text_iron};
    font-weight: 600;
    padding: 6px 10px;
    margin-right: 2px;
}}
QWidget#scrapePanel QToolButton:hover {{
    color: {text_signal};
    background: {cvops_rgba('accent_active', 0.05)};
}}
QWidget#scrapePanel QToolButton:checked {{
    color: {selection_text};
    background: {selection_fill};
    border: none;
    border-bottom: 2px solid {selection_edge};
}}
/* Lists & logs — drop global left/right strokes */
QWidget#scrapePanel QListWidget {{
    border: none;
    background: {cvops_rgba('bg_panel', 0.42)};
    border-radius: 0px;
    outline: none;
}}
QWidget#scrapePanel QListWidget#scrapeJobList {{
    background: {cvops_rgba('bg_panel', 0.30)};
    alternate-background-color: {cvops_rgba('bg_graphite', 0.30)};
    border-top: 1px solid {cvops_rgba('line_light', 0.38)};
    border-bottom: 1px solid {cvops_rgba('line_light', 0.38)};
}}
QWidget#scrapePanel QListWidget#scrapeJobList::item {{
    padding: 6px 8px;
    margin: 0px;
    min-height: 22px;
}}
QWidget#scrapePanel QListWidget#scrapeJobList::item:selected,
QWidget#scrapePanel QListWidget#scrapeJobList::item:selected:active,
QWidget#scrapePanel QListWidget#scrapeJobList::item:selected:!active {{
    background: {job_selected_fill};
    color: {selection_text};
}}
QWidget#scrapePanel QWidget#scrapeJobListRow,
QWidget#scrapePanel QWidget#scrapeJobListRow QWidget {{
    background: transparent;
    border: none;
}}
QWidget#scrapePanel QWidget#scrapeJobListRow[selected="true"] {{
    background: {job_selected_row_fill};
    border-left: 2px solid {job_selected_edge};
}}
QWidget#scrapePanel QListWidget:focus {{
    border: none;
}}
QWidget#scrapePanel QTextEdit {{
    border: none;
    background: {cvops_rgba('bg_panel', 0.42)};
    border-radius: 0px;
}}
QWidget#scrapePanel QListWidget#rawGalleryList {{
    background: {cvops_rgba('bg_panel', 0.38)};
}}
QWidget#scrapePanel QListWidget#rawGalleryList::item {{
    padding: 4px;
    margin: 2px;
    min-height: 0px;
}}
QWidget#scrapePanel QListWidget#rawGalleryList::item:selected {{
    background: {selection_fill};
    color: {selection_text};
}}
QWidget#scrapePanel QListWidget#stagedGalleryList,
QWidget#scrapePanel QListWidget#fullGalleryList {{
    background: {cvops_rgba('bg_panel', 0.38)};
}}
QWidget#scrapePanel QListWidget#stagedGalleryList::item,
QWidget#scrapePanel QListWidget#fullGalleryList::item {{
    padding: 4px;
    margin: 2px;
    min-height: 0px;
}}
QWidget#scrapePanel QListWidget#stagedGalleryList::item:selected,
QWidget#scrapePanel QListWidget#fullGalleryList::item:selected {{
    background: {selection_fill};
    color: {selection_text};
}}
QWidget#scrapePanel QFrame#rawPreviewDataCard {{
    background: {cvops_rgba('accent_active', 0.055)};
    border: none;
    border-left: 3px solid {cvops_rgba('accent_active', 0.42)};
    border-radius: 4px;
}}
QWidget#scrapePanel QLabel#rawPreviewEyebrow {{
    color: {a};
    font-weight: 700;
    font-size: 11px;
    font-family: {WB_FONT_MONO};
    border: none;
}}
QWidget#scrapePanel QLabel#rawPreviewName {{
    color: {text_bright};
    font-weight: 700;
    font-size: 12px;
    border: none;
}}
QWidget#scrapePanel QLabel#rawPreviewMeta {{
    color: {text_iron};
    font-size: 10px;
    font-family: {WB_FONT_MONO};
    border: none;
}}
QWidget#scrapePanel QGraphicsView {{
    border: none;
    background: {surface};
    border-radius: 0px;
}}
QWidget#scrapePanel QProgressBar {{
    border: none;
    border-radius: 0px;
    background: {cvops_rgba('line_light', 0.20)};
}}
QWidget#scrapePanel QProgressBar::chunk {{
    background: {cvops_rgba('accent_active', 0.36)};
}}
QWidget#scrapePanel QLineEdit,
QWidget#scrapePanel QSpinBox,
QWidget#scrapePanel QDoubleSpinBox {{
    border: none;
    border-radius: 0px;
    background: {cvops_rgba('bg_panel', 0.54)};
}}
QWidget#scrapePanel QSpinBox:focus,
QWidget#scrapePanel QDoubleSpinBox:focus,
QWidget#scrapePanel QLineEdit:focus {{
    border: none;
}}
/* Buttons: solid fills instead of dashed side rules */
QWidget#scrapePanel QPushButton {{
    border: none;
    padding: 5px 10px;
    background: {cvops_rgba('bg_panel', 0.66)};
}}
QWidget#scrapePanel QPushButton:hover {{
    background: {cvops_rgba('accent_active', 0.12)};
}}
QWidget#scrapePanel QPushButton:pressed,
QWidget#scrapePanel QPushButton:checked {{
    background: {selection_fill_strong};
    color: {selection_text};
}}
QWidget#scrapePanel QPushButton[isPrimary="true"] {{
    border: none;
    background: {cvops_rgba('accent_active', 0.18)};
    color: {text_bright};
    font-weight: 600;
}}
QWidget#scrapePanel QPushButton[isPrimary="true"]:hover {{
    background: {cvops_rgba('accent_active', 0.28)};
}}
QWidget#scrapePanel QListWidget::item {{
    padding: 6px 8px;
    min-height: 18px;
}}
QWidget#scrapePanel QLabel#scrapeJobBadge {{
    color: {a};
    font-weight: 700;
    font-size: 10px;
    font-family: {WB_FONT_MONO};
    border: none;
    min-width: 36px;
}}
QWidget#scrapePanel QLabel#scrapeJobBadge[tone="error"] {{
    color: {cvops_color('accent_alert')};
}}
QWidget#scrapePanel QLabel#scrapeJobBadge[tone="paused"] {{
    color: {text_iron};
    font-weight: 600;
}}
QWidget#scrapePanel QLabel#scrapeJobNameRow {{
    color: {text_signal};
    font-weight: 600;
    font-size: 12px;
    border: none;
}}
QWidget#scrapePanel QWidget#scrapeJobListRow[selected="true"] QLabel#scrapeJobNameRow {{
    color: {text_bright};
}}
QWidget#scrapePanel QWidget#scrapeJobListRow[selected="true"] QLabel#scrapeJobBadge {{
    color: {text_bright};
}}
"""


# ------------------------------------------------------------------ #
# Annotation canvas
# ------------------------------------------------------------------ #

class _AnnotationCanvas(QGraphicsView):
    """Rubber-band bounding boxes with per-box class labels drawn on canvas.

    Boxes are stored as ``[cls_idx, cx, cy, w, h]`` (YOLO normalised).
    When ``class_picker`` is set, each finished rectangle opens that callback
    to choose ``cls_idx`` (interactive label); otherwise ``active_class_idx`` is used.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._img_w = 1
        self._img_h = 1
        self._start_pt: Optional[tuple[float, float]] = None
        self._rubber: Optional[QGraphicsRectItem] = None
        self._draw_items: list[QGraphicsRectItem] = []
        self._label_items: list[QGraphicsSimpleTextItem] = []
        self._model_items: list[QGraphicsRectItem] = []
        self.boxes: list[list[float]] = []
        self.active_class_idx: int = 0
        self.class_names: list[str] = []
        self.class_picker: Optional[Callable[[], int]] = None

    def load_image(self, path: Path) -> None:
        self._scene.clear()
        self._draw_items.clear()
        self._label_items.clear()
        self._model_items.clear()
        self.boxes.clear()
        pix = QPixmap(str(path))
        if pix.isNull():
            return
        self._pix_item = self._scene.addPixmap(pix)
        self._img_w = pix.width()
        self._img_h = pix.height()
        self._scene.setSceneRect(0, 0, self._img_w, self._img_h)
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def load_boxes(self, boxes: list[list[float]]) -> None:
        """Re-draw saved boxes (YOLO normalised) over the current image."""
        for item in self._draw_items:
            self._scene.removeItem(item)
        for item in self._label_items:
            self._scene.removeItem(item)
        self._draw_items.clear()
        self._label_items.clear()
        self.boxes = [list(b) for b in boxes]
        for b in self.boxes:
            cls_idx = int(b[0])
            _, cx, cy, nw, nh = b
            x = (cx - nw / 2) * self._img_w
            y = (cy - nh / 2) * self._img_h
            w = nw * self._img_w
            h = nh * self._img_h
            self._add_rect_item(x, y, w, h, cls_idx)

    def _class_title(self, cls_idx: int) -> str:
        ci = int(cls_idx)
        if 0 <= ci < len(self.class_names):
            return self.class_names[ci]
        return f"class_{ci}"

    def _add_rect_item(self, x: float, y: float, w: float, h: float, cls_idx: int) -> None:
        pen = QPen(_DETECTION_BOX_RED, 2)
        brush = QBrush(_DETECTION_BOX_FILL_RED)
        item = self._scene.addRect(x, y, w, h, pen, brush)
        item.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
        item.setZValue(3)
        self._draw_items.append(item)

        title = self._class_title(cls_idx)
        ti = QGraphicsSimpleTextItem(title)
        ti.setBrush(QBrush(cvops_qcolor("text_bright")))
        mono = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        if mono.pointSize() <= 0:
            mono.setPointSize(10)
        ti.setFont(mono)
        ti.setPos(x + 2, max(0.0, y - 18))
        ti.setZValue(4)
        self._scene.addItem(ti)
        self._label_items.append(ti)

    def clear_boxes(self) -> None:
        for item in self._draw_items:
            self._scene.removeItem(item)
        for item in self._label_items:
            self._scene.removeItem(item)
        self._draw_items.clear()
        self._label_items.clear()
        self.boxes.clear()

    def clear_model_detections(self) -> None:
        for item in self._model_items:
            self._scene.removeItem(item)
        self._model_items.clear()

    def reset(self) -> None:
        self._scene.clear()
        self._pix_item = None
        self._img_w = 1
        self._img_h = 1
        self._start_pt = None
        self._rubber = None
        self._draw_items.clear()
        self._label_items.clear()
        self._model_items.clear()
        self.boxes.clear()

    def set_model_detections(
        self,
        detections: list[dict[str, Any]],
        *,
        highlight_idx: int = -1,
    ) -> None:
        self.clear_model_detections()
        for idx, det in enumerate(detections or []):
            try:
                x1 = float(det["x1"])
                y1 = float(det["y1"])
                x2 = float(det["x2"])
                y2 = float(det["y2"])
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            highlight = idx == int(highlight_idx)
            pen = QPen(
                _DETECTION_BOX_RED,
                2 if highlight else 1,
                Qt.PenStyle.DashLine,
            )
            brush = QBrush(_DETECTION_BOX_HIGHLIGHT_FILL_RED if highlight else _DETECTION_BOX_FILL_RED)
            rect_item = self._scene.addRect(x1, y1, x2 - x1, y2 - y1, pen, brush)
            rect_item.setZValue(1)
            self._model_items.append(rect_item)

    # -- mouse drawing --

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            sp = self.mapToScene(event.pos())
            self._start_pt = (sp.x(), sp.y())
            pen = QPen(_DETECTION_BOX_RED, 2, Qt.PenStyle.DashLine)
            self._rubber = self._scene.addRect(sp.x(), sp.y(), 0, 0, pen)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._start_pt and self._rubber is not None:
            ep = self.mapToScene(event.pos())
            x = min(self._start_pt[0], ep.x())
            y = min(self._start_pt[1], ep.y())
            w = abs(ep.x() - self._start_pt[0])
            h = abs(ep.y() - self._start_pt[1])
            self._rubber.setRect(x, y, w, h)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton and self._start_pt and self._rubber is not None:
            ep = self.mapToScene(event.pos())
            x = min(self._start_pt[0], ep.x())
            y = min(self._start_pt[1], ep.y())
            w = abs(ep.x() - self._start_pt[0])
            h = abs(ep.y() - self._start_pt[1])
            self._scene.removeItem(self._rubber)
            self._rubber = None
            self._start_pt = None
            if w > 2 and h > 2:
                if self.class_picker is not None:
                    cls_idx = int(self.class_picker())
                else:
                    cls_idx = int(self.active_class_idx)
                if cls_idx < 0:
                    super().mouseReleaseEvent(event)
                    return
                cx = (x + w / 2) / self._img_w
                cy = (y + h / 2) / self._img_h
                nw = w / self._img_w
                nh = h / self._img_h
                self._add_rect_item(x, y, w, h, cls_idx)
                self.boxes.append(
                    [
                        float(cls_idx),
                        max(0.0, min(1.0, cx)),
                        max(0.0, min(1.0, cy)),
                        max(0.0, min(1.0, nw)),
                        max(0.0, min(1.0, nh)),
                    ]
                )
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._pix_item is not None:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)


# ------------------------------------------------------------------ #
# Main panel
# ------------------------------------------------------------------ #

class ScrapePanel(QWidget):
    """Native PyQt6 port of the Streamlit ``scrap_panel.render()`` function."""

    errorRaised = pyqtSignal(str)
    qaStateChanged = pyqtSignal()
    qaScanCompletedForPromotion = pyqtSignal(str, int)
    jobSelected = pyqtSignal(str)
    stageChanged = pyqtSignal(str)  # "collect" | "label" | "gallery" | "emit" | "status"
    importDatasetRequested = pyqtSignal()  # user wants to import an already-downloaded dataset folder

    # Stack page indices
    _PAGE_EMPTY = 0
    _PAGE_ACTIVITY = 1
    _PAGE_LABEL = 2
    _PAGE_GALLERY = 3
    _PAGE_STATUS = 4

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        *,
        side_panes: Optional[list[tuple[str, QWidget]]] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("scrapePanel")
        set_cvops_stylesheet(self, _scrape_panel_stylesheet)

        self._side_pane_defs: list[tuple[str, QWidget]] = list(side_panes or [])
        self._side_pane_stack: Optional[DropdownPaneStack] = None
        self._active_slug: Optional[str] = None
        self._staged_images: list[Path] = []
        self._label_strip: list[tuple[Path, str]] = []
        self._gallery_views: list[tuple[QListWidget, QSize, QSize]] = []
        self._gallery_summaries: list[QLabel] = []
        self._current_idx: int = 0
        self._qa_lock = threading.RLock()
        self._qa_state: dict[str, Any] = {
            "slug": "",
            "running": False,
            "message": "",
            "error": "",
            "items": {},
            "image_names": [],
            "model_path": _QA_DEFAULT_MODEL,
            "resolved_model_path": "",
            "conf_threshold": _QA_DEFAULT_CONF,
            "completed": 0,
            "total": 0,
            "last_scan_at": 0.0,
        }
        self._qa_live: dict[str, threading.Event] = {}
        self._qa_filter_key: str = _QA_ALL_FILTER
        self._qa_highlight_det_idx: int = -1
        self._target_count_spins: list[QSpinBox] = []
        self._raw_thumb_cache_slug = ""
        self._raw_thumb_cache: dict[str, QIcon] = {}

        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(1000)
        self._poll_timer.timeout.connect(self._poll_job)

        # One Event per job slug — set while a scrape worker thread is active.
        self._scrape_live: dict[str, threading.Event] = {}

        self._build_ui()
        self._populate_qa_models()
        self.qaStateChanged.connect(self._on_qa_state_changed)
        self._refresh_job_list()

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(12, 14, 12, 14)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(4)
        root.addWidget(splitter)

        # Left: compact job catalog — scraper controls and optional helper panes
        # share a dropdown-backed side pane to keep the collect workspace narrow.
        left = QWidget()
        left.setObjectName("scrapeSidebar")
        left.setMinimumWidth(268)
        left.setMaximumWidth(380)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 4, 12, 8)
        ll.setSpacing(8)

        lbl = QLabel("Job scrape catalog")
        lbl.setProperty("isTitle", True)
        repolish(lbl)
        ll.addWidget(lbl)

        self._side_pane_stack = DropdownPaneStack()
        self._side_pane_stack.setObjectName("scrapeSidePaneStack")
        ll.addWidget(self._side_pane_stack, stretch=1)

        jobs_page = QWidget()
        jobs_page.setObjectName("scrapeJobsPane")
        jobs_l = QVBoxLayout(jobs_page)
        jobs_l.setContentsMargins(0, 0, 0, 0)
        jobs_l.setSpacing(10)

        hint = QLabel(
            "Scrape the web for fresh images, or import a dataset you already downloaded. "
            "Web images are for research use only — respect site terms and robots rules."
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        repolish(hint)
        jobs_l.addWidget(hint)

        self._new_job_btn = QPushButton("New job")
        self._new_job_btn.setProperty("isPrimary", True)
        self._new_job_btn.setMinimumHeight(34)
        repolish(self._new_job_btn)
        self._new_job_btn.clicked.connect(self._toggle_new_job_card)
        jobs_l.addWidget(self._new_job_btn)

        # Bring-your-own ingestion: drop in an already-downloaded dataset folder
        # (YOLO / ImageFolder / audio / ...) and go straight to making a model.
        self._import_dataset_btn = QPushButton("Import dataset")
        self._import_dataset_btn.setMinimumHeight(34)
        self._import_dataset_btn.setToolTip(
            "Import an already-downloaded dataset folder into the library, then create a model from it."
        )
        self._import_dataset_btn.clicked.connect(self.importDatasetRequested.emit)
        jobs_l.addWidget(self._import_dataset_btn)

        self._delete_job_btn = QPushButton("Delete job + all data")
        self._delete_job_btn.setMinimumHeight(34)
        self._delete_job_btn.setEnabled(False)
        self._delete_job_btn.setToolTip(
            "Permanently delete the selected job, its raw/ images, staged/ images, labels, and QA cache."
        )
        self._delete_job_btn.clicked.connect(self._on_delete_job)
        jobs_l.addWidget(self._delete_job_btn)

        self._job_list = QListWidget()
        self._job_list.setObjectName("scrapeJobList")
        self._job_list.setFrameShape(QFrame.Shape.NoFrame)
        self._job_list.setAlternatingRowColors(True)
        self._job_list.currentItemChanged.connect(self._on_job_selected)
        jobs_l.addWidget(self._job_list, stretch=1)

        self._new_job_card = self._build_new_job_card()
        self._new_job_card.setVisible(False)
        jobs_l.addWidget(self._new_job_card)

        self._side_pane_stack.addTab(jobs_page, "Scrape Jobs")
        for title, pane in self._side_pane_defs:
            self.add_side_pane(title, pane)

        splitter.addWidget(left)

        # Right: stacked pages
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 8, 8, 12)
        rl.setSpacing(22)

        self._job_context = QLabel("")
        self._job_context.setWordWrap(True)
        self._job_context.setProperty("muted", True)
        repolish(self._job_context)
        self._job_context.setVisible(False)
        rl.addWidget(self._job_context)

        nav_wrap = QWidget()
        nav_wrap.setObjectName("scrapeNavStrip")
        self._nav_wrap = nav_wrap
        nav_lay = QHBoxLayout(nav_wrap)
        nav_lay.setContentsMargins(0, 0, 0, 0)
        nav_lay.setSpacing(4)
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        for label, page in (
            ("Activity", self._PAGE_ACTIVITY),
            ("Label", self._PAGE_LABEL),
            ("Gallery", self._PAGE_GALLERY),
            ("Status", self._PAGE_STATUS),
        ):
            tb = QToolButton()
            tb.setText(label)
            tb.setCheckable(True)
            tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
            tb.setAutoRaise(True)
            tb.setMinimumHeight(36)
            tb.setProperty("_page", page)
            tb.clicked.connect(self._on_nav_clicked)
            self._nav_group.addButton(tb)
            nav_lay.addWidget(tb)
        nav_lay.addStretch()
        rl.addWidget(nav_wrap)

        self._right_stack = QStackedWidget()
        rl.addWidget(self._right_stack, stretch=1)

        # Page 0: empty
        empty_page = QWidget()
        ev = QVBoxLayout(empty_page)
        ev.setContentsMargins(24, 28, 28, 28)
        ev.setSpacing(16)
        empty_title = QLabel("No job selected")
        empty_title.setProperty("isTitle", True)
        repolish(empty_title)
        ev.addWidget(empty_title)
        empty_body = QLabel(
            "Choose an existing job or start a new scrape. "
            "Use Activity for scrape state and log; use Label (gallery + data card) to draw boxes, assign a class per box, "
            "and publish the dataset from the Emit / integrate section at the bottom of the editor."
        )
        empty_body.setWordWrap(True)
        empty_body.setProperty("muted", True)
        repolish(empty_body)
        ev.addWidget(empty_body)
        ev.addStretch()
        self._right_stack.addWidget(empty_page)

        # Page 1: Activity — standalone output area (state, log, labeling coverage)
        self._right_stack.addWidget(self._build_activity_page())

        # Page 2: Label — gallery + annotation + QA in one data card
        self._right_stack.addWidget(self._build_label_page())

        # Page 3: Gallery — full-size inspection grid synced to Label
        self._right_stack.addWidget(self._build_gallery_page())

        # Page 4: raw status JSON. (Emit now lives inside the Label editor as a
        # collapsible section rather than its own islanded page.)
        self._right_stack.addWidget(self._build_status_page())

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 760])

        self._set_nav_enabled(False)

    def add_side_pane(self, title: str, pane: QWidget) -> None:
        stack = getattr(self, "_side_pane_stack", None)
        if stack is None:
            self._side_pane_defs.append((str(title or "Pane"), pane))
            return
        for idx in range(stack.count()):
            if stack.widget(idx) is pane:
                stack.setTabText(idx, str(title or "Pane"))
                return
        stack.addTab(pane, str(title or "Pane"))

    def refresh_theme_styles(self) -> None:
        set_cvops_stylesheet(self, _scrape_panel_stylesheet)
        for child in self.findChildren(_ScrapeAnnotator):
            child.update()

    def _build_new_job_card(self) -> QFrame:
        """Inline form (Ecosystem-style data card) — replaces the modal new-job dialog."""
        card = QFrame()
        card.setObjectName("scrapeNewJobCard")
        set_cvops_stylesheet(
            card,
            lambda: (
                "QFrame#scrapeNewJobCard {"
                f" background: {cvops_rgba('accent_active', 0.055)};"
                " border: none;"
                f" border-left: 3px solid {cvops_rgba('accent_active', 0.42)};"
                " border-radius: 4px; }"
                f"QLabel#scrapeCardTitle {{ color: {cvops_color('accent_active')}; font-weight: 700;"
                f" font-size: 11px; font-family: {WB_FONT_MONO}; border: none; }}"
                f"QLabel#scrapeCardHint {{ color: {cvops_color('text_iron')}; font-size: 10px;"
                f" font-family: {WB_FONT_MONO}; line-height: 1.45; border: none; }}"
            ),
        )

        outer = QVBoxLayout(card)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(16)

        title = QLabel("[DATA CARD] New scrape job")
        title.setObjectName("scrapeCardTitle")
        outer.addWidget(title)

        form = QFormLayout()
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(14)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._new_topic = QLineEdit()
        self._new_topic.setPlaceholderText("e.g. African elephant")
        self._new_topic.setMinimumHeight(30)
        form.addRow("Topic", self._new_topic)

        self._new_query = QLineEdit()
        self._new_query.setPlaceholderText("Optional; defaults to topic if empty")
        self._new_query.setMinimumHeight(30)
        form.addRow("Search query", self._new_query)

        self._new_count = QSpinBox()
        self._new_count.setRange(10, 500)
        self._new_count.setSingleStep(10)
        self._new_count.setMinimumHeight(30)
        self._new_count.setValue(50)
        form.addRow("Target image count", self._new_count)

        outer.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        start_btn = QPushButton("Start scrape")
        start_btn.setProperty("isPrimary", True)
        start_btn.setMinimumHeight(34)
        repolish(start_btn)
        start_btn.clicked.connect(self._submit_new_job_from_card)
        btn_row.addWidget(start_btn)
        close_btn = QPushButton("Close")
        close_btn.setMinimumHeight(34)
        close_btn.clicked.connect(self._hide_new_job_card)
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        outer.addLayout(btn_row)

        return card

    def _toggle_new_job_card(self) -> None:
        vis = not self._new_job_card.isVisible()
        self._new_job_card.setVisible(vis)
        self._new_job_btn.setText("Hide form" if vis else "New job")

    def _hide_new_job_card(self) -> None:
        self._new_job_card.setVisible(False)
        self._new_job_btn.setText("New job")

    def _set_nav_enabled(self, on: bool) -> None:
        for b in self._nav_group.buttons():
            b.setEnabled(on)
        if not on:
            for b in self._nav_group.buttons():
                b.setChecked(False)
        self._delete_job_btn.setEnabled(on)

    def _new_target_count_spin(self) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(10, 5000)
        spin.setSingleStep(10)
        spin.setMinimumHeight(34)
        spin.setValue(50)
        spin.setToolTip(
            "Set a higher raw-image target, then pull more images into the same scrape job."
        )
        spin.valueChanged.connect(
            lambda value, source=spin: self._sync_target_count_spins(int(value), source=source)
        )
        self._target_count_spins.append(spin)
        return spin

    def _sync_target_count_spins(self, value: int, *, source: QSpinBox | None = None) -> None:
        safe_value = max(10, int(value or 0))
        for spin in list(self._target_count_spins):
            if spin is source or spin.value() == safe_value:
                continue
            prior = spin.blockSignals(True)
            spin.setValue(safe_value)
            spin.blockSignals(prior)

    def _refresh_target_count_controls(self, target_count: int) -> None:
        safe_target = max(10, int(target_count or 0))
        for spin in list(self._target_count_spins):
            if spin.hasFocus() or spin.value() == safe_target:
                continue
            prior = spin.blockSignals(True)
            spin.setValue(safe_target)
            spin.blockSignals(prior)

    def _current_target_count_value(self) -> int:
        for spin in list(self._target_count_spins):
            if spin.hasFocus():
                spin.interpretText()
                return int(spin.value() or 0)
        return int(self._target_count_spin.value() or 0)

    # -- Activity page (pipeline output: state / log / coverage) --

    def _build_activity_page(self) -> QWidget:
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(6, 10, 10, 20)
        vl.setSpacing(22)


        self._activity_status = QLabel("")
        self._activity_status.setWordWrap(True)
        self._activity_status.setMinimumHeight(28)
        vl.addWidget(self._activity_status)

        progress_row = QHBoxLayout()
        progress_row.setContentsMargins(0, 2, 0, 2)
        progress_row.addStretch()
        self._scrape_progress_circle = _ScrapeProgressCircle()
        progress_row.addWidget(self._scrape_progress_circle)
        progress_row.addStretch()
        vl.addLayout(progress_row)

        ctr = QHBoxLayout()
        ctr.setSpacing(10)
        self._btn_pause_play = QPushButton("Pause scraping")
        self._btn_pause_play.setMinimumHeight(34)
        self._btn_pause_play.setEnabled(False)
        self._btn_pause_play.clicked.connect(self._on_toggle_pause_play)
        ctr.addWidget(self._btn_pause_play)

        self._btn_continue_scrape = QPushButton("Continue downloads")
        self._btn_continue_scrape.setMinimumHeight(34)
        self._btn_continue_scrape.setToolTip(
            "Download only the images still missing in raw/, then re-stage (same query)."
        )
        self._btn_continue_scrape.clicked.connect(self._on_continue_scrape)
        ctr.addWidget(self._btn_continue_scrape)

        self._btn_restart_scrape = QPushButton("Restart downloads")
        self._btn_restart_scrape.setMinimumHeight(34)
        self._btn_restart_scrape.setToolTip(
            "Clear raw/ for this job and scrape again toward the saved target count."
        )
        self._btn_restart_scrape.clicked.connect(self._on_restart_scrape_downloads)
        ctr.addWidget(self._btn_restart_scrape)

        ctr.addStretch()
        vl.addLayout(ctr)

        target_row = QHBoxLayout()
        target_row.setSpacing(10)
        target_lbl = QLabel("Target raw images")
        target_lbl.setProperty("muted", True)
        repolish(target_lbl)
        target_row.addWidget(target_lbl)

        self._target_count_spin = self._new_target_count_spin()
        target_row.addWidget(self._target_count_spin)

        self._btn_add_more_images = QPushButton("Add more images")
        self._btn_add_more_images.setMinimumHeight(34)
        self._btn_add_more_images.setToolTip(
            "Update the job target and re-run downloads without clearing raw/."
        )
        self._btn_add_more_images.clicked.connect(self._on_add_more_images)
        target_row.addWidget(self._btn_add_more_images)
        target_row.addStretch()
        vl.addLayout(target_row)

        self._activity_progress = QProgressBar()
        self._activity_progress.setRange(0, 100)
        self._activity_progress.setFormat("Images labeled · %p%")
        self._activity_progress.setTextVisible(True)
        self._activity_progress.setMinimumHeight(26)
        vl.addWidget(self._activity_progress)

        log_hdr = QLabel("Pipeline output")
        log_hdr.setProperty("muted", True)
        repolish(log_hdr)
        vl.addWidget(log_hdr)

        self._scrape_trace = QTextEdit()
        self._scrape_trace.setObjectName("scrapeTraceLog")
        self._scrape_trace.setReadOnly(True)
        self._scrape_trace.setFrameShape(QFrame.Shape.NoFrame)
        self._scrape_trace.document().setDocumentMargin(12)
        self._scrape_trace.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        # Disable internal scrollbars — the outer _ScrollTabPage handles all
        # vertical scrolling.  Without this, two nested QScrollAreas compete
        # and cause macOS rubber-band snap-back.
        self._scrape_trace.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scrape_trace.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scrape_trace.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self._scrape_trace.setPlaceholderText(
            "Verbose processing log (Selenium, download, staging). "
            "No messages yet once the scrape has finished unless you restart a job."
        )
        self._scrape_trace.setMinimumHeight(240)
        vl.addWidget(self._scrape_trace)

        return page

    # -- Label page: staged gallery + data card (canvas, classes, save) + QA --

    def _make_gallery_summary(self) -> QLabel:
        summary = QLabel("")
        summary.setWordWrap(True)
        summary.setProperty("muted", True)
        repolish(summary)
        self._gallery_summaries.append(summary)
        return summary

    def _make_gallery_list(
        self,
        object_name: str,
        *,
        icon_size: QSize,
        grid_size: QSize,
        fixed_height: Optional[int] = None,
    ) -> QListWidget:
        gallery = QListWidget()
        gallery.setObjectName(object_name)
        gallery.setFrameShape(QFrame.Shape.NoFrame)
        gallery.setViewMode(QListView.ViewMode.IconMode)
        gallery.setResizeMode(QListView.ResizeMode.Adjust)
        gallery.setMovement(QListView.Movement.Static)
        gallery.setWrapping(True)
        gallery.setUniformItemSizes(True)
        gallery.setIconSize(icon_size)
        gallery.setGridSize(grid_size)
        gallery.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gallery.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        if fixed_height is not None:
            gallery.setFixedHeight(int(fixed_height))
            gallery.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )
        else:
            gallery.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
        gallery.currentItemChanged.connect(self._on_gallery_selected)
        self._gallery_views.append((gallery, QSize(icon_size), QSize(grid_size)))
        return gallery

    def _build_label_page(self) -> QWidget:
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(6, 10, 10, 20)
        vl.setSpacing(12)

        self._staged_gallery_summary = self._make_gallery_summary()
        self._staged_gallery = self._make_gallery_list(
            "stagedGalleryList",
            icon_size=_RAW_THUMB_SIZE,
            grid_size=_LABEL_GALLERY_GRID_SIZE,
        )

        edit_tools = QHBoxLayout()
        edit_tools.setSpacing(10)
        edit_tools.addStretch(1)
        self._show_gallery_btn = QPushButton("Show gallery")
        self._show_gallery_btn.setCheckable(True)
        self._show_gallery_btn.setProperty("navToggle", True)
        self._show_gallery_btn.setMinimumHeight(34)
        self._show_gallery_btn.setToolTip("Show or hide the thumbnail gallery beside the edited preview.")
        self._show_gallery_btn.toggled.connect(self._set_label_gallery_visible)
        edit_tools.addWidget(self._show_gallery_btn)
        vl.addLayout(edit_tools)

        workspace_shell = QWidget()
        workspace_shell.setObjectName("scrapeSection")
        workspace_shell.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        workspace_layout = QHBoxLayout(workspace_shell)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(12)

        data_card = QFrame()
        data_card.setObjectName("rawPreviewDataCard")
        # Shares the top row evenly with the transform preview (50/50). A floor
        # keeps the drawing canvas usable on narrow windows.
        data_card.setMinimumWidth(380)
        data_card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        pl = QVBoxLayout(data_card)
        pl.setContentsMargins(14, 14, 14, 14)
        pl.setSpacing(10)

        eyebrow = QLabel("[DATA CARD] Image & labels (raw or staged)")
        eyebrow.setObjectName("rawPreviewEyebrow")
        pl.addWidget(eyebrow)

        self._label_card_title = QLabel("Select an image from the gallery")
        self._label_card_title.setObjectName("rawPreviewName")
        self._label_card_title.setWordWrap(True)
        pl.addWidget(self._label_card_title)

        canvas_hdr = QLabel("Drag on the image to draw a box — then pick the class for that box.")
        canvas_hdr.setProperty("muted", True)
        repolish(canvas_hdr)
        pl.addWidget(canvas_hdr)

        self._canvas = _AnnotationCanvas()
        self._canvas.setObjectName("annotationCanvas")
        # Lives directly in the card (NOT inside a scroll area) so the view size
        # equals the visible area — the QGraphicsView then re-fits the image to
        # what you see, scaling to fit exactly like the transform preview instead
        # of being cropped behind a scrollbar. Expanding + a low floor lets it
        # shrink/grow with the splitter.
        self._canvas.setMinimumHeight(160)
        self._canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        pl.addWidget(self._canvas, stretch=1)

        self._label_card_meta = QLabel("")
        self._label_card_meta.setWordWrap(True)
        self._label_card_meta.setObjectName("rawPreviewMeta")
        pl.addWidget(self._label_card_meta)

        ch = QLabel("Classes — highlighted row suggests the default in the class picker")
        ch.setProperty("muted", True)
        repolish(ch)
        pl.addWidget(ch)
        cls_row = QHBoxLayout()
        cls_row.setSpacing(12)
        self._class_input = QLineEdit()
        self._class_input.setPlaceholderText("New class name")
        self._class_input.setMinimumHeight(34)
        cls_row.addWidget(self._class_input)
        add_cls_btn = QPushButton("Add class")
        add_cls_btn.setMinimumHeight(36)
        add_cls_btn.clicked.connect(self._on_add_class)
        cls_row.addWidget(add_cls_btn)
        pl.addLayout(cls_row)
        self._class_list = QListWidget()
        self._class_list.setObjectName("scrapeClassList")
        self._class_list.setFrameShape(QFrame.Shape.NoFrame)
        self._class_list.setMinimumHeight(80)
        self._class_list.setMaximumHeight(140)
        self._class_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._class_list.currentRowChanged.connect(self._on_class_selected)
        pl.addWidget(self._class_list)

        self._save_boxes_btn = QPushButton("Save boxes for this image")
        self._save_boxes_btn.setProperty("isPrimary", True)
        self._save_boxes_btn.setMinimumHeight(40)
        repolish(self._save_boxes_btn)
        self._save_boxes_btn.clicked.connect(self._on_save_boxes)
        pl.addWidget(self._save_boxes_btn)

        self._label_data_card = data_card
        self._transform_pane = self._build_transform_preview_pane()
        self._qa_panel = self._build_qa_panel()
        self._qa_panel.setMinimumHeight(200)

        self._label_gallery_panel = QWidget()
        self._label_gallery_panel.setObjectName("scrapeSection")
        self._label_gallery_panel.setMinimumWidth(260)
        self._label_gallery_panel.setMaximumWidth(360)
        self._label_gallery_panel.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Expanding,
        )
        gallery_layout = QVBoxLayout(self._label_gallery_panel)
        gallery_layout.setContentsMargins(8, 0, 0, 8)
        gallery_layout.setSpacing(6)
        gallery_layout.addWidget(self._staged_gallery_summary)
        gallery_layout.addWidget(self._staged_gallery, stretch=1)
        self._label_gallery_panel.setVisible(False)

        workspace_layout.addWidget(self._label_data_card, 1)
        workspace_layout.addWidget(self._transform_pane, 1)
        workspace_layout.addWidget(self._label_gallery_panel, 0)

        editor_splitter = QSplitter(Qt.Orientation.Vertical)
        editor_splitter.setChildrenCollapsible(False)
        editor_splitter.setHandleWidth(3)
        editor_splitter.addWidget(workspace_shell)
        editor_splitter.addWidget(self._qa_panel)
        editor_splitter.setStretchFactor(0, 3)
        editor_splitter.setStretchFactor(1, 2)
        editor_splitter.setSizes([420, 400])
        self._label_editor_splitter = editor_splitter
        vl.addWidget(editor_splitter, stretch=1)

        nav_outer = QVBoxLayout()
        nav_outer.setSpacing(14)
        nav1 = QHBoxLayout()
        nav1.setSpacing(12)
        self._prev_btn = QPushButton("Previous")
        self._prev_btn.setMinimumHeight(36)
        self._prev_btn.clicked.connect(self._on_prev)
        nav1.addWidget(self._prev_btn)
        self._next_btn = QPushButton("Next")
        self._next_btn.setMinimumHeight(36)
        self._next_btn.clicked.connect(self._on_next)
        nav1.addWidget(self._next_btn)
        self._next_unlabeled_btn = QPushButton("Next unlabeled")
        self._next_unlabeled_btn.setMinimumHeight(36)
        self._next_unlabeled_btn.clicked.connect(self._on_next_unlabeled)
        nav1.addWidget(self._next_unlabeled_btn)
        nav1.addStretch()
        nav_outer.addLayout(nav1)
        nav2 = QHBoxLayout()
        nav2.setSpacing(12)
        self._clear_btn = QPushButton("Clear this image")
        self._clear_btn.setMinimumHeight(36)
        self._clear_btn.clicked.connect(self._on_clear)
        nav2.addWidget(self._clear_btn)
        self._img_counter = QLabel("")
        self._img_counter.setProperty("muted", True)
        repolish(self._img_counter)
        nav2.addWidget(self._img_counter)
        nav2.addStretch()
        nav_outer.addLayout(nav2)
        vl.addLayout(nav_outer)

        # Emit / integrate lives here in the editor (collapsed) instead of on its
        # own islanded page, so the whole flow stays in one place.
        self._emit_section = self._build_emit_section()
        vl.addWidget(self._emit_section)

        return page

    def _set_label_gallery_visible(self, visible: bool) -> None:
        panel = getattr(self, "_label_gallery_panel", None)
        if panel is not None:
            panel.setVisible(bool(visible))
        button = getattr(self, "_show_gallery_btn", None)
        if button is not None and button.isChecked() != bool(visible):
            button.blockSignals(True)
            button.setChecked(bool(visible))
            button.blockSignals(False)

    # -- Gallery page: larger inspection grid synced to the Label tab --

    def _build_gallery_page(self) -> QWidget:
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(6, 10, 10, 20)
        vl.setSpacing(14)


        self._full_gallery_summary = self._make_gallery_summary()
        vl.addWidget(self._full_gallery_summary)

        self._full_gallery = self._make_gallery_list(
            "fullGalleryList",
            icon_size=_GALLERY_THUMB_SIZE,
            grid_size=_FULL_GALLERY_GRID_SIZE,
        )
        vl.addWidget(self._full_gallery, stretch=1)

        actions = QHBoxLayout()
        actions.setSpacing(12)
        open_label_btn = QPushButton("Open selected in Label")
        open_label_btn.setMinimumHeight(36)
        open_label_btn.clicked.connect(self._open_current_in_label)
        actions.addWidget(open_label_btn)
        refresh_btn = QPushButton("Refresh gallery")
        refresh_btn.setMinimumHeight(36)
        refresh_btn.clicked.connect(self._refresh_label_page)
        actions.addWidget(refresh_btn)
        actions.addStretch(1)
        vl.addLayout(actions)

        return page

    def _build_qa_panel(self) -> QWidget:
        cell = QWidget()
        cell.setObjectName("scrapeSection")
        cell.setMinimumWidth(300)
        cell.setMaximumWidth(16777215)
        cell.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        outer = QVBoxLayout(cell)
        outer.setContentsMargins(0, 0, 8, 8)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("scrapeQaScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        scroll.viewport().setObjectName("scrapeQaViewport")
        scroll.viewport().setAutoFillBackground(False)

        body = QWidget()
        body.setObjectName("scrapeQaBody")
        body.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        ql = QVBoxLayout(body)
        ql.setContentsMargins(0, 0, 0, 0)
        ql.setSpacing(10)

        # Reflowed for the wide/horizontal workspace: the QA + editor controls are
        # split across three side-by-side columns so the panel stays short instead
        # of forcing a tall vertical scroll the way the old vertical build did.
        #   col1  Scan & grow   col2  Detections   col3  Detection editor
        columns_row = QHBoxLayout()
        columns_row.setContentsMargins(0, 0, 0, 0)
        columns_row.setSpacing(18)

        def _qa_column() -> tuple[QWidget, QVBoxLayout]:
            holder = QWidget()
            # Ignored width so equal stretch factors split the panel into true
            # equal thirds regardless of each column's preferred content width
            # (otherwise the wider Model-QA column would steal horizontal space).
            holder.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            lay = QVBoxLayout(holder)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(10)
            return holder, lay

        col1_w, col1 = _qa_column()
        col2_w, col2 = _qa_column()
        col3_w, col3 = _qa_column()
        columns_row.addWidget(col1_w, 1)
        columns_row.addWidget(col2_w, 1)
        columns_row.addWidget(col3_w, 1)
        ql.addLayout(columns_row, 1)

        hdr = QLabel("Model QA — detector-assisted scrub")
        hdr.setProperty("muted", True)
        repolish(hdr)
        col1.addWidget(hdr)

        self._qa_status = QLabel("No QA scan yet.")
        self._qa_status.setWordWrap(True)
        self._qa_status.setMinimumHeight(34)
        self._qa_status.setSizePolicy(
            QSizePolicy.Policy.Preferred,
            QSizePolicy.Policy.Preferred,
        )
        col1.addWidget(self._qa_status)

        qa_circle_row = QHBoxLayout()
        qa_circle_row.addStretch()
        self._qa_progress_circle = _ScrapeProgressCircle(
            diameter=118,
            tooltip="QA scan coverage across the current staged image set.",
            detail="qa 0 / 0",
        )
        qa_circle_row.addWidget(self._qa_progress_circle)
        qa_circle_row.addStretch()
        col1.addLayout(qa_circle_row)

        growth_hdr = QLabel("Dataset growth")
        growth_hdr.setProperty("muted", True)
        repolish(growth_hdr)
        col1.addWidget(growth_hdr)

        growth_row = QHBoxLayout()
        growth_row.setSpacing(10)
        self._qa_target_count_spin = self._new_target_count_spin()
        growth_row.addWidget(self._qa_target_count_spin, stretch=1)
        self._qa_add_more_btn = QPushButton("Add more images")
        self._qa_add_more_btn.setMinimumHeight(36)
        self._qa_add_more_btn.setToolTip(
            "Raise the target and download more staged candidates into this same job."
        )
        self._qa_add_more_btn.clicked.connect(self._on_add_more_images)
        growth_row.addWidget(self._qa_add_more_btn)
        col1.addLayout(growth_row)

        self._qa_growth_status = QLabel("")
        self._qa_growth_status.setWordWrap(True)
        self._qa_growth_status.setProperty("muted", True)
        repolish(self._qa_growth_status)
        col1.addWidget(self._qa_growth_status)

        model_row = QHBoxLayout()
        model_row.setSpacing(10)
        self._qa_model_combo = QComboBox()
        self._qa_model_combo.setEditable(True)
        self._qa_model_combo.setMinimumHeight(34)
        self._qa_model_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._qa_model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        if self._qa_model_combo.lineEdit() is not None:
            self._qa_model_combo.lineEdit().setPlaceholderText(
                "Detector model reference or path"
            )
        model_row.addWidget(self._qa_model_combo, stretch=1)
        self._qa_conf = QDoubleSpinBox()
        self._qa_conf.setRange(0.01, 0.99)
        self._qa_conf.setDecimals(2)
        self._qa_conf.setSingleStep(0.05)
        self._qa_conf.setValue(_QA_DEFAULT_CONF)
        self._qa_conf.setPrefix("conf ")
        self._qa_conf.setMinimumHeight(34)
        model_row.addWidget(self._qa_conf)
        self._qa_scan_btn = QPushButton("Run QA")
        self._qa_scan_btn.setMinimumHeight(36)
        self._qa_scan_btn.clicked.connect(self._on_run_qa_scan)
        model_row.addWidget(self._qa_scan_btn)
        col1.addLayout(model_row)
        col1.addStretch(1)

        groups_hdr = QLabel("Detection groups")
        groups_hdr.setProperty("muted", True)
        repolish(groups_hdr)
        col2.addWidget(groups_hdr)

        self._qa_group_list = QListWidget()
        self._qa_group_list.setFrameShape(QFrame.Shape.NoFrame)
        self._qa_group_list.setMinimumHeight(104)
        self._qa_group_list.setMaximumHeight(172)
        self._qa_group_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._qa_group_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._qa_group_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._qa_group_list.currentItemChanged.connect(self._on_qa_group_changed)
        col2.addWidget(self._qa_group_list)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(10)
        self._qa_prev_btn = QPushButton("Prev match")
        self._qa_prev_btn.setMinimumHeight(34)
        self._qa_prev_btn.clicked.connect(self._on_qa_prev_match)
        nav_row.addWidget(self._qa_prev_btn, stretch=1)
        self._qa_next_btn = QPushButton("Next match")
        self._qa_next_btn.setMinimumHeight(34)
        self._qa_next_btn.clicked.connect(self._on_qa_next_match)
        nav_row.addWidget(self._qa_next_btn, stretch=1)
        col2.addLayout(nav_row)

        promote_row = QHBoxLayout()
        promote_row.setSpacing(10)
        self._promote_dataset_btn = QToolButton()
        self._promote_dataset_btn.setText("Promote to dataset")
        self._promote_dataset_btn.setMinimumHeight(34)
        self._promote_dataset_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self._promote_dataset_btn.setToolTip(
            "Copy staged images for a detection group into another library dataset's staged/ folder. "
            "Use the arrow for a menu of groups, or click the button for the current group."
        )
        self._promote_menu = QMenu(self)
        self._promote_menu.aboutToShow.connect(self._populate_promote_menu)
        self._promote_dataset_btn.setMenu(self._promote_menu)
        self._promote_dataset_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self._promote_dataset_btn.clicked.connect(self._on_promote_dataset_default)
        promote_row.addWidget(self._promote_dataset_btn)
        col2.addLayout(promote_row)

        self._qa_current_summary = QLabel("")
        self._qa_current_summary.setWordWrap(True)
        self._qa_current_summary.setMinimumHeight(28)
        col2.addWidget(self._qa_current_summary)

        current_hdr = QLabel("Current image detections")
        current_hdr.setProperty("muted", True)
        repolish(current_hdr)
        col2.addWidget(current_hdr)

        self._qa_detection_list = QListWidget()
        self._qa_detection_list.setFrameShape(QFrame.Shape.NoFrame)
        self._qa_detection_list.setMinimumHeight(80)
        self._qa_detection_list.setMaximumHeight(140)
        self._qa_detection_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._qa_detection_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._qa_detection_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._qa_detection_list.currentRowChanged.connect(self._on_qa_detection_selected)
        col2.addWidget(self._qa_detection_list)
        col2.addStretch(1)

        editor_hdr = QLabel("Detection editor")
        editor_hdr.setProperty("muted", True)
        repolish(editor_hdr)
        col3.addWidget(editor_hdr)

        editor_hint = QLabel(
            "Filter by label, drop bad boxes everywhere, or relabel in bulk. "
            "Labels match the detector string exactly (case-sensitive). "
            "From label accepts several classes separated by commas, e.g. human, zebra, cat."
        )
        editor_hint.setWordWrap(True)
        editor_hint.setProperty("muted", True)
        repolish(editor_hint)
        col3.addWidget(editor_hint)

        editor_form = QGridLayout()
        editor_form.setHorizontalSpacing(8)
        editor_form.setVerticalSpacing(6)
        from_lbl = QLabel("From label")
        from_lbl.setProperty("muted", True)
        repolish(from_lbl)
        editor_form.addWidget(from_lbl, 0, 0)
        self._qa_edit_from_combo = QComboBox()
        self._qa_edit_from_combo.setEditable(True)
        self._qa_edit_from_combo.setMinimumHeight(32)
        self._qa_edit_from_combo.setToolTip(
            "One label or several separated by commas (e.g. human, zebra, cat) for bulk remove/relabel. "
            "Editable if the class is not in the list."
        )
        editor_form.addWidget(self._qa_edit_from_combo, 0, 1)
        to_lbl = QLabel("To label")
        to_lbl.setProperty("muted", True)
        repolish(to_lbl)
        editor_form.addWidget(to_lbl, 1, 0)
        self._qa_edit_to_edit = QLineEdit()
        self._qa_edit_to_edit.setPlaceholderText("Target label for bulk relabel…")
        self._qa_edit_to_edit.setMinimumHeight(32)
        editor_form.addWidget(self._qa_edit_to_edit, 1, 1)
        editor_form.setColumnStretch(1, 1)
        col3.addLayout(editor_form)

        editor_bulk_row = QVBoxLayout()
        editor_bulk_row.setSpacing(10)
        self._qa_remove_label_all_btn = QPushButton("Remove label (all images)")
        self._qa_remove_label_all_btn.setMinimumHeight(34)
        self._qa_remove_label_all_btn.setToolTip(
            "Delete every box whose label is listed in From label (comma-separated allowed) across all images."
        )
        self._qa_remove_label_all_btn.clicked.connect(self._on_qa_remove_label_all)
        editor_bulk_row.addWidget(self._qa_remove_label_all_btn)
        self._qa_relabel_all_btn = QPushButton("Relabel (all images)")
        self._qa_relabel_all_btn.setMinimumHeight(34)
        self._qa_relabel_all_btn.setToolTip(
            "Set every box matching any From label (comma-separated) to the To label across all images."
        )
        self._qa_relabel_all_btn.clicked.connect(self._on_qa_relabel_all)
        editor_bulk_row.addWidget(self._qa_relabel_all_btn)
        col3.addLayout(editor_bulk_row)

        editor_one_row = QVBoxLayout()
        editor_one_row.setSpacing(10)
        self._qa_remove_one_btn = QPushButton("Remove selected box")
        self._qa_remove_one_btn.setMinimumHeight(34)
        self._qa_remove_one_btn.setToolTip(
            "Remove the highlighted row in Current image detections for this frame only."
        )
        self._qa_remove_one_btn.clicked.connect(self._on_qa_remove_selected_detection)
        editor_one_row.addWidget(self._qa_remove_one_btn)
        self._qa_relabel_selected_btn = QPushButton("Relabel selected…")
        self._qa_relabel_selected_btn.setMinimumHeight(34)
        self._qa_relabel_selected_btn.setToolTip(
            "Change the label of the highlighted detection on the current image."
        )
        self._qa_relabel_selected_btn.clicked.connect(self._on_qa_relabel_selected_detection)
        editor_one_row.addWidget(self._qa_relabel_selected_btn)
        col3.addLayout(editor_one_row)

        action_row = QVBoxLayout()
        action_row.setSpacing(10)
        self._qa_delete_current_btn = QPushButton("Delete current")
        self._qa_delete_current_btn.setMinimumHeight(36)
        self._qa_delete_current_btn.clicked.connect(self._on_delete_current_staged_image)
        action_row.addWidget(self._qa_delete_current_btn)
        self._qa_delete_matching_btn = QPushButton("Delete matching")
        self._qa_delete_matching_btn.setMinimumHeight(36)
        self._qa_delete_matching_btn.clicked.connect(self._on_delete_matching_staged_images)
        action_row.addWidget(self._qa_delete_matching_btn)
        col3.addLayout(action_row)
        col3.addStretch(1)

        scroll.setWidget(body)
        outer.addWidget(scroll)

        return cell

    # -- Inline transform preview pane (Label tab, right of QA panel) --

    def _build_transform_preview_pane(self) -> QWidget:
        from .transform_preview_dialog import _ReadOnlyBoxCanvas  # noqa: PLC0415

        cell = QWidget()
        cell.setObjectName("scrapeSection")
        cell.setMinimumWidth(260)
        cell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        vl = QVBoxLayout(cell)
        vl.setContentsMargins(8, 0, 0, 8)
        vl.setSpacing(6)

        hdr = QLabel("[TRANSFORM PREVIEW] — label output for current image")
        hdr.setProperty("muted", True)
        repolish(hdr)
        vl.addWidget(hdr)

        self._inline_filter_lbl = QLabel("")
        self._inline_filter_lbl.setStyleSheet(
            "font-size: 10px; color: rgba(10,143,168,0.85); font-weight: 600;"
        )
        vl.addWidget(self._inline_filter_lbl)

        self._inline_canvas: _ReadOnlyBoxCanvas = _ReadOnlyBoxCanvas()
        # Low floor so the top pane can shrink and the QA panel below can grow.
        self._inline_canvas.setMinimumHeight(120)
        self._inline_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        vl.addWidget(self._inline_canvas, stretch=1)

        txt_hdr = QLabel("[OUTPUT] — file content")
        txt_hdr.setStyleSheet(
            "font-size: 10px; font-weight: 600; color: rgba(10,143,168,0.75);"
        )
        vl.addWidget(txt_hdr)

        self._inline_txt = QTextEdit()
        self._inline_txt.setReadOnly(True)
        self._inline_txt.setMaximumHeight(120)
        self._inline_txt.setMinimumHeight(70)
        from PyQt6.QtGui import QFont as _QFont  # noqa: PLC0415
        _mf = _QFont("JetBrains Mono", 9)
        _mf.setStyleHint(_QFont.StyleHint.Monospace)
        self._inline_txt.setFont(_mf)
        self._inline_txt.setStyleSheet(
            "QTextEdit { background: rgba(10,143,168,0.06); "
            "border: 1px solid rgba(10,143,168,0.22); }"
        )
        vl.addWidget(self._inline_txt)

        self._inline_source_lbl = QLabel("")
        self._inline_source_lbl.setWordWrap(True)
        self._inline_source_lbl.setStyleSheet(
            "font-size: 9px; color: rgba(133,153,0,0.55);"
        )
        vl.addWidget(self._inline_source_lbl)

        return cell

    # -- Emit section (lives inside the Label editor, not a separate page) --

    def _build_emit_section(self) -> CollapsibleSection:
        """Compact "publish this dataset" panel folded into the editing layer.

        Turns the staged/labeled job into a properly integrated YOLO dataset +
        training scenario without leaving the editor. Collapsed by default so it
        stays out of the way until the labeling is done."""
        section = CollapsibleSection("Emit / integrate dataset", expanded=False)
        body = section.body_layout()
        body.setContentsMargins(2, 2, 2, 4)
        body.setSpacing(10)

        self._emit_summary = QLabel("")
        self._emit_summary.setWordWrap(True)
        self._emit_summary.setMinimumHeight(28)
        body.addWidget(self._emit_summary)

        controls = QHBoxLayout()
        controls.setSpacing(14)
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        self._emit_val_frac = QDoubleSpinBox()
        self._emit_val_frac.setRange(0.05, 0.50)
        self._emit_val_frac.setSingleStep(0.05)
        self._emit_val_frac.setValue(0.20)
        self._emit_val_frac.setDecimals(2)
        self._emit_val_frac.setMinimumHeight(32)
        form.addRow("Validation fraction", self._emit_val_frac)

        self._emit_epochs = QSpinBox()
        self._emit_epochs.setRange(1, 300)
        self._emit_epochs.setValue(20)
        self._emit_epochs.setMinimumHeight(32)
        form.addRow("Default training epochs", self._emit_epochs)

        self._emit_base_model = QLineEdit("assets/models/yolov10n.pt")
        self._emit_base_model.setMinimumHeight(34)
        form.addRow("Base model path", self._emit_base_model)
        controls.addLayout(form, stretch=1)
        body.addLayout(controls)

        emit_btn = QPushButton("Emit YOLO dataset and scenario")
        emit_btn.setProperty("isPrimary", True)
        emit_btn.setMinimumHeight(40)
        repolish(emit_btn)
        emit_btn.clicked.connect(self._on_emit)
        body.addWidget(emit_btn)

        self._emit_result = QLabel("")
        self._emit_result.setWordWrap(True)
        self._emit_result.setMinimumHeight(28)
        body.addWidget(self._emit_result)

        return section

    def _reveal_emit_section(self) -> None:
        """Bring the Label editor forward and expand the Emit section in place."""
        self._refresh_emit_page()
        section = getattr(self, "_emit_section", None)
        if section is not None:
            section.set_expanded(True)

    # -- Status page --

    def _build_status_page(self) -> QWidget:
        page = QWidget()
        vl = QVBoxLayout(page)
        vl.setContentsMargins(6, 10, 10, 20)
        vl.setSpacing(18)

        self._status_log = QTextEdit()
        self._status_log.setObjectName("logView")
        self._status_log.setReadOnly(True)
        self._status_log.setFrameShape(QFrame.Shape.NoFrame)
        self._status_log.document().setDocumentMargin(14)
        self._status_log.setFont(
            QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
        )
        vl.addWidget(self._status_log, stretch=1)

        refresh_btn = QPushButton("Refresh status")
        refresh_btn.setMinimumHeight(36)
        refresh_btn.clicked.connect(self._refresh_status_page)
        vl.addWidget(refresh_btn)

        return page

    # ------------------------------------------------------------------ #
    # Job list management
    # ------------------------------------------------------------------ #

    def _job_list_working_and_badge(self, slug: str) -> tuple[bool, str]:
        """Whether the scrape worker is active, and a short status label when idle."""
        worker = self._scrape_worker_live(slug)
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
        except Exception:
            job = None
        if job is None:
            return (False, "")
        state = str(job.state or "")
        if worker or state in ("pending", "scraping"):
            return (True, "")
        if state in ("staged", "labeling", "emitted"):
            return (False, "Done")
        if state == "error":
            return (False, "Failed")
        if state == "paused_downloads":
            return (False, "Paused")
        return (False, "")

    def _refresh_job_list(self) -> None:
        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            names = reg.list_library_dataset_names()
            slugs = [n for n in names if JobStore.exists(n)]
        except Exception:
            slugs = []

        current_slug = self._active_slug
        self._job_list.blockSignals(True)
        self._job_list.clear()
        for slug in slugs:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, slug)
            working, badge = self._job_list_working_and_badge(slug)
            row = _ScrapeJobListRow(slug, working=working, badge=badge)
            row.setMinimumHeight(26)
            item.setSizeHint(row.sizeHint().expandedTo(QSize(0, 26)))
            self._job_list.addItem(item)
            self._job_list.setItemWidget(item, row)
        self._job_list.blockSignals(False)

        if current_slug:
            for i in range(self._job_list.count()):
                if self._job_list.item(i).data(Qt.ItemDataRole.UserRole) == current_slug:
                    self._job_list.setCurrentRow(i)
                    break
        self._sync_job_list_row_selection()

    def _sync_job_list_row_selection(self) -> None:
        current = self._job_list.currentItem()
        for i in range(self._job_list.count()):
            item = self._job_list.item(i)
            row = self._job_list.itemWidget(item)
            if isinstance(row, _ScrapeJobListRow):
                row.set_selected(item is current)

    def _on_job_selected(self, current: Optional[QListWidgetItem], _prev) -> None:
        self._sync_job_list_row_selection()
        if current is None:
            self._active_slug = None
            self._qa_filter_key = _QA_ALL_FILTER
            self._qa_highlight_det_idx = -1
            self._job_context.setVisible(False)
            self._job_context.clear()
            self._right_stack.setCurrentIndex(self._PAGE_EMPTY)
            self._set_nav_enabled(False)
            self.jobSelected.emit("")
            return
        slug = current.data(Qt.ItemDataRole.UserRole)
        self._active_slug = slug
        self._job_context.setText(f"Active job: {slug}")
        self._job_context.setVisible(True)
        self._load_job(slug)
        self.jobSelected.emit(str(slug or ""))

    def _preferred_tab_for_job(self, slug: str) -> int:
        """Open Activity during scrape setup; Label when there is something to annotate."""
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
        except Exception:
            job = None
        if job is None:
            return self._PAGE_ACTIVITY
        if job.state in ("pending", "scraping", "paused_downloads"):
            return self._PAGE_ACTIVITY
        if self._get_staged_images(slug):
            return self._PAGE_LABEL
        return self._PAGE_ACTIVITY

    # Map stack pages to the outward-facing stage names used by the Collect & Edit hub.
    _PAGE_TO_STAGE = {
        _PAGE_ACTIVITY: "collect",
        _PAGE_LABEL: "label",
        _PAGE_GALLERY: "gallery",
        _PAGE_STATUS: "status",
    }
    _STAGE_TO_PAGE = {
        "collect": _PAGE_ACTIVITY,
        "label": _PAGE_LABEL,
        "gallery": _PAGE_GALLERY,
        # Emit is folded into the Label editor; route it there and reveal the section.
        "emit": _PAGE_LABEL,
        "status": _PAGE_STATUS,
    }

    def _activate_stack_tab(self, page: int) -> None:
        self._right_stack.setCurrentIndex(page)
        for btn in self._nav_group.buttons():
            if btn.property("_page") == page:
                btn.blockSignals(True)
                btn.setChecked(True)
                btn.blockSignals(False)
                break
        stage = self._PAGE_TO_STAGE.get(int(page), "")
        if stage:
            self.stageChanged.emit(stage)

    def set_nav_strip_visible(self, visible: bool) -> None:
        """Public hook (Collect & Edit hub): hide ScrapePanel's own nav strip when an
        outer stage strip drives navigation, so there is a single set of stage tabs."""
        nav_wrap = getattr(self, "_nav_wrap", None)
        if nav_wrap is not None:
            nav_wrap.setVisible(bool(visible))

    def select_stage(self, stage: str) -> None:
        """Public hook: switch to a named stage (collect/label/gallery/emit/status).

        Refreshes the target page the same way the internal nav does, then activates it.
        When no job is selected the right workspace stays on its empty landing so the
        Collect catalog remains usable without first picking a job.
        """
        stage_norm = str(stage or "").strip().lower()
        page = self._STAGE_TO_PAGE.get(stage_norm)
        if page is None:
            return
        if self._active_slug is None:
            self._right_stack.setCurrentIndex(self._PAGE_EMPTY)
            self.stageChanged.emit(self._PAGE_TO_STAGE.get(page, "collect"))
            return
        if page == self._PAGE_STATUS:
            self._refresh_status_page()
        elif page == self._PAGE_GALLERY:
            self._refresh_label_page()
        elif page == self._PAGE_ACTIVITY:
            self._refresh_activity_page()
        self._activate_stack_tab(page)
        # "emit" routes to the Label page; surface its folded-in section.
        if stage_norm == "emit":
            self._reveal_emit_section()

    def _load_job(self, slug: str) -> None:
        self._staged_images = self._get_staged_images(slug)
        self._current_idx = 0
        self._qa_filter_key = _QA_ALL_FILTER
        self._qa_highlight_det_idx = -1
        self._populate_qa_models()
        self._ensure_qa_state_for_job(slug)
        self._refresh_activity_page()
        self._refresh_label_page()
        self._refresh_emit_page()
        self._refresh_status_page()
        self._set_nav_enabled(True)
        self._activate_stack_tab(self._preferred_tab_for_job(slug))

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # Delete job
    # ------------------------------------------------------------------ #

    def _on_delete_job(self) -> None:
        import shutil  # noqa: PLC0415

        slug = self._active_slug
        if not slug:
            return

        scrape_live = self._scrape_worker_live(slug)
        qa_live = self._qa_scan_live(slug)

        if scrape_live or qa_live:
            running_desc = " and ".join(
                ([" scrape download"] if scrape_live else [])
                + (["QA scan"] if qa_live else [])
            )
            QMessageBox.warning(
                self,
                "Job Is Running",
                f"A {running_desc} is still active for '{slug}'.\n\n"
                "Stop the running operation before deleting the job.",
            )
            return

        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
            dataset_path = reg.resolve_library_dataset_path(slug)
        except Exception as exc:
            self.errorRaised.emit(f"Could not resolve job path: {exc}")
            return

        raw_cnt = sum(1 for p in (dataset_path / "raw").iterdir() if p.is_file()) if (dataset_path / "raw").exists() else 0
        staged_cnt = sum(1 for p in (dataset_path / "staged").iterdir() if p.is_file()) if (dataset_path / "staged").exists() else 0
        topic = job.topic if job else slug

        confirm = QMessageBox.warning(
            self,
            "Delete Job — Cannot Be Undone",
            f"Permanently delete job '{slug}' ({topic})?\n\n"
            f"  raw/     {raw_cnt} file(s)\n"
            f"  staged/  {staged_cnt} file(s)\n"
            f"  scrap.json, QA cache, all labels\n\n"
            f"Full path:  {dataset_path}\n\n"
            "This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            shutil.rmtree(dataset_path, ignore_errors=False)
        except Exception as exc:
            self.errorRaised.emit(f"Delete job failed: {exc}")
            return

        # Clean up in-memory state.
        self._scrape_live.pop(slug, None)
        self._qa_live.pop(slug, None)
        with self._qa_lock:
            if self._qa_state.get("slug") == slug:
                self._qa_state = {}

        if self._active_slug == slug:
            self._active_slug = None
            self._label_strip = []
            self._staged_images = []
            self._current_idx = 0
            self._job_context.setVisible(False)
            self._job_context.clear()
            self._right_stack.setCurrentIndex(self._PAGE_EMPTY)
            self._set_nav_enabled(False)

        self._refresh_job_list()

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    def _show_qa_promotion_flow(self, slug: str, scanned_count: int) -> None:
        """Same promotion guidance as video index export: path + Emit / Database next steps."""
        if self._active_slug != str(slug or "").strip():
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
        except Exception:
            job = None
        if job is None or str(getattr(job, "state", "") or "") == "emitted":
            return
        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415

            ds_path = reg.resolve_library_dataset_path(slug)
            path_txt = str(ds_path.resolve())
        except Exception:
            path_txt = "(could not resolve dataset path)"

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Information)
        msg.setWindowTitle("Scrape QA")
        msg.setText(
            f"QA scan finished for `{slug}` ({int(scanned_count)} staged image(s))."
        )
        msg.setInformativeText(
            f"Dataset folder (staged labels stay here until you emit):\n{path_txt}\n\n"
            "Finish any labeling fixes on the Label tab, then open the Emit / integrate "
            "section at the bottom of the editor and click "
            "\"Emit YOLO dataset and scenario\" to publish the YOLO layout under database/ and "
            "register a training scenario — the same promotion step as the video index pipeline "
            "after frames land in a dataset folder.\n\n"
            "After emit, use the Database tab to review images and labels, and Workbench to train."
        )
        msg.setTextFormat(Qt.TextFormat.PlainText)
        emit_btn = msg.addButton("Open emit panel", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Ok)
        msg.exec()
        if msg.clickedButton() == emit_btn:
            self.select_stage("emit")

    def _on_nav_clicked(self) -> None:
        btn = self.sender()
        if btn is None:
            return
        page = btn.property("_page")
        if self._active_slug is None:
            self._right_stack.setCurrentIndex(self._PAGE_EMPTY)
            return
        if page == self._PAGE_STATUS:
            self._refresh_status_page()
        elif page == self._PAGE_GALLERY:
            self._refresh_label_page()
        elif page == self._PAGE_ACTIVITY:
            self._refresh_activity_page()
        self._right_stack.setCurrentIndex(page)

    def _open_current_in_label(self) -> None:
        if self._active_slug is None:
            return
        self._activate_stack_tab(self._PAGE_LABEL)
        self._reload_canvas_for_current_image()

    # ------------------------------------------------------------------ #
    # Activity + label views
    # ------------------------------------------------------------------ #

    def _get_staged_images(self, slug: str) -> list[Path]:
        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415
            base = reg.resolve_library_dataset_path(slug)
            staged = base / "staged"
            if not staged.exists():
                return []
            return sorted(p for p in staged.iterdir() if p.is_file())
        except Exception:
            return []

    def _get_raw_image_paths(self, slug: str) -> list[Path]:
        """All previewable image files under raw/ (no minimum size — small assets included)."""
        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415

            base = reg.resolve_library_dataset_path(slug)
            raw_dir = base / "raw"
            if not raw_dir.exists():
                return []
            return sorted(
                p
                for p in raw_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
            )
        except Exception:
            return []

    def _build_label_strip(self, slug: str, *, staged_paths: Optional[list[Path]] = None) -> list[tuple[Path, str]]:
        """Unified Label-tab gallery: every raw/ image first, then every staged/ image."""
        raw_paths = self._get_raw_image_paths(slug)
        staged = list(staged_paths) if staged_paths is not None else self._get_staged_images(slug)
        strip: list[tuple[Path, str]] = []
        for p in raw_paths:
            strip.append((p, "raw"))
        for p in staged:
            strip.append((p, "staged"))
        return strip

    def _current_label_path(self) -> Optional[Path]:
        if not self._label_strip or not (0 <= self._current_idx < len(self._label_strip)):
            return None
        return self._label_strip[self._current_idx][0]

    def _raw_thumb_icon(
        self,
        slug: str,
        image_path: Path,
        size: QSize = _RAW_THUMB_SIZE,
    ) -> QIcon:
        if self._raw_thumb_cache_slug != slug:
            self._raw_thumb_cache_slug = slug
            self._raw_thumb_cache.clear()
        thumb_size = QSize(size)
        try:
            stat = image_path.stat()
            key = (
                f"{thumb_size.width()}x{thumb_size.height()}:"
                f"{image_path.name}:{stat.st_size}:{stat.st_mtime_ns}"
            )
        except Exception:
            key = f"{thumb_size.width()}x{thumb_size.height()}:{image_path.name}"
        cached = self._raw_thumb_cache.get(key)
        if cached is not None:
            return cached
        pix = QPixmap(str(image_path))
        if not pix.isNull():
            pix = pix.scaled(
                thumb_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        icon = QIcon(pix) if not pix.isNull() else QIcon()
        self._raw_thumb_cache[key] = icon
        if len(self._raw_thumb_cache) > 1200:
            for old_key in list(self._raw_thumb_cache)[:200]:
                self._raw_thumb_cache.pop(old_key, None)
        return icon

    def _populate_staged_gallery(self) -> None:
        if not self._gallery_views:
            return
        slug = self._active_slug
        if not slug:
            for gallery, _icon_size, _grid_size in self._gallery_views:
                gallery.clear()
            for summary in self._gallery_summaries:
                summary.setText("")
            return
        raw_total = len(self._get_raw_image_paths(slug))
        staged_total = len(self._staged_images)
        total = len(self._label_strip)
        for summary in self._gallery_summaries:
            if total:
                summary.setText(
                    f"Gallery: {raw_total} raw/ + {staged_total} staged/ = {total} thumbnail(s). "
                    "Scroll the grid — nothing is hidden; tiny images stay listed."
                )
            else:
                summary.setText("No image files in raw/ or staged/ yet.")

        row = max(0, min(self._current_idx, total - 1)) if total else -1
        for gallery, icon_size, grid_size in self._gallery_views:
            gallery.blockSignals(True)
            gallery.clear()
            for idx, (image_path, src) in enumerate(self._label_strip):
                caption = f"[{src}] {image_path.name}"
                item = QListWidgetItem(self._raw_thumb_icon(slug, image_path, icon_size), caption)
                item.setData(Qt.ItemDataRole.UserRole, idx)
                item.setToolTip(f"{src.upper()}\n{image_path}")
                item.setSizeHint(grid_size)
                gallery.addItem(item)
            gallery.setCurrentRow(row)
            gallery.blockSignals(False)

    def _sync_gallery_selection(self, source: Optional[QListWidget] = None) -> None:
        row = (
            max(0, min(self._current_idx, len(self._label_strip) - 1))
            if self._label_strip
            else -1
        )
        for gallery, _icon_size, _grid_size in self._gallery_views:
            if gallery is source:
                continue
            gallery.blockSignals(True)
            gallery.setCurrentRow(row)
            gallery.blockSignals(False)

    def _update_label_card_chrome(self, img_path: Path) -> None:
        title = getattr(self, "_label_card_title", None)
        meta = getattr(self, "_label_card_meta", None)
        if title is not None:
            title.setText(img_path.name)
        if meta is None:
            return
        pix = QPixmap(str(img_path))
        size_text = "unreadable"
        if not pix.isNull():
            size_text = f"{pix.width()} x {pix.height()} px"
        try:
            stat = img_path.stat()
            file_text = f"{stat.st_size / (1024 * 1024):.2f} MB"
        except Exception:
            file_text = "unknown size"
        meta.setText(f"{size_text} · {file_text}\n{img_path}")

    def _append_class_and_sync(self, name: str) -> int:
        slug = self._active_slug
        if not slug or not str(name or "").strip():
            return -1
        name = str(name).strip()
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
            if job is None:
                return -1
            if name in job.classes:
                idx = job.classes.index(name)
                self._canvas.class_names = list(job.classes)
                self._class_list.blockSignals(True)
                self._class_list.setCurrentRow(idx)
                self._class_list.blockSignals(False)
                self._canvas.active_class_idx = idx
                return idx
            JobStore.update(slug, classes=job.classes + [name])
            job = JobStore.load(slug)
            if job is None:
                return -1
            classes = list(job.classes)
            self._canvas.class_names = classes
            self._class_list.blockSignals(True)
            self._class_list.clear()
            for c in classes:
                self._class_list.addItem(c)
            ni = classes.index(name)
            self._class_list.setCurrentRow(ni)
            self._class_list.blockSignals(False)
            self._canvas.active_class_idx = ni
            return ni
        except Exception:
            return -1

    def _pick_class_for_new_box(self) -> int:
        slug = self._active_slug
        if not slug:
            return -1
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
        except Exception:
            return -1
        if job is None:
            return -1
        names = list(job.classes)
        if not names:
            text, ok = QInputDialog.getText(
                self,
                "Class for box",
                "No classes yet — type a name for this box (it will be added to the job):",
            )
            if not ok or not str(text or "").strip():
                return -1
            return self._append_class_and_sync(str(text).strip())
        row = self._class_list.currentRow()
        if row < 0:
            row = 0
        default_row = min(row, len(names) - 1)
        choice, ok = QInputDialog.getItem(
            self,
            "Class for box",
            "Label this box:",
            names,
            default_row,
            False,
        )
        if not ok:
            return -1
        try:
            return names.index(str(choice))
        except ValueError:
            return -1

    def _on_gallery_selected(
        self,
        current: Optional[QListWidgetItem],
        _prev: Optional[QListWidgetItem],
    ) -> None:
        if current is None or not self._label_strip:
            return
        source = self.sender()
        gallery = source if isinstance(source, QListWidget) else None
        raw_idx = current.data(Qt.ItemDataRole.UserRole)
        if raw_idx is None:
            idx = gallery.row(current) if gallery is not None else -1
        else:
            idx = int(raw_idx)
        if idx < 0 or idx >= len(self._label_strip):
            return
        if idx == self._current_idx:
            self._sync_gallery_selection(source=gallery)
            return
        self._qa_highlight_det_idx = -1
        self._current_idx = idx
        self._sync_gallery_selection(source=gallery)
        self._reload_canvas_for_current_image()

    def _on_staged_gallery_selected(
        self,
        current: Optional[QListWidgetItem],
        prev: Optional[QListWidgetItem],
    ) -> None:
        self._on_gallery_selected(current, prev)

    def _reload_canvas_for_current_image(self) -> None:
        slug = self._active_slug
        if not slug:
            return
        self._sync_gallery_selection()
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
        except Exception:
            job = None
        if job is None:
            self._canvas.reset()
            return
        self._canvas.class_names = list(job.classes)
        self._canvas.class_picker = self._pick_class_for_new_box
        total = len(self._label_strip)
        self._img_counter.setText(f"{self._current_idx + 1} / {total}" if total else "0 / 0")
        if self._label_strip and 0 <= self._current_idx < len(self._label_strip):
            img_path = self._label_strip[self._current_idx][0]
            self._update_label_card_chrome(img_path)
            self._canvas.load_image(img_path)
            qa_entry = dict(self._qa_snapshot().get("items") or {}).get(img_path.name) or {}
            self._canvas.set_model_detections(
                list(qa_entry.get("detections") or []),
                highlight_idx=self._qa_highlight_det_idx,
            )
            saved = job.labels.get(img_path.name, [])
            if saved:
                self._canvas.load_boxes(saved)
        else:
            lt = getattr(self, "_label_card_title", None)
            lm = getattr(self, "_label_card_meta", None)
            if lt is not None:
                lt.setText("No images in raw/ or staged/")
            if lm is not None:
                lm.clear()
            self._canvas.reset()
        self._refresh_qa_panel()
        self._refresh_transform_pane()

    def _refresh_transform_pane(self) -> None:
        """Recompute and display the live YOLO label transform for the current image."""
        inline_canvas = getattr(self, "_inline_canvas", None)
        inline_txt = getattr(self, "_inline_txt", None)
        inline_filter_lbl = getattr(self, "_inline_filter_lbl", None)
        inline_source_lbl = getattr(self, "_inline_source_lbl", None)
        if inline_canvas is None or inline_txt is None:
            return

        filter_key = getattr(self, "_qa_filter_key", _QA_ALL_FILTER)

        # -- Update filter label --
        if inline_filter_lbl is not None:
            if filter_key in (_QA_ALL_FILTER, _QA_NONE_FILTER):
                inline_filter_lbl.setText(f"Filter: {filter_key}")
            else:
                inline_filter_lbl.setText(f"Filter: '{filter_key}'  →  promoted class")

        if not self._label_strip or not (0 <= self._current_idx < len(self._label_strip)):
            inline_canvas._scene.clear()
            inline_canvas._pix_item = None
            inline_txt.clear()
            if inline_source_lbl is not None:
                inline_source_lbl.setText("No image selected.")
            return

        img_path = self._label_strip[self._current_idx][0]

        # Load job labels for hand-drawn box detection
        slug = self._active_slug
        source_job_labels: dict[str, list] = {}
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
            source_job_labels = dict(job.labels or {})
        except Exception:
            pass

        qa_items = dict(self._qa_snapshot().get("items") or {})

        # Use promoted_class = filter_key if it's a real label, else empty string
        promoted_class = (
            filter_key
            if filter_key not in (_QA_ALL_FILTER, _QA_NONE_FILTER)
            else ""
        )

        preview = compute_transform_preview(
            paths=[img_path],
            filter_key=filter_key,
            promoted_class=promoted_class,
            promoted_class_idx=0,
            source_job_labels=source_job_labels,
            qa_items=qa_items,
        )

        entry = preview.get(img_path.name) or {}
        transformed_boxes: list = list(entry.get("transformed_boxes") or [])
        original_dets: list = list(entry.get("original_dets") or [])
        original_source = str(entry.get("original_source") or "full_frame_fallback")
        yolo_txt = str(entry.get("yolo_txt") or "")

        from PyQt6.QtGui import QColor as _QColor  # noqa: PLC0415
        _TEAL = _QColor(10, 143, 168)
        _TEAL_FILL = _QColor(10, 143, 168, 52)

        inline_canvas.load_image_path(img_path)
        inline_canvas.draw_yolo_boxes(
            transformed_boxes,
            promoted_class or "?",
            _TEAL,
            _TEAL_FILL,
        )

        # Format the text display
        if yolo_txt.strip():
            txt_content = yolo_txt.rstrip()
        else:
            txt_content = "# no boxes"
        inline_txt.setPlainText(txt_content)

        _SOURCE_MAP = {
            "hand_drawn": "hand-drawn boxes (class remapped)",
            "qa_detection": "QA detections → YOLO normalised",
            "full_frame_fallback": "no detections — full-frame fallback",
        }
        if inline_source_lbl is not None:
            inline_source_lbl.setText(_SOURCE_MAP.get(original_source, original_source))

    def _count_raw_files(self, slug: str) -> int:
        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415
            base = reg.resolve_library_dataset_path(slug)
            raw_dir = base / "raw"
            if not raw_dir.exists():
                return 0
            return sum(1 for p in raw_dir.iterdir() if p.is_file())
        except Exception:
            return 0

    def _qa_cache_path(self, slug: str) -> Path:
        from mlops.pipeline import registry as reg  # noqa: PLC0415

        return reg.resolve_library_dataset_path(slug) / _QA_CACHE_NAME

    def _qa_scan_live(self, slug: str | None) -> bool:
        if not slug:
            return False
        ev = self._qa_live.get(slug)
        return bool(ev and ev.is_set())

    def _current_qa_model_path(self) -> str:
        combo = getattr(self, "_qa_model_combo", None)
        if combo is None:
            return _QA_DEFAULT_MODEL
        value = str(combo.currentText() or "").strip()
        return value or _QA_DEFAULT_MODEL

    def _populate_qa_models(self) -> None:
        combo = getattr(self, "_qa_model_combo", None)
        if combo is None:
            return
        current_ref = self._current_qa_model_path()
        combo.blockSignals(True)
        combo.clear()
        options = _list_qa_model_refs()
        for item in options:
            ref = str(item.get("ref") or "").strip()
            if not ref:
                continue
            combo.addItem(ref, userData=ref)
            idx = combo.count() - 1
            combo.setItemData(idx, str(item.get("path") or ""), Qt.ItemDataRole.ToolTipRole)
        if combo.count() == 0:
            combo.addItem(_QA_DEFAULT_MODEL, userData=_QA_DEFAULT_MODEL)
            combo.setItemData(0, str(_resolve_qa_model_reference(_QA_DEFAULT_MODEL)), Qt.ItemDataRole.ToolTipRole)
        target_idx = combo.findText(current_ref)
        if target_idx >= 0:
            combo.setCurrentIndex(target_idx)
        else:
            combo.setEditText(current_ref or _QA_DEFAULT_MODEL)
        combo.blockSignals(False)

    def _current_qa_conf(self) -> float:
        spin = getattr(self, "_qa_conf", None)
        if spin is None:
            return _QA_DEFAULT_CONF
        return float(spin.value())

    def _replace_qa_state(
        self,
        slug: str,
        *,
        running: bool,
        message: str,
        error: str = "",
        items: dict[str, Any] | None = None,
        image_names: list[str] | None = None,
        model_path: str = "",
        resolved_model_path: str = "",
        conf_threshold: float = _QA_DEFAULT_CONF,
        completed: int = 0,
        total: int = 0,
        last_scan_at: float = 0.0,
    ) -> None:
        with self._qa_lock:
            self._qa_state = {
                "slug": slug,
                "running": bool(running),
                "message": str(message or ""),
                "error": str(error or ""),
                "items": dict(items or {}),
                "image_names": list(image_names or []),
                "model_path": str(model_path or _QA_DEFAULT_MODEL),
                "resolved_model_path": str(resolved_model_path or ""),
                "conf_threshold": float(conf_threshold),
                "completed": int(completed),
                "total": int(total),
                "last_scan_at": float(last_scan_at or 0.0),
            }

    def _qa_snapshot(self) -> dict[str, Any]:
        with self._qa_lock:
            snap = dict(self._qa_state)
            snap["items"] = dict(self._qa_state.get("items") or {})
            snap["image_names"] = list(self._qa_state.get("image_names") or [])
            return snap

    def _load_qa_cache(self, slug: str) -> dict[str, Any] | None:
        try:
            path = self._qa_cache_path(slug)
        except Exception:
            return None
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return data if isinstance(data, dict) else None

    def _save_qa_cache(self, slug: str, payload: dict[str, Any]) -> None:
        path = self._qa_cache_path(slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _persist_qa_items(self, slug: str, items: dict[str, Any], *, log_line: str = "") -> None:
        """Write updated QA detection items to memory, scrap_qa.json, and refresh the UI."""
        snap = self._qa_snapshot()
        self._replace_qa_state(
            slug,
            running=bool(snap.get("running")),
            message=str(snap.get("message") or ""),
            error=str(snap.get("error") or ""),
            items=dict(items),
            image_names=list(snap.get("image_names") or []),
            model_path=str(snap.get("model_path") or _QA_DEFAULT_MODEL),
            resolved_model_path=str(snap.get("resolved_model_path") or ""),
            conf_threshold=float(snap.get("conf_threshold") or _QA_DEFAULT_CONF),
            completed=int(snap.get("completed") or 0),
            total=int(snap.get("total") or 0),
            last_scan_at=float(snap.get("last_scan_at") or 0.0),
        )
        cache = self._load_qa_cache(slug) or {}
        cache["items"] = dict(items)
        cache["image_names"] = list(snap.get("image_names") or [])
        cache["model_path"] = str(snap.get("model_path") or cache.get("model_path") or _QA_DEFAULT_MODEL)
        cache["resolved_model_path"] = str(
            snap.get("resolved_model_path") or cache.get("resolved_model_path") or ""
        )
        cache["conf_threshold"] = float(
            snap.get("conf_threshold") or cache.get("conf_threshold") or _QA_DEFAULT_CONF
        )
        cache["generated_at"] = time.time()
        self._save_qa_cache(slug, cache)
        if log_line:
            try:
                from mlops.scrap.jobs import JobStore  # noqa: PLC0415

                JobStore.append_log(slug, log_line)
            except Exception:
                pass
        self._reload_canvas_for_current_image()

    def _qa_cache_is_current(
        self,
        payload: dict[str, Any],
        images: list[Path],
        *,
        model_path: str,
        conf_threshold: float,
    ) -> bool:
        try:
            resolved = str(_resolve_qa_model_reference(model_path))
        except Exception:
            return False
        cached_names = list(payload.get("image_names") or [])
        cached_resolved = str(payload.get("resolved_model_path") or "")
        cached_conf = float(payload.get("conf_threshold") or 0.0)
        return (
            cached_names == _qa_signature(images)
            and cached_resolved == resolved
            and abs(cached_conf - float(conf_threshold)) < 1e-9
        )

    def _ensure_qa_state_for_job(self, slug: str, *, force: bool = False) -> None:
        if not slug:
            return
        images = [p for p, _ in self._label_strip] if self._label_strip else list(self._staged_images)
        model_path = self._current_qa_model_path()
        conf_threshold = self._current_qa_conf()
        if not images:
            self._replace_qa_state(
                slug,
                running=False,
                message="No images to QA.",
                items={},
                image_names=[],
                model_path=model_path,
                conf_threshold=conf_threshold,
                total=0,
            )
            return
        if not force:
            snap = self._qa_snapshot()
            if snap.get("slug") == slug and (snap.get("running") or snap.get("items")):
                if snap.get("running"):
                    return
                if self._qa_cache_is_current(
                    {
                        "image_names": snap.get("image_names") or [],
                        "resolved_model_path": snap.get("resolved_model_path") or "",
                        "conf_threshold": snap.get("conf_threshold") or 0.0,
                    },
                    images,
                    model_path=model_path,
                    conf_threshold=conf_threshold,
                ):
                    return
        payload = None if force else self._load_qa_cache(slug)
        if payload and self._qa_cache_is_current(
            payload,
            images,
            model_path=model_path,
            conf_threshold=conf_threshold,
        ):
            self._replace_qa_state(
                slug,
                running=False,
                message=f"Cached QA ready for {len(images)} staged image(s).",
                items=dict(payload.get("items") or {}),
                image_names=list(payload.get("image_names") or []),
                model_path=str(payload.get("model_path") or model_path),
                resolved_model_path=str(payload.get("resolved_model_path") or ""),
                conf_threshold=float(payload.get("conf_threshold") or conf_threshold),
                completed=len(payload.get("items") or {}),
                total=len(images),
                last_scan_at=float(payload.get("generated_at") or 0.0),
            )
            return
        if self._qa_scan_live(slug):
            self._replace_qa_state(
                slug,
                running=True,
                message="QA scan already running in the background…",
                items={},
                image_names=_qa_signature(images),
                model_path=model_path,
                resolved_model_path=str(_resolve_qa_model_reference(model_path)),
                conf_threshold=conf_threshold,
                completed=0,
                total=len(images),
            )
            return
        if not self._qa_scan_live(slug):
            self._start_qa_scan(slug, images, model_path=model_path, conf_threshold=conf_threshold)

    def _start_qa_scan(
        self,
        slug: str,
        images: list[Path],
        *,
        model_path: str,
        conf_threshold: float,
    ) -> None:
        if self._qa_scan_live(slug):
            return
        image_names = _qa_signature(images)
        resolved_model = str(_resolve_qa_model_reference(model_path))
        if self._active_slug == slug:
            self._replace_qa_state(
                slug,
                running=True,
                message=f"Scanning {len(images)} image(s) with {Path(resolved_model).name}…",
                items={},
                image_names=image_names,
                model_path=model_path,
                resolved_model_path=resolved_model,
                conf_threshold=conf_threshold,
                completed=0,
                total=len(images),
            )
            self.qaStateChanged.emit()

        ev = threading.Event()
        ev.set()
        self._qa_live[slug] = ev

        def _run() -> None:
            items: dict[str, Any] = {}
            try:
                from mlops.scrap.jobs import JobStore  # noqa: PLC0415
                from ..detection_backends import (  # noqa: PLC0415
                    YuNetFaceDetectorBackend,
                    extract_yolo_detections,
                    is_yunet_face_detector_model,
                )

                JobStore.append_log(
                    slug,
                    f"QA scan started with {Path(resolved_model).name} "
                    f"(conf {conf_threshold:.2f}) on {len(images)} staged image(s).",
                )
                resolved_path = Path(resolved_model)
                if not resolved_path.exists():
                    raise FileNotFoundError(f"QA model not found: {resolved_path}")

                device = _resolve_auto_device()
                face_backend: YuNetFaceDetectorBackend | None = None
                yolo_model = None
                supports_to = False
                if is_yunet_face_detector_model(resolved_path):
                    import cv2  # type: ignore

                    face_backend = YuNetFaceDetectorBackend(resolved_path)
                    cv2_mod = cv2
                else:
                    try:
                        from ultralytics import YOLO  # type: ignore
                    except Exception as exc:
                        raise RuntimeError(f"ultralytics unavailable: {exc}") from exc
                    yolo_model = YOLO(str(resolved_path))
                    supports_to = _model_supports_to(str(resolved_path))
                    if supports_to:
                        yolo_model.to(device)
                    cv2_mod = None

                for idx, image_path in enumerate(images, start=1):
                    if not ev.is_set():
                        return
                    try:
                        if face_backend is not None:
                            frame = cv2_mod.imread(str(image_path)) if cv2_mod is not None else None
                            dets = face_backend.predict(frame) if frame is not None else []
                            dets = [
                                det
                                for det in dets
                                if float(det.get("conf") or 0.0) >= float(conf_threshold)
                            ]
                        else:
                            predict_kwargs: dict[str, Any] = {
                                "verbose": False,
                                "conf": float(conf_threshold),
                                "max_det": 50,
                            }
                            if supports_to:
                                predict_kwargs["device"] = device
                            results = yolo_model.predict(source=str(image_path), **predict_kwargs)
                            first = results[0] if results else None
                            if first is not None and getattr(first, "orig_shape", None):
                                img_h, img_w = first.orig_shape[:2]
                            else:
                                img_w = 0
                                img_h = 0
                            dets = extract_yolo_detections(results, int(img_w), int(img_h))
                        items[image_path.name] = _qa_entry_for_detections(dets)
                    except Exception as exc:
                        items[image_path.name] = _qa_entry_for_detections([], error=str(exc))

                    if idx == 1 or idx % 10 == 0 or idx >= len(images):
                        JobStore.append_log(
                            slug,
                            f"QA scan {idx}/{len(images)}: {image_path.name}",
                        )

                    if self._active_slug == slug:
                        with self._qa_lock:
                            self._qa_state["running"] = True
                            self._qa_state["message"] = (
                                f"QA scan {idx}/{len(images)} with {Path(resolved_model).name}…"
                            )
                            self._qa_state["items"] = dict(items)
                            self._qa_state["completed"] = idx
                            self._qa_state["total"] = len(images)
                        self.qaStateChanged.emit()

                generated_at = time.time()
                payload = {
                    "model_path": model_path,
                    "resolved_model_path": resolved_model,
                    "conf_threshold": float(conf_threshold),
                    "image_names": image_names,
                    "generated_at": generated_at,
                    "items": items,
                }
                self._save_qa_cache(slug, payload)
                if self._active_slug == slug:
                    self._replace_qa_state(
                        slug,
                        running=False,
                        message=(
                            f"QA ready — scanned {len(images)} image(s) with "
                            f"{Path(resolved_model).name}."
                        ),
                        items=items,
                        image_names=image_names,
                        model_path=model_path,
                        resolved_model_path=resolved_model,
                        conf_threshold=conf_threshold,
                        completed=len(images),
                        total=len(images),
                        last_scan_at=generated_at,
                    )
                JobStore.append_log(
                    slug,
                    f"QA scan finished: scanned={len(images)} model={Path(resolved_model).name}",
                )
                self.qaScanCompletedForPromotion.emit(slug, len(images))
            except Exception as exc:
                if self._active_slug == slug:
                    self._replace_qa_state(
                        slug,
                        running=False,
                        message="QA scan failed.",
                        error=str(exc),
                        items=items,
                        image_names=image_names,
                        model_path=model_path,
                        resolved_model_path=resolved_model,
                        conf_threshold=conf_threshold,
                        completed=len(items),
                        total=len(images),
                    )
                try:
                    from mlops.scrap.jobs import JobStore  # noqa: PLC0415

                    JobStore.append_log(slug, f"QA ERROR: {exc}")
                except Exception:
                    pass
            finally:
                ev.clear()
                if self._active_slug == slug:
                    self.qaStateChanged.emit()

        threading.Thread(target=_run, name=f"scrap-qa-{slug}", daemon=True).start()

    def _qa_match_indices(self, filter_key: str | None = None) -> list[int]:
        key = str(filter_key or self._qa_filter_key or _QA_ALL_FILTER)
        snap = self._qa_snapshot()
        items = dict(snap.get("items") or {})
        out: list[int] = []
        # Indices refer to the unified label strip (raw + staged); QA scan only covers staged/.
        for idx, (image_path, src) in enumerate(self._label_strip):
            if src != "staged":
                continue
            entry = items.get(image_path.name)
            if key == _QA_ALL_FILTER:
                out.append(idx)
            elif key == _QA_NONE_FILTER:
                if entry is not None and int(entry.get("detection_count") or 0) == 0:
                    out.append(idx)
            elif entry is not None and key in set(entry.get("label_counts") or {}):
                out.append(idx)
        return out

    def _qa_build_group_rows(self) -> list[tuple[str, str]]:
        """Rows for detection groups (same keys as QA group list)."""
        snap = self._qa_snapshot()
        items = dict(snap.get("items") or {})
        total = len(self._label_strip) if self._label_strip else len(self._staged_images)
        image_hits: Counter[str] = Counter()
        box_hits: Counter[str] = Counter()
        none_count = 0
        all_scan_paths = [p for p, _ in self._label_strip] if self._label_strip else self._staged_images
        for image_path in all_scan_paths:
            entry = items.get(image_path.name)
            if entry is None:
                continue
            label_counts = dict(entry.get("label_counts") or {})
            if int(entry.get("detection_count") or 0) <= 0:
                none_count += 1
            for label, count in label_counts.items():
                image_hits[str(label)] += 1
                box_hits[str(label)] += int(count or 0)

        rows: list[tuple[str, str]] = [
            (_QA_ALL_FILTER, f"All images · {total} image(s)"),
        ]
        rows.append((_QA_NONE_FILTER, f"[no detections] · {none_count} image(s)"))
        for label in sorted(image_hits):
            rows.append(
                (
                    label,
                    f"{label} · {image_hits[label]} image(s) · {box_hits[label]} box(es)",
                )
            )
        return rows

    def _qa_staged_paths_for_filter(self, filter_key: str) -> list[Path]:
        paths: list[Path] = []
        for idx in self._qa_match_indices(filter_key):
            if 0 <= idx < len(self._label_strip):
                p, src = self._label_strip[idx]
                if src == "staged":
                    paths.append(p)
        return paths

    def _qa_group_row_text(self, rows: list[tuple[str, str]], key: str) -> str:
        for k, text in rows:
            if k == key:
                return text
        return str(key)

    def _populate_promote_menu(self) -> None:
        self._promote_menu.clear()
        if not self._active_slug:
            act = self._promote_menu.addAction("(Select a scrape job)")
            act.setEnabled(False)
            return
        rows = self._qa_build_group_rows()
        for key, text in rows:
            action = self._promote_menu.addAction(text)
            action.setData(key)
            action.triggered.connect(
                lambda *_, k=key: self._open_promote_for_group(k)
            )

    def _on_promote_dataset_default(self) -> None:
        self._open_promote_for_group(self._qa_filter_key)

    def _open_promote_for_group(self, filter_key: str) -> None:
        slug = self._active_slug
        if not slug:
            return
        if self._qa_scan_live(slug):
            self.errorRaised.emit("Wait for the QA scan to finish before promoting.")
            return
        rows = self._qa_build_group_rows()
        paths = self._qa_staged_paths_for_filter(filter_key)
        if not paths:
            QMessageBox.information(
                self,
                "Promote to dataset",
                "No staged images match this detection group.",
            )
            return
        summary = self._qa_group_row_text(rows, filter_key)
        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            existing = [n for n in reg.list_library_dataset_names() if n != slug]
            source_job_pre = JobStore.load(slug)
        except Exception:
            existing = []
            source_job_pre = None
        if filter_key not in (_QA_ALL_FILTER, _QA_NONE_FILTER):
            default_label = str(filter_key)
        elif source_job_pre and source_job_pre.classes:
            default_label = str(source_job_pre.classes[0])
        else:
            default_label = "object"
        dlg = ScrapePromoteDatasetDialog(
            self,
            group_summary=summary,
            match_count=len(paths),
            source_slug=slug,
            existing_slugs=existing,
            default_promoted_label=default_label,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        target_slug, is_new, promoted_class = dlg.result()

        # Resolve promoted_class_idx from target's current class list (read-only — no files written).
        promoted_class_idx = 0
        try:
            from mlops.pipeline import registry as _reg_prev  # noqa: PLC0415

            _dest_prev = _reg_prev.resolve_library_dataset_path(target_slug)
            _ec_prev, _ = _scrape_existing_yolo_items(_dest_prev)
            if not _ec_prev:
                try:
                    from mlops.scrap.jobs import JobStore as _JS_prev  # noqa: PLC0415

                    if _JS_prev.exists(target_slug):
                        _tj = _JS_prev.load(target_slug)
                        _ec_prev = list(_tj.classes)
                except Exception:
                    pass
            _, promoted_class_idx = _scrape_ensure_class_index(_ec_prev, promoted_class)
        except Exception:
            promoted_class_idx = 0

        _source_labels_prev = dict(getattr(source_job_pre, "labels", None) or {})
        _snap_prev = self._qa_snapshot()
        _qa_items_prev = dict(_snap_prev.get("items") or {})
        preview = compute_transform_preview(
            paths=paths,
            filter_key=filter_key,
            promoted_class=promoted_class,
            promoted_class_idx=promoted_class_idx,
            source_job_labels=_source_labels_prev,
            qa_items=_qa_items_prev,
        )

        from .transform_preview_dialog import TransformPreviewDialog  # noqa: PLC0415

        preview_dlg = TransformPreviewDialog(
            self,
            preview=preview,
            promoted_class=promoted_class,
            promoted_class_idx=promoted_class_idx,
            group_summary=summary,
            filter_key=filter_key,
        )
        if preview_dlg.exec() != QDialog.DialogCode.Accepted:
            return

        self._execute_promote_to_dataset(
            filter_key, target_slug, is_new=is_new, promoted_class=promoted_class
        )

    def _execute_promote_to_dataset(
        self,
        filter_key: str,
        target_slug_raw: str,
        *,
        is_new: bool,
        promoted_class: str,
    ) -> None:
        source_slug = self._active_slug
        if not source_slug:
            return
        promoted = str(promoted_class or "").strip() or "object"
        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            target_slug = reg.sanitize_library_dataset_slug(target_slug_raw)
        except Exception as exc:
            self.errorRaised.emit(f"Invalid dataset name: {exc}")
            return
        if target_slug == source_slug:
            QMessageBox.warning(
                self,
                "Promote to dataset",
                "Choose a different dataset than the current job.",
            )
            return
        paths = self._qa_staged_paths_for_filter(filter_key)
        if not paths:
            return
        source_job = JobStore.load(source_slug)
        try:
            if is_new:
                reg.create_library_dataset_root(target_slug)
            dest_base = reg.resolve_library_dataset_path(target_slug)
        except Exception as exc:
            self.errorRaised.emit(f"Could not prepare target dataset: {exc}")
            return

        target_job = JobStore.load(target_slug) if JobStore.exists(target_slug) else None
        existing_classes, existing_items = _scrape_existing_yolo_items(dest_base)
        if not existing_classes and target_job:
            existing_classes = list(target_job.classes)
        classes, final_idx = _scrape_ensure_class_index(
            existing_classes,
            promoted,
        )
        snap = self._qa_snapshot()
        qa_items = dict(snap.get("items") or {})
        emitted_items = list(existing_items)
        promoted_labels: dict[str, list[list[float]]] = {}
        promoted_images = 0
        hand_label_images = 0
        qa_label_images = 0
        full_frame_images = 0

        from mlops.scrap.emit import LabeledItem, emit_yolo_dataset  # noqa: PLC0415

        for src in paths:
            if not src.is_file():
                continue
            name = src.name
            new_boxes: list[list[float]] = []
            raw = source_job.labels.get(name) if source_job else None
            if raw:
                for b in raw:
                    if len(b) < 5:
                        continue
                    cx, cy, w, h = float(b[1]), float(b[2]), float(b[3]), float(b[4])
                    new_boxes.append([float(final_idx), cx, cy, w, h])
                if new_boxes:
                    hand_label_images += 1
            else:
                entry = qa_items.get(name)
                dets = list(entry.get("detections") or []) if isinstance(entry, dict) else []
                if filter_key not in (_QA_ALL_FILTER, _QA_NONE_FILTER):
                    dets = [d for d in dets if str(d.get("label") or "").strip() == str(filter_key)]
                elif filter_key == _QA_NONE_FILTER:
                    dets = []
                for det in dets:
                    box = _scrape_detection_to_yolo_box(dict(det), final_idx)
                    if box is not None:
                        new_boxes.append(box)
                if new_boxes:
                    qa_label_images += 1
            if not new_boxes:
                new_boxes = [_scrape_full_frame_box(final_idx)]
                full_frame_images += 1
            promoted_labels[name] = [list(b) for b in new_boxes]
            emitted_items.append(
                LabeledItem(
                    image_path=src,
                    boxes=tuple((int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])) for b in new_boxes),
                )
            )
            promoted_images += 1

        if len(emitted_items) < 2:
            QMessageBox.warning(
                self,
                "Promote to dataset",
                "Need at least 2 labeled images to create a train/val YOLO dataset.",
            )
            return

        try:
            ds_root = emit_yolo_dataset(
                slug=target_slug,
                classes=classes,
                items=emitted_items,
                val_frac=0.2,
            )
        except Exception as exc:
            self.errorRaised.emit(f"Promote emit failed: {exc}")
            return

        if target_job:
            try:
                # If an in-memory/UI copy still has the old target job loaded,
                # leave no stale label map behind.
                JobStore.update(target_slug, labels=promoted_labels, classes=classes, state="emitted")
            except Exception as exc:
                self.errorRaised.emit(f"Dataset emitted, but legacy scraper metadata cleanup failed: {exc}")
        # The promoted target is now a database dataset, not a raw scraper job.
        # Clean old intermediate state from previous buggy promotions.
        shutil.rmtree(dest_base / "staged", ignore_errors=True)
        for aux in (dest_base / "scrap.json", dest_base / _QA_CACHE_NAME):
            try:
                aux.unlink(missing_ok=True)
            except Exception:
                pass
        if source_job:
            try:
                JobStore.append_log(
                    source_slug,
                    f"Promoted {promoted_images} image(s) to final YOLO dataset {target_slug} "
                    f"(group {filter_key!r}, class {promoted!r}).",
                )
            except Exception:
                pass

        msg = (
            f"Promoted {promoted_images} image(s) into final YOLO layout:\n{ds_root}\n\n"
            f"Target class: `{promoted}` (index {final_idx}).\n"
            "Created/updated images/train, images/val, labels/train, labels/val, classes.txt, and data.yaml."
        )
        details = []
        if hand_label_images:
            details.append(f"{hand_label_images} from hand-drawn boxes")
        if qa_label_images:
            details.append(f"{qa_label_images} from QA detector boxes")
        if full_frame_images:
            details.append(f"{full_frame_images} full-image fallback labels")
        if details:
            msg += "\n\nLabels: " + ", ".join(details) + "."
        QMessageBox.information(self, "Promote to dataset", msg)
        self._refresh_job_list()
        if self._active_slug == source_slug:
            self._refresh_label_page()

    def _scrape_worker_live(self, slug: str | None) -> bool:
        if not slug:
            return False
        ev = self._scrape_live.get(slug)
        return bool(ev and ev.is_set())

    def _refresh_activity_page(self) -> None:
        slug = self._active_slug
        if not slug:
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
        except Exception:
            job = None
        if job is None:
            self._activity_status.setText(f"[{slug}] job state not found")
            self._scrape_trace.clear()
            self._activity_progress.setValue(0)
            self._scrape_progress_circle.set_progress(0, headline="0%", detail="raw 0 / 0")
            return

        worker_live = self._scrape_worker_live(slug)
        raw_cnt = self._count_raw_files(slug)
        target_count = int(job.target_count or 0)
        remaining = max(0, target_count - raw_cnt)
        q_avail = bool((job.last_scrape_query or "").strip())
        raw_pct = 0 if target_count <= 0 else min(100, int(round((raw_cnt / max(1, target_count)) * 100)))

        if job.state == "scraping" and not worker_live:
            stale_note = (
                " (no scrape thread in this session — Continue downloads or Restart if stuck)"
            )
        else:
            stale_note = ""
        status_line = (
            f"State: {job.state}{stale_note} — {job.message or '…'} "
            f"· raw {raw_cnt} / target {target_count}"
        )
        self._activity_status.setText(status_line)
        self._scrape_progress_circle.set_progress(
            raw_pct,
            headline=f"{raw_pct}%",
            detail=f"raw {raw_cnt} / {target_count}",
        )

        total = len(self._staged_images)
        labeled_count = sum(1 for p in self._staged_images if p.name in job.labels)
        pct = int(labeled_count / max(1, total) * 100)
        self._activity_progress.setValue(pct)

        ctrl_scraping = job.state == "scraping" and worker_live
        self._btn_pause_play.setEnabled(ctrl_scraping)
        if job.scrape_paused and ctrl_scraping:
            self._btn_pause_play.setText("Resume scraping")
        else:
            self._btn_pause_play.setText("Pause scraping")

        can_resume = remaining > 0 and (q_avail or bool(job.topic))
        idle = not worker_live
        self._btn_continue_scrape.setEnabled(idle and can_resume and job.state != "emitted")
        self._btn_restart_scrape.setEnabled(idle and (q_avail or bool(job.topic)) and job.state != "emitted")
        self._btn_add_more_images.setEnabled(idle and (q_avail or bool(job.topic)) and job.state != "emitted")
        self._refresh_target_count_controls(target_count)

        log_text = "\n".join(job.processing_log)
        self._scrape_trace.setPlainText(log_text)
        if job.state in ("pending", "scraping") and not log_text:
            self._scrape_trace.setPlaceholderText(
                "Waiting for pipeline events (browser, scroll, downloads)…"
            )
        elif not log_text:
            self._scrape_trace.setPlaceholderText(
                "No log lines recorded for this step yet."
            )

    def _on_toggle_pause_play(self) -> None:
        slug = self._active_slug
        if not slug or not self._scrape_worker_live(slug):
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
            if job is None:
                return
            JobStore.update(slug, scrape_paused=not job.scrape_paused)
        except Exception as exc:
            self.errorRaised.emit(f"Pause toggle failed: {exc}")
            return
        self._refresh_activity_page()

    def _active_scrape_query(self, job) -> str:
        return (job.last_scrape_query or "").strip() or (job.topic or "").strip()

    def _on_continue_scrape(self) -> None:
        slug = self._active_slug
        if not slug or self._scrape_worker_live(slug):
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
            if job is None:
                return
            query = self._active_scrape_query(job)
            if not query:
                self.errorRaised.emit("No search query on file — restart from a job that ran a scrape.")
                return
            self._start_scrape_thread(slug, query, clear_raw=False)
        except Exception as exc:
            self.errorRaised.emit(f"Continue failed: {exc}")
            return
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        self._refresh_job_list()
        self._refresh_activity_page()

    def _on_restart_scrape_downloads(self) -> None:
        slug = self._active_slug
        if not slug or self._scrape_worker_live(slug):
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
            if job is None:
                return
            query = self._active_scrape_query(job)
            if not query:
                self.errorRaised.emit("No search query on file — recreate the job.")
                return
            self._start_scrape_thread(slug, query, clear_raw=True)
        except Exception as exc:
            self.errorRaised.emit(f"Restart failed: {exc}")
            return
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        self._refresh_job_list()
        self._refresh_activity_page()

    def _on_add_more_images(self) -> None:
        slug = self._active_slug
        if not slug or self._scrape_worker_live(slug):
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
            if job is None:
                return
            query = self._active_scrape_query(job)
            if not query:
                self.errorRaised.emit("No search query on file — recreate the job.")
                return
            raw_now = self._count_raw_files(slug)
            new_target = self._current_target_count_value()
            if new_target <= raw_now:
                self.errorRaised.emit(
                    f"Choose a target above the current raw total ({raw_now}) to add more images."
                )
                return
            old_target = int(job.target_count or 0)
            if new_target != old_target:
                JobStore.update(
                    slug,
                    target_count=new_target,
                    message=f"target raised from {old_target} to {new_target}; downloading more",
                )
                JobStore.append_log(
                    slug,
                    f"Target raw count updated from {old_target} to {new_target} for add-more run.",
                )
            self._start_scrape_thread(slug, query, clear_raw=False)
        except Exception as exc:
            self.errorRaised.emit(f"Add more images failed: {exc}")
            return
        if not self._poll_timer.isActive():
            self._poll_timer.start()
        self._refresh_job_list()
        self._refresh_activity_page()

    def _on_qa_state_changed(self) -> None:
        if not self._active_slug:
            return
        self._refresh_label_page()

    def _refresh_qa_panel(self) -> None:
        slug = self._active_slug
        qa_circle = getattr(self, "_qa_progress_circle", None)
        if not slug:
            self._qa_status.clear()
            self._qa_growth_status.clear()
            self._qa_current_summary.clear()
            self._qa_group_list.clear()
            self._qa_detection_list.clear()
            if qa_circle is not None:
                qa_circle.set_progress(0, headline="0%", detail="qa 0 / 0")
            ec = getattr(self, "_qa_edit_from_combo", None)
            if ec is not None:
                ec.blockSignals(True)
                ec.clear()
                ec.blockSignals(False)
            te = getattr(self, "_qa_edit_to_edit", None)
            if te is not None:
                te.clear()
            for attr in (
                "_qa_remove_label_all_btn",
                "_qa_relabel_all_btn",
                "_qa_remove_one_btn",
                "_qa_relabel_selected_btn",
            ):
                b = getattr(self, attr, None)
                if b is not None:
                    b.setEnabled(False)
            pr = getattr(self, "_promote_dataset_btn", None)
            if pr is not None:
                pr.setEnabled(False)
            return

        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
        except Exception:
            job = None

        snap = self._qa_snapshot()
        items = dict(snap.get("items") or {})
        total = len(self._label_strip) if self._label_strip else len(self._staged_images)
        completed = int(snap.get("completed") or 0)
        running = bool(snap.get("running"))
        error = str(snap.get("error") or "").strip()
        conf_threshold = float(snap.get("conf_threshold") or self._current_qa_conf())
        model_label = Path(
            str(snap.get("resolved_model_path") or snap.get("model_path") or _QA_DEFAULT_MODEL)
        ).name
        scanned = min(max(0, total), max(len(items), completed))
        pending = max(0, total - scanned)
        qa_pct = int(round((scanned / max(1, total)) * 100)) if total > 0 else 0
        raw_cnt = self._count_raw_files(slug)
        target_count = int(job.target_count or 0) if job is not None else 0
        remaining = max(0, target_count - raw_cnt)
        self._refresh_target_count_controls(target_count)

        if qa_circle is not None:
            qa_circle.set_progress(
                qa_pct,
                headline=f"{qa_pct}%",
                detail=f"qa {scanned} / {total}",
            )

        if error:
            self._qa_status.setText(f"QA error — {error}")
        elif running:
            self._qa_status.setText(
                f"Scanning {completed}/{max(1, total)} with {model_label} @ conf {conf_threshold:.2f}."
            )
        elif total <= 0:
            self._qa_status.setText("No images to QA.")
        elif items:
            ready_bits = [f"{scanned} image(s) scanned with {model_label}"]
            if pending:
                ready_bits.append(f"{pending} pending")
            self._qa_status.setText("QA ready — " + " · ".join(ready_bits))
        else:
            self._qa_status.setText("No QA scan yet.")

        scrape_live = self._scrape_worker_live(slug)
        query_ready = bool(job is not None and self._active_scrape_query(job))
        if job is None:
            self._qa_growth_status.setText("No scrape job loaded.")
        elif job.state == "emitted":
            self._qa_growth_status.setText(
                f"Raw {raw_cnt} / target {target_count} — emitted jobs are locked from more downloads."
            )
        elif scrape_live:
            self._qa_growth_status.setText(
                f"Raw {raw_cnt} / target {target_count} — scrape run in progress."
            )
        elif remaining > 0:
            self._qa_growth_status.setText(
                f"Raw {raw_cnt} / target {target_count} — {remaining} more can be fetched before raising the target."
            )
        else:
            self._qa_growth_status.setText(
                f"Raw {raw_cnt} / target {target_count} — raise the target above {raw_cnt} to add more."
            )
        self._qa_target_count_spin.setEnabled(
            job is not None and not scrape_live and job.state != "emitted"
        )
        self._qa_add_more_btn.setEnabled(
            job is not None and not scrape_live and query_ready and job.state != "emitted"
        )

        rows = self._qa_build_group_rows()

        current_key = self._qa_filter_key if any(key == self._qa_filter_key for key, _ in rows) else _QA_ALL_FILTER
        self._qa_filter_key = current_key

        self._qa_group_list.blockSignals(True)
        self._qa_group_list.clear()
        selected_row = 0
        for idx, (key, text) in enumerate(rows):
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, key)
            self._qa_group_list.addItem(item)
            if key == current_key:
                selected_row = idx
        if rows:
            self._qa_group_list.setCurrentRow(selected_row)
        self._qa_group_list.blockSignals(False)

        current_entry = None
        current_name = ""
        cur_path = self._current_label_path()
        if cur_path is not None:
            current_name = cur_path.name
            current_entry = items.get(current_name)

        if current_entry is None:
            if running and current_name:
                self._qa_current_summary.setText(f"Current image QA pending — {current_name}")
            elif current_name:
                self._qa_current_summary.setText(f"No QA data for {current_name}")
            else:
                self._qa_current_summary.setText("No image selected.")
        elif current_entry.get("error"):
            self._qa_current_summary.setText(
                f"Current image QA error — {current_entry.get('error')}"
            )
        else:
            det_count = int(current_entry.get("detection_count") or 0)
            label_count = len(dict(current_entry.get("label_counts") or {}))
            if det_count <= 0:
                self._qa_current_summary.setText("Current image: no detections.")
            else:
                self._qa_current_summary.setText(
                    f"Current image: {det_count} detection(s) across {label_count} label(s)."
                )

        detections = list((current_entry or {}).get("detections") or [])
        if self._qa_highlight_det_idx >= len(detections):
            self._qa_highlight_det_idx = -1

        self._qa_detection_list.blockSignals(True)
        self._qa_detection_list.clear()
        if current_entry and current_entry.get("error"):
            self._qa_detection_list.addItem(f"[scan error] {current_entry.get('error')}")
        elif current_entry is None:
            if current_name:
                self._qa_detection_list.addItem("(QA pending for this image)")
        elif not detections:
            self._qa_detection_list.addItem("(no detections)")
        else:
            for idx, det in enumerate(detections, start=1):
                label = str(det.get("label") or "unknown")
                conf = float(det.get("conf") or 0.0)
                item = QListWidgetItem(f"{idx}. {label} · {conf:.2f}")
                item.setData(Qt.ItemDataRole.UserRole, idx - 1)
                self._qa_detection_list.addItem(item)
            if 0 <= self._qa_highlight_det_idx < len(detections):
                self._qa_detection_list.setCurrentRow(self._qa_highlight_det_idx)
        self._qa_detection_list.blockSignals(False)

        match_count = len(self._qa_match_indices())
        self._qa_prev_btn.setEnabled(match_count > 0)
        self._qa_next_btn.setEnabled(match_count > 0)
        self._qa_delete_current_btn.setEnabled(bool(self._label_strip) and not running)
        can_delete_matching = self._qa_filter_key != _QA_ALL_FILTER and match_count > 0
        self._qa_delete_matching_btn.setEnabled(can_delete_matching and not running)
        if self._qa_filter_key == _QA_NONE_FILTER:
            self._qa_delete_matching_btn.setText("Delete no-det images")
        elif self._qa_filter_key not in {"", _QA_ALL_FILTER}:
            self._qa_delete_matching_btn.setText(f"Delete '{self._qa_filter_key}' images")
        else:
            self._qa_delete_matching_btn.setText("Delete matching")

        edit_combo = getattr(self, "_qa_edit_from_combo", None)
        if edit_combo is not None:
            prior = edit_combo.currentText()
            edit_combo.blockSignals(True)
            edit_combo.clear()
            for key, _text in rows:
                if key in (_QA_ALL_FILTER, _QA_NONE_FILTER):
                    continue
                edit_combo.addItem(key)
            if prior.strip():
                edit_combo.setEditText(prior)
            edit_combo.blockSignals(False)

        can_bulk = bool(items) and not running
        row = self._qa_detection_list.currentRow()
        sel_item = self._qa_detection_list.item(row) if row >= 0 else None
        has_det_idx = sel_item is not None and sel_item.data(Qt.ItemDataRole.UserRole) is not None
        entry_ok = bool(
            current_entry
            and not current_entry.get("error")
            and detections
        )
        can_one = can_bulk and entry_ok and has_det_idx
        rb = getattr(self, "_qa_remove_label_all_btn", None)
        if rb is not None:
            rb.setEnabled(can_bulk)
        rb = getattr(self, "_qa_relabel_all_btn", None)
        if rb is not None:
            rb.setEnabled(can_bulk)
        rb = getattr(self, "_qa_remove_one_btn", None)
        if rb is not None:
            rb.setEnabled(can_one)
        rb = getattr(self, "_qa_relabel_selected_btn", None)
        if rb is not None:
            rb.setEnabled(can_one)

        self._qa_scan_btn.setText("Scanning…" if running else "Run QA")
        self._qa_scan_btn.setEnabled(not running and bool(self._label_strip or self._staged_images))

        pr = getattr(self, "_promote_dataset_btn", None)
        if pr is not None:
            pr.setEnabled(bool(slug) and bool(self._staged_images) and not running)

    def _on_qa_remove_label_all(self) -> None:
        slug = self._active_slug
        if not slug or self._qa_scan_live(slug):
            return
        from_labels = _qa_parse_from_labels(self._qa_edit_from_combo.currentText())
        if not from_labels:
            return
        from_set = frozenset(from_labels)
        snap = self._qa_snapshot()
        items = dict(snap.get("items") or {})
        removed_boxes = 0
        for entry in items.values():
            if not isinstance(entry, dict):
                continue
            dets = list(entry.get("detections") or [])
            removed_boxes += sum(
                1 for d in dets if str(d.get("label") or "").strip() in from_set
            )
        if removed_boxes <= 0:
            listed = ", ".join(from_labels)
            QMessageBox.information(
                self,
                "Remove label",
                f"No boxes with any of these labels in the current QA results: {listed}",
            )
            return
        listed = ", ".join(from_labels)
        confirm = QMessageBox.question(
            self,
            "Remove label everywhere",
            f"Remove {removed_boxes} box(es) with label(s) [{listed}] across all scanned images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        new_items: dict[str, Any] = {}
        for name, entry in items.items():
            if not isinstance(entry, dict):
                new_items[name] = entry
                continue
            e = dict(entry)
            new_dets = [
                d
                for d in list(e.get("detections") or [])
                if str(d.get("label") or "").strip() not in from_set
            ]
            e["detections"] = new_dets
            _qa_entry_refresh_metadata(e)
            new_items[name] = e
        self._qa_highlight_det_idx = -1
        self._persist_qa_items(
            slug,
            new_items,
            log_line=f"QA editor: removed labels [{listed}] ({removed_boxes} boxes)",
        )

    def _on_qa_relabel_all(self) -> None:
        slug = self._active_slug
        if not slug or self._qa_scan_live(slug):
            return
        from_labels = _qa_parse_from_labels(self._qa_edit_from_combo.currentText())
        to_label = str(self._qa_edit_to_edit.text() or "").strip()
        if not from_labels or not to_label:
            QMessageBox.information(
                self,
                "Relabel",
                "Set From label (one or more comma-separated) and To label for a bulk relabel.",
            )
            return
        from_set = frozenset(from_labels)
        if len(from_set) == 1 and to_label in from_set:
            return
        snap = self._qa_snapshot()
        items = dict(snap.get("items") or {})
        changed = 0
        for entry in items.values():
            if not isinstance(entry, dict):
                continue
            for d in list(entry.get("detections") or []):
                lbl = str(d.get("label") or "").strip()
                if lbl in from_set and lbl != to_label:
                    changed += 1
        if changed <= 0:
            listed = ", ".join(from_labels)
            QMessageBox.information(
                self,
                "Relabel",
                f"No boxes to relabel (none of [{listed}] found, or all already '{to_label}').",
            )
            return
        listed = ", ".join(from_labels)
        confirm = QMessageBox.question(
            self,
            "Relabel everywhere",
            f"Relabel {changed} box(es) from [{listed}] to '{to_label}' across all images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        new_items: dict[str, Any] = {}
        for name, entry in items.items():
            if not isinstance(entry, dict):
                new_items[name] = entry
                continue
            e = dict(entry)
            new_dets = []
            for d in list(e.get("detections") or []):
                dc = dict(d)
                lbl = str(dc.get("label") or "").strip()
                if lbl in from_set:
                    dc["label"] = to_label
                new_dets.append(dc)
            e["detections"] = new_dets
            _qa_entry_refresh_metadata(e)
            new_items[name] = e
        self._qa_highlight_det_idx = -1
        self._persist_qa_items(
            slug,
            new_items,
            log_line=f"QA editor: relabeled [{listed}] -> '{to_label}' ({changed} boxes)",
        )

    def _on_qa_remove_selected_detection(self) -> None:
        slug = self._active_slug
        if not slug or self._qa_scan_live(slug):
            return
        row = self._qa_detection_list.currentRow()
        lw_item = self._qa_detection_list.item(row)
        if lw_item is None:
            return
        raw_idx = lw_item.data(Qt.ItemDataRole.UserRole)
        if raw_idx is None:
            return
        det_idx = int(raw_idx)
        cur = self._current_label_path()
        if cur is None:
            return
        snap = self._qa_snapshot()
        items = dict(snap.get("items") or {})
        entry = items.get(cur.name)
        if not isinstance(entry, dict) or entry.get("error"):
            return
        dets = list(entry.get("detections") or [])
        if det_idx < 0 or det_idx >= len(dets):
            return
        dets.pop(det_idx)
        new_entry = dict(entry)
        new_entry["detections"] = dets
        _qa_entry_refresh_metadata(new_entry)
        items[cur.name] = new_entry
        self._qa_highlight_det_idx = -1
        self._persist_qa_items(
            slug,
            items,
            log_line=f"QA editor: removed one box from {cur.name}",
        )

    def _on_qa_relabel_selected_detection(self) -> None:
        slug = self._active_slug
        if not slug or self._qa_scan_live(slug):
            return
        row = self._qa_detection_list.currentRow()
        lw_item = self._qa_detection_list.item(row)
        if lw_item is None:
            return
        raw_idx = lw_item.data(Qt.ItemDataRole.UserRole)
        if raw_idx is None:
            return
        det_idx = int(raw_idx)
        cur = self._current_label_path()
        if cur is None:
            return
        snap = self._qa_snapshot()
        items = dict(snap.get("items") or {})
        entry = items.get(cur.name)
        if not isinstance(entry, dict) or entry.get("error"):
            return
        dets = list(entry.get("detections") or [])
        if det_idx < 0 or det_idx >= len(dets):
            return
        old = str(dets[det_idx].get("label") or "")
        text, ok = QInputDialog.getText(
            self,
            "Relabel box",
            "New label:",
            text=old,
        )
        if not ok:
            return
        new_l = str(text or "").strip()
        if not new_l:
            return
        new_entry = dict(entry)
        new_dets = list(new_entry.get("detections") or [])
        dc = dict(new_dets[det_idx])
        dc["label"] = new_l
        new_dets[det_idx] = dc
        new_entry["detections"] = new_dets
        _qa_entry_refresh_metadata(new_entry, resort=False)
        items[cur.name] = new_entry
        self._persist_qa_items(
            slug,
            items,
            log_line=f"QA editor: relabeled one box on {cur.name} ({old!r} -> {new_l!r})",
        )

    def _on_qa_group_changed(self, current: Optional[QListWidgetItem], _prev) -> None:
        if current is None:
            return
        self._qa_filter_key = str(current.data(Qt.ItemDataRole.UserRole) or _QA_ALL_FILTER)
        self._qa_highlight_det_idx = -1
        matches = self._qa_match_indices()
        if matches and self._current_idx not in matches:
            self._current_idx = matches[0]
            self._refresh_label_page()
            return
        self._refresh_qa_panel()
        self._refresh_transform_pane()

    def _on_qa_detection_selected(self, row: int) -> None:
        self._qa_highlight_det_idx = int(row) if row >= 0 else -1
        self._refresh_label_page()

    def _on_qa_prev_match(self) -> None:
        matches = self._qa_match_indices()
        if not matches:
            return
        self._qa_highlight_det_idx = -1
        if self._current_idx not in matches:
            self._current_idx = matches[-1]
        else:
            pos = matches.index(self._current_idx)
            self._current_idx = matches[(pos - 1) % len(matches)]
        self._refresh_label_page()

    def _on_qa_next_match(self) -> None:
        matches = self._qa_match_indices()
        if not matches:
            return
        self._qa_highlight_det_idx = -1
        if self._current_idx not in matches:
            self._current_idx = matches[0]
        else:
            pos = matches.index(self._current_idx)
            self._current_idx = matches[(pos + 1) % len(matches)]
        self._refresh_label_page()

    def _on_run_qa_scan(self) -> None:
        slug = self._active_slug
        if not slug or not (self._label_strip or self._staged_images):
            return
        self._qa_highlight_det_idx = -1
        self._ensure_qa_state_for_job(slug, force=True)
        self._refresh_label_page()

    def _delete_staged_images(self, image_names: list[str], *, reason: str) -> int:
        slug = self._active_slug
        if not slug:
            return 0
        targets = {str(name) for name in image_names if str(name)}
        if not targets:
            return 0
        removed = 0
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415

            job = JobStore.load(slug)
            if job is None:
                return 0
            new_labels = dict(job.labels)
            for name in targets:
                new_labels.pop(name, None)
            for image_path in list(self._staged_images):
                if image_path.name not in targets:
                    continue
                try:
                    image_path.unlink(missing_ok=True)
                    removed += 1
                except Exception as exc:
                    self.errorRaised.emit(f"Delete failed for {image_path.name}: {exc}")
            self._staged_images = [p for p in self._staged_images if p.name not in targets]
            new_total = len(self._staged_images)
            JobStore.update(
                slug,
                staged_count=new_total,
                labels=new_labels,
                message=f"{reason}; staged now {new_total}",
            )
            cache = self._load_qa_cache(slug) or {}
            cache_items = dict(cache.get("items") or {})
            for name in targets:
                cache_items.pop(name, None)
            cache["items"] = cache_items
            cache["image_names"] = _qa_signature(self._staged_images)
            cache["generated_at"] = time.time()
            if cache:
                self._save_qa_cache(slug, cache)
            snap = self._qa_snapshot()
            qa_items = dict(snap.get("items") or {})
            for name in targets:
                qa_items.pop(name, None)
            self._replace_qa_state(
                slug,
                running=self._qa_scan_live(slug),
                message=f"{reason}; removed {removed} image(s).",
                error=str(snap.get("error") or ""),
                items=qa_items,
                image_names=_qa_signature(self._staged_images),
                model_path=str(snap.get("model_path") or self._current_qa_model_path()),
                resolved_model_path=str(snap.get("resolved_model_path") or ""),
                conf_threshold=float(snap.get("conf_threshold") or self._current_qa_conf()),
                completed=min(int(snap.get("completed") or 0), len(self._staged_images)),
                total=len(self._staged_images),
                last_scan_at=float(snap.get("last_scan_at") or 0.0),
            )
            JobStore.append_log(slug, f"QA delete: removed {removed} staged image(s) ({reason})")
        except Exception as exc:
            self.errorRaised.emit(f"Delete staged images failed: {exc}")
            return removed
        self._qa_highlight_det_idx = -1
        self._refresh_activity_page()
        self._refresh_label_page()
        self._refresh_emit_page()
        self._refresh_job_list()
        return removed

    def _on_delete_current_staged_image(self) -> None:
        path = self._current_label_path()
        slug = self._active_slug
        if path is None or not slug:
            return
        image_name = path.name
        if path.parent.name.lower() == "raw":
            confirm = QMessageBox.warning(
                self,
                "Delete Raw Image",
                f"Permanently delete '{image_name}' from raw/?\n\nThis cannot be undone.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            if confirm != QMessageBox.StandardButton.Yes:
                return
            try:
                path.unlink(missing_ok=True)
            except Exception as exc:
                self.errorRaised.emit(f"Delete raw file failed: {exc}")
                return
            try:
                from mlops.scrap.jobs import JobStore  # noqa: PLC0415

                job = JobStore.load(slug)
                if job is not None and image_name in job.labels:
                    new_labels = dict(job.labels)
                    new_labels.pop(image_name, None)
                    JobStore.update(slug, labels=new_labels)
                JobStore.append_log(slug, f"Deleted raw image {image_name}")
            except Exception as exc:
                self.errorRaised.emit(f"Update job after raw delete failed: {exc}")
                return
            self._qa_highlight_det_idx = -1
            self._refresh_label_page()
            self._refresh_activity_page()
            return

        if path.parent.name.lower() != "staged":
            return
        confirm = QMessageBox.warning(
            self,
            "Delete Current Staged Image",
            f"Delete '{image_name}' from staged/?\n\nThis removes it from scrape QA and future dataset emit.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        self._delete_staged_images([image_name], reason=f"deleted staged image {image_name}")

    def _on_delete_matching_staged_images(self) -> None:
        if self._qa_filter_key == _QA_ALL_FILTER:
            return
        match_names = [
            self._label_strip[idx][0].name
            for idx in self._qa_match_indices()
            if 0 <= idx < len(self._label_strip)
        ]
        if not match_names:
            return
        title = "Delete Matching Staged Images"
        if self._qa_filter_key == _QA_NONE_FILTER:
            prompt = (
                f"Delete {len(match_names)} staged image(s) with no detections?\n\n"
                "This cannot be undone."
            )
        else:
            prompt = (
                f"Delete {len(match_names)} staged image(s) matched by '{self._qa_filter_key}'?\n\n"
                "This cannot be undone."
            )
        confirm = QMessageBox.warning(
            self,
            title,
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        reason = (
            "deleted no-detection staged images"
            if self._qa_filter_key == _QA_NONE_FILTER
            else f"deleted staged images matched by {self._qa_filter_key}"
        )
        self._delete_staged_images(match_names, reason=reason)

    def _refresh_label_page(self) -> None:
        slug = self._active_slug
        if not slug:
            for gallery, _icon_size, _grid_size in self._gallery_views:
                gallery.clear()
            for summary in self._gallery_summaries:
                summary.clear()
            return

        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
        except Exception:
            job = None

        if job is None:
            self._img_counter.clear()
            self._class_list.blockSignals(True)
            self._class_list.clear()
            self._class_list.blockSignals(False)
            self._canvas.reset()
            self._canvas.class_names = []
            self._canvas.class_picker = None
            self._qa_group_list.clear()
            self._qa_detection_list.clear()
            self._qa_status.clear()
            self._qa_current_summary.clear()
            for gallery, _icon_size, _grid_size in self._gallery_views:
                gallery.clear()
            for summary in self._gallery_summaries:
                summary.clear()
            lt = getattr(self, "_label_card_title", None)
            if lt is not None:
                lt.setText("Select a job")
            lm = getattr(self, "_label_card_meta", None)
            if lm is not None:
                lm.clear()
            self._label_strip = []
            self._current_idx = 0
            return

        self._staged_images = self._get_staged_images(slug)
        self._label_strip = self._build_label_strip(slug, staged_paths=self._staged_images)
        if self._label_strip:
            self._current_idx = max(0, min(self._current_idx, len(self._label_strip) - 1))
        else:
            self._current_idx = 0

        prior_row = self._class_list.currentRow()
        self._class_list.blockSignals(True)
        self._class_list.clear()
        for cls_name in job.classes:
            self._class_list.addItem(cls_name)
        if job.classes:
            nr = max(0, min(prior_row if prior_row >= 0 else 0, len(job.classes) - 1))
            self._class_list.setCurrentRow(nr)
            self._canvas.active_class_idx = nr
        self._class_list.blockSignals(False)

        self._canvas.class_names = list(job.classes)
        self._canvas.class_picker = self._pick_class_for_new_box

        self._populate_staged_gallery()
        self._reload_canvas_for_current_image()

    def _on_prev(self) -> None:
        if not self._label_strip:
            return
        self._qa_highlight_det_idx = -1
        self._current_idx = (self._current_idx - 1) % len(self._label_strip)
        self._refresh_label_page()

    def _on_next(self) -> None:
        if not self._label_strip:
            return
        self._qa_highlight_det_idx = -1
        self._current_idx = (self._current_idx + 1) % len(self._label_strip)
        self._refresh_label_page()

    def _on_next_unlabeled(self) -> None:
        slug = self._active_slug
        if not slug or not self._label_strip:
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
        except Exception:
            return
        if job is None:
            return
        total = len(self._label_strip)
        for offset in range(1, total + 1):
            j = (self._current_idx + offset) % total
            path = self._label_strip[j][0]
            if path.name not in job.labels:
                self._qa_highlight_det_idx = -1
                self._current_idx = j
                break
        self._refresh_label_page()

    def _on_clear(self) -> None:
        slug = self._active_slug
        current_img = self._current_label_path()
        if not slug or current_img is None:
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
            if job is None:
                return
            new_labels = dict(job.labels)
            new_labels.pop(current_img.name, None)
            JobStore.update(slug, labels=new_labels)
        except Exception as exc:
            self.errorRaised.emit(f"Clear failed: {exc}")
            return
        self._canvas.clear_boxes()
        self._refresh_label_page()
        self._refresh_activity_page()

    def _on_class_selected(self, row: int) -> None:
        self._canvas.active_class_idx = max(0, row)

    def _on_add_class(self) -> None:
        slug = self._active_slug
        if not slug:
            return
        name = self._class_input.text().strip()
        if not name:
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
            if job is None:
                return
            if name not in job.classes:
                JobStore.update(slug, classes=job.classes + [name])
        except Exception as exc:
            self.errorRaised.emit(f"Add class failed: {exc}")
            return
        self._class_input.clear()
        self._refresh_label_page()

    def _on_save_boxes(self) -> None:
        slug = self._active_slug
        current_img = self._current_label_path()
        if not slug or current_img is None:
            return
        boxes = self._canvas.boxes
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
            if job is None:
                return
            new_labels = dict(job.labels)
            new_labels[current_img.name] = [list(b) for b in boxes]
            JobStore.update(slug, labels=new_labels)
        except Exception as exc:
            self.errorRaised.emit(f"Save boxes failed: {exc}")
            return
        self._refresh_label_page()
        self._refresh_activity_page()

    # ------------------------------------------------------------------ #
    # Emit page logic
    # ------------------------------------------------------------------ #

    def _refresh_emit_page(self) -> None:
        slug = self._active_slug
        if not slug:
            self._emit_summary.setText("")
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
        except Exception:
            job = None
        if job is None:
            self._emit_summary.setText("Job not found.")
            return
        labeled = sum(1 for v in job.labels.values() if v)
        self._emit_summary.setText(
            f"Staged images: {job.staged_count}\n\n"
            f"Labeled images: {labeled}\n\n"
            f"Classes defined: {len(job.classes)}"
        )

    def _on_emit(self) -> None:
        slug = self._active_slug
        if not slug:
            return
        try:
            from mlops.scrap.emit import LabeledItem, emit_yolo_dataset  # noqa: PLC0415
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            from mlops.pipeline import registry as reg  # noqa: PLC0415

            job = JobStore.load(slug)
            if job is None:
                self.errorRaised.emit("Job not found.")
                return

            images = self._get_staged_images(slug)
            labeled_images = [p for p in images if p.name in job.labels and job.labels[p.name]]
            if len(labeled_images) < 2:
                self._emit_result.setText("[ERROR] Need at least 2 labeled images for a train/val split.")
                return

            items = [
                LabeledItem(
                    image_path=p,
                    boxes=tuple(
                        (int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4]))
                        for b in job.labels[p.name]
                    ),
                )
                for p in labeled_images
            ]

            ds_root = emit_yolo_dataset(
                slug=slug,
                classes=list(job.classes),
                items=items,
                val_frac=self._emit_val_frac.value(),
            )

            reg.create_scenario_profile(
                name=slug,
                display_name=job.topic.title(),
                description=f"Scrap-built scenario for topic '{job.topic}'.",
                base_model=self._emit_base_model.text().strip() or "assets/models/yolov10n.pt",
                dataset=slug,
                classes=list(job.classes),
                hyperparams={"epochs": self._emit_epochs.value(), "imgsz": 640},
                guard_profile="balanced",
                backbone_type="yolo_detection",
            )

            JobStore.update(slug, state="emitted", message="dataset + scenario emitted")
            self._emit_result.setText(
                f"Emitted dataset to {ds_root} and scenario {slug}.yaml. "
                "Ready for training."
            )
            self._refresh_activity_page()
        except Exception as exc:
            log.exception("emit failed for %s", slug)
            self._emit_result.setText(f"[ERROR] {exc}")
            self.errorRaised.emit(str(exc))

    # ------------------------------------------------------------------ #
    # Status page logic
    # ------------------------------------------------------------------ #

    def _refresh_status_page(self) -> None:
        slug = self._active_slug
        if not slug:
            self._status_log.setPlainText("")
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
        except Exception as exc:
            self._status_log.setPlainText(f"Could not load job: {exc}")
            return
        if job is None:
            self._status_log.setPlainText("Job not found.")
            return
        trace = "\n".join(job.processing_log)
        trace_block = (
            f"--- processing_log ({len(job.processing_log)} lines) ---\n{trace}\n\n"
            if job.processing_log
            else "--- processing_log: (empty) ---\n\n"
        )
        qa = self._qa_snapshot() if slug == self._active_slug else {}
        payload = {
            "slug": job.slug,
            "topic": job.topic,
            "state": job.state,
            "message": job.message,
            "raw_count": job.raw_count,
            "staged_count": job.staged_count,
            "classes": job.classes,
            "labeled_images": sum(1 for v in job.labels.values() if v),
            "qa_running": bool(qa.get("running")),
            "qa_model_path": qa.get("model_path", ""),
            "qa_scanned_images": len(qa.get("items") or {}),
            "qa_filter": self._qa_filter_key,
        }
        self._status_log.setPlainText(trace_block + json.dumps(payload, indent=2))

    # ------------------------------------------------------------------ #
    # New job (inline data card)
    # ------------------------------------------------------------------ #

    def _submit_new_job_from_card(self) -> None:
        topic = self._new_topic.text().strip()
        if not topic:
            self.errorRaised.emit("Topic is required.")
            return
        q = self._new_query.text().strip()
        query = q if q else topic
        count = self._new_count.value()

        try:
            from mlops.pipeline import registry as reg  # noqa: PLC0415
            from mlops.scrap.jobs import JobState, JobStore  # noqa: PLC0415

            raw_slug = topic.lower().replace(" ", "_").replace("-", "_")
            raw_slug = "".join(ch for ch in raw_slug if ch.isalnum() or ch == "_") or "topic"
            slug = reg.pick_unique_library_dataset_slug(f"scrap_{raw_slug}")
            reg.create_library_dataset_root(slug)

            JobStore.save(
                JobState(
                    slug=slug,
                    topic=topic,
                    target_count=count,
                    state="pending",
                    message="job created",
                    processing_log=[],
                )
            )
        except Exception as exc:
            self.errorRaised.emit(f"Could not create job: {exc}")
            return

        self._start_scrape_thread(slug, query, clear_raw=False)
        self._refresh_job_list()

        for i in range(self._job_list.count()):
            if self._job_list.item(i).data(Qt.ItemDataRole.UserRole) == slug:
                self._job_list.setCurrentRow(i)
                break

        self._poll_timer.start()
        self._hide_new_job_card()

    def _start_scrape_thread(self, slug: str, query: str, *, clear_raw: bool = False) -> None:
        from mlops.scrap.jobs import JobStore  # noqa: PLC0415

        ev_existing = self._scrape_live.get(slug)
        if ev_existing is not None and ev_existing.is_set():
            self.errorRaised.emit("A scrape worker is already running for this job.")
            return

        preload = JobStore.load(slug)
        if preload is None:
            return

        worker_gen = JobStore.bump_scrape_generation(slug)
        if worker_gen is None:
            return

        ev = threading.Event()
        ev.set()
        self._scrape_live[slug] = ev

        def poll_continue() -> bool:
            """False = abort (generation superseded); blocks while scrape_paused is True."""
            while True:
                job_live = JobStore.load(slug)
                if job_live is None:
                    return False
                if int(job_live.scrape_generation) != int(worker_gen):
                    return False
                if not job_live.scrape_paused:
                    return True
                time.sleep(0.25)

        def _run() -> None:
            try:
                from mlops.scrap.jobs import JobStore  # noqa: PLC0415

                JobStore.append_log(
                    slug,
                    "Worker thread entered — importing Selenium/Chrome stack "
                    "(first run can take tens of seconds before the next log line)…",
                )
                from mlops.scrap.selenium_search import search_google_images  # noqa: PLC0415
                from mlops.scrap import filter as scrap_filter  # noqa: PLC0415
                from mlops.pipeline import registry as reg  # noqa: PLC0415

                def jlog(msg: str) -> None:
                    JobStore.append_log(slug, msg)

                def still_owner() -> bool:
                    jc = JobStore.load(slug)
                    return jc is not None and int(jc.scrape_generation) == int(worker_gen)

                jlog(f"Worker thread started for slug={slug!r} (generation {worker_gen})")
                jlog(f"Parameters: query={query!r}, restart_raw={clear_raw}")

                snapshot = JobStore.load(slug)
                if snapshot is None:
                    return
                target_count = int(snapshot.target_count or 0)
                jlog(f"Target raw count={target_count}")

                JobStore.update(
                    slug,
                    state="scraping",
                    last_scrape_query=query,
                    message=f"searching '{query}'",
                    scrape_paused=False,
                )
                jlog("State set to scraping; resolving dataset paths…")
                base = reg.resolve_library_dataset_path(slug)
                raw_dir = base / "raw"
                staged_dir = base / "staged"
                raw_dir.mkdir(parents=True, exist_ok=True)
                staged_dir.mkdir(parents=True, exist_ok=True)
                jlog(f"raw_dir={raw_dir}")
                jlog(f"staged_dir={staged_dir}")

                if clear_raw:
                    jlog("Clearing raw/ (restart downloads)")
                    shutil.rmtree(raw_dir, ignore_errors=True)
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    JobStore.update(slug, raw_count=0, message="cleared raw/; downloading")

                if not still_owner():
                    jlog("Aborted — job superseded before download phase.")
                    return

                raw_now = sum(1 for p in raw_dir.iterdir() if p.is_file())
                remaining_dl = max(0, target_count - raw_now)
                jlog(
                    f"Raw files on disk={raw_now}; requesting up to "
                    f"{remaining_dl} new download(s)."
                )

                result = search_google_images(
                    query,
                    remaining_dl,
                    raw_dir,
                    on_progress=jlog,
                    poll_continue=poll_continue,
                )

                if not still_owner():
                    return

                if result.cancelled:
                    raw_after = sum(1 for p in raw_dir.iterdir() if p.is_file())
                    jlog(
                        f"Download phase cancelled/superseded "
                        f"(saved this session={len(result.saved)}; raw_total={raw_after})."
                    )
                    JobStore.update(
                        slug,
                        state="paused_downloads",
                        raw_count=raw_after,
                        message="downloads interrupted — use Continue or Restart",
                    )
                    return

                raw_after_done = sum(1 for p in raw_dir.iterdir() if p.is_file())
                summary_dl = (
                    f"Download phase finished: saved={len(result.saved)} "
                    f"attempted={result.attempted} skipped={result.skipped} raw_total={raw_after_done}"
                )
                jlog(summary_dl)
                JobStore.update(
                    slug,
                    raw_count=raw_after_done,
                    message=(
                        f"downloaded to raw_total={raw_after_done}"
                        f" (attempted {result.attempted}, skipped {result.skipped}); staging"
                    ),
                )

                if not poll_continue():
                    return

                jlog("Beginning dedupe_and_stage (perceptual hash, keeping small images)…")
                stage = scrap_filter.dedupe_and_stage(
                    raw_dir,
                    staged_dir,
                    min_size=0,
                    on_progress=jlog,
                    poll_continue=poll_continue,
                )

                if not still_owner():
                    return

                jlog(
                    f"Staging finished: staged_files={len(stage.staged)} "
                    f"skipped_small={stage.skipped_small} "
                    f"skipped_dup={stage.skipped_dup} "
                    f"skipped_unreadable={stage.skipped_unreadable}"
                )
                JobStore.update(
                    slug,
                    state="staged",
                    staged_count=len(stage.staged),
                    message=(
                        f"staged {len(stage.staged)}; "
                        "kept readable small images; "
                        f"dup={stage.skipped_dup} "
                        f"unreadable={stage.skipped_unreadable}"
                    ),
                )
                jlog("Job state set to staged. You can label images on the Label tab.")
            except Exception as exc:
                log.exception("scrape thread failed for %s", slug)
                try:
                    from mlops.scrap.jobs import JobStore  # noqa: PLC0415

                    jc = JobStore.load(slug)
                    if jc is not None and int(jc.scrape_generation) != int(worker_gen):
                        return
                    JobStore.append_log(slug, f"ERROR: {exc}")
                    JobStore.update(slug, state="error", message=f"scrape failed: {exc}")
                except Exception:
                    pass
            finally:
                ev.clear()

        threading.Thread(target=_run, name=f"scrap-{slug}", daemon=True).start()

    # ------------------------------------------------------------------ #
    # Poll timer (updates job list + label page during active scrape)
    # ------------------------------------------------------------------ #

    def _poll_job(self) -> None:
        slug = self._active_slug
        if not slug:
            self._poll_timer.stop()
            return
        try:
            from mlops.scrap.jobs import JobStore  # noqa: PLC0415
            job = JobStore.load(slug)
        except Exception:
            return
        if job is None:
            return

        live = self._scrape_worker_live(slug)
        if live or job.state in ("pending", "scraping"):
            self._refresh_activity_page()
            self._refresh_label_page()
            return

        self._poll_timer.stop()
        self._staged_images = self._get_staged_images(slug)
        self._ensure_qa_state_for_job(slug)
        self._refresh_activity_page()
        self._refresh_label_page()
        self._refresh_emit_page()
        self._refresh_job_list()
        self._activate_stack_tab(self._preferred_tab_for_job(slug))
