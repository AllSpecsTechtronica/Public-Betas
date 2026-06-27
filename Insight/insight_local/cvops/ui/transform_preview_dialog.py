"""Transform preview dialog for the detect-group → dataset promote flow.

Shows a side-by-side before/after view of every staged image in the selected
detection group so the operator can verify exactly what will be written to disk
before clicking Promote.

Left side  — original model detections or hand-drawn boxes (red)
Right side — transformed YOLO boxes with target class (teal), plus the raw
             file content that would be written in the selected output format.

Output formats supported in the preview:
  YOLO .txt  — one line per box:  class_idx cx cy w h  (fall_detection style)
  CSV        — id,label rows  +  bare class-name .txt  (FaceRecognition style)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_RED = QColor(220, 50, 50)
_RED_FILL = QColor(220, 50, 50, 48)
_TEAL = QColor(10, 143, 168)
_TEAL_FILL = QColor(10, 143, 168, 52)

_SOURCE_LABELS = {
    "hand_drawn": "hand-drawn boxes (geometry preserved, class remapped)",
    "qa_detection": "QA model detections (converted to YOLO normalized coords)",
    "full_frame_fallback": "no detections — full-frame fallback box applied",
}


def _mono_font(size: int = 9) -> QFont:
    f = QFont("JetBrains Mono", size)
    if not f.exactMatch():
        f = QFont("IBM Plex Mono", size)
    f.setStyleHint(QFont.StyleHint.Monospace)
    return f


class _ReadOnlyBoxCanvas(QGraphicsView):
    """Read-only QGraphicsView that renders an image and overlays coloured boxes.

    Supports two box inputs:
      draw_yolo_boxes  — normalised [cls_idx, cx, cy, nw, nh] coords
      draw_qa_dets     — raw detection dicts whose x1/y1/x2/y2 are in the
                         model's frame space; we normalise using frame_w/frame_h
                         then scale back to image pixels so the boxes align
                         correctly regardless of the QA model's input resolution.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._pix_item: Optional[QGraphicsPixmapItem] = None
        self._img_w = 1
        self._img_h = 1

    def load_image_path(self, path: Path) -> None:
        self._scene.clear()
        self._pix_item = None
        pix = QPixmap(str(path))
        if pix.isNull():
            self._img_w = 1
            self._img_h = 1
            return
        self._pix_item = self._scene.addPixmap(pix)
        self._img_w = pix.width()
        self._img_h = pix.height()
        self._scene.setSceneRect(0.0, 0.0, float(self._img_w), float(self._img_h))
        # fitInView here is best-effort — showEvent / resizeEvent will redo it
        # once the widget has its real allocated size.
        self._fit()

    def draw_yolo_boxes(
        self,
        boxes: list[list[float]],
        class_label: str,
        color: QColor,
        fill: QColor,
    ) -> None:
        """Draw YOLO-normalised [cls, cx, cy, nw, nh] boxes onto the loaded image."""
        font = _mono_font(9)
        for b in boxes:
            if len(b) < 5:
                continue
            _, cx, cy, nw, nh = b
            x = (cx - nw / 2.0) * self._img_w
            y = (cy - nh / 2.0) * self._img_h
            w = nw * self._img_w
            h = nh * self._img_h
            if w <= 0 or h <= 0:
                continue
            rect = self._scene.addRect(x, y, w, h, QPen(color, 2), QBrush(fill))
            rect.setZValue(3)
            ti = QGraphicsSimpleTextItem(class_label)
            ti.setFont(font)
            ti.setBrush(QBrush(color))
            ti.setPos(x + 2, max(0.0, y - 16))
            ti.setZValue(4)
            self._scene.addItem(ti)

    def draw_qa_dets(
        self,
        dets: list[dict[str, Any]],
        color: QColor,
        fill: QColor,
    ) -> None:
        """Draw QA detection dicts.

        x1/y1/x2/y2 are in the QA model's frame coordinate space
        (frame_w × frame_h).  We normalise them first, then scale to
        the actual image pixel dimensions so boxes align with the image
        regardless of the resolution the model ran at.
        """
        font = _mono_font(9)
        for det in dets:
            try:
                fw = float(det.get("frame_w") or 0)
                fh = float(det.get("frame_h") or 0)
                x1 = float(det["x1"])
                y1 = float(det["y1"])
                x2 = float(det["x2"])
                y2 = float(det["y2"])
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            # Normalise to [0,1] using the QA model's frame dims (fall back to
            # treating the coords as already normalised if frame_w/h are 0).
            if fw > 0 and fh > 0:
                nx1, nx2 = x1 / fw, x2 / fw
                ny1, ny2 = y1 / fh, y2 / fh
            else:
                nx1, nx2, ny1, ny2 = x1, x2, y1, y2
            px1 = nx1 * self._img_w
            py1 = ny1 * self._img_h
            pw = (nx2 - nx1) * self._img_w
            ph = (ny2 - ny1) * self._img_h
            if pw <= 0 or ph <= 0:
                continue
            label = str(det.get("label") or "")
            conf = float(det.get("conf") or 0.0)
            rect = self._scene.addRect(
                px1, py1, pw, ph,
                QPen(color, 2, Qt.PenStyle.DashLine),
                QBrush(fill),
            )
            rect.setZValue(3)
            ti = QGraphicsSimpleTextItem(f"{label} {conf:.2f}")
            ti.setFont(font)
            ti.setBrush(QBrush(color))
            ti.setPos(px1 + 2, max(0.0, py1 - 16))
            ti.setZValue(4)
            self._scene.addItem(ti)

    # -- Qt overrides --

    def _fit(self) -> None:
        if self._pix_item is not None and self.viewport().width() > 0:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._fit()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._fit()


class TransformPreviewDialog(QDialog):
    """Side-by-side before/after label preview for a detection-group promotion.

    Constructed from the dict returned by ``compute_transform_preview()``.
    Calling ``exec()`` returns Accepted only when the operator clicks Promote,
    giving the caller a chance to bail out without writing anything.
    """

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        preview: dict[str, dict[str, Any]],
        promoted_class: str,
        promoted_class_idx: int,
        group_summary: str,
        filter_key: str,
    ) -> None:
        super().__init__(parent)
        self._preview = preview
        self._names = list(preview.keys())
        self._index = 0
        self._promoted_class = promoted_class
        self._promoted_class_idx = promoted_class_idx
        self._fmt = "yolo"

        self.setWindowTitle("Transform Preview — verify before promote")
        self.resize(1140, 740)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # -- Header --
        hdr = QLabel(
            f"Group: {group_summary}   →   "
            f'class "{promoted_class}" (index {promoted_class_idx})'
        )
        hdr.setStyleSheet("font-weight: 700; font-size: 12px;")
        outer.addWidget(hdr)

        hint = QLabel(
            "Review the label transform for each image. "
            "Left shows what the model detected; right shows exactly what will be written. "
            "Click Promote when satisfied, or Cancel to go back."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.65);")
        outer.addWidget(hint)

        # -- Nav bar --
        nav = QHBoxLayout()
        self._prev_btn = QPushButton("← Prev")
        self._prev_btn.clicked.connect(self._go_prev)
        nav.addWidget(self._prev_btn)
        self._nav_label = QLabel("")
        self._nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._nav_label.setStyleSheet("font-weight: 600; font-size: 11px;")
        nav.addWidget(self._nav_label, stretch=1)
        self._next_btn = QPushButton("Next →")
        self._next_btn.clicked.connect(self._go_next)
        nav.addWidget(self._next_btn)
        outer.addLayout(nav)

        # -- Split view --
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(3)

        # Left: BEFORE
        left_w = QWidget()
        lv = QVBoxLayout(left_w)
        lv.setContentsMargins(0, 0, 4, 0)
        lv.setSpacing(4)
        before_hdr = QLabel("[BEFORE] — model detections / hand-drawn boxes")
        before_hdr.setStyleSheet("font-size: 10px; font-weight: 600; color: rgba(220,50,50,0.9);")
        lv.addWidget(before_hdr)
        self._before_canvas = _ReadOnlyBoxCanvas()
        self._before_canvas.setMinimumSize(380, 280)
        lv.addWidget(self._before_canvas, stretch=1)
        self._before_info = QLabel("")
        self._before_info.setWordWrap(True)
        self._before_info.setFont(_mono_font(9))
        self._before_info.setStyleSheet("color: rgba(220,50,50,0.85);")
        lv.addWidget(self._before_info)
        splitter.addWidget(left_w)

        # Right: AFTER
        right_w = QWidget()
        rv = QVBoxLayout(right_w)
        rv.setContentsMargins(4, 0, 0, 0)
        rv.setSpacing(4)
        after_hdr = QLabel("[AFTER] — transformed labels (will be written)")
        after_hdr.setStyleSheet("font-size: 10px; font-weight: 600; color: rgba(10,143,168,0.9);")
        rv.addWidget(after_hdr)
        self._after_canvas = _ReadOnlyBoxCanvas()
        self._after_canvas.setMinimumSize(380, 280)
        rv.addWidget(self._after_canvas, stretch=1)

        # Format toggle
        fmt_row = QHBoxLayout()
        fmt_lbl = QLabel("Output format:")
        fmt_lbl.setStyleSheet("font-size: 10px;")
        fmt_row.addWidget(fmt_lbl)
        self._fmt_group = QButtonGroup(self)
        self._radio_yolo = QRadioButton("YOLO .txt")
        self._radio_yolo.setChecked(True)
        self._radio_yolo.toggled.connect(self._on_fmt_changed)
        self._fmt_group.addButton(self._radio_yolo)
        fmt_row.addWidget(self._radio_yolo)
        self._radio_csv = QRadioButton("CSV (id,label)")
        self._radio_csv.toggled.connect(self._on_fmt_changed)
        self._fmt_group.addButton(self._radio_csv)
        fmt_row.addWidget(self._radio_csv)
        fmt_row.addStretch(1)
        rv.addLayout(fmt_row)

        self._label_path_info = QLabel("")
        self._label_path_info.setStyleSheet(
            "font-size: 10px; color: rgba(133,153,0,0.7); font-family: monospace;"
        )
        rv.addWidget(self._label_path_info)

        self._txt_display = QTextEdit()
        self._txt_display.setReadOnly(True)
        self._txt_display.setMaximumHeight(110)
        self._txt_display.setFont(_mono_font(10))
        self._txt_display.setStyleSheet(
            "QTextEdit { background: rgba(10,143,168,0.06); border: 1px solid rgba(10,143,168,0.25); }"
        )
        rv.addWidget(self._txt_display)

        self._source_info = QLabel("")
        self._source_info.setWordWrap(True)
        self._source_info.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        rv.addWidget(self._source_info)

        splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, stretch=1)

        # -- Bottom buttons --
        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)
        btn_row.addStretch(1)
        count_lbl = QLabel(f"{len(self._names)} image(s) in this group")
        count_lbl.setStyleSheet("font-size: 10px; color: rgba(133,153,0,0.6);")
        btn_row.addWidget(count_lbl)
        btn_row.addSpacing(12)
        promote_btn = QPushButton("Promote →")
        promote_btn.setMinimumWidth(120)
        promote_btn.setStyleSheet(
            "QPushButton { background: #0A8FA8; color: #fff; font-weight: 700; "
            "border-radius: 3px; padding: 4px 14px; }"
            "QPushButton:hover { background: #0d9fba; }"
        )
        promote_btn.clicked.connect(self.accept)
        btn_row.addWidget(promote_btn)
        outer.addLayout(btn_row)

        # Defer the first refresh so the dialog is fully laid out (and canvases
        # have real pixel dimensions) before we try to load images.  Without this,
        # fitInView is called against a 0×0 viewport during __init__ and the image
        # never scales to fill the canvas.
        self._nav_label.setText(
            f"1 / {len(self._names)}  ·  {self._names[0]}"
            if self._names else "No images in group"
        )
        self._prev_btn.setEnabled(False)
        self._next_btn.setEnabled(len(self._names) > 1)
        QTimer.singleShot(0, self._refresh)

    # -- Navigation --

    def _go_prev(self) -> None:
        if self._index > 0:
            self._index -= 1
            self._refresh()

    def _go_next(self) -> None:
        if self._index < len(self._names) - 1:
            self._index += 1
            self._refresh()

    def _on_fmt_changed(self) -> None:
        self._fmt = "csv" if self._radio_csv.isChecked() else "yolo"
        self._refresh_txt()

    # -- Rendering --

    def _current_entry(self) -> dict[str, Any]:
        if not self._names:
            return {}
        return self._preview.get(self._names[self._index]) or {}

    def _refresh(self) -> None:
        n = len(self._names)
        if not n:
            self._nav_label.setText("No images in group")
            self._prev_btn.setEnabled(False)
            self._next_btn.setEnabled(False)
            self._before_info.setText("")
            self._source_info.setText("")
            self._txt_display.clear()
            return

        name = self._names[self._index]
        entry = self._current_entry()
        img_path = entry.get("image_path")

        self._nav_label.setText(f"{self._index + 1} / {n}  ·  {name}")
        self._prev_btn.setEnabled(self._index > 0)
        self._next_btn.setEnabled(self._index < n - 1)

        original_source = str(entry.get("original_source") or "full_frame_fallback")
        original_dets: list[dict] = list(entry.get("original_dets") or [])
        original_hand_boxes: list[list[float]] = list(entry.get("original_hand_boxes") or [])
        transformed_boxes: list[list[float]] = list(entry.get("transformed_boxes") or [])

        valid_path = (
            img_path is not None
            and isinstance(img_path, Path)
            and img_path.is_file()
        )
        if valid_path:
            self._before_canvas.load_image_path(img_path)  # type: ignore[arg-type]
            self._after_canvas.load_image_path(img_path)   # type: ignore[arg-type]
        else:
            self._before_canvas._scene.clear()
            self._before_canvas._pix_item = None
            self._after_canvas._scene.clear()
            self._after_canvas._pix_item = None

        # -- Before canvas --
        if original_source == "hand_drawn":
            self._before_canvas.draw_yolo_boxes(original_hand_boxes, "hand-drawn", _RED, _RED_FILL)
            lines = [
                f"cls={int(b[0])}  cx={b[1]:.4f}  cy={b[2]:.4f}  w={b[3]:.4f}  h={b[4]:.4f}"
                for b in original_hand_boxes
                if len(b) >= 5
            ]
            self._before_info.setText("\n".join(lines[:8]))
        elif original_source == "qa_detection":
            self._before_canvas.draw_qa_dets(original_dets, _RED, _RED_FILL)
            lines = [
                f"{d.get('label', '?')}  conf={float(d.get('conf', 0)):.3f}"
                for d in original_dets[:8]
            ]
            self._before_info.setText("\n".join(lines))
        else:
            self._before_info.setText("(no detections — full-frame fallback)")

        # -- After canvas --
        self._after_canvas.draw_yolo_boxes(
            transformed_boxes, self._promoted_class, _TEAL, _TEAL_FILL
        )

        self._source_info.setText(
            "Transform source: "
            + _SOURCE_LABELS.get(original_source, original_source)
        )
        self._refresh_txt()

    def _refresh_txt(self) -> None:
        entry = self._current_entry()
        name = self._names[self._index] if self._names else ""
        stem = Path(name).stem if name else ""

        if self._fmt == "yolo":
            self._label_path_info.setText(f"labels/{{split}}/{stem}.txt")
            self._txt_display.setPlainText(str(entry.get("yolo_txt") or "").rstrip())
        else:
            csv_row = str(entry.get("csv_row") or "")
            bare = self._promoted_class
            self._label_path_info.setText(
                f"Dataset.csv (id,label)     labels/{{split}}/{stem}.txt (bare class name)"
            )
            note = "# geometry is dropped in CSV/classification format"
            self._txt_display.setPlainText(
                f"# Dataset.csv row:\n{csv_row}\n\n# labels/{{split}}/{stem}.txt:\n{bare}\n\n{note}"
            )
