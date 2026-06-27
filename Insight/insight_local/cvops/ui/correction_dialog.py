"""Modal for flagging a misdetection during video testing.

Workflow:
    1. Caller pauses the video, captures a QImage of the current frame, and
       gathers the model's current detections.
    2. Dialog opens showing the frame. Model detections render as RED ghost
       boxes (visual reference; not stored unless re-confirmed). User-drawn
       ground-truth boxes render in green.
    3. User can:
         - Draw a new box and pick / type a label  (false negative)
         - Click "Confirm model box" to copy a model detection into GT
         - Click "Whole frame: <label>" for whole-frame classifier flows
           (e.g. fall detection where the entire scene is the label).
         - Add a free-text label not in the model's class list.
    4. On accept, returns the list of ground-truth boxes + notes; caller
       writes a Correction record + JPEG via corrections_store.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QImage, QPen, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGraphicsRectItem,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .annotation_editor import YoloBox, _AnnotationView
from .cvops_theme import cvops_color, cvops_qcolor


@dataclass
class CorrectionResult:
    ground_truth: list[dict]   # [{label, x1, y1, x2, y2}, ...]
    notes: str
    kind: str                  # "fn" / "fp" / "relabel" / "mixed" / "whole_frame"


class CorrectionDialog(QDialog):
    """One-shot dialog. Use `result_payload` after exec() returns Accepted."""

    def __init__(
        self,
        *,
        frame: QImage,
        frame_ts_ms: int,
        video_name: str,
        model_name: str,
        model_detections: list[dict],
        known_classes: list[str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._model_detections = list(model_detections or [])
        # Ground-truth boxes the user has added/confirmed. Each is a dict
        # with label + pixel xyxy in the original frame's coordinate space.
        self._gt: list[dict] = []
        self._known_classes: list[str] = []
        self._frame_w = int(frame.width())
        self._frame_h = int(frame.height())
        # Outcome the caller reads after exec().
        self.result_payload: Optional[CorrectionResult] = None

        ts_label = self._fmt_ts(frame_ts_ms)
        self.setWindowTitle(f"Flag detection — {video_name} @ {ts_label}")
        self.resize(1100, 760)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        # ----- header -----
        header = QHBoxLayout()
        title = QLabel(
            f"<b>{video_name}</b> &nbsp; @ {ts_label} &nbsp; "
            f"<span style='color:{cvops_color('text_iron')}'>· model: {model_name}</span>"
        )
        title.setTextFormat(Qt.TextFormat.RichText)
        header.addWidget(title, stretch=1)
        legend = QLabel(
            f"<span style='color:{cvops_color('accent_alert')}'>[#] model detection</span> &nbsp; "
            f"<span style='color:{cvops_color('accent_select')}'>[#] your correction</span>"
        )
        legend.setTextFormat(Qt.TextFormat.RichText)
        header.addWidget(legend)
        outer.addLayout(header)

        # ----- body: image + sidebar -----
        body = QHBoxLayout()
        body.setSpacing(8)

        self._view = _AnnotationView()
        self._view.set_image(QPixmap.fromImage(frame))
        # Render model detections as ghost boxes for reference. These are
        # NOT stored in self._gt; the user must explicitly confirm or draw
        # for a box to count as ground truth.
        self._draw_model_ghosts()
        self._view.rectFinalized.connect(self._on_rect_drawn)
        body.addWidget(self._view, stretch=1)

        side = self._build_sidebar()
        body.addLayout(side)
        outer.addLayout(body, stretch=1)

        # ----- footer -----
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        btns.button(QDialogButtonBox.StandardButton.Save).setText("Save correction")
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        outer.addWidget(btns)

        # Pre-populate class combo from model + observed detection labels.
        seen: list[str] = []
        for c in known_classes or []:
            c = str(c).strip()
            if c and c not in seen:
                seen.append(c)
        for d in self._model_detections:
            lbl = str(d.get("label", "")).strip()
            if lbl and lbl not in seen:
                seen.append(lbl)
        self._known_classes = seen
        self._refresh_class_combo()

    # ------------------------------------------------------------------
    # Sidebar
    # ------------------------------------------------------------------

    def _build_sidebar(self) -> QVBoxLayout:
        side = QVBoxLayout()
        side.setSpacing(8)

        # --- Class picker ---
        klass_card = QFrame()
        klass_card.setFrameShape(QFrame.Shape.StyledPanel)
        kl = QVBoxLayout(klass_card)
        kl.setContentsMargins(8, 6, 8, 8)
        kl.setSpacing(4)
        kl.addWidget(QLabel("<b>Class</b>"))
        self._class_combo = QComboBox()
        self._class_combo.setEditable(True)
        self._class_combo.setMinimumWidth(220)
        self._class_combo.setToolTip(
            "Pick a known class or type a new label (e.g. 'fallen'). New "
            "labels are saved with the correction and surface in future runs."
        )
        kl.addWidget(self._class_combo)

        draw_row = QHBoxLayout()
        self._draw_btn = QPushButton("Draw box")
        self._draw_btn.setCheckable(True)
        self._draw_btn.toggled.connect(self._view.set_draw_mode)
        draw_row.addWidget(self._draw_btn)

        self._whole_btn = QPushButton("Whole frame")
        self._whole_btn.setToolTip(
            "Add a full-image box with the current class. Use this for "
            "whole-frame classifiers (e.g. fall vs no-fall)."
        )
        self._whole_btn.clicked.connect(self._add_whole_frame_box)
        draw_row.addWidget(self._whole_btn)
        kl.addLayout(draw_row)
        side.addWidget(klass_card)

        # --- Model detections list (clickable to confirm) ---
        det_card = QFrame()
        det_card.setFrameShape(QFrame.Shape.StyledPanel)
        dl = QVBoxLayout(det_card)
        dl.setContentsMargins(8, 6, 8, 8)
        dl.setSpacing(4)
        dl.addWidget(QLabel("<b>Model detections</b>"))
        if not self._model_detections:
            empty = QLabel("<i>(none — model returned nothing for this frame)</i>")
            empty.setStyleSheet("color: rgba(160,160,160,0.85);")
            dl.addWidget(empty)
        else:
            for i, det in enumerate(self._model_detections):
                row = QHBoxLayout()
                lbl = QLabel(
                    f"{det.get('label','?')} &nbsp;<span style='color:{cvops_color('text_iron')}'>"
                    f"{float(det.get('conf',0.0)):.2f}</span>"
                )
                lbl.setTextFormat(Qt.TextFormat.RichText)
                row.addWidget(lbl, stretch=1)
                btn = QPushButton("Confirm")
                btn.setToolTip(
                    "Treat this model box as correct ground truth (copies it "
                    "into the green box list)."
                )
                btn.clicked.connect(lambda _checked, idx=i: self._confirm_model_box(idx))
                row.addWidget(btn)
                dl.addLayout(row)
        side.addWidget(det_card)

        # --- Ground truth (live count) + tools ---
        gt_card = QFrame()
        gt_card.setFrameShape(QFrame.Shape.StyledPanel)
        gl = QVBoxLayout(gt_card)
        gl.setContentsMargins(8, 6, 8, 8)
        gl.setSpacing(4)
        gl.addWidget(QLabel("<b>Your corrections</b>"))
        self._gt_count_label = QLabel("0 boxes")
        gl.addWidget(self._gt_count_label)
        gt_btns = QHBoxLayout()
        undo = QPushButton("Undo")
        undo.clicked.connect(self._undo_gt)
        clr = QPushButton("Clear")
        clr.clicked.connect(self._clear_gt)
        gt_btns.addWidget(undo)
        gt_btns.addWidget(clr)
        gl.addLayout(gt_btns)
        side.addWidget(gt_card)

        # --- Notes ---
        notes_card = QFrame()
        notes_card.setFrameShape(QFrame.Shape.StyledPanel)
        nl = QVBoxLayout(notes_card)
        nl.setContentsMargins(8, 6, 8, 8)
        nl.setSpacing(4)
        nl.addWidget(QLabel(f"<b>Notes</b> <span style='color:{cvops_color('text_iron')}'>(optional)</span>"))
        self._notes = QPlainTextEdit()
        self._notes.setPlaceholderText(
            "Why is the model wrong here? Anything to flag for retraining?"
        )
        self._notes.setMaximumHeight(120)
        nl.addWidget(self._notes)
        side.addWidget(notes_card)

        side.addStretch(1)
        return side

    # ------------------------------------------------------------------
    # Class combo
    # ------------------------------------------------------------------

    def _refresh_class_combo(self) -> None:
        cur = str(self._class_combo.currentText() or "").strip()
        self._class_combo.blockSignals(True)
        self._class_combo.clear()
        for name in self._known_classes:
            self._class_combo.addItem(name)
        if cur:
            idx = self._class_combo.findText(cur)
            if idx >= 0:
                self._class_combo.setCurrentIndex(idx)
            else:
                self._class_combo.setEditText(cur)
        self._class_combo.blockSignals(False)

    def _current_label(self) -> str:
        return str(self._class_combo.currentText() or "").strip()

    def _ensure_class(self, label: str) -> str:
        label = label.strip()
        if not label:
            return ""
        if label not in self._known_classes:
            self._known_classes.append(label)
            self._refresh_class_combo()
        return label

    # ------------------------------------------------------------------
    # Model ghost boxes (red) + GT boxes (green)
    # ------------------------------------------------------------------

    def _draw_model_ghosts(self) -> None:
        scene = self._view.scene()
        if scene is None:
            return
        pen = QPen(cvops_qcolor("accent_alert", 220), 2)
        pen.setStyle(Qt.PenStyle.DashLine)
        for det in self._model_detections:
            try:
                x1 = float(det["x1"]); y1 = float(det["y1"])
                x2 = float(det["x2"]); y2 = float(det["y2"])
            except Exception:
                continue
            item = QGraphicsRectItem(QRectF(x1, y1, x2 - x1, y2 - y1))
            item.setPen(pen)
            item.setBrush(cvops_qcolor("accent_alert", 30))
            item.setZValue(5)  # below GT boxes
            scene.addItem(item)

    def _on_rect_drawn(self, rect: QRectF) -> None:
        label = self._ensure_class(self._current_label())
        if not label:
            QMessageBox.warning(
                self,
                "Pick a class",
                "Choose or type a class name before drawing a correction box.",
            )
            return
        self._gt.append({
            "label": label,
            "x1": float(rect.left()),
            "y1": float(rect.top()),
            "x2": float(rect.right()),
            "y2": float(rect.bottom()),
        })
        self._sync_gt_view()

    def _confirm_model_box(self, idx: int) -> None:
        if not (0 <= idx < len(self._model_detections)):
            return
        det = self._model_detections[idx]
        label = str(det.get("label", "")).strip()
        if not label:
            label = self._current_label()
        label = self._ensure_class(label)
        if not label:
            return
        try:
            self._gt.append({
                "label": label,
                "x1": float(det["x1"]), "y1": float(det["y1"]),
                "x2": float(det["x2"]), "y2": float(det["y2"]),
            })
        except Exception:
            return
        self._sync_gt_view()

    def _add_whole_frame_box(self) -> None:
        label = self._ensure_class(self._current_label())
        if not label:
            QMessageBox.warning(
                self,
                "Pick a class",
                "Choose or type a class name before adding a whole-frame box.",
            )
            return
        if self._frame_w <= 0 or self._frame_h <= 0:
            return
        self._gt.append({
            "label": label,
            "x1": 0.0, "y1": 0.0,
            "x2": float(self._frame_w), "y2": float(self._frame_h),
        })
        self._sync_gt_view()

    def _undo_gt(self) -> None:
        if not self._gt:
            return
        self._gt.pop()
        self._sync_gt_view()

    def _clear_gt(self) -> None:
        if not self._gt:
            return
        self._gt.clear()
        self._sync_gt_view()

    def _sync_gt_view(self) -> None:
        # Convert GT dicts into YoloBox-pixel boxes for the existing renderer.
        # YoloBox carries class_id but the renderer ignores it for paint.
        boxes = [
            YoloBox(0, b["x1"], b["y1"], b["x2"], b["y2"]) for b in self._gt
        ]
        self._view.set_boxes(boxes)
        # Re-draw model ghosts since set_boxes clears them too.
        self._draw_model_ghosts()
        self._gt_count_label.setText(
            f"{len(self._gt)} box{'es' if len(self._gt) != 1 else ''}"
        )

    # ------------------------------------------------------------------
    # Accept
    # ------------------------------------------------------------------

    def _infer_kind(self) -> str:
        gt = self._gt
        model_n = len(self._model_detections)
        if not gt and model_n > 0:
            return "fp"
        if gt and model_n == 0:
            return "fn"
        if gt and model_n > 0:
            # Heuristic: same count + same labels in same order = relabel,
            # otherwise mixed. Good enough for a summary tag; the diff is
            # always reconstructable from model_detections + ground_truth.
            if len(gt) == model_n:
                gt_labels = [b.get("label") for b in gt]
                m_labels = [d.get("label") for d in self._model_detections]
                if gt_labels != m_labels:
                    return "relabel"
            return "mixed"
        return "fp"  # nothing on either side: treat as "model overshot"

    def _accept(self) -> None:
        if not self._gt and not self._model_detections:
            QMessageBox.information(
                self,
                "Nothing to save",
                "There are no model detections to flag and no ground-truth "
                "boxes drawn. Add at least one before saving.",
            )
            return
        # Empty GT with non-empty model output is a valid correction
        # (everything the model returned is a false positive).
        self.result_payload = CorrectionResult(
            ground_truth=list(self._gt),
            notes=self._notes.toPlainText().strip(),
            kind=self._infer_kind(),
        )
        self.accept()

    @staticmethod
    def _fmt_ts(ms: int) -> str:
        ms = max(0, int(ms))
        s, ms = divmod(ms, 1000)
        m, s = divmod(s, 60)
        h, m = divmod(m, 60)
        if h:
            return f"{h:d}:{m:02d}:{s:02d}.{ms:03d}"
        return f"{m:02d}:{s:02d}.{ms:03d}"
