from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QPen, QPixmap, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsView

from ...ui.media_utils import pixmap_from_b64_jpeg
from .cvops_theme import cvops_qcolor


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _norm_rect(rect: QRectF) -> QRectF:
    x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
    return QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))


def _parse_yolo_boxes(text: str, *, w: int, h: int) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for raw in (text or "").splitlines():
        ln = raw.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 5:
            continue
        try:
            cid = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            bw = float(parts[3])
            bh = float(parts[4])
        except Exception:
            continue
        if w <= 0 or h <= 0:
            continue
        x1 = (xc - bw / 2.0) * float(w)
        x2 = (xc + bw / 2.0) * float(w)
        y1 = (yc - bh / 2.0) * float(h)
        y2 = (yc + bh / 2.0) * float(h)
        boxes.append(YoloBox(cid, x1, y1, x2, y2))
    return boxes


def _boxes_to_yolo_text(boxes: list[YoloBox], *, w: int, h: int) -> str:
    if w <= 0 or h <= 0:
        return ""
    lines: list[str] = []
    for b in boxes:
        x1 = _clamp(min(b.x1, b.x2), 0.0, float(w))
        x2 = _clamp(max(b.x1, b.x2), 0.0, float(w))
        y1 = _clamp(min(b.y1, b.y2), 0.0, float(h))
        y2 = _clamp(max(b.y1, b.y2), 0.0, float(h))
        bw = max(0.0, x2 - x1) / float(w)
        bh = max(0.0, y2 - y1) / float(h)
        if bw <= 0.0 or bh <= 0.0:
            continue
        xc = ((x1 + x2) / 2.0) / float(w)
        yc = ((y1 + y2) / 2.0) / float(h)
        lines.append(f"{int(b.class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


class _AnnotationView(QGraphicsView):
    rectFinalized = pyqtSignal(QRectF)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._rect_items: list[QGraphicsRectItem] = []
        self._temp_rect: Optional[QGraphicsRectItem] = None
        self._drawing = False
        self._draw_mode = False
        self._start = QPointF()
        self._image_w = 0
        self._image_h = 0

    def image_size(self) -> tuple[int, int]:
        return self._image_w, self._image_h

    def set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = bool(enabled)
        if self._draw_mode:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._drawing = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def set_image(self, pix: QPixmap) -> None:
        scene = self.scene()
        if scene is None:
            return
        scene.clear()
        self._rect_items.clear()
        self._temp_rect = None
        self._pixmap_item = None
        self._image_w = int(pix.width())
        self._image_h = int(pix.height())
        item = QGraphicsPixmapItem(pix)
        item.setZValue(0)
        scene.addItem(item)
        self._pixmap_item = item
        scene.setSceneRect(QRectF(0, 0, float(self._image_w), float(self._image_h)))
        self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def clear_boxes(self) -> None:
        scene = self.scene()
        if scene is None:
            return
        for it in self._rect_items:
            scene.removeItem(it)
        self._rect_items.clear()

    def set_boxes(self, boxes: list[YoloBox]) -> None:
        self.clear_boxes()
        for b in boxes:
            rect = QRectF(min(b.x1, b.x2), min(b.y1, b.y2), abs(b.x2 - b.x1), abs(b.y2 - b.y1))
            self._rect_items.append(self._add_rect_item(rect))

    def selected_box_indices(self) -> list[int]:
        scene = self.scene()
        if scene is None:
            return []
        out: list[int] = []
        for it in scene.selectedItems():
            if isinstance(it, QGraphicsRectItem) and it in self._rect_items:
                out.append(self._rect_items.index(it))
        return sorted(set(out))

    def refresh_theme_styles(self) -> None:
        for item in self._rect_items:
            self._apply_box_style(item)
        if self._temp_rect is not None:
            self._apply_box_style(self._temp_rect, temporary=True)
        self.viewport().update()

    def _apply_box_style(self, item: QGraphicsRectItem, *, temporary: bool = False) -> None:
        pen = QPen(cvops_qcolor("accent_select", 200 if temporary else 220), 2)
        if temporary:
            pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)
        item.setBrush(cvops_qcolor("accent_select", 30 if temporary else 40))

    def _add_rect_item(self, rect: QRectF) -> QGraphicsRectItem:
        scene = self.scene()
        if scene is None:
            raise RuntimeError("scene missing")
        item = QGraphicsRectItem(rect)
        item.setZValue(10)
        self._apply_box_style(item)
        item.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable, True)
        scene.addItem(item)
        return item

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        scene = self.scene()
        if scene is None:
            return
        if self._pixmap_item is not None:
            self.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        # Ctrl+wheel zoom; otherwise keep default scrolling.
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.scale(1.15, 1.15)
            elif delta < 0:
                self.scale(0.87, 0.87)
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if not self._draw_mode or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        scene = self.scene()
        if scene is None or self._pixmap_item is None:
            return
        self._drawing = True
        self._start = self.mapToScene(event.pos())
        self._start.setX(_clamp(float(self._start.x()), 0.0, float(self._image_w)))
        self._start.setY(_clamp(float(self._start.y()), 0.0, float(self._image_h)))
        if self._temp_rect is None:
            self._temp_rect = QGraphicsRectItem()
            self._temp_rect.setZValue(20)
            self._apply_box_style(self._temp_rect, temporary=True)
            scene.addItem(self._temp_rect)
        self._temp_rect.setRect(QRectF(self._start, self._start))

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not self._draw_mode or not self._drawing or self._temp_rect is None:
            return super().mouseMoveEvent(event)
        pos = self.mapToScene(event.pos())
        pos.setX(_clamp(float(pos.x()), 0.0, float(self._image_w)))
        pos.setY(_clamp(float(pos.y()), 0.0, float(self._image_h)))
        rect = _norm_rect(QRectF(self._start, pos))
        self._temp_rect.setRect(rect)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if not self._draw_mode or event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        if not self._drawing or self._temp_rect is None:
            return
        self._drawing = False
        rect = _norm_rect(self._temp_rect.rect())
        # Remove the temp rectangle; the dialog will add a real box item if accepted.
        scene = self.scene()
        if scene is not None:
            scene.removeItem(self._temp_rect)
        self._temp_rect = None
        if rect.width() < 4 or rect.height() < 4:
            return
        self.rectFinalized.emit(rect)


class AnnotationEditorDialog(QDialog):
    def __init__(
        self,
        *,
        base_url: str,
        dataset_slug: str,
        start_relative_path: str = "",
        classes_override: Optional[list[str]] = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._slug = str(dataset_slug or "").strip()
        self._entries: list[dict[str, Any]] = []
        self._classes: list[str] = []
        self._classes_override = [str(c) for c in (classes_override or []) if str(c)]
        self._index = 0
        self._dirty = False
        self._boxes: list[YoloBox] = []

        self.setWindowTitle(f"Annotation Editor — {self._slug}")
        self.resize(1100, 760)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        top = QHBoxLayout()
        self._title = QLabel("—")
        self._title.setStyleSheet("font-weight: 700; font-size: 12px;")
        top.addWidget(self._title, stretch=1)
        self._pos = QLabel("")
        self._pos.setStyleSheet("color: rgba(133,153,0,0.65);")
        top.addWidget(self._pos)
        outer.addLayout(top)

        self._view = _AnnotationView()
        outer.addWidget(self._view, stretch=1)

        row = QHBoxLayout()
        row.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        self._class_combo.setEditable(True)
        self._class_combo.setMinimumWidth(220)
        row.addWidget(self._class_combo)
        self._add_label_btn = QPushButton("Add Label")
        self._add_label_btn.setToolTip("Add this label to the dataset class list (classes.txt).")
        self._add_label_btn.clicked.connect(self._add_label)
        row.addWidget(self._add_label_btn)

        self._draw_btn = QPushButton("Draw Box")
        self._draw_btn.setCheckable(True)
        self._draw_btn.toggled.connect(self._view.set_draw_mode)
        row.addWidget(self._draw_btn)

        self._undo_btn = QPushButton("Undo")
        self._undo_btn.clicked.connect(self._undo)
        row.addWidget(self._undo_btn)

        self._delete_sel_btn = QPushButton("Delete selected")
        self._delete_sel_btn.setToolTip("Select a box (click it), then press Delete or use this button.")
        self._delete_sel_btn.clicked.connect(self._delete_selected)
        row.addWidget(self._delete_sel_btn)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.clicked.connect(self._clear)
        row.addWidget(self._clear_btn)

        row.addStretch(1)

        self._prev_btn = QPushButton("Prev")
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn = QPushButton("Next")
        self._next_btn.clicked.connect(lambda: self._step(1))
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._save_clicked)
        row.addWidget(self._prev_btn)
        row.addWidget(self._next_btn)
        row.addWidget(self._save_btn)
        outer.addLayout(row)

        quick = QHBoxLayout()
        quick.addWidget(QLabel("Quick Add:"))
        self._add_full_btn = QPushButton("Full Image")
        self._add_full_btn.clicked.connect(self._add_full_image_box)
        quick.addWidget(self._add_full_btn)

        self._center_w = QDoubleSpinBox()
        self._center_w.setRange(0.05, 1.0)
        self._center_w.setSingleStep(0.05)
        self._center_w.setDecimals(2)
        self._center_w.setValue(0.50)
        self._center_h = QDoubleSpinBox()
        self._center_h.setRange(0.05, 1.0)
        self._center_h.setSingleStep(0.05)
        self._center_h.setDecimals(2)
        self._center_h.setValue(0.50)
        quick.addWidget(QLabel("Center w:"))
        quick.addWidget(self._center_w)
        quick.addWidget(QLabel("h:"))
        quick.addWidget(self._center_h)
        self._add_center_btn = QPushButton("Center Box")
        self._add_center_btn.clicked.connect(self._add_center_box)
        quick.addWidget(self._add_center_btn)
        quick.addStretch(1)
        outer.addLayout(quick)

        bulk = QHBoxLayout()
        bulk.addWidget(QLabel("Bulk Apply:"))
        self._bulk_scope = QComboBox()
        self._bulk_scope.addItem("All images", "all")
        self._bulk_scope.addItem("Current split", "split")
        self._bulk_scope.addItem("Class folder", "class_folder")
        bulk.addWidget(self._bulk_scope)
        self._bulk_only_missing = QCheckBox("Only missing")
        self._bulk_only_missing.setChecked(True)
        bulk.addWidget(self._bulk_only_missing)
        self._bulk_replace = QCheckBox("Overwrite")
        self._bulk_replace.setChecked(True)
        bulk.addWidget(self._bulk_replace)
        self._bulk_full_btn = QPushButton("Full Image")
        self._bulk_full_btn.clicked.connect(lambda: self._bulk_apply("full_image"))
        self._bulk_center_btn = QPushButton("Center")
        self._bulk_center_btn.clicked.connect(lambda: self._bulk_apply("center"))
        bulk.addWidget(self._bulk_full_btn)
        bulk.addWidget(self._bulk_center_btn)
        bulk.addStretch(1)
        outer.addLayout(bulk)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size: 10px;")
        outer.addWidget(self._status)

        self._view.rectFinalized.connect(self._on_rect_finalized)

        # Shortcuts
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save_clicked)
        QShortcut(QKeySequence("Left"), self, activated=lambda: self._step(-1))
        QShortcut(QKeySequence("Right"), self, activated=lambda: self._step(1))
        QShortcut(QKeySequence.StandardKey.Delete, self, activated=self._delete_selected)
        QShortcut(QKeySequence(Qt.Key.Key_Backspace), self, activated=self._delete_selected)

        self._load_catalog()
        if start_relative_path:
            idx = self._find_index_by_relative_path(start_relative_path)
            if idx >= 0:
                self._index = idx
        self._load_current()

    # ---------- HTTP ----------

    def _http_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        *,
        timeout: float = 10.0,
    ) -> dict[str, Any]:
        url = self._base_url + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        with urllib.request.urlopen(req, timeout=float(timeout)) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    # ---------- Data ----------

    def _load_catalog(self) -> None:
        if not self._slug:
            return
        enc = urllib.parse.quote(self._slug, safe="")
        try:
            payload = self._http_json("GET", f"/database/{enc}")
        except Exception as exc:
            QMessageBox.critical(self, "Dataset Load Failed", str(exc))
            return
        self._entries = [e for e in (payload.get("images") or []) if isinstance(e, dict)]
        classes = [str(c) for c in (payload.get("classes") or []) if str(c)]
        if self._classes_override and not classes:
            classes = list(self._classes_override)
        self._classes = classes
        self._class_combo.clear()
        if self._classes:
            for i, name in enumerate(self._classes):
                self._class_combo.addItem(name, i)
            self._class_combo.setCurrentIndex(0)
        else:
            self._class_combo.addItem("0", 0)
            self._class_combo.setCurrentIndex(0)

    def _find_index_by_relative_path(self, relative_path: str) -> int:
        want = str(relative_path or "").strip()
        if not want:
            return -1
        for i, e in enumerate(self._entries):
            if str(e.get("relative_path") or "") == want:
                return i
        return -1

    def _current_entry(self) -> dict[str, Any]:
        if 0 <= self._index < len(self._entries):
            return self._entries[self._index]
        return {}

    # ---------- UI ----------

    def _set_dirty(self, value: bool) -> None:
        self._dirty = bool(value)
        base = f"Annotation Editor — {self._slug}"
        self.setWindowTitle(base + (" *" if self._dirty else ""))

    def _load_current(self) -> None:
        entry = self._current_entry()
        rel_path = str(entry.get("relative_path") or "")
        disp = str(entry.get("display_name") or rel_path)
        self._title.setText(disp or "—")
        self._pos.setText(f"{self._index + 1}/{max(1, len(self._entries))}")
        self._status.setText("")
        self._set_dirty(False)

        if not rel_path:
            return
        enc_slug = urllib.parse.quote(self._slug, safe="")
        enc_path = urllib.parse.quote(rel_path, safe="")
        try:
            img_payload = self._http_json("GET", f"/database/{enc_slug}/image/{enc_path}?max_side=1600")
            b64 = str(img_payload.get("image_b64") or "")
            pix = pixmap_from_b64_jpeg(b64)
            if pix.isNull():
                raise ValueError("could not decode image payload")
        except Exception as exc:
            self._view.set_image(QPixmap())
            self._status.setText(f"Failed to load image: {exc}")
            return

        self._view.set_image(pix)
        w, h = self._view.image_size()

        # Load label text and draw existing boxes.
        try:
            label_payload = self._http_json("GET", f"/database/{enc_slug}/label/{enc_path}")
            text = str(label_payload.get("text") or "")
        except Exception:
            text = ""
        self._boxes = _parse_yolo_boxes(text, w=w, h=h)
        self._view.set_boxes(self._boxes)

        # Best-effort class hint for ImageFolder sources: label endpoint returns "class: <name>".
        if text.startswith("class:"):
            class_name = text.replace("class:", "").strip().splitlines()[0].strip()
            if class_name:
                self._class_combo.setCurrentText(class_name)

    def _current_class_id(self) -> Optional[int]:
        raw = str(self._class_combo.currentText() or "").strip()
        if not raw:
            return None
        if raw.isdigit():
            return int(raw)
        # When using class names, require an exact name match so editable-combo text
        # cannot silently map to a stale `currentData()` selection.
        if self._classes:
            for i, name in enumerate(self._classes):
                if name.lower() == raw.lower():
                    return i
            return None
        return None

    def _persist_classes(self) -> bool:
        if not self._slug:
            return False
        enc_slug = urllib.parse.quote(self._slug, safe="")
        try:
            self._http_json("PUT", f"/database/{enc_slug}/classes", {"classes": list(self._classes)})
        except Exception as exc:
            self._status.setText(f"Failed to save classes: {exc}")
            return False
        return True

    def _add_label(self) -> None:
        name = str(self._class_combo.currentText() or "").strip()
        if not name:
            self._status.setText("Enter a label name, then click Add Label.")
            return
        if name.isdigit():
            self._status.setText("Label name must be text (not a number).")
            return

        for i, existing in enumerate(self._classes):
            if existing.lower() == name.lower():
                self._class_combo.setCurrentIndex(i)
                self._status.setText(f"Label already exists: {existing}")
                return

        new_id = len(self._classes)
        self._classes.append(name)
        self._class_combo.addItem(name, new_id)
        added_index = self._class_combo.count() - 1
        self._class_combo.setCurrentIndex(added_index)
        if not self._persist_classes():
            # Roll back local state if the write fails.
            try:
                self._class_combo.removeItem(added_index)
            except Exception:
                pass
            try:
                if self._classes and self._classes[-1].lower() == name.lower():
                    self._classes.pop()
            except Exception:
                pass
            return
        self._status.setText(f"Added label: {name} (id {new_id})")

    def _on_rect_finalized(self, rect: QRectF) -> None:
        cid = self._current_class_id()
        if cid is None:
            QMessageBox.warning(
                self,
                "Missing Class",
                "Select a known class (or Add Label) before drawing a box.",
            )
            return
        norm = _norm_rect(rect)
        self._boxes.append(YoloBox(int(cid), norm.left(), norm.top(), norm.right(), norm.bottom()))
        self._view.set_boxes(self._boxes)
        self._set_dirty(True)

    def _add_full_image_box(self) -> None:
        cid = self._current_class_id()
        if cid is None:
            QMessageBox.warning(
                self,
                "Missing Class",
                "Select a known class (or Add Label) before adding a box.",
            )
            return
        w, h = self._view.image_size()
        if w <= 0 or h <= 0:
            return
        self._boxes.append(YoloBox(int(cid), 0.0, 0.0, float(w), float(h)))
        self._view.set_boxes(self._boxes)
        self._set_dirty(True)

    def _add_center_box(self) -> None:
        cid = self._current_class_id()
        if cid is None:
            QMessageBox.warning(
                self,
                "Missing Class",
                "Select a known class (or Add Label) before adding a box.",
            )
            return
        w, h = self._view.image_size()
        if w <= 0 or h <= 0:
            return
        wf = float(self._center_w.value())
        hf = float(self._center_h.value())
        wf = _clamp(wf, 0.01, 1.0)
        hf = _clamp(hf, 0.01, 1.0)
        bw = float(w) * wf
        bh = float(h) * hf
        x1 = (float(w) - bw) / 2.0
        y1 = (float(h) - bh) / 2.0
        self._boxes.append(YoloBox(int(cid), x1, y1, x1 + bw, y1 + bh))
        self._view.set_boxes(self._boxes)
        self._set_dirty(True)

    @staticmethod
    def _match_class_folder(rel_path: str, class_folder: str) -> bool:
        if not rel_path or not class_folder:
            return False
        want = class_folder.lower()
        parts = urllib.parse.unquote(rel_path).split("/")
        # Prefer images/<split>/<class_folder>/... when present.
        if "images" in parts:
            idx = parts.index("images")
            if len(parts) > idx + 2:
                folder = parts[idx + 2]
                if folder and folder.lower() == want:
                    return True
        return any(p.lower() == want for p in parts if p)

    def _bulk_candidates_count(self) -> int:
        scope = str(self._bulk_scope.currentData() or "all")
        only_missing = bool(self._bulk_only_missing.isChecked())
        replace = bool(self._bulk_replace.isChecked())
        _ = replace  # unused for counting; kept for parity in case we change logic later.

        split = ""
        class_folder = ""
        if scope == "split":
            split = str(self._current_entry().get("split") or "")
        elif scope == "class_folder":
            class_folder = str(self._class_combo.currentText() or "").strip()

        n = 0
        for e in self._entries:
            if scope == "split" and split and str(e.get("split") or "") != split:
                continue
            if scope == "class_folder" and class_folder:
                if not self._match_class_folder(str(e.get("relative_path") or ""), class_folder):
                    continue
            if only_missing and bool(e.get("has_label")):
                continue
            n += 1
        return n

    def _bulk_apply(self, geometry: str) -> None:
        if not self._entries:
            return
        if not self._maybe_save():
            return
        cid = self._current_class_id()
        if cid is None:
            QMessageBox.warning(
                self,
                "Missing Class",
                "Select a known class (or Add Label) before bulk applying.",
            )
            return

        scope = str(self._bulk_scope.currentData() or "all")
        split = ""
        class_folder = ""
        if scope == "split":
            split = str(self._current_entry().get("split") or "")
            if not split:
                QMessageBox.warning(self, "Invalid Split", "Current item has no split.")
                return
        elif scope == "class_folder":
            class_folder = str(self._class_combo.currentText() or "").strip()
            if not class_folder:
                QMessageBox.warning(self, "Missing Folder", "Enter/select a class name for class-folder scope.")
                return

        total = self._bulk_candidates_count()
        if total <= 0:
            self._status.setText("Bulk apply: no matching images.")
            return
        if total >= 500:
            resp = QMessageBox.question(
                self,
                "Confirm Bulk Apply",
                f"This will write labels for about {total} image(s). Continue?",
            )
            if resp != QMessageBox.StandardButton.Yes:
                return

        self._status.setText(f"Bulk apply {geometry}... ({total} image(s))")
        enc_slug = urllib.parse.quote(self._slug, safe="")
        payload = {
            "class_id": int(cid),
            "geometry": str(geometry or "full_image"),
            "center_w": float(self._center_w.value()),
            "center_h": float(self._center_h.value()),
            "scope": scope,
            "split": split,
            "class_folder_name": class_folder,
            "only_missing": bool(self._bulk_only_missing.isChecked()),
            "replace": bool(self._bulk_replace.isChecked()),
        }
        try:
            res = self._http_json(
                "POST",
                f"/database/{enc_slug}/labels/bulk_apply",
                payload,
                timeout=240.0,
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._status.setText(f"Bulk apply failed: HTTP {exc.code} {detail[:160]}")
            return
        except Exception as exc:
            self._status.setText(f"Bulk apply failed: {exc}")
            return

        applied = int(res.get("applied") or 0)
        skipped = int(res.get("skipped") or 0)
        errs = res.get("errors") or []
        msg = f"Bulk applied to {applied}. Skipped {skipped}."
        if errs:
            msg += f" Error: {errs[0]}"
        self._status.setText(msg)

        # Refresh catalog so has_label is up to date, then reload the current image.
        cur_rel = str(self._current_entry().get("relative_path") or "")
        self._load_catalog()
        idx = self._find_index_by_relative_path(cur_rel)
        if idx >= 0:
            self._index = idx
        self._load_current()

    def _undo(self) -> None:
        if not self._boxes:
            return
        self._boxes.pop()
        self._view.set_boxes(self._boxes)
        self._set_dirty(True)

    def _clear(self) -> None:
        if not self._boxes:
            return
        self._boxes.clear()
        self._view.clear_boxes()
        self._set_dirty(True)

    def _delete_selected(self) -> None:
        indices = self._view.selected_box_indices()
        if not indices:
            return
        for i in sorted(indices, reverse=True):
            if 0 <= i < len(self._boxes):
                self._boxes.pop(i)
        self._view.set_boxes(self._boxes)
        self._set_dirty(True)

    def _save_internal(self) -> bool:
        entry = self._current_entry()
        rel_path = str(entry.get("relative_path") or "")
        if not self._slug or not rel_path:
            return False
        if not self._dirty:
            self._status.setText("No changes to save.")
            return True
        w, h = self._view.image_size()
        text = _boxes_to_yolo_text(self._boxes, w=w, h=h)
        enc_slug = urllib.parse.quote(self._slug, safe="")
        enc_path = urllib.parse.quote(rel_path, safe="")
        try:
            self._http_json("PUT", f"/database/{enc_slug}/label/{enc_path}", {"text": text})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._status.setText(f"Save failed: HTTP {exc.code} {detail[:120]}")
            return False
        except Exception as exc:
            self._status.setText(f"Save failed: {exc}")
            return False
        self._status.setText("Saved.")
        self._set_dirty(False)
        return True

    def _save_clicked(self) -> None:
        ok = self._save_internal()
        if not ok or self._dirty:
            return
        # Offer to return to the dataset panel (which refreshes on close) or keep editing.
        mb = QMessageBox(self)
        mb.setWindowTitle("Saved")
        mb.setIcon(QMessageBox.Icon.Information)
        mb.setText("Saved labels for this image.")
        mb.setInformativeText("Go back to view results, or continue editing?")
        continue_btn = mb.addButton("Continue editing", QMessageBox.ButtonRole.AcceptRole)
        back_btn = mb.addButton("Back and view results", QMessageBox.ButtonRole.DestructiveRole)
        mb.setDefaultButton(continue_btn)  # type: ignore[arg-type]
        mb.exec()
        if mb.clickedButton() == back_btn:
            self.accept()

    def _maybe_save(self) -> bool:
        if not self._dirty:
            return True
        ok = self._save_internal()
        return ok and not self._dirty

    def _step(self, delta: int) -> None:
        if not self._entries:
            return
        if not self._maybe_save():
            return
        self._index = int(_clamp(float(self._index + int(delta)), 0.0, float(len(self._entries) - 1)))
        self._load_current()
