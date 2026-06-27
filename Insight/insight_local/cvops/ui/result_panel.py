from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QBrush, QImage, QPixmap
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .path_actions import reveal_in_file_manager
from .run_artifacts_panel import RunArtifactsPanel
from .viewport_geometry import reference_layout_size
from .cvops_theme import cvops_qcolor, repolish
from .training_graph import TrainingGraphWidget
from .test_range_subroutine import (
    SubroutineControlsWidget,
    SubroutineImageOverlay,
    SubroutineSession,
    crop_qimage,
    offset_detections,
    qimage_to_bgr_ndarray,
)


def _pixmap_from_b64_any(b64: str) -> QPixmap:
    if not b64:
        return QPixmap()
    try:
        raw = base64.b64decode(b64)
    except Exception:
        return QPixmap()
    img = QImage()
    img.loadFromData(raw)
    return QPixmap.fromImage(img)


class ResultPanel(QWidget):
    """Displays job result with preview, summary, and final artifact paths."""

    flagRequested = pyqtSignal(object)
    activeContextChanged = pyqtSignal(bool)

    _IMAGE_MIN_WIDTH = 160
    _IMAGE_MIN_HEIGHT = 120
    # Compact preview thumbnail height: the wide result-image column was removed,
    # so the preview lives inside the subroutine column purely as a small canvas
    # for drawing [SUBROUTINE ROI] crops.
    _PREVIEW_MAX_HEIGHT = 240

    def __init__(
        self,
        *,
        base_url: str = "",
        http_get: Optional[Callable[[str], dict[str, Any]]] = None,
        http_get_text: Optional[Callable[[str], str]] = None,
        show_detection_table: bool = True,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._current_job_id = ""
        self._overlay_pixmap: Optional[QPixmap] = None
        # Clean (un-annotated) uploaded image kept so the subroutine can run its
        # own models on the full frame after the initial scenario-model result.
        self._subroutine_source_image: Optional[QImage] = None
        self._result_path = ""
        self._weights_path = ""
        self._feedback_context: dict[str, Any] = {}
        self._base_url = str(base_url or "").rstrip("/")
        self._http_get = http_get
        self._http_get_text = http_get_text
        self._active_context = False
        self._show_detection_table = bool(show_detection_table)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self._header = QLabel("No result selected.")
        self._header.setStyleSheet("font-size: 10px; font-weight: 600;")
        layout.addWidget(self._header)

        self._notice = QLabel("")
        self._notice.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._notice.setWordWrap(True)
        self._notice.setVisible(False)
        self._notice.setStyleSheet("font-size: 11px; font-weight: 700; padding: 2px 8px;")
        layout.addWidget(self._notice)

        # Small result preview — the canvas for drawing [SUBROUTINE ROI] crops.
        # The result-image preview is no longer shown on its own. The uploaded
        # image, its scenario-model detections, and any later subroutine-model
        # runs are all surfaced inside the subroutine panel's Result / Raw crop
        # areas. The preview widgets below stay constructed (apply_result and
        # the subroutine ROI overlay still reference them) but live off-screen
        # in the hidden container.
        self._left_wrap = QScrollArea()
        self._left_wrap.setWidgetResizable(True)
        self._left_wrap.setMaximumHeight(self._PREVIEW_MAX_HEIGHT)
        self._preview_host = QWidget()
        preview_layout = QVBoxLayout(self._preview_host)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(0)
        self._image_label = QLabel()
        self._image_label.setObjectName("overlayPreview")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(self._IMAGE_MIN_WIDTH, self._IMAGE_MIN_HEIGHT)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        preview_layout.addWidget(self._image_label)
        self._image_caption = QLabel("")
        self._image_caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_caption.setWordWrap(True)
        self._image_caption.setProperty("muted", True)
        self._image_caption.setVisible(False)
        preview_layout.addWidget(self._image_caption)
        self._subroutine_overlay = SubroutineImageOverlay(self._preview_host)
        self._subroutine_overlay.setVisible(False)
        self._subroutine_overlay.subroutineClicked.connect(self._on_subroutine_button)
        self._subroutine_overlay.roiCommitted.connect(self._on_subroutine_roi_committed)
        self._subroutine_overlay.cleared.connect(self._on_subroutine_overlay_cleared)
        self._subroutine_session = SubroutineSession(self)
        self._subroutine_roi_rect: Optional[QRect] = None
        self._left_wrap.setWidget(self._preview_host)

        # The ML-training graph + run-artifacts panel are still driven by
        # apply_result() for non-image (tabular/custom-code/LLM) runs, but the
        # compact Range layout no longer surfaces them — keep them
        # constructed inside a hidden, off-layout container so that code path
        # stays valid without occupying space.
        self._left_container = QWidget(self)
        self._left_container.setVisible(False)
        left_layout = QVBoxLayout(self._left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(5)
        left_layout.addWidget(self._left_wrap)

        self._ml_graph = TrainingGraphWidget()
        self._ml_graph.setMinimumHeight(160)
        self._ml_graph.setVisible(False)
        left_layout.addWidget(self._ml_graph)

        if self._http_get is not None and self._http_get_text is not None:
            self._artifacts_panel = RunArtifactsPanel(
                base_url=self._base_url,
                http_get=self._http_get,  # type: ignore[arg-type]
                http_get_text=self._http_get_text,  # type: ignore[arg-type]
            )
        else:
            self._artifacts_panel = None
        if self._artifacts_panel is not None:
            self._artifacts_panel.setVisible(False)
            left_layout.addWidget(self._artifacts_panel, stretch=1)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(5)

        self._signal_pill = QLabel("[SIGNAL] —")
        self._signal_pill.setObjectName("signalPill")
        self._signal_pill.setProperty("signal", "clear")
        rl.addWidget(self._signal_pill, alignment=Qt.AlignmentFlag.AlignLeft)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        rl.addWidget(self._summary)

        self._paths = QLabel("")
        self._paths.setWordWrap(True)
        rl.addWidget(self._paths)

        path_row = QHBoxLayout()
        self._reveal_result_btn = QPushButton("Reveal Result")
        self._reveal_result_btn.clicked.connect(self._reveal_result_path)
        path_row.addWidget(self._reveal_result_btn)
        self._reveal_weights_btn = QPushButton("Reveal Weights")
        self._reveal_weights_btn.clicked.connect(self._reveal_weights_path)
        path_row.addWidget(self._reveal_weights_btn)
        self._flag_weights_btn = QPushButton("[FLAG]")
        self._flag_weights_btn.setToolTip(
            "Save feedback against this Console run and its weights."
        )
        self._flag_weights_btn.clicked.connect(self._on_flag_weights)
        path_row.addWidget(self._flag_weights_btn)
        self._subroutine_roi_btn = QPushButton("[SUBROUTINE ROI]")
        self._subroutine_roi_btn.setCheckable(True)
        self._subroutine_roi_btn.setToolTip(
            "Drag a rectangle on the image preview to run a subroutine model on that crop."
        )
        self._subroutine_roi_btn.toggled.connect(self._on_subroutine_roi_mode)
        self._subroutine_roi_btn.setEnabled(False)
        # No standalone preview to draw on anymore — the subroutine now runs on
        # the whole uploaded image, so the ROI button is hidden.
        self._subroutine_roi_btn.setVisible(False)
        path_row.addWidget(self._subroutine_roi_btn)
        path_row.addStretch(1)
        rl.addLayout(path_row)

        self._subroutine_panel = SubroutineControlsWidget(http_get=self._http_get, parent=self)
        self._subroutine_panel.runRequested.connect(self._on_subroutine_run)
        self._subroutine_panel.dismissed.connect(self._on_subroutine_dismissed)
        rl.addWidget(self._subroutine_panel)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Label", "Confidence", "BBox", "Track"])
        self._table.setStyleSheet("QHeaderView::section { font-weight: 600; }")
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setVisible(self._show_detection_table)
        if self._show_detection_table:
            rl.addWidget(self._table, stretch=1)

        toggle_row = QHBoxLayout()
        self._toggle_raw = QPushButton("Show raw JSON")
        self._toggle_raw.setCheckable(True)
        self._toggle_raw.toggled.connect(self._on_toggle_raw)
        toggle_row.addWidget(self._toggle_raw)
        toggle_row.addStretch(1)
        rl.addLayout(toggle_row)

        self._raw = QTextEdit()
        self._raw.setObjectName("rawJson")
        self._raw.setReadOnly(True)
        self._raw.setVisible(False)
        self._raw.setMaximumHeight(220)
        rl.addWidget(self._raw)

        layout.addWidget(right, stretch=1)
        self._sync_responsive_geometry()

    def clear(self) -> None:
        self._current_job_id = ""
        self._overlay_pixmap = None
        self._subroutine_source_image = None
        self._header.setText("No result selected.")
        self._notice.clear()
        self._notice.setVisible(False)
        self._image_label.clear()
        self._image_label.setText("[NO IMAGE]")
        self._image_caption.clear()
        self._image_caption.setVisible(False)
        self._ml_graph.clear()
        self._ml_graph.setVisible(False)
        if self._artifacts_panel is not None:
            self._artifacts_panel.clear()
            self._artifacts_panel.setVisible(False)
        self._left_wrap.setVisible(True)
        self._signal_pill.setText("[SIGNAL] —")
        self._summary.setText("")
        self._paths.setText("")
        self._result_path = ""
        self._weights_path = ""
        self._feedback_context = {}
        self._reveal_result_btn.setEnabled(False)
        self._reveal_weights_btn.setEnabled(False)
        self._flag_weights_btn.setEnabled(False)
        self._subroutine_roi_btn.setEnabled(False)
        self._on_subroutine_dismissed()
        self._table.setRowCount(0)
        self._table.setColumnCount(4)
        self._table.setHorizontalHeaderLabels(["Label", "Confidence", "BBox", "Track"])
        self._raw.clear()
        self._set_active_context(False)

    def show_message(self, text: str) -> None:
        self.clear()
        msg = str(text or "").strip() or "No result selected."
        self._header.setText(msg)
        self._notice.clear()
        self._notice.setVisible(False)
        self._set_active_context(msg != "No result selected.")

    def select_job(self, job_id: str) -> None:
        """Fetch a job's result and render it. Used by the Ecosystem quick-nav."""
        jid = str(job_id or "").strip()
        if not jid:
            return
        if self._http_get is None:
            self.show_message(f"Job {jid} — result fetch unavailable")
            return
        try:
            payload = self._http_get(f"/jobs/{jid}/result")
        except Exception as exc:
            self.show_message(f"Job {jid} — fetch error: {exc}")
            return
        if not isinstance(payload, dict):
            self.show_message(f"Job {jid} — no result yet")
            return
        # Service returns either the bare result dict or {"result": {...}}.
        result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
        self.apply_result(jid, result)

    def apply_result(self, job_id: str, result: dict[str, Any]) -> None:
        self._on_subroutine_dismissed()
        self._subroutine_source_image = None
        self._current_job_id = job_id
        self._set_active_context(True)
        self._header.setText(f"Job {job_id}  |  {result.get('scenario', '')}")
        self._notice.clear()
        self._notice.setVisible(False)
        btype = str(result.get("backbone_type") or "yolo_detection")
        is_artifact_run = btype in {"torch_tabular", "custom_code", "llm_fine_tuning"}
        # For non-image ML runs, swap the left side to artifacts/charts by default.
        if is_artifact_run and self._artifacts_panel is not None:
            self._left_wrap.setVisible(False)
            self._ml_graph.setVisible(True)
            self._load_ml_training_graph(job_id)
            self._artifacts_panel.setVisible(True)
            try:
                self._artifacts_panel.load_job(job_id)
            except Exception:
                pass
        else:
            if self._artifacts_panel is not None:
                self._artifacts_panel.setVisible(False)
            self._ml_graph.setVisible(False)
            self._left_wrap.setVisible(True)

        if not is_artifact_run:
            overlay_b64 = str(result.get("overlay_image") or "")
            source_b64 = str(result.get("source_image_b64") or "")
            source_name = str(result.get("source_name") or "").strip()
            pix = QPixmap()
            if overlay_b64:
                pix = _pixmap_from_b64_any(overlay_b64)
            if pix.isNull() and source_b64:
                pix = _pixmap_from_b64_any(source_b64)
            # Clean source frame (prefer the un-annotated upload) is what later
            # subroutine-model runs operate on.
            source_pix = _pixmap_from_b64_any(source_b64) if source_b64 else QPixmap()
            if source_pix.isNull():
                source_pix = pix
            self._subroutine_source_image = (
                source_pix.toImage() if not source_pix.isNull() else None
            )
            if not pix.isNull():
                self._overlay_pixmap = pix
                self._render_overlay_pixmap()
                self._subroutine_roi_btn.setEnabled(True)
                self._subroutine_overlay.set_image_size(pix.width(), pix.height())
                self._image_caption.setText(source_name or "Uploaded image")
                self._image_caption.setVisible(True)
            elif overlay_b64 or source_b64:
                self._overlay_pixmap = None
                self._image_label.clear()
                self._image_label.setText("[INVALID IMAGE]")
                self._image_caption.clear()
                self._image_caption.setVisible(False)
                self._subroutine_roi_btn.setEnabled(False)
            else:
                self._overlay_pixmap = None
                self._image_label.clear()
                self._image_label.setText("[NO OVERLAY]")
                self._image_caption.clear()
                self._image_caption.setVisible(False)
                self._subroutine_roi_btn.setEnabled(False)
        else:
            self._overlay_pixmap = None
            self._image_caption.clear()
            self._image_caption.setVisible(False)
            self._subroutine_roi_btn.setEnabled(False)

        raw_result = result.get("raw") if isinstance(result.get("raw"), dict) else result
        signal = raw_result.get("signal") if isinstance(raw_result.get("signal"), dict) else {}
        flag = bool(signal.get("flag")) if signal else False
        summary = str(result.get("summary") or signal.get("summary") or "")
        self._signal_pill.setProperty("signal", "flagged" if flag else "clear")
        repolish(self._signal_pill)
        self._signal_pill.setText(f"[SIGNAL] {'FLAGGED' if flag else 'CLEAR'}")
        elapsed = result.get("elapsed_ms")
        elapsed_str = f"  |  {elapsed} ms" if elapsed is not None else ""
        err = str(result.get("error") or "")
        err_str = f"  |  error: {err}" if err else ""
        self._summary.setText(f"{summary}{elapsed_str}{err_str}")

        self._result_path = str(result.get("result_path") or "")
        self._weights_path = str(result.get("weights") or "")
        path_lines: list[str] = []
        if self._result_path:
            path_lines.append(f"run_dir: {self._result_path}")
        if self._weights_path:
            weight_name = Path(self._weights_path).name or self._weights_path
            path_lines.append(f"weights: {weight_name}")
        if btype == "torch_tabular":
            metrics = signal.get("metrics") if isinstance(signal, dict) else {}
            if isinstance(metrics, dict):
                task = str(metrics.get("task") or "")
                if task:
                    path_lines.append(f"task: {task}")
        self._paths.setText("\n".join(path_lines))
        self._reveal_result_btn.setEnabled(bool(self._result_path))
        self._reveal_weights_btn.setEnabled(bool(self._weights_path))
        self._flag_weights_btn.setEnabled(bool(self._weights_path))
        detections_obj = result.get("detections")
        if not isinstance(detections_obj, list):
            detections_obj = raw_result.get("detections") if isinstance(raw_result.get("detections"), list) else []
        self._feedback_context = {
            "source": "console_result",
            "job_id": str(job_id or ""),
            "scenario": str(result.get("scenario") or ""),
            "weights_path": self._weights_path,
            "result_path": self._result_path,
            "summary": self._summary.text(),
            "backbone_type": btype,
            "elapsed_ms": elapsed,
            "error": err,
            "signal": signal if isinstance(signal, dict) else {},
            "detection_count": len(detections_obj) if isinstance(detections_obj, list) else 0,
        }

        if is_artifact_run:
            # Show key metrics rather than detections.
            metrics = signal.get("metrics") if isinstance(signal, dict) else {}
            metrics = metrics if isinstance(metrics, dict) else {}
            show: list[tuple[str, Any]] = []
            for key in (
                "task",
                "final_val_metric",
                "val_metric",
                "best_val_loss",
                "num_features",
                "rows",
                "num_classes",
                "train_examples",
                "val_examples",
                "dry_run",
            ):
                if key in metrics:
                    show.append((key, metrics.get(key)))
            # Fallback: show whatever is present (but keep it short).
            if not show:
                for k in sorted(metrics.keys(), key=lambda s: str(s).lower())[:18]:
                    show.append((str(k), metrics.get(k)))

            self._table.setColumnCount(2)
            self._table.setHorizontalHeaderLabels(["Metric", "Value"])
            self._table.setRowCount(len(show))
            text_brush = QBrush(cvops_qcolor("text_signal"))
            for row, (k, v) in enumerate(show):
                k_item = QTableWidgetItem(str(k))
                k_item.setForeground(text_brush)
                v_item = QTableWidgetItem(str(v))
                v_item.setForeground(text_brush)
                self._table.setItem(row, 0, k_item)
                self._table.setItem(row, 1, v_item)
            self._table.resizeColumnsToContents()
        else:
            detections = detections_obj or []
            self._table.setColumnCount(4)
            self._table.setHorizontalHeaderLabels(["Label", "Confidence", "BBox", "Track"])
            self._table.setRowCount(len(detections))
            text_brush = QBrush(cvops_qcolor("text_signal"))
            for row, det in enumerate(detections):
                if not isinstance(det, dict):
                    continue
                label = str(det.get("label") or det.get("class") or "")
                conf = det.get("confidence", det.get("score"))
                try:
                    conf_str = f"{float(conf):.3f}" if conf is not None else ""
                except Exception:
                    conf_str = str(conf)
                bbox = det.get("bbox") or det.get("box") or []
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    try:
                        bbox_str = (
                            f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
                        )
                    except Exception:
                        bbox_str = str(bbox)
                else:
                    bbox_str = str(bbox)
                track = str(det.get("track_id", "") or "")
                for col, value in enumerate((label, conf_str, bbox_str, track)):
                    item = QTableWidgetItem(value)
                    item.setForeground(text_brush)
                    self._table.setItem(row, col, item)
            self._table.resizeColumnsToContents()
            if not detections and not err:
                self._notice.setText("No detections found in the uploaded image.")
                self._notice.setVisible(True)
            # Surface the upload + its scenario-model detections inside the
            # subroutine panel, and arm it so the user can then run other
            # (subroutine) models on the same image.
            self._feed_result_to_subroutine(detections)

        self._raw.setPlainText(json.dumps(result, indent=2, ensure_ascii=True, default=str))

    def has_active_context(self) -> bool:
        return bool(self._active_context)

    def _set_active_context(self, active: bool) -> None:
        active = bool(active)
        if self._active_context == active:
            return
        self._active_context = active
        self.activeContextChanged.emit(active)

    def refresh_theme_styles(self) -> None:
        text_brush = QBrush(cvops_qcolor("text_signal"))
        for row in range(self._table.rowCount()):
            for col in range(self._table.columnCount()):
                item = self._table.item(row, col)
                if item is not None:
                    item.setForeground(text_brush)

    def _load_ml_training_graph(self, job_id: str) -> None:
        if self._http_get is None:
            self._ml_graph.clear()
            return
        try:
            payload = self._http_get(f"/jobs/{job_id}/training_progress")
        except Exception:
            self._ml_graph.clear()
            return
        events = payload.get("epoch_events") if isinstance(payload, dict) else None
        if not isinstance(events, list) or not events:
            # Torch tabular runs may not emit /training_progress events; fall back to metrics.json history
            # if it exists in the run artifacts.
            if self._http_get_text is None:
                self._ml_graph.clear()
                return
            try:
                text = self._http_get_text(f"/jobs/{job_id}/artifacts/metrics.json")
                data = json.loads(text)
            except Exception:
                self._ml_graph.clear()
                return
            if not isinstance(data, dict):
                self._ml_graph.clear()
                return
            history = data.get("history")
            if not isinstance(history, list) or not history:
                self._ml_graph.clear()
                return
            points: list[dict[str, Any]] = []
            for row in history:
                if not isinstance(row, dict):
                    continue
                try:
                    ep = int(float(row.get("epoch") or 0)) - 1
                except Exception:
                    ep = len(points)
                point = {"event": "epoch", "epoch": ep}
                for k in ("train_loss", "val_loss", "val_mae", "val_acc", "map50"):
                    if k in row:
                        point[k] = row.get(k)
                points.append(point)
            total = len(points)
            for i, p in enumerate(points):
                p["epochs"] = total
                p["progress"] = (float(i + 1) / float(max(1, total))) * 100.0
            self._ml_graph.set_points(points)
            return
        points = [e for e in events if isinstance(e, dict)]
        if not points:
            self._ml_graph.clear()
            return
        self._ml_graph.set_points(points)

    def _on_toggle_raw(self, checked: bool) -> None:
        self._raw.setVisible(checked)
        self._toggle_raw.setText("Hide raw JSON" if checked else "Show raw JSON")

    def _reveal_result_path(self) -> None:
        if not self._result_path:
            return
        try:
            reveal_in_file_manager(self._result_path)
        except Exception as exc:
            self._paths.setText(f"{self._paths.text()}\nreveal failed: {exc}".strip())

    def _reveal_weights_path(self) -> None:
        if not self._weights_path:
            return
        try:
            reveal_in_file_manager(self._weights_path)
        except Exception as exc:
            self._paths.setText(f"{self._paths.text()}\nreveal failed: {exc}".strip())

    def _on_flag_weights(self) -> None:
        if not self._weights_path:
            return
        payload = dict(self._feedback_context)
        payload["source"] = payload.get("source") or "console_result"
        payload["weights_path"] = str(self._weights_path)
        payload["result_path"] = str(self._result_path or payload.get("result_path") or "")
        self.flagRequested.emit(payload)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_responsive_geometry()
        if self._overlay_pixmap is not None:
            self._render_overlay_pixmap()
        self._sync_subroutine_overlay_geometry()

    def _sync_subroutine_overlay_geometry(self) -> None:
        if not hasattr(self, "_subroutine_overlay"):
            return
        host = self._preview_host
        self._subroutine_overlay.setGeometry(0, 0, host.width(), host.height())
        self._subroutine_overlay.raise_()

    def _on_subroutine_roi_mode(self, checked: bool) -> None:
        if self._overlay_pixmap is None:
            self._subroutine_roi_btn.setChecked(False)
            return
        self._subroutine_overlay.set_select_mode(checked)
        self._subroutine_overlay.setVisible(checked or self._subroutine_roi_rect is not None)
        self._sync_subroutine_overlay_geometry()

    def _on_subroutine_roi_committed(self, rect: QRect) -> None:
        self._subroutine_roi_rect = QRect(rect)
        self._subroutine_roi_btn.setChecked(False)
        self._subroutine_overlay.set_select_mode(False)

    def _on_subroutine_button(self) -> None:
        if self._subroutine_roi_rect is None:
            return
        self._subroutine_panel.open_for_roi()

    def _on_subroutine_overlay_cleared(self) -> None:
        self._subroutine_roi_rect = None

    def _on_subroutine_dismissed(self) -> None:
        if getattr(self, "_subroutine_resetting", False):
            return
        self._subroutine_resetting = True
        try:
            self._subroutine_session.stop()
            self._subroutine_roi_rect = None
            self._subroutine_overlay.clear(emit=False)
            self._subroutine_overlay.setVisible(False)
            self._subroutine_panel.hide_panel()
            self._subroutine_roi_btn.blockSignals(True)
            self._subroutine_roi_btn.setChecked(False)
            self._subroutine_roi_btn.blockSignals(False)
            self._subroutine_overlay.set_select_mode(False)
        finally:
            self._subroutine_resetting = False

    def _source_qimage(self) -> Optional[QImage]:
        if self._overlay_pixmap is None or self._overlay_pixmap.isNull():
            return None
        return self._overlay_pixmap.toImage()

    @staticmethod
    def _registry_dets_to_subroutine(detections: list) -> list[dict[str, Any]]:
        """Map scenario-model result detections (bbox = [x1,y1,x2,y2]) into the
        flat x1/y1/x2/y2 dicts the subroutine panel renders and tabulates."""
        out: list[dict[str, Any]] = []
        for det in detections or []:
            if not isinstance(det, dict):
                continue
            bbox = det.get("bbox") or det.get("box")
            if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            else:
                x1, y1, x2, y2 = det.get("x1"), det.get("y1"), det.get("x2"), det.get("y2")
            if any(v is None for v in (x1, y1, x2, y2)):
                continue
            mapped: dict[str, Any] = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "label": str(det.get("label") or det.get("class") or ""),
                "conf": det.get("confidence", det.get("score", det.get("conf"))),
                "type": str(det.get("type") or "det"),
            }
            for k in ("keypoints", "mask_xy", "model_label", "track_id"):
                if det.get(k) is not None:
                    mapped[k] = det.get(k)
            out.append(mapped)
        return out

    def _feed_result_to_subroutine(self, detections: list) -> None:
        """Render the uploaded image + its scenario-model detections inside the
        subroutine panel and arm it to run further models on the whole frame."""
        img = self._subroutine_source_image
        if img is None or img.isNull():
            return
        # The whole uploaded frame is the working crop for subroutine runs.
        self._subroutine_roi_rect = QRect(0, 0, img.width(), img.height())
        self._subroutine_panel.show_results(
            crop_image=img,
            detections=self._registry_dets_to_subroutine(detections),
            frame_w=img.width(),
            frame_h=img.height(),
        )

    def _on_subroutine_run(self, model_path: str, _device: str) -> None:
        # Subroutine models run on the whole uploaded frame (the clean source),
        # not on the annotated overlay — so an initial registry result must
        # exist first.
        frame_img = self._subroutine_source_image
        if frame_img is None or frame_img.isNull():
            self._subroutine_panel.show_error(
                "Run a registry model on an image first, then run subroutine models on it."
            )
            return
        rect = self._subroutine_roi_rect or QRect(0, 0, frame_img.width(), frame_img.height())
        crop = crop_qimage(frame_img, rect)
        if crop.isNull():
            self._subroutine_panel.show_error("Invalid image for subroutine.")
            return
        try:
            crop_bgr = qimage_to_bgr_ndarray(crop)
        except Exception as exc:
            self._subroutine_panel.show_error(f"Crop conversion failed: {exc}")
            return
        self._subroutine_panel.show_running()
        ox, oy = rect.x(), rect.y()

        def _done(detections: list) -> None:
            full = offset_detections(detections, ox, oy)
            fw, fh = frame_img.width(), frame_img.height()
            for det in full:
                det["frame_w"] = fw
                det["frame_h"] = fh
            self._subroutine_overlay.set_detection_boxes(full)
            self._subroutine_panel.show_results(
                crop_image=crop,
                detections=detections,
                frame_w=crop.width(),
                frame_h=crop.height(),
            )

        def _fail(msg: str) -> None:
            self._subroutine_panel.show_error(msg)
            self._subroutine_overlay.set_detection_boxes([])

        self._subroutine_session.start(
            model_path=model_path,
            device="",
            frame_bgr=crop_bgr,
            on_finished=_done,
            on_failed=_fail,
        )

    def _render_overlay_pixmap(self) -> None:
        if self._overlay_pixmap is None:
            return
        scaled = self._overlay_pixmap.scaled(
            max(self._image_label.width(), self._IMAGE_MIN_WIDTH),
            max(self._image_label.height(), self._IMAGE_MIN_HEIGHT),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def _sync_responsive_geometry(self) -> None:
        _, panel_h = reference_layout_size(self)
        raw_max = int(max(140, min(360, panel_h * 0.3)))
        self._raw.setMaximumHeight(raw_max)

    def refresh_subroutine_models(self) -> None:
        if hasattr(self, "_subroutine_panel"):
            self._subroutine_panel.refresh_models()

    def refresh_responsive_layout(self) -> None:
        self._sync_responsive_geometry()
        if self._overlay_pixmap is not None:
            self._render_overlay_pixmap()
        self._sync_subroutine_overlay_geometry()
