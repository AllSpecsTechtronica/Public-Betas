from __future__ import annotations

import json
import math
import re
import shutil
import sys
import time
from array import array
from collections import deque
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRect,
    QRectF,
    QSize,
    QSizeF,
    Qt,
    QProcess,
    QThread,
    QTimer,
    QUrl,
    pyqtSignal,
    pyqtSlot,
)
from PyQt6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPen, QShortcut
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer, QVideoFrame
from PyQt6.QtMultimediaWidgets import QGraphicsVideoItem

from .. import corrections_store
from ..libav_log_capture import (
    install_libav_log_capture,
    recent_libav_corruption_lines,
)
from .correction_dialog import CorrectionDialog
from PyQt6.QtWidgets import (
    QBoxLayout,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QMessageBox,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...config import heatmap_category
from ...filtering import matches_detection_view
from ...ui.theme import text_css, text_qcolor, theme_hex, theme_rgba
from ..detection_backends import (
    YuNetFaceDetectorBackend,
    extract_yolo_detections,
    is_supported_video_test_model,
    is_yunet_face_detector_model,
)
from .audio_timeline import AudioTimeline
from .test_range_subroutine import (
    SubroutineBoxOverlay,
    SubroutineBlinkHighlight,
    SubroutineControlsWidget,
    SubroutineRoiController,
    SubroutineSession,
    collect_video_test_models,
    crop_qimage,
    offset_detections,
    qimage_to_bgr_ndarray,
)
from .model_converter_panel import ModelConverterPanel
from .video_index_export_dialog import VideoIndexExportDialog


HttpCall = Callable[..., object]


_ASSETS_VIDEOS = Path(__file__).resolve().parents[4] / "assets" / "videos"


def _filter_scoped_detections(
    detections: list[dict],
    *,
    label_filter: str = "",
    categories: set[str] | tuple[str, ...] | list[str] = (),
) -> list[dict]:
    query = str(label_filter or "").strip().lower()
    active_categories = {
        str(item).strip().lower()
        for item in categories
        if str(item).strip()
    }
    if not query and not active_categories:
        return list(detections or [])

    filtered: list[dict] = []
    for det in detections or []:
        label = str(det.get("label") or "")
        category = heatmap_category(label)
        if matches_detection_view(query, active_categories, label, category):
            filtered.append(det)
    return filtered


def _surface_frame_css(*, raised: bool = False) -> str:
    fill = theme_rgba("panel", 0.44 if raised else 0.34)
    border = theme_rgba("accent_dark", 0.20 if raised else 0.15)
    return (
        "QFrame {"
        f" border: 1px solid {border};"
        f" background: {fill};"
        " border-radius: 0px;"
        " }"
    )


def _section_caption_css() -> str:
    return (
        f"color: {theme_rgba('accent_dark', 0.86)};"
        " font-size: 10px;"
        " font-weight: 700;"
        " letter-spacing: 0.05em;"
    )


def _section_hint_css() -> str:
    return f"font-size: 10px; color: {text_css(0.84)};"


def _theme_qcolor(role: str, alpha: int = 255) -> QColor:
    color = QColor(theme_hex(role))
    color.setAlpha(max(0, min(255, int(alpha))))
    return color


def _detect_devices() -> list[tuple[str, str]]:
    """Return [(label, device_id)] for devices that torch can actually use.

    Always includes Auto and CPU; adds CUDA / MPS only when available so the
    UI never offers a path that will throw at predict() time.
    """
    options: list[tuple[str, str]] = [("Auto", "")]
    try:
        import torch  # type: ignore
    except Exception:
        options.append(("CPU", "cpu"))
        return options
    try:
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            for i in range(count):
                name = torch.cuda.get_device_name(i)
                options.append((f"CUDA:{i} ({name})", f"cuda:{i}"))
    except Exception:
        pass
    try:
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            options.append(("MPS (Apple GPU)", "mps"))
    except Exception:
        pass
    options.append(("CPU", "cpu"))
    return options


def _model_supports_to(model_path: str) -> bool:
    """Whether YOLO(...).to(device) is meaningful for this model format.

    Only torch-backed formats (.pt, .torchscript) carry tensors that can be
    moved to a device. Exported runtimes (ONNX, CoreML, TensorRT, TFLite)
    are pinned to their provider/runtime at export time; calling .to()
    raises. For those, device selection (when applicable) flows through
    predict(device=...).
    """
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


def _format_ms(ms: int) -> str:
    if ms < 0:
        ms = 0
    total = ms // 1000
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _format_secs(secs: float) -> str:
    return _format_ms(int(max(0.0, secs) * 1000))


class DetectionTimeline(QWidget):
    """Stacked-bar timeline that shows where detections were observed.

    Each tick is keyed by absolute video time (ms). Bars are coloured per
    class label. The timeline reflows when the duration is set and rescales
    automatically.
    """

    PALETTE = [
        "#ff6b6b", "#4dabf7", "#69db7c", "#ffd43b", "#cc5de8",
        "#ffa94d", "#3bc9db", "#f783ac", "#a9e34b", "#9775fa",
    ]
    _BASE_HEIGHT = 56
    _BASE_HINT_HEIGHT = 64
    _BASE_BOTTOM_H = 14

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._layout_scale: float = 1.0
        self._apply_layout_scale()
        self._duration_ms: int = 0
        self._cursor_ms: int = 0
        self._events: list[tuple[int, str]] = []
        self._class_colours: dict[str, QColor] = {}
        # Spans (start_ms, end_ms) of footage that failed to decode.
        self._corrupt_ranges: list[tuple[int, int]] = []

    def _scaled_px(self, px: int, *, minimum: int = 1) -> int:
        return max(minimum, int(round(px * self._layout_scale)))

    def _apply_layout_scale(self) -> None:
        height = self._scaled_px(self._BASE_HEIGHT, minimum=28)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.updateGeometry()

    def set_visual_scale(self, scale: float) -> None:
        self._layout_scale = max(0.5, min(1.5, float(scale)))
        self._apply_layout_scale()
        self.update()

    def set_duration(self, duration_ms: int) -> None:
        self._duration_ms = max(0, int(duration_ms))
        self.update()

    def set_cursor(self, position_ms: int) -> None:
        self._cursor_ms = max(0, int(position_ms))
        self.update()

    def clear_events(self) -> None:
        self._events.clear()
        self._class_colours.clear()
        self._corrupt_ranges.clear()
        self.update()

    def add_corrupt_ranges(self, ranges: list) -> None:
        """Record decode-failure spans. Deferred repaint (see add_event note)."""
        for item in ranges:
            try:
                start_ms = max(0, int(item[0]))
                end_ms = max(start_ms, int(item[1]))
            except Exception:
                continue
            self._corrupt_ranges.append((start_ms, end_ms))
        self._corrupt_ranges = self._merge_ranges(self._corrupt_ranges)

    def corrupt_ranges(self) -> list[tuple[int, int]]:
        return list(self._corrupt_ranges)

    @staticmethod
    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not ranges:
            return []
        ordered = sorted(ranges)
        merged: list[tuple[int, int]] = [ordered[0]]
        for start, end in ordered[1:]:
            last_start, last_end = merged[-1]
            # Merge touching/overlapping spans (small gap tolerance bridges
            # frame-by-frame failures sampled a few ms apart).
            if start <= last_end + 250:
                merged[-1] = (last_start, max(last_end, end))
            else:
                merged.append((start, end))
        return merged

    def add_event(self, position_ms: int, label: str = "") -> None:
        # NOTE: does not call self.update(). Callers should batch many adds
        # and trigger a single repaint on a coalesced cadence — repainting
        # this widget is O(events) and a per-frame update() during indexing
        # saturates the GUI thread once events accumulate.
        self._events.append((max(0, int(position_ms)), label or ""))
        if label and label not in self._class_colours:
            colour = QColor(self.PALETTE[len(self._class_colours) % len(self.PALETTE)])
            self._class_colours[label] = colour

    def add_events(self, position_ms: int, labels: list[str]) -> None:
        # See note on add_event re: deferred repaint.
        ts = max(0, int(position_ms))
        for label in labels:
            self._events.append((ts, label or ""))
            if label and label not in self._class_colours:
                colour = QColor(
                    self.PALETTE[len(self._class_colours) % len(self.PALETTE)]
                )
                self._class_colours[label] = colour

    def event_count(self) -> int:
        return len(self._events)

    def labels(self) -> list[str]:
        return list(self._class_colours.keys())

    def colour_for(self, label: str) -> QColor:
        return self._class_colours.get(label, _theme_qcolor("strip_soft"))

    def sizeHint(self) -> QSize:
        return QSize(400, self._scaled_px(self._BASE_HINT_HEIGHT, minimum=32))

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.rect()
        font = QFont(painter.font())
        font.setPixelSize(self._scaled_px(10, minimum=8))
        painter.setFont(font)
        painter.fillRect(rect, _theme_qcolor("input_fill"))

        track_top = rect.top() + self._scaled_px(4, minimum=3)
        track_bottom = rect.bottom() - self._scaled_px(self._BASE_BOTTOM_H, minimum=8)
        track_h = max(4, track_bottom - track_top)

        painter.setPen(QPen(_theme_qcolor("hover"), 1))
        painter.drawRect(rect.left(), track_top, rect.width() - 1, track_h)

        if self._duration_ms <= 0:
            painter.setPen(QPen(text_qcolor(0.58), 1))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No video loaded.")
            return

        width = max(1, rect.width())

        # Corrupt spans first, as a translucent red band under the detection
        # ticks, so good-footage detections still read clearly on top.
        if self._corrupt_ranges:
            corrupt_fill = QColor(255, 70, 70, 70)
            corrupt_edge = QColor(255, 90, 90, 200)
            for start_ms, end_ms in self._corrupt_ranges:
                x1 = int((start_ms / self._duration_ms) * width)
                x2 = int((end_ms / self._duration_ms) * width)
                span = max(2, x2 - x1)
                painter.fillRect(x1, track_top + 1, span, track_h - 2, corrupt_fill)
                painter.setPen(QPen(corrupt_edge, 1))
                painter.drawLine(x1, track_top + 1, x1, track_bottom - 1)
                painter.drawLine(x1 + span, track_top + 1, x1 + span, track_bottom - 1)

        for ts, label in self._events:
            x = int((ts / self._duration_ms) * width)
            colour = self._class_colours.get(label, _theme_qcolor("strip_soft"))
            painter.setPen(QPen(colour, 2))
            painter.drawLine(x, track_top + 1, x, track_bottom - 1)

        if 0 <= self._cursor_ms <= self._duration_ms:
            cx = int((self._cursor_ms / self._duration_ms) * width)
            painter.setPen(QPen(_theme_qcolor("privacy_warn"), 2))
            painter.drawLine(cx, rect.top(), cx, rect.bottom())

        painter.setPen(QPen(text_qcolor(0.58), 1))
        ticks = 6
        for i in range(ticks + 1):
            tx = int(i * (width - 1) / ticks)
            ts = int(i * self._duration_ms / ticks)
            painter.drawLine(
                tx,
                track_bottom,
                tx,
                track_bottom + self._scaled_px(4, minimum=3),
            )
            painter.drawText(
                tx + 2,
                rect.bottom(),
                _format_ms(ts),
            )


class _LegendWrapWidget(QWidget):
    """Detection summary + wrapped class chips."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._specs: list[tuple[str, str]] = []
        self._chip_widgets: list[QLabel] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        self._summary = QLabel("0 detections.")
        self._summary.setStyleSheet(_section_hint_css())
        self._summary.setWordWrap(True)
        outer.addWidget(self._summary)

        self._chips_host = QWidget()
        self._chips_grid = QGridLayout(self._chips_host)
        self._chips_grid.setContentsMargins(0, 0, 0, 0)
        self._chips_grid.setHorizontalSpacing(6)
        self._chips_grid.setVerticalSpacing(6)
        outer.addWidget(self._chips_host)
        self._chips_host.setVisible(False)

    def refresh_theme_styles(self) -> None:
        self._summary.setStyleSheet(_section_hint_css())
        for chip in self._chip_widgets:
            chip.update()

    def set_entries(self, count: int, entries: list[tuple[str, QColor]]) -> None:
        total = max(0, int(count))
        if not entries:
            self._summary.setText(f"{total:,} detections.")
            self._chips_host.setVisible(False)
            self._clear_chips()
            self._specs = []
            return

        self._summary.setText(f"{total:,} detections across {len(entries)} labels.")
        specs = [(label, colour.name()) for label, colour in entries]
        self._chips_host.setVisible(True)
        if specs != self._specs:
            self._specs = specs
            self._rebuild_chips()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._specs:
            self._reflow_chips()

    def _clear_chips(self) -> None:
        while self._chips_grid.count():
            item = self._chips_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._chip_widgets = []

    def _rebuild_chips(self) -> None:
        self._clear_chips()
        for label, color_name in self._specs:
            color = QColor(color_name)
            bg = f"rgba({color.red()},{color.green()},{color.blue()},0.14)"
            border = f"rgba({color.red()},{color.green()},{color.blue()},0.44)"
            chip = QLabel(label)
            chip.setStyleSheet(
                "QLabel {"
                f" background: {bg};"
                f" border: 1px solid {border};"
                f" border-left: 3px solid {color_name};"
                " padding: 3px 8px 3px 7px;"
                " font-size: 10px;"
                " font-weight: 600;"
                " }"
            )
            self._chip_widgets.append(chip)
        self._reflow_chips()

    def _reflow_chips(self) -> None:
        while self._chips_grid.count():
            self._chips_grid.takeAt(0)
        if not self._chip_widgets:
            return
        available = max(240, self.width() - 4)
        row = 0
        col = 0
        line_width = 0
        spacing = self._chips_grid.horizontalSpacing()
        for chip in self._chip_widgets:
            hint = chip.sizeHint().width()
            needed = hint if line_width == 0 else line_width + spacing + hint
            if line_width > 0 and needed > available:
                row += 1
                col = 0
                line_width = 0
            self._chips_grid.addWidget(chip, row, col)
            line_width = hint if line_width == 0 else line_width + spacing + hint
            col += 1


class _DetectorWorker(QObject):
    """Runs a CV model over a video file in a background thread.

    Emits one batch per sampled frame. A status signal reports milestones
    (loading model, scanning) so the UI never goes silent during the cold
    YOLO load.
    """

    status = pyqtSignal(str)
    progress = pyqtSignal(int, int)            # frame_index, total_frames
    # Per-frame detections kept for compatibility, but the index-mode UI
    # subscribes to the batched signal below — emitting once per frame at
    # 14-20Hz floods the GUI event queue and starves UI input.
    frame_detections = pyqtSignal(int, list)   # timestamp_ms, list[dict]
    frame_detections_batch = pyqtSignal(list)  # [(ts_ms, [dets]), ...]
    finished = pyqtSignal(int)                 # total events
    failed = pyqtSignal(str)

    # Worker emits at most one progress / detections-batch per this interval
    # of wall time. 100ms = 10Hz: small enough to feel live, sparse enough
    # that the GUI thread keeps up with UI events between dispatches.
    _EMIT_INTERVAL_S = 0.1

    def __init__(self) -> None:
        super().__init__()
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(
        self,
        video_path: str,
        model_path: str,
        sample_every: int,
        device: str = "",
    ) -> None:
        self._cancel = False
        try:
            import cv2  # type: ignore
        except Exception as exc:
            self.failed.emit(f"opencv unavailable: {exc}")
            return

        resolved_device = device.strip() or _resolve_auto_device()
        face_backend: YuNetFaceDetectorBackend | None = None
        model = None
        supports_to = False
        if is_yunet_face_detector_model(model_path):
            self.status.emit(f"Loading face detector {Path(model_path).name}...")
            try:
                face_backend = YuNetFaceDetectorBackend(model_path)
            except Exception as exc:
                self.failed.emit(f"face detector load failed: {exc}")
                return
        else:
            try:
                from ultralytics import YOLO  # type: ignore
            except Exception as exc:
                self.failed.emit(f"ultralytics unavailable: {exc}")
                return
            self.status.emit(
                f"Loading model {Path(model_path).name} on {resolved_device}..."
            )
            try:
                model = YOLO(model_path)
            except Exception as exc:
                self.failed.emit(f"model load failed: {exc}")
                return
            supports_to = _model_supports_to(model_path)
            if supports_to:
                try:
                    model.to(resolved_device)
                except Exception as exc:
                    self.failed.emit(
                        f"could not move model to {resolved_device}: {exc} "
                        f"(falling back requires re-running with CPU)"
                    )
                    return
        if self._cancel:
            self.finished.emit(0)
            return

        self.status.emit("Opening video...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.failed.emit(f"cannot open video: {video_path}")
            return

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(sample_every))
            count = 0
            frame_idx = 0
            self.status.emit(
                f"Scanning {total} frames @ {fps:.1f} fps (sample every {step})..."
            )
            self.progress.emit(0, total)
            # Local buffers + last-emit clocks. We dispatch to the GUI thread
            # at most once every _EMIT_INTERVAL_S regardless of frame rate so
            # the GUI event queue can drain mouse/keyboard events between
            # batches. Anything still buffered is flushed in the finally arm.
            batch_buf: list[tuple[int, list]] = []
            last_batch_emit = time.perf_counter()
            last_progress_emit = 0.0
            while True:
                if self._cancel:
                    break
                if step > 1:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    break
                ts_ms = int((frame_idx / fps) * 1000.0) if fps > 0 else 0
                try:
                    if face_backend is not None:
                        detections = face_backend.predict(frame)
                    else:
                        predict_kwargs = {"verbose": False}
                        if supports_to:
                            predict_kwargs["device"] = resolved_device
                        results = model.predict(frame, **predict_kwargs)
                        fh, fw = frame.shape[:2]
                        detections = extract_yolo_detections(results, fw, fh)
                except Exception as exc:
                    self.failed.emit(f"inference failed at frame {frame_idx}: {exc}")
                    return
                if detections:
                    batch_buf.append((ts_ms, detections))
                    count += len(detections)

                now = time.perf_counter()
                if batch_buf and (now - last_batch_emit) >= self._EMIT_INTERVAL_S:
                    self.frame_detections_batch.emit(batch_buf)
                    batch_buf = []
                    last_batch_emit = now
                if (now - last_progress_emit) >= self._EMIT_INTERVAL_S:
                    self.progress.emit(frame_idx, total)
                    last_progress_emit = now

                frame_idx += step
                if total and frame_idx >= total:
                    break
            # Final drain so no detections are lost on completion or cancel.
            if batch_buf:
                self.frame_detections_batch.emit(batch_buf)
            self.progress.emit(frame_idx, total)
            self.finished.emit(count)
        finally:
            cap.release()


class _LiveDetectorWorker(QObject):
    """Long-lived worker that holds an open YOLO model and cv2 capture, then
    answers single-frame inference requests dispatched from the UI as the user
    scrubs or plays the video. Re-entrancy is handled by `frame_done`: the UI
    only re-arms a request after the previous one completes."""

    ready = pyqtSignal()
    failed = pyqtSignal(str)
    frame_detections = pyqtSignal(int, list)  # ts_ms, list[str]
    frame_done = pyqtSignal(int)               # ts_ms (always emitted)

    def __init__(self) -> None:
        super().__init__()
        self._cap = None
        self._model = None
        self._face_backend: YuNetFaceDetectorBackend | None = None
        self._device = "cpu"
        self._supports_to = True
        self._fps = 30.0
        self._current_frame_idx = -1

    def _read_frame_at(self, target_frame_idx: int):
        cap = self._cap
        if cap is None:
            return (False, None)
        try:
            import cv2  # type: ignore
        except Exception:
            cv2 = None

        target = max(0, int(target_frame_idx))
        current = int(self._current_frame_idx)

        # Normal playback advances by a handful of frames between requests.
        # In that case, sequential grabs are much cheaper than a fresh seek.
        can_step_forward = current >= 0 and target > current and (target - current) <= 12
        if can_step_forward:
            while (self._current_frame_idx + 1) < target:
                if not cap.grab():
                    return (False, None)
                self._current_frame_idx += 1
        else:
            if cv2 is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            else:
                cap.set(1, target)
            self._current_frame_idx = target - 1

        ok, frame = cap.read()
        if ok:
            self._current_frame_idx += 1
        return (ok, frame)

    @pyqtSlot(str, str, str)
    def open_session(self, video_path: str, model_path: str, device: str) -> None:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            self.failed.emit(f"opencv unavailable: {exc}")
            return

        resolved = device.strip() or _resolve_auto_device()
        supports_to = False
        model = None
        face_backend: YuNetFaceDetectorBackend | None = None
        if is_yunet_face_detector_model(model_path):
            try:
                face_backend = YuNetFaceDetectorBackend(model_path)
            except Exception as exc:
                self.failed.emit(f"face detector load failed: {exc}")
                return
        else:
            try:
                from ultralytics import YOLO  # type: ignore
            except Exception as exc:
                self.failed.emit(f"ultralytics unavailable: {exc}")
                return
            supports_to = _model_supports_to(model_path)
            try:
                model = YOLO(model_path)
                if supports_to:
                    model.to(resolved)
            except Exception as exc:
                self.failed.emit(f"model load failed: {exc}")
                return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.failed.emit(f"cannot open video: {video_path}")
            return

        self._model = model
        self._face_backend = face_backend
        self._device = resolved
        self._supports_to = supports_to
        self._cap = cap
        self._fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._current_frame_idx = -1
        self.ready.emit()

    @pyqtSlot(int)
    def infer_at(self, ts_ms: int) -> None:
        cap = self._cap
        model = self._model
        if cap is None or (model is None and self._face_backend is None):
            self.frame_done.emit(int(ts_ms))
            return
        try:
            frame_idx = max(0, int((ts_ms / 1000.0) * self._fps))
            ok, frame = self._read_frame_at(frame_idx)
            if not ok:
                self.frame_done.emit(int(ts_ms))
                return
            if self._face_backend is not None:
                detections = self._face_backend.predict(frame)
            else:
                predict_kwargs = {"verbose": False}
                if self._supports_to:
                    predict_kwargs["device"] = self._device
                results = model.predict(frame, **predict_kwargs)
                fh, fw = frame.shape[:2]
                detections = extract_yolo_detections(results, fw, fh)
            # Always emit so the UI can clear stale boxes on frames with nothing.
            self.frame_detections.emit(int(ts_ms), detections)
        except Exception as exc:
            self.failed.emit(f"live inference failed: {exc}")
        finally:
            self.frame_done.emit(int(ts_ms))

    @pyqtSlot()
    def close_session(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self._model = None
        self._face_backend = None
        self._current_frame_idx = -1


class _BoundingBoxItem(QGraphicsItem):
    """Graphics item that paints YOLO boxes over a QGraphicsVideoItem.

    Sized and positioned to mirror the video item, so detection coords map
    directly into the displayed video region. Drawn in the same scene as the
    video, which avoids the macOS native-layer occlusion problem that bit
    QVideoWidget + sibling-widget overlays.
    """

    PALETTE = [
        "#ff6b6b", "#4dabf7", "#69db7c", "#ffd43b", "#cc5de8",
        "#ffa94d", "#3bc9db", "#f783ac", "#a9e34b", "#9775fa",
    ]

    def __init__(self, video_item: QGraphicsVideoItem) -> None:
        super().__init__()
        self._video_item = video_item
        self._boxes: list[dict] = []
        self._colors: dict[str, QColor] = {}
        self._enabled: bool = True
        self.setZValue(10.0)
        self.setAcceptedMouseButtons(Qt.MouseButton.NoButton)

    def sync_to_video(self) -> None:
        self.prepareGeometryChange()
        self.setPos(self._video_item.pos())
        self.update()

    def boundingRect(self) -> QRectF:  # type: ignore[override]
        size = self._video_item.size()
        return QRectF(0.0, 0.0, size.width(), size.height())

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        self.setVisible(self._enabled)
        if self._enabled:
            self.update()

    def set_boxes(self, boxes: list[dict]) -> None:
        self._boxes = list(boxes or [])
        for det in self._boxes:
            label = det.get("label", "")
            if label and label not in self._colors:
                idx = len(self._colors) % len(self.PALETTE)
                self._colors[label] = QColor(self.PALETTE[idx])
        self.update()

    def clear(self) -> None:
        self._boxes = []
        self.update()

    def paint(self, painter, _option, _widget=None) -> None:  # type: ignore[override]
        if not self._enabled or not self._boxes:
            return
        rect = self.boundingRect()
        widget_w = rect.width()
        widget_h = rect.height()
        if widget_w <= 0 or widget_h <= 0:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for det in self._boxes:
            fw = int(det.get("frame_w") or 0)
            fh = int(det.get("frame_h") or 0)
            if fw <= 0 or fh <= 0:
                continue
            scale_x = widget_w / fw
            scale_y = widget_h / fh
            x1 = det["x1"] * scale_x
            y1 = det["y1"] * scale_y
            x2 = det["x2"] * scale_x
            y2 = det["y2"] * scale_y

            label = det.get("label", "")
            colour = self._colors.get(label, _theme_qcolor("strip_soft"))
            painter.setPen(QPen(colour, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

            conf = det.get("conf", 0.0)
            text = f"{label} {conf:.2f}" if label else f"{conf:.2f}"
            metrics = painter.fontMetrics()
            text_w = metrics.horizontalAdvance(text) + 8
            text_h = metrics.height() + 2
            tag_y = max(0.0, y1 - text_h)
            painter.fillRect(QRectF(x1, tag_y, text_w, text_h), colour)
            painter.setPen(QPen(QColor("#000000")))
            painter.drawText(QPointF(x1 + 4, tag_y + metrics.ascent() + 1), text)


class _TimedGraphicsView(QGraphicsView):
    """QGraphicsView that records paintEvent durations for a HUD readout.

    Used to tell paint-bound lag (slow rasterization, many overlay items)
    apart from inference-bound lag (model latency, signal queueing).
    """

    def __init__(self, scene: QGraphicsScene, parent: Optional[QWidget] = None) -> None:
        super().__init__(scene, parent)
        self._paint_ms: deque[float] = deque(maxlen=120)
        self._paint_clocks: deque[float] = deque(maxlen=120)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        t0 = time.perf_counter()
        super().paintEvent(event)
        now = time.perf_counter()
        self._paint_ms.append((now - t0) * 1000.0)
        self._paint_clocks.append(now)

    def paint_stats(self) -> tuple[float, float]:
        """Return (avg_paint_ms, fps) over the rolling window."""
        if not self._paint_ms:
            return (0.0, 0.0)
        avg = sum(self._paint_ms) / len(self._paint_ms)
        if len(self._paint_clocks) >= 2:
            span = self._paint_clocks[-1] - self._paint_clocks[0]
            fps = (len(self._paint_clocks) - 1) / span if span > 0 else 0.0
        else:
            fps = 0.0
        return avg, fps


class VideoTestPanel(QWidget):
    """Basic video player tuned for testing CV models against pre-recorded clips.

    Layout (top to bottom):
      - Toolbar: open file, model picker, sample rate, run/cancel
      - Video display (QGraphicsView)
      - Playback row (transport controls, seek timeline, speed)
      - Audio detection timeline strip
      - Detection timeline strip
      - Detection controls
      - Status line
    """

    errorRaised = pyqtSignal(str)

    # Signals fired into the live worker thread (queued connections).
    _liveOpenRequested = pyqtSignal(str, str, str)
    _liveInferRequested = pyqtSignal(int)
    _liveCloseRequested = pyqtSignal()

    # Tick fast and rely on `_live_busy` to gate. With Δ ~50ms, this lets
    # inferences fire back-to-back instead of idling 150ms per cycle. If the
    # model genuinely takes longer than the tick, the gate just skips ticks.
    _LIVE_TICK_MS = 33  # ~30Hz timer; effective rate is min(timer, model)
    _LIVE_FACE_SAMPLE_CAP = 2  # keep face preview fluid even if index sampling is coarse
    _TIMELINE_VISUAL_SCALE = 0.6
    _INDEX_CATEGORY_CHOICES = (
        ("people", "People"),
        ("animals", "Animals"),
        ("tech", "Tech"),
        ("objects", "Objects"),
    )

    def __init__(
        self,
        *,
        http_get: Optional[HttpCall] = None,
        http_post: Optional[HttpCall] = None,
        http_delete: Optional[HttpCall] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._http_get = http_get
        self._http_post = http_post
        self._http_delete = http_delete
        self._video_path: Optional[Path] = None
        self._duration_ms: int = 0
        self._user_seeking: bool = False

        # Index-mode (full-pass) worker. The current path uses QProcess so
        # heavy YOLO import/inference cannot starve the Qt event loop via the
        # Python GIL. The older QThread fields are kept for defensive teardown
        # of stale sessions from prior builds.
        self._worker_thread: Optional[QThread] = None
        self._worker: Optional[_DetectorWorker] = None
        self._index_process: Optional[QProcess] = None
        self._index_stdout_buffer: str = ""
        self._index_stderr_buffer: str = ""
        self._index_reported_total_events: Optional[int] = None
        self._index_reported_failure: str = ""
        self._index_process_error_message: str = ""
        self._index_ui_token: int = 0
        # Indexing run-state for progress reporting. Reset each run.
        self._index_start_wall: float = 0.0
        self._index_total_detections: int = 0
        self._index_cancel_requested: bool = False
        # Class names observed in detections so far. Fed into the correction
        # dialog so the picker pre-fills with whatever the model has been
        # producing this session, even if we don't have model.names handy.
        self._observed_classes: list[str] = []
        # Coalesced repaint: detector signals fire per-frame; the timeline +
        # legend repaints them at this rate so we don't lock the GUI thread.
        self._timeline_dirty: bool = False
        self._timeline_refresh_timer = QTimer(self)
        self._timeline_refresh_timer.setInterval(100)
        self._timeline_refresh_timer.timeout.connect(self._flush_timeline_repaint)
        self._timeline_refresh_timer.timeout.connect(self._poll_playback_corruption)
        self._timeline_refresh_timer.start()

        # Surface FFmpeg/libav decode-corruption messages that QMediaPlayer
        # swallows (the "feed goes black" case). install is best-effort.
        install_libav_log_capture()
        self._corruption_dialog_at: float = 0.0
        self._corruption_box: Optional[QMessageBox] = None
        self._CORRUPTION_DIALOG_COOLDOWN_S = 12.0

        # Live-mode (single-frame on demand) worker
        self._live_thread: Optional[QThread] = None
        self._live_worker: Optional[_LiveDetectorWorker] = None
        self._live_active: bool = False
        self._live_busy: bool = False
        self._live_last_ts: int = -1
        self._live_timer = QTimer(self)
        self._live_timer.setInterval(self._LIVE_TICK_MS)
        self._live_timer.timeout.connect(self._on_live_tick)

        # Custom (user-opened) videos that aren't in the assets folder
        self._extra_videos: list[Path] = []

        # Bounding-box state. _index_boxes maps ts_ms -> list[detection dict];
        # _index_box_keys is a sorted list of those ts for nearest-neighbour
        # lookup as the player position changes. _current_boxes is the latest
        # live-mode inference (replaces on each tick).
        self._show_boxes: bool = True
        self._index_boxes: dict[int, list[dict]] = {}
        self._index_box_keys: list[int] = []
        self._current_boxes: list[dict] = []
        self._raw_index_boxes: dict[int, list[dict]] = {}
        self._raw_current_boxes: list[dict] = []
        self._raw_detection_events: list[tuple[int, list[dict]]] = []
        self._mode: str = "idle"  # "idle" | "live" | "index"
        self._index_category_buttons: dict[str, QPushButton] = {}
        self._audio_waveform_process: Optional[QProcess] = None
        self._audio_waveform_buffer = bytearray()
        self._audio_waveform_token: int = 0

        self.setObjectName("videoTestPanel")
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setHandleWidth(4)
        root.addWidget(top_splitter, stretch=1)

        # ---------------- Sidebar: video library ----------------
        sidebar = QFrame()
        sidebar.setStyleSheet(_surface_frame_css())
        sidebar.setMinimumWidth(198)
        sidebar.setMaximumWidth(280)
        sb = QVBoxLayout(sidebar)
        sb.setContentsMargins(10, 10, 10, 10)
        sb.setSpacing(8)
        sb_title = QLabel("Video Assets")
        sb_title.setStyleSheet("font-weight: 700; font-size: 11px;")
        sb.addWidget(sb_title)
        sb_hint = QLabel("Pick a video, or open one from disk.")
        sb_hint.setStyleSheet(_section_hint_css())
        sb_hint.setWordWrap(True)
        sb.addWidget(sb_hint)
        self._video_list = QListWidget()
        self._video_list.setAlternatingRowColors(True)
        self._video_list.itemActivated.connect(self._on_video_list_activated)
        self._video_list.currentItemChanged.connect(
            lambda cur, _prev: self._on_video_list_current(cur)
        )
        sb.addWidget(self._video_list, stretch=1)
        sb_btns = QHBoxLayout()
        sb_btns.setSpacing(4)
        self._open_btn = QPushButton("Open File...")
        self._open_btn.clicked.connect(self._on_open_video)
        sb_btns.addWidget(self._open_btn)
        self._refresh_videos_btn = QPushButton("Refresh")
        self._refresh_videos_btn.clicked.connect(self._populate_video_library)
        sb_btns.addWidget(self._refresh_videos_btn)
        sb.addLayout(sb_btns)
        top_splitter.addWidget(sidebar)

        # ---------------- Center: main footage stage ----------------
        center_stage = QWidget()
        rl = QVBoxLayout(center_stage)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)

        # Header strip: file label
        header_card = QFrame()
        header_card.setStyleSheet(_surface_frame_css())
        header_row = QHBoxLayout(header_card)
        header_row.setContentsMargins(10, 8, 10, 8)
        header_row.setSpacing(8)
        self._file_label = QLabel("No video loaded.")
        self._file_label.setStyleSheet("font-weight: 700; font-size: 11px;")
        header_row.addWidget(self._file_label, stretch=1)
        header_hint = QLabel("Single-clip inspection with live playback or a full indexed pass.")
        header_hint.setStyleSheet(_section_hint_css())
        header_row.addWidget(header_hint)
        rl.addWidget(header_card)

        # -- Video surface --
        # QGraphicsScene composes the video frame and the box overlay in a
        # single Qt paint pass. Avoids the QVideoWidget native-NSView layer
        # on macOS, which previously occluded any sibling-widget overlay.
        self._video_scene = QGraphicsScene(self)
        self._video_scene.setBackgroundBrush(QColor("#000000"))
        self._video_item = QGraphicsVideoItem()
        self._video_item.setAspectRatioMode(Qt.AspectRatioMode.KeepAspectRatio)
        self._video_scene.addItem(self._video_item)
        self._box_overlay = _BoundingBoxItem(self._video_item)
        self._video_scene.addItem(self._box_overlay)
        self._subroutine_overlay = SubroutineBoxOverlay(self._video_item)
        self._video_scene.addItem(self._subroutine_overlay)
        self._blink_item = SubroutineBlinkHighlight(self._video_item)
        self._video_scene.addItem(self._blink_item)
        self._subroutine_session = SubroutineSession(self)
        self._subroutine_roi_rect: Optional[QRect] = None
        self._subroutine_full_dets: list[dict] = []
        # frame context for blink highlight — set at ROI commit, reused on re-run
        self._subroutine_crop_ox: int = 0
        self._subroutine_crop_oy: int = 0
        self._subroutine_frame_w: int = 0
        self._subroutine_frame_h: int = 0

        self._video_surface = _TimedGraphicsView(self._video_scene)
        # NOTE: do not set a QOpenGLWidget viewport here. On macOS,
        # QGraphicsVideoItem renders into the default raster surface and
        # goes black when the view's viewport is GL-backed. Performance is
        # tuned with optimization flags + a cached overlay item instead.
        self._video_surface.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate
        )
        self._video_surface.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontSavePainterState, True
        )
        self._video_surface.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True
        )
        self._video_surface.setStyleSheet("background: #000; border: 0;")
        self._video_surface.setFrameShape(QFrame.Shape.NoFrame)
        self._video_surface.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._video_surface.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._video_surface.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )
        self._video_surface.setMinimumHeight(280)
        self._video_surface.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._video_surface.installEventFilter(self)
        self._video_surface.viewport().installEventFilter(self)
        rl.addWidget(self._video_surface, stretch=1)

        self._subroutine_roi = SubroutineRoiController(
            view=self._video_surface,
            video_item=self._video_item,
            scene=self._video_scene,
            parent=self,
        )

        self._player = QMediaPlayer(self)
        self._audio = QAudioOutput(self)
        self._player.setAudioOutput(self._audio)
        self._player.setVideoOutput(self._video_item)
        self._video_item.nativeSizeChanged.connect(self._layout_video_item)
        self._player.positionChanged.connect(self._on_position_changed)
        self._player.durationChanged.connect(self._on_duration_changed)
        self._player.playbackStateChanged.connect(self._on_playback_state)
        self._player.hasAudioChanged.connect(self._on_has_audio_changed)
        self._player.errorOccurred.connect(self._on_player_error)
        self._player.mediaStatusChanged.connect(self._on_media_status)

        # -- Playback controls + seek timeline --
        transport_card = QFrame()
        transport_card.setStyleSheet(_surface_frame_css())
        self._transport_layout = QHBoxLayout(transport_card)
        self._transport_layout.setContentsMargins(10, 8, 10, 8)
        self._transport_layout.setSpacing(10)

        transport_cluster = QWidget()
        transport_controls = QHBoxLayout(transport_cluster)
        transport_controls.setContentsMargins(0, 0, 0, 0)
        transport_controls.setSpacing(6)

        self._back_btn = QPushButton("[<< 5s]")
        self._back_btn.clicked.connect(lambda: self._jump_relative(-5000))
        transport_controls.addWidget(self._back_btn)

        self._play_btn = QPushButton("[PLAY]")
        self._play_btn.clicked.connect(self._on_play_pause)
        transport_controls.addWidget(self._play_btn)

        self._stop_btn = QPushButton("[STOP]")
        self._stop_btn.clicked.connect(self._on_stop)
        transport_controls.addWidget(self._stop_btn)

        self._forward_btn = QPushButton("[5s >>]")
        self._forward_btn.clicked.connect(lambda: self._jump_relative(5000))
        transport_controls.addWidget(self._forward_btn)

        self._mute_btn = QPushButton("[AUDIO]")
        self._mute_btn.setCheckable(True)
        self._mute_btn.setChecked(False)
        self._mute_btn.setToolTip("Mute or unmute video playback audio.")
        self._mute_btn.toggled.connect(self._on_mute_toggled)
        transport_controls.addWidget(self._mute_btn)

        seek_cluster = QWidget()
        seek_controls = QHBoxLayout(seek_cluster)
        seek_controls.setContentsMargins(0, 0, 0, 0)
        seek_controls.setSpacing(8)

        self._current_label = QLabel("00:00")
        self._current_label.setMinimumWidth(52)
        self._current_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        seek_controls.addWidget(self._current_label)

        self._position_slider = QSlider(Qt.Orientation.Horizontal)
        self._position_slider.setRange(0, 0)
        self._position_slider.setMinimumWidth(180)
        self._position_slider.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._position_slider.sliderPressed.connect(self._on_slider_pressed)
        self._position_slider.sliderReleased.connect(self._on_slider_released)
        self._position_slider.sliderMoved.connect(self._on_slider_moved)
        seek_controls.addWidget(self._position_slider, stretch=1)

        self._duration_label = QLabel("00:00")
        self._duration_label.setMinimumWidth(52)
        self._duration_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        seek_controls.addWidget(self._duration_label)

        speed_cluster = QWidget()
        speed_controls = QHBoxLayout(speed_cluster)
        speed_controls.setContentsMargins(0, 0, 0, 0)
        speed_controls.setSpacing(6)
        self._speed_label = QLabel("Speed:")
        speed_controls.addWidget(self._speed_label)
        self._speed_combo = QComboBox()
        for label, rate in (
            ("0.25x", 0.25),
            ("0.5x", 0.5),
            ("1x", 1.0),
            ("1.5x", 1.5),
            ("2x", 2.0),
            ("4x", 4.0),
        ):
            self._speed_combo.addItem(label, userData=rate)
        self._speed_combo.setCurrentIndex(2)
        self._speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        speed_controls.addWidget(self._speed_combo)

        self._transport_layout.addWidget(transport_cluster, 0)
        self._transport_layout.addWidget(seek_cluster, 1)
        self._transport_layout.addWidget(speed_cluster, 0)
        rl.addWidget(transport_card)

        top_splitter.addWidget(center_stage)

        # ---------------- Right: tabbed sidebar ----------------
        controls_sidebar = QFrame()
        controls_sidebar.setStyleSheet(_surface_frame_css())
        controls_sidebar.setMinimumWidth(286)
        controls_sidebar.setMaximumWidth(420)
        _cs_outer = QVBoxLayout(controls_sidebar)
        _cs_outer.setContentsMargins(0, 0, 0, 0)
        _cs_outer.setSpacing(0)

        self._sidebar_tabs = QTabWidget()
        self._sidebar_tabs.setDocumentMode(True)
        _cs_outer.addWidget(self._sidebar_tabs)

        # ---- Tab 0: Options & Toggles ----
        _options_tab = QWidget()
        controls_layout = QVBoxLayout(_options_tab)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(8)
        self._sidebar_tabs.addTab(_options_tab, "Options & Toggles")

        signal_card = QFrame()
        signal_card.setStyleSheet(_surface_frame_css())
        sc = QVBoxLayout(signal_card)
        sc.setContentsMargins(10, 8, 10, 10)
        sc.setSpacing(8)

        audio_timeline_label = QLabel("Audio Detection Timeline")
        audio_timeline_label.setStyleSheet("font-weight: 700; font-size: 11px;")
        sc.addWidget(audio_timeline_label)

        self._audio_timeline = AudioTimeline()
        self._audio_timeline.set_visual_scale(self._TIMELINE_VISUAL_SCALE)
        self._audio_timeline.set_aurora_waveform_cyan(True)
        self._audio_timeline.seek_requested.connect(self._on_waveform_seek)
        sc.addWidget(self._audio_timeline)

        timeline_label = QLabel("Detection Timeline")
        timeline_label.setStyleSheet("font-weight: 700; font-size: 11px;")
        sc.addWidget(timeline_label)

        self._detection_timeline = DetectionTimeline()
        self._detection_timeline.set_visual_scale(self._TIMELINE_VISUAL_SCALE)
        sc.addWidget(self._detection_timeline)

        self._legend = _LegendWrapWidget()
        sc.addWidget(self._legend)
        controls_layout.addWidget(signal_card)

        # -- Detection deck --
        detection_card = QFrame()
        detection_card.setStyleSheet(_surface_frame_css(raised=True))
        dc = QVBoxLayout(detection_card)
        dc.setContentsMargins(10, 9, 10, 10)
        dc.setSpacing(8)

        dc_title_row = QHBoxLayout()
        dc_title_row.setSpacing(8)
        dc_title = QLabel("Detection Deck")
        dc_title.setStyleSheet("font-weight: 700; font-size: 11px;")
        dc_title_row.addWidget(dc_title)
        self._live_indicator = QLabel("[REC]")
        self._live_indicator.setStyleSheet(
            f"color: {theme_rgba('privacy_warn', 0.96)}; font-weight: 700; font-size: 11px;"
        )
        self._live_indicator.setVisible(False)
        dc_title_row.addWidget(self._live_indicator)
        dc_title_row.addStretch(1)
        dc.addLayout(dc_title_row)

        self._config_deck = QVBoxLayout()
        self._config_deck.setSpacing(8)

        setup_group = QFrame()
        setup_group.setStyleSheet(_surface_frame_css())
        setup_layout = QVBoxLayout(setup_group)
        setup_layout.setContentsMargins(9, 8, 9, 8)
        setup_layout.setSpacing(6)
        setup_caption = QLabel("Inference Setup")
        setup_caption.setStyleSheet(_section_caption_css())
        setup_layout.addWidget(setup_caption)
        setup_grid = QGridLayout()
        setup_grid.setContentsMargins(0, 0, 0, 0)
        setup_grid.setHorizontalSpacing(8)
        setup_grid.setVerticalSpacing(6)
        model_lbl = QLabel("Model")
        model_lbl.setStyleSheet(_section_hint_css())
        setup_grid.addWidget(model_lbl, 0, 0)
        self._model_combo = QComboBox()
        self._model_combo.setMinimumWidth(180)
        self._populate_models()
        setup_grid.addWidget(self._model_combo, 0, 1)
        sample_lbl = QLabel("Sample every")
        sample_lbl.setStyleSheet(_section_hint_css())
        setup_grid.addWidget(sample_lbl, 1, 0)
        self._sample_combo = QComboBox()
        for step in (1, 2, 5, 10, 15, 30):
            self._sample_combo.addItem(f"{step} frame(s)", userData=step)
        self._sample_combo.setCurrentIndex(2)
        self._sample_combo.setToolTip(
            "Frame step for full index runs. Live face preview may sample more "
            "tightly to keep bounding boxes responsive."
        )
        setup_grid.addWidget(self._sample_combo, 1, 1)
        device_lbl = QLabel("Device")
        device_lbl.setStyleSheet(_section_hint_css())
        setup_grid.addWidget(device_lbl, 2, 0)
        self._device_combo = QComboBox()
        for label, device_id in _detect_devices():
            self._device_combo.addItem(label, userData=device_id)
        setup_grid.addWidget(self._device_combo, 2, 1)
        setup_grid.setColumnStretch(1, 1)
        setup_layout.addLayout(setup_grid)

        scope_group = QFrame()
        scope_group.setStyleSheet(_surface_frame_css())
        scope_layout = QVBoxLayout(scope_group)
        scope_layout.setContentsMargins(9, 8, 9, 8)
        scope_layout.setSpacing(6)
        scope_caption = QLabel("Index Scope")
        scope_caption.setStyleSheet(_section_caption_css())
        scope_layout.addWidget(scope_caption)
        scope_grid = QGridLayout()
        scope_grid.setContentsMargins(0, 0, 0, 0)
        scope_grid.setHorizontalSpacing(8)
        scope_grid.setVerticalSpacing(6)
        categories_lbl = QLabel("Categories")
        categories_lbl.setStyleSheet(_section_hint_css())
        scope_grid.addWidget(categories_lbl, 0, 0)
        categories_host = QWidget()
        categories_row = QHBoxLayout(categories_host)
        categories_row.setContentsMargins(0, 0, 0, 0)
        categories_row.setSpacing(4)
        for key, label in self._INDEX_CATEGORY_CHOICES:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setToolTip(f"Include {label.lower()} detections in the next index run.")
            btn.toggled.connect(self._on_scope_filter_changed)
            self._index_category_buttons[key] = btn
            categories_row.addWidget(btn)
        categories_row.addStretch(1)
        scope_grid.addWidget(categories_host, 0, 1)
        label_filter_lbl = QLabel("Label filter")
        label_filter_lbl.setStyleSheet(_section_hint_css())
        scope_grid.addWidget(label_filter_lbl, 1, 0)
        self._index_filter_edit = QLineEdit()
        self._index_filter_edit.setPlaceholderText("person, car, cell phone...")
        self._index_filter_edit.setClearButtonEnabled(True)
        self._index_filter_edit.setMinimumWidth(180)
        self._index_filter_edit.setToolTip("Optional label filter for the next index run.")
        self._index_filter_edit.textChanged.connect(self._on_scope_filter_changed)
        scope_grid.addWidget(self._index_filter_edit, 1, 1)
        scope_grid.setColumnStretch(1, 1)
        scope_layout.addLayout(scope_grid)

        self._config_deck.addWidget(setup_group, 0)
        self._config_deck.addWidget(scope_group, 1)
        dc.addLayout(self._config_deck)

        self._action_deck = QVBoxLayout()
        self._action_deck.setSpacing(8)

        review_group = QFrame()
        review_group.setStyleSheet(_surface_frame_css())
        review_layout = QVBoxLayout(review_group)
        review_layout.setContentsMargins(9, 8, 9, 8)
        review_layout.setSpacing(6)
        review_caption = QLabel("Review Tools")
        review_caption.setStyleSheet(_section_caption_css())
        review_layout.addWidget(review_caption)
        review_row = QHBoxLayout()
        review_row.setContentsMargins(0, 0, 0, 0)
        review_row.setSpacing(6)
        self._boxes_btn = QPushButton("[BOXES]")
        self._boxes_btn.setCheckable(True)
        self._boxes_btn.setChecked(True)
        self._boxes_btn.setToolTip("Toggle bounding-box overlay on the video.")
        self._boxes_btn.toggled.connect(self._on_boxes_toggled)
        review_row.addWidget(self._boxes_btn)
        self._subroutine_roi_btn = QPushButton("[SUBROUTINE ROI]")
        self._subroutine_roi_btn.setCheckable(True)
        self._subroutine_roi_btn.setToolTip(
            "Drag a rectangle on the video, then switch to the Subroutine tab."
        )
        self._subroutine_roi_btn.toggled.connect(self._on_subroutine_roi_mode)
        review_row.addWidget(self._subroutine_roi_btn)
        self._clear_btn = QPushButton("Clear timeline")
        self._clear_btn.clicked.connect(self._on_clear_detections)
        review_row.addWidget(self._clear_btn)
        self._flag_btn = QPushButton("[FLAG]")
        self._flag_btn.setToolTip(
            "Flag the current frame as a wrong/missed detection (Ctrl+F). "
            "Pauses the video, captures the frame, opens the correction "
            "dialog, and saves the result to the local corrections store."
        )
        self._flag_btn.setEnabled(False)
        self._flag_btn.clicked.connect(self._on_flag_frame)
        review_row.addWidget(self._flag_btn)
        review_row.addStretch(1)
        review_layout.addLayout(review_row)

        run_group = QFrame()
        run_group.setStyleSheet(_surface_frame_css())
        run_layout = QVBoxLayout(run_group)
        run_layout.setContentsMargins(9, 8, 9, 8)
        run_layout.setSpacing(6)
        run_caption = QLabel("Execution")
        run_caption.setStyleSheet(_section_caption_css())
        run_layout.addWidget(run_caption)
        run_row = QHBoxLayout()
        run_row.setContentsMargins(0, 0, 0, 0)
        run_row.setSpacing(6)
        self._live_btn = QPushButton("[LIVE DETECT]")
        self._live_btn.setCheckable(True)
        self._live_btn.setToolTip(
            "Run inference while the video plays. Detections are added to "
            "the timeline as they happen."
        )
        self._live_btn.toggled.connect(self._on_live_toggle)
        run_row.addWidget(self._live_btn)
        self._run_btn = QPushButton("[INDEX VIDEO]")
        self._run_btn.setToolTip(
            "Scan the entire video upfront and place every detection on the "
            "timeline so you can scrub through them."
        )
        self._run_btn.clicked.connect(self._on_run_detector)
        self._run_btn.setEnabled(False)
        run_row.addWidget(self._run_btn)
        self._cancel_btn = QPushButton("[CANCEL]")
        self._cancel_btn.clicked.connect(self._on_cancel_detector)
        self._cancel_btn.setEnabled(False)
        run_row.addWidget(self._cancel_btn)
        run_row.addStretch(1)
        run_layout.addLayout(run_row)

        debug_caption = QLabel("Diagnostics")
        debug_caption.setStyleSheet(_section_caption_css())
        run_layout.addWidget(debug_caption)
        self._debug_view = QLabel("paint --\ninfer --")
        self._debug_view.setWordWrap(True)
        self._debug_view.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._debug_view.setStyleSheet(
            f"color: {text_css(0.84)};"
            " font-family: Menlo, Consolas, monospace; font-size: 10px;"
            f" background: {theme_rgba('panel', 0.65)};"
            " padding: 6px 8px; border-radius: 3px;"
        )
        run_layout.addWidget(self._debug_view)

        # Inference cadence + latency samples for the diagnostics panel.
        self._infer_clocks: deque[float] = deque(maxlen=60)
        self._last_box_count: int = 0
        self._last_box_ts_ms: int = -1
        self._infer_request_wall: float = 0.0
        self._last_infer_wall_ms: float = 0.0
        self._lag_samples: deque[int] = deque(maxlen=40)
        self._hud_timer = QTimer(self)
        self._hud_timer.setInterval(250)
        self._hud_timer.timeout.connect(self._update_hud)
        self._hud_timer.start()

        self._action_deck.addWidget(review_group, 0)
        self._action_deck.addWidget(run_group, 1)
        dc.addLayout(self._action_deck)

        self._index_progress = QProgressBar()
        self._index_progress.setRange(0, 100)
        self._index_progress.setValue(0)
        self._index_progress.setTextVisible(True)
        self._index_progress.setFormat("Idle")
        self._index_progress.setVisible(False)
        self._index_progress.setMinimumHeight(18)
        dc.addWidget(self._index_progress)

        self._index_substatus = QLabel("")
        self._index_substatus.setStyleSheet(
            "color: rgba(180,180,180,0.85); font-size: 10px;"
        )
        self._index_substatus.setVisible(False)
        dc.addWidget(self._index_substatus)

        controls_layout.addWidget(detection_card, stretch=1)

        # ---- Tab 1: Subroutine ----
        self._subroutine_panel = SubroutineControlsWidget(
            http_get=self._http_get,
            parent=self,
        )
        self._subroutine_panel.runRequested.connect(self._on_subroutine_run)
        self._subroutine_panel.dismissed.connect(self._on_subroutine_dismissed)
        self._subroutine_panel.highlightRequested.connect(self._on_subroutine_highlight)
        self._subroutine_panel.fusionRequested.connect(self._on_subroutine_fusion)
        self._subroutine_panel.reAnalyzeRequested.connect(self._on_subroutine_reanalyze)
        self._sidebar_tabs.addTab(self._subroutine_panel, "Subroutine")

        # ---- Tab 2: Convert ----
        self._converter_panel = ModelConverterPanel(parent=self)
        self._converter_panel.modelRegistered.connect(self._on_converter_model_registered)
        self._sidebar_tabs.addTab(self._converter_panel, "Convert")

        self._status = QLabel("Open a video to begin.")
        self._status.setProperty("muted", True)
        root.addWidget(self._status)

        top_splitter.addWidget(controls_sidebar)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setStretchFactor(2, 0)
        top_splitter.setSizes([220, 960, 340])

        self._update_transport_enabled(False)
        self._populate_video_library()
        self._maybe_load_default_video()
        self.refresh_responsive_layout()
        self._sync_mute_ui()

        # Ctrl+F: flag the current frame. Scoped to this panel so the
        # shortcut doesn't fire from other tabs.
        self._flag_shortcut = QShortcut(QKeySequence("Ctrl+F"), self)
        self._flag_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._flag_shortcut.activated.connect(self._on_flag_frame)

        self._subroutine_roi.roiCommitted.connect(self._on_subroutine_roi_committed)
        self._subroutine_roi.cleared.connect(self._on_subroutine_roi_cleared)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reload(self) -> None:
        self._populate_models()
        if hasattr(self, "_subroutine_panel"):
            self._subroutine_panel.refresh_models()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.refresh_responsive_layout()

    def refresh_theme_styles(self) -> None:
        if hasattr(self, "_live_indicator"):
            self._live_indicator.setStyleSheet(
                f"color: {theme_rgba('privacy_warn', 0.96)}; font-weight: 700; font-size: 11px;"
            )
        if hasattr(self, "_index_substatus"):
            self._index_substatus.setStyleSheet(
                f"color: {text_css(0.85)}; font-size: 10px;"
            )
        if hasattr(self, "_video_list"):
            for i in range(self._video_list.count()):
                item = self._video_list.item(i)
                if item is not None and not (item.flags() & Qt.ItemFlag.ItemIsSelectable):
                    item.setForeground(text_qcolor(0.58))
        for attr in ("_detection_timeline", "_audio_timeline", "_boxes_item"):
            obj = getattr(self, attr, None)
            if obj is not None:
                try:
                    obj.update()
                except Exception:
                    pass

    def refresh_responsive_layout(self) -> None:
        if not hasattr(self, "_transport_layout"):
            return
        width = max(0, self.width())
        transport_vertical = width < 1120
        self._transport_layout.setDirection(
            QBoxLayout.Direction.TopToBottom
            if transport_vertical
            else QBoxLayout.Direction.LeftToRight
        )

    def _indexing_active(self) -> bool:
        return self._index_process is not None or self._worker_thread is not None

    def _selected_index_categories(self) -> list[str]:
        return [
            key
            for key, button in self._index_category_buttons.items()
            if button.isChecked()
        ]

    def _current_scope_filters(self) -> tuple[set[str], str]:
        return (
            {key for key in self._selected_index_categories()},
            str(self._index_filter_edit.text() or "").strip().lower(),
        )

    def _filter_visible_detections(self, detections: list[dict]) -> list[dict]:
        categories, label_filter = self._current_scope_filters()
        return _filter_scoped_detections(
            detections,
            label_filter=label_filter,
            categories=categories,
        )

    def _index_filter_summary(self, categories: list[str], label_filter: str) -> str:
        parts: list[str] = []
        if categories:
            labels = {
                key: label
                for key, label in self._INDEX_CATEGORY_CHOICES
            }
            parts.append(", ".join(labels.get(key, key) for key in categories))
        if label_filter:
            parts.append(f"label '{label_filter}'")
        return " + ".join(parts) if parts else "all detections"

    def _rebuild_detection_views_from_raw(self) -> None:
        self._detection_timeline.clear_events()
        self._index_boxes = {}

        for ts_ms, raw_detections in self._raw_detection_events:
            visible = self._filter_visible_detections(raw_detections)
            labels = [str(det.get("label", "")) for det in visible]
            if labels:
                self._detection_timeline.add_events(ts_ms, labels)

        for ts_key, raw_detections in self._raw_index_boxes.items():
            self._index_boxes[ts_key] = self._filter_visible_detections(raw_detections)

        self._current_boxes = self._filter_visible_detections(self._raw_current_boxes)

        if not self._show_boxes:
            self._box_overlay.clear()
        elif self._mode == "live":
            self._box_overlay.set_boxes(self._current_boxes)
        elif self._mode == "index":
            self._update_index_overlay(self._player.position())
        else:
            self._box_overlay.clear()

        self._timeline_dirty = True
        self._flush_timeline_repaint()

    def _on_scope_filter_changed(self, *_args) -> None:
        self._rebuild_detection_views_from_raw()

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _populate_models(self) -> None:
        current = self._model_combo.currentData() if hasattr(self, "_model_combo") else None
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        seen_paths: set[str] = set()
        try:
            models = collect_video_test_models(http_get=getattr(self, "_http_get", None))
        except Exception:
            models = []
        for label, path in models:
            path_str = str(path or "").strip()
            if not path_str:
                continue
            try:
                key = str(Path(path_str).resolve())
            except Exception:
                key = path_str
            if key in seen_paths:
                continue
            seen_paths.add(key)
            self._model_combo.addItem(label, userData=path_str)
        if self._model_combo.count() == 0:
            self._model_combo.addItem("(no models found)", userData="")
            self._model_combo.setEnabled(False)
        else:
            self._model_combo.setEnabled(True)
        if current:
            idx = self._model_combo.findData(current)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)
        if hasattr(self, "_subroutine_panel"):
            self._subroutine_panel.refresh_models()

    def _maybe_load_default_video(self) -> None:
        default = _ASSETS_VIDEOS / "frenchpeoplewalkinglong.mp4"
        if default.exists():
            self._load_video(default)

    # ------------------------------------------------------------------
    # Video library
    # ------------------------------------------------------------------

    _VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

    def _populate_video_library(self) -> None:
        prev = self._video_path
        self._video_list.blockSignals(True)
        self._video_list.clear()

        if _ASSETS_VIDEOS.exists():
            assets = sorted(
                p for p in _ASSETS_VIDEOS.iterdir()
                if p.is_file() and p.suffix.lower() in self._VIDEO_EXTS
            )
        else:
            assets = []

        if assets:
            header = QListWidgetItem("— assets/videos —")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setForeground(text_qcolor(0.58))
            self._video_list.addItem(header)
            for path in assets:
                item = QListWidgetItem(path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setToolTip(str(path))
                self._video_list.addItem(item)

        # De-duplicate the user-opened list against assets.
        asset_set = {p.resolve() for p in assets}
        extras = [p for p in self._extra_videos if p.resolve() not in asset_set]
        if extras:
            header = QListWidgetItem("— opened —")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setForeground(text_qcolor(0.58))
            self._video_list.addItem(header)
            for path in extras:
                item = QListWidgetItem(path.name)
                item.setData(Qt.ItemDataRole.UserRole, str(path))
                item.setToolTip(str(path))
                self._video_list.addItem(item)

        if self._video_list.count() == 0:
            empty = QListWidgetItem("(no videos found)")
            empty.setFlags(Qt.ItemFlag.NoItemFlags)
            empty.setForeground(text_qcolor(0.58))
            self._video_list.addItem(empty)

        self._video_list.blockSignals(False)

        if prev is not None:
            self._highlight_in_library(prev)

    def _highlight_in_library(self, path: Path) -> None:
        target = str(path.resolve())
        for i in range(self._video_list.count()):
            item = self._video_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if not data:
                continue
            try:
                if str(Path(data).resolve()) == target:
                    self._video_list.blockSignals(True)
                    self._video_list.setCurrentRow(i)
                    self._video_list.blockSignals(False)
                    return
            except Exception:
                pass

    def _on_video_list_activated(self, item: QListWidgetItem) -> None:
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if path_str:
            self._load_video(Path(path_str))

    def _on_video_list_current(self, item: Optional[QListWidgetItem]) -> None:
        if item is None:
            return
        path_str = item.data(Qt.ItemDataRole.UserRole)
        if not path_str:
            return
        path = Path(path_str)
        if self._video_path is not None and path.resolve() == self._video_path.resolve():
            return
        self._load_video(path)

    # ------------------------------------------------------------------
    # File handling
    # ------------------------------------------------------------------

    def _on_open_video(self) -> None:
        start = str(_ASSETS_VIDEOS) if _ASSETS_VIDEOS.exists() else ""
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            start,
            "Video Files (*.mp4 *.mov *.avi *.mkv *.webm);;All Files (*.*)",
        )
        if not path_str:
            return
        self._load_video(Path(path_str))

    def _load_video(self, path: Path) -> None:
        self._on_subroutine_dismissed()
        self._subroutine_roi_btn.setChecked(False)
        self._discard_audio_waveform_process(kill=True)
        if self._indexing_active():
            self._discard_index_process(kill=True)
            self._teardown_worker()
            self._index_cancel_requested = False
            self._index_progress.setVisible(False)
            self._index_substatus.setVisible(False)

        # Switching videos: tear down any active live session first since the
        # worker holds an open capture against the previous file.
        if self._live_active:
            self._live_btn.setChecked(False)

        # If user opened a file outside the assets folder, remember it.
        try:
            in_assets = (
                _ASSETS_VIDEOS.exists()
                and path.resolve().is_relative_to(_ASSETS_VIDEOS.resolve())
            )
        except Exception:
            in_assets = False
        if not in_assets:
            resolved = path.resolve()
            if not any(p.resolve() == resolved for p in self._extra_videos):
                self._extra_videos.append(path)
                self._populate_video_library()

        self._video_path = path
        self._file_label.setText(path.name)
        self._detection_timeline.clear_events()
        self._audio_timeline.reset()
        self._index_boxes.clear()
        self._index_box_keys.clear()
        self._current_boxes = []
        self._raw_index_boxes.clear()
        self._raw_current_boxes = []
        self._raw_detection_events.clear()
        self._box_overlay.clear()
        self._mode = "idle"
        self._update_legend()
        self._player.stop()
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._start_audio_waveform_analysis(path)
        self._status.setText(f"Loaded {path.name}.")
        self._update_transport_enabled(True)
        self._highlight_in_library(path)

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    def _update_transport_enabled(self, enabled: bool) -> None:
        for btn in (
            self._play_btn,
            self._stop_btn,
            self._back_btn,
            self._forward_btn,
        ):
            btn.setEnabled(enabled)
        self._position_slider.setEnabled(enabled)
        self._flag_btn.setEnabled(enabled)
        has_model = bool(self._model_combo.currentData())
        # Index disabled while live is on (and vice-versa) so the two pipelines
        # don't fight over the same model load.
        indexing = self._indexing_active()
        self._run_btn.setEnabled(
            enabled and has_model and not self._live_active and not indexing
        )
        self._live_btn.setEnabled(enabled and has_model and not indexing)
        self._sync_mute_ui()

    def _sync_mute_ui(self) -> None:
        """Enable mute control when a clip with an audio stream is loaded."""
        loaded = self._video_path is not None
        transport_on = self._play_btn.isEnabled()
        has_audio = self._player.hasAudio()
        can_mute = loaded and transport_on and has_audio
        self._mute_btn.blockSignals(True)
        try:
            self._mute_btn.setChecked(self._audio.isMuted())
            self._mute_btn.setText("[MUTED]" if self._audio.isMuted() else "[AUDIO]")
            self._mute_btn.setEnabled(can_mute)
        finally:
            self._mute_btn.blockSignals(False)

    def _on_mute_toggled(self, muted: bool) -> None:
        self._audio.setMuted(muted)
        self._mute_btn.setText("[MUTED]" if muted else "[AUDIO]")

    def _on_play_pause(self) -> None:
        if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self._player.pause()
        else:
            self._player.play()

    def _on_stop(self) -> None:
        self._player.stop()

    def _jump_relative(self, delta_ms: int) -> None:
        target = max(0, min(self._duration_ms, self._player.position() + delta_ms))
        self._player.setPosition(target)

    def _on_speed_changed(self, _idx: int) -> None:
        rate = float(self._speed_combo.currentData() or 1.0)
        self._player.setPlaybackRate(rate)

    def _on_waveform_seek(self, position_ms: int) -> None:
        self._player.setPosition(position_ms)
        self._position_slider.setValue(position_ms)
        self._current_label.setText(_format_ms(position_ms))
        self._detection_timeline.set_cursor(position_ms)

    def _on_slider_pressed(self) -> None:
        self._user_seeking = True

    def _on_slider_released(self) -> None:
        self._player.setPosition(self._position_slider.value())
        self._user_seeking = False

    def _on_slider_moved(self, value: int) -> None:
        self._current_label.setText(_format_ms(value))
        self._audio_timeline.set_cursor(value)
        self._detection_timeline.set_cursor(value)
        if self._mode == "index" and self._show_boxes:
            self._update_index_overlay(value)

    # ------------------------------------------------------------------
    # Player callbacks
    # ------------------------------------------------------------------

    def _on_position_changed(self, position: int) -> None:
        if not self._user_seeking:
            self._position_slider.setValue(position)
            self._current_label.setText(_format_ms(position))
        self._audio_timeline.set_cursor(position)
        self._detection_timeline.set_cursor(position)
        if self._mode == "index" and self._show_boxes:
            self._update_index_overlay(position)

    def eventFilter(self, obj, event):  # type: ignore[override]
        is_view = obj is self._video_surface
        is_viewport = (
            hasattr(self, "_video_surface")
            and obj is self._video_surface.viewport()
        )
        if is_view or is_viewport:
            if hasattr(self, "_subroutine_roi") and self._subroutine_roi.handle_event(event):
                return True
            if is_view and event.type() == QEvent.Type.Resize:
                self._layout_video_item()
        return super().eventFilter(obj, event)

    def _layout_video_item(self) -> None:
        """Resize/center the video item inside the view, then sync the overlay.

        Sizing the video item explicitly (rather than relying on the view's
        fitInView) keeps the overlay's coordinate system identical to the
        video item's local rect, so detection xy maps cleanly without extra
        letterbox math.
        """
        viewport = self._video_surface.viewport()
        view_w = max(1, viewport.width())
        view_h = max(1, viewport.height())
        self._video_scene.setSceneRect(0.0, 0.0, float(view_w), float(view_h))

        native = self._video_item.nativeSize()
        if native.isValid() and native.width() > 0 and native.height() > 0:
            scale = min(view_w / native.width(), view_h / native.height())
            disp_w = native.width() * scale
            disp_h = native.height() * scale
        else:
            disp_w = float(view_w)
            disp_h = float(view_h)

        self._video_item.setSize(QSizeF(disp_w, disp_h))
        self._video_item.setPos(
            (view_w - disp_w) / 2.0, (view_h - disp_h) / 2.0
        )
        self._box_overlay.sync_to_video()
        if hasattr(self, "_subroutine_overlay"):
            self._subroutine_overlay.sync_to_video()
        if hasattr(self, "_subroutine_roi"):
            self._subroutine_roi.sync_layout()

    def _on_subroutine_roi_mode(self, checked: bool) -> None:
        self._subroutine_roi.set_select_mode(checked)
        self._video_surface.setMouseTracking(bool(checked))
        self._video_surface.viewport().setMouseTracking(bool(checked))
        if checked:
            self._status.setText("Drag a box on the video to mark a subroutine region.")
        elif self._subroutine_roi.current_roi() is None:
            self._status.setText("Subroutine ROI select off.")

    def _on_subroutine_roi_committed(self, rect: QRect) -> None:
        # Store the rect BEFORE calling set_select_mode so the clear-on-deactivate
        # guard in the controller sees a non-None rect and skips emitting cleared.
        self._subroutine_roi_rect = QRect(rect)

        self._subroutine_roi_btn.blockSignals(True)
        self._subroutine_roi_btn.setChecked(False)
        self._subroutine_roi_btn.blockSignals(False)
        self._subroutine_roi.set_select_mode(False)
        self._video_surface.setMouseTracking(False)
        self._video_surface.viewport().setMouseTracking(False)

        # Capture the crop immediately and show it as the raw image.
        frame_img = self._capture_current_frame()
        raw_crop: Optional[QImage] = None
        if frame_img is not None:
            frame_roi = self._roi_display_to_frame(self._subroutine_roi_rect)
            raw_crop = crop_qimage(frame_img, frame_roi)
            # Save frame context so the blink highlight can map crop-local coords
            # back into full-frame space even on re-runs where crop is already set.
            self._subroutine_crop_ox = frame_roi.x()
            self._subroutine_crop_oy = frame_roi.y()
            self._subroutine_frame_w = frame_img.width()
            self._subroutine_frame_h = frame_img.height()

        self._subroutine_panel.open_for_roi(raw_crop=raw_crop)
        self._sidebar_tabs.setCurrentWidget(self._subroutine_panel)

        # Auto-fire: prefer fusion set if active, otherwise single-model
        if self._subroutine_panel.has_fusion_set() and raw_crop is not None and not raw_crop.isNull():
            self._status.setText("Region captured — running fusion...")
            self._on_subroutine_fusion(self._subroutine_panel.fusion_paths())
        else:
            model_path = str(self._subroutine_panel.current_model_path()).strip()
            if model_path and raw_crop is not None and not raw_crop.isNull():
                self._status.setText("Region captured — running subroutine...")
                self._on_subroutine_run(model_path, "")
            else:
                self._status.setText("Region captured — select a model and run.")

    def _on_subroutine_roi_cleared(self) -> None:
        self._subroutine_roi_rect = None
        self._subroutine_overlay.clear()

    def _on_subroutine_dismissed(self) -> None:
        if getattr(self, "_subroutine_resetting", False):
            return
        self._subroutine_resetting = True
        try:
            self._subroutine_session.stop()
            self._subroutine_overlay.clear()
            self._blink_item.stop()
            self._subroutine_full_dets = []
            self._subroutine_roi_rect = None
            self._subroutine_crop_ox = 0
            self._subroutine_crop_oy = 0
            self._subroutine_frame_w = 0
            self._subroutine_frame_h = 0
            self._subroutine_panel.hide_panel()
            self._subroutine_roi_btn.blockSignals(True)
            self._subroutine_roi_btn.setChecked(False)
            self._subroutine_roi_btn.blockSignals(False)
            self._subroutine_roi.set_select_mode(False)
            self._subroutine_roi.clear(emit=False)
        finally:
            self._subroutine_resetting = False

    def _on_converter_model_registered(self, output_path: str) -> None:
        """Refresh the subroutine model combo and the main inference combo after a conversion."""
        if hasattr(self, "_subroutine_panel"):
            self._subroutine_panel.refresh_models()
        self._populate_models()

    def _on_subroutine_highlight(self, index: int) -> None:
        self._blink_item.stop()
        if index < 0 or index >= len(self._subroutine_full_dets):
            return
        self._blink_item.start_detection(self._subroutine_full_dets[index])

    def _roi_display_to_frame(self, roi: QRect) -> QRect:
        """Map video-item display coords to native frame pixels."""
        native = self._video_item.nativeSize()
        size = self._video_item.size()
        if not native.isValid() or size.width() <= 0 or size.height() <= 0:
            return roi
        sx = native.width() / size.width()
        sy = native.height() / size.height()
        return QRect(
            int(roi.x() * sx),
            int(roi.y() * sy),
            max(1, int(roi.width() * sx)),
            max(1, int(roi.height() * sy)),
        )

    def _on_subroutine_run(self, model_path: str, _device: str) -> None:
        if self._subroutine_roi_rect is None:
            self._subroutine_panel.show_error("No region selected — draw a box on the video first.")
            return

        # Prefer the crop already captured at commit time; re-capture for Try Again.
        crop: Optional[QImage] = self._subroutine_panel.raw_crop()

        if crop is None or crop.isNull():
            # Capture straight off the video sink (works while playing) so the
            # clip keeps flowing — the subroutine runs in its own worker thread
            # against this snapshot rather than freezing playback.
            frame_img = self._capture_current_frame()
            if frame_img is None:
                self._subroutine_panel.show_error(
                    "Could not capture frame — try pausing on the target frame."
                )
                return
            frame_roi = self._roi_display_to_frame(self._subroutine_roi_rect)
            crop = crop_qimage(frame_img, frame_roi)
            if crop.isNull():
                self._subroutine_panel.show_error("Invalid crop region.")
                return
            self._subroutine_panel.show_raw_crop(crop)
            # Update stored frame context in case this is a re-capture mid-session
            self._subroutine_crop_ox = frame_roi.x()
            self._subroutine_crop_oy = frame_roi.y()
            self._subroutine_frame_w = frame_img.width()
            self._subroutine_frame_h = frame_img.height()

        try:
            crop_bgr = qimage_to_bgr_ndarray(crop)
        except Exception as exc:
            self._subroutine_panel.show_error(f"Crop conversion failed: {exc}")
            return

        device = str(self._device_combo.currentData() or "")
        self._subroutine_panel.show_running()
        _crop_ref = crop
        _ox = self._subroutine_crop_ox
        _oy = self._subroutine_crop_oy
        _fw = self._subroutine_frame_w or crop.width()
        _fh = self._subroutine_frame_h or crop.height()

        def _done(detections: list) -> None:
            # Always produce full-frame coords so the blink highlight can map
            # them against the full video item regardless of crop origin.
            full = offset_detections(detections, _ox, _oy)
            for det in full:
                det["frame_w"] = _fw
                det["frame_h"] = _fh
            # Don't show detection boxes inside the ROI on the video — they live
            # in the subroutine result panel instead.
            self._subroutine_overlay.clear()
            self._subroutine_full_dets = full
            self._subroutine_panel.show_results(
                crop_image=_crop_ref,
                detections=detections,
                frame_w=_crop_ref.width(),
                frame_h=_crop_ref.height(),
            )

        def _fail(msg: str) -> None:
            self._subroutine_panel.show_error(msg)
            self._subroutine_overlay.clear()

        self._subroutine_session.start(
            model_path=model_path,
            device=device,
            frame_bgr=crop_bgr,
            on_finished=_done,
            on_failed=_fail,
        )

    def _on_subroutine_fusion(self, model_paths: list) -> None:
        if self._subroutine_roi_rect is None:
            self._subroutine_panel.show_error("No region selected — draw a box on the video first.")
            return
        paths = [str(p) for p in model_paths if p]
        if len(paths) < 2:
            self._subroutine_panel.show_error("Pick at least two models for fusion.")
            return

        # Reuse the same crop-capture path as _on_subroutine_run
        crop: Optional[QImage] = self._subroutine_panel.raw_crop()
        if crop is None or crop.isNull():
            # Capture straight off the video sink (works while playing) so the
            # clip keeps flowing while the fusion models run in worker threads.
            frame_img = self._capture_current_frame()
            if frame_img is None:
                self._subroutine_panel.show_error(
                    "Could not capture frame — try pausing on the target frame."
                )
                return
            frame_roi = self._roi_display_to_frame(self._subroutine_roi_rect)
            crop = crop_qimage(frame_img, frame_roi)
            if crop.isNull():
                self._subroutine_panel.show_error("Invalid crop region.")
                return
            self._subroutine_panel.show_raw_crop(crop)
            self._subroutine_crop_ox = frame_roi.x()
            self._subroutine_crop_oy = frame_roi.y()
            self._subroutine_frame_w = frame_img.width()
            self._subroutine_frame_h = frame_img.height()

        try:
            crop_bgr = qimage_to_bgr_ndarray(crop)
        except Exception as exc:
            self._subroutine_panel.show_error(f"Crop conversion failed: {exc}")
            return

        device = str(self._device_combo.currentData() or "")
        _crop_ref = crop
        _ox = self._subroutine_crop_ox
        _oy = self._subroutine_crop_oy
        _fw = self._subroutine_frame_w or crop.width()
        _fh = self._subroutine_frame_h or crop.height()
        streaming = self._subroutine_panel.is_stream_enabled()

        per_model_local: list[tuple[str, list[dict]]] = []
        full_dets: list[dict] = []

        if streaming:
            self._subroutine_panel.begin_fusion_stream(
                crop_image=_crop_ref,
                model_labels=[Path(p).name for p in paths],
                frame_w=_crop_ref.width(),
                frame_h=_crop_ref.height(),
            )
        else:
            self._subroutine_panel.show_running()

        def _each(path: str, detections: list) -> None:
            label = Path(path).name
            per_model_local.append((label, list(detections or [])))
            # accumulate full-frame coords for the video-side overlay/blink
            offs = offset_detections(detections or [], _ox, _oy)
            for det in offs:
                det["frame_w"] = _fw
                det["frame_h"] = _fh
                det["model_label"] = Path(path).stem
            full_dets.extend(offs)
            # Keep _subroutine_full_dets in sync so blink-on-row-click works mid-stream
            self._subroutine_full_dets = list(full_dets)
            if streaming:
                self._subroutine_panel.append_fusion_model(label, detections or [])

        def _each_failed(path: str, msg: str) -> None:
            label = Path(path).name
            per_model_local.append((label, []))
            if streaming:
                # Still notify the panel so it strips the "running…" suffix
                self._subroutine_panel.append_fusion_model(label, [])
            else:
                self._status.setText(f"Fusion: {label} failed — {msg}")

        def _all_done() -> None:
            self._subroutine_overlay.clear()
            self._subroutine_full_dets = full_dets
            if streaming:
                self._subroutine_panel.finish_fusion_stream()
            else:
                self._subroutine_panel.show_fusion_results(
                    crop_image=_crop_ref,
                    per_model=per_model_local,
                    frame_w=_crop_ref.width(),
                    frame_h=_crop_ref.height(),
                )

        self._subroutine_session.start_multi(
            model_paths=paths,
            device=device,
            frame_bgr=crop_bgr,
            on_each=_each,
            on_each_failed=_each_failed,
            on_all_done=_all_done,
        )

    def _on_subroutine_reanalyze(self) -> None:
        """Recapture the current frame and re-run whichever subroutine mode is active."""
        if self._subroutine_roi_rect is None:
            self._subroutine_panel.show_error("No region selected — draw a box on the video first.")
            return

        frame_img = self._capture_current_frame()
        if frame_img is None:
            self._subroutine_panel.show_error(
                "Could not capture frame — try pausing on the target frame."
            )
            return
        frame_roi = self._roi_display_to_frame(self._subroutine_roi_rect)
        crop = crop_qimage(frame_img, frame_roi)
        if crop.isNull():
            self._subroutine_panel.show_error("Invalid crop region.")
            return

        # Replace cached crop + frame context with the fresh capture so subsequent
        # _on_subroutine_run / _on_subroutine_fusion calls use the new image.
        self._subroutine_panel.show_raw_crop(crop)
        self._subroutine_crop_ox = frame_roi.x()
        self._subroutine_crop_oy = frame_roi.y()
        self._subroutine_frame_w = frame_img.width()
        self._subroutine_frame_h = frame_img.height()

        # Dispatch to whichever mode is active
        if self._subroutine_panel.has_fusion_set():
            self._on_subroutine_fusion(self._subroutine_panel.fusion_paths())
        else:
            model_path = str(self._subroutine_panel.current_model_path()).strip()
            if not model_path:
                self._subroutine_panel.show_error("Open the catalog and pick a model first.")
                return
            self._on_subroutine_run(model_path, "")

    def _update_hud(self) -> None:
        """Refresh the diagnostics panel from rolling paint/inference stats."""
        avg_ms, fps = self._video_surface.paint_stats()
        if len(self._infer_clocks) >= 2:
            span = self._infer_clocks[-1] - self._infer_clocks[0]
            infer_hz = (len(self._infer_clocks) - 1) / span if span > 0 else 0.0
            infer_dt_ms = (span / (len(self._infer_clocks) - 1)) * 1000.0
        else:
            infer_hz = 0.0
            infer_dt_ms = 0.0
        # Lag = how far the player has advanced past the frame the boxes were
        # computed on. Sample on every HUD tick, then report min/avg/max so
        # we can tell a flat-high lag (real delay) from an oscillation
        # (sampling phase on top of a healthy 50->250ms cycle).
        if self._last_box_ts_ms >= 0:
            lag_now = max(0, int(self._player.position()) - self._last_box_ts_ms)
            self._lag_samples.append(lag_now)
        if self._lag_samples:
            lag_min = min(self._lag_samples)
            lag_max = max(self._lag_samples)
            lag_avg = sum(self._lag_samples) // len(self._lag_samples)
        else:
            lag_min = lag_max = lag_avg = 0
        text = (
            f"paint  {avg_ms:5.1f} ms   {fps:5.1f} fps\n"
            f"infer  {infer_dt_ms:5.0f} ms / cycle   {infer_hz:5.1f}/s\n"
            f"Δ      {self._last_infer_wall_ms:5.0f} ms (request->response)\n"
            f"lag    {lag_avg:4d} ms avg   [{lag_min:4d} - {lag_max:4d}]\n"
            f"boxes  {self._last_box_count:<3d}        mode {self._mode}"
        )
        self._debug_view.setText(text)
        self._debug_view.updateGeometry()

    def _on_duration_changed(self, duration: int) -> None:
        self._duration_ms = max(0, int(duration))
        self._position_slider.setRange(0, self._duration_ms)
        self._duration_label.setText(_format_ms(self._duration_ms))
        self._audio_timeline.set_duration(self._duration_ms)
        self._detection_timeline.set_duration(self._duration_ms)

    def _on_playback_state(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self._play_btn.setText("[PAUSE]" if playing else "[PLAY]")
        # The live timer only ticks while the player is actually playing — we
        # don't want to keep inferring on a paused frame.
        if self._live_active:
            if playing:
                if not self._live_timer.isActive():
                    self._live_timer.start()
            else:
                self._live_timer.stop()

    def _on_has_audio_changed(self, has_audio: bool) -> None:
        self._sync_mute_ui()
        if has_audio or self._video_path is None:
            return
        process = self._audio_waveform_process
        if process is None or process.state() == QProcess.ProcessState.NotRunning:
            self._audio_timeline.set_muted()

    def _on_player_error(self, error: QMediaPlayer.Error, error_string: str) -> None:
        if error == QMediaPlayer.Error.NoError:
            return
        msg = error_string or "Unknown playback error."
        self._status.setText(f"Player error: {msg}")
        self.errorRaised.emit(f"player: {msg}")
        self._show_playback_corruption_dialog(
            int(self._player.position()),
            recent_libav_corruption_lines(within_s=10.0),
            headline=f"Playback error: {msg}",
        )

    def _on_media_status(self, status: QMediaPlayer.MediaStatus) -> None:
        if status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._show_playback_corruption_dialog(
                int(self._player.position()),
                recent_libav_corruption_lines(within_s=10.0),
                headline="This media is invalid or corrupt and cannot be decoded.",
            )
        elif status == QMediaPlayer.MediaStatus.StalledMedia:
            self._status.setText(
                "Playback stalled (possible corrupt segment) — skipping ahead..."
            )

    def _poll_playback_corruption(self) -> None:
        """Watchdog: libav prints corruption straight to stderr without raising a
        QMediaPlayer error, so the feed just goes black. We tap those lines, mark
        the live position on the timeline, and surface a (deduped) error dialog."""
        if self._video_path is None:
            return
        if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
            return
        lines = recent_libav_corruption_lines(within_s=0.8, limit=12)
        if not lines:
            return
        pos = int(self._player.position())
        # Mark the corrupt moment on the timeline (merges into a span over time).
        self._detection_timeline.add_corrupt_ranges([(max(0, pos - 200), pos + 200)])
        self._timeline_dirty = True
        self._status.setText(
            "Corrupt packets during playback — marked on the timeline; the player "
            "skips the bad region."
        )
        self._show_playback_corruption_dialog(pos, lines)

    def _show_playback_corruption_dialog(
        self,
        pos_ms: int,
        lines: list,
        *,
        headline: str = "",
    ) -> None:
        """Non-blocking warning so the clip keeps playing the good parts. Deduped
        with a cooldown so a long corrupt stretch doesn't spawn a dialog storm."""
        now = time.perf_counter()
        if (now - self._corruption_dialog_at) < self._CORRUPTION_DIALOG_COOLDOWN_S:
            return
        self._corruption_dialog_at = now
        where = _format_ms(max(0, int(pos_ms)))
        head = headline or "Corrupt footage detected during playback."
        detail_lines = list(lines or [])
        body = (
            f"{head}\n\n"
            f"At ~{where} the decoder hit packets it could not read, so the "
            f"video may go black or skip. Playback continues past the bad region; "
            f"the affected span is marked in red on the detection timeline."
        )
        if not detail_lines:
            body += "\n\n(No decoder detail captured for this event.)"
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Corrupt video segment")
        box.setText(body)
        if detail_lines:
            box.setDetailedText("\n".join(detail_lines))
        box.setStandardButtons(QMessageBox.StandardButton.Ok)
        box.setWindowModality(Qt.WindowModality.NonModal)
        box.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        # Replace any prior still-open notice so they don't stack up.
        prior = self._corruption_box
        if prior is not None:
            try:
                prior.close()
            except Exception:
                pass
        self._corruption_box = box
        box.show()

    # ------------------------------------------------------------------
    # Detector
    # ------------------------------------------------------------------

    def _on_run_detector(self) -> None:
        if self._video_path is None:
            return
        model_path = str(self._model_combo.currentData() or "")
        if not model_path:
            self._status.setText("Pick a model first.")
            return
        if self._indexing_active():
            return
        if self._live_active:
            self._live_btn.setChecked(False)

        sample_every = int(self._sample_combo.currentData() or 5)
        device = str(self._device_combo.currentData() or "")
        index_categories = self._selected_index_categories()
        index_label_filter = str(self._index_filter_edit.text() or "").strip().lower()
        index_filter_summary = self._index_filter_summary(
            index_categories,
            index_label_filter,
        )

        dlg = VideoIndexExportDialog(
            video_path=self._video_path,
            http_get=self._http_get,
            http_post=self._http_post,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        export_dir = str(dlg.export_dir or "").strip()
        file_prefix = re.sub(r"[^a-zA-Z0-9._-]", "_", (self._video_path.stem or "video"))[:48] or "video"

        self._detection_timeline.clear_events()
        self._index_boxes.clear()
        self._index_box_keys.clear()
        self._current_boxes = []
        self._raw_index_boxes.clear()
        self._raw_current_boxes = []
        self._raw_detection_events.clear()
        self._box_overlay.clear()
        self._mode = "index"
        self._update_legend()

        # Reset run-state. _index_start_wall is captured here (before model
        # load) so the elapsed timer reflects total user-visible duration,
        # not just the scan loop.
        self._index_start_wall = time.perf_counter()
        self._index_total_detections = 0
        self._index_cancel_requested = False
        self._index_reported_total_events = None
        self._index_reported_failure = ""
        self._index_process_error_message = ""
        self._index_ui_token += 1

        process = QProcess(self)
        process.setProgram(sys.executable)
        worker_path = Path(__file__).resolve().parents[1] / "video_index_worker.py"
        args_list = [
            str(worker_path),
            "--video",
            str(self._video_path),
            "--model",
            model_path,
            "--sample-every",
            str(sample_every),
            "--device",
            device,
            "--categories",
            ",".join(index_categories),
            "--label-filter",
            index_label_filter,
        ]
        if export_dir:
            args_list.extend(
                [
                    "--export-frames-dir",
                    export_dir,
                    "--export-file-prefix",
                    file_prefix,
                ]
            )
        process.setArguments(args_list)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        process.readyReadStandardOutput.connect(self._on_index_stdout_ready)
        process.readyReadStandardError.connect(self._on_index_stderr_ready)
        process.errorOccurred.connect(self._on_index_process_error)
        process.finished.connect(self._on_index_process_finished)
        self._index_process = process
        self._index_stdout_buffer = ""
        self._index_stderr_buffer = ""
        process.start()

        self._index_progress.setRange(0, 0)  # busy/indeterminate until total known
        self._index_progress.setFormat("Loading model...")
        self._index_progress.setValue(0)
        self._index_progress.setVisible(True)
        self._index_substatus.setText("Preparing...")
        self._index_substatus.setVisible(True)
        self._run_btn.setEnabled(False)
        self._cancel_btn.setText("[CANCEL]")
        self._cancel_btn.setEnabled(True)
        self._update_transport_enabled(self._video_path is not None)
        self._status.setText(
            f"Indexing {index_filter_summary} in {self._video_path.name} with {Path(model_path).name}..."
        )

    def _on_cancel_detector(self) -> None:
        if not self._indexing_active() or self._index_cancel_requested:
            return
        self._index_cancel_requested = True
        if self._worker is not None:
            self._worker.cancel()
        self._stop_index_process(kill=False)
        self._cancel_btn.setText("[CANCELLING...]")
        self._cancel_btn.setEnabled(False)
        self._index_progress.setFormat("Cancelling...")
        self._status.setText("Cancelling detector...")

    def _on_detector_status(self, message: str) -> None:
        self._status.setText(message)

    def _on_index_stdout_ready(self) -> None:
        process = self._index_process
        if process is None:
            return
        chunk = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
        if not chunk:
            return
        self._index_stdout_buffer += chunk
        while "\n" in self._index_stdout_buffer:
            line, self._index_stdout_buffer = self._index_stdout_buffer.split("\n", 1)
            self._handle_index_worker_line(line)

    def _on_index_stderr_ready(self) -> None:
        process = self._index_process
        if process is None:
            return
        chunk = bytes(process.readAllStandardError()).decode("utf-8", "replace")
        if chunk:
            self._index_stderr_buffer = (self._index_stderr_buffer + chunk)[-4000:]

    def _handle_index_worker_line(self, line: str) -> None:
        line = str(line or "").strip()
        if not line:
            return
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            # Third-party libraries occasionally write banners to stdout.
            # Keep them out of the UI protocol; stderr is still captured for
            # real process failures.
            return
        kind = str(payload.get("type") or "")
        if kind == "status":
            self._on_detector_status(str(payload.get("message") or ""))
        elif kind == "progress":
            self._on_detector_progress(
                int(payload.get("frame_idx") or 0),
                int(payload.get("total") or 0),
            )
        elif kind == "detections_batch":
            batch = payload.get("batch")
            if isinstance(batch, list):
                self._on_frame_detections_batch(batch)
        elif kind == "corrupt_batch":
            ranges = payload.get("ranges")
            if isinstance(ranges, list) and ranges:
                self._detection_timeline.add_corrupt_ranges(ranges)
                self._timeline_dirty = True
                status_n = len(self._detection_timeline.corrupt_ranges())
                self._status.setText(
                    f"Corrupt footage detected — {status_n} span(s) marked on the timeline; scanning good frames..."
                )
        elif kind == "finished":
            self._index_reported_total_events = int(
                payload.get("total_events") or self._index_total_detections
            )
        elif kind == "export_summary":
            saved = int(payload.get("saved") or 0)
            directory = str(payload.get("directory") or "").strip()
            if directory:
                self._status.setText(
                    f"Saved {saved} frame JPEG(s) under {directory}. Open the Database tab to label and promote."
                )
        elif kind == "failed":
            self._index_reported_failure = str(payload.get("message") or "index worker failed")
            self._status.setText(f"Index failed: {self._index_reported_failure}")

    def _on_index_process_error(self, error: QProcess.ProcessError) -> None:
        self._index_process_error_message = str(getattr(error, "name", error))
        if error == QProcess.ProcessError.FailedToStart:
            self._on_detector_failed("could not start indexing worker process")

    def _on_index_process_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        process = self.sender()
        if process is self._index_process:
            self._index_process = None
        if isinstance(process, QProcess):
            try:
                chunk = bytes(process.readAllStandardOutput()).decode("utf-8", "replace")
                if chunk:
                    self._index_stdout_buffer += chunk
                if self._index_stdout_buffer.strip():
                    self._handle_index_worker_line(self._index_stdout_buffer)
                self._index_stdout_buffer = ""
                err_chunk = bytes(process.readAllStandardError()).decode("utf-8", "replace")
                if err_chunk:
                    self._index_stderr_buffer = (self._index_stderr_buffer + err_chunk)[-4000:]
            finally:
                process.deleteLater()

        if self._index_cancel_requested:
            self._on_detector_finished(self._index_total_detections)
            return
        if self._index_reported_failure:
            self._on_detector_failed(self._index_reported_failure)
            return
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            detail = self._index_process_error_message or f"exit code {exit_code}"
            stderr = self._index_stderr_buffer.strip()
            if stderr:
                detail = f"{detail}: {stderr.splitlines()[-1]}"
            self._on_detector_failed(detail)
            return
        self._on_detector_finished(
            self._index_reported_total_events
            if self._index_reported_total_events is not None
            else self._index_total_detections
        )

    def _on_frame_detections_batch(self, batch: list) -> None:
        """Index-mode bulk handler. One signal carries a full ~100ms window
        of detections so the GUI thread sees one slot call instead of dozens.
        """
        for ts_ms, detections in batch:
            self._on_frame_detections(int(ts_ms), detections)

    def _on_frame_detections(self, ts_ms: int, detections: list) -> None:
        # detections is a list of dicts (label, conf, xyxy, frame_w/h).
        now = time.perf_counter()
        self._infer_clocks.append(now)
        raw_detections = [dict(det) for det in detections]
        visible_detections = self._filter_visible_detections(raw_detections)
        self._last_box_count = len(visible_detections)
        self._last_box_ts_ms = int(ts_ms)
        if self._infer_request_wall > 0.0:
            self._last_infer_wall_ms = (now - self._infer_request_wall) * 1000.0
        labels = [str(d.get("label", "")) for d in visible_detections]
        self._raw_detection_events.append((int(ts_ms), raw_detections))
        for lbl in (str(d.get("label", "")) for d in raw_detections):
            if lbl and lbl not in self._observed_classes:
                self._observed_classes.append(lbl)
        if labels:
            self._detection_timeline.add_events(ts_ms, labels)
            self._timeline_dirty = True
        if self._mode == "index":
            self._index_total_detections += len(visible_detections)
            ts_key = int(ts_ms)
            if raw_detections:
                self._raw_index_boxes[ts_key] = list(raw_detections)
                self._index_boxes[ts_key] = list(visible_detections)
                # Maintain raw keys for nearest-neighbour lookup even when the
                # active scope hides everything at this timestamp.
                if not self._index_box_keys or self._index_box_keys[-1] < ts_key:
                    self._index_box_keys.append(ts_key)
                else:
                    import bisect
                    bisect.insort(self._index_box_keys, ts_key)
        else:
            # Live mode: latest inference replaces the overlay.
            self._raw_current_boxes = list(raw_detections)
            self._current_boxes = list(visible_detections)
            if self._show_boxes:
                self._box_overlay.set_boxes(self._current_boxes)

    def _on_boxes_toggled(self, checked: bool) -> None:
        self._show_boxes = bool(checked)
        self._box_overlay.set_enabled(self._show_boxes)
        self._boxes_btn.setText("[BOXES]" if checked else "[BOXES OFF]")
        if not self._show_boxes:
            return
        # Re-render whatever is appropriate for the current mode.
        if self._mode == "live":
            self._box_overlay.set_boxes(self._current_boxes)
        elif self._mode == "index":
            self._update_index_overlay(self._player.position())

    def _update_index_overlay(self, position_ms: int) -> None:
        if not self._show_boxes or self._mode != "index" or not self._index_box_keys:
            return
        import bisect
        idx = bisect.bisect_left(self._index_box_keys, int(position_ms))
        candidates = []
        if idx < len(self._index_box_keys):
            candidates.append(self._index_box_keys[idx])
        if idx > 0:
            candidates.append(self._index_box_keys[idx - 1])
        if not candidates:
            self._box_overlay.clear()
            return
        nearest = min(candidates, key=lambda k: abs(k - int(position_ms)))
        # Stale tolerance: ~one sample step worth at 30fps. Keeps boxes from
        # smearing across the whole timeline when the user scrubs far away.
        sample_every = int(self._sample_combo.currentData() or 5)
        tol_ms = max(500, int(sample_every * (1000 / 30) * 2))
        if abs(nearest - int(position_ms)) > tol_ms:
            self._box_overlay.clear()
            return
        self._box_overlay.set_boxes(self._index_boxes.get(nearest, []))

    def _on_detector_progress(self, frame_idx: int, total: int) -> None:
        if self._index_cancel_requested:
            # Keep the user's intent visible; let _on_detector_finished
            # write the final state.
            return
        elapsed = max(0.001, time.perf_counter() - self._index_start_wall)
        scan_fps = frame_idx / elapsed if frame_idx > 0 else 0.0
        if total > 0:
            if self._index_progress.maximum() != total:
                self._index_progress.setRange(0, total)
            self._index_progress.setValue(frame_idx)
            pct = int(100 * frame_idx / total)
            self._index_progress.setFormat(f"{frame_idx} / {total}  ({pct}%)")
            remaining = max(0, total - frame_idx)
            eta_s = int(remaining / scan_fps) if scan_fps > 0 else 0
            self._index_substatus.setText(
                f"elapsed {_format_secs(elapsed)}   "
                f"ETA {_format_secs(eta_s)}   "
                f"{scan_fps:5.1f} fps   "
                f"{self._index_total_detections} detections"
            )
            self._status.setText(f"Indexing frame {frame_idx}/{total} ({pct}%)")
        else:
            self._index_progress.setRange(0, 0)
            self._index_progress.setFormat(f"frame {frame_idx}")
            self._index_substatus.setText(
                f"elapsed {_format_secs(elapsed)}   "
                f"{scan_fps:5.1f} fps   "
                f"{self._index_total_detections} detections"
            )
            self._status.setText(f"Indexing frame {frame_idx}")

    def _on_detector_finished(self, total_events: int) -> None:
        self._teardown_worker()
        # Force a final timeline/legend repaint so end-state matches reality
        # without waiting for the next coalesced tick.
        self._timeline_dirty = True
        self._flush_timeline_repaint()
        elapsed = max(0.0, time.perf_counter() - self._index_start_wall)
        cancelled = self._index_cancel_requested
        self._index_progress.setRange(0, 1)
        self._index_progress.setValue(1)
        if cancelled:
            self._index_progress.setFormat(
                f"Cancelled — {total_events} detections so far"
            )
            self._index_substatus.setText(
                f"stopped after {_format_secs(elapsed)}   "
                f"{total_events} detections kept"
            )
            self._status.setText(
                f"Index cancelled. Kept {total_events} detections."
            )
        else:
            self._index_progress.setFormat(f"Done — {total_events} detections")
            self._index_substatus.setText(
                f"finished in {_format_secs(elapsed)}   "
                f"{total_events} detections"
            )
            self._status.setText(
                f"Index finished. {total_events} detections plotted."
            )
        self._run_btn.setEnabled(self._video_path is not None)
        self._cancel_btn.setText("[CANCEL]")
        self._cancel_btn.setEnabled(False)
        self._index_cancel_requested = False
        self._update_transport_enabled(self._video_path is not None)
        ui_token = self._index_ui_token
        QTimer.singleShot(2500, lambda token=ui_token: self._hide_index_progress(token))

    def _on_detector_failed(self, message: str) -> None:
        self._discard_index_process(kill=True)
        self._teardown_worker()
        self._index_progress.setVisible(False)
        self._index_substatus.setVisible(False)
        self._status.setText(f"Index failed: {message}")
        self.errorRaised.emit(f"detector: {message}")
        self._run_btn.setEnabled(self._video_path is not None)
        self._cancel_btn.setText("[CANCEL]")
        self._cancel_btn.setEnabled(False)
        self._index_cancel_requested = False
        self._update_transport_enabled(self._video_path is not None)

    def _hide_index_progress(self, token: int) -> None:
        if token != self._index_ui_token or self._indexing_active():
            return
        self._index_progress.setVisible(False)
        self._index_substatus.setVisible(False)

    # ------------------------------------------------------------------
    # Live detect
    # ------------------------------------------------------------------

    def _on_live_toggle(self, checked: bool) -> None:
        if checked:
            if not self._start_live_session():
                # Setup failed; bounce the toggle back without recursing.
                self._live_btn.blockSignals(True)
                self._live_btn.setChecked(False)
                self._live_btn.blockSignals(False)
                return
            self._live_btn.setText("[STOP LIVE]")
            self._live_indicator.setVisible(True)
            self._live_active = True
            self._mode = "live"
            self._current_boxes = []
            self._box_overlay.clear()
            # Auto-start playback so detections accumulate as the video plays.
            if self._player.playbackState() != QMediaPlayer.PlaybackState.PlayingState:
                self._player.play()
            else:
                self._live_timer.start()
            self._status.setText("Live detect armed. Inferring while playing...")
        else:
            self._stop_live_session()
            self._live_btn.setText("[LIVE DETECT]")
            self._live_indicator.setVisible(False)
            self._live_active = False
            self._live_timer.stop()
            self._current_boxes = []
            self._box_overlay.clear()
            self._mode = "idle"
            self._status.setText("Live detect stopped.")
        self._update_transport_enabled(self._video_path is not None)

    def _start_live_session(self) -> bool:
        if self._video_path is None:
            return False
        model_path = str(self._model_combo.currentData() or "")
        if not model_path:
            self._status.setText("Pick a model first.")
            return False
        if self._live_thread is not None:
            return True

        self._live_busy = False
        self._live_last_ts = -1
        device = str(self._device_combo.currentData() or "")

        self._live_worker = _LiveDetectorWorker()
        self._live_thread = QThread(self)
        self._live_worker.moveToThread(self._live_thread)
        self._live_worker.frame_detections.connect(self._on_frame_detections)
        self._live_worker.frame_done.connect(self._on_live_frame_done)
        self._live_worker.failed.connect(self._on_live_failed)
        self._live_worker.ready.connect(
            lambda: self._status.setText("Live detector ready.")
        )
        self._liveOpenRequested.connect(
            self._live_worker.open_session, type=Qt.ConnectionType.QueuedConnection
        )
        self._liveInferRequested.connect(
            self._live_worker.infer_at, type=Qt.ConnectionType.QueuedConnection
        )
        self._liveCloseRequested.connect(
            self._live_worker.close_session, type=Qt.ConnectionType.QueuedConnection
        )
        self._live_thread.start()
        self._liveOpenRequested.emit(str(self._video_path), model_path, device)
        return True

    def _stop_live_session(self) -> None:
        if self._live_thread is None:
            return
        try:
            self._liveCloseRequested.emit()
        except Exception:
            pass
        self._live_thread.quit()
        self._live_thread.wait(2000)
        self._live_thread.deleteLater()
        self._live_thread = None
        if self._live_worker is not None:
            self._live_worker.deleteLater()
            self._live_worker = None
        self._live_busy = False
        self._live_last_ts = -1

    def _on_live_tick(self) -> None:
        if not self._live_active or self._live_worker is None:
            return
        if self._live_busy:
            return
        # Throttle inference to roughly the chosen sample rate. With sample_every=N,
        # we infer at most every (N / fps) seconds — but never faster than the
        # tick interval and never on a duplicate timestamp.
        ts_ms = int(self._player.position())
        sample_every = int(self._sample_combo.currentData() or 5)
        model_path = str(self._model_combo.currentData() or "")
        if is_yunet_face_detector_model(model_path):
            sample_every = min(sample_every, self._LIVE_FACE_SAMPLE_CAP)
        # Assume ~30fps; if we already inferred near this ts, skip.
        min_step_ms = max(self._LIVE_TICK_MS, int(sample_every * (1000 / 30)))
        if self._live_last_ts >= 0 and abs(ts_ms - self._live_last_ts) < min_step_ms:
            return
        self._live_busy = True
        self._live_last_ts = ts_ms
        self._infer_request_wall = time.perf_counter()
        self._liveInferRequested.emit(ts_ms)

    def _on_live_frame_done(self, _ts_ms: int) -> None:
        self._live_busy = False

    def _on_live_failed(self, message: str) -> None:
        self._status.setText(f"Live detect failed: {message}")
        self.errorRaised.emit(f"live-detector: {message}")
        if self._live_active:
            self._live_btn.setChecked(False)

    def _stop_index_process(self, *, kill: bool) -> None:
        process = self._index_process
        if process is None:
            return
        if process.state() == QProcess.ProcessState.NotRunning:
            return
        if kill:
            process.kill()
            process.waitForFinished(8000)
            return
        process.terminate()
        QTimer.singleShot(
            1500,
            lambda proc=process: (
                proc.kill()
                if self._index_process is proc
                and proc.state() != QProcess.ProcessState.NotRunning
                else None
            ),
        )

    def _discard_index_process(self, *, kill: bool) -> None:
        process = self._index_process
        self._index_process = None
        if process is None:
            return
        for signal, slot in (
            (process.readyReadStandardOutput, self._on_index_stdout_ready),
            (process.readyReadStandardError, self._on_index_stderr_ready),
            (process.errorOccurred, self._on_index_process_error),
            (process.finished, self._on_index_process_finished),
        ):
            try:
                signal.disconnect(slot)
            except (TypeError, RuntimeError):
                pass
        process.blockSignals(True)
        if process.state() != QProcess.ProcessState.NotRunning:
            if kill:
                process.kill()
            else:
                process.terminate()
            process.waitForFinished(8000)
        process.blockSignals(False)
        process.deleteLater()

    def _start_audio_waveform_analysis(self, path: Path) -> None:
        self._audio_waveform_token += 1
        token = self._audio_waveform_token
        self._audio_waveform_buffer.clear()
        self._audio_timeline.set_analyzing()

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            self._audio_timeline.set_unavailable("Audio visualizer unavailable.")
            return

        process = QProcess(self)
        self._audio_waveform_process = process
        process.setProgram(ffmpeg)
        process.setArguments([
            "-v", "error",
            "-i", str(path),
            "-vn",
            "-ac", "1",
            "-ar", "80",
            "-f", "f32le",
            "pipe:1",
        ])
        process.readyReadStandardOutput.connect(
            lambda token=token: self._on_audio_waveform_stdout(token)
        )
        process.errorOccurred.connect(
            lambda _error, token=token: self._on_audio_waveform_error(token)
        )
        process.finished.connect(
            lambda exit_code, exit_status, token=token: self._on_audio_waveform_finished(
                token,
                exit_code,
                exit_status,
            )
        )
        process.start()

    def _on_audio_waveform_stdout(self, token: int) -> None:
        if token != self._audio_waveform_token:
            return
        process = self._audio_waveform_process
        if process is None:
            return
        try:
            chunk = bytes(process.readAllStandardOutput())
        except RuntimeError:
            return
        self._audio_waveform_buffer.extend(chunk)

    def _on_audio_waveform_error(self, token: int) -> None:
        if token != self._audio_waveform_token:
            return
        try:
            self._audio_timeline.set_unavailable("Audio visualizer unavailable.")
        except RuntimeError:
            pass

    def _on_audio_waveform_finished(
        self,
        token: int,
        _exit_code: int,
        _exit_status: QProcess.ExitStatus,
    ) -> None:
        if token != self._audio_waveform_token:
            return
        process = self._audio_waveform_process
        if process is not None:
            try:
                self._audio_waveform_buffer.extend(bytes(process.readAllStandardOutput()))
            except Exception:
                pass
            try:
                process.deleteLater()
            except RuntimeError:
                pass
        self._audio_waveform_process = None

        raw = bytes(self._audio_waveform_buffer)
        self._audio_waveform_buffer.clear()
        
        try:
            _ = self._audio_timeline.objectName()
        except RuntimeError:
            return

        usable = len(raw) - (len(raw) % 4)
        if usable <= 0:
            self._audio_timeline.set_muted()
            return

        samples = array("f")
        try:
            samples.frombytes(raw[:usable])
        except Exception:
            self._audio_timeline.set_unavailable("Audio visualizer unavailable.")
            return
        if not samples:
            self._audio_timeline.set_muted()
            return

        bucket_size = max(1, len(samples) // 2400)
        levels: list[float] = []
        peak = 0.0
        for idx in range(0, len(samples), bucket_size):
            bucket = samples[idx:idx + bucket_size]
            if not bucket:
                continue
            rms = math.sqrt(sum(float(v) * float(v) for v in bucket) / len(bucket))
            levels.append(rms)
            if rms > peak:
                peak = rms
        if peak <= 0.0001:
            self._audio_timeline.set_muted()
            return
        self._audio_timeline.set_levels([min(1.0, value / peak) for value in levels])

    def _discard_audio_waveform_process(self, *, kill: bool) -> None:
        self._audio_waveform_token += 1
        self._audio_waveform_buffer.clear()
        process = self._audio_waveform_process
        self._audio_waveform_process = None
        if process is None:
            return
        try:
            process.blockSignals(True)
            for sig in (
                process.readyReadStandardOutput,
                process.errorOccurred,
                process.finished,
            ):
                try:
                    sig.disconnect()
                except TypeError:
                    pass
            if process.state() != QProcess.ProcessState.NotRunning:
                if kill:
                    process.kill()
                else:
                    process.terminate()
                process.waitForFinished(8000)
            process.blockSignals(False)
            process.deleteLater()
        except RuntimeError:
            return

    def shutdown_background_processes(self) -> None:
        """Stop ffmpeg / index workers and media timers (call from QMainWindow.closeEvent).

        Child widgets do not receive QWidget.closeEvent when the main window
        closes, so this must be invoked explicitly to avoid destroying QProcess
        while ffmpeg is still running (Qt warning + possible crash on exit).
        """
        if self._worker is not None:
            try:
                self._worker.cancel()
            except Exception:
                pass
        self._discard_audio_waveform_process(kill=True)
        self._discard_index_process(kill=True)
        self._teardown_worker()
        self._live_timer.stop()
        if self._live_active or self._live_thread is not None:
            self._stop_live_session()
        self._live_active = False
        try:
            self._timeline_refresh_timer.stop()
        except Exception:
            pass
        try:
            self._player.stop()
        except Exception:
            pass

    def _teardown_worker(self) -> None:
        if self._worker_thread is not None:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
            self._worker_thread.deleteLater()
            self._worker_thread = None
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None

    # ------------------------------------------------------------------
    # Correction flagging
    # ------------------------------------------------------------------

    def _capture_current_frame(self) -> Optional[QImage]:
        """Pull the most recent decoded frame off the QVideoSink.

        We sink straight from QGraphicsVideoItem so this works whether the
        player is paused or playing. Returns None if no frame is available
        yet (e.g. before the first decode completes after seek).
        """
        sink = self._video_item.videoSink() if self._video_item else None
        if sink is None:
            return None
        frame: QVideoFrame = sink.videoFrame()
        if not frame.isValid():
            return None
        img = frame.toImage()
        if img.isNull():
            return None
        # Force ARGB32 so downstream JPEG encoders never trip on an exotic
        # source format (some decoders deliver YUV-backed QImages).
        return img.convertToFormat(QImage.Format.Format_RGB32)

    def _current_detections_for_flag(self) -> list[dict]:
        """Return the detections the user is currently looking at."""
        if self._mode == "live":
            return list(self._current_boxes)
        if self._mode == "index":
            return list(self._box_overlay._boxes)  # already nearest-neighbour selection
        return []

    # ------------------------------------------------------------------
    # External entrypoints (cross-tab handoff)
    # ------------------------------------------------------------------

    def select_model_path(self, weights_path: str) -> bool:
        """Select these weights in the model picker, adding them on the fly
        if they live outside `assets/models/`.

        Used by the Console tab's [FLAG] button so flagging picks up the
        exact weights from the run the user was inspecting, not whatever
        was last selected here.
        """
        target = str(weights_path or "").strip()
        if not target:
            return False
        idx = self._model_combo.findData(target)
        if idx < 0 and Path(target).exists():
            label = Path(target).name + "  (run)"
            self._model_combo.addItem(label, userData=target)
            self._model_combo.setEnabled(True)
            idx = self._model_combo.count() - 1
        if idx < 0:
            self._status.setText(f"Could not load weights: {target}")
            return False
        self._model_combo.setCurrentIndex(idx)
        self._update_transport_enabled(self._video_path is not None)
        self._status.setText(f"Loaded weights from run: {Path(target).name}")
        return True

    def _on_flag_frame(self) -> None:
        if self._video_path is None:
            return
        # Pause first so the captured frame matches what the user is seeing.
        was_playing = (
            self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )
        if was_playing:
            self._player.pause()
        img = self._capture_current_frame()
        if img is None:
            self._status.setText(
                "Couldn't capture frame for correction (try pausing first)."
            )
            return

        ts_ms = int(self._player.position())
        model_path = str(self._model_combo.currentData() or "")
        model_name = Path(model_path).name if model_path else "(no model)"
        detections = self._current_detections_for_flag()

        dlg = CorrectionDialog(
            frame=img,
            frame_ts_ms=ts_ms,
            video_name=self._video_path.name,
            model_name=model_name,
            model_detections=detections,
            known_classes=list(self._observed_classes),
            parent=self,
        )
        if dlg.exec() != dlg.DialogCode.Accepted or dlg.result_payload is None:
            return

        result = dlg.result_payload
        correction = corrections_store.Correction.new(
            video_path=str(self._video_path),
            frame_ts_ms=ts_ms,
            frame_w=int(img.width()),
            frame_h=int(img.height()),
            model_path=model_path,
            model_classes=list(self._observed_classes),
            model_detections=detections,
            ground_truth=result.ground_truth,
            notes=result.notes,
            kind=result.kind,
        )
        # Image must hit disk before the JSONL line so a reader never sees a
        # record with a missing snapshot.
        out_path = corrections_store.image_path(correction)
        if not img.save(str(out_path), "JPG", 90):
            self._status.setText(f"Failed to save snapshot: {out_path}")
            return
        corrections_store.append_correction(correction)
        self._status.setText(
            f"Saved correction {correction.id} ({result.kind}, "
            f"{len(result.ground_truth)} GT box(es))."
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def _on_clear_detections(self) -> None:
        self._detection_timeline.clear_events()
        self._index_boxes.clear()
        self._index_box_keys.clear()
        self._current_boxes = []
        self._raw_index_boxes.clear()
        self._raw_current_boxes = []
        self._raw_detection_events.clear()
        self._box_overlay.clear()
        self._update_legend()
        self._status.setText("Detection timeline cleared.")

    def _flush_timeline_repaint(self) -> None:
        """Coalesced timeline repaint. Runs on a timer; cheap when idle."""
        if not self._timeline_dirty:
            return
        self._timeline_dirty = False
        self._detection_timeline.update()
        self._update_legend()

    def _update_legend(self) -> None:
        labels = self._detection_timeline.labels()
        entries = [
            (label, self._detection_timeline.colour_for(label))
            for label in labels
        ]
        self._legend.set_entries(self._detection_timeline.event_count(), entries)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.shutdown_background_processes()
        super().closeEvent(event)
