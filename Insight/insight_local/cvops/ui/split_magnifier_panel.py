from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QGuiApplication, QImage, QMouseEvent, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...ui.theme import text_css, theme_rgba
from .test_range_subroutine import (
    SubroutineControlsWidget,
    SubroutineSession,
    qimage_to_bgr_ndarray,
)


HttpCall = Callable[..., Any]


class _SplitMagnifierLens(QFrame):
    """Transparent capture surface for the split magnifier."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("splitMagnifierLens")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAutoFillBackground(False)
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setMouseTracking(True)
        self.setStyleSheet(
            "QFrame#splitMagnifierLens { background: transparent; border: none; }"
        )

    def paintEvent(self, event) -> None:  # type: ignore[override]
        event.accept()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(2, 2, -3, -3)
        if rect.width() <= 0 or rect.height() <= 0:
            painter.end()
            return
        line = QColor(34, 211, 238, 190)
        # Leave the capture area unpainted so the desktop/app below reads through.
        painter.setPen(QPen(line, 2, Qt.PenStyle.DashLine))
        painter.drawRect(rect)
        painter.end()


class SplitMagnifierWindow(QWidget):
    """Top-level transparent Range panel for screen-surface inference.

    The left split pane is a transparent lens placed over any visible desktop
    surface. Capture hides the window briefly, grabs the lens rectangle from the
    underlying screen, and feeds that image to the existing Range subroutine
    controls.
    """

    def __init__(
        self,
        *,
        http_get: Optional[HttpCall] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Split Magnifier")
        self.setObjectName("splitMagnifierWindow")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAutoFillBackground(False)
        self.setStyleSheet("QWidget#splitMagnifierWindow { background: transparent; }")
        self.setMinimumSize(760, 420)
        self.resize(1040, 560)

        self._http_get = http_get
        self._capture_pending = False
        self._capture_auto_run = True
        self._last_capture = QImage()
        self._drag_origin: Optional[QPoint] = None
        self._session = SubroutineSession(self)

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        header = QFrame()
        header.setObjectName("splitMagnifierHeader")
        header.setStyleSheet(
            "QFrame#splitMagnifierHeader {"
            f" background: {theme_rgba('panel', 0.82)};"
            f" border: 1px solid {theme_rgba('accent_dark', 0.32)};"
            "}"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(8, 5, 8, 5)
        header_layout.setSpacing(8)

        title = QLabel("SPLIT MAGNIFIER")
        title.setProperty("isTitle", True)
        header_layout.addWidget(title)

        self._status = QLabel("Place the transparent lens over another app, then capture.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet(f"font-size: 10px; color: {text_css(0.78)}; border: none;")
        header_layout.addWidget(self._status, stretch=1)

        self._auto_run = QCheckBox("Run after capture")
        self._auto_run.setChecked(True)
        self._auto_run.setStyleSheet(f"font-size: 10px; color: {text_css(0.84)};")
        header_layout.addWidget(self._auto_run)

        self._capture_btn = QPushButton("[CAPTURE]")
        self._capture_btn.setToolTip("Hide this panel, capture the lens area underneath it, then optionally run selected model(s).")
        self._capture_btn.clicked.connect(lambda: self.capture_lens(auto_run=self._auto_run.isChecked()))
        header_layout.addWidget(self._capture_btn)

        capture_only_btn = QPushButton("[CAPTURE ONLY]")
        capture_only_btn.clicked.connect(lambda: self.capture_lens(auto_run=False))
        header_layout.addWidget(capture_only_btn)

        close_btn = QPushButton("[CLOSE]")
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)
        root.addWidget(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setObjectName("splitMagnifierSplitter")
        splitter.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        splitter.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        splitter.setAutoFillBackground(False)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            "QSplitter#splitMagnifierSplitter { background: transparent; border: none; }"
            f"QSplitter#splitMagnifierSplitter::handle {{ background: {theme_rgba('accent_dark', 0.20)}; }}"
            f"QSplitter#splitMagnifierSplitter::handle:hover {{ background: {theme_rgba('accent_dark', 0.34)}; }}"
        )
        root.addWidget(splitter, stretch=1)

        self._lens = _SplitMagnifierLens()
        splitter.addWidget(self._lens)

        controls = QFrame()
        controls.setObjectName("splitMagnifierControls")
        controls.setMinimumWidth(360)
        controls.setStyleSheet(
            "QFrame#splitMagnifierControls {"
            f" background: {theme_rgba('panel', 0.86)};"
            f" border: 1px solid {theme_rgba('accent_dark', 0.28)};"
            "}"
        )
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(8, 8, 8, 8)
        controls_layout.setSpacing(6)
        hint = QLabel("Captured surface")
        hint.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {text_css(0.70)};")
        controls_layout.addWidget(hint)
        self._subroutine_panel = SubroutineControlsWidget(http_get=http_get, parent=self)
        self._subroutine_panel.runRequested.connect(self._on_subroutine_run)
        self._subroutine_panel.fusionRequested.connect(self._on_subroutine_fusion)
        self._subroutine_panel.reAnalyzeRequested.connect(self._on_reanalyze)
        self._subroutine_panel.dismissed.connect(self._on_subroutine_dismissed)
        controls_layout.addWidget(self._subroutine_panel, stretch=1)
        splitter.addWidget(controls)
        splitter.setSizes([620, 420])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_for_parent(self, parent: Optional[QWidget]) -> None:
        if parent is not None and not self.isVisible():
            top_left = parent.mapToGlobal(QPoint(40, 80))
            self.move(top_left)
        self.show()
        self.raise_()
        self.activateWindow()

    def refresh_models(self) -> None:
        self._subroutine_panel.refresh_models()

    def capture_lens(self, *, auto_run: bool) -> None:
        if self._capture_pending:
            return
        rect = self._lens_global_rect()
        if rect.width() < 16 or rect.height() < 16:
            self._status.setText("Lens is too small to capture.")
            return
        self._capture_pending = True
        self._capture_auto_run = bool(auto_run)
        self._status.setText("Capturing lens area...")
        self.hide()
        QTimer.singleShot(140, lambda r=QRect(rect): self._grab_hidden_lens(r))

    # ------------------------------------------------------------------
    # Capture + inference
    # ------------------------------------------------------------------

    def _lens_global_rect(self) -> QRect:
        origin = self._lens.mapToGlobal(QPoint(0, 0))
        return QRect(origin, self._lens.size())

    def _grab_hidden_lens(self, rect: QRect) -> None:
        try:
            image = self._grab_screen_rect(rect)
        finally:
            self.show()
            self.raise_()
            self.activateWindow()
            self._capture_pending = False
        if image.isNull():
            self._status.setText("Screen capture failed. Check OS screen-recording permission.")
            self._subroutine_panel.show_error("Screen capture failed.")
            return

        self._last_capture = image
        self._subroutine_panel.open_for_roi(raw_crop=image)
        self._status.setText(f"Captured {image.width()} x {image.height()} px from the lens.")
        if self._capture_auto_run:
            self._run_active_selection()

    @staticmethod
    def _grab_screen_rect(rect: QRect) -> QImage:
        center = rect.center()
        screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
        if screen is None:
            return QImage()
        pix = screen.grabWindow(0, rect.x(), rect.y(), rect.width(), rect.height())
        if pix.isNull():
            return QImage()
        return pix.toImage().convertToFormat(QImage.Format.Format_RGB32)

    def _run_active_selection(self) -> None:
        if self._subroutine_panel.has_fusion_set():
            self._on_subroutine_fusion(self._subroutine_panel.fusion_paths())
            return
        model_path = str(self._subroutine_panel.current_model_path()).strip()
        if not model_path:
            self._status.setText("Capture ready. Open the model catalog and pick a model.")
            return
        self._on_subroutine_run(model_path, "")

    def _current_crop(self) -> Optional[QImage]:
        crop = self._subroutine_panel.raw_crop()
        if crop is not None and not crop.isNull():
            return crop
        if not self._last_capture.isNull():
            return self._last_capture
        return None

    def _on_subroutine_run(self, model_path: str, _device: str) -> None:
        crop = self._current_crop()
        if crop is None or crop.isNull():
            self._subroutine_panel.show_error("Capture the lens area first.")
            return
        try:
            crop_bgr = qimage_to_bgr_ndarray(crop)
        except Exception as exc:
            self._subroutine_panel.show_error(f"Capture conversion failed: {exc}")
            return

        self._subroutine_panel.show_running()
        self._status.setText(f"Running {Path(model_path).name} on captured surface...")
        crop_ref = QImage(crop)

        def _done(detections: list) -> None:
            self._status.setText(f"Surface inference complete: {len(detections or [])} detection(s).")
            self._subroutine_panel.show_results(
                crop_image=crop_ref,
                detections=list(detections or []),
                frame_w=crop_ref.width(),
                frame_h=crop_ref.height(),
            )

        def _fail(message: str) -> None:
            self._status.setText(f"Inference failed: {message}")
            self._subroutine_panel.show_error(message)

        self._session.start(
            model_path=model_path,
            device="",
            frame_bgr=crop_bgr,
            on_finished=_done,
            on_failed=_fail,
        )

    def _on_subroutine_fusion(self, model_paths: list) -> None:
        paths = [str(path) for path in model_paths if path]
        if len(paths) < 2:
            self._subroutine_panel.show_error("Pick at least two models for fusion.")
            return
        crop = self._current_crop()
        if crop is None or crop.isNull():
            self._subroutine_panel.show_error("Capture the lens area first.")
            return
        try:
            crop_bgr = qimage_to_bgr_ndarray(crop)
        except Exception as exc:
            self._subroutine_panel.show_error(f"Capture conversion failed: {exc}")
            return

        crop_ref = QImage(crop)
        streaming = self._subroutine_panel.is_stream_enabled()
        per_model: list[tuple[str, list[dict]]] = []
        self._status.setText(f"Running fusion ({len(paths)}) on captured surface...")

        if streaming:
            self._subroutine_panel.begin_fusion_stream(
                crop_image=crop_ref,
                model_labels=[Path(p).name for p in paths],
                frame_w=crop_ref.width(),
                frame_h=crop_ref.height(),
            )
        else:
            self._subroutine_panel.show_running()

        def _each(path: str, detections: list) -> None:
            label = Path(path).name
            dets = list(detections or [])
            per_model.append((label, dets))
            if streaming:
                self._subroutine_panel.append_fusion_model(label, dets)

        def _each_failed(path: str, message: str) -> None:
            label = Path(path).name
            per_model.append((label, []))
            if streaming:
                self._subroutine_panel.append_fusion_model(label, [])
            self._status.setText(f"Fusion model failed: {label}: {message}")

        def _all_done() -> None:
            total = sum(len(dets) for _label, dets in per_model)
            self._status.setText(f"Surface fusion complete: {total} detection(s).")
            if streaming:
                self._subroutine_panel.finish_fusion_stream()
            else:
                self._subroutine_panel.show_fusion_results(
                    crop_image=crop_ref,
                    per_model=per_model,
                    frame_w=crop_ref.width(),
                    frame_h=crop_ref.height(),
                )

        self._session.start_multi(
            model_paths=paths,
            device="",
            frame_bgr=crop_bgr,
            on_each=_each,
            on_each_failed=_each_failed,
            on_all_done=_all_done,
        )

    def _on_reanalyze(self) -> None:
        self.capture_lens(auto_run=True)

    def _on_subroutine_dismissed(self) -> None:
        self._session.stop()
        self._last_capture = QImage()
        self._status.setText("Place the transparent lens over another app, then capture.")

    # ------------------------------------------------------------------
    # Qt overrides
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._session.stop()
        super().closeEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._drag_origin is not None and event.buttons() & Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_origin)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        self._drag_origin = None
        super().mouseReleaseEvent(event)
