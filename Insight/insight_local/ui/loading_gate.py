from __future__ import annotations
from pathlib import Path
from PyQt6.QtCore import Qt, QTimer, pyqtSignal

from .theme import beacon_title_tag_css, contrast_text_hex, current_color_scheme, get_hud_meter_css, text_css, theme_rgba
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class LoadingGateOverlay(QWidget):
    """Minimal loading gate: single bar + status text. Blinks DONE or FAILED on finish."""

    cancel_clicked = pyqtSignal()
    confirm_clicked = pyqtSignal()
    finished = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self._panel = QFrame(self)
        self._panel.setObjectName("loadingPanel")
        self._panel.setFixedWidth(340)
        outer = QVBoxLayout(self)
        outer.addStretch(13)
        mid = QHBoxLayout()
        mid.addStretch(1)
        mid.addWidget(self._panel, stretch=0)
        mid.addStretch(1)
        outer.addLayout(mid)
        outer.addStretch(7)

        pl = QVBoxLayout(self._panel)
        pl.setContentsMargins(16, 12, 16, 12)
        pl.setSpacing(8)

        self._bar = QProgressBar()
        self._bar.setRange(0, 100)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(3)
        pl.addWidget(self._bar)

        self._copy = QLabel("Preparing signaling, transport, and live video feed.")
        self._copy.setWordWrap(True)
        self._copy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(self._copy)

        act = QHBoxLayout()
        act.setContentsMargins(0, 4, 0, 0)
        self._btn_cancel = QPushButton("Stay Here")
        self._btn_confirm = QPushButton("Swap View")
        self._btn_cancel.hide()
        self._btn_confirm.hide()
        self._btn_cancel.clicked.connect(self.cancel_clicked.emit)
        self._btn_confirm.clicked.connect(self.confirm_clicked.emit)
        act.addWidget(self._btn_cancel)
        act.addWidget(self._btn_confirm)
        pl.addLayout(act)

        self._p = {"signal": 0.04, "data": 0.0, "video": 0.0, "live": 0.0}
        self._fallback = QTimer(self)
        self._fallback.setSingleShot(True)
        self._fallback.timeout.connect(self._on_fallback)
        self._blink_timer = QTimer(self)
        self._blink_state = False
        self._blink_remaining = 0
        self._blink_on_done = None
        self._failure_blink = False

        self._apply_base_style()
        self._refresh_bar()

    # -- styles ----------------------------------------------------------

    def _apply_base_style(self) -> None:
        self._panel.setStyleSheet(
            f"QFrame#loadingPanel {{ background: {theme_rgba('panel', 0.90)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.28)}; }}"
        )
        self._bar.setStyleSheet(get_hud_meter_css(1))
        self._copy.setStyleSheet(
            f"font-size: 10px; color: {text_css(0.78)}; letter-spacing: 0.5px;"
        )

    def _apply_success_style(self, on: bool) -> None:
        if on:
            self._panel.setStyleSheet(
                f"QFrame#loadingPanel {{ background: {theme_rgba('panel', 0.92)}; "
                f"border: 1px solid {theme_rgba('accent_dark', 0.54)}; }}"
            )
            self._copy.setStyleSheet(
                f"font-size: 11px; color: {text_css(0.92)}; letter-spacing: 2px; font-weight: 600;"
            )
        else:
            self._panel.setStyleSheet(
                f"QFrame#loadingPanel {{ background: {theme_rgba('panel', 0.90)}; "
                f"border: 1px solid {theme_rgba('accent_dark', 0.14)}; }}"
            )
            self._copy.setStyleSheet(
                f"font-size: 11px; color: {text_css(0.32)}; letter-spacing: 2px; font-weight: 600;"
            )

    def _apply_failure_style(self, on: bool) -> None:
        if on:
            self._panel.setStyleSheet(
                f"QFrame#loadingPanel {{ background: {theme_rgba('accent_dark', 0.94)}; "
                f"border: 1px solid {theme_rgba('pressed', 0.68)}; }}"
            )
            self._copy.setStyleSheet(
                f"font-size: 11px; color: {contrast_text_hex('accent_dark')}; letter-spacing: 2px; font-weight: 600;"
            )
        else:
            self._panel.setStyleSheet(
                f"QFrame#loadingPanel {{ background: {theme_rgba('panel', 0.90)}; "
                f"border: 1px solid {theme_rgba('accent_dark', 0.14)}; }}"
            )
            self._copy.setStyleSheet(
                f"font-size: 11px; color: {text_css(0.32)}; letter-spacing: 2px; font-weight: 600;"
            )

    def refresh_theme(self) -> None:
        self._apply_base_style()
        self._refresh_bar()

    # -- resize ----------------------------------------------------------

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.setGeometry(self.parent().rect() if self.parent() else self.rect())

    # -- progress --------------------------------------------------------

    def start_fallback(self, ms: int = 9000) -> None:
        self._fallback.start(ms)

    def stop_fallback(self) -> None:
        self._fallback.stop()

    def _on_fallback(self) -> None:
        if self._p.get("video", 0) >= 1.0 and self._p.get("signal", 0) >= 1.0:
            self.set_progress("live", 1.0)
            self.hide_with_success(blinks=1)

    def _clamp01(self, x: float) -> float:
        return max(0.0, min(1.0, x))

    def set_progress(self, key: str, value: float) -> None:
        if key in self._p:
            self._p[key] = max(self._p[key], self._clamp01(value))
        self._refresh_bar()

    def set_progress_exact(self, updates: dict[str, float]) -> None:
        for k, v in updates.items():
            if k in self._p:
                self._p[k] = self._clamp01(v)
        self._refresh_bar()

    def reset_progress(self) -> None:
        self._p = {"signal": 0.04, "data": 0.0, "video": 0.0, "live": 0.0}
        self._refresh_bar()

    def boot_ready(self) -> bool:
        return all(self._p.get(k, 0.0) >= 0.99 for k in ("signal", "data", "video", "live"))

    def _refresh_bar(self) -> None:
        avg = sum(self._p.values()) / 4.0
        self._bar.setValue(int(round(avg * 100)))

    def set_copy(self, state: str, detail: str) -> None:
        self._copy.setText(detail)

    def set_swap_actions_visible(self, visible: bool) -> None:
        self._btn_cancel.setVisible(visible)
        self._btn_confirm.setVisible(visible)

    # -- blink success ---------------------------------------------------

    def _disconnect_blink(self) -> None:
        try:
            self._blink_timer.timeout.disconnect()
        except TypeError:
            pass

    def hide_with_success(self, blinks: int = 2, on_done: object | None = None) -> None:
        self.stop_fallback()
        self._disconnect_blink()
        self._blink_on_done = on_done
        self._bar.hide()
        self._copy.setText("DONE")
        self._blink_remaining = max(0, blinks * 2)
        if self._blink_remaining == 0:
            self._finish_hide()
            return
        self._blink_timer.timeout.connect(self._blink_step)
        self._blink_timer.start(120)

    def _blink_step(self) -> None:
        self._blink_state = not self._blink_state
        self._blink_remaining -= 1
        self._apply_success_style(self._blink_state)
        if self._blink_remaining <= 0:
            self._blink_timer.stop()
            self._disconnect_blink()
            self._finish_hide()

    # -- blink failure ---------------------------------------------------

    def hide_with_failure(self, blinks: int = 3, on_done: object | None = None) -> None:
        self.stop_fallback()
        self._disconnect_blink()
        self._blink_on_done = on_done
        self._bar.hide()
        self._copy.setText("FAILED")
        self._blink_remaining = max(0, blinks * 2)
        if self._blink_remaining == 0:
            self._finish_hide()
            return
        self._failure_blink = True
        self._blink_timer.timeout.connect(self._blink_step_failure)
        self._blink_timer.start(120)

    def _blink_step_failure(self) -> None:
        self._blink_state = not self._blink_state
        self._blink_remaining -= 1
        self._apply_failure_style(self._blink_state)
        if self._blink_remaining <= 0:
            self._blink_timer.stop()
            self._disconnect_blink()
            self._failure_blink = False
            self._finish_hide()

    # -- finish ----------------------------------------------------------

    def _finish_hide(self) -> None:
        self._apply_base_style()
        self._bar.show()
        self.hide()
        self.set_swap_actions_visible(False)
        cb = self._blink_on_done
        self._blink_on_done = None
        if callable(cb):
            cb()
        self.finished.emit()


class BootSelectorOverlay(QWidget):
    """Full-window startup overlay shown before session boot.

    Emits source_selected("camera", int) or source_selected("video", Path)
    when the user picks an input source.
    """

    source_selected = pyqtSignal(str, object)  # ("camera", index) | ("video", Path)

    def __init__(self, default_video_path: Path, parent=None) -> None:
        super().__init__(parent)
        self._default_video_path = default_video_path

        # Semi-transparent scrim over the whole window.
        if current_color_scheme() == "fire":
            self.setStyleSheet(f"background: {theme_rgba('pressed', 0.72)};")
        else:
            self.setStyleSheet("background: rgba(6, 2, 2, 0.78);")
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

        self._panel = QFrame(self)
        self._panel.setObjectName("bootPanel")
        self._panel.setFixedWidth(390)

        outer = QVBoxLayout(self)
        outer.addStretch(2)
        mid = QHBoxLayout()
        mid.addStretch(1)
        mid.addWidget(self._panel, stretch=0)
        mid.addStretch(1)
        outer.addLayout(mid)
        outer.addStretch(3)

        pl = QVBoxLayout(self._panel)
        pl.setContentsMargins(22, 18, 22, 18)
        pl.setSpacing(10)

        self._title = QLabel("ATLAS HUD")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub = QLabel("select an input source to begin")
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(self._title)
        pl.addWidget(self._sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {theme_rgba('accent_dark', 0.22)}; background: transparent;")
        pl.addWidget(sep)

        # [CAMERA] Live feed row
        cam_row = QHBoxLayout()
        cam_row.setSpacing(8)
        self._btn_camera = QPushButton("[C] Live Camera")
        self._btn_camera.clicked.connect(self._select_camera)
        cam_row.addWidget(self._btn_camera, stretch=1)

        self._cam_index = QSpinBox()
        self._cam_index.setRange(0, 9)
        self._cam_index.setValue(0)
        self._cam_index.setFixedWidth(54)
        self._cam_index.setToolTip("Camera device index")
        cam_row.addWidget(self._cam_index)
        pl.addLayout(cam_row)

        # [VIDEO] Sample demo row
        self._btn_video = QPushButton("[V] Sample Video")
        self._btn_video.clicked.connect(self._select_video)
        pl.addWidget(self._btn_video)

        # [FILE] Custom file row
        self._btn_file = QPushButton("[F] Open Video File...")
        self._btn_file.clicked.connect(self._select_file)
        pl.addWidget(self._btn_file)

        self._apply_style()

    def _apply_style(self) -> None:
        if current_color_scheme() == "fire":
            self.setStyleSheet(f"background: {theme_rgba('pressed', 0.72)};")
        else:
            self.setStyleSheet("background: rgba(6, 2, 2, 0.78);")

        if current_color_scheme() == "beacon":
            self._title.setStyleSheet(beacon_title_tag_css(font_size=12, padding="6px 12px"))
        else:
            self._title.setStyleSheet(
                f"font-size: 14px; font-weight: 700; color: {text_css(0.90)}; "
                "letter-spacing: 3px; background: transparent;"
            )
        self._sub.setStyleSheet(
            f"font-size: 10px; color: {text_css(0.58)}; "
            "letter-spacing: 0.8px; background: transparent;"
        )
        self._panel.setStyleSheet(
            f"QFrame#bootPanel {{ background: {theme_rgba('panel', 0.94)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.34)}; }}"
        )
        _btn = (
            f"QPushButton {{ border: 1px solid {theme_rgba('accent_dark', 0.22)}; "
            f"background: {theme_rgba('panel', 0.36)}; color: {text_css(0.76)}; "
            f"padding: 8px 14px; font-size: 11px; text-align: left; }}"
            f"QPushButton:hover {{ background: {theme_rgba('panel', 0.60)}; "
            f"color: {text_css(0.92)}; border-color: {theme_rgba('accent_dark', 0.50)}; }}"
            f"QPushButton:pressed {{ background: {theme_rgba('accent_dark', 0.60)}; "
            f"color: {contrast_text_hex('accent_dark')}; }}"
        )
        _spin = (
            f"QSpinBox {{ border: 1px solid {theme_rgba('accent_dark', 0.22)}; "
            f"background: {theme_rgba('panel', 0.36)}; color: {text_css(0.76)}; "
            f"padding: 4px 6px; font-size: 11px; }}"
        )
        self._btn_camera.setStyleSheet(_btn)
        self._btn_video.setStyleSheet(_btn)
        self._btn_file.setStyleSheet(_btn)
        self._cam_index.setStyleSheet(_spin)

    def refresh_theme(self) -> None:
        self._apply_style()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.parent():
            self.setGeometry(self.parent().rect())

    def _select_camera(self) -> None:
        self.source_selected.emit("camera", self._cam_index.value())

    def _select_video(self) -> None:
        self.source_selected.emit("video", self._default_video_path)

    def _select_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv *.m4v *.webm *.ts);;All Files (*)",
        )
        if path:
            self.source_selected.emit("video", Path(path))
