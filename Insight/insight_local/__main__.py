from __future__ import annotations
from pathlib import Path
import importlib.abc
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

# Allow direct execution via:
#   python /path/to/Insight/insight_local/__main__.py
# as well as package execution:
#   python -m insight_local
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PyQt6.QtCore import QRect, QRectF, QTimer, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QFontDatabase, QLinearGradient, QPainter, QPen
from PyQt6.QtWidgets import QApplication
from insight_local.config import CVOPS_BASE_URL, CVOPS_HOST, CVOPS_PORT, parse_args
from insight_local.export_apple import main as export_runtime_main


def _preinit_webengine() -> None:
    """Qt WebEngine requires AA_ShareOpenGLContexts set and QtWebEngineWidgets
    imported BEFORE QApplication is constructed. Without this, QWebEngineView
    fails to instantiate at runtime and grid cells fall back to a text label.

    Also force Chromium GPU acceleration so video playback (e.g. YouTube)
    doesn't stutter — without these flags it falls back to software rasterizing
    every frame on the main thread."""
    try:
        existing = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        flags = [
            "--enable-gpu-rasterization",
            "--enable-zero-copy",
            "--ignore-gpu-blocklist",
            "--enable-accelerated-video-decode",
            "--enable-features=VaapiVideoDecoder",
        ]
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (existing + " " + " ".join(flags)).strip()
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        import PyQt6.QtWebEngineWidgets as _webengine  # noqa: F401
        _ = _webengine
    except Exception:
        pass


class _BlockAvFinder(importlib.abc.MetaPathFinder):
    """Prevent PyAV import in insight_local to avoid cv2/av FFmpeg dylib clashes."""

    def find_spec(self, fullname: str, path, target=None):
        if fullname == "av" or fullname.startswith("av."):
            raise ModuleNotFoundError(
                "PyAV is disabled for insight_local (cv2/av dylib conflict mitigation)"
            )
        return None


def _install_av_blocker() -> None:
    # Allow opt-out for debugging: INSIGHT_BLOCK_PYAV=0
    if str(os.environ.get("INSIGHT_BLOCK_PYAV", "1")).strip() in {"0", "false", "False"}:
        return
    if "av" in sys.modules:
        sys.modules.pop("av", None)
    if not any(isinstance(f, _BlockAvFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _BlockAvFinder())


def _cvops_health_url(base_url: str) -> str:
    value = str(base_url or "").strip()
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}/health"
    return f"{value.rstrip('/')}/health"


def _cvops_is_reachable(base_url: str, timeout: float = 0.75) -> bool:
    url = _cvops_health_url(base_url)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return int(getattr(resp, "status", 0) or 0) == 200
    except Exception:
        return False


def _cvops_target_host_port(base_url: str) -> tuple[str, int]:
    parsed = urllib.parse.urlparse(str(base_url or "").strip())
    host = parsed.hostname or str(CVOPS_HOST or "127.0.0.1")
    port = int(parsed.port or CVOPS_PORT or 8787)
    return host, port


def _is_local_host(host: str) -> bool:
    h = str(host or "").strip().lower()
    return h in {"127.0.0.1", "localhost", "::1"}


def _wait_for_cvops(base_url: str, timeout_sec: float = 6.0) -> bool:
    deadline = time.monotonic() + max(0.5, float(timeout_sec))
    while time.monotonic() < deadline:
        if _cvops_is_reachable(base_url, timeout=0.7):
            return True
        time.sleep(0.2)
    return _cvops_is_reachable(base_url, timeout=0.7)


class _InsightLaunchSplash:
    """Launch splash shown while Insight Local bootstraps services and UI."""

    _RUN_BLUES = ("#2bd9ff", "#22b8f0", "#1d8ed8", "#55d9ff", "#0a8fa8", "#5a9fd6")

    def __init__(self) -> None:
        from PyQt6.QtWidgets import QWidget

        self._widget = QWidget(
            None,
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self._widget.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._widget.setFixedSize(420, 260)
        self._tick = 0
        self._timer = QTimer(self._widget)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._on_tick)
        self._widget.paintEvent = self._paint_event  # type: ignore[method-assign]

    def start(self) -> None:
        self._center()
        self._timer.start()
        self._widget.show()
        self._widget.raise_()

    def finish(self) -> None:
        self._timer.stop()
        self._widget.hide()
        self._widget.deleteLater()

    def _center(self) -> None:
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen is not None else QRect(0, 0, 1180, 800)
        self._widget.move(
            geo.x() + max(0, (geo.width() - self._widget.width()) // 2),
            geo.y() + max(0, (geo.height() - self._widget.height()) // 2),
        )

    def _on_tick(self) -> None:
        self._tick += 1
        self._widget.update()

    def _paint_event(self, event) -> None:
        del event
        painter = QPainter(self._widget)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(1, 1, self._widget.width() - 2, self._widget.height() - 2)
        panel = QLinearGradient(rect.topLeft(), rect.bottomRight())
        panel.setColorAt(0.0, QColor(9, 15, 20, 242))
        panel.setColorAt(0.55, QColor(11, 20, 26, 246))
        panel.setColorAt(1.0, QColor(14, 27, 34, 238))
        painter.setBrush(QBrush(panel))
        painter.setPen(QPen(QColor(94, 131, 145, 220), 1.25))
        painter.drawRect(rect)
        painter.setPen(QPen(QColor(62, 91, 103, 180), 1.0))
        painter.drawLine(18, 38, self._widget.width() - 18, 38)
        painter.drawLine(18, self._widget.height() - 36, self._widget.width() - 18, self._widget.height() - 36)
        self._draw_matrix(painter)
        self._draw_text(painter)
        painter.end()

    def _draw_matrix(self, painter: QPainter) -> None:
        cx = self._widget.width() / 2.0
        cy = 96.0
        pulse = (self._tick % 16) / 16.0
        side = 122.0 + (2.0 if pulse < 0.5 else 0.0)
        x0 = cx - side / 2.0
        y0 = cy - side / 2.0
        outer = QRectF(x0, y0, side, side)
        outer_fill = QLinearGradient(outer.topLeft(), outer.bottomRight())
        outer_fill.setColorAt(0.0, QColor("#041723"))
        outer_fill.setColorAt(1.0, QColor("#0b2b3a"))
        painter.setPen(QPen(QColor("#8ceaff"), 1.2))
        painter.setBrush(QBrush(outer_fill))
        painter.drawRect(outer)
        cells = 6
        gap = 3.0
        inner_margin = 8.0
        inner_side = side - inner_margin * 2.0
        cell_side = (inner_side - gap * (cells - 1)) / cells
        x_cell = x0 + inner_margin
        y_cell = y0 + inner_margin
        painter.setPen(Qt.PenStyle.NoPen)
        for row in range(cells):
            for col in range(cells):
                r = random.random()
                if r > 0.7:
                    color = QColor(random.choice(self._RUN_BLUES))
                    color.setAlpha(220)
                elif r > 0.34:
                    color = QColor("#13465e")
                    color.setAlpha(180)
                else:
                    color = QColor("#0a2735")
                    color.setAlpha(148)
                painter.setBrush(QBrush(color))
                painter.drawRect(
                    QRectF(
                        x_cell + col * (cell_side + gap),
                        y_cell + row * (cell_side + gap),
                        cell_side,
                        cell_side,
                    )
                )
        glow = QColor(random.choice(self._RUN_BLUES))
        glow.setAlpha(72)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(glow, 3.5))
        painter.drawRect(outer.adjusted(-2, -2, 2, 2))

    def _draw_text(self, painter: QPainter) -> None:
        title_font = QFont("IBM Plex Mono", 16)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor(220, 237, 246))
        painter.drawText(QRectF(0, 160, self._widget.width(), 28), Qt.AlignmentFlag.AlignCenter, "test: loading")


def _maybe_start_cvops_service():
    """Start embedded CV Ops API if not already running."""
    if str(os.environ.get("INSIGHT_AUTO_START_CVOPS", "1")).strip() in {"0", "false", "False"}:
        return None 
    if _cvops_is_reachable(CVOPS_BASE_URL):
        return None
    host, port = _cvops_target_host_port(CVOPS_BASE_URL)
    if not _is_local_host(host):
        # Respect remote CVOPS_URL targets; don't spin up a local daemon.
        return None
    try:
        from insight_local.cvops.service import CvOpsServerHandle
    except Exception as exc:
        print(f"[insight_local] CV Ops import failed: {exc}", flush=True)
        return None
    try:
        handle = CvOpsServerHandle(host=host, port=port)
        handle.start()
        if not _wait_for_cvops(CVOPS_BASE_URL, timeout_sec=float(os.environ.get("INSIGHT_CVOPS_BOOT_WAIT_SEC", "6.0"))):
            print(
                f"[insight_local] CV Ops started on {host}:{port} but health did not become ready in time.",
                flush=True,
            )
        return handle
    except Exception as exc:
        print(f"[insight_local] CV Ops start failed: {exc}", flush=True)
        return None


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] in {"export-apple", "export-runtime"}:
        raise SystemExit(export_runtime_main(sys.argv[2:]))
    # Install before cvops/main imports so any transitive `av` import is blocked.
    _install_av_blocker()
    if len(sys.argv) > 1 and sys.argv[1] == "cvops":
        from insight_local.cvops.window import main as cvops_main

        raise SystemExit(cvops_main(sys.argv[2:]))
    _preinit_webengine()
    config = parse_args()
    app = QApplication(sys.argv)
    app.setApplicationName("Insight Local")
    from insight_local.ui.theme import (
        apply_text_palette,
        configure_color_scheme,
        configure_text_mode,
        install_text_palette_filter,
    )

    configure_color_scheme(config.color_scheme)
    configure_text_mode(config.text_color)
    install_text_palette_filter(app)
    preferred_fonts = (
        "Roboto",
        "Google Sans",
        "Segoe UI",
        "Helvetica Neue",
        "SF Pro Text",
        "Arial",
    )
    installed_families = set(QFontDatabase.families())
    chosen_family = next((name for name in preferred_fonts if name in installed_families), "")
    ui_font = QFont(chosen_family, 13) if chosen_family else QFont()
    if not chosen_family:
        ui_font.setPointSize(13)
    ui_font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(ui_font)
    splash = _InsightLaunchSplash()
    splash.start()
    cvops_result: dict[str, object] = {"handle": None}
    cvops_thread_done = threading.Event()
    state: dict[str, object] = {"win": None}

    def _boot_cvops_worker() -> None:
        try:
            cvops_result["handle"] = _maybe_start_cvops_service()
        finally:
            cvops_thread_done.set()

    cvops_thread = threading.Thread(target=_boot_cvops_worker, name="cvops-bootstrap", daemon=True)
    cvops_thread.start()

    def _build_main_window() -> None:
        from insight_local.ui.main_window import MainWindow

        win = MainWindow(config)
        apply_text_palette(win)
        win.show()
        splash.finish()
        state["win"] = win

    # Let the event loop start first so splash animation is live during boot.
    QTimer.singleShot(0, _build_main_window)
    exit_code = int(app.exec())
    if not cvops_thread_done.is_set():
        cvops_thread.join(timeout=1.5)
    cvops_handle = cvops_result.get("handle")
    if cvops_handle is not None:
        try:
            cvops_handle.stop()
        except Exception:
            pass
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
