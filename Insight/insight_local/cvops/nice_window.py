from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import QObject, QTimer, Qt, QUrl, pyqtSlot
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication, QFileDialog, QLabel, QMainWindow, QStyleFactory

from .service import CVOPS_DB_PATH, CvOpsServerHandle


class CvOpsNativeBridge(QObject):
    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._local_proc: Optional[subprocess.Popen[str]] = None

    @pyqtSlot(result="QVariant")
    def openLocalWorkbench(self) -> dict[str, object]:
        if self._local_proc is not None and self._local_proc.poll() is None:
            return {"ok": True, "message": "Local workbench is already running."}
        try:
            self._local_proc = subprocess.Popen(
                [sys.executable, "-m", "insight_local.cvops", "--local"],
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception as exc:
            return {"ok": False, "message": f"Could not start local workbench: {exc}"}
        return {"ok": True, "message": "Starting local workbench with --local."}

    @pyqtSlot(str, result="QVariant")
    def revealPath(self, path: str) -> dict[str, object]:
        raw = str(path or "").strip()
        if not raw:
            return {"ok": False, "message": "No path provided."}
        try:
            p = Path(raw).expanduser()
            target = p if p.exists() else p.parent
            if sys.platform == "darwin":
                subprocess.Popen(["open", "-R", str(p if p.exists() else target)])
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", "/select,", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(target)])
        except Exception as exc:
            return {"ok": False, "message": f"Could not reveal path: {exc}"}
        return {"ok": True, "message": f"Reveal requested: {raw}"}

    @pyqtSlot(result="QVariant")
    def pickFolder(self) -> dict[str, object]:
        path = QFileDialog.getExistingDirectory(None, "Select folder")
        return {"ok": bool(path), "path": path, "message": path or "No folder selected."}

    @pyqtSlot(result="QVariant")
    def pickFile(self) -> dict[str, object]:
        path, _filter = QFileDialog.getOpenFileName(None, "Select file")
        return {"ok": bool(path), "path": path, "message": path or "No file selected."}


class CvOpsNiceWindow(QMainWindow):
    def __init__(self, *, host: str, port: int, dev_url: str = "") -> None:
        super().__init__()
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{host}:{port}"
        self.entry_url = str(dev_url or "").strip() or f"{self.base_url}/nice"
        self._server = CvOpsServerHandle(host=host, port=self.port, db_path=CVOPS_DB_PATH)
        self._server.start()
        self._native_bridge = CvOpsNativeBridge(self)

        self.setObjectName("cvOpsNiceWindow")
        self.setWindowTitle("CV Ops")
        self.resize(1320, 860)
        self._web_view = None
        self._build_view()

        self._load_timer = QTimer(self)
        self._load_timer.setInterval(150)
        self._load_timer.setSingleShot(True)
        self._load_timer.timeout.connect(self._load_when_ready)
        self._load_timer.start()

    def _build_view(self) -> None:
        try:
            from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            from PyQt6.QtWebChannel import QWebChannel
        except Exception as exc:
            fallback = QLabel(
                "CV Ops --nice requires PyQt6-WebEngine.\n\n"
                f"Backend started at {self.base_url}\n"
                f"Reason: {exc}"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setWordWrap(True)
            self.setCentralWidget(fallback)
            return

        class _NicePage(QWebEnginePage):
            def javaScriptConsoleMessage(
                self, level, message, line_number, source_id
            ) -> None:  # type: ignore[override]
                text = str(message or "")
                if "generate_204" in text and "preloaded" in text:
                    return
                print(f"[CVOPS NICE JS] {source_id}:{line_number} {text}", flush=True)

        view = QWebEngineView(self)
        view.setObjectName("cvOpsNiceWebView")
        page = _NicePage(view)
        channel = QWebChannel(page)
        channel.registerObject("cvopsNative", self._native_bridge)
        page.setWebChannel(channel)
        view.setPage(page)
        settings = view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)
        self._web_view = view
        self.setCentralWidget(view)

    def _load_when_ready(self) -> None:
        if self._web_view is None:
            return
        if self.entry_url.startswith(self.base_url):
            try:
                with urllib.request.urlopen(f"{self.base_url}/health", timeout=0.5) as response:
                    if int(getattr(response, "status", 200)) >= 500:
                        raise RuntimeError("health check failed")
            except Exception:
                self._load_timer.start()
                return
        self._web_view.load(QUrl(self.entry_url))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self._server.stop()
        finally:
            super().closeEvent(event)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV Ops React gateway window")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument(
        "--dev-url",
        default="",
        help="Load a Vite/React dev server instead of the bundled /nice route.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prog = ""
        try:
            prog = str(sys.argv[0] or "").strip()
        except Exception:
            prog = ""
        app = QApplication([prog or "insight-cvops"])
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setApplicationName("CV Ops")
    font = QFont("IBM Plex Sans", 10)
    if not font.exactMatch():
        font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)

    win = CvOpsNiceWindow(host=str(args.host), port=int(args.port), dev_url=str(args.dev_url or ""))
    win.show()
    if owns_app:
        return int(app.exec())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
