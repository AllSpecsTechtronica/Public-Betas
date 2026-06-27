from __future__ import annotations
import importlib.abc
import json
import os
from pathlib import Path
import sys
import threading
import time
# Allow direct execution via:
#   python /path/to/Insight/insight_local/cvops/__main__.py
# as well as package execution:
#   python -m insight_local.cvops
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_BOOT_T0: float = time.perf_counter()
_BOOT_TOTAL_STEPS = 23   # total _boot_step() calls across the startup chain
_BAR_WIDTH = 30          # number of block characters in the filled/empty track
_BAR_LABEL_MAX = 36      # max characters for the stage label


class _BootTrace:
    """Small JSON timing ledger for CV Ops startup."""

    def __init__(self, t0: float) -> None:
        self._t0 = t0
        self._lock = threading.Lock()
        self._marks: list[dict[str, object]] = []
        self._seen_once: set[str] = set()
        self._path: Path | None = None
        self.mark("process_start", wall_time=time.time(), once=True)

    def mark(self, name: str, *, once: bool = False, **fields: object) -> None:
        label = str(name or "").strip()
        if not label:
            return
        now_perf = time.perf_counter()
        now_wall = time.time()
        with self._lock:
            if once and label in self._seen_once:
                return
            if once:
                self._seen_once.add(label)
            prev_ms = float(self._marks[-1]["t_ms"]) if self._marks else 0.0
            t_ms = (now_perf - self._t0) * 1000.0
            mark: dict[str, object] = {
                "name": label,
                "t_ms": round(t_ms, 3),
                "delta_ms": round(t_ms - prev_ms, 3),
                "wall_time": round(now_wall, 6),
            }
            mark.update(fields)
            self._marks.append(mark)
            self._write_locked()

    def _trace_path(self) -> Path | None:
        if self._path is not None:
            return self._path
        try:
            from insight_local.cvops.paths import CVOPS_STATE_DIR  # noqa: PLC0415
        except Exception:
            return None
        self._path = CVOPS_STATE_DIR / "boot_trace.json"
        return self._path

    def _write_locked(self) -> None:
        path = self._trace_path()
        if path is None:
            return
        payload = {
            "version": 1,
            "pid": os.getpid(),
            "started_at": self._marks[0].get("wall_time") if self._marks else time.time(),
            "updated_at": round(time.time(), 6),
            "elapsed_ms": self._marks[-1].get("t_ms") if self._marks else 0.0,
            "marks": list(self._marks),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
            tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            tmp.replace(path)
        except Exception:
            pass


class _BootBar:
    """Single-line overwriting terminal progress bar for startup sequencing."""

    _FULL  = "█"
    _EMPTY = "░"
    _DONE_MSG = "ready"

    def __init__(self, total_steps: int) -> None:
        self._total   = total_steps
        self._step    = 0
        self._label   = "starting..."
        self._done    = False
        self._t0      = time.perf_counter()
        self._lock    = threading.Lock()
        self._stream  = sys.stderr
        self._tty     = bool(getattr(self._stream, "isatty", lambda: False)())
        self._last_line_len = 0
        self._qt_handler_installed = False
        self._qt_prev_handler = None
        self._deferred_qt_logs: list[str] = []

    def install_qt_message_handler(self) -> None:
        if self._qt_handler_installed:
            return
        try:
            from PyQt6.QtCore import qInstallMessageHandler  # noqa: PLC0415
        except Exception:
            return

        def _handler(_msg_type, _context, message) -> None:
            text = " ".join(str(message or "").split())
            if not text:
                return
            with self._lock:
                if self._done:
                    self._emit_log_line(text)
                    return
                if not self._deferred_qt_logs or self._deferred_qt_logs[-1] != text:
                    self._deferred_qt_logs.append(text)

        self._qt_prev_handler = qInstallMessageHandler(_handler)
        self._qt_handler_installed = True

    def _restore_qt_message_handler(self) -> None:
        if not self._qt_handler_installed:
            return
        try:
            from PyQt6.QtCore import qInstallMessageHandler  # noqa: PLC0415
        except Exception:
            return
        qInstallMessageHandler(self._qt_prev_handler)
        self._qt_prev_handler = None
        self._qt_handler_installed = False

    # ------------------------------------------------------------------
    def step(self, label: str) -> None:
        with self._lock:
            if self._done:
                return
            self._step  = min(self._step + 1, self._total - 1)
            self._label = label
            self._render()

    def finish(self) -> None:
        with self._lock:
            if self._done:
                return
            self._done = True
            elapsed_ms = (time.perf_counter() - self._t0) * 1000
            bar = self._FULL * _BAR_WIDTH
            self._write_progress_line(
                f"CV Ops  [{bar}]  {self._DONE_MSG}  {elapsed_ms:.0f}ms",
                newline=True,
            )
            self._restore_qt_message_handler()
            for text in self._deferred_qt_logs:
                self._emit_log_line(text)
            self._deferred_qt_logs.clear()

    # ------------------------------------------------------------------
    def _render(self) -> None:
        pct      = int(self._step * 100 / self._total)
        filled   = int(_BAR_WIDTH * pct / 100)
        bar      = self._FULL * filled + self._EMPTY * (_BAR_WIDTH - filled)
        label    = self._label[:_BAR_LABEL_MAX].ljust(_BAR_LABEL_MAX)
        self._write_progress_line(f"CV Ops  [{bar}]  {pct:2d}%  {label}")

    def _write_progress_line(self, text: str, *, newline: bool = False) -> None:
        clear_prefix = "\r\x1b[2K" if self._tty else "\r"
        pad = ""
        if not self._tty and len(text) < self._last_line_len:
            pad = " " * (self._last_line_len - len(text))
        self._stream.write(clear_prefix + text + pad + ("\n" if newline else ""))
        self._stream.flush()
        self._last_line_len = 0 if newline else len(text)

    def _emit_log_line(self, text: str) -> None:
        self._stream.write(str(text).rstrip() + "\n")
        self._stream.flush()


_bar = _BootBar(_BOOT_TOTAL_STEPS)
_trace = _BootTrace(_BOOT_T0)


def _boot_step(msg: str) -> None:
    """Advance the boot bar by one step with the given label."""
    _bar.step(msg)


def _boot_mark(name: str, *, once: bool = False, **fields: object) -> None:
    """Record a named startup timing mark in CVOPS_STATE_DIR/boot_trace.json."""
    _trace.mark(name, once=once, **fields)


def _boot_finish() -> None:  # noqa: F401 — imported by window.py
    """Seal the boot bar (idempotent — safe to call multiple times)."""
    _boot_mark("boot_bar_finished", once=True)
    _bar.finish()


# Legacy alias used by window.py's _bprint fallback.
_BOOT_T0 = _BOOT_T0  # re-export so window.py can read elapsed time


from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication


def _preinit_webengine() -> None:
    """Configure Qt WebEngine flags before QApplication construction.

    The actual QtWebEngineWidgets import is intentionally deferred until the
    Ecosystem view opens or the delayed warmup runs; importing it here adds a
    visible cold-start penalty.
    """
    _boot_mark("webengine_config_start", once=True)
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
    except Exception:
        pass
    _boot_mark("webengine_config_done", once=True)


class _BlockAvFinder(importlib.abc.MetaPathFinder):
    """Prevent PyAV import in CV Ops to avoid cv2/av dylib clashes."""

    def find_spec(self, fullname: str, path, target=None):
        if fullname == "av" or fullname.startswith("av."):
            raise ModuleNotFoundError(
                "PyAV is disabled for insight_local.cvops (cv2/av dylib conflict mitigation)"
            )
        return None


def _install_av_blocker() -> None:
    if str(os.environ.get("INSIGHT_BLOCK_PYAV", "1")).strip() in {"0", "false", "False"}:
        return
    if "av" in sys.modules:
        sys.modules.pop("av", None)
    if not any(isinstance(f, _BlockAvFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _BlockAvFinder())


def main(argv: list[str] | None = None) -> int:
    args = list(argv or [])

    if "--nice" in args:
        print("[cvops] --nice/web mode is retired; starting local workbench.", file=sys.stderr)
    args = [arg for arg in args if arg not in {"--local", "--nice"}]
    _bar.install_qt_message_handler()

    _boot_step("av blocker")
    _install_av_blocker()

    _boot_step("webengine config")
    _preinit_webengine()

    _boot_step("importing UI patches")
    import insight_local.cvops.ui.patch_parallelogram_buttons  # noqa: F401

    _boot_step("importing window module")
    _boot_mark("window_import_start", once=True)
    from insight_local.cvops.window import main as cvops_main
    _boot_mark("window_import_done", once=True)
    _boot_step("window module ready")

    return int(cvops_main(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
