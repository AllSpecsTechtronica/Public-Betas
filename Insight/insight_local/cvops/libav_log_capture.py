"""Best-effort capture of libav/FFmpeg decoder messages.

QtMultimedia's FFmpeg backend (and OpenCV) emit decode diagnostics — "Packet
corrupt", "Invalid NAL unit size", "missing picture", "partial file", … — by
writing straight to file descriptor 2 via av_log's default callback. They do
NOT travel through Qt's logging, so qInstallMessageHandler can't see them and
QMediaPlayer often does not raise errorOccurred for them either.

To surface those messages in the UI we tap fd 2: redirect it through a pipe,
tee everything back to the real terminal (so normal logging is unaffected), and
keep a small ring buffer of the lines that match known corruption patterns.

Everything here is best-effort and fully guarded: if the redirection can't be
installed (e.g. fd 2 is not a normal stream), capture is silently disabled and
the application is left untouched.
"""

from __future__ import annotations

import os
import re
import threading
import time
from collections import deque

# Decoder-level corruption / decode-failure signatures worth surfacing.
_CORRUPTION_RE = re.compile(
    r"corrupt"
    r"|invalid nal"
    r"|missing picture"
    r"|partial file"
    r"|error splitting"
    r"|non-existing pps"
    r"|decode_slice_header"
    r"|concealing"
    r"|error while decoding"
    r"|no frame",
    re.IGNORECASE,
)


class _LibavLogTap:
    def __init__(self) -> None:
        self._lines: deque[tuple[float, str]] = deque(maxlen=120)
        self._lock = threading.Lock()
        self._installed = False
        self._orig_fd: int | None = None
        self._read_fd: int | None = None
        self._thread: threading.Thread | None = None

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> bool:
        if self._installed:
            return True
        try:
            # Keep a handle to the real stderr so we can tee output to it.
            self._orig_fd = os.dup(2)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, 2)
            os.close(write_fd)
            self._read_fd = read_fd
            self._thread = threading.Thread(
                target=self._reader, name="LibavLogTap", daemon=True
            )
            self._thread.start()
            self._installed = True
        except Exception:
            # Roll back as best we can and leave stderr exactly as it was.
            try:
                if self._orig_fd is not None:
                    os.dup2(self._orig_fd, 2)
                    os.close(self._orig_fd)
            except Exception:
                pass
            self._orig_fd = None
            self._read_fd = None
            self._installed = False
        return self._installed

    def _reader(self) -> None:
        orig = self._orig_fd
        read_fd = self._read_fd
        if read_fd is None or orig is None:
            return
        buf = b""
        while True:
            try:
                data = os.read(read_fd, 4096)
            except Exception:
                break
            if not data:
                break
            # Tee straight back to the real terminal so nothing is swallowed.
            try:
                os.write(orig, data)
            except Exception:
                pass
            buf += data
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                text = raw.decode("utf-8", "replace").rstrip()
                if text and _CORRUPTION_RE.search(text):
                    with self._lock:
                        self._lines.append((time.time(), text))

    def recent(self, within_s: float = 8.0, limit: int = 12) -> list[str]:
        now = time.time()
        with self._lock:
            matches = [text for (ts, text) in self._lines if now - ts <= within_s]
        return matches[-limit:]


_TAP = _LibavLogTap()


def install_libav_log_capture() -> bool:
    """Install the fd-2 tap once. Safe to call repeatedly; returns success."""
    return _TAP.install()


def recent_libav_corruption_lines(within_s: float = 8.0, limit: int = 12) -> list[str]:
    """Return decoder corruption lines seen in the last ``within_s`` seconds."""
    if not _TAP.installed:
        return []
    return _TAP.recent(within_s=within_s, limit=limit)
