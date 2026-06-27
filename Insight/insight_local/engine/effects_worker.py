# Effects Worker Thread
# =====================
# Dedicated thread for all heavy visual-overlay operations:
#   - Hairline edge background (Sobel + threshold + blend)
#   - Thermal edge recolor inside detection bounds
#   - Segmentation draw_overlays (contour tracking + mask composite)
#
# Architecture:
#   capture thread  -->  drops (frame, overlays) into a 1-slot mailbox
#   effects thread  -->  picks up the latest, runs all heavy ops, stores
#                        a composed uint8 frame behind a lock
#   UI thread       -->  reads the latest composed frame (zero numpy work)
#
# If the effects thread falls behind, stale frames are silently dropped
# (the mailbox only holds the most recent item).  The UI thread never
# blocks and never does heavy work.

from __future__ import annotations

import threading
from typing import Any, Optional

import cv2
import numpy as np


class EffectsWorker:
    """Runs all heavy per-frame visual effects on its own thread."""

    def __init__(self) -> None:
        # -- mailbox: single-slot (latest wins) --
        self._inbox_lock = threading.Lock()
        self._inbox: Optional[tuple[np.ndarray, list[dict[str, Any]]]] = None

        # -- output: latest composed frame --
        self._out_lock = threading.Lock()
        self._composed_frame: Optional[np.ndarray] = None

        # -- pluggable effect callbacks --
        # Each is called as  fn(frame, overlays) -> frame
        # They run sequentially on the effects thread.
        self._effects: list[tuple[str, Any]] = []
        self._effects_lock = threading.Lock()

        # -- lifecycle --
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="EffectsWorker",
        )

    # ------------------------------------------------------------------
    # Effect registration
    # ------------------------------------------------------------------

    def register_effect(self, name: str, fn: Any) -> None:
        """Register an effect callback.

        *fn* signature:  ``(frame: ndarray, overlays: list) -> ndarray``
        Effects run in registration order.
        """
        with self._effects_lock:
            self._effects.append((name, fn))

    def remove_effect(self, name: str) -> None:
        with self._effects_lock:
            self._effects = [(n, f) for n, f in self._effects if n != name]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        self._thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    # Producer API (called from capture thread)
    # ------------------------------------------------------------------

    def submit(self, frame: np.ndarray, overlays: list[dict[str, Any]]) -> None:
        """Drop a frame + overlays into the mailbox.

        Only the most recent submission is kept.  If the effects thread
        hasn't picked up the previous one yet, it is silently replaced.
        """
        with self._inbox_lock:
            self._inbox = (frame, overlays)
        self._wake.set()

    # ------------------------------------------------------------------
    # Consumer API (called from UI thread)
    # ------------------------------------------------------------------

    def get_composed_frame(self) -> Optional[np.ndarray]:
        """Return the latest composed frame, or None if nothing ready yet."""
        with self._out_lock:
            return self._composed_frame

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=0.05)
            self._wake.clear()

            # Grab the latest submission
            with self._inbox_lock:
                item = self._inbox
                self._inbox = None

            if item is None:
                continue

            frame, overlays = item

            # Run each registered effect
            with self._effects_lock:
                effects = list(self._effects)

            for _name, fn in effects:
                try:
                    frame = fn(frame, overlays)
                except Exception:
                    pass

            # Publish
            with self._out_lock:
                self._composed_frame = frame
