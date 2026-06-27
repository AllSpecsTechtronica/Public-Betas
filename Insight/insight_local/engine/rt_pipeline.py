from __future__ import annotations

import os
import sys
import threading
import time
from typing import TYPE_CHECKING

from ..config import FOCUS_MAX_DIM, PREVIEW_MAX_DIM
from .imaging import encode_jpeg_base64, render_status_frame

if TYPE_CHECKING:
    import numpy as np

    from .effects_worker import EffectsWorker
    from .frame_source import FrameSource
    from .perception import InsightPerceptionEngine
    from .ui_adapter import SessionUiAdapter


class InsightRtPipeline:
    """Dedicated runtime threads for capture, encoding, and UI publishing."""

    def __init__(
        self,
        *,
        frame_source: FrameSource,
        perception: InsightPerceptionEngine,
        ui: SessionUiAdapter,
        source_label_getter,
    ) -> None:
        self.frame_source = frame_source
        self.perception = perception
        self.ui = ui
        self.source_label_getter = source_label_getter
        self.effects_worker: EffectsWorker | None = None
        self._stop_event = threading.Event()
        # [PERF] Event signals replace polling in encode/publish loops
        self._encode_event = threading.Event()
        self._publish_event = threading.Event()
        # [BACKPRESSURE] Only allow one video_frame payload in flight at a
        # time. Capture thread blocks until the UI dequeues the previous
        # frame, which makes playback pace to whatever the UI can handle
        # (slower but smooth) instead of dropping every Nth frame (skittery).
        self._frame_inflight = threading.BoundedSemaphore(1)
        # [DIAG] Counters for the optional diagnostic printer (env INSIGHT_PIPELINE_STATS=1).
        self._stats_emitted = 0
        self._stats_dropped_backpressure = 0
        self._stats_dropped_stale = 0
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="InsightCapture")
        self._encode_thread = threading.Thread(target=self._encode_loop, daemon=True, name="InsightSerialize")
        self._publish_thread = threading.Thread(target=self._publish_loop, daemon=True, name="InsightPublish")

    def start(self) -> None:
        self.perception.start()
        self._capture_thread.start()
        self._encode_thread.start()
        self._publish_thread.start()
        if os.environ.get("INSIGHT_PIPELINE_STATS", "0").strip() not in ("", "0", "false", "False"):
            threading.Thread(
                target=self._stats_loop, daemon=True, name="InsightPipelineStats",
            ).start()

    def _stats_loop(self) -> None:
        import resource  # noqa: WPS433 — stdlib, only imported when diagnostics are on
        while not self._stop_event.is_set():
            self._stop_event.wait(3.0)
            if self._stop_event.is_set():
                break
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # On Darwin ru_maxrss is bytes; on Linux it's kilobytes. Normalize to MB.
            rss_mb = rss_kb / (1024 * 1024) if sys.platform == "darwin" else rss_kb / 1024
            try:
                vbuf_depth = len(self.frame_source._vbuf)  # type: ignore[attr-defined]
            except Exception:
                vbuf_depth = -1
            sys.stderr.write(
                f"[pipeline] rss={rss_mb:.1f}MB emitted={self._stats_emitted} "
                f"bp_drops={self._stats_dropped_backpressure} "
                f"stale_drops={self._stats_dropped_stale} "
                f"vbuf={vbuf_depth}\n"
            )
            sys.stderr.flush()

    def stop(self) -> None:
        self._stop_event.set()
        # Unblock any threads waiting on events so they can exit cleanly
        self._encode_event.set()
        self._publish_event.set()
        self.perception.stop()
        for thread in (self._capture_thread, self._encode_thread, self._publish_thread):
            if thread.is_alive():
                thread.join(timeout=1.0)

    def _capture_loop(self) -> None:
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            target_fps = max(10, int(getattr(self.perception, "target_fps", 30) or 30))
            # For video files, cap pacing at the video's native FPS — pulling
            # faster advances the file cursor past real-time and floods the UI
            # with frames faster than the video actually contains content.
            if getattr(self.frame_source, "is_video", False):
                native = float(getattr(self.frame_source, "video_fps", 0) or 0)
                if native > 0:
                    target_fps = min(target_fps, max(1, int(round(native))))
            interval = 1.0 / target_fps
            now = time.monotonic()
            if now < next_tick:
                self._stop_event.wait(next_tick - now)
                if self._stop_event.is_set():
                    break
            # [PERF] Advance by fixed interval to prevent cumulative drift, but
            # clamp to `now` if we fell behind for longer than one interval —
            # otherwise the gate above is bypassed forever and the loop spins
            # at max speed (in video mode frame_source.read() is non-blocking,
            # so there is no natural backstop).
            next_tick = max(next_tick + interval, time.monotonic())
            frame_ok = True
            try:
                frame = self.frame_source.read()
            except Exception as exc:
                frame_ok = False
                height, width = self.frame_source.last_shape
                frame = render_status_frame(str(exc), width=width, height=height)
            is_new = bool(getattr(self.frame_source, "frame_is_new", True))

            # Bail early on stale repeats — no downstream consumer wants them
            # and copying a ~2.6MB array 25 times/sec just to drop it churns
            # the allocator.
            should_emit = (not frame_ok) or is_new
            perception_wants = frame_ok and is_new and hasattr(self.perception, "update_frame")
            effects_wants = self.effects_worker is not None and frame_ok and is_new
            if not (should_emit or perception_wants or effects_wants):
                self._stats_dropped_stale += 1
                continue

            overlays = self._get_live_overlays()

            # Block until the UI has dequeued the previous video_frame payload.
            # Timeout keeps us alive if the UI thread ever stalls (e.g. modal
            # dialog) — we drop the frame instead of hanging the capture loop.
            acquired = False
            if should_emit:
                acquired = self._frame_inflight.acquire(timeout=0.5)
                if not acquired:
                    self._stats_dropped_backpressure += 1

            # [PERF] Single copy shared by UI + perception (both read-only).
            # Effects worker gets its own copy since effect callbacks may write
            # to the frame array in-place.
            frame_snap = frame.copy()

            if effects_wants:
                self.effects_worker.submit(frame_snap.copy(), overlays)

            if acquired:
                self.ui.emit_payload(
                    {
                        "type": "video_frame",
                        "frame": frame_snap,
                        "frame_ok": frame_ok,
                        "is_new": is_new,
                        "overlays": overlays,
                        "active_focus": self._get_active_focus_track_id(),
                        "source": self.source_label_getter(),
                        "ts": round(time.time(), 3),
                    }
                )
                self._stats_emitted += 1
            if perception_wants:
                self.perception.update_frame(frame_snap)
                # Signal encode/publish loops that new work may be available
                self._encode_event.set()
                self._publish_event.set()

    def _encode_loop(self) -> None:
        # [PERF] Block on event instead of polling every 20-100ms.
        # _encode_event is set by _capture_loop after each update_frame(); 0.5s
        # timeout is a safety fallback for work queued by other code paths.
        while not self._stop_event.is_set():
            self._encode_event.wait(timeout=0.5)
            self._encode_event.clear()
            if self._stop_event.is_set():
                break
            if not hasattr(self.perception, "drain_encode_work"):
                continue
            work = self.perception.drain_encode_work()
            if not work:
                continue

            encoded: dict[int, tuple[str, str]] = {}
            for track_id, crop, is_focused in work:
                preview_b64 = encode_jpeg_base64(crop, PREVIEW_MAX_DIM)
                focus_b64 = encode_jpeg_base64(crop, FOCUS_MAX_DIM, quality=78) if is_focused else ""
                encoded[int(track_id)] = (preview_b64, focus_b64)
            if hasattr(self.perception, "apply_encoded_crops"):
                self.perception.apply_encoded_crops(encoded)

    def _publish_loop(self) -> None:
        # [PERF] Block on event instead of polling every 30-100ms.
        # Drains all queued payloads before returning to wait.
        while not self._stop_event.is_set():
            self._publish_event.wait(timeout=0.5)
            self._publish_event.clear()
            if self._stop_event.is_set():
                break
            if not hasattr(self.perception, "collect_broadcast_payloads"):
                continue
            while not self._stop_event.is_set():
                payloads = self.perception.collect_broadcast_payloads()
                if not payloads:
                    break
                self.ui.emit_many(payloads)

    def notify_frame_consumed(self) -> None:
        """UI thread calls this when it dequeues a video_frame payload, freeing
        a slot in the inflight semaphore so the capture loop can emit again."""
        try:
            self._frame_inflight.release()
        except ValueError:
            pass

    def _get_active_focus_track_id(self):
        getter = getattr(self.perception, "get_active_focus_track_id", None)
        if callable(getter):
            return getter()
        return None

    def _get_live_overlays(self) -> list[dict[str, object]]:
        getter = getattr(self.perception, "get_live_overlays", None)
        if callable(getter):
            return getter()
        return list(getattr(self.perception, "live_overlays", []) or [])
