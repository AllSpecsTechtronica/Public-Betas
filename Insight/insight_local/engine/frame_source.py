from __future__ import annotations
import threading
import time
from collections import deque
from typing import Callable, Optional

import cv2
import numpy as np

from ..config import DEFAULT_FPS, RuntimeConfig, normalize_fps

_VIDEO_BUFFER_SIZE = 8  # pre-decoded frames to keep ahead


class FrameSource:
    def __init__(self, config: RuntimeConfig, status_callback: Callable[[str, str], None]):
        self.config = config
        self.status_callback = status_callback
        self.requested_fps = normalize_fps(getattr(config, "fps", DEFAULT_FPS))
        self.lock = threading.RLock()
        self.capture: Optional[cv2.VideoCapture] = None
        self.prepared_capture: Optional[cv2.VideoCapture] = None
        self.prepared_source: Optional[str] = None
        self.prepared_frame: Optional[np.ndarray] = None
        self._prepare_generation = 0
        self.current_source = config.source
        self.last_shape = (720, 1280)
        self.pending_frame: Optional[np.ndarray] = None
        self.last_ok_ts = 0.0
        self.last_error = ""
        self.failure_count = 0
        self.frame_is_new = True
        self._video_speed = 1.0
        self._video_speed_accum = 0.0
        self._paused = False
        # Video read-ahead buffer
        self._vbuf: deque[np.ndarray] = deque(maxlen=_VIDEO_BUFFER_SIZE)
        self._vbuf_lock = threading.Lock()
        self._vbuf_stop = threading.Event()
        self._vbuf_thread: Optional[threading.Thread] = None
        self._vbuf_last: Optional[np.ndarray] = None  # fallback when buffer empty

    def _source_label(self, source: str) -> str:
        if source == "camera":
            return f"camera {self.config.camera_index}"
        return f"video {self.config.video_path.name}"

    def _active_status_text(self, source: str) -> str:
        if source == "camera":
            return f"Camera {self.config.camera_index} active"
        return f"Video demo active: {self.config.video_path.name}"

    def _open_capture_for_source(self, source: str) -> cv2.VideoCapture:
        if source == "camera":
            capture = cv2.VideoCapture(self.config.camera_index)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if self.requested_fps > 0:
                capture.set(cv2.CAP_PROP_FPS, self.requested_fps)
            if not capture.isOpened():
                capture.release()
                raise RuntimeError(f"Unable to open camera index {self.config.camera_index}")
            return capture

        if not self.config.video_path.exists():
            raise FileNotFoundError(f"Sample video not found: {self.config.video_path}")
        capture = cv2.VideoCapture(str(self.config.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Unable to open video file: {self.config.video_path}")
        return capture

    def _frame_has_signal(self, frame: np.ndarray) -> bool:
        if frame is None or frame.size == 0:
            return False
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        return mean_val > 1.5 or std_val > 1.5

    def _read_from_capture(self, capture: cv2.VideoCapture, source: str) -> np.ndarray:
        ok, frame = capture.read()
        if not ok:
            if source == "video":
                capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Unable to read from {source}")
        if frame is None or frame.size == 0:
            raise RuntimeError(f"Empty frame from {source}")
        return frame

    def _warm_capture(self, capture: cv2.VideoCapture, source: str) -> np.ndarray:
        deadline = time.monotonic() + (1.6 if source == "camera" else 0.8)
        target_reads = 6 if source == "camera" else 2
        successful_reads = 0
        fallback_frame: Optional[np.ndarray] = None

        while time.monotonic() < deadline:
            try:
                frame = self._read_from_capture(capture, source)
            except RuntimeError:
                time.sleep(0.03)
                continue

            fallback_frame = frame
            successful_reads += 1
            if successful_reads >= target_reads and self._frame_has_signal(frame):
                return frame
            time.sleep(0.03 if source == "camera" else 0.01)

        if fallback_frame is not None:
            return fallback_frame
        raise RuntimeError(f"{self._source_label(source)} did not produce a ready frame")

    def _stop_video_reader(self) -> None:
        """Signal the reader thread to stop and flush the buffer. Safe to call under self.lock."""
        self._vbuf_stop.set()
        # Don't join here — the reader may be waiting on self.lock.
        # It will exit on its own once it sees the stop flag.
        old = self._vbuf_thread
        self._vbuf_thread = None
        with self._vbuf_lock:
            self._vbuf.clear()
            self._vbuf_last = None
        # Join outside lock if possible (best-effort, thread is daemon anyway)
        if old is not None and old.is_alive():
            try:
                old.join(timeout=0.1)
            except RuntimeError:
                pass

    def _start_video_reader(self) -> None:
        if self.current_source == "video":
            if self._paused or abs(self._video_speed - 1.0) > 0.001:
                return
        self._stop_video_reader()
        self._vbuf_stop.clear()
        self._vbuf_thread = threading.Thread(
            target=self._video_reader_loop, daemon=True, name="VideoReadAhead"
        )
        self._vbuf_thread.start()

    def _video_reader_loop(self) -> None:
        """Background thread: decode video frames paced at the video's native
        FPS so playback runs at real-time speed. Pulling faster than native
        just plays the video fast and flooods the consumer with frames it
        can never catch up on."""
        native_fps = 0.0
        if self.capture is not None:
            try:
                native_fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
            except Exception:
                native_fps = 0.0
        pace_fps = native_fps if native_fps > 0 else float(self.requested_fps or 0)
        if self.requested_fps > 0 and pace_fps > 0:
            pace_fps = min(pace_fps, float(self.requested_fps))
        next_ts = time.monotonic()
        while not self._vbuf_stop.is_set():
            now = time.monotonic()
            if now < next_ts:
                self._vbuf_stop.wait(next_ts - now)
                if self._vbuf_stop.is_set():
                    break
            interval = 0.0 if pace_fps <= 0 else (1.0 / pace_fps)
            next_ts = time.monotonic() + interval
            with self._vbuf_lock:
                if len(self._vbuf) >= _VIDEO_BUFFER_SIZE:
                    continue
            with self.lock:
                if self.capture is None or self.current_source != "video":
                    break
                if self._paused or abs(self._video_speed - 1.0) > 0.001:
                    continue
                try:
                    frame = self._read_from_capture(self.capture, "video")
                except Exception:
                    continue
            with self._vbuf_lock:
                self._vbuf.append(frame)

    def _set_active_capture_locked(self, capture: cv2.VideoCapture, source: str, frame: Optional[np.ndarray]) -> None:
        self._stop_video_reader()
        previous_capture = self.capture
        self.capture = capture
        self.current_source = source
        self.pending_frame = frame.copy() if frame is not None else None
        if frame is not None and frame.size != 0:
            self.last_shape = frame.shape[:2]
            self.last_ok_ts = time.time()
            self.last_error = ""
            self.failure_count = 0
        if previous_capture is not None and previous_capture is not capture:
            previous_capture.release()
        if source == "video" and not self._paused and abs(self._video_speed - 1.0) <= 0.001:
            self._start_video_reader()

    def _clear_prepared_locked(self) -> None:
        if self.prepared_capture is not None:
            self.prepared_capture.release()
        self.prepared_capture = None
        self.prepared_source = None
        self.prepared_frame = None

    def _open_capture_locked(self) -> None:
        capture = self._open_capture_for_source(self.current_source)
        ready_frame = self._warm_capture(capture, self.current_source)
        self._set_active_capture_locked(capture, self.current_source, ready_frame)
        self.status_callback(self._active_status_text(self.current_source), "info")

    def switch_source(self, requested: Optional[str] = None) -> str:
        with self.lock:
            target = self.prepare_switch(requested)
            return self.commit_prepared_switch(target)

    def prepare_switch(self, requested: Optional[str] = None) -> str:
        with self.lock:
            target = requested or ("video" if self.current_source == "camera" else "camera")
            if target == self.current_source and self.capture is not None:
                return self.current_source
            self._prepare_generation += 1
            generation = self._prepare_generation
            stale_prepared = self.prepared_capture
            self.prepared_capture = None
            self.prepared_source = None
            self.prepared_frame = None
        if stale_prepared is not None:
            stale_prepared.release()

        new_capture: Optional[cv2.VideoCapture] = None
        try:
            new_capture = self._open_capture_for_source(target)
            ready_frame = self._warm_capture(new_capture, target)
        except Exception:
            if new_capture is not None:
                new_capture.release()
            raise

        prior_prepared: Optional[cv2.VideoCapture] = None
        with self.lock:
            if self._prepare_generation != generation:
                new_capture.release()
                return target
            if target == self.current_source and self.capture is not None:
                new_capture.release()
                return self.current_source
            prior_prepared = self.prepared_capture
            self.prepared_capture = new_capture
            self.prepared_source = target
            self.prepared_frame = ready_frame.copy()
        if prior_prepared is not None and prior_prepared is not new_capture:
            prior_prepared.release()
        return target

    def commit_prepared_switch(self, expected_source: Optional[str] = None) -> str:
        with self.lock:
            if self.prepared_capture is None or self.prepared_source is None:
                raise RuntimeError("No prepared source is ready to switch")
            if expected_source is not None and self.prepared_source != expected_source:
                raise RuntimeError("Prepared source no longer matches the requested target")

            target = self.prepared_source
            capture = self.prepared_capture
            frame = self.prepared_frame
            self.prepared_capture = None
            self.prepared_source = None
            self.prepared_frame = None
            self._prepare_generation += 1
            assert capture is not None
            self._set_active_capture_locked(capture, target, frame)
            self.status_callback(self._active_status_text(self.current_source), "info")
            return self.current_source

    def cancel_prepared_switch(self) -> None:
        with self.lock:
            self._prepare_generation += 1
            self._clear_prepared_locked()

    @property
    def is_video(self) -> bool:
        return self.current_source == "video"

    @property
    def video_fps(self) -> float:
        with self.lock:
            if self.capture is not None and self.current_source == "video":
                fps = self.capture.get(cv2.CAP_PROP_FPS)
                return fps if fps > 0 else 30.0
            return 30.0

    @property
    def video_frame_count(self) -> int:
        with self.lock:
            if self.capture is not None and self.current_source == "video":
                return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            return 0

    @property
    def video_position_frame(self) -> int:
        with self.lock:
            if self.capture is not None and self.current_source == "video":
                return int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
            return 0

    @property
    def video_duration_sec(self) -> float:
        count = self.video_frame_count
        fps = self.video_fps
        return count / fps if fps > 0 else 0.0

    @property
    def video_position_sec(self) -> float:
        pos = self.video_position_frame
        fps = self.video_fps
        return pos / fps if fps > 0 else 0.0

    def seek_frame(self, frame_index: int) -> None:
        with self.lock:
            if self.capture is not None and self.current_source == "video":
                frame_index = max(0, min(frame_index, self.video_frame_count - 1))
                # Flush read-ahead buffer so stale frames aren't shown after seek
                with self._vbuf_lock:
                    self._vbuf.clear()
                    self._vbuf_last = None
                self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                self._video_speed_accum = 0.0
                self.pending_frame = None

    def seek_fraction(self, fraction: float) -> None:
        """Seek to a fraction (0.0–1.0) of the video duration."""
        count = self.video_frame_count
        if count > 0:
            self.seek_frame(int(fraction * count))

    @property
    def video_paused(self) -> bool:
        return self._paused

    @video_paused.setter
    def video_paused(self, value: bool) -> None:
        self._paused = bool(value)
        with self.lock:
            if self.current_source != "video":
                return
            if self._paused or abs(self._video_speed - 1.0) > 0.001:
                self._stop_video_reader()
            else:
                self._start_video_reader()

    @property
    def video_speed(self) -> float:
        return float(self._video_speed)

    def set_video_speed(self, speed: float) -> None:
        try:
            parsed = float(speed)
        except (TypeError, ValueError):
            parsed = 1.0
        clamped = max(-5.0, min(5.0, parsed))
        if abs(clamped) < 0.01:
            clamped = 0.0
        self._video_speed = clamped
        self._video_speed_accum = 0.0
        with self.lock:
            if self.current_source != "video":
                return
            if self._paused or abs(self._video_speed - 1.0) > 0.001:
                self._stop_video_reader()
            else:
                self._start_video_reader()

    def reopen_current_source(self) -> None:
        self._stop_video_reader()
        with self.lock:
            if self.capture is not None:
                self.capture.release()
                self.capture = None
            self.pending_frame = None
            self._open_capture_locked()

    def update_settings(self, settings: dict[str, object]) -> None:
        if "fps" not in settings:
            return
        fps = normalize_fps(settings["fps"])
        self.requested_fps = fps
        self.config.fps = fps
        with self.lock:
            if self.capture is not None and self.current_source == "camera" and fps > 0:
                self.capture.set(cv2.CAP_PROP_FPS, fps)
            if self.prepared_capture is not None and self.prepared_source == "camera" and fps > 0:
                self.prepared_capture.set(cv2.CAP_PROP_FPS, fps)

    def _record_failure(self, message: str) -> None:
        self.last_error = message
        self.failure_count += 1
        self.status_callback(message, "error")

    def read(self) -> np.ndarray:
        with self.lock:
            try:
                if self.capture is None:
                    self._open_capture_locked()

                if self.pending_frame is not None:
                    frame = self.pending_frame
                    self.pending_frame = None
                elif self.current_source == "video":
                    if not self._paused and abs(self._video_speed - 1.0) <= 0.001:
                        # Pull from read-ahead buffer (non-blocking) at normal speed.
                        frame = self._pop_video_frame()
                    else:
                        frame = self._read_video_speed_locked()
                else:
                    assert self.capture is not None
                    frame = self._read_from_capture(self.capture, self.current_source)
            except Exception as exc:
                self._record_failure(str(exc))
                raise
            self.last_shape = frame.shape[:2]
            self.last_ok_ts = time.time()
            self.last_error = ""
            self.failure_count = 0
            return frame

    def _read_video_speed_locked(self) -> np.ndarray:
        assert self.capture is not None
        frame_count = max(1, self.video_frame_count)
        pos = int(self.capture.get(cv2.CAP_PROP_POS_FRAMES))
        current_idx = pos - 1 if pos > 0 else 0
        current_idx = max(0, min(frame_count - 1, current_idx))
        speed = self._video_speed
        if self._paused or abs(speed) < 0.01:
            target_idx = current_idx
            self.frame_is_new = False
        else:
            self._video_speed_accum += abs(speed)
            whole_steps = int(self._video_speed_accum)
            if whole_steps > 0:
                self._video_speed_accum -= whole_steps
            step = whole_steps if whole_steps > 0 else 0
            if step == 0:
                target_idx = current_idx
                self.frame_is_new = False
            else:
                direction = 1 if speed > 0 else -1
                target_idx = (current_idx + (direction * step)) % frame_count
                self.frame_is_new = True
        with self._vbuf_lock:
            self._vbuf.clear()
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, int(target_idx))
        frame = self._read_from_capture(self.capture, "video")
        with self._vbuf_lock:
            self._vbuf_last = frame
        return frame

    def _pop_video_frame(self) -> np.ndarray:
        """Pop a pre-decoded frame from the read-ahead buffer, or fall back to direct read."""
        with self._vbuf_lock:
            if self._vbuf:
                frame = self._vbuf.popleft()
                self._vbuf_last = frame
                self.frame_is_new = True
                return frame
            if self._vbuf_last is not None:
                self.frame_is_new = False
                return self._vbuf_last
        # Buffer empty and no fallback — direct read (startup or stall)
        self.frame_is_new = True
        assert self.capture is not None
        return self._read_from_capture(self.capture, self.current_source)

    def describe_source(self) -> str:
        if self.current_source == "camera":
            return f"camera:{self.config.camera_index}"
        return f"video:{self.config.video_path.name}"

    def cleanup(self) -> None:
        self._stop_video_reader()
        with self.lock:
            self._clear_prepared_locked()
            if self.capture is not None:
                self.capture.release()
                self.capture = None
