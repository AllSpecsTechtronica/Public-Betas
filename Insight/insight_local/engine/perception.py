from __future__ import annotations
import math
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Optional

import cv2
import numpy as np

from ..config import (
    ATTENDANCE_CHECKOUT_SECONDS,
    DEFAULT_FPS,
    FOCUS_MAX_DIM,
    HISTORY_PUBLISH_INTERVAL,
    HISTORY_TTL_SECONDS,
    NEW_TRACK_SECONDS,
    PERSISTENT_SECONDS,
    PREVIEW_MAX_DIM,
    PREVIEW_QUALITY,
    PUBLISH_INTERVAL_SECONDS,
    RECOGNITION_CONFIRM_FRAMES,
    RECOGNITION_MAX_MISSES,
    RECOGNITION_PERSON_ONLY,
    RECOGNITION_REQUEST_INTERVAL,
    RECOGNITION_THRESHOLD,
    RECOGNITION_TOP_K,
    normalize_fps,
    RuntimeConfig,
    TRACK_STALE_FRAMES,
    TRACK_STALE_SECONDS,
    CONF_EWMA_ALPHA,
    CONF_TIER_MEDIUM,
    CONF_TIER_ALPHA_LOCK,
    CONF_RAPID_DROP_PTS,
    CONF_RAPID_DROP_FRAMES,
    CONF_GRACE_FRAMES_MULT,
    CONF_GRACE_SECS_MULT,
    normalize_detection_mode,
    normalize_image_size,
)
from .imaging import (
    bbox_iou,
    center_of_bbox,
    clamp,
    class_priority,
    crop_with_padding,
    describe_track_event,
    downsample_gray,
    encode_jpeg_base64,
)
from .services import Detector
from .types import HistoryEntry, HudState, PreviewCard, TrackState


@dataclass
class AttendanceRecord:
    identity: str
    track_id: int
    checked_in_at: float
    last_seen_at: float
    confidence: float


class InsightPerceptionEngine:
    def __init__(
        self,
        config: RuntimeConfig,
        broadcaster: Callable[[dict[str, Any]], None],
        source_label_getter: Callable[[], str],
        detector: Optional[Detector] = None,
        recognition_worker: Optional[Any] = None,
        detector_state_callback: Optional[Callable[[str, str], None]] = None,
        detector_recovery_callback: Optional[Callable[[str, str], None]] = None,
        recognition_control_callback: Optional[Callable[[bool, str], None]] = None,
    ):
        self.config = config
        self.broadcaster = broadcaster
        self.source_label_getter = source_label_getter
        self.detector = detector
        self.recognition_worker: Optional[Any] = recognition_worker
        self.detector_state_callback = detector_state_callback
        self.detector_recovery_callback = detector_recovery_callback
        self.recognition_control_callback = recognition_control_callback

        self.frame_queue: Queue[np.ndarray] = Queue(maxsize=1)
        self.last_processed_frame: Optional[np.ndarray] = None
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run_loop, daemon=True, name="InsightPerception")

        self.state_lock = threading.RLock()
        self._frame_lock = threading.Lock()
        self.track_states: dict[int, TrackState] = {}
        self.focused_track_id: Optional[int] = None
        self.event_feed: deque[dict[str, Any]] = deque(maxlen=8)
        self.status_text = "Waiting for frames"
        self.status_level = "info"
        self.status_ts = time.time()
        self.detection_times: deque[float] = deque(maxlen=30)
        self.processing_latency_ms = 0.0
        self.frames_received = 0
        self.frames_dropped = 0
        self.frames_processed = 0
        self.frame_counter = 0
        self.inference_interval = 1
        self.max_cards = max(3, min(6, config.max_cards))
        self.track_stale_seconds = TRACK_STALE_SECONDS
        self.track_stale_frames = TRACK_STALE_FRAMES
        self.new_track_seconds = NEW_TRACK_SECONDS
        self.persistent_seconds = PERSISTENT_SECONDS
        self.recognition_confirm_frames = max(1, RECOGNITION_CONFIRM_FRAMES)
        self.recognition_max_misses = max(1, RECOGNITION_MAX_MISSES)
        self.recognition_person_only = RECOGNITION_PERSON_ONLY
        self.recognition_request_interval = max(0.1, RECOGNITION_REQUEST_INTERVAL)
        self.attendance_checkout_seconds = max(0.5, ATTENDANCE_CHECKOUT_SECONDS)
        self.history_ttl_seconds = HISTORY_TTL_SECONDS
        self.preview_quality = PREVIEW_QUALITY
        self.target_fps = normalize_fps(getattr(config, "fps", DEFAULT_FPS))
        self.last_publish_ts = 0.0
        self.next_track_id = 1
        self.roi: Optional[tuple[float, float, float, float]] = None
        self.roi_shape: str = "rect"
        self.detection_history: list[dict[str, Any]] = []
        self.next_history_id = 1
        self.last_history_publish_ts = 0.0
        self.history_dirty = False
        self.last_roi_capture: Optional[dict[str, Any]] = None
        self.live_overlays: list[dict[str, Any]] = []
        self.detector_consecutive_errors = 0
        self.detector_restart_count = 0
        self.detector_latched = False
        self.detector_last_error = ""
        self.detector_last_ok_ts = 0.0
        self.recognition_pressure_disabled = False
        self.attendance_active: dict[str, AttendanceRecord] = {}
        self._base_target_fps = self.target_fps
        self._base_image_size = config.image_size
        self._base_max_det = config.max_det
        self.detection_mode = normalize_detection_mode(getattr(config, "detection_mode", "boxes"))
        self.config.detection_mode = self.detection_mode
        self.segmentation_resource = 1.0
        self.segmentation_ready = True
        self._segmentation_model_ready = False
        self.segmentation_backend = "yolo"
        self._resource_stage = 0
        self._high_latency_streak = 0
        self._last_performance_issue_ts = 0.0
        self._drop_window_start = time.monotonic()
        self._drop_window_count = 0
        self._next_inference_ts = 0.0
        self._pending_encode_work: deque[list[tuple[int, np.ndarray, bool]]] = deque(maxlen=1)
        self._pending_broadcasts: deque[dict[str, Any]] = deque()
        self._publish_dirty = True
        self._force_publish = False

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def set_status(self, message: str, level: str = "info") -> None:
        with self.state_lock:
            self.status_text = message
            self.status_level = level
            self.status_ts = time.time()
            self._publish_dirty = True
        self.broadcaster(
            {
                "type": "status",
                "message": message,
                "level": level,
                "ts": self.status_ts,
            }
        )

    def set_roi(self, x1: float, y1: float, x2: float, y2: float, shape: str = "rect") -> None:
        norm_shape = shape if shape in ("rect", "circle", "minimal") else "rect"
        new_roi = (
            clamp(min(x1, x2), 0.0, 1.0),
            clamp(min(y1, y2), 0.0, 1.0),
            clamp(max(x1, x2), 0.0, 1.0),
            clamp(max(y1, y2), 0.0, 1.0),
        )
        prev_roi: Optional[tuple[float, float, float, float]] = None
        prev_shape = "rect"
        with self.state_lock:
            prev_roi = self.roi
            prev_shape = self.roi_shape
            self.roi = new_roi
            self.roi_shape = norm_shape
        # Avoid status/toast spam while dragging — only announce activation or shape changes.
        if prev_roi is None:
            self.set_status(f"ROI active ({self.roi_shape})", "info")
        elif prev_shape != self.roi_shape:
            self.set_status(f"ROI active ({self.roi_shape})", "info")

    def clear_roi(self) -> None:
        with self.state_lock:
            self.roi = None
            self.roi_shape = "rect"
        self.set_status("ROI cleared", "info")

    @staticmethod
    def _to_numpy_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            array = value.numpy()
        else:
            array = np.asarray(value)
        if dtype is not None:
            array = array.astype(dtype, copy=False)
        return array

    def _result_arrays(self, result: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, str] | Any]:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            empty_boxes = np.empty((0, 4), dtype=np.float32)
            empty_scores = np.empty((0,), dtype=np.float32)
            empty_classes = np.empty((0,), dtype=np.int32)
            return empty_boxes, empty_scores, empty_classes, getattr(result, "names", {})

        xyxy = self._to_numpy_array(getattr(boxes, "xyxy", ()), dtype=np.float32).reshape(-1, 4)
        conf = self._to_numpy_array(getattr(boxes, "conf", ()), dtype=np.float32).reshape(-1)
        cls = self._to_numpy_array(getattr(boxes, "cls", ()), dtype=np.int32).reshape(-1)
        return xyxy, conf, cls, getattr(result, "names", {})

    def _result_masks(
        self,
        result: Any,
        *,
        frame_w: int,
        frame_h: int,
    ) -> list[list[tuple[float, float]]]:
        masks = getattr(result, "masks", None)
        if masks is None:
            return []
        polys = getattr(masks, "xy", None)
        if polys is None:
            return []
        output: list[list[tuple[float, float]]] = []
        for poly in polys:
            arr = self._to_numpy_array(poly, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] < 3:
                output.append([])
                continue
            points = [
                (
                    clamp(float(px) / max(1, frame_w), 0.0, 1.0),
                    clamp(float(py) / max(1, frame_h), 0.0, 1.0),
                )
                for px, py in arr
            ]
            output.append(points)
        return output

    def capture_roi_snapshot(self) -> None:
        crop, shape = self.get_current_roi_crop()
        if crop is None:
            return
        if crop.size == 0:
            self.set_status("ROI capture failed: empty region", "warn")
            return

        scan_results: list[dict[str, Any]] = []
        if self.detector is not None and not self.detector_latched:
            try:
                results = self.detector.predict(
                    crop,
                    image_size=self.config.image_size,
                    confidence=self.config.confidence,
                    iou=self.config.iou,
                    max_det=self.config.max_det,
                )
                ch, cw = crop.shape[:2]
                for result in results:
                    xyxy, conf, cls_ids, names = self._result_arrays(result)
                    for coords, confidence, cls_id in zip(xyxy, conf, cls_ids):
                        if any(math.isnan(v) or math.isinf(v) for v in coords) or math.isnan(confidence):
                            continue
                        label = names.get(int(cls_id), f"class_{int(cls_id)}") if isinstance(names, dict) else f"class_{int(cls_id)}"
                        bx1, by1, bx2, by2 = coords
                        obj_w = bx2 - bx1
                        obj_h = by2 - by1
                        det_crop = crop[max(0, int(by1)):min(ch, int(by2)), max(0, int(bx1)):min(cw, int(bx2))]
                        det_crop_b64 = ""
                        if det_crop.size > 0:
                            det_crop_b64 = encode_jpeg_base64(det_crop, PREVIEW_MAX_DIM, quality=72)
                        scan_results.append({
                            "label": label,
                            "confidence": round(confidence, 3),
                            "bbox_pct": [
                                round(bx1 / cw * 100, 1),
                                round(by1 / ch * 100, 1),
                                round(bx2 / cw * 100, 1),
                                round(by2 / ch * 100, 1),
                            ],
                            "area_pct": round((obj_w * obj_h) / (cw * ch) * 100, 1),
                            "crop_b64": det_crop_b64,
                        })
                scan_results.sort(key=lambda d: d["confidence"], reverse=True)
            except Exception as exc:
                self.set_status(f"ROI scan error: {exc}", "warn")

        image_b64 = encode_jpeg_base64(crop, FOCUS_MAX_DIM, quality=82)
        captured_at = round(time.time(), 3)

        focus_payload = {
            "active": True,
            "track_id": 0,
            "label": "ROI CAPTURE",
            "confidence": 1.0,
            "motion_score": 0,
            "age_seconds": 0,
            "event_tag": f"roi-{shape}",
            "image": image_b64,
            "silhouette": None,
            "scan_results": scan_results,
            "captured_at": captured_at,
        }
        with self.state_lock:
            self.last_roi_capture = {
                "image": image_b64,
                "scan_results": [dict(item) for item in scan_results],
                "shape": shape,
                "captured_at": captured_at,
            }
        det_count = len(scan_results)
        self.broadcaster({"type": "roi_capture", "focus": focus_payload})
        self.set_status(f"ROI scan complete - {det_count} object{'s' if det_count != 1 else ''} detected", "info")

    def get_current_roi_crop(self) -> tuple[Optional[np.ndarray], str]:
        with self._frame_lock:
            frame = self.last_processed_frame
        if frame is None:
            self.set_status("No frame available for ROI capture", "warn")
            return None, "rect"
        with self.state_lock:
            roi = self.roi
            shape = self.roi_shape
        if roi is None:
            self.set_status("No ROI active", "warn")
            return None, shape
        crop = self._extract_roi_crop(frame, roi, shape)
        if crop is None or crop.size == 0:
            self.set_status("ROI capture failed: empty region", "warn")
            return None, shape
        return crop, shape

    @staticmethod
    def _extract_roi_crop(
        frame: np.ndarray,
        roi: tuple[float, float, float, float],
        shape: str,
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        rx1, ry1 = int(roi[0] * w), int(roi[1] * h)
        rx2, ry2 = int(roi[2] * w), int(roi[3] * h)

        if shape == "circle":
            cx_px, cy_px = (rx1 + rx2) // 2, (ry1 + ry2) // 2
            radius = min(rx2 - rx1, ry2 - ry1) // 2
            sq_x1 = max(0, cx_px - radius)
            sq_y1 = max(0, cy_px - radius)
            sq_x2 = min(w, cx_px + radius)
            sq_y2 = min(h, cy_px + radius)
            crop = frame[sq_y1:sq_y2, sq_x1:sq_x2].copy()
        else:
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            crop = frame[ry1:ry2, rx1:rx2].copy()
        return crop if crop.size > 0 else None

    def get_last_roi_capture_context(self) -> Optional[dict[str, Any]]:
        with self.state_lock:
            if self.last_roi_capture is None:
                return None
            return {
                "image": self.last_roi_capture.get("image", ""),
                "scan_results": [dict(item) for item in self.last_roi_capture.get("scan_results", [])],
                "shape": self.last_roi_capture.get("shape", "rect"),
                "captured_at": self.last_roi_capture.get("captured_at", 0),
            }

    def _archive_track(self, state: TrackState) -> None:
        entry = asdict(
            HistoryEntry(
                entry_id=self.next_history_id,
                track_id=state.track_id,
                label=state.label,
                confidence=round(state.confidence, 3),
                event_tag=state.event_tag,
                age_seconds=round(state.age_seconds, 1),
                score=round(state.editorial_score, 3),
                image=state.latest_crop_b64,
                captured_at=round(time.time(), 2),
                recognized_identity=state.recognized_identity,
                recognition_confidence=round(state.recognition_confidence, 3),
            )
        )
        self.next_history_id += 1
        self.detection_history.append(entry)
        self.history_dirty = True

    def _prune_history(self) -> None:
        cutoff = time.time() - self.history_ttl_seconds
        before = len(self.detection_history)
        self.detection_history = [
            e for e in self.detection_history if e["captured_at"] >= cutoff
        ]
        if len(self.detection_history) != before:
            self.history_dirty = True

    def clear_history(self) -> None:
        with self.state_lock:
            self.detection_history.clear()
            self.history_dirty = True
        self.set_status("Detection history cleared", "info")
        self._publish_history()

    def delete_history_entry(self, entry_id: int) -> None:
        with self.state_lock:
            before = len(self.detection_history)
            self.detection_history = [
                e for e in self.detection_history if e["entry_id"] != entry_id
            ]
            if len(self.detection_history) != before:
                self.history_dirty = True
        self._publish_history()

    def _publish_history(self) -> None:
        with self.state_lock:
            self._prune_history()
            payload = list(self.detection_history)
            self.history_dirty = False
            self.last_history_publish_ts = time.monotonic()
        self.broadcaster({"type": "detection_history", "entries": payload})

    def reset_scene(self) -> None:
        with self.state_lock:
            self.track_states.clear()
            self.attendance_active.clear()
            self.focused_track_id = None
            self.event_feed.clear()
            self.live_overlays = []
            self._publish_dirty = True
        self.publish_state(force=True)

    def update_settings(self, settings: dict[str, Any]) -> None:
        if "detection_mode" in settings:
            self.detection_mode = normalize_detection_mode(settings.get("detection_mode"))
            self.config.detection_mode = self.detection_mode
        _int_settings: list[tuple[str, str, int, int, str | None]] = [
            ("max_cards", "max_cards", 3, 6, None),
            ("max_det", "config.max_det", 10, 300, None),
            ("image_size", "config.image_size", 640, 640, None),
            ("stale_frames", "track_stale_frames", 2, 30, None),
            ("preview_quality", "preview_quality", 30, 100, None),
            ("recog_top_k", "_recog_top_k", 1, 20, None),
        ]
        _float_settings: list[tuple[str, str, float, float]] = [
            ("confidence", "config.confidence", 0.05, 0.95),
            ("iou", "config.iou", 0.05, 0.95),
            ("stale_seconds", "track_stale_seconds", 0.4, 5.0),
            ("new_track_sec", "new_track_seconds", 0.5, 5.0),
            ("persistent_sec", "persistent_seconds", 1.0, 15.0),
            ("recog_threshold", "_recog_threshold", 0.30, 0.99),
        ]
        for key, attr, lo, hi, _ in _int_settings:
            if key in settings:
                try:
                    if key == "image_size":
                        v = normalize_image_size(settings[key])
                    else:
                        v = max(lo, min(hi, int(settings[key])))
                    if "." in attr:
                        obj_name, field = attr.split(".", 1)
                        setattr(getattr(self, obj_name), field, v)
                    else:
                        setattr(self, attr, v)
                except (TypeError, ValueError):
                    pass
        if "fps" in settings:
            try:
                fps = normalize_fps(settings["fps"])
                self.target_fps = fps
                self._base_target_fps = fps
                self.config.fps = fps
                self._next_inference_ts = 0.0
            except (TypeError, ValueError):
                pass
        for key, attr, lo, hi in _float_settings:
            if key in settings:
                try:
                    v = max(lo, min(hi, float(settings[key])))
                    if "." in attr:
                        obj_name, field = attr.split(".", 1)
                        setattr(getattr(self, obj_name), field, v)
                    else:
                        setattr(self, attr, v)
                except (TypeError, ValueError):
                    pass
        if "image_size" in settings:
            self._base_image_size = self.config.image_size
        if "max_det" in settings:
            self._base_max_det = self.config.max_det
        self.publish_state(force=True)

    def update_frame(self, frame: np.ndarray) -> None:
        self.frames_received += 1
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
                self.frames_dropped += 1
            self.frame_queue.put_nowait(frame)
        except Full:
            self.frames_dropped += 1

    @property
    def detector_available(self) -> bool:
        return self.detector is not None and self.detector.is_ready and not self.detector_latched

    def attach_detector(self, detector: Optional[Detector]) -> None:
        self.detector = detector
        self.detector_latched = False
        self.detector_consecutive_errors = 0
        self.detector_last_error = ""

    def get_runtime_snapshot(self) -> dict[str, Any]:
        with self.state_lock:
            roi_payload = None
            if self.roi is not None:
                roi_payload = {
                    "x1": self.roi[0],
                    "y1": self.roi[1],
                    "x2": self.roi[2],
                    "y2": self.roi[3],
                    "shape": self.roi_shape,
                }
            return {
                "settings": {
                    "confidence": self.config.confidence,
                    "iou": self.config.iou,
                    "detection_mode": self.detection_mode,
                    "segmentation_backend": self.segmentation_backend,
                    "image_size": self.config.image_size,
                    "max_det": self.config.max_det,
                    "max_cards": self.max_cards,
                    "stale_seconds": self.track_stale_seconds,
                    "stale_frames": self.track_stale_frames,
                    "new_track_sec": self.new_track_seconds,
                    "persistent_sec": self.persistent_seconds,
                    "preview_quality": self.preview_quality,
                    "recog_threshold": float(getattr(self, "_recog_threshold", RECOGNITION_THRESHOLD)),
                    "recog_top_k": int(getattr(self, "_recog_top_k", RECOGNITION_TOP_K)),
                    "fps": self.target_fps,
                },
                "roi": roi_payload,
                "metrics": self.get_runtime_metrics(),
            }

    def get_runtime_metrics(self) -> dict[str, Any]:
        with self.state_lock:
            return {
                "frames_received": self.frames_received,
                "frames_dropped": self.frames_dropped,
                "frames_processed": self.frames_processed,
                "queue_depth": self.frame_queue.qsize(),
                "detector_errors": self.detector_consecutive_errors,
                "detector_restarts": self.detector_restart_count,
                "detector_latched": self.detector_latched,
                "resource_stage": self._resource_stage,
                "target_fps": self.target_fps,
                "detection_mode": self.detection_mode,
                "segmentation_backend": self.segmentation_backend,
                "segmentation_resource": round(self.segmentation_resource, 3),
                "segmentation_ready": self.segmentation_ready,
                "image_size": self.config.image_size,
                "max_det": self.config.max_det,
            }

    def _compute_segmentation_resource(self, queue_depth: int = 0) -> tuple[float, bool]:
        if not self.detector_available:
            return 0.0, False
        model_penalty = 0.0
        if self.detection_mode == "segmentation" and not self._segmentation_model_ready:
            model_penalty = 0.55
        latency_penalty = clamp(self.processing_latency_ms / 260.0, 0.0, 1.0) * 0.45
        stage_penalty = clamp(float(self._resource_stage) / 3.0, 0.0, 1.0) * 0.42
        queue_penalty = 0.14 if queue_depth > 0 else 0.0
        pressure_penalty = 0.08 if self.recognition_pressure_disabled else 0.0
        score = clamp(
            1.0 - model_penalty - latency_penalty - stage_penalty - queue_penalty - pressure_penalty,
            0.0,
            1.0,
        )
        is_ready = score >= 0.45 and (self.detection_mode != "segmentation" or self._segmentation_model_ready)
        return score, is_ready

    def _emit_detector_health(self, state: str, message: str) -> None:
        if self.detector_state_callback is not None:
            self.detector_state_callback(state, message)

    def _emit_recovery_event(self, action: str, detail: str) -> None:
        if self.detector_recovery_callback is not None:
            self.detector_recovery_callback(action, detail)

    def _degrade_resource_policy(self, now: float) -> None:
        # image_size is locked to 640 for YOLO -- never degrade it.
        if self._resource_stage == 0:
            if self._base_target_fps <= 0:
                self.target_fps = DEFAULT_FPS
            else:
                self.target_fps = max(10, int(self._base_target_fps * 0.6))
        elif self._resource_stage == 1:
            self.config.max_det = max(10, int(self._base_max_det * 0.5))
        else:
            return
        self._resource_stage += 1
        self._last_performance_issue_ts = now

    def _restore_resource_policy(self) -> None:
        if self._resource_stage <= 0:
            return
        self._resource_stage -= 1
        if self._resource_stage <= 0:
            self.target_fps = self._base_target_fps
        if self._resource_stage <= 1:
            self.config.max_det = self._base_max_det

    def _handle_detector_error(self, exc: Exception) -> None:
        self.detector_consecutive_errors += 1
        self.detector_last_error = str(exc)
        self._emit_detector_health("degraded", str(exc))
        self.set_status(f"Detection error: {exc}", "error")
        if self.detector_consecutive_errors == self.config.detector_error_threshold:
            self._emit_recovery_event("detector_recovery_requested", self.detector_last_error or "Detector recovery requested")
        if self.detector_consecutive_errors >= self.config.fault_latch_threshold:
            self.detector_latched = True
            self._emit_detector_health("failed", self.detector_last_error or "Detector fault latched")
            self._emit_recovery_event("detector_latched", self.detector_last_error or "Detector fault latched")
            self.set_status("Detector latched failed. Manual mode only.", "error")

    def set_focus(self, track_id: int) -> None:
        with self.state_lock:
            if track_id not in self.track_states:
                self.status_text = f"Track {track_id} is no longer available"
                self.status_level = "warn"
                self.status_ts = time.time()
                self.publish_state(force=True)
                return
            self.focused_track_id = track_id
            state = self.track_states[track_id]
            state.event_tag = "focused"
            if not state.focus_crop_b64:
                state.focus_crop_b64 = state.latest_crop_b64
            self._append_event_locked(track_id, state.label, "focused")
        self.publish_state(force=True)

    def clear_focus(self) -> None:
        with self.state_lock:
            self.focused_track_id = None
        self.publish_state(force=True)

    def get_live_overlays(self) -> list[dict[str, Any]]:
        with self.state_lock:
            return [dict(item) for item in self.live_overlays]

    def get_active_focus_track_id(self) -> Optional[int]:
        with self.state_lock:
            return self.focused_track_id

    def drain_encode_work(self) -> Optional[list[tuple[int, np.ndarray, bool]]]:
        with self.state_lock:
            if not self._pending_encode_work:
                return None
            return self._pending_encode_work.pop()

    def apply_encoded_crops(self, encoded: dict[int, tuple[str, str]]) -> None:
        with self.state_lock:
            changed = False
            for track_id, (preview_b64, focus_b64) in encoded.items():
                state = self.track_states.get(track_id)
                if state is None:
                    continue
                state.latest_crop_b64 = preview_b64
                if focus_b64:
                    state.focus_crop_b64 = focus_b64
                changed = True
            if changed:
                self._publish_dirty = True

    def collect_broadcast_payloads(self, force: bool = False) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        with self.state_lock:
            if force:
                self._force_publish = True
            while self._pending_broadcasts:
                payloads.append(self._pending_broadcasts.popleft())
        publish_payloads = self._build_publish_payloads(force=force)
        if publish_payloads:
            payloads.extend(publish_payloads)
        return payloads

    def publish_state(self, force: bool = False) -> None:
        for payload in self.collect_broadcast_payloads(force=force):
            self.broadcaster(payload)

    def _build_publish_payloads(self, force: bool = False) -> list[dict[str, Any]]:
        with self.state_lock:
            now = time.monotonic()
            should_publish = force or self._force_publish or self._publish_dirty
            if not should_publish and now - self.last_publish_ts < PUBLISH_INTERVAL_SECONDS:
                return []
            self.last_publish_ts = now
            self._publish_dirty = False
            self._force_publish = False

            active_tracks = [
                state
                for state in self.track_states.values()
                if now - state.last_seen <= self.track_stale_seconds
            ]
            active_tracks.sort(key=lambda state: state.editorial_score, reverse=True)
            top_tracks = active_tracks[: self.max_cards]
            cards = []
            frame_h = 0
            frame_w = 0
            with self._frame_lock:
                if self.last_processed_frame is not None:
                    frame_h, frame_w = self.last_processed_frame.shape[:2]
            for rank, state in enumerate(top_tracks, start=1):
                bx1, by1, bx2, by2 = state.bbox
                bbox_norm = None
                if frame_w > 0 and frame_h > 0:
                    bbox_norm = (
                        max(0.0, min(1.0, bx1 / frame_w)),
                        max(0.0, min(1.0, by1 / frame_h)),
                        max(0.0, min(1.0, bx2 / frame_w)),
                        max(0.0, min(1.0, by2 / frame_h)),
                    )
                cards.append(
                    asdict(
                        PreviewCard(
                            track_id=state.track_id,
                            label=state.label,
                            confidence=round(state.confidence, 3),
                            motion_score=round(state.motion_score, 3),
                            age_seconds=round(state.age_seconds, 1),
                            event_tag=state.event_tag,
                            rank=rank,
                            score=round(state.editorial_score, 3),
                            image=state.latest_crop_b64,
                            recognized_identity=state.recognized_identity,
                            recognition_confidence=round(state.recognition_confidence, 3),
                            bbox_norm=bbox_norm,
                        )
                    )
                )

            focus_payload: dict[str, Any] = {"active": False}
            if self.focused_track_id is not None:
                state = self.track_states.get(self.focused_track_id)
                if state is not None and now - state.last_seen <= self.track_stale_seconds:
                    focus_payload = {
                        "active": True,
                        "track_id": state.track_id,
                        "label": state.label,
                        "confidence": round(state.confidence, 3),
                        "motion_score": round(state.motion_score, 3),
                        "age_seconds": round(state.age_seconds, 1),
                        "event_tag": state.event_tag,
                        "image": state.focus_crop_b64 or state.latest_crop_b64,
                        "silhouette": state.focus_silhouette_b64,
                    }
                else:
                    self.focused_track_id = None

            hud_state = asdict(
                HudState(
                    mode="ROI" if self.roi is not None else "PASSIVE",
                    source=self.source_label_getter(),
                    model=Path(self.config.model_path).name,
                    fps=float(self.target_fps),
                    latency_ms=round(self.processing_latency_ms, 1),
                    track_count=len(active_tracks),
                    active_focus=self.focused_track_id,
                    status=self.status_text,
                    roi_active=self.roi is not None,
                    detection_mode=self.detection_mode,
                    segmentation_backend=self.segmentation_backend,
                    segmentation_resource=round(self.segmentation_resource, 3),
                    segmentation_ready=self.segmentation_ready,
                )
            )
            status_payload = {
                "type": "status",
                "message": self.status_text,
                "level": self.status_level,
                "ts": self.status_ts,
            }
            events_payload = list(self.event_feed)

            publish_history = False
            if self.history_dirty or (
                now - self.last_history_publish_ts >= HISTORY_PUBLISH_INTERVAL
            ):
                self._prune_history()
                history_payload = list(self.detection_history)
                self.history_dirty = False
                self.last_history_publish_ts = now
                publish_history = True
            else:
                history_payload = None

        payloads = [
            {"type": "hud_state", "state": hud_state},
            {"type": "preview_tiles", "tiles": cards},
            {"type": "event_feed", "items": events_payload},
            {"type": "focus_state", "focus": focus_payload},
            status_payload,
        ]
        if publish_history:
            payloads.append({"type": "detection_history", "entries": history_payload})
        return payloads

    def _append_custom_event_locked(self, track_id: int, tag: str, text: str, label: str = "attendance") -> None:
        self.event_feed.appendleft(
            {
                "track_id": track_id,
                "tag": tag,
                "label": label,
                "text": text,
                "ts": round(time.time(), 2),
            }
        )

    def _mark_attendance_locked(
        self,
        identity: str,
        track_id: int,
        confidence: float,
        now: float,
    ) -> Optional[dict[str, Any]]:
        record = self.attendance_active.get(identity)
        if record is None:
            self.attendance_active[identity] = AttendanceRecord(
                identity=identity,
                track_id=track_id,
                checked_in_at=now,
                last_seen_at=now,
                confidence=confidence,
            )
            self._append_custom_event_locked(track_id, "checkin", f"{identity} checked in")
            return {
                "type": "attendance_event",
                "event": "check_in",
                "identity": identity,
                "track_id": track_id,
                "confidence": round(confidence, 4),
                "ts": round(time.time(), 3),
            }

        record.track_id = track_id
        record.last_seen_at = now
        record.confidence = max(record.confidence, confidence)
        return None

    def _expire_attendance_locked(self, now: float) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for identity, record in list(self.attendance_active.items()):
            if record.track_id and record.track_id in self.track_states:
                continue
            if now - record.last_seen_at < self.attendance_checkout_seconds:
                continue
            self._append_custom_event_locked(record.track_id, "checkout", f"{identity} checked out")
            payloads.append(
                {
                    "type": "attendance_event",
                    "event": "check_out",
                    "identity": identity,
                    "track_id": record.track_id,
                    "confidence": round(record.confidence, 4),
                    "ts": round(time.time(), 3),
                }
            )
            del self.attendance_active[identity]
        return payloads

    def apply_recognition_result(self, result: dict[str, Any]) -> None:
        """
        Called from LocalInsightSession when a recognition_result payload arrives.
        Promotes a candidate identity only after repeated consistent recognitions.
        """
        track_id = int(result.get("track_id", 0))
        identity = str(result.get("identity", "unknown"))
        confidence = float(result.get("confidence", 0.0))
        threshold_met = bool(result.get("threshold_met", False))
        source = str(result.get("source", "auto"))
        attendance_payload: Optional[dict[str, Any]] = None
        now = time.monotonic()

        with self.state_lock:
            state = self.track_states.get(track_id)
            if state is None:
                return
            state.recognition_last_update = now

            if not threshold_met or identity == "unknown":
                state.recognition_miss_streak += 1
                if state.recognition_miss_streak >= self.recognition_max_misses:
                    state.recognition_candidate = ""
                    state.recognition_candidate_streak = 0
                    if not state.attendance_identity:
                        state.recognized_identity = ""
                        state.recognition_confidence = 0.0
                return

            if state.recognized_identity and state.recognized_identity != identity:
                return

            state.recognition_miss_streak = 0
            if state.recognition_candidate == identity:
                state.recognition_candidate_streak += 1
            else:
                state.recognition_candidate = identity
                state.recognition_candidate_streak = 1

            confirm_frames = 1 if source == "manual" else self.recognition_confirm_frames
            if state.recognized_identity == identity:
                state.recognition_confidence = max(state.recognition_confidence * 0.7, confidence)
                if state.attendance_identity == identity:
                    attendance_payload = self._mark_attendance_locked(identity, track_id, confidence, state.last_seen)
                return

            if state.recognition_candidate_streak < confirm_frames:
                state.recognition_confidence = max(state.recognition_confidence, confidence * 0.5)
                return

            state.recognized_identity = identity
            state.recognition_confidence = confidence
            state.attendance_identity = identity
            self._append_custom_event_locked(track_id, "identified", f"{identity} confirmed on track {track_id}")
            attendance_payload = self._mark_attendance_locked(identity, track_id, confidence, state.last_seen)

        if attendance_payload is not None:
            self.broadcaster(attendance_payload)

    def _run_loop(self) -> None:
        consecutive_errors = 0
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue
            self.frame_counter += 1
            now = time.monotonic()
            interval = 0.0 if self.target_fps <= 0 else (1.0 / self.target_fps)
            if interval > 0.0 and now < self._next_inference_ts:
                continue
            self._next_inference_ts = now + interval if interval > 0.0 else 0.0
            try:
                self._process_frame(frame)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                self.set_status(f"Perception worker error: {exc}", "error")
                if consecutive_errors >= 10:
                    self.set_status("Perception paused: too many consecutive errors", "error")
                    self.stop_event.wait(timeout=5.0)
                    consecutive_errors = 0

    def _process_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self.last_processed_frame = frame
        start = time.monotonic()
        if not self.detector_available:
            self.live_overlays = []
            with self.state_lock:
                self._segmentation_model_ready = False
                self.segmentation_resource, self.segmentation_ready = 0.0, False
                self._publish_dirty = True
            return
        using_segmentation = self.detection_mode == "segmentation"
        try:
            assert self.detector is not None
            results = self.detector.predict(
                frame,
                image_size=self.config.image_size,
                confidence=self.config.confidence,
                iou=self.config.iou,
                max_det=self.config.max_det,
            )
            if using_segmentation:
                self._segmentation_model_ready = True
        except Exception as exc:
            self._handle_detector_error(exc)
            return
        self.detector_consecutive_errors = 0
        self.detector_last_error = ""
        self.detector_last_ok_ts = time.time()
        self._emit_detector_health("healthy", "Detector online")

        now = time.monotonic()
        detections: list[dict[str, Any]] = []
        frame_h, frame_w = frame.shape[:2]
        for result in results:
            xyxy, conf, cls_ids, names = self._result_arrays(result)
            mask_polygons = (
                self._result_masks(result, frame_w=frame_w, frame_h=frame_h)
                if using_segmentation
                else []
            )
            for idx, (coords, confidence, cls_id) in enumerate(zip(xyxy, conf, cls_ids)):
                if any(math.isnan(v) or math.isinf(v) for v in coords) or math.isnan(confidence):
                    continue
                label = names.get(int(cls_id), f"class_{int(cls_id)}") if isinstance(names, dict) else f"class_{int(cls_id)}"
                x1, y1, x2, y2 = [int(v) for v in coords]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w, x2)
                y2 = min(frame_h, y2)
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                        "mask_norm": [],
                    }
                )

        roi = self.roi
        if roi is not None:
            h, w = frame.shape[:2]
            rx1 = int(roi[0] * w)
            ry1 = int(roi[1] * h)
            rx2 = int(roi[2] * w)
            ry2 = int(roi[3] * h)
            roi_cx = (rx1 + rx2) / 2
            roi_cy = (ry1 + ry2) / 2
            roi_rx = (rx2 - rx1) / 2
            roi_ry = (ry2 - ry1) / 2
            is_circle = self.roi_shape == "circle"
            filtered = []
            for det in detections:
                bx1, by1, bx2, by2 = det["bbox"]
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                if is_circle:
                    r = min(roi_rx, roi_ry)
                    dist_sq = (cx - roi_cx) ** 2 + (cy - roi_cy) ** 2
                    if dist_sq <= r * r:
                        filtered.append(det)
                else:
                    if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                        filtered.append(det)
            detections = filtered

        active_ids: set[int] = set()
        encode_work: list[tuple[int, np.ndarray, bool]] = []
        attendance_payloads: list[dict[str, Any]] = []

        with self.state_lock:
            focused_id = self.focused_track_id
            available_tracks = {
                track_id: state
                for track_id, state in self.track_states.items()
                if now - state.last_seen <= self.track_stale_seconds
            }
            matched_track_ids: set[int] = set()

            for detection in detections:
                label = detection["label"]
                confidence = detection["confidence"]
                bbox = detection["bbox"]
                bbox_center = center_of_bbox(bbox)

                best_track_id: Optional[int] = None
                best_match_score = 0.0
                for track_id, state in available_tracks.items():
                    if track_id in matched_track_ids:
                        continue
                    if state.label != label:
                        continue
                    iou = bbox_iou(bbox, state.bbox)
                    distance = math.dist(bbox_center, center_of_bbox(state.bbox))
                    diag = max(20.0, math.hypot(state.bbox[2] - state.bbox[0], state.bbox[3] - state.bbox[1]))
                    proximity = clamp(1.0 - (distance / (diag * 3.5)), 0.0, 1.0)
                    match_score = (iou * 0.72) + (proximity * 0.28)
                    if match_score > best_match_score:
                        best_match_score = match_score
                        best_track_id = track_id

                if best_track_id is None or best_match_score < 0.18:
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    state = TrackState(
                        track_id=track_id,
                        label=label,
                        confidence=confidence,
                        bbox=bbox,
                        first_seen=now,
                        last_seen=now,
                        mask_norm=list(detection.get("mask_norm", []) or []),
                    )
                    self.track_states[track_id] = state
                    available_tracks[track_id] = state
                else:
                    track_id = best_track_id
                    state = self.track_states[track_id]

                matched_track_ids.add(track_id)
                active_ids.add(track_id)
                crop_pad_ratio = 0.52 if label.lower() == "face" else 0.14
                crop, _, _ = crop_with_padding(frame, bbox, pad_ratio=crop_pad_ratio)
                if crop.size == 0:
                    continue

                previous_gray = state.prev_gray_crop
                gray_small = downsample_gray(crop)
                state.center_history.append(bbox_center)

                motion_geo = 0.0
                if len(state.center_history) >= 2:
                    last_center = state.center_history[-2]
                    delta = math.dist(last_center, bbox_center)
                    diag = max(20.0, math.hypot(bbox[2] - bbox[0], bbox[3] - bbox[1]))
                    motion_geo = clamp(delta / (diag * 0.18), 0.0, 1.0)

                motion_roi = 0.0
                if previous_gray is not None and previous_gray.shape == gray_small.shape:
                    diff = cv2.absdiff(previous_gray, gray_small)
                    motion_roi = clamp(float(np.mean(diff)) / 30.0, 0.0, 1.0)

                state.prev_gray_crop = gray_small
                state.label = label
                state.confidence = confidence
                self._update_conf_state(state, confidence, now)
                state.bbox = bbox
                state.mask_norm = list(detection.get("mask_norm", []) or [])
                state.last_seen = now
                state.age_seconds = now - state.first_seen
                state.roi_energy = motion_roi
                state.motion_score = clamp((motion_geo * 0.58) + (motion_roi * 0.42), 0.0, 1.0)
                state.persistence_score = clamp(state.age_seconds / self.persistent_seconds, 0.0, 1.0)
                state.novelty_score = clamp(1.0 - (state.age_seconds / 4.0), 0.0, 1.0)
                state.missing_frames = 0

                encode_work.append((track_id, crop, focused_id == track_id))

                should_request_recognition = (
                    self.recognition_worker is not None
                    and (not self.recognition_person_only or label.lower() in {"person", "face"})
                    and (state.recognized_identity in {"", "unknown"} or state.attendance_identity == "")
                    and (now - state.last_recognition_request_ts >= self.recognition_request_interval)
                )
                if should_request_recognition:
                    state.last_recognition_request_ts = now
                    self.recognition_worker.enqueue(
                        entry_id=track_id,
                        crop=crop,
                        label=label,
                        source="auto",
                        track_id=track_id,
                    )

                state.event_tag = self._derive_event_tag(track_id, state.age_seconds, state.motion_score)
                raw_score = self._editorial_score(track_id, state)
                if state.editorial_score <= 0.0:
                    state.editorial_score = raw_score
                else:
                    state.editorial_score = (state.editorial_score * 0.65) + (raw_score * 0.35)

                if state.event_tag != state.last_announced_tag:
                    self._append_event_locked(track_id, state.label, state.event_tag)
                    state.last_announced_tag = state.event_tag

            stale_ids = []
            for track_id, state in self.track_states.items():
                if track_id not in active_ids:
                    state.missing_frames += 1
                    # [ANTI-FLICKER] Extend grace period for medium/alpha tier tracks
                    # with steady confidence. Rapid drops revert to standard threshold.
                    if self._is_rapid_conf_drop(state) or state.conf_tier == "lowest":
                        eff_frames  = self.track_stale_frames
                        eff_seconds = self.track_stale_seconds
                    else:
                        eff_frames  = int(self.track_stale_frames * CONF_GRACE_FRAMES_MULT)
                        eff_seconds = self.track_stale_seconds * CONF_GRACE_SECS_MULT
                    if state.missing_frames > eff_frames or now - state.last_seen > eff_seconds:
                        stale_ids.append(track_id)

            for track_id in stale_ids:
                departed = self.track_states.pop(track_id, None)
                if departed is not None and departed.latest_crop_b64:
                    self._archive_track(departed)
                if departed is not None and departed.attendance_identity:
                    record = self.attendance_active.get(departed.attendance_identity)
                    if record is not None and record.track_id == track_id:
                        record.track_id = 0
                        record.last_seen_at = departed.last_seen
                if self.focused_track_id == track_id:
                    self.focused_track_id = None

            attendance_payloads = self._expire_attendance_locked(now)
            self._prune_history()

        if encode_work:
            self._pending_encode_work.append(encode_work)

        with self.state_lock:
            dt = time.monotonic() - start
            self.detection_times.append(dt)
            total = sum(self.detection_times)
            self.processing_latency_ms = (total / len(self.detection_times)) * 1000.0 if self.detection_times else 0.0
            self.frames_processed += 1
            self.inference_interval = 1

            if dt > 0.18:
                self._high_latency_streak += 1
                self._last_performance_issue_ts = now
            else:
                self._high_latency_streak = 0
                if self._resource_stage > 0 and self._last_performance_issue_ts > 0 and now - self._last_performance_issue_ts >= 30.0:
                    self._restore_resource_policy()
                    self._last_performance_issue_ts = now

            if self._high_latency_streak >= 10:
                self._degrade_resource_policy(now)
                self._high_latency_streak = 0

            if now - self._drop_window_start >= 10.0:
                dropped = self.frames_dropped - self._drop_window_count
                self._drop_window_count = self.frames_dropped
                self._drop_window_start = now
                if dropped > 20 and self.recognition_worker is not None and not self.recognition_pressure_disabled:
                    self.recognition_pressure_disabled = True
                    self.recognition_worker = None
                    if self.recognition_control_callback is not None:
                        self.recognition_control_callback(False, "Recognition disabled under sustained queue pressure")

            self.status_text = "CV active" if active_ids else "Insight scanning"
            self.status_level = "info"
            self.status_ts = time.time()
            self.segmentation_resource, self.segmentation_ready = self._compute_segmentation_resource(
                self.frame_queue.qsize()
            )

            # Build live overlay data for active mode bboxes
            # [ANTI-FLICKER] Include all non-stale tracks (not just active_ids) so
            # grace-period tracks remain visible at reduced opacity instead of vanishing.
            _tier_active_scale  = {"alpha": 1.0,  "medium": 0.82, "lowest": 0.58}
            _tier_grace_base    = {"alpha": 0.72, "medium": 0.55, "lowest": 0.38}
            overlays = []
            for tid, st in self.track_states.items():
                bx1, by1, bx2, by2 = st.bbox
                if tid in active_ids:
                    tier_scale = _tier_active_scale.get(st.conf_tier, 1.0)
                else:
                    eff_frames = (
                        int(self.track_stale_frames * CONF_GRACE_FRAMES_MULT)
                        if st.conf_tier != "lowest" and not self._is_rapid_conf_drop(st)
                        else self.track_stale_frames
                    )
                    fade = max(0.0, 1.0 - (st.missing_frames / max(1, eff_frames)))
                    tier_scale = round(fade * _tier_grace_base.get(st.conf_tier, 0.55), 3)
                    if tier_scale < 0.05:
                        continue
                overlays.append({
                    "track_id": tid,
                    "label": st.label,
                    "confidence": round(st.smoothed_conf, 3),
                    "identity": st.recognized_identity or "",
                    "bbox_norm": (bx1 / frame_w, by1 / frame_h, bx2 / frame_w, by2 / frame_h),
                    "mask_norm": list(st.mask_norm),
                    "alpha_scale": tier_scale,
                })
            self.live_overlays = overlays
            self._publish_dirty = True
            for payload in attendance_payloads:
                self._pending_broadcasts.append(payload)

    @staticmethod
    def _update_conf_state(state: TrackState, raw_conf: float, now: float) -> None:
        """Update EWMA smoothed confidence and tier classification."""
        state.conf_history.append((now, raw_conf))
        if state.smoothed_conf == 0.0:
            state.smoothed_conf = raw_conf          # bootstrap on first detection
        else:
            state.smoothed_conf = (
                CONF_EWMA_ALPHA * raw_conf
                + (1.0 - CONF_EWMA_ALPHA) * state.smoothed_conf
            )
        if state.smoothed_conf >= CONF_TIER_ALPHA_LOCK:
            state.conf_tier = "alpha"
        elif state.smoothed_conf >= CONF_TIER_MEDIUM:
            state.conf_tier = "medium"
        else:
            state.conf_tier = "lowest"

    @staticmethod
    def _is_rapid_conf_drop(state: TrackState) -> bool:
        """Return True if confidence dropped >= CONF_RAPID_DROP_PTS within last CONF_RAPID_DROP_FRAMES frames."""
        history = list(state.conf_history)
        if len(history) < 2:
            return False
        window = history[-CONF_RAPID_DROP_FRAMES:]
        max_conf = max(c for _, c in window)
        min_conf = min(c for _, c in window)
        return (max_conf - min_conf) >= CONF_RAPID_DROP_PTS

    def _derive_event_tag(self, track_id: int, age_seconds: float, motion_score: float) -> str:
        if self.focused_track_id == track_id:
            return "focused"
        if age_seconds <= self.new_track_seconds:
            return "new"
        if motion_score >= 0.18:
            return "moving"
        if age_seconds >= self.persistent_seconds:
            return "persistent"
        return "moving" if motion_score >= 0.10 else "new"

    def _editorial_score(self, track_id: int, state: TrackState) -> float:
        focus_boost = 2.8 if self.focused_track_id == track_id else 0.0
        tag_boost = {
            "focused": 1.8,
            "new": 1.15,
            "moving": 1.0,
            "persistent": 0.75,
        }.get(state.event_tag, 0.5)
        return (
            focus_boost
            + tag_boost
            + (state.motion_score * 1.65)
            + (state.novelty_score * 1.20)
            + (state.persistence_score * 0.90)
            + (state.confidence * 1.15)
            + class_priority(state.label)
        )

    def _append_event_locked(self, track_id: int, label: str, tag: str) -> None:
        if tag not in {"new", "moving", "persistent", "focused"}:
            return
        self.event_feed.appendleft(
            {
                "track_id": track_id,
                "tag": tag,
                "label": label,
                "text": describe_track_event(label, tag),
                "ts": round(time.time(), 2),
            }
        )
