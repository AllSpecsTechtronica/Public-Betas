#!/usr/bin/env python3
from __future__ import annotations
import argparse
import asyncio
import base64
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import asynccontextmanager
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from queue import Empty, Full, Queue
from typing import Any, Callable, Optional

import cv2
import numpy as np
import uvicorn
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
from aiortc import rtcpeerconnection as _rtcpc, sdp as _rtcsdp
from aiortc.mediastreams import VideoFrame
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from ultralytics import YOLO

from insight_local.runtime_profile import pick_torch_device

# aiortc 1.14.0 bug: and_direction() crashes with ValueError when
# _offerDirection is None (happens with Chrome-generated SDP for data
# channel transceivers).  Patch to treat None as pass-through.
_orig_and_direction = _rtcpc.and_direction


def _safe_and_direction(a: str, b: str) -> str:
    if a is None:
        return b if b is not None else "inactive"
    if b is None:
        return a
    return _orig_and_direction(a, b)


_rtcpc.and_direction = _safe_and_direction


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

INSIGHT_DIR = Path(__file__).resolve().parent
LOCAL_ASSETS_DIR = INSIGHT_DIR / "insight_local" / "Insight_assets"
LEGACY_ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR = LOCAL_ASSETS_DIR if LOCAL_ASSETS_DIR.exists() else LEGACY_ASSETS_DIR
MODELS_DIR = ASSETS_DIR / "models"
VIDEOS_DIR = ASSETS_DIR / "videos"

DEFAULT_MODEL_PATH = MODELS_DIR / "yolo26n.pt"
DEFAULT_VIDEO_PATH = VIDEOS_DIR / "frenchpeoplewalkinglong.mp4"
MODEL_CHOICES = {
    "yolo26n": MODELS_DIR / "yolo26n.pt",
    "yolo26s": MODELS_DIR / "yolo26s.pt",
    "yolo26m": MODELS_DIR / "yolo26m.pt",
}

DEFAULT_CONFIDENCE = float(os.environ.get("INSIGHT_CONFIDENCE", "0.25"))
DEFAULT_IOU = float(os.environ.get("INSIGHT_IOU", "0.20"))
DEFAULT_IMG_SIZE = int(os.environ.get("INSIGHT_IMG_SIZE", "256"))
DEFAULT_MAX_DET = int(os.environ.get("INSIGHT_MAX_DET", "100"))
DEFAULT_FPS = int(os.environ.get("INSIGHT_FPS", "30"))
TRACK_STALE_SECONDS = float(os.environ.get("INSIGHT_STALE_SEC", "1.4"))
TRACK_STALE_FRAMES = int(os.environ.get("INSIGHT_STALE_FRAMES", "8"))
NEW_TRACK_SECONDS = float(os.environ.get("INSIGHT_NEW_TRACK_SEC", "1.5"))
PERSISTENT_SECONDS = float(os.environ.get("INSIGHT_PERSISTENT_SEC", "4.0"))
PUBLISH_INTERVAL_SECONDS = float(os.environ.get("INSIGHT_PUBLISH_INTERVAL", "0.25"))
HISTORY_TTL_SECONDS = int(os.environ.get("INSIGHT_HISTORY_TTL", "40"))
HISTORY_PUBLISH_INTERVAL = float(os.environ.get("INSIGHT_HISTORY_PUB_INTERVAL", "1.0"))
MAX_HISTORY_ENTRIES = int(os.environ.get("INSIGHT_HISTORY_MAX", "60"))
PREVIEW_MAX_DIM = int(os.environ.get("INSIGHT_PREVIEW_DIM", "240"))
FOCUS_MAX_DIM = int(os.environ.get("INSIGHT_FOCUS_DIM", "420"))
PREVIEW_QUALITY = int(os.environ.get("INSIGHT_PREVIEW_QUALITY", "72"))
AI_HTTP_TIMEOUT_SECONDS = float(os.environ.get("INSIGHT_AI_TIMEOUT_SEC", "35"))

INSIGHT_OLLAMA_URL = os.environ.get("INSIGHT_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
INSIGHT_OLLAMA_MODEL = os.environ.get("INSIGHT_OLLAMA_MODEL", "llava:latest")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
INSIGHT_OPENAI_MODEL = os.environ.get("INSIGHT_OPENAI_MODEL", "gpt-4.1-mini")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
INSIGHT_ANTHROPIC_MODEL = os.environ.get("INSIGHT_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

from mlops.pipeline import infer as mlops_infer
from mlops.pipeline import registry as mlops_registry
from mlops.pipeline.integration import append_integration_event

CLASS_PRIORITY = {
    "person": 1.00,
    "car": 0.92,
    "truck": 0.90,
    "bus": 0.88,
    "motorcycle": 0.86,
    "bicycle": 0.82,
    "cell phone": 0.76,
    "knife": 0.75,
    "backpack": 0.70,
    "dog": 0.68,
    "cat": 0.68,
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def pick_device() -> str:
    return pick_torch_device()


def debug_print(enabled: bool, *parts: Any) -> None:
    if enabled:
        print(*parts, flush=True)


def render_status_frame(message: str, width: int = 1280, height: int = 720) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    gradient = np.linspace(22, 3, height, dtype=np.uint8)
    frame[:, :, 0] = gradient[:, None]
    frame[:, :, 1] = (gradient[:, None] * 2.5).astype(np.uint8)
    frame[:, :, 2] = (gradient[:, None] * 4.5).astype(np.uint8)
    cv2.putText(
        frame,
        "INSIGHT",
        (50, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.8,
        (255, 240, 190),
        3,
        cv2.LINE_AA,
    )
    wrapped = []
    words = message.split()
    line = []
    max_chars = 48
    for word in words:
        candidate = " ".join(line + [word])
        if len(candidate) > max_chars and line:
            wrapped.append(" ".join(line))
            line = [word]
        else:
            line.append(word)
    if line:
        wrapped.append(" ".join(line))
    y = 180
    for text in wrapped[:6]:
        cv2.putText(
            frame,
            text,
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (190, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 48
    return frame


def resize_for_encoding(image: np.ndarray, max_dim: int) -> np.ndarray:
    if image is None or image.size == 0:
        return image
    height, width = image.shape[:2]
    largest_dim = max(height, width)
    if largest_dim <= max_dim:
        return image
    scale = max_dim / float(largest_dim)
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def encode_jpeg_base64(image: np.ndarray, max_dim: int, quality: int = PREVIEW_QUALITY) -> str:
    image = resize_for_encoding(image, max_dim)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        return ""
    return base64.b64encode(encoded).decode("ascii")


def decode_jpeg_base64(image_b64: str) -> Optional[np.ndarray]:
    if not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64)
    except Exception:
        return None
    arr = np.frombuffer(raw, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_png_base64(image: np.ndarray, max_dim: int) -> str:
    image = resize_for_encoding(image, max_dim)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return ""
    return base64.b64encode(encoded).decode("ascii")


def crop_with_padding(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    pad_ratio: float = 0.12,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]]:
    x1, y1, x2, y2 = bbox
    height, width = frame.shape[:2]
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(width, x2 + pad_x)
    crop_y2 = min(height, y2 + pad_y)
    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()

    inner_bbox = (x1 - crop_x1, y1 - crop_y1, x2 - crop_x1, y2 - crop_y1)
    return crop, (crop_x1, crop_y1, crop_x2, crop_y2), inner_bbox


def build_focus_silhouette(crop: np.ndarray, inner_bbox: tuple[int, int, int, int]) -> str:
    if crop is None or crop.size == 0:
        return ""

    mask = np.zeros(crop.shape[:2], dtype=np.uint8)
    ix1, iy1, ix2, iy2 = inner_bbox
    ix1, iy1 = max(0, ix1), max(0, iy1)
    ix2, iy2 = min(crop.shape[1], ix2), min(crop.shape[0], iy2)
    if ix2 - ix1 < 4 or iy2 - iy1 < 4:
        return ""

    roi = crop[iy1:iy2, ix1:ix2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 45, 135)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 40]
    if not contours:
        return ""

    roi_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(roi_mask, [largest], -1, 255, thickness=-1)
    roi_mask = cv2.GaussianBlur(roi_mask, (9, 9), 0)
    mask[iy1:iy2, ix1:ix2] = roi_mask

    overlay = np.zeros((crop.shape[0], crop.shape[1], 4), dtype=np.uint8)
    overlay[:, :, 0] = 255
    overlay[:, :, 1] = 235
    overlay[:, :, 2] = 70
    overlay[:, :, 3] = (mask.astype(np.float32) * 0.72).astype(np.uint8)
    return encode_png_base64(overlay, FOCUS_MAX_DIM)


def downsample_gray(image: np.ndarray, side: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA)




def center_of_bbox(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def class_priority(label: str) -> float:
    return CLASS_PRIORITY.get(label, 0.45)


def bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def describe_track_event(label: str, tag: str) -> str:
    nice_label = label.replace("_", " ")
    if tag == "new":
        return f"{nice_label} entered scene"
    if tag == "moving":
        return f"{nice_label} moving"
    if tag == "persistent":
        return f"{nice_label} persisted in view"
    if tag == "focused":
        return f"Focused {nice_label}"
    return f"{nice_label} updated"


@dataclass
class RuntimeConfig:
    source: str = "camera"
    camera_index: int = 0
    video_path: Path = DEFAULT_VIDEO_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    host: str = "0.0.0.0"
    port: int = 8000
    max_cards: int = 4
    debug: bool = False
    confidence: float = DEFAULT_CONFIDENCE
    iou: float = DEFAULT_IOU
    image_size: int = DEFAULT_IMG_SIZE
    max_det: int = DEFAULT_MAX_DET


@dataclass
class PreviewCard:
    track_id: int
    label: str
    confidence: float
    motion_score: float
    age_seconds: float
    event_tag: str
    rank: int
    score: float
    image: str


@dataclass
class HistoryEntry:
    entry_id: int
    track_id: int
    label: str
    confidence: float
    event_tag: str
    age_seconds: float
    score: float
    image: str  # base64 JPEG thumbnail
    captured_at: float  # time.time() wallclock


@dataclass
class HudState:
    mode: str
    source: str
    model: str
    fps: float
    latency_ms: float
    track_count: int
    active_focus: Optional[int]
    status: str
    roi_active: bool = False


@dataclass
class TrackState:
    track_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    age_seconds: float = 0.0
    motion_score: float = 0.0
    persistence_score: float = 0.0
    novelty_score: float = 1.0
    roi_energy: float = 0.0
    event_tag: str = "new"
    latest_crop_b64: str = ""
    focus_crop_b64: str = ""
    focus_silhouette_b64: str = ""
    editorial_score: float = 0.0
    missing_frames: int = 0
    last_announced_tag: str = ""
    center_history: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=6), repr=False)
    prev_gray_crop: Optional[np.ndarray] = field(default=None, repr=False)


class FrameSource:
    def __init__(self, config: RuntimeConfig, status_callback: Callable[[str, str], None]):
        self.config = config
        self.status_callback = status_callback
        self.lock = threading.RLock()
        self.capture: Optional[cv2.VideoCapture] = None
        self.prepared_capture: Optional[cv2.VideoCapture] = None
        self.prepared_source: Optional[str] = None
        self.prepared_frame: Optional[np.ndarray] = None
        self.current_source = config.source
        self.last_shape = (720, 1280)
        self.pending_frame: Optional[np.ndarray] = None

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
            capture.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)
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

    def _set_active_capture_locked(self, capture: cv2.VideoCapture, source: str, frame: Optional[np.ndarray]) -> None:
        previous_capture = self.capture
        self.capture = capture
        self.current_source = source
        self.pending_frame = frame.copy() if frame is not None else None
        if frame is not None and frame.size != 0:
            self.last_shape = frame.shape[:2]
        if previous_capture is not None and previous_capture is not capture:
            previous_capture.release()

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

    def read(self) -> np.ndarray:
        with self.lock:
            if self.capture is None:
                self._open_capture_locked()

            if self.pending_frame is not None:
                frame = self.pending_frame
                self.pending_frame = None
            else:
                assert self.capture is not None
                frame = self._read_from_capture(self.capture, self.current_source)
            self.last_shape = frame.shape[:2]
            return frame

    def switch_source(self, requested: Optional[str] = None) -> str:
        with self.lock:
            target = self.prepare_switch(requested)
            return self.commit_prepared_switch(target)

    def prepare_switch(self, requested: Optional[str] = None) -> str:
        with self.lock:
            target = requested or ("video" if self.current_source == "camera" else "camera")
            if target == self.current_source and self.capture is not None:
                return self.current_source

            self._clear_prepared_locked()
            new_capture: Optional[cv2.VideoCapture] = None
            try:
                new_capture = self._open_capture_for_source(target)
                ready_frame = self._warm_capture(new_capture, target)
                self.prepared_capture = new_capture
                self.prepared_source = target
                self.prepared_frame = ready_frame.copy()
            except Exception:
                if new_capture is not None:
                    new_capture.release()
                raise
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
            assert capture is not None
            self._set_active_capture_locked(capture, target, frame)
            self.status_callback(self._active_status_text(self.current_source), "info")
            return self.current_source

    def cancel_prepared_switch(self) -> None:
        with self.lock:
            self._clear_prepared_locked()

    def describe_source(self) -> str:
        if self.current_source == "camera":
            return f"camera:{self.config.camera_index}"
        return f"video:{self.config.video_path.name}"

    def cleanup(self) -> None:
        with self.lock:
            self._clear_prepared_locked()
            if self.capture is not None:
                self.capture.release()
                self.capture = None


class InsightPerceptionEngine:
    def __init__(
        self,
        config: RuntimeConfig,
        broadcaster: Callable[[dict[str, Any]], None],
        source_label_getter: Callable[[], str],
    ):
        self.config = config
        self.broadcaster = broadcaster
        self.source_label_getter = source_label_getter
        self.device = pick_device()
        self.model = YOLO(str(config.model_path))
        self.model.to(self.device)

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
        self.detection_fps = 0.0
        self.processing_latency_ms = 0.0
        self.frames_received = 0
        self.frames_dropped = 0
        self.frames_processed = 0
        self.frame_counter = 0
        self.inference_interval = 1
        self.max_cards = max(3, min(6, config.max_cards))
        self.last_publish_ts = 0.0
        self.next_track_id = 1
        # ROI filter: normalised (0-1) coords (x1, y1, x2, y2) or None
        self.roi: Optional[tuple[float, float, float, float]] = None
        self.roi_shape: str = "rect"  # "rect" or "circle"
        # Detection history — snapshots of tracks when they go stale
        self.detection_history: list[dict[str, Any]] = []
        self.next_history_id = 1
        self.last_history_publish_ts = 0.0
        self.history_dirty = False
        self._history_snapshot_ids: set[int] = set()
        self._history_force_full = True
        self.last_roi_capture: Optional[dict[str, Any]] = None
        self._encode_queue: Queue[list[tuple[int, np.ndarray, bool]]] = Queue(maxsize=2)
        self._encode_thread = threading.Thread(target=self._encode_loop, daemon=True, name="InsightEncode")

    def start(self) -> None:
        self.thread.start()
        self._encode_thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self._encode_thread.is_alive():
            self._encode_thread.join(timeout=2.0)

    def set_status(self, message: str, level: str = "info") -> None:
        with self.state_lock:
            self.status_text = message
            self.status_level = level
            self.status_ts = time.time()
        self.broadcaster(
            {
                "type": "status",
                "message": message,
                "level": level,
                "ts": self.status_ts,
            }
        )

    def set_roi(self, x1: float, y1: float, x2: float, y2: float, shape: str = "rect") -> None:
        with self.state_lock:
            self.roi = (
                clamp(min(x1, x2), 0.0, 1.0),
                clamp(min(y1, y2), 0.0, 1.0),
                clamp(max(x1, x2), 0.0, 1.0),
                clamp(max(y1, y2), 0.0, 1.0),
            )
            self.roi_shape = shape if shape in ("rect", "circle") else "rect"
        self.set_status(f"ROI active ({self.roi_shape})", "info")

    def clear_roi(self) -> None:
        with self.state_lock:
            self.roi = None
            self.roi_shape = "rect"
        self.set_status("ROI cleared", "info")

    def capture_roi_snapshot(self) -> None:
        """Capture the current frame region inside the ROI, run detection, and push results to focus."""
        with self._frame_lock:
            frame = self.last_processed_frame
        if frame is None:
            self.set_status("No frame available for ROI capture", "warn")
            return
        with self.state_lock:
            roi = self.roi
            shape = self.roi_shape
        if roi is None:
            self.set_status("No ROI active", "warn")
            return

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

        if crop.size == 0:
            self.set_status("ROI capture failed: empty region", "warn")
            return

        # --- Run detection on the ROI crop ---
        scan_results: list[dict[str, Any]] = []
        try:
            results = self.model.predict(
                source=crop,
                device=self.device,
                imgsz=self.config.image_size,
                conf=self.config.confidence,
                iou=self.config.iou,
                max_det=self.config.max_det,
                verbose=False,
                stream=False,
            )
            ch, cw = crop.shape[:2]
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box, conf_score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                    coords = box.tolist()
                    confidence = float(conf_score)
                    if any(math.isnan(v) or math.isinf(v) for v in coords) or math.isnan(confidence):
                        continue
                    label = self.model.names.get(int(cls), f"class_{int(cls)}")
                    bx1, by1, bx2, by2 = coords
                    obj_w = bx2 - bx1
                    obj_h = by2 - by1
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
        """Snapshot a departing track into the detection history (caller holds lock)."""
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
            )
        )
        self.next_history_id += 1
        self.detection_history.append(entry)
        if len(self.detection_history) > MAX_HISTORY_ENTRIES:
            del self.detection_history[: len(self.detection_history) - MAX_HISTORY_ENTRIES]
        self.history_dirty = True

    def _prune_history(self) -> None:
        """Remove history entries older than HISTORY_TTL_SECONDS (caller holds lock)."""
        cutoff = time.time() - HISTORY_TTL_SECONDS
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
            self._history_force_full = True
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

    def _build_history_message_locked(self) -> dict[str, Any]:
        """Caller holds state_lock. Returns full or delta payload and updates snapshot."""
        current_ids = {e["entry_id"] for e in self.detection_history}
        if self._history_force_full:
            payload = {
                "type": "detection_history",
                "mode": "full",
                "entries": list(self.detection_history),
            }
            self._history_snapshot_ids = current_ids
            self._history_force_full = False
            return payload
        added_ids = current_ids - self._history_snapshot_ids
        removed_ids = self._history_snapshot_ids - current_ids
        added = [e for e in self.detection_history if e["entry_id"] in added_ids]
        self._history_snapshot_ids = current_ids
        return {
            "type": "detection_history",
            "mode": "delta",
            "added": added,
            "removed_ids": sorted(removed_ids),
        }

    def _publish_history(self) -> None:
        with self.state_lock:
            self._prune_history()
            payload = self._build_history_message_locked()
            self.history_dirty = False
            self.last_history_publish_ts = time.monotonic()
        self.broadcaster(payload)

    def reset_scene(self) -> None:
        with self.state_lock:
            self.track_states.clear()
            self.focused_track_id = None
            self.event_feed.clear()
        self.publish_state(force=True)

    def update_settings(self, settings: dict[str, Any]) -> None:
        if "max_cards" in settings:
            try:
                self.max_cards = max(3, min(6, int(settings["max_cards"])))
            except (TypeError, ValueError):
                self.set_status("Invalid max_cards setting ignored", "warn")
        self.publish_state(force=True)

    def update_frame(self, frame: np.ndarray) -> None:
        frame_copy = frame.copy()
        self.frames_received += 1
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
                self.frames_dropped += 1
            self.frame_queue.put_nowait(frame_copy)
        except Full:
            self.frames_dropped += 1

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

    def publish_state(self, force: bool = False) -> None:
        with self.state_lock:
            now = time.monotonic()
            if not force and now - self.last_publish_ts < PUBLISH_INTERVAL_SECONDS:
                return
            self.last_publish_ts = now
            if force:
                self._history_force_full = True

            active_tracks = [
                state
                for state in self.track_states.values()
                if now - state.last_seen <= TRACK_STALE_SECONDS
            ]
            active_tracks.sort(key=lambda state: state.editorial_score, reverse=True)
            top_tracks = active_tracks[: self.max_cards]
            cards = []
            for rank, state in enumerate(top_tracks, start=1):
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
                        )
                    )
                )

            focus_payload = {"active": False}
            if self.focused_track_id is not None:
                state = self.track_states.get(self.focused_track_id)
                if state is not None and now - state.last_seen <= TRACK_STALE_SECONDS:
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
                    fps=round(self.detection_fps, 1),
                    latency_ms=round(self.processing_latency_ms, 1),
                    track_count=len(active_tracks),
                    active_focus=self.focused_track_id,
                    status=self.status_text,
                    roi_active=self.roi is not None,
                )
            )
            status_payload = {
                "type": "status",
                "message": self.status_text,
                "level": self.status_level,
                "ts": self.status_ts,
            }
            events_payload = list(self.event_feed)

            # Publish history only when it changed or periodically
            history_message: Optional[dict[str, Any]] = None
            if self.history_dirty or self._history_force_full or (
                now - self.last_history_publish_ts >= HISTORY_PUBLISH_INTERVAL
            ):
                self._prune_history()
                history_message = self._build_history_message_locked()
                self.history_dirty = False
                self.last_history_publish_ts = now

        bundle = {
            "type": "state_update",
            "hud": hud_state,
            "tiles": cards,
            "events": events_payload,
            "focus": focus_payload,
            "status": status_payload,
        }
        if history_message is not None:
            bundle["history"] = history_message
        self.broadcaster(bundle)

    def _run_loop(self) -> None:
        consecutive_errors = 0
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except Empty:
                continue
            self.frame_counter += 1
            if self.frame_counter % self.inference_interval != 0:
                continue
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

    def _encode_loop(self) -> None:
        """Background thread: encode detection crops to base64 JPEG."""
        while not self.stop_event.is_set():
            try:
                work = self._encode_queue.get(timeout=0.1)
            except Empty:
                continue
            encoded: dict[int, tuple[str, str]] = {}
            for tid, crop, is_focused in work:
                preview_b64 = encode_jpeg_base64(crop, PREVIEW_MAX_DIM)
                focus_b64 = encode_jpeg_base64(crop, FOCUS_MAX_DIM, quality=78) if is_focused else ""
                encoded[tid] = (preview_b64, focus_b64)
            with self.state_lock:
                for tid, (preview_b64, focus_b64) in encoded.items():
                    state = self.track_states.get(tid)
                    if state is None:
                        continue
                    state.latest_crop_b64 = preview_b64
                    if focus_b64:
                        state.focus_crop_b64 = focus_b64

    def _process_frame(self, frame: np.ndarray) -> None:
        with self._frame_lock:
            self.last_processed_frame = frame
        start = time.monotonic()
        try:
            results = self.model.predict(
                source=frame,
                device=self.device,
                imgsz=self.config.image_size,
                conf=self.config.confidence,
                iou=self.config.iou,
                max_det=self.config.max_det,
                verbose=False,
                stream=False,
            )
        except Exception as exc:
            self.set_status(f"Detection error: {exc}", "error")
            return

        now = time.monotonic()
        detections: list[dict[str, Any]] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box, conf_score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
                coords = box.tolist()
                confidence = float(conf_score)
                if any(math.isnan(v) or math.isinf(v) for v in coords) or math.isnan(confidence):
                    continue
                label = result.names[int(cls)]
                x1, y1, x2, y2 = [int(v) for v in coords]
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                if x2 - x1 < 6 or y2 - y1 < 6:
                    continue
                detections.append(
                    {
                        "label": label,
                        "confidence": confidence,
                        "bbox": (x1, y1, x2, y2),
                    }
                )

        # Filter detections by ROI if active
        roi = self.roi
        if roi is not None:
            h, w = frame.shape[:2]
            rx1 = int(roi[0] * w)
            ry1 = int(roi[1] * h)
            rx2 = int(roi[2] * w)
            ry2 = int(roi[3] * h)
            roi_cx = (rx1 + rx2) / 2
            roi_cy = (ry1 + ry2) / 2
            roi_rx = (rx2 - rx1) / 2  # horizontal radius
            roi_ry = (ry2 - ry1) / 2  # vertical radius
            is_circle = self.roi_shape == "circle"
            filtered = []
            for det in detections:
                bx1, by1, bx2, by2 = det["bbox"]
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                if is_circle:
                    # Use smaller radius for true circle
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

        # Phase 1: Under lock — match detections, update numeric fields, collect crops
        with self.state_lock:
            focused_id = self.focused_track_id
            available_tracks = {
                track_id: state
                for track_id, state in self.track_states.items()
                if now - state.last_seen <= TRACK_STALE_SECONDS
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
                    )
                    self.track_states[track_id] = state
                    available_tracks[track_id] = state
                else:
                    track_id = best_track_id
                    state = self.track_states[track_id]

                matched_track_ids.add(track_id)
                active_ids.add(track_id)
                crop, _, _ = crop_with_padding(frame, bbox, pad_ratio=0.14)
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
                state.bbox = bbox
                state.last_seen = now
                state.age_seconds = now - state.first_seen
                state.roi_energy = motion_roi
                state.motion_score = clamp((motion_geo * 0.58) + (motion_roi * 0.42), 0.0, 1.0)
                state.persistence_score = clamp(state.age_seconds / PERSISTENT_SECONDS, 0.0, 1.0)
                state.novelty_score = clamp(1.0 - (state.age_seconds / 4.0), 0.0, 1.0)
                state.missing_frames = 0

                # Collect crop for encoding outside lock
                encode_work.append((track_id, crop.copy(), focused_id == track_id))

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
                    if state.missing_frames > TRACK_STALE_FRAMES or now - state.last_seen > TRACK_STALE_SECONDS:
                        stale_ids.append(track_id)

            for track_id in stale_ids:
                departed = self.track_states.pop(track_id, None)
                if departed is not None and departed.latest_crop_b64:
                    self._archive_track(departed)
                if self.focused_track_id == track_id:
                    self.focused_track_id = None

            self._prune_history()

        # Offload image encoding to background thread
        if encode_work:
            if self._encode_queue.full():
                try:
                    self._encode_queue.get_nowait()
                except Empty:
                    pass
            self._encode_queue.put_nowait(encode_work)

        with self.state_lock:
            dt = time.monotonic() - start
            self.detection_times.append(dt)
            total = sum(self.detection_times)
            self.detection_fps = (len(self.detection_times) / total) if total > 0 else 0.0
            self.processing_latency_ms = (total / len(self.detection_times)) * 1000.0 if self.detection_times else 0.0
            self.frames_processed += 1

            if dt > 0.18:
                self.inference_interval = min(3, self.inference_interval + 1)
            elif dt < 0.10:
                self.inference_interval = max(1, self.inference_interval - 1)

            self.status_text = "CV active" if active_ids else "Insight scanning"
            self.status_level = "info"
            self.status_ts = time.time()

        self.publish_state(force=False)

    def _derive_event_tag(self, track_id: int, age_seconds: float, motion_score: float) -> str:
        if self.focused_track_id == track_id:
            return "focused"
        if age_seconds <= NEW_TRACK_SECONDS:
            return "new"
        if motion_score >= 0.18:
            return "moving"
        if age_seconds >= PERSISTENT_SECONDS:
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


class InsightSession:
    def __init__(self, config: RuntimeConfig, loop: Optional[asyncio.AbstractEventLoop] = None):
        self.config = config
        self.channels: set[Any] = set()
        self.channels_lock = threading.RLock()
        self._scenario_lock = threading.RLock()
        self._scenario_run_seq = 0
        self.closed = False
        self._loop: Optional[asyncio.AbstractEventLoop] = loop

        self.frame_source = FrameSource(config, self._source_status_update)
        self.perception = InsightPerceptionEngine(
            config=config,
            broadcaster=self.broadcast,
            source_label_getter=self.frame_source.describe_source,
        )
        self.video_track = InsightVideoTrack(self)
        self.perception.start()

    def _next_run_id(self) -> str:
        with self._scenario_lock:
            self._scenario_run_seq += 1
            seq = self._scenario_run_seq
        return f"run-{int(time.time() * 1000)}-{seq}"

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def _source_status_update(self, message: str, level: str) -> None:
        self.perception.set_status(message, level)

    def register_channel(self, channel: Any) -> None:
        with self.channels_lock:
            self.channels.add(channel)

    def unregister_channel(self, channel: Any) -> None:
        with self.channels_lock:
            self.channels.discard(channel)

    @staticmethod
    def _channel_ready_state(channel: Any) -> str:
        try:
            return str(getattr(channel, "readyState", "open")).lower()
        except Exception:
            return "open"

    def broadcast(self, payload: dict[str, Any]) -> None:
        try:
            serialized = json.dumps(payload)
        except TypeError:
            return
        # aiortc RTCDataChannel.send() must be called from the asyncio
        # event-loop thread.  The perception engine calls broadcast from a
        # background thread, so we detect that case and dispatch via
        # call_soon_threadsafe to avoid silent send failures that would
        # permanently remove the channel from the set.
        loop = self._loop
        if loop is not None:
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop is not loop:
                loop.call_soon_threadsafe(self._send_to_channels, serialized)
                return
        self._send_to_channels(serialized)

    def _send_to_channels(self, serialized: str) -> None:
        with self.channels_lock:
            stale = []
            for channel in list(self.channels):
                state = self._channel_ready_state(channel)
                if state in {"connecting", "connecting()"}:
                    continue
                if state in {"closed", "closing"}:
                    stale.append(channel)
                    continue
                try:
                    channel.send(serialized)
                except Exception:
                    stale.append(channel)
            for channel in stale:
                self.channels.discard(channel)

    @staticmethod
    def _post_json(url: str, payload: dict[str, Any], headers: Optional[dict[str, str]] = None) -> dict[str, Any]:
        request_headers = {"Content-Type": "application/json"}
        if headers:
            request_headers.update(headers)
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=AI_HTTP_TIMEOUT_SECONDS) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            raise RuntimeError(f"HTTP {exc.code} from AI provider: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"AI provider connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("AI provider request timed out") from exc
        try:
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError as exc:
            raise RuntimeError("AI provider returned non-JSON response") from exc

    @staticmethod
    def _extract_openai_text(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"output_text", "text"}:
                    text = str(block.get("text", "")).strip()
                    if text:
                        parts.append(text)
            return "\n".join(parts).strip()
        return ""

    @staticmethod
    def _extract_anthropic_text(content: Any) -> str:
        if not isinstance(content, list):
            return ""
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = str(block.get("text", "")).strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()

    @staticmethod
    def _build_roi_ai_prompt(scan_results: list[dict[str, Any]], user_prompt: str) -> str:
        clean_prompt = (user_prompt or "").strip()
        if scan_results:
            summary_items = []
            for item in scan_results[:10]:
                label = str(item.get("label", "object"))
                confidence = int(round(float(item.get("confidence", 0.0)) * 100))
                area_pct = float(item.get("area_pct", 0.0))
                summary_items.append(f"- {label}: {confidence}% confidence, {area_pct:.1f}% area")
            scan_summary = "Detected objects in ROI:\n" + "\n".join(summary_items)
        else:
            scan_summary = "No objects were detected by the ROI scan."
        if not clean_prompt:
            clean_prompt = (
                "Describe what is happening in this ROI image, call out notable risks or anomalies, "
                "and suggest the most useful next check."
            )
        return f"{clean_prompt}\n\n{scan_summary}\n\nKeep the response concise and actionable."

    @staticmethod
    def _normalize_ai_provider(provider: Any) -> str:
        value = str(provider or "auto").strip().lower()
        aliases = {
            "claude": "anthropic",
            "gpt": "openai",
        }
        return aliases.get(value, value)

    def _resolve_ai_provider(self, requested: str) -> str:
        provider = self._normalize_ai_provider(requested)
        if provider not in {"auto", "ollama", "openai", "anthropic"}:
            raise RuntimeError(f"Unsupported AI provider: {provider}")
        if provider == "auto":
            if OPENAI_API_KEY:
                return "openai"
            if ANTHROPIC_API_KEY:
                return "anthropic"
            return "ollama"
        if provider == "openai" and not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY is missing")
        if provider == "anthropic" and not ANTHROPIC_API_KEY:
            raise RuntimeError("ANTHROPIC_API_KEY is missing")
        return provider

    def _ask_ollama(self, image_b64: str, prompt: str) -> str:
        payload = {
            "model": INSIGHT_OLLAMA_MODEL,
            "prompt": prompt,
            "images": [image_b64],
            "stream": False,
        }
        data = self._post_json(INSIGHT_OLLAMA_URL, payload)
        text = str(data.get("response", "")).strip()
        if not text:
            raise RuntimeError("Ollama returned an empty response")
        return text

    def _ask_openai(self, image_b64: str, prompt: str) -> str:
        payload = {
            "model": INSIGHT_OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0.2,
        }
        data = self._post_json(
            "https://api.openai.com/v1/chat/completions",
            payload,
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
        )
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("OpenAI returned no choices")
        message = choices[0].get("message", {})
        text = self._extract_openai_text(message.get("content"))
        if not text:
            raise RuntimeError("OpenAI returned an empty response")
        return text

    def _ask_anthropic(self, image_b64: str, prompt: str) -> str:
        payload = {
            "model": INSIGHT_ANTHROPIC_MODEL,
            "max_tokens": 600,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                    ],
                }
            ],
        }
        data = self._post_json(
            "https://api.anthropic.com/v1/messages",
            payload,
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
            },
        )
        text = self._extract_anthropic_text(data.get("content"))
        if not text:
            raise RuntimeError("Anthropic returned an empty response")
        return text

    def request_roi_ai_analysis(self, data: dict[str, Any]) -> None:
        capture = self.perception.get_last_roi_capture_context()
        if capture is None or not capture.get("image"):
            self.perception.set_status("Capture ROI first before Ask AI", "warn")
            return
        prompt = str(data.get("prompt", "") or "")
        requested_provider = self._normalize_ai_provider(data.get("provider", "auto"))
        self.broadcast(
            {
                "type": "roi_ai_status",
                "stage": "started",
                "provider": requested_provider,
                "captured_at": capture.get("captured_at", 0),
            }
        )
        thread = threading.Thread(
            target=self._run_roi_ai_analysis,
            args=(capture, requested_provider, prompt),
            daemon=True,
            name="InsightRoiAI",
        )
        thread.start()

    def _run_roi_ai_analysis(self, capture: dict[str, Any], requested_provider: str, prompt: str) -> None:
        captured_at = capture.get("captured_at", 0)
        try:
            provider = self._resolve_ai_provider(requested_provider)
            built_prompt = self._build_roi_ai_prompt(capture.get("scan_results", []), prompt)
            image_b64 = str(capture.get("image", ""))
            if provider == "openai":
                result = self._ask_openai(image_b64, built_prompt)
            elif provider == "anthropic":
                result = self._ask_anthropic(image_b64, built_prompt)
            else:
                result = self._ask_ollama(image_b64, built_prompt)
            self.broadcast(
                {
                    "type": "roi_ai_result",
                    "provider": provider,
                    "text": result,
                    "error": "",
                    "captured_at": captured_at,
                }
            )
            self.perception.set_status(f"ROI AI complete ({provider})", "info")
        except Exception as exc:
            self.broadcast(
                {
                    "type": "roi_ai_result",
                    "provider": requested_provider,
                    "text": "",
                    "error": str(exc),
                    "captured_at": captured_at,
                }
            )
            self.perception.set_status(f"ROI AI error: {exc}", "warn")

    def send_scenario_catalog(self) -> None:
        try:
            scenarios = mlops_registry.list_enabled_scenarios()
            self.broadcast({"type": "scenario_catalog", "scenarios": scenarios, "error": ""})
        except Exception as exc:
            self.broadcast({"type": "scenario_catalog", "scenarios": [], "error": str(exc)})
            self.perception.set_status(f"Scenario catalog error: {exc}", "warn")

    def _resolve_scenario_image(self, data: dict[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
        source = str(data.get("source") or "").strip().lower()
        if source not in {"roi", "history"}:
            raise RuntimeError("Scenario source must be 'roi' or 'history'")

        if source == "roi":
            capture = self.perception.get_last_roi_capture_context()
            if capture is None or not capture.get("image"):
                raise RuntimeError("Capture ROI first before running a scenario")
            image = decode_jpeg_base64(str(capture.get("image", "")))
            if image is None:
                raise RuntimeError("Failed to decode ROI capture image")
            source_ref = {"captured_at": capture.get("captured_at", 0)}
            return image, source, source_ref

        entry_id = data.get("entry_id")
        try:
            entry_id = int(entry_id)
        except (TypeError, ValueError):
            raise RuntimeError("History source requires a valid entry_id")
        history_image = ""
        with self.perception.state_lock:
            for entry in self.perception.detection_history:
                if int(entry.get("entry_id", -1)) == entry_id:
                    history_image = str(entry.get("image", ""))
                    break
        if not history_image:
            raise RuntimeError(f"History entry not found: {entry_id}")
        image = decode_jpeg_base64(history_image)
        if image is None:
            raise RuntimeError(f"Failed to decode history image for entry_id={entry_id}")
        return image, source, {"entry_id": entry_id}

    def request_scenario_run(self, data: dict[str, Any]) -> None:
        scenario = str(data.get("scenario", "") or "").strip()
        if not scenario:
            self.perception.set_status("Scenario name is required", "warn")
            return

        run_id = self._next_run_id()
        started_at = round(time.time(), 3)
        try:
            image_bgr, source, source_ref = self._resolve_scenario_image(data)
        except Exception as exc:
            finished_at = round(time.time(), 3)
            self.broadcast(
                {
                    "type": "scenario_status",
                    "run_id": run_id,
                    "stage": "error",
                    "scenario": scenario,
                    "source": str(data.get("source", "")),
                    "source_ref": {},
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error": str(exc),
                }
            )
            self.broadcast(
                {
                    "type": "scenario_result",
                    "run_id": run_id,
                    "scenario": scenario,
                    "source": str(data.get("source", "")),
                    "source_ref": {},
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "ran_at": started_at,
                    "elapsed_ms": 0.0,
                    "detections": [],
                    "signal": {"flag": False, "summary": "", "metrics": {}},
                    "overlay_image": "",
                    "error": str(exc),
                }
            )
            self.perception.set_status(f"Scenario run error: {exc}", "warn")
            return

        self.broadcast(
            {
                "type": "scenario_status",
                "run_id": run_id,
                "stage": "queued",
                "scenario": scenario,
                "source": source,
                "source_ref": source_ref,
                "started_at": started_at,
                "error": "",
            }
        )
        thread = threading.Thread(
            target=self._run_scenario_inference,
            args=(run_id, scenario, source, source_ref, started_at, image_bgr),
            daemon=True,
            name=f"InsightScenario-{scenario}",
        )
        thread.start()

    def _run_scenario_inference(
        self,
        run_id: str,
        scenario: str,
        source: str,
        source_ref: dict[str, Any],
        started_at: float,
        image_bgr: np.ndarray,
    ) -> None:
        self.broadcast(
            {
                "type": "scenario_status",
                "run_id": run_id,
                "stage": "running",
                "scenario": scenario,
                "source": source,
                "source_ref": source_ref,
                "started_at": started_at,
                "error": "",
            }
        )
        finished_at = round(time.time(), 3)
        try:
            def _cell_cb(evt: Any) -> None:
                try:
                    payload = evt if isinstance(evt, dict) else {"message": str(evt)}
                except Exception:
                    payload = {"message": "unknown"}
                ts = round(time.time(), 3)
                if isinstance(payload, dict) and str(payload.get("type") or "").strip().lower() == "log":
                    self.broadcast(
                        {
                            "type": "scenario_log",
                            "run_id": run_id,
                            "scenario": scenario,
                            "ts": ts,
                            "phase": str(payload.get("phase") or ""),
                            "message": str(payload.get("message") or ""),
                        }
                    )
                    return
                if isinstance(payload, dict) and payload.get("cell_name") and payload.get("cell_status"):
                    self.broadcast(
                        {
                            "type": "scenario_cell",
                            "run_id": run_id,
                            "scenario": scenario,
                            "ts": ts,
                            "cell_index": payload.get("cell_index", -1),
                            "cell_name": payload.get("cell_name", ""),
                            "cell_status": payload.get("cell_status", ""),
                            "output": payload.get("output", ""),
                            "elapsed_ms": payload.get("elapsed_ms", 0),
                        }
                    )

            result = mlops_infer.run_scenario(scenario, image_bgr, cell_callback=_cell_cb, job_id=run_id)
            finished_at = round(time.time(), 3)
            stage = "error" if result.get("error") else "done"
            payload = {
                "type": "scenario_result",
                "run_id": run_id,
                "scenario": str(result.get("scenario", scenario)),
                "source": source,
                "source_ref": source_ref,
                "started_at": started_at,
                "finished_at": finished_at,
                "ran_at": result.get("ran_at", started_at),
                "elapsed_ms": result.get("elapsed_ms", 0.0),
                "detections": result.get("detections", []),
                "signal": result.get("signal", {"flag": False, "summary": "", "metrics": {}}),
                "overlay_image": result.get("overlay_image", ""),
                "error": str(result.get("error", "")),
            }
            self.broadcast(
                {
                    "type": "scenario_status",
                    "run_id": run_id,
                    "stage": stage,
                    "scenario": payload["scenario"],
                    "source": source,
                    "source_ref": source_ref,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error": payload["error"],
                }
            )
            self.broadcast(payload)
            integration_event = {
                "event_id": run_id,
                "run_id": run_id,
                "scenario": payload["scenario"],
                "source": source,
                "source_ref": source_ref,
                "stage": stage,
                "started_at": started_at,
                "finished_at": finished_at,
                "ran_at": payload["ran_at"],
                "elapsed_ms": payload["elapsed_ms"],
                "detections": payload["detections"],
                "signal": payload["signal"],
                "error": payload["error"],
                "overlay_image_present": bool(payload["overlay_image"]),
            }
            append_integration_event(integration_event)
            if stage == "done":
                self.perception.set_status(f"Scenario '{payload['scenario']}' complete", "info")
            else:
                self.perception.set_status(f"Scenario '{payload['scenario']}' error: {payload['error']}", "warn")
        except Exception as exc:
            finished_at = round(time.time(), 3)
            error_message = str(exc)
            self.broadcast(
                {
                    "type": "scenario_status",
                    "run_id": run_id,
                    "stage": "error",
                    "scenario": scenario,
                    "source": source,
                    "source_ref": source_ref,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "error": error_message,
                }
            )
            self.broadcast(
                {
                    "type": "scenario_result",
                    "run_id": run_id,
                    "scenario": scenario,
                    "source": source,
                    "source_ref": source_ref,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "ran_at": started_at,
                    "elapsed_ms": 0.0,
                    "detections": [],
                    "signal": {"flag": False, "summary": "", "metrics": {}},
                    "overlay_image": "",
                    "error": error_message,
                }
            )
            append_integration_event(
                {
                    "event_id": run_id,
                    "run_id": run_id,
                    "scenario": scenario,
                    "source": source,
                    "source_ref": source_ref,
                    "stage": "error",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "ran_at": started_at,
                    "elapsed_ms": 0.0,
                    "detections": [],
                    "signal": {"flag": False, "summary": "", "metrics": {}},
                    "error": error_message,
                    "overlay_image_present": False,
                }
            )
            self.perception.set_status(f"Scenario run error: {error_message}", "warn")

    def _switch_target_label(self, target_source: str) -> str:
        if target_source == "camera":
            return f"camera {self.config.camera_index}"
        return f"video {self.config.video_path.name}"

    async def _async_switch_source(self, requested: Optional[str]) -> None:
        current_source = self.frame_source.current_source
        target_source = requested or ("video" if current_source == "camera" else "camera")
        target_label = self._switch_target_label(target_source)
        self.broadcast(
            {"type": "source_switch", "stage": "starting", "target": target_source, "label": target_label}
        )
        loop = asyncio.get_running_loop()
        max_attempts = 3
        last_exc: Optional[Exception] = None
        for attempt in range(max_attempts):
            try:
                await loop.run_in_executor(None, self.frame_source.prepare_switch, requested)
                self.broadcast(
                    {"type": "source_switch", "stage": "prepared", "target": target_source, "label": target_label}
                )
                return
            except Exception as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    await asyncio.sleep(0.4)
        self.broadcast(
            {"type": "source_switch", "stage": "failed", "target": requested, "message": str(last_exc)}
        )
        self.perception.set_status(str(last_exc), "error")

    async def _async_confirm_switch(self, requested: Optional[str]) -> None:
        self.broadcast({"type": "source_switch", "stage": "committing", "target": requested})
        loop = asyncio.get_running_loop()
        try:
            active_source = await loop.run_in_executor(
                None, self.frame_source.commit_prepared_switch, requested
            )
            self.perception.reset_scene()
            self.broadcast(
                {
                    "type": "source_switch",
                    "stage": "ready",
                    "target": active_source,
                    "label": self._switch_target_label(active_source),
                }
            )
        except Exception as exc:
            self.broadcast(
                {"type": "source_switch", "stage": "failed", "target": requested, "message": str(exc)}
            )
            self.perception.set_status(str(exc), "error")

    def handle_client_message(self, data: dict[str, Any]) -> None:
        message_type = data.get("type")
        if message_type == "client_ready":
            self.perception.publish_state(force=True)
            return
        if message_type == "list_scenarios":
            self.send_scenario_catalog()
            return
        if message_type == "select_track":
            try:
                self.perception.set_focus(int(data["track_id"]))
            except (KeyError, TypeError, ValueError):
                self.perception.set_status("Invalid track selection", "warn")
            return
        if message_type == "clear_focus":
            self.perception.clear_focus()
            return
        if message_type == "switch_source":
            requested = data.get("source")
            if requested not in (None, "camera", "video"):
                self.perception.set_status("Invalid source switch request", "warn")
                return
            loop = self._loop
            if loop:
                loop.create_task(self._async_switch_source(requested))
            return
        if message_type == "confirm_source_switch":
            requested = data.get("source")
            if requested not in (None, "camera", "video"):
                self.perception.set_status("Invalid source confirmation request", "warn")
                return
            loop = self._loop
            if loop:
                loop.create_task(self._async_confirm_switch(requested))
            return
        if message_type == "cancel_source_switch":
            self.frame_source.cancel_prepared_switch()
            self.broadcast({"type": "source_switch", "stage": "cancelled"})
            return
        if message_type == "set_roi":
            try:
                self.perception.set_roi(
                    float(data["x1"]),
                    float(data["y1"]),
                    float(data["x2"]),
                    float(data["y2"]),
                    shape=data.get("shape", "rect"),
                )
            except (KeyError, TypeError, ValueError):
                self.perception.set_status("Invalid ROI coordinates", "warn")
            return
        if message_type == "clear_roi":
            self.perception.clear_roi()
            return
        if message_type == "capture_roi":
            self.perception.capture_roi_snapshot()
            return
        if message_type == "ask_ai_roi":
            self.request_roi_ai_analysis(data)
            return
        if message_type == "run_scenario":
            self.request_scenario_run(data)
            return
        if message_type == "clear_history":
            self.perception.clear_history()
            return
        if message_type == "delete_history_entry":
            try:
                self.perception.delete_history_entry(int(data["entry_id"]))
            except (KeyError, TypeError, ValueError):
                self.perception.set_status("Invalid history entry id", "warn")
            return
        if message_type == "update_settings":
            settings = data.get("settings")
            if isinstance(settings, dict):
                self.perception.update_settings(settings)
            else:
                self.perception.set_status("Settings payload ignored", "warn")
            return
        self.perception.set_status(f"Unknown message type: {message_type}", "warn")

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.perception.stop()
        self.frame_source.cleanup()

    async def close_async(self) -> None:
        if self.closed:
            return
        self.closed = True
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._close_blocking)

    def _close_blocking(self) -> None:
        self.perception.stop()
        self.frame_source.cleanup()


class InsightVideoTrack(VideoStreamTrack):
    def __init__(self, session: InsightSession):
        super().__init__()
        self.session = session

    async def recv(self) -> VideoFrame:
        pts, time_base = await self.next_timestamp()
        loop = asyncio.get_running_loop()
        try:
            frame = await loop.run_in_executor(None, self.session.frame_source.read)
            self.session.perception.update_frame(frame)
        except Exception as exc:
            self.session.perception.set_status(str(exc), "error")
            height, width = self.session.frame_source.last_shape
            frame = render_status_frame(str(exc), width=width, height=height)
            await asyncio.sleep(0.05)

        video_frame = VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="bgr24")
        video_frame.pts = pts
        video_frame.time_base = time_base
        return video_frame


INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="ViewPort" content="width=device-width, initial-scale=1" />
  <title>ATLAS Tactical HUD</title>
  <style>
    /* ============================================================
       ATLAS — Multi-Band Red / Hard UI Tactical Display

       RED HIERARCHY (5 tiers):
         T1 iron    — structural: rails, dividers, panel skeleton
         T2 signal  — operational: active states, selection, module markers
         T3 caution — attention: warm-shifted, not catastrophic
         T4 hot     — high alert: immediate recognition required
         T5 crit    — emergency: spatially rare, luminance spike

       SUPPORT: white text, yellow detection titles, muted gray metadata
       DOCTRINE: urgency = convergence of placement + luminance + wording
       ============================================================ */
    :root {
      /* -- Backgrounds / Planes — deep iron-red, not black -- */
      --hud-bg: #140808;
      --hud-panel: rgba(26, 10, 10, 0.55);
      --hud-panel-strong: rgba(20, 8, 8, 0.60);

      /* -- T1: Iron red — structural skeleton, neon-tinted -- */
      --red-iron: #8a2222;
      --red-iron-alpha: rgba(138, 34, 34, 0.50);

      /* -- T2: Signal red — operational accents, bright neon -- */
      --red-signal: #ff2a2a;
      --red-signal-alpha: rgba(255, 42, 42, 0.30);

      /* -- T3: Caution — attention-worthy, warm neon -- */
      --red-caution: #ff6030;

      /* -- T4: Hot — high alert, full neon -- */
      --red-hot: #ff1a1a;

      /* -- T5: Critical — emergency only, peak neon + glow -- */
      --red-crit: #ff0020;
      --red-crit-glow: 0 0 12px rgba(255, 0, 32, 0.7), 0 0 4px rgba(255, 0, 32, 0.9);

      /* -- Support palette -- */
      --hud-yellow: #ffe040;
      --hud-text: #e0e0e0;
      --hud-text-bright: #f4f4f4;
      --hud-muted: rgba(190, 180, 175, 0.50);

      /* -- Borders (mapped to red tiers) -- */
      --border-iron: rgba(138, 34, 34, 0.55);
      --border-signal: rgba(255, 42, 42, 0.40);
      --border-hot: rgba(255, 26, 26, 0.65);

      /* -- Infrastructure -- */
      --hud-shadow: 0 2px 8px rgba(20, 4, 4, 0.60);
      --mono: "IBM Plex Mono", "SFMono-Regular", Menlo, monospace;
      --line-light: 1px;
      --line-medium: 2px;
      --line-heavy: 3px;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      background: #120808;
      color: var(--hud-text);
      font-family: var(--mono);
      font-size: 12px;
      overflow: hidden;
      -webkit-font-smoothing: antialiased;
    }

    #sceneVideo {
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      object-fit: cover;
      z-index: 0;
      background: #0e0606;
    }

    .veil {
      position: fixed;
      inset: 0;
      background: linear-gradient(180deg, rgba(18,6,6,0.02) 0%, rgba(18,6,6,0.15) 100%);
      pointer-events: none;
      z-index: 1;
    }

    .loading-screen {
      position: fixed;
      left: 50%;
      top: 50%;
      transform: translate(-50%, -50%);
      z-index: 6;
      display: flex;
      align-items: center;
      justify-content: center;
      width: min(620px, calc(100vw - 28px));
      pointer-events: none;
      opacity: 1;
      transition: opacity 180ms linear, visibility 180ms linear;
    }
    .loading-screen.hidden {
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
    }
    .loading-panel {
      width: 100%;
      padding: 10px 14px 10px;
      border: var(--line-light) solid rgba(255, 42, 42, 0.20);
      border-radius: 10px;
      background: rgba(14, 6, 6, 0.74);
      backdrop-filter: blur(8px);
      box-shadow: 0 0 18px rgba(255, 26, 26, 0.06);
      pointer-events: auto;
      transition: border-color 90ms linear, box-shadow 90ms linear, background 90ms linear;
    }
    .loading-panel.blink-success {
      border-color: rgba(255, 208, 96, 0.78);
      background: rgba(34, 14, 10, 0.88);
      box-shadow: 0 0 18px rgba(255, 198, 84, 0.18);
    }
    .loading-panel.blink-failure {
      border-color: rgba(255, 42, 42, 0.82);
      background: rgba(34, 8, 8, 0.9);
      box-shadow: 0 0 20px rgba(255, 42, 42, 0.22);
    }
    .loading-line {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      min-width: 0;
      margin-bottom: 8px;
      white-space: nowrap;
    }
    .loading-title {
      flex: 0 0 auto;
      font-size: 12px;
      letter-spacing: 1.4px;
      color: var(--hud-text-bright);
      text-transform: uppercase;
      line-height: 1.05;
    }
    .loading-sep {
      flex: 0 0 12px;
      height: 1px;
      background: rgba(255, 42, 42, 0.28);
    }
    .loading-state {
      font-size: 9px;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      color: var(--hud-muted);
      flex: 0 0 auto;
    }
    .loading-copy {
      font-size: 10px;
      line-height: 1.2;
      letter-spacing: 0.2px;
      color: var(--hud-text);
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .loading-bar {
      position: relative;
      height: 3px;
      overflow: hidden;
      background: rgba(138, 34, 34, 0.26);
      border-radius: 999px;
    }
    .loading-bar::before {
      content: "";
      position: absolute;
      inset: 0;
      background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.05), transparent);
      transform: translateX(-100%);
      animation: loading-sweep 1.4s linear infinite;
      opacity: 0.45;
    }
    .loading-fill {
      position: absolute;
      inset: 0 auto 0 0;
      width: 0%;
      background: linear-gradient(90deg, rgba(140, 20, 20, 0.55) 0%, rgba(255, 42, 42, 0.95) 65%, rgba(255, 198, 84, 0.9) 100%);
      box-shadow: 0 0 12px rgba(255, 42, 42, 0.20);
      border-radius: 999px;
      transition: width 220ms ease;
    }
    .loading-actions {
      display: none;
      align-items: center;
      justify-content: center;
      gap: 8px;
      margin-top: 8px;
    }
    .loading-actions.visible {
      display: flex;
    }
    .loading-action {
      padding: 4px 10px;
      border: var(--line-light) solid rgba(255, 42, 42, 0.22);
      border-radius: 999px;
      background: rgba(24, 10, 10, 0.72);
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 1.1px;
      cursor: pointer;
      transition: background 60ms linear, color 60ms linear, border-color 60ms linear;
    }
    .loading-action:hover {
      color: var(--hud-text);
      background: rgba(255, 42, 42, 0.10);
    }
    .loading-action.primary {
      color: var(--hud-text-bright);
      border-color: rgba(255, 42, 42, 0.42);
      background: rgba(255, 42, 42, 0.12);
    }
    .loading-action.primary:hover {
      background: rgba(255, 42, 42, 0.20);
      color: var(--hud-text-bright);
    }
    @keyframes loading-sweep {
      from { transform: translateX(-100%); }
      to { transform: translateX(100%); }
    }

    /* ============ FLIGHT STRIP — T1 iron structural ============ */
    .flight-strip {
      position: fixed;
      top: 0; left: 0; right: 0;
      z-index: 3;
      display: flex;
      gap: 0;
      padding: 0;
      border-bottom: var(--line-light) solid var(--red-iron-alpha);
      background: var(--hud-panel-strong);
      align-items: stretch;
      justify-content: space-between;
      transition: transform 0.25s ease, opacity 0.25s ease;
    }
    .flight-strip.dismissed {
      transform: translateY(-100%);
      opacity: 0;
      pointer-events: none;
    }
    .flight-strip-restore {
      position: fixed;
      top: 0; left: 50%;
      transform: translateX(-50%);
      z-index: 3;
      padding: 2px 18px;
      border: var(--line-light) solid var(--red-iron-alpha);
      border-top: none;
      border-radius: 0 0 4px 4px;
      background: var(--hud-panel-strong);
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      cursor: pointer;
      display: none;
      transition: background 60ms linear;
    }
    .flight-strip-restore:hover {
      background: var(--red-signal-alpha);
      color: var(--hud-text);
    }
    .flight-strip-restore.visible {
      display: block;
    }
    .flight-dismiss {
      display: flex;
      align-items: center;
      padding: 0 10px;
      cursor: pointer;
      color: var(--hud-muted);
      font-size: 14px;
      font-family: var(--mono);
      transition: color 60ms linear;
      border: none;
      background: none;
    }
    .flight-dismiss:hover {
      color: var(--hud-text);
    }

    .flight-group {
      display: flex;
      gap: 0;
      align-items: stretch;
      flex-wrap: nowrap;
    }

    .flight-chip {
      min-width: 80px;
      padding: 6px 12px;
      background: transparent;
      border-right: var(--line-light) solid rgba(138, 34, 34, 0.25);
      display: flex;
      flex-direction: column;
      gap: 1px;
      justify-content: center;
    }

    .flight-chip.buttonish {
      cursor: pointer;
      transition: background 80ms linear;
    }

    .flight-chip.buttonish:hover {
      background: var(--red-signal-alpha);
    }

    .chip-label {
      font-size: 9px;
      letter-spacing: 1.6px;
      text-transform: uppercase;
      color: var(--hud-muted);
    }

    .chip-value {
      font-size: 14px;
      letter-spacing: 0.5px;
      color: var(--hud-text-bright);
      font-family: var(--mono);
      white-space: nowrap;
      font-weight: 600;
    }

    .flight-title {
      font-size: 16px;
      font-weight: 700;
      letter-spacing: 3px;
      color: var(--red-signal);
      padding: 0 16px;
      line-height: 1;
    }

    .flight-subtitle {
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: var(--hud-muted);
      padding: 0 16px;
    }

    /* ============ SIDEBAR — T1 iron skeleton, horizontally draggable ============ */
    .sidebar {
      position: fixed;
      top: 46px; bottom: 0;
      width: min(28vw, 340px);
      z-index: 3;
      display: flex;
      flex-direction: column;
      border-left: var(--line-light) solid var(--red-iron-alpha);
      border-right: var(--line-light) solid var(--red-iron-alpha);
      background: var(--hud-panel-strong);
      overflow: hidden;
      transform-origin: right center;
      transform: perspective(900px) rotateY(-4deg);
      transition: transform 0.25s ease, opacity 0.25s ease;
    }
    .sidebar.dismissed {
      transform: translateX(calc(100% + 24px));
      opacity: 0;
      pointer-events: none;
    }

    .sidebar-drag {
      position: absolute;
      top: 0; left: -8px;
      width: 8px; height: 100%;
      cursor: ew-resize;
      z-index: 8;
      background: transparent;
    }
    .sidebar-drag:hover,
    .sidebar-drag.dragging {
      background: var(--red-signal-alpha);
    }
    .sidebar-drag::after {
      content: "";
      position: absolute;
      top: 50%; left: 2px;
      width: 3px; height: 28px;
      transform: translateY(-50%);
      background: var(--red-iron);
      border-right: var(--line-light) solid var(--red-iron-alpha);
    }

    .sidebar-tabs {
      display: flex;
      flex-shrink: 0;
      border-bottom: var(--line-light) solid var(--red-iron-alpha);
      background: rgba(12, 4, 4, 0.60);
    }

    .sidebar-tab {
      flex: 1;
      padding: 8px 0;
      text-align: center;
      font-size: 10px;
      font-family: var(--mono);
      text-transform: uppercase;
      letter-spacing: 1.6px;
      color: var(--hud-muted);
      cursor: pointer;
      border: none;
      background: none;
      transition: color 60ms linear, background 60ms linear;
      position: relative;
    }

    .sidebar-tab:hover {
      color: var(--hud-text);
      background: rgba(255, 255, 255, 0.04);
    }

    /* T2 signal red — active tab marker */
    .sidebar-tab.active {
      color: var(--hud-text-bright);
      background: rgba(138, 34, 34, 0.18);
      text-shadow: 0 0 6px rgba(255, 42, 42, 0.30);
    }

    .sidebar-tab.active::after {
      content: "";
      position: absolute;
      bottom: 0; left: 0; right: 0;
      height: var(--line-light);
      background: var(--red-signal);
      box-shadow: 0 0 4px rgba(255, 42, 42, 0.30);
    }
    .sidebar-dismiss {
      flex: 0 0 36px;
      border: none;
      border-left: var(--line-light) solid var(--red-iron-alpha);
      background: none;
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 13px;
      cursor: pointer;
      transition: color 60ms linear, background 60ms linear;
    }
    .sidebar-dismiss:hover {
      color: var(--hud-text);
      background: rgba(255, 42, 42, 0.08);
    }
    .sidebar-restore {
      position: fixed;
      top: 50%;
      right: 0;
      transform: translateY(-50%);
      z-index: 3;
      display: none;
      padding: 10px 8px;
      border: var(--line-light) solid var(--red-iron-alpha);
      border-right: none;
      border-radius: 4px 0 0 4px;
      background: var(--hud-panel-strong);
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 1.6px;
      writing-mode: vertical-rl;
      text-orientation: mixed;
      cursor: pointer;
      transition: background 60ms linear, color 60ms linear;
    }
    .sidebar-restore:hover {
      background: var(--red-signal-alpha);
      color: var(--hud-text);
    }
    .sidebar-restore.visible {
      display: block;
    }

    .sidebar-pane {
      display: none;
      flex: 1;
      overflow-y: auto;
      background: rgba(18, 7, 7, 0.25);
      scrollbar-width: thin;
      scrollbar-color: var(--red-iron-alpha) transparent;
    }

    .sidebar-pane.active {
      display: flex;
      flex-direction: column;
    }

    .pane-dismiss-row {
      display: flex;
      justify-content: flex-end;
      padding: 8px 8px 0;
      flex-shrink: 0;
    }

    .pane-dismiss {
      min-width: 34px;
      height: 28px;
      border: var(--line-light) solid var(--red-iron-alpha);
      background: rgba(12, 4, 4, 0.6);
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 12px;
      cursor: pointer;
      transition: color 60ms linear, background 60ms linear, border-color 60ms linear;
    }

    .pane-dismiss:hover {
      color: var(--hud-text);
      background: rgba(255, 42, 42, 0.08);
      border-color: rgba(255, 42, 42, 0.24);
    }

    /* ============ PREVIEWS PANE ============ */
    .tile-rail {
      display: flex;
      flex-direction: column;
      gap: 0;
      padding: 0;
    }

    .tile-card {
      width: 100%;
      display: grid;
      grid-template-columns: 96px 1fr;
      gap: 10px;
      padding: 10px;
      border: none;
      border-bottom: var(--line-light) solid var(--red-iron-alpha);
      background: rgba(24, 10, 10, 0.55);
      cursor: pointer;
      transition: background 60ms linear;
      text-align: left;
      color: inherit;
      font-family: var(--mono);
    }

    /* T2 signal — hover selection */
    .tile-card:hover {
      background: rgba(138, 34, 34, 0.14);
    }

    .tile-thumb {
      width: 96px;
      height: 96px;
      object-fit: cover;
      background: rgba(10, 4, 4, 0.70);
      border: var(--line-light) solid var(--red-iron-alpha);
    }

    .tile-meta {
      display: flex;
      flex-direction: column;
      gap: 6px;
      min-width: 0;
      background: rgba(30, 12, 12, 0.30);
      padding: 4px 6px;
      margin: -4px -6px;
    }

    .tile-heading {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }

    /* Detection title — YELLOW (support palette) */
    .tile-title {
      font-size: 14px;
      font-weight: 700;
      color: var(--hud-yellow);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    .tile-track {
      color: var(--hud-muted);
      font-size: 11px;
    }

    /* T1 iron — tag/pill trim */
    .tile-pill {
      align-self: flex-start;
      padding: 2px 8px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      border: var(--line-light) solid var(--red-iron-alpha);
      background: rgba(138, 34, 34, 0.18);
      color: var(--hud-text);
    }

    .tile-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 4px;
      font-size: 11px;
      color: var(--hud-text);
      padding-top: 4px;
      border-top: var(--line-light) solid rgba(138, 34, 34, 0.20);
    }

    .tile-stat span {
      display: block;
      font-size: 9px;
      letter-spacing: 1.2px;
      text-transform: uppercase;
      color: var(--hud-muted);
      margin-bottom: 1px;
    }

    /* ============ FOCUS PANE ============ */
    .focus-pane-inner { padding: 12px; }

    .focus-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 10px;
    }

    .focus-title {
      font-size: 16px;
      font-weight: 700;
      color: var(--hud-yellow);
      margin: 0;
    }

    .focus-copy {
      font-size: 10px;
      letter-spacing: 1.4px;
      text-transform: uppercase;
      color: var(--hud-muted);
      margin-top: 2px;
    }

    .focus-clear {
      border: var(--line-light) solid var(--red-iron-alpha);
      background: rgba(138, 34, 34, 0.15);
      color: var(--hud-text);
      padding: 4px 10px;
      cursor: pointer;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      flex-shrink: 0;
      transition: background 60ms linear;
    }

    .focus-clear:hover {
      background: var(--red-signal-alpha);
    }

    /* ============ FOCUS CONTROLS ============ */
    .focus-controls {
      display: flex;
      align-items: center;
      gap: 6px;
      padding: 6px 8px;
      background: rgba(30, 12, 12, 0.35);
      border-bottom: var(--line-light) solid var(--red-iron-alpha);
      margin-bottom: 0;
    }
    .focus-control-btn {
      padding: 3px 8px;
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 9px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 1px;
      transition: background 60ms linear, color 60ms linear;
    }
    .focus-control-btn:hover {
      background: var(--red-signal-alpha);
      color: var(--hud-text);
    }
    .focus-control-btn:active {
      opacity: 0.7;
    }
    .focus-zoom-level {
      font-family: var(--mono);
      font-size: 9px;
      color: var(--hud-muted);
      margin-left: auto;
      white-space: nowrap;
    }

    .focus-visual {
      position: relative;
      overflow: auto;
      background: rgba(8, 3, 3, 0.80);
      border: var(--line-light) solid var(--red-iron-alpha);
      border: var(--line-light) solid var(--red-iron-alpha);
      aspect-ratio: 1.25 / 1;
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .focus-visual-inner {
      position: relative;
      width: 100%;
      height: 100%;
      transform-origin: center;
      transition: transform 0.2s ease;
    }

    .focus-image,
    .focus-silhouette {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      object-fit: cover;
    }

    .focus-silhouette {
      mix-blend-mode: screen;
      opacity: 0.92;
      pointer-events: none;
    }

    .focus-stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      font-size: 11px;
      background: rgba(30, 12, 12, 0.35);
      padding: 8px;
      border-top: var(--line-light) solid var(--red-iron-alpha);
    }

    .focus-stat span {
      display: block;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--hud-muted);
      margin-bottom: 2px;
    }


    .focus-scan {
      border-top: var(--line-light) solid var(--red-iron-alpha);
      padding: 8px;
      background: rgba(30, 12, 12, 0.35);
    }
    .focus-runtime {
      border-top: var(--line-light) solid var(--red-iron-alpha);
      padding: 8px;
      background: rgba(30, 12, 12, 0.35);
      margin-top: 8px;
    }
    .runtime-minimal {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.1px;
      color: var(--hud-muted);
    }
    .runtime-card-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      margin-bottom: 6px;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--hud-muted);
    }
    .runtime-flag-pill {
      border: var(--line-light) solid var(--border-iron);
      padding: 2px 8px;
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 1px;
      background: rgba(138, 34, 34, 0.12);
      color: var(--hud-text);
    }
    .runtime-flag-pill.hot {
      border-color: var(--red-hot);
      color: var(--red-hot);
      background: rgba(255, 26, 26, 0.14);
    }
    .runtime-summary {
      font-size: 11px;
      line-height: 1.4;
      color: var(--hud-text-bright);
      margin-bottom: 8px;
    }
    .runtime-meta {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-bottom: 8px;
      font-size: 10px;
    }
    .runtime-meta div {
      border: var(--line-light) solid var(--red-iron-alpha);
      padding: 4px 6px;
      background: rgba(20, 8, 8, 0.45);
    }
    .runtime-catalog-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .runtime-catalog-item {
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-text);
      text-align: left;
      padding: 6px 8px;
      font-family: var(--mono);
      cursor: pointer;
      transition: background 60ms linear, border-color 60ms linear;
    }
    .runtime-catalog-item:hover {
      border-color: var(--border-signal);
      background: var(--red-signal-alpha);
    }
    .runtime-catalog-name {
      display: block;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--hud-yellow);
      margin-bottom: 3px;
    }
    .runtime-catalog-desc {
      display: block;
      font-size: 10px;
      color: var(--hud-muted);
      line-height: 1.3;
    }
    .runtime-overlay-img {
      width: 100%;
      max-height: 180px;
      object-fit: contain;
      border: var(--line-light) solid var(--red-iron-alpha);
      background: rgba(20, 8, 8, 0.45);
    }
    .focus-scan-header {
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--hud-muted);
      margin-bottom: 6px;
    }
    .focus-scan-empty {
      font-size: 10px;
      color: var(--hud-muted);
      font-style: italic;
      padding: 4px 0;
    }
    .scan-item {
      display: flex;
      align-items: center;
      gap: 8px;
      padding: 4px 0;
      border-bottom: 1px solid rgba(138, 34, 34, 0.20);
    }
    .scan-item:last-child {
      border-bottom: none;
    }
    .scan-label {
      font-size: 11px;
      color: var(--hud-yellow);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      min-width: 72px;
    }
    .scan-conf {
      font-size: 10px;
      color: var(--hud-text-bright);
      min-width: 36px;
      text-align: right;
    }
    .scan-conf-bar {
      flex: 1;
      height: 3px;
      background: rgba(138, 34, 34, 0.30);
      position: relative;
      border-radius: 1px;
    }
    .scan-conf-fill {
      position: absolute;
      top: 0; left: 0;
      height: 100%;
      border-radius: 1px;
      background: var(--red-signal);
      transition: width 0.3s ease;
    }
    .scan-area {
      font-size: 9px;
      color: var(--hud-muted);
      min-width: 40px;
      text-align: right;
    }

    /* ============ FOCUS OVERLAY CANVAS ============ */
    .focus-overlay-canvas {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 2;
    }

    /* Scan overlay toggle buttons */
    .scan-toggles {
      display: flex;
      gap: 6px;
      margin-bottom: 6px;
    }
    .scan-toggle-btn {
      padding: 3px 8px;
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 9px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 1px;
      transition: background 60ms linear, color 60ms linear;
    }
    .scan-toggle-btn:hover {
      background: var(--red-signal-alpha);
      color: var(--hud-text);
    }
    .scan-toggle-btn.active {
      border-color: var(--red-signal);
      background: rgba(255, 42, 42, 0.14);
      color: var(--red-signal);
    }
    .scan-toggle-btn.ai {
      color: var(--hud-yellow);
    }
    .scan-toggle-btn.ai.active {
      border-color: var(--hud-yellow);
      color: var(--hud-yellow);
      background: rgba(255, 214, 120, 0.08);
    }
    .ai-compose {
      display: none;
      border: var(--line-light) solid var(--border-iron);
      background: rgba(22, 9, 9, 0.32);
      padding: 8px;
      margin-bottom: 8px;
      gap: 6px;
      flex-direction: column;
    }
    .ai-compose.visible {
      display: flex;
    }
    .ai-compose-row {
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .ai-provider-select,
    .ai-send-btn {
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-text);
      font-family: var(--mono);
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 0.8px;
      padding: 4px 6px;
    }
    .ai-provider-select {
      min-width: 110px;
    }
    .ai-send-btn {
      cursor: pointer;
      margin-left: auto;
    }
    .ai-send-btn:disabled {
      opacity: 0.6;
      cursor: default;
    }
    .ai-prompt-input {
      width: 100%;
      min-height: 54px;
      resize: vertical;
      border: var(--line-light) solid var(--border-iron);
      background: rgba(18, 7, 7, 0.7);
      color: var(--hud-text-bright);
      font-family: var(--mono);
      font-size: 10px;
      line-height: 1.35;
      padding: 6px;
      box-sizing: border-box;
    }
    .ai-response {
      margin-top: 8px;
      border-top: 1px solid rgba(138, 34, 34, 0.25);
      padding-top: 8px;
    }
    .ai-response-head {
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--hud-muted);
      margin-bottom: 5px;
    }
    .ai-response-text {
      font-size: 11px;
      line-height: 1.45;
      color: var(--hud-text-bright);
      white-space: normal;
    }
    .ai-response-error {
      color: var(--red-caution);
    }
    /* Heat map transparency slider */
    .heatmap-opacity-row {
      display: none;
      align-items: center;
      gap: 6px;
      margin-bottom: 6px;
    }
    .heatmap-opacity-row.visible {
      display: flex;
    }
    .heatmap-opacity-label {
      font-family: var(--mono);
      font-size: 9px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: var(--hud-muted);
      white-space: nowrap;
    }
    .heatmap-opacity-slider {
      -webkit-appearance: none;
      appearance: none;
      flex: 1;
      height: 3px;
      background: rgba(138, 34, 34, 0.30);
      border-radius: 2px;
      outline: none;
    }
    .heatmap-opacity-slider::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 10px;
      height: 10px;
      border-radius: 50%;
      background: var(--red-signal);
      cursor: pointer;
    }


    .focus-empty {
      padding: 32px 16px;
      text-align: center;
      color: var(--hud-muted);
      background: rgba(22, 9, 9, 0.30);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
    }

    /* ============ EVENTS PANE ============ */
    .event-list-inner {
      display: flex;
      flex-direction: column;
      gap: 0;
      padding: 0;
    }

    .event-item {
      padding: 8px 10px;
      background: rgba(26, 10, 10, 0.40);
      border-bottom: var(--line-light) solid var(--red-iron-alpha);
      border-left: var(--line-light) solid var(--red-iron-alpha);
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .event-item:nth-child(even) {
      background: rgba(22, 9, 9, 0.30);
    }

    /* T2 signal — event tag accent */
    .event-item strong {
      color: var(--red-signal);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      flex-shrink: 0;
    }

    .event-item span {
      flex: 1;
      font-size: 11px;
    }
    .event-item.runtime {
      border-left-color: rgba(255, 214, 120, 0.45);
      background: rgba(28, 14, 8, 0.45);
    }
    .event-item.runtime strong {
      color: var(--hud-yellow);
    }
    .event-item.runtime.done strong {
      color: rgba(94, 224, 141, 0.95);
    }
    .event-item.runtime.error strong {
      color: var(--red-hot);
    }
    .event-item.runtime.clickable {
      cursor: pointer;
    }
    .event-item.runtime.clickable:hover {
      background: rgba(255, 42, 42, 0.08);
    }

    /* ============ STATUS TOAST — T3 caution when warn ============ */
    .status-toast {
      position: fixed;
      left: 50%;
      transform: translateX(-50%);
      bottom: 24px;
      z-index: 4;
      padding: 6px 14px;
      background: var(--hud-panel-strong);
      border: var(--line-light) solid var(--border-iron);
      color: var(--hud-text);
      font-size: 11px;
      letter-spacing: 0.6px;
      opacity: 0;
      pointer-events: none;
      transition: opacity 100ms linear;
    }

    .status-toast.visible { opacity: 1; }

    .empty-state {
      padding: 16px;
      border: var(--line-light) dashed var(--red-iron-alpha);
      color: var(--hud-muted);
      font-size: 11px;
      background: rgba(24, 10, 10, 0.30);
      text-align: center;
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    @media (max-width: 980px) {
      .flight-strip {
        flex-direction: column;
        align-items: stretch;
      }
      .sidebar {
        width: 100vw;
        top: auto; left: 0 !important; right: 0; bottom: 0;
        height: 38vh;
        border-left: none;
        border-right: none;
        border-top: var(--line-light) solid var(--red-iron-alpha);
        transform: none;
        transform-origin: center center;
      }
      .sidebar.dismissed {
        transform: translateY(100%);
      }
      .sidebar-drag { display: none; }
      .sidebar-restore {
        top: auto;
        bottom: 38vh;
        right: 12px;
        transform: none;
        writing-mode: horizontal-tb;
        border-right: var(--line-light) solid var(--red-iron-alpha);
        border-radius: 4px;
        padding: 6px 10px;
      }
    }

    /* ============ TIMELINE — T1 iron frame, T2 signal interaction ============ */
    .timeline-bar {
      position: fixed;
      bottom: 0; left: 0; right: 0;
      z-index: 5;
      background: var(--hud-panel-strong);
      border-top: var(--line-light) solid var(--red-iron-alpha);
      display: flex;
      flex-direction: column;
      max-height: 200px;
      transition: transform 120ms linear;
      transform: translateY(100%);
    }
    .timeline-bar.open { transform: translateY(0); }

    .timeline-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 6px 12px;
      border-bottom: var(--line-light) solid var(--red-iron-alpha);
      flex-shrink: 0;
    }
    .timeline-title {
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 2px;
      color: var(--hud-muted);
    }
    /* T2 signal — count accent */
    .timeline-count {
      font-size: 10px;
      color: var(--red-signal);
      margin-left: 6px;
    }
    .timeline-actions { display: flex; gap: 4px; }
    .timeline-btn {
      border: var(--line-light) solid var(--red-iron-alpha);
      background: rgba(138, 34, 34, 0.12);
      color: var(--hud-text);
      padding: 3px 10px;
      cursor: pointer;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      font-family: var(--mono);
      transition: background 60ms linear;
    }
    .timeline-btn:hover {
      background: var(--red-signal-alpha);
    }
    /* T4 hot — danger button escalation */
    .timeline-btn.danger:hover {
      background: rgba(255, 26, 26, 0.14);
      border-color: var(--red-hot);
      color: var(--red-hot);
    }

    .timeline-scroll {
      display: flex;
      gap: 6px;
      padding: 8px 12px 12px;
      overflow-x: auto;
      overflow-y: hidden;
      scrollbar-width: auto;
      scrollbar-color: var(--red-iron) rgba(20, 8, 8, 0.40);
      flex: 1;
      align-items: flex-start;
    }
    .timeline-scroll::-webkit-scrollbar {
      height: 10px;
    }
    .timeline-scroll::-webkit-scrollbar-track {
      background: rgba(20, 8, 8, 0.40);
      border-top: var(--line-light) solid var(--red-iron-alpha);
    }
    .timeline-scroll::-webkit-scrollbar-thumb {
      background: var(--red-iron);
      border: 1px solid rgba(255, 42, 42, 0.20);
    }
    .timeline-scroll::-webkit-scrollbar-thumb:hover {
      background: rgba(180, 40, 40, 0.80);
    }

    .timeline-card {
      flex-shrink: 0;
      width: 132px;
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel);
      cursor: pointer;
      overflow: hidden;
      transition: background 60ms linear, border-color 60ms linear;
      display: flex;
      flex-direction: column;
      font-family: var(--mono);
      color: inherit;
      text-align: left;
      padding: 0;
      position: relative;
    }
    /* T2 signal — card hover */
    .timeline-card:hover {
      border-color: var(--border-signal);
      background: rgba(255, 42, 42, 0.06);
    }
    .timeline-card-img {
      width: 100%;
      height: 80px;
      object-fit: cover;
      background: rgba(20, 8, 8, 0.45);
      display: block;
      border-bottom: var(--line-light) solid var(--red-iron-alpha);
    }
    .timeline-card-body {
      padding: 4px 6px;
      display: flex;
      flex-direction: column;
      gap: 1px;
    }
    .timeline-card-label {
      font-size: 11px;
      font-weight: 700;
      color: var(--hud-yellow);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .timeline-card-meta {
      font-size: 9px;
      color: var(--hud-muted);
      display: flex;
      justify-content: space-between;
    }
    /* T4 hot — delete action */
    .timeline-card-del {
      position: absolute;
      top: 2px; right: 2px;
      width: 18px; height: 18px;
      border: var(--line-light) solid rgba(255, 26, 26, 0.35);
      background: rgba(20, 6, 6, 0.85);
      color: var(--red-hot);
      font-size: 12px;
      line-height: 18px;
      text-align: center;
      cursor: pointer;
      display: none;
      padding: 0;
    }
    .timeline-card:hover .timeline-card-del { display: block; }

    .timeline-ttl {
      position: absolute;
      top: 3px; left: 3px;
      width: 20px; height: 20px;
    }
    .timeline-ttl-bg {
      fill: none;
      stroke: var(--red-iron-alpha);
      stroke-width: 2.5;
    }
    /* TTL ring: white healthy -> T3 caution -> T4 hot */
    .timeline-ttl-fg {
      fill: none;
      stroke: var(--hud-text);
      stroke-width: 2.5;
      stroke-linecap: butt;
      transition: stroke 0.3s linear;
    }
    .timeline-ttl-fg.warn {
      stroke: var(--red-caution);
    }
    .timeline-ttl-fg.danger {
      stroke: var(--red-hot);
    }

    /* T2 signal — toggle buttons */
    .timeline-toggle {
      position: fixed;
      bottom: 12px; left: 156px;
      z-index: 10;
      padding: 5px 12px;
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-text);
      font-family: var(--mono);
      font-size: 10px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      transition: background 60ms linear;
    }
    .timeline-toggle:hover {
      background: var(--red-signal-alpha);
    }
    .timeline-toggle.active {
      border-color: var(--red-signal);
      background: rgba(255, 42, 42, 0.14);
      color: var(--red-signal);
    }

    .timeline-empty {
      padding: 16px;
      text-align: center;
      color: var(--hud-muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      width: 100%;
    }

    /* ============ ROI OVERLAY ============ */
    .roi-box {
      position: fixed;
      border: var(--line-light) solid rgba(138, 34, 34, 0.55);
      background: rgba(255, 42, 42, 0.02);
      box-shadow: 0 0 6px rgba(255, 42, 42, 0.06);
      z-index: 6;
      pointer-events: none;
      display: none;
      overflow: visible;
    }
    .roi-box.active {
      display: block;
      pointer-events: auto;
    }
    .roi-handle {
      position: absolute;
      width: 8px; height: 8px;
      border: var(--line-light) solid rgba(138, 34, 34, 0.60);
      background: rgba(255, 42, 42, 0.12);
      z-index: 7;
      transition: background 80ms linear, box-shadow 80ms linear;
    }
    .roi-handle:hover {
      background: rgba(255, 42, 42, 0.30);
      box-shadow: 0 0 4px rgba(255, 42, 42, 0.25);
    }
    .roi-handle.tl { top: -4px; left: -4px; cursor: nwse-resize; border-right: none; border-bottom: none; }
    .roi-handle.tr { top: -4px; right: -4px; cursor: nesw-resize; border-left: none; border-bottom: none; }
    .roi-handle.bl { bottom: -4px; left: -4px; cursor: nesw-resize; border-right: none; border-top: none; }
    .roi-handle.br { bottom: -4px; right: -4px; cursor: nwse-resize; border-left: none; border-top: none; }
    .roi-label {
      position: absolute;
      top: -22px; left: 0;
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      color: rgba(138, 34, 34, 0.70);
      white-space: nowrap;
      padding: 1px 6px;
      background: rgba(20, 8, 8, 0.65);
    }

    .roi-toggle {
      position: fixed;
      bottom: 12px; left: 12px;
      z-index: 10;
      padding: 5px 12px;
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-text);
      font-family: var(--mono);
      font-size: 10px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      transition: background 60ms linear;
    }
    .roi-toggle:hover {
      background: var(--red-signal-alpha);
    }
    .roi-toggle.active {
      border-color: var(--red-signal);
      background: rgba(255, 42, 42, 0.14);
      color: var(--red-signal);
    }

    /* ROI shape toggle button */
    .roi-shape-toggle {
      position: fixed;
      bottom: 12px; left: 84px;
      z-index: 10;
      padding: 5px 12px;
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 10px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 1.4px;
      transition: background 60ms linear;
      display: none;
    }
    .roi-shape-toggle.visible { display: block; }
    .roi-shape-toggle:hover {
      background: var(--red-signal-alpha);
      color: var(--hud-text);
    }

    /* ROI size presets — anchored below the ROI box */
    .roi-size-presets {
      position: absolute;
      bottom: -28px; left: 50%;
      transform: translateX(-50%);
      z-index: 8;
      display: none;
      gap: 4px;
      pointer-events: auto;
      white-space: nowrap;
    }
    .roi-box.active .roi-size-presets { display: flex; }
    .roi-size-btn {
      padding: 5px 10px;
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-muted);
      font-family: var(--mono);
      font-size: 10px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 1px;
      transition: background 60ms linear, color 60ms linear;
    }
    .roi-size-btn:hover {
      background: var(--red-signal-alpha);
      color: var(--hud-text);
    }
    .roi-size-btn.active {
      border-color: var(--red-signal);
      background: rgba(255, 42, 42, 0.14);
      color: var(--red-signal);
    }
    .roi-center-btn {
      padding: 5px 10px;
      border: var(--line-light) solid var(--border-iron);
      background: var(--hud-panel-strong);
      color: var(--hud-text-bright);
      font-family: var(--mono);
      font-size: 10px;
      cursor: pointer;
      text-transform: uppercase;
      letter-spacing: 1px;
      transition: background 60ms linear;
      flex: 1;
      text-align: center;
    }
    .roi-center-btn:hover {
      background: var(--red-signal-alpha);
    }

    /* ---- Circle ROI inner elements (hidden in rect mode) ---- */
    .roi-crosshair,
    .roi-inner-ring,
    .roi-inner-fill { display: none; }

    /* Circle ROI mode */
    .roi-box.circle {
      border-radius: 50%;
      box-shadow: 0 0 8px rgba(255, 42, 42, 0.06);
    }
    .roi-box.circle .roi-handle { display: none; }
    .roi-box.circle .roi-handle.tl,
    .roi-box.circle .roi-handle.br {
      display: block;
      border-radius: 50%;
      border: var(--line-light) solid rgba(138, 34, 34, 0.60);
      width: 8px; height: 8px;
    }

    /* Dashed crosshair lines */
    .roi-box.circle .roi-crosshair {
      display: block;
      position: absolute;
      pointer-events: none;
    }
    .roi-box.circle .roi-cross-h {
      top: 50%; left: 8%;
      width: 84%; height: 0;
      border-top: 1px dashed rgba(138, 34, 34, 0.30);
      transform: translateY(-0.5px);
    }
    .roi-box.circle .roi-cross-v {
      display: none;
    }

    /* Inner circle ring — 40% of outer diameter */
    .roi-box.circle .roi-inner-ring {
      display: block;
      position: absolute;
      top: 30%; left: 30%;
      width: 40%; height: 40%;
      border-radius: 50%;
      border: 1px solid rgba(138, 34, 34, 0.25);
      pointer-events: none;
      transition: transform 0.1s ease-out, border-color 0.1s ease-out, box-shadow 0.1s ease-out;
    }

    /* Bottom-half shaded fill of inner circle — textured */
    .roi-box.circle .roi-inner-fill {
      display: block;
      position: absolute;
      top: 50%; left: 30%;
      width: 40%; height: 20%;
      border-radius: 0 0 999px 999px;
      overflow: hidden;
      pointer-events: none;
      background:
        repeating-linear-gradient(
          0deg,
          rgba(255, 42, 42, 0.10) 0px,
          rgba(255, 42, 42, 0.10) 1px,
          transparent 1px,
          transparent 3px
        ),
        rgba(255, 42, 42, 0.06);
      transition: transform 0.1s ease-out, opacity 0.1s ease-out;
    }

    /* Monocle capture animation */
    @keyframes monocle-push {
      0%   { transform: perspective(400px) translateZ(0) scale(1); border-color: rgba(255, 42, 42, 0.30); box-shadow: none; }
      35%  { transform: perspective(400px) translateZ(60px) scale(1.18); border-color: var(--red-signal); box-shadow: 0 0 20px rgba(255, 42, 42, 0.50), inset 0 0 12px rgba(255, 42, 42, 0.15); }
      70%  { transform: perspective(400px) translateZ(30px) scale(1.06); border-color: rgba(255, 42, 42, 0.50); box-shadow: 0 0 10px rgba(255, 42, 42, 0.25); }
      100% { transform: perspective(400px) translateZ(0) scale(1); border-color: rgba(255, 42, 42, 0.30); box-shadow: none; }
    }
    @keyframes monocle-fill-push {
      0%   { transform: perspective(400px) translateZ(0) scale(1); opacity: 1; }
      35%  { transform: perspective(400px) translateZ(60px) scale(1.18); opacity: 0.6; }
      70%  { transform: perspective(400px) translateZ(30px) scale(1.06); opacity: 0.85; }
      100% { transform: perspective(400px) translateZ(0) scale(1); opacity: 1; }
    }
    .roi-box.circle.monocle-capture .roi-inner-ring {
      animation: monocle-push 0.45s ease-out forwards;
    }
    .roi-box.circle.monocle-capture .roi-inner-fill {
      animation: monocle-fill-push 0.45s ease-out forwards;
    }
    .roi-box.circle.monocle-capture .roi-cross-h {
      border-top-color: rgba(255, 42, 42, 0.60);
      transition: border-top-color 0.15s ease-out;
    }

    /* ============ SUBTITLE BAR ============ */
    .subtitle-bar {
      position: fixed;
      bottom: 48px; left: 50%;
      transform: translateX(-50%);
      z-index: 4;
      max-width: 70vw;
      min-width: 280px;
      padding: 6px 16px;
      background: rgba(20, 8, 8, 0.75);
      border: var(--line-light) solid var(--red-iron-alpha);
      color: var(--hud-text-bright);
      font-family: var(--mono);
      font-size: 13px;
      letter-spacing: 0.4px;
      text-align: center;
      line-height: 1.4;
      opacity: 0;
      pointer-events: none;
      transition: opacity 150ms linear;
    }
    .subtitle-bar.visible {
      opacity: 1;
    }
    .subtitle-bar {
      overflow: hidden;
    }
    .subtitle-bar canvas {
      position: absolute;
      top: 0; left: 0;
      width: 100%; height: 100%;
      z-index: 0;
      pointer-events: none;
    }
    .subtitle-bar #subtitleText {
      position: relative;
      z-index: 1;
    }
    .subtitle-bar .interim {
      color: var(--hud-muted);
    }
  </style>
</head>
<body>
  <video id="sceneVideo" autoplay playsinline muted></video>
  <div class="veil"></div>
  <div class="loading-screen" id="loadingScreen">
    <div class="loading-panel" id="loadingPanel">
      <div class="loading-line">
        <div class="loading-title">Atlas Link</div>
        <div class="loading-sep"></div>
        <div class="loading-state" id="loadingState">Starting</div>
        <div class="loading-copy" id="loadingCopy">Preparing signaling, transport, and live video feed.</div>
      </div>
      <div class="loading-bar"><div class="loading-fill" id="loadingFill"></div></div>
      <div class="loading-actions" id="loadingActions">
        <button class="loading-action" id="loadingCancelBtn" type="button">Stay Here</button>
        <button class="loading-action primary" id="loadingConfirmBtn" type="button">Swap View</button>
      </div>
    </div>
  </div>

  <div class="flight-strip" id="flightStrip">
    <div class="flight-group">
      <div>
        <div class="flight-title">ATLAS</div>
        <div class="flight-subtitle">Tactical vision system</div>
      </div>
    </div>
    <div class="flight-group" id="flightMetrics"></div>
    <button class="flight-dismiss" id="flightDismiss" type="button">x</button>
  </div>
  <button class="flight-strip-restore" id="flightRestore" type="button">HUD</button>

  <div class="sidebar" id="sidebar">
    <div class="sidebar-drag" id="sidebarDrag"></div>
    <div class="sidebar-tabs">
      <button class="sidebar-tab active" id="tabPreviews" type="button">Previews</button>
      <button class="sidebar-tab" id="tabFocus" type="button">Focus</button>
      <button class="sidebar-tab" id="tabEvents" type="button">Events</button>
      <button class="sidebar-dismiss" id="sidebarDismiss" type="button">x</button>
    </div>
    <div class="sidebar-pane active" id="panePreviews">
      <div class="pane-dismiss-row">
        <button class="pane-dismiss" type="button" aria-label="Dismiss previews panel">x</button>
      </div>
      <div class="tile-rail" id="tileRail"></div>
    </div>
    <div class="sidebar-pane" id="paneFocus">
      <div class="pane-dismiss-row">
        <button class="pane-dismiss" type="button" aria-label="Dismiss focus panel">x</button>
      </div>
      <div class="focus-pane-inner" id="focusPanel">
        <div class="focus-empty" id="focusEmpty">Select a preview card or timeline entry to inspect it.</div>
        <div class="focus-header" id="focusHeader" style="display:none">
          <div>
            <h2 class="focus-title" id="focusTitle"></h2>
            <div class="focus-copy" id="focusCopy"></div>
          </div>
          <button class="focus-clear" id="focusClear" type="button">Clear</button>
        </div>
        <div class="focus-controls" id="focusControls" style="display:none">
          <button class="focus-control-btn" id="focusZoomOut" type="button">[−] Zoom Out</button>
          <span class="focus-zoom-level" id="focusZoomLevel">100%</span>
          <button class="focus-control-btn" id="focusZoomIn" type="button">[+] Zoom In</button>
          <button class="focus-control-btn" id="focusRefresh" type="button">[↻] Refresh</button>
          <button class="focus-control-btn" id="focusRuntimeCatalogBtn" type="button">[CV] Catalog</button>
        </div>
        <div class="focus-visual" id="focusVisual" style="display:none">
          <div class="focus-visual-inner" id="focusVisualInner">
            <img class="focus-image" id="focusImage" alt="Focused track crop" />
            <img class="focus-silhouette" id="focusSilhouette" alt="Focused track silhouette" />
            <canvas class="focus-overlay-canvas" id="focusOverlayCanvas"></canvas>
          </div>
        </div>
        <div class="focus-stats" id="focusStats"></div>
        <div class="focus-scan" id="focusScan" style="display:none"></div>
        <div class="focus-runtime" id="focusRuntime" style="display:none"></div>
      </div>
    </div>
    <div class="sidebar-pane" id="paneEvents">
      <div class="pane-dismiss-row">
        <button class="pane-dismiss" type="button" aria-label="Dismiss events panel">x</button>
      </div>
      <div class="event-list-inner" id="eventList"></div>
    </div>
  </div>
  <button class="sidebar-restore" id="sidebarRestore" type="button">Data</button>

  <div class="roi-box" id="roiBox">
    <div class="roi-label">ROI</div>
    <div class="roi-crosshair roi-cross-h"></div>
    <div class="roi-crosshair roi-cross-v"></div>
    <div class="roi-inner-ring"></div>
    <div class="roi-inner-fill"></div>
    <div class="roi-handle tl" data-handle="tl"></div>
    <div class="roi-handle tr" data-handle="tr"></div>
    <div class="roi-handle bl" data-handle="bl"></div>
    <div class="roi-handle br" data-handle="br"></div>
    <div class="roi-size-presets" id="roiSizePresets">
      <button class="roi-size-btn" data-scale="0.5">.5x</button>
      <button class="roi-size-btn active" data-scale="1">1x</button>
      <button class="roi-center-btn" id="roiCenterBtn" type="button">[C] Center</button>
      <button class="roi-size-btn" data-scale="2">2x</button>
      <button class="roi-size-btn" data-scale="3">3x</button>
    </div>
  </div>
  <button class="roi-toggle" id="roiToggle" type="button">[R] ROI</button>
  <button class="roi-shape-toggle" id="roiShapeToggle" type="button">Rect</button>
  <button class="timeline-toggle" id="timelineToggle" type="button">[T] Timeline</button>

  <div class="subtitle-bar" id="subtitleBar"><canvas id="audioVizCanvas"></canvas><span id="subtitleText"></span></div>

  <div class="timeline-bar" id="timelineBar">
    <div class="timeline-header">
      <div>
        <span class="timeline-title">Detection History</span>
        <span class="timeline-count" id="timelineCount">0</span>
      </div>
      <div class="timeline-actions">
        <button class="timeline-btn danger" id="timelineClear" type="button">Clear All</button>
        <button class="timeline-btn" id="timelineClose" type="button">Close</button>
      </div>
    </div>
    <div class="timeline-scroll" id="timelineScroll"></div>
  </div>

  <div class="status-toast" id="statusToast"></div>


  <script>
    const video = document.getElementById("sceneVideo");
    const loadingScreen = document.getElementById("loadingScreen");
    const loadingPanel = document.getElementById("loadingPanel");
    const loadingState = document.getElementById("loadingState");
    const loadingCopy = document.getElementById("loadingCopy");
    const loadingFill = document.getElementById("loadingFill");
    const loadingActions = document.getElementById("loadingActions");
    const loadingCancelBtn = document.getElementById("loadingCancelBtn");
    const loadingConfirmBtn = document.getElementById("loadingConfirmBtn");
    const tileRail = document.getElementById("tileRail");
    const eventList = document.getElementById("eventList");
    const focusPanel = document.getElementById("focusPanel");
    const focusTitle = document.getElementById("focusTitle");
    const focusCopy = document.getElementById("focusCopy");
    const focusImage = document.getElementById("focusImage");
    const focusSilhouette = document.getElementById("focusSilhouette");
    const focusStats = document.getElementById("focusStats");
    const focusHeader = document.getElementById("focusHeader");
    const focusVisual = document.getElementById("focusVisual");
    const focusEmpty = document.getElementById("focusEmpty");
    const focusScan = document.getElementById("focusScan");
    const focusOverlayCanvas = document.getElementById("focusOverlayCanvas");
    const focusControls = document.getElementById("focusControls");
    const focusVisualInner = document.getElementById("focusVisualInner");
    const focusZoomOut = document.getElementById("focusZoomOut");
    const focusZoomIn = document.getElementById("focusZoomIn");
    const focusZoomLevel = document.getElementById("focusZoomLevel");
    const focusRefresh = document.getElementById("focusRefresh");
    const focusRuntimeCatalogBtn = document.getElementById("focusRuntimeCatalogBtn");
    const focusRuntime = document.getElementById("focusRuntime");
    const flightMetrics = document.getElementById("flightMetrics");
    const statusToast = document.getElementById("statusToast");
    const focusClear = document.getElementById("focusClear");
    const flightStrip = document.getElementById("flightStrip");
    const flightDismiss = document.getElementById("flightDismiss");
    const flightRestore = document.getElementById("flightRestore");
    const sidebarEl = document.getElementById("sidebar");
    const sidebarDrag = document.getElementById("sidebarDrag");
    const sidebarDismiss = document.getElementById("sidebarDismiss");
    const sidebarRestore = document.getElementById("sidebarRestore");
    const roiBox = document.getElementById("roiBox");
    const roiToggle = document.getElementById("roiToggle");
    const roiShapeToggle = document.getElementById("roiShapeToggle");
    const roiSizePresets = document.getElementById("roiSizePresets");
    const subtitleBar = document.getElementById("subtitleBar");
    const subtitleText = document.getElementById("subtitleText");
    const audioVizCanvas = document.getElementById("audioVizCanvas");
    const timelineBar = document.getElementById("timelineBar");
    const timelineToggle = document.getElementById("timelineToggle");
    const timelineScroll = document.getElementById("timelineScroll");
    const timelineCount = document.getElementById("timelineCount");
    const timelineClear = document.getElementById("timelineClear");
    const timelineClose = document.getElementById("timelineClose");
    const tabPreviews = document.getElementById("tabPreviews");
    const tabFocus = document.getElementById("tabFocus");
    const tabEvents = document.getElementById("tabEvents");
    const panePreviews = document.getElementById("panePreviews");
    const paneFocus = document.getElementById("paneFocus");
    const paneEvents = document.getElementById("paneEvents");
    const paneDismissButtons = document.querySelectorAll(".pane-dismiss");

    let pc = null;
    let dc = null;
    let toastTimer = null;
    let lastHudState = null;
    let hasReceivedLivePayload = false;
    let loadingDismissTimer = null;
    let loadingBlinkTimers = [];
    let sourceSwapPending = null;
    let loadingFallbackTimer = null;
    const loadingProgress = {
      signal: 0.04,
      data: 0,
      video: 0,
      live: 0,
    };

    function clamp01(value) {
      return Math.max(0, Math.min(1, value));
    }

    function updateLoadingUi() {
      const avg =
        (loadingProgress.signal + loadingProgress.data + loadingProgress.video + loadingProgress.live) / 4;
      loadingFill.style.width = Math.round(clamp01(avg) * 100) + "%";
    }

    function setLoadingActionsVisible(visible) {
      loadingActions.classList.toggle("visible", visible);
      loadingConfirmBtn.disabled = !visible;
      loadingCancelBtn.disabled = !visible;
    }

    function clearLoadingBlink() {
      loadingBlinkTimers.forEach((timerId) => window.clearTimeout(timerId));
      loadingBlinkTimers = [];
      loadingPanel.classList.remove("blink-success", "blink-failure");
    }
    function clearLoadingFallbackTimer() {
      if (loadingFallbackTimer) {
        window.clearTimeout(loadingFallbackTimer);
        loadingFallbackTimer = null;
      }
    }

    function blinkLoading(kind, count, onComplete = null) {
      clearLoadingBlink();
      if (count <= 0) {
        if (onComplete) onComplete();
        return;
      }
      const className = kind === "failure" ? "blink-failure" : "blink-success";
      const stepMs = 110;
      for (let i = 0; i < count * 2; i++) {
        const timerId = window.setTimeout(() => {
          loadingPanel.classList.toggle(className, i % 2 === 0);
        }, i * stepMs);
        loadingBlinkTimers.push(timerId);
      }
      loadingBlinkTimers.push(
        window.setTimeout(() => {
          loadingPanel.classList.remove(className);
          if (onComplete) onComplete();
        }, count * 2 * stepMs)
      );
    }

    function setLoadingExact(nextValues) {
      Object.entries(nextValues).forEach(([key, value]) => {
        if (key in loadingProgress) {
          loadingProgress[key] = clamp01(value);
        }
      });
      updateLoadingUi();
    }

    function setLoadingProgress(key, value) {
      if (!(key in loadingProgress)) return;
      loadingProgress[key] = Math.max(loadingProgress[key], clamp01(value));
      updateLoadingUi();
    }

    function showLoading(stateText, copyText) {
      window.clearTimeout(loadingDismissTimer);
      clearLoadingBlink();
      loadingState.textContent = stateText;
      loadingCopy.textContent = copyText;
      loadingScreen.classList.remove("hidden");
    }

    function hideLoading(blinkCount = 2) {
      window.clearTimeout(loadingDismissTimer);
      setLoadingActionsVisible(false);
      blinkLoading("success", blinkCount, () => {
        loadingDismissTimer = window.setTimeout(() => {
          loadingScreen.classList.add("hidden");
        }, 120);
      });
    }

    function resetLoading(reasonText) {
      clearLoadingFallbackTimer();
      setLoadingExact({
        signal: 0.04,
        data: 0,
        video: 0,
        live: 0,
      });
      hasReceivedLivePayload = false;
      sourceSwapPending = null;
      setLoadingActionsVisible(false);
      showLoading("Linking", reasonText);
    }

    updateLoadingUi();

    function sourceKind(sourceValue) {
      const text = String(sourceValue || "").toLowerCase();
      if (text.startsWith("camera:")) return "camera";
      if (text.startsWith("video:")) return "video";
      return null;
    }

    function sourceLabel(sourceValue) {
      const kind = sourceKind(sourceValue);
      if (kind === "camera") return "camera feed";
      if (kind === "video") return "video feed";
      return "source feed";
    }

    function waitForRenderedFrame(callback) {
      if (typeof video.requestVideoFrameCallback === "function") {
        video.requestVideoFrameCallback(() => callback());
        return;
      }
      requestAnimationFrame(() => callback());
    }

    function completeSourceSwap(labelText) {
      setLoadingExact({ signal: 1, data: 1, video: 1, live: 1 });
      loadingState.textContent = "Source Ready";
      loadingCopy.textContent = `${labelText} is active and rendering live frames.`;
      const completedLabel = labelText;
      sourceSwapPending = null;
      setLoadingActionsVisible(false);
      waitForRenderedFrame(() => {
        hideLoading();
        showStatus(`Source ready: ${completedLabel}`);
      });
    }

    function beginSourceSwap(targetSource, labelText) {
      sourceSwapPending = {
        target: targetSource,
        label: labelText,
        prepared: false,
        committing: false,
      };
      setLoadingExact({ signal: 1, data: 1, video: 0.08, live: 0.02 });
      setLoadingActionsVisible(false);
      showLoading("Switching Source", `Preparing ${labelText}. Holding the current view until the new feed is ready.`);
    }

    function requestSourceSwitch(requested = null) {
      if (sourceSwapPending) return;
      const currentKind = sourceKind(lastHudState?.source);
      const target = requested || (currentKind === "camera" ? "video" : "camera");
      const labelText = target === "camera" ? "camera feed" : "video feed";
      beginSourceSwap(target, labelText);
      if (requested) {
        sendMessage({ type: "switch_source", source: requested });
      } else {
        sendMessage({ type: "switch_source" });
      }
    }

    function confirmSourceSwap() {
      if (!sourceSwapPending || !sourceSwapPending.prepared || sourceSwapPending.committing) return;
      sourceSwapPending.committing = true;
      setLoadingActionsVisible(false);
      loadingState.textContent = "Swapping";
      loadingCopy.textContent = `Switching to ${sourceSwapPending.label}. Keeping the current view visible until the new feed lands.`;
      setLoadingExact({ signal: 1, data: 1, video: 0.98, live: 0.82 });
      sendMessage({ type: "confirm_source_switch", source: sourceSwapPending.target });
    }

    function cancelSourceSwap() {
      if (!sourceSwapPending) return;
      sendMessage({ type: "cancel_source_switch" });
      sourceSwapPending = null;
      hideLoading(0);
      showStatus("Source swap cancelled");
    }

    loadingConfirmBtn.addEventListener("click", confirmSourceSwap);
    loadingCancelBtn.addEventListener("click", cancelSourceSwap);

    /* ---- ROI state ---- */
    let roiActive = false;
    let roiShape = "rect"; // "rect" or "circle"
    let roiScale = 1; // current size multiplier (1, 2, or 3)
    // normalised 0-1 coords relative to the video
    let roiNorm = { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 };
    let roiDrag = null; // { mode: "move"|"tl"|"tr"|"bl"|"br", startX, startY, origNorm }
    let timelineOpen = false;
    let historyEntries = [];
    let historyFocusLock = false; // true while viewing a timeline card in the focus panel
    let hasFocusContent = false; // whether the focus pane has something to show
    let focusContext = null; // { source: "roi"|"history", captured_at?, entry_id? }
    let scenarioCatalog = [];
    let scenarioCatalogOpen = false;
    let selectedScenarioRunId = null;
    const runtimeResultsByRunId = new Map();
    let runtimeEvents = [];
    const runtimeLogsByRunId = new Map(); // run_id -> [{ts, kind, tag, text, stage}]
    let detectionEvents = [];
    const MAX_RUNTIME_EVENTS = 30;
    const MAX_RUNTIME_LOGS_PER_RUN = 160;

    /* ---- Scan overlay state ---- */
    let showBboxOverlay = false;
    let showHeatmap = false;
    let heatmapOpacity = 0.55;
    let lastScanResults = null; // cached for redraw on toggle

    // Category-to-color mapping (shared by heat maps, tiles, and scan list)
    const HEAT_CATS = {
      "person": "human", "potted plant": "plant",
      "dog": "animal", "cat": "animal", "bird": "animal", "horse": "animal",
      "sheep": "animal", "cow": "animal", "elephant": "animal", "bear": "animal",
      "zebra": "animal", "giraffe": "animal",
      "laptop": "tech", "cell phone": "tech", "keyboard": "tech", "mouse": "tech",
      "remote": "tech", "tv": "tech", "microwave": "tech", "oven": "tech", "toaster": "tech",
    };
    const HEAT_CAT_CSS = {
      human: "rgb(255,160,60)", plant: "rgb(60,210,60)", animal: "rgb(0,200,190)",
      inorganic: "rgb(60,140,255)", tech: "rgb(170,80,255)",
    };
    function catColor(label) { return HEAT_CAT_CSS[HEAT_CATS[label] || "inorganic"] || HEAT_CAT_CSS.inorganic; }
    let focusZoom = 1.0; // zoom level for ROI capture image
    let roiCapturePayload = null; // store last capture for refresh

    /* ---- AI Analysis state ---- */
    let roiAiComposerOpen = false;
    let roiAiBusy = false;
    let roiAiResultText = "";
    let roiAiErrorText = "";
    let roiAiProvider = "auto";
    let roiAiDraftPrompt = "";
    let roiAiCaptureAt = null;

    /* ---- Sidebar tab switching ---- */
    const allTabs = [tabPreviews, tabFocus, tabEvents];
    const allPanes = [panePreviews, paneFocus, paneEvents];
    const tabMap = { previews: 0, focus: 1, events: 2 };

    function switchTab(tabId) {
      const idx = tabMap[tabId] ?? 0;
      allTabs.forEach((t) => t.classList.remove("active"));
      allPanes.forEach((p) => p.classList.remove("active"));
      allTabs[idx].classList.add("active");
      allPanes[idx].classList.add("active");
    }

    tabPreviews.addEventListener("click", () => switchTab("previews"));
    tabFocus.addEventListener("click", () => switchTab("focus"));
    tabEvents.addEventListener("click", () => switchTab("events"));
    paneDismissButtons.forEach((button) => {
      button.addEventListener("click", hideSidebar);
    });

    /* ---- Sidebar horizontal drag ---- */
    let sidebarX = window.innerWidth * 0.8 - sidebarEl.offsetWidth;
    sidebarEl.style.left = sidebarX + "px";
    let sbDrag = null;
    let sidebarHidden = false;

    function hideSidebar() {
      sidebarHidden = true;
      sidebarEl.classList.add("dismissed");
      sidebarRestore.classList.add("visible");
    }

    function showSidebar() {
      sidebarHidden = false;
      sidebarEl.classList.remove("dismissed");
      sidebarRestore.classList.remove("visible");
    }

    function clampSidebarX(x) {
      const w = sidebarEl.offsetWidth;
      return Math.max(0, Math.min(window.innerWidth - w, x));
    }

    sidebarDrag.addEventListener("mousedown", (e) => {
      if (sidebarHidden) return;
      e.preventDefault();
      sidebarDrag.classList.add("dragging");
      sbDrag = { startX: e.clientX, origLeft: sidebarX };
    });

    window.addEventListener("mousemove", (e) => {
      if (!sbDrag) return;
      const dx = e.clientX - sbDrag.startX;
      sidebarX = clampSidebarX(sbDrag.origLeft + dx);
      sidebarEl.style.left = sidebarX + "px";
    });

    window.addEventListener("mouseup", () => {
      if (sbDrag) {
        sbDrag = null;
        sidebarDrag.classList.remove("dragging");
      }
    });

    window.addEventListener("resize", () => {
      sidebarX = clampSidebarX(sidebarX);
      sidebarEl.style.left = sidebarX + "px";
    });

    sidebarDismiss.addEventListener("click", hideSidebar);
    sidebarRestore.addEventListener("click", showSidebar);

    function dataUriFromBase64(type, value) {
      if (!value) return "";
      return `data:${type};base64,${value}`;
    }
    function esc(str) {
      const d = document.createElement("div");
      d.textContent = String(str);
      return d.innerHTML;
    }
    function escMultiline(str) {
      return esc(str).replace(/\n/g, "<br>");
    }

    let pendingMessages = [];
    const MAX_PENDING = 50;
    function sendMessage(payload) {
      if (dc && dc.readyState === "open") {
        dc.send(JSON.stringify(payload));
      } else if (pendingMessages.length < MAX_PENDING) {
        pendingMessages.push(payload);
      }
    }
    function flushPendingMessages() {
      while (pendingMessages.length > 0 && dc && dc.readyState === "open") {
        dc.send(JSON.stringify(pendingMessages.shift()));
      }
    }

    /* ---- ROI helpers ---- */
    function videoRect() {
      return video.getBoundingClientRect();
    }

    function roiToScreen() {
      const vr = videoRect();
      return {
        left: vr.left + roiNorm.x1 * vr.width,
        top: vr.top + roiNorm.y1 * vr.height,
        width: (roiNorm.x2 - roiNorm.x1) * vr.width,
        height: (roiNorm.y2 - roiNorm.y1) * vr.height,
      };
    }

    function renderRoi() {
      if (!roiActive) {
        roiBox.classList.remove("active", "circle");
        roiToggle.classList.remove("active");
        roiShapeToggle.classList.remove("visible");
        return;
      }
      roiBox.classList.add("active");
      roiBox.classList.toggle("circle", roiShape === "circle");
      roiToggle.classList.add("active");
      roiShapeToggle.classList.add("visible");
      roiShapeToggle.textContent = roiShape === "rect" ? "Rect" : "Circle";

      const s = roiToScreen();
      if (roiShape === "circle") {
        // Force square aspect ratio for circle — use smaller dimension
        const side = Math.min(s.width, s.height);
        const cx = s.left + s.width / 2;
        const cy = s.top + s.height / 2;
        roiBox.style.left = (cx - side / 2) + "px";
        roiBox.style.top = (cy - side / 2) + "px";
        roiBox.style.width = side + "px";
        roiBox.style.height = side + "px";
      } else {
        roiBox.style.left = s.left + "px";
        roiBox.style.top = s.top + "px";
        roiBox.style.width = s.width + "px";
        roiBox.style.height = s.height + "px";
      }
    }

    function sendRoi() {
      sendMessage({
        type: "set_roi",
        shape: roiShape,
        x1: roiNorm.x1,
        y1: roiNorm.y1,
        x2: roiNorm.x2,
        y2: roiNorm.y2,
      });
    }

    function toggleRoi() {
      roiActive = !roiActive;
      renderRoi();
      if (roiActive) {
        sendRoi();
      } else {
        sendMessage({ type: "clear_roi" });
      }
    }

    function cycleRoiShape() {
      roiShape = roiShape === "rect" ? "circle" : "rect";
      renderRoi();
      if (roiActive) sendRoi();
    }

    roiToggle.addEventListener("click", toggleRoi);
    roiShapeToggle.addEventListener("click", cycleRoiShape);

    // ROI size presets: 1x, 2x, 3x
    const ROI_BASE_SIZE = 0.16; // base half-size at 1x (total width/height = 0.32 of video)
    function applyRoiScale(scale) {
      roiScale = scale;
      const cx = (roiNorm.x1 + roiNorm.x2) / 2;
      const cy = (roiNorm.y1 + roiNorm.y2) / 2;
      const half = ROI_BASE_SIZE * scale;
      roiNorm.x1 = Math.max(0, cx - half);
      roiNorm.y1 = Math.max(0, cy - half);
      roiNorm.x2 = Math.min(1, cx + half);
      roiNorm.y2 = Math.min(1, cy + half);
      // Re-center if clamped against edges
      const w = roiNorm.x2 - roiNorm.x1;
      const h = roiNorm.y2 - roiNorm.y1;
      if (roiNorm.x1 === 0) roiNorm.x2 = w;
      if (roiNorm.y1 === 0) roiNorm.y2 = h;
      if (roiNorm.x2 === 1) roiNorm.x1 = 1 - w;
      if (roiNorm.y2 === 1) roiNorm.y1 = 1 - h;
      // Update active button state
      roiSizePresets.querySelectorAll(".roi-size-btn").forEach((btn) => {
        btn.classList.toggle("active", Number(btn.dataset.scale) === scale);
      });
      renderRoi();
      if (roiActive) sendRoi();
    }
    roiSizePresets.addEventListener("mousedown", (e) => e.stopPropagation());
    roiSizePresets.addEventListener("click", (e) => {
      const sizeBtn = e.target.closest(".roi-size-btn");
      if (sizeBtn) {
        applyRoiScale(Number(sizeBtn.dataset.scale));
        return;
      }
      const centerBtn = e.target.closest(".roi-center-btn");
      if (centerBtn) {
        // Recenter to middle of screen (0.5, 0.5)
        const cx = 0.5;
        const cy = 0.5;
        const w = roiNorm.x2 - roiNorm.x1;
        const h = roiNorm.y2 - roiNorm.y1;
        roiNorm.x1 = Math.max(0, Math.min(1 - w, cx - w / 2));
        roiNorm.y1 = Math.max(0, Math.min(1 - h, cy - h / 2));
        roiNorm.x2 = roiNorm.x1 + w;
        roiNorm.y2 = roiNorm.y1 + h;
        renderRoi();
        if (roiActive) sendRoi();
      }
    });

    // Double-click inside ROI to screenshot the region into Focus
    let roiClickTimer = null;
    roiBox.addEventListener("dblclick", (e) => {
      if (!roiActive) return;
      e.preventDefault();
      e.stopPropagation();
      // Cancel any drag that started from the first click
      roiDrag = null;
      clearTimeout(roiClickTimer);
      roiClickTimer = null;
      // Trigger monocle capture animation on circle mode
      if (roiBox.classList.contains("circle")) {
        roiBox.classList.remove("monocle-capture");
        void roiBox.offsetWidth; // force reflow to restart animation
        roiBox.classList.add("monocle-capture");
        roiBox.addEventListener("animationend", () => {
          roiBox.classList.remove("monocle-capture");
        }, { once: true });
      }
      sendMessage({ type: "capture_roi" });
    });

    // Make ROI draggable / resizable — delay to let dblclick win
    roiBox.addEventListener("mousedown", (e) => {
      if (!roiActive) return;
      const handle = e.target.dataset?.handle;
      const mode = handle || "move";
      const pending = {
        mode,
        startX: e.clientX,
        startY: e.clientY,
        origNorm: { ...roiNorm },
      };
      e.preventDefault();
      // Handles start immediately; move waits briefly so dblclick can cancel
      if (handle) {
        roiDrag = pending;
      } else {
        clearTimeout(roiClickTimer);
        roiClickTimer = setTimeout(() => {
          roiDrag = pending;
          roiClickTimer = null;
        }, 200);
      }
    });

    window.addEventListener("mousemove", (e) => {
      if (!roiDrag) return;
      const vr = videoRect();
      const dx = (e.clientX - roiDrag.startX) / vr.width;
      const dy = (e.clientY - roiDrag.startY) / vr.height;
      const o = roiDrag.origNorm;
      const clamp01 = (v) => Math.max(0, Math.min(1, v));

      if (roiDrag.mode === "move") {
        const w = o.x2 - o.x1;
        const h = o.y2 - o.y1;
        let nx1 = clamp01(o.x1 + dx);
        let ny1 = clamp01(o.y1 + dy);
        if (nx1 + w > 1) nx1 = 1 - w;
        if (ny1 + h > 1) ny1 = 1 - h;
        roiNorm.x1 = Math.max(0, nx1);
        roiNorm.y1 = Math.max(0, ny1);
        roiNorm.x2 = roiNorm.x1 + w;
        roiNorm.y2 = roiNorm.y1 + h;
      } else {
        const updated = { ...o };
        if (roiDrag.mode.includes("l")) updated.x1 = clamp01(o.x1 + dx);
        if (roiDrag.mode.includes("r")) updated.x2 = clamp01(o.x2 + dx);
        if (roiDrag.mode.includes("t")) updated.y1 = clamp01(o.y1 + dy);
        if (roiDrag.mode.includes("b")) updated.y2 = clamp01(o.y2 + dy);
        // enforce minimum size
        if (updated.x2 - updated.x1 < 0.05) { updated.x1 = o.x1; updated.x2 = o.x2; }
        if (updated.y2 - updated.y1 < 0.05) { updated.y1 = o.y1; updated.y2 = o.y2; }
        roiNorm.x1 = updated.x1;
        roiNorm.y1 = updated.y1;
        roiNorm.x2 = updated.x2;
        roiNorm.y2 = updated.y2;
      }
      renderRoi();
    });

    window.addEventListener("mouseup", () => {
      if (roiDrag) {
        roiDrag = null;
        if (roiActive) sendRoi();
      }
    });

    // Keep ROI box in sync when window resizes
    window.addEventListener("resize", () => { if (roiActive) renderRoi(); });

    /* ---- Timeline helpers ---- */
    function toggleTimeline() {
      timelineOpen = !timelineOpen;
      timelineBar.classList.toggle("open", timelineOpen);
      timelineToggle.classList.toggle("active", timelineOpen);
    }

    function formatTime(ts) {
      const d = new Date(ts * 1000);
      return d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    }

    const HISTORY_TTL = 40;
    const TTL_CIRC = 2 * Math.PI * 7; // r=7 => ~43.98

    function buildTimelineCard(entry) {
      const card = document.createElement("button");
      card.className = "timeline-card";
      card.type = "button";
      card.dataset.entryId = String(entry.entry_id);
      card.innerHTML = `
        <img class="timeline-card-img" src="${dataUriFromBase64("image/jpeg", entry.image)}" alt="${esc(entry.label)}" />
        <svg class="timeline-ttl" viewBox="0 0 20 20">
          <circle class="timeline-ttl-bg" cx="10" cy="10" r="7" />
          <circle class="timeline-ttl-fg" cx="10" cy="10" r="7"
            stroke-dasharray="${TTL_CIRC}"
            stroke-dashoffset="0"
            transform="rotate(-90 10 10)"
            data-captured="${entry.captured_at}" />
        </svg>
        <div class="timeline-card-body">
          <div class="timeline-card-label">${esc(entry.label)}</div>
          <div class="timeline-card-meta">
            <span>${Math.round(entry.confidence * 100)}%</span>
            <span>${formatTime(entry.captured_at)}</span>
          </div>
        </div>
        <button class="timeline-card-del" data-eid="${entry.entry_id}" title="Delete">&times;</button>
      `;
      card.addEventListener("click", (e) => {
        if (e.target.classList.contains("timeline-card-del")) return;
        historyFocusLock = true;
        renderFocus({
          active: true,
          track_id: entry.track_id,
          label: entry.label,
          confidence: entry.confidence,
          motion_score: 0,
          age_seconds: entry.age_seconds,
          event_tag: entry.event_tag,
          image: entry.image,
          silhouette: null,
          source: "history",
          entry_id: entry.entry_id,
          captured_at: entry.captured_at,
        });
      });
      card.querySelector(".timeline-card-del").addEventListener("click", (e) => {
        e.stopPropagation();
        sendMessage({ type: "delete_history_entry", entry_id: entry.entry_id });
      });
      return card;
    }

    let timelineUserScrolling = false;
    let timelineScrollTimer = null;
    timelineScroll.addEventListener("scroll", () => {
      timelineUserScrolling = true;
      clearTimeout(timelineScrollTimer);
      timelineScrollTimer = setTimeout(() => { timelineUserScrolling = false; }, 1500);
    }, { passive: true });

    function applyHistoryMessage(msg) {
      if (!msg) return;
      if (msg.mode === "full" || Array.isArray(msg.entries)) {
        renderTimeline(msg.entries || []);
        return;
      }
      if (msg.mode === "delta") {
        const removed = new Set((msg.removed_ids || []).map((id) => Number(id)));
        const added = msg.added || [];
        let next = historyEntries.filter((e) => !removed.has(Number(e.entry_id)));
        if (added.length) {
          const existingIds = new Set(next.map((e) => Number(e.entry_id)));
          for (const entry of added) {
            if (!existingIds.has(Number(entry.entry_id))) next.push(entry);
          }
          next.sort((a, b) => Number(a.entry_id) - Number(b.entry_id));
        }
        if (removed.size === 0 && added.length === 0) return;
        renderTimeline(next);
      }
    }

    function renderTimeline(entries) {
      historyEntries = entries || [];
      timelineCount.textContent = String(historyEntries.length);

      if (historyEntries.length === 0) {
        timelineScroll.innerHTML = `<div class="timeline-empty">No detections captured yet. Cards appear here when tracks leave the scene.</div>`;
        return;
      }

      // Build a set of incoming IDs (newest-first order)
      const incoming = [...historyEntries].reverse();
      const incomingIds = new Set(incoming.map((e) => String(e.entry_id)));

      // Remove cards that no longer exist in the incoming list
      const existingCards = timelineScroll.querySelectorAll(".timeline-card");
      const existingIds = new Set();
      existingCards.forEach((card) => {
        const eid = card.dataset.entryId;
        if (!incomingIds.has(eid)) {
          card.remove();
        } else {
          existingIds.add(eid);
        }
      });

      // Clear empty-state placeholder if present
      const placeholder = timelineScroll.querySelector(".timeline-empty");
      if (placeholder) placeholder.remove();

      // Snapshot scroll state before DOM mutations
      const scrollBefore = timelineScroll.scrollLeft;
      const widthBefore = timelineScroll.scrollWidth;

      // Prepend any new entries that don't already have a DOM node.
      const firstExisting = timelineScroll.querySelector(".timeline-card");
      for (let i = incoming.length - 1; i >= 0; i--) {
        const entry = incoming[i];
        if (existingIds.has(String(entry.entry_id))) continue;
        const card = buildTimelineCard(entry);
        if (firstExisting) {
          timelineScroll.insertBefore(card, firstExisting);
        } else {
          timelineScroll.appendChild(card);
        }
      }

      // Re-order DOM to match newest-first without tearing (only move if needed)
      let prevNode = null;
      for (const entry of incoming) {
        const eid = String(entry.entry_id);
        const node = timelineScroll.querySelector(`.timeline-card[data-entry-id="${eid}"]`);
        if (!node) continue;
        if (prevNode === null) {
          if (node !== timelineScroll.firstElementChild) {
            timelineScroll.insertBefore(node, timelineScroll.firstElementChild);
          }
        } else if (node.previousElementSibling !== prevNode) {
          prevNode.after(node);
        }
        prevNode = node;
      }

      // Compensate scroll position if user was scrolling — keep their ViewPort stable
      if (timelineUserScrolling && scrollBefore > 0) {
        const widthAdded = timelineScroll.scrollWidth - widthBefore;
        if (widthAdded > 0) {
          timelineScroll.scrollLeft = scrollBefore + widthAdded;
        }
      }

      tickTtlRings();
    }

    function tickTtlRings() {
      const nowSec = Date.now() / 1000;
      document.querySelectorAll(".timeline-ttl-fg").forEach((ring) => {
        const captured = parseFloat(ring.dataset.captured);
        const elapsed = nowSec - captured;
        const remaining = Math.max(0, 1 - elapsed / HISTORY_TTL);
        ring.setAttribute("stroke-dashoffset", String(TTL_CIRC * (1 - remaining)));
        ring.classList.remove("warn", "danger");
        if (remaining <= 0.2) ring.classList.add("danger");
        else if (remaining <= 0.45) ring.classList.add("warn");
      });
    }
    const ttlIntervalId = setInterval(tickTtlRings, 500);

    timelineToggle.addEventListener("click", toggleTimeline);
    timelineClose.addEventListener("click", () => {
      timelineOpen = false;
      timelineBar.classList.remove("open");
      timelineToggle.classList.remove("active");
    });
    timelineClear.addEventListener("click", () => {
      sendMessage({ type: "clear_history" });
    });

    function showStatus(message, level = "info") {
      if (!message) return;
      statusToast.textContent = message;
      statusToast.classList.add("visible");
      statusToast.style.borderColor =
        level === "error" ? "rgba(255, 26, 26, 0.65)" :
        level === "warn" ? "rgba(232, 88, 48, 0.50)" :
        "rgba(138, 34, 34, 0.50)";
      window.clearTimeout(toastTimer);
      toastTimer = window.setTimeout(() => {
        statusToast.classList.remove("visible");
      }, 2200);
    }

    function renderHud(state) {
      lastHudState = state;
      if (sourceSwapPending && sourceKind(state.source) === sourceSwapPending.target) {
        completeSourceSwap(sourceSwapPending.label);
      }
      const entries = [
        { key: "MODE", value: state.mode || "PASSIVE" },
        { key: "SOURCE", value: state.source || "--", buttonish: true, action: "switch" },
        { key: "MODEL", value: state.model || "--" },
        { key: "TRACKS", value: String(state.track_count ?? 0) },
        { key: "FPS", value: `${state.fps ?? 0}` },
        { key: "LAT", value: `${state.latency_ms ?? 0} ms` },
        { key: "FOCUS", value: state.active_focus == null ? "none" : `#${state.active_focus}` },
      ];

      if (!flightMetrics._chipCache) flightMetrics._chipCache = new Map();
      const cache = flightMetrics._chipCache;
      const seen = new Set();
      entries.forEach((entry) => {
        seen.add(entry.key);
        let node = cache.get(entry.key);
        if (!node) {
          node = document.createElement("div");
          node.dataset.key = entry.key;
          const label = document.createElement("div");
          label.className = "chip-label";
          label.textContent = entry.key;
          const value = document.createElement("div");
          value.className = "chip-value";
          node.appendChild(label);
          node.appendChild(value);
          if (entry.action === "switch") {
            node.addEventListener("click", () => requestSourceSwitch());
          }
          cache.set(entry.key, node);
          flightMetrics.appendChild(node);
        }
        const newClass = `flight-chip${entry.buttonish ? " buttonish" : ""}`;
        if (node.className !== newClass) node.className = newClass;
        const valueEl = node.lastElementChild;
        const newVal = String(entry.value);
        if (valueEl.textContent !== newVal) valueEl.textContent = newVal;
      });
      for (const [key, node] of cache) {
        if (!seen.has(key)) {
          node.remove();
          cache.delete(key);
        }
      }
    }

    function buildTileCard(tile) {
      const card = document.createElement("button");
      card.className = "tile-card";
      card.type = "button";
      card.dataset.trackId = String(tile.track_id);
      card.innerHTML = `
        <img class="tile-thumb" alt="" />
        <div class="tile-meta">
          <div class="tile-heading">
            <div class="tile-title"><span class="tile-dot"></span><span class="tile-label-text"></span></div>
            <div class="tile-track"></div>
          </div>
          <div class="tile-pill"></div>
          <div class="tile-stats">
            <div class="tile-stat"><span>Confidence</span><span class="tile-conf"></span></div>
            <div class="tile-stat"><span>Motion</span><span class="tile-motion"></span></div>
            <div class="tile-stat"><span>Age</span><span class="tile-age"></span></div>
            <div class="tile-stat"><span>Score</span><span class="tile-score"></span></div>
          </div>
        </div>
      `;
      const dot = card.querySelector(".tile-dot");
      dot.style.cssText = "display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:middle;";
      card.addEventListener("click", () => {
        const tid = Number(card.dataset.trackId);
        historyFocusLock = false;
        sendMessage({ type: "select_track", track_id: tid });
        switchTab("focus");
      });
      return card;
    }

    function updateTileCard(card, tile) {
      card.dataset.trackId = String(tile.track_id);
      const img = card.querySelector(".tile-thumb");
      const newSrc = dataUriFromBase64("image/jpeg", tile.image);
      if (card.dataset.imgHash !== tile.image) {
        img.src = newSrc;
        card.dataset.imgHash = tile.image || "";
        img.alt = tile.label || "";
      }
      const setText = (sel, val) => {
        const el = card.querySelector(sel);
        if (el && el.textContent !== val) el.textContent = val;
      };
      const dot = card.querySelector(".tile-dot");
      const nextColor = catColor(tile.label);
      if (dot.dataset.color !== nextColor) {
        dot.style.background = nextColor;
        dot.dataset.color = nextColor;
      }
      setText(".tile-label-text", tile.label || "");
      setText(".tile-track", `T${tile.track_id}`);
      setText(".tile-pill", tile.event_tag || "");
      setText(".tile-conf", `${Math.round((tile.confidence || 0) * 100)}%`);
      setText(".tile-motion", `${Math.round((tile.motion_score || 0) * 100)}%`);
      setText(".tile-age", `${(tile.age_seconds || 0).toFixed(1)}s`);
      setText(".tile-score", `${(tile.score || 0).toFixed(2)}`);
    }

    function renderTiles(tiles) {
      if (!Array.isArray(tiles) || tiles.length === 0) {
        if (tileRail._state !== "empty") {
          tileRail.innerHTML = `<div class="empty-state">Insight is scanning. Ranked preview cards will appear here.</div>`;
          tileRail._state = "empty";
          tileRail._cache = new Map();
        }
        return;
      }
      if (tileRail._state !== "tiles") {
        tileRail.innerHTML = "";
        tileRail._state = "tiles";
        tileRail._cache = new Map();
      }
      const cache = tileRail._cache;
      const seen = new Set();
      let prev = null;
      tiles.forEach((tile) => {
        const tid = String(tile.track_id);
        seen.add(tid);
        let card = cache.get(tid);
        if (!card) {
          card = buildTileCard(tile);
          cache.set(tid, card);
        }
        updateTileCard(card, tile);
        const expectedNext = prev ? prev.nextSibling : tileRail.firstChild;
        if (card !== expectedNext) {
          tileRail.insertBefore(card, expectedNext);
        }
        prev = card;
      });
      for (const [tid, node] of cache) {
        if (!seen.has(tid)) {
          node.remove();
          cache.delete(tid);
        }
      }
    }

    function runtimeStageTag(stage) {
      if (stage === "running") return "RT RUN";
      if (stage === "done") return "RT DONE";
      if (stage === "error") return "RT ERR";
      return "RT QUEUED";
    }

    function runtimeStageText(item) {
      const scenario = item.scenario || "scenario";
      const source = item.source || "unknown";
      if (item.stage === "running") return `${scenario} running on ${source}`;
      if (item.stage === "done") return `${scenario} complete on ${source}`;
      if (item.stage === "error") return `${scenario} failed: ${item.error || "unknown error"}`;
      return `${scenario} queued on ${source}`;
    }

    function upsertRuntimeEvent(partial) {
      const runId = String(partial.run_id || "");
      if (!runId) return;
      const nowTs = Math.round(Date.now() / 1000);
      const idx = runtimeEvents.findIndex((evt) => evt.run_id === runId);
      const existing = idx >= 0 ? runtimeEvents[idx] : null;
      const next = {
        kind: "runtime",
        run_id: runId,
        scenario: partial.scenario || existing?.scenario || "",
        source: partial.source || existing?.source || "",
        source_ref: partial.source_ref || existing?.source_ref || {},
        stage: partial.stage || existing?.stage || "queued",
        started_at: partial.started_at || existing?.started_at || nowTs,
        finished_at: partial.finished_at || existing?.finished_at || null,
        error: partial.error || existing?.error || "",
        ts: partial.finished_at || partial.started_at || existing?.ts || nowTs,
      };
      if (idx >= 0) runtimeEvents[idx] = next;
      else runtimeEvents.unshift(next);
      runtimeEvents.sort((a, b) => (Number(b.ts) || 0) - (Number(a.ts) || 0));
      if (runtimeEvents.length > MAX_RUNTIME_EVENTS) runtimeEvents = runtimeEvents.slice(0, MAX_RUNTIME_EVENTS);
    }

    function appendRuntimeLog(runId, item) {
      const rid = String(runId || "");
      if (!rid) return;
      const list = runtimeLogsByRunId.get(rid) || [];
      list.push(item);
      if (list.length > MAX_RUNTIME_LOGS_PER_RUN) {
        runtimeLogsByRunId.set(rid, list.slice(list.length - MAX_RUNTIME_LOGS_PER_RUN));
      } else {
        runtimeLogsByRunId.set(rid, list);
      }
    }

    function getRuntimeLogs(runId) {
      const rid = String(runId || "");
      if (!rid) return [];
      return runtimeLogsByRunId.get(rid) || [];
    }

    function openRuntimeResult(runId) {
      const result = runtimeResultsByRunId.get(runId);
      if (!result) {
        showStatus("No result payload found for this run", "warn");
        return;
      }
      selectedScenarioRunId = runId;
      scenarioCatalogOpen = false;
      renderFocusRuntimePanel();
      switchTab("focus");
    }

    function renderEvents() {
      const detection = Array.isArray(detectionEvents) ? detectionEvents.map((item) => ({
        kind: "detection",
        ts: item.ts,
        tag: item.tag,
        text: item.text,
      })) : [];
      const runtime = runtimeEvents.map((item) => ({
        kind: "runtime",
        ts: item.ts,
        run_id: item.run_id,
        stage: item.stage,
        tag: runtimeStageTag(item.stage),
        text: runtimeStageText(item),
      }));
      const runtimeLogs = [];
      for (const [rid, logs] of runtimeLogsByRunId.entries()) {
        (logs || []).forEach((it) => {
          runtimeLogs.push({
            kind: it.kind || "runtime_log",
            ts: it.ts,
            run_id: rid,
            stage: it.stage || "",
            tag: it.tag || "RT LOG",
            text: it.text || "",
          });
        });
      }
      const combined = [...runtime, ...runtimeLogs, ...detection].sort((a, b) => (Number(b.ts) || 0) - (Number(a.ts) || 0));
      if (combined.length === 0) {
        if (eventList._state !== "empty") {
          eventList.innerHTML = `<div class="empty-state">No event transitions yet.</div>`;
          eventList._state = "empty";
          eventList._sig = "";
        }
        return;
      }
      const sig = combined.map((i) => `${i.kind}|${i.ts}|${i.tag}|${i.text}|${i.run_id || ""}|${i.stage || ""}`).join("\n");
      if (eventList._sig === sig) return;
      eventList._sig = sig;
      eventList._state = "items";
      eventList.innerHTML = "";
      combined.forEach((item) => {
        const row = document.createElement("div");
        if (item.kind === "runtime") {
          const clickable = item.stage === "done" || item.stage === "error";
          const stageClass = String(item.stage || "").replace(/[^a-z0-9_-]/gi, "");
          row.className = `event-item runtime ${stageClass}${clickable ? " clickable" : ""}`;
          row.innerHTML = `
            <strong>${esc(item.tag)}</strong>
            <span>${esc(item.text)}</span>
          `;
          if (clickable) {
            row.addEventListener("click", () => openRuntimeResult(item.run_id));
          }
        } else if (String(item.kind || "").startsWith("runtime_")) {
          row.className = "event-item runtime log";
          row.innerHTML = `
            <strong>${esc(item.tag)}</strong>
            <span>${esc(item.text)}</span>
          `;
        } else {
          row.className = "event-item";
          row.innerHTML = `
            <strong>${esc(item.tag)}</strong>
            <span>${esc(item.text)}</span>
          `;
        }
        eventList.appendChild(row);
      });
    }

    /* ---- Scan overlay drawing ---- */
    function drawFocusOverlays() {
      const canvas = focusOverlayCanvas;
      const container = focusVisual;
      if (!canvas || !container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (w === 0 || h === 0) return;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, w, h);

      const scan = lastScanResults;
      if (!scan || !Array.isArray(scan) || scan.length === 0) return;

      const _HEAT_COLORS = {
        human:    { outer: ["255,180,80","255,140,60","255,100,40"], mid: ["255,130,40","255,100,20","255,80,10"],  core: ["220,80,20","200,60,10"] },
        plant:    { outer: ["80,220,80","40,200,60","20,160,40"],    mid: ["40,200,40","20,180,30","10,140,20"],    core: ["20,180,40","10,140,20"] },
        animal:   { outer: ["0,210,200","0,190,180","0,160,150"],    mid: ["0,200,190","0,170,160","0,140,130"],    core: ["0,190,180","0,150,140"] },
        inorganic:{ outer: ["80,160,255","40,120,255","20,80,220"],  mid: ["40,120,255","20,80,220","10,60,180"],   core: ["20,80,255","10,60,200"] },
        tech:     { outer: ["180,100,255","140,60,220","100,40,180"],mid: ["160,80,240","120,40,200","80,20,160"],  core: ["160,40,255","120,20,200"] },
      };

      // Draw thermal clouds per detection, colored by object category
      if (showHeatmap) {
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(0, 0, w, h, 6);
        ctx.clip();

        scan.forEach(item => {
          const bp = item.bbox_pct;
          if (!bp) return;
          const bx1 = bp[0] / 100 * w;
          const by1 = bp[1] / 100 * h;
          const bx2 = bp[2] / 100 * w;
          const by2 = bp[3] / 100 * h;
          const cx = (bx1 + bx2) / 2;
          const cy = (by1 + by2) / 2;
          const conf = item.confidence || 0.5;
          const bw = Math.max(2, bx2 - bx1);
          const bh = Math.max(2, by2 - by1);
          const cloudRadius = Math.max(bw, bh) * (0.25 + conf * 0.20) * 0.4;
          const hc = _HEAT_COLORS[HEAT_CATS[item.label] || "inorganic"] || _HEAT_COLORS.inorganic;

          // Outer halo
          const grad1 = ctx.createRadialGradient(cx, cy, 0, cx, cy, cloudRadius);
          grad1.addColorStop(0, `rgba(${hc.outer[0]}, ${0.25 * heatmapOpacity})`);
          grad1.addColorStop(0.7, `rgba(${hc.outer[1]}, ${0.15 * heatmapOpacity})`);
          grad1.addColorStop(1, `rgba(${hc.outer[2]}, 0)`);
          ctx.fillStyle = grad1;
          ctx.fillRect(cx - cloudRadius, cy - cloudRadius, cloudRadius * 2, cloudRadius * 2);

          // Middle
          const grad2 = ctx.createRadialGradient(cx, cy, 0, cx, cy, cloudRadius * 0.65);
          grad2.addColorStop(0, `rgba(${hc.mid[0]}, ${0.40 * heatmapOpacity})`);
          grad2.addColorStop(0.8, `rgba(${hc.mid[1]}, ${0.20 * heatmapOpacity})`);
          grad2.addColorStop(1, `rgba(${hc.mid[2]}, 0)`);
          ctx.fillStyle = grad2;
          ctx.fillRect(cx - cloudRadius * 0.65, cy - cloudRadius * 0.65, cloudRadius * 1.3, cloudRadius * 1.3);

          // Core
          const grad3 = ctx.createRadialGradient(cx, cy, 0, cx, cy, cloudRadius * 0.35);
          grad3.addColorStop(0, `rgba(${hc.core[0]}, ${0.55 * heatmapOpacity})`);
          grad3.addColorStop(1, `rgba(${hc.core[1]}, ${0.30 * heatmapOpacity})`);
          ctx.fillStyle = grad3;
          ctx.beginPath();
          ctx.arc(cx, cy, cloudRadius * 0.35, 0, Math.PI * 2);
          ctx.fill();
        });

        ctx.restore();
      }

      // Draw bounding boxes
      if (showBboxOverlay) {
        scan.forEach(item => {
          const bp = item.bbox_pct;
          if (!bp) return;
          const bx1 = bp[0] / 100 * w;
          const by1 = bp[1] / 100 * h;
          const bx2 = bp[2] / 100 * w;
          const by2 = bp[3] / 100 * h;
          const conf = Math.round((item.confidence || 0) * 100);
          const alpha = 0.4 + (item.confidence || 0) * 0.5;

          ctx.strokeStyle = `rgba(255, 42, 42, ${alpha})`;
          ctx.lineWidth = 1.5;
          ctx.setLineDash([4, 3]);
          ctx.strokeRect(bx1, by1, bx2 - bx1, by2 - by1);
          ctx.setLineDash([]);

          // Label background
          const label = `${item.label} ${conf}%`;
          ctx.font = "bold 9px monospace";
          const tm = ctx.measureText(label);
          const lw = tm.width + 6;
          const lh = 14;
          ctx.fillStyle = `rgba(20, 8, 8, 0.75)`;
          ctx.fillRect(bx1, by1 - lh, lw, lh);
          // Label text
          ctx.fillStyle = `rgba(255, 42, 42, ${alpha + 0.1})`;
          ctx.fillText(label, bx1 + 3, by1 - 3);

          // Corner ticks
          const tick = 6;
          ctx.strokeStyle = `rgba(255, 42, 42, ${alpha})`;
          ctx.lineWidth = 2;
          // top-left
          ctx.beginPath(); ctx.moveTo(bx1, by1 + tick); ctx.lineTo(bx1, by1); ctx.lineTo(bx1 + tick, by1); ctx.stroke();
          // top-right
          ctx.beginPath(); ctx.moveTo(bx2 - tick, by1); ctx.lineTo(bx2, by1); ctx.lineTo(bx2, by1 + tick); ctx.stroke();
          // bottom-left
          ctx.beginPath(); ctx.moveTo(bx1, by2 - tick); ctx.lineTo(bx1, by2); ctx.lineTo(bx1 + tick, by2); ctx.stroke();
          // bottom-right
          ctx.beginPath(); ctx.moveTo(bx2 - tick, by2); ctx.lineTo(bx2, by2); ctx.lineTo(bx2, by2 - tick); ctx.stroke();
        });
      }
    }

    /* ---- Focus image zoom control ---- */
    function updateFocusZoom() {
      if (!focusVisualInner) return;
      focusVisualInner.style.transform = `scale(${focusZoom})`;
      if (focusZoomLevel) focusZoomLevel.textContent = Math.round(focusZoom * 100) + "%";
    }

    function zoomFocusImage(delta) {
      focusZoom = Math.max(1.0, Math.min(3.5, focusZoom + delta));
      updateFocusZoom();
    }

    if (focusZoomOut) focusZoomOut.addEventListener("click", () => zoomFocusImage(-0.2));
    if (focusZoomIn) focusZoomIn.addEventListener("click", () => zoomFocusImage(0.2));

    if (focusRefresh) focusRefresh.addEventListener("click", () => {
      if (roiCapturePayload) {
        sendMessage({ type: "capture_roi" });
      }
    });

    function runtimeMatchesFocusContext(runtimeEvent) {
      if (!focusContext || !runtimeEvent) return false;
      if (focusContext.source !== runtimeEvent.source) return false;
      const ref = runtimeEvent.source_ref || {};
      if (focusContext.source === "history") {
        return Number(ref.entry_id) === Number(focusContext.entry_id);
      }
      if (focusContext.source === "roi") {
        if (focusContext.captured_at == null) return true;
        return Number(ref.captured_at || 0) === Number(focusContext.captured_at || 0);
      }
      return false;
    }

    function getActiveRuntimeForFocus() {
      return runtimeEvents.find((evt) => runtimeMatchesFocusContext(evt) && (evt.stage === "queued" || evt.stage === "running")) || null;
    }

    function requestScenarioRun(scenarioName) {
      if (!focusContext) {
        showStatus("Select ROI capture or history entry first", "warn");
        return;
      }
      const payload = {
        type: "run_scenario",
        scenario: scenarioName,
        source: focusContext.source,
      };
      if (focusContext.source === "history") {
        payload.entry_id = Number(focusContext.entry_id);
      } else if (focusContext.source === "roi" && focusContext.captured_at != null) {
        payload.captured_at = focusContext.captured_at;
      }
      scenarioCatalogOpen = false;
      selectedScenarioRunId = null;
      sendMessage(payload);
      renderFocusRuntimePanel();
      switchTab("events");
    }

    function renderFocusRuntimePanel() {
      if (!focusRuntime) return;
      if (!hasFocusContent && !selectedScenarioRunId) {
        focusRuntime.style.display = "none";
        focusRuntime.innerHTML = "";
        return;
      }
      if (scenarioCatalogOpen) {
        focusRuntime.style.display = "";
        if (!focusContext) {
          focusRuntime.innerHTML = `<div class="runtime-minimal">Catalog is available for ROI captures and history entries only.</div>`;
          return;
        }
        if (!Array.isArray(scenarioCatalog) || scenarioCatalog.length === 0) {
          focusRuntime.innerHTML = `<div class="runtime-minimal">No runtime scenarios found. Refresh catalog.</div>`;
          sendMessage({ type: "list_scenarios" });
          return;
        }
        let html = `<div class="focus-scan-header">[CV] Runtime Catalog</div><div class="runtime-catalog-list">`;
        scenarioCatalog.forEach((item) => {
          html += `<button class="runtime-catalog-item" type="button" data-scenario="${esc(item.name)}">`;
          html += `<span class="runtime-catalog-name">${esc(item.display_name || item.name)}</span>`;
          html += `<span class="runtime-catalog-desc">${esc(item.description || "")}</span>`;
          html += `</button>`;
        });
        html += `</div>`;
        focusRuntime.innerHTML = html;
        focusRuntime.querySelectorAll(".runtime-catalog-item").forEach((btn) => {
          btn.addEventListener("click", () => {
            const name = btn.dataset.scenario || "";
            if (!name) return;
            requestScenarioRun(name);
          });
        });
        return;
      }

      if (selectedScenarioRunId) {
        const result = runtimeResultsByRunId.get(selectedScenarioRunId);
        if (result) {
          if (!hasFocusContent) {
            focusEmpty.style.display = "none";
          }
          const signal = result.signal || {};
          const isHot = !!signal.flag;
          const detections = Array.isArray(result.detections) ? result.detections.length : 0;
          let html = `<div class="runtime-card-head"><span>[CV] ${esc(result.scenario || "scenario")}</span><span>${esc(result.source || "")}</span></div>`;
          html += `<div class="runtime-summary">${esc(signal.summary || "no summary")}</div>`;
          html += `<div class="runtime-meta">`;
          html += `<div>Run: ${esc(result.run_id || "")}</div>`;
          html += `<div>Detections: ${detections}</div>`;
          html += `<div>Elapsed: ${Math.round(Number(result.elapsed_ms || 0))} ms</div>`;
          html += `<div><span class="runtime-flag-pill${isHot ? " hot" : ""}">${isHot ? "flagged" : "clear"}</span></div>`;
          html += `</div>`;
          if (result.error) {
            html += `<div class="ai-response-text ai-response-error">${esc(result.error)}</div>`;
          }
          if (result.overlay_image) {
            html += `<img class="runtime-overlay-img" src="${dataUriFromBase64("image/jpeg", result.overlay_image)}" alt="Runtime overlay preview" />`;
          }
          focusRuntime.style.display = "";
          focusRuntime.innerHTML = html;
          return;
        }
      }

      const active = getActiveRuntimeForFocus();
      if (active) {
        focusRuntime.style.display = "";
        const logs = getRuntimeLogs(active.run_id);
        const tail = logs.slice(Math.max(0, logs.length - 10));
        let logHtml = "";
        if (tail.length > 0) {
          logHtml = `<pre class="runtime-log">${tail.map((it) => esc(it.text || "")).join("\n")}</pre>`;
        }
        focusRuntime.innerHTML = `
          <div class="runtime-minimal">Runtime ${esc(active.scenario || "")} ${esc(active.stage)}.</div>
          <div class="runtime-minimal">Open Events for live progress.</div>
          ${logHtml}
        `;
        return;
      }

      focusRuntime.style.display = "none";
      focusRuntime.innerHTML = "";
    }

    if (focusRuntimeCatalogBtn) focusRuntimeCatalogBtn.addEventListener("click", () => {
      if (!focusContext) {
        showStatus("Catalog is available for ROI captures and history entries", "warn");
        return;
      }
      scenarioCatalogOpen = !scenarioCatalogOpen;
      selectedScenarioRunId = null;
      renderFocusRuntimePanel();
    });

    function renderFocus(focus, autoSwitch) {
      if (!focus || !focus.active) {
        hasFocusContent = false;
        focusHeader.style.display = "none";
        focusVisual.style.display = "none";
        focusStats.innerHTML = "";
        focusScan.style.display = "none";
        focusScan.innerHTML = "";
        if (focusRuntime) {
          focusRuntime.style.display = "none";
          focusRuntime.innerHTML = "";
        }
        focusEmpty.style.display = "";
        if (focusControls) focusControls.style.display = "none";
        lastScanResults = null;
        roiCapturePayload = null;
        focusContext = null;
        scenarioCatalogOpen = false;
        selectedScenarioRunId = null;
        roiAiCaptureAt = null;
        roiAiComposerOpen = false;
        roiAiBusy = false;
        roiAiResultText = "";
        roiAiErrorText = "";
        roiAiDraftPrompt = "";
        roiAiProvider = "auto";
        const _ctx = focusOverlayCanvas.getContext("2d");
        _ctx.clearRect(0, 0, focusOverlayCanvas.width, focusOverlayCanvas.height);
        return;
      }
      hasFocusContent = true;
      focusEmpty.style.display = "none";
      focusHeader.style.display = "";
      focusVisual.style.display = "";
      focusTitle.textContent = `${focus.label} #${focus.track_id}`;
      focusCopy.textContent = `Event: ${focus.event_tag}  |  Confidence ${Math.round((focus.confidence || 0) * 100)}%`;
      focusImage.src = dataUriFromBase64("image/jpeg", focus.image);
      focusSilhouette.src = focus.silhouette ? dataUriFromBase64("image/png", focus.silhouette) : "";
      // Show controls for ROI capture, reset zoom
      const isRoiCapture = focus.event_tag && focus.event_tag.startsWith("roi-");
      const isHistoryFocus = focus.source === "history" || Number.isFinite(Number(focus.entry_id));
      if (isRoiCapture) {
        focusContext = { source: "roi", captured_at: focus.captured_at };
      } else if (isHistoryFocus) {
        focusContext = { source: "history", entry_id: Number(focus.entry_id) };
      } else {
        focusContext = null;
      }
      if (focusControls) focusControls.style.display = (isRoiCapture || isHistoryFocus) ? "" : "none";
      if (focusRefresh) focusRefresh.style.display = isRoiCapture ? "" : "none";
      if (focusRuntimeCatalogBtn) focusRuntimeCatalogBtn.style.display = focusContext ? "" : "none";
      if (isRoiCapture) {
        if (roiAiCaptureAt !== focus.captured_at) {
          roiAiCaptureAt = focus.captured_at;
          roiAiComposerOpen = false;
          roiAiBusy = false;
          roiAiResultText = "";
          roiAiErrorText = "";
          roiAiDraftPrompt = "";
          roiAiProvider = "auto";
        }
        focusZoom = 1.0;
        updateFocusZoom();
        roiCapturePayload = focus;
      } else {
        roiAiCaptureAt = null;
        roiAiComposerOpen = false;
        roiAiBusy = false;
        roiAiResultText = "";
        roiAiErrorText = "";
        roiAiDraftPrompt = "";
        roiAiProvider = "auto";
      }
      focusStats.innerHTML = `
        <div class="focus-stat"><span>Track</span>#${focus.track_id}</div>
        <div class="focus-stat"><span>Motion</span>${Math.round((focus.motion_score || 0) * 100)}%</div>
        <div class="focus-stat"><span>Age</span>${(focus.age_seconds || 0).toFixed(1)}s</div>
        <div class="focus-stat"><span>State</span>${esc(focus.event_tag)}</div>
      `;

      // --- ROI scan results ---
      const scan = focus.scan_results;
      lastScanResults = scan;
      if (scan && Array.isArray(scan) && scan.length > 0) {
        focusScan.style.display = "";
        let html = `<div class="focus-scan-header">[SCAN] ${scan.length} object${scan.length !== 1 ? "s" : ""} detected</div>`;
        // Overlay toggle buttons
        html += `<div class="scan-toggles">`;
        if (isRoiCapture) {
          html += `<button class="scan-toggle-btn ai${roiAiComposerOpen ? " active" : ""}" id="askAiToggleBtn" type="button">[AI] Ask AI</button>`;
        }
        html += `<button class="scan-toggle-btn${showBboxOverlay ? " active" : ""}" id="bboxToggleBtn" type="button">[B] Boxes</button>`;
        html += `<button class="scan-toggle-btn${showHeatmap ? " active" : ""}" id="heatmapToggleBtn" type="button">[H] Heat Map</button>`;
        html += `</div>`;
        if (isRoiCapture) {
          html += `<div class="ai-compose${roiAiComposerOpen ? " visible" : ""}" id="aiComposePanel">`;
          html += `<div class="ai-compose-row">`;
          html += `<select class="ai-provider-select" id="aiProviderSelect">`;
          html += `<option value="auto"${roiAiProvider === "auto" ? " selected" : ""}>Auto</option>`;
          html += `<option value="ollama"${roiAiProvider === "ollama" ? " selected" : ""}>Ollama</option>`;
          html += `<option value="openai"${roiAiProvider === "openai" ? " selected" : ""}>GPT API</option>`;
          html += `<option value="anthropic"${roiAiProvider === "anthropic" ? " selected" : ""}>Claude API</option>`;
          html += `</select>`;
          html += `<button class="ai-send-btn" id="aiSendBtn" type="button"${roiAiBusy ? " disabled" : ""}>${roiAiBusy ? "Sending..." : "Send"}</button>`;
          html += `</div>`;
          html += `<textarea class="ai-prompt-input" id="aiPromptInput" placeholder="Optional prompt. Leave empty for automatic analysis.">${esc(roiAiDraftPrompt)}</textarea>`;
          html += `</div>`;
        }
        html += `<div class="heatmap-opacity-row${showHeatmap ? " visible" : ""}" id="heatmapOpacityRow">`;
        html += `<span class="heatmap-opacity-label">Opacity</span>`;
        html += `<input type="range" class="heatmap-opacity-slider" id="heatmapOpacitySlider" min="0.1" max="1" step="0.05" value="${heatmapOpacity}">`;
        html += `<span class="heatmap-opacity-label" id="heatmapOpacityVal">${Math.round(heatmapOpacity * 100)}%</span>`;
        html += `</div>`;
        scan.forEach(item => {
          const pct = Math.round((item.confidence || 0) * 100);
          html += `<div class="scan-item">
            <div class="scan-label"><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${catColor(item.label)};margin-right:4px;vertical-align:middle;"></span>${esc(item.label)}</div>
            <div class="scan-conf-bar"><div class="scan-conf-fill" style="width:${pct}%"></div></div>
            <div class="scan-conf">${pct}%</div>
            <div class="scan-area">${item.area_pct || 0}%</div>
          </div>`;
        });
        if (isRoiCapture && (roiAiBusy || roiAiResultText || roiAiErrorText)) {
          html += `<div class="ai-response">`;
          html += `<div class="ai-response-head">[AI] ${esc(roiAiProvider)} ${roiAiBusy ? " - running" : ""}</div>`;
          if (roiAiErrorText) {
            html += `<div class="ai-response-text ai-response-error">${escMultiline(roiAiErrorText)}</div>`;
          } else if (roiAiResultText) {
            html += `<div class="ai-response-text">${escMultiline(roiAiResultText)}</div>`;
          } else {
            html += `<div class="ai-response-text">Waiting for response...</div>`;
          }
          html += `</div>`;
        }
        focusScan.innerHTML = html;
        // Wire up toggle buttons
        const askAiBtn = document.getElementById("askAiToggleBtn");
        const bboxBtn = document.getElementById("bboxToggleBtn");
        const heatBtn = document.getElementById("heatmapToggleBtn");
        const opacityRow = document.getElementById("heatmapOpacityRow");
        const opacitySlider = document.getElementById("heatmapOpacitySlider");
        const opacityVal = document.getElementById("heatmapOpacityVal");
        const aiProviderSelect = document.getElementById("aiProviderSelect");
        const aiPromptInput = document.getElementById("aiPromptInput");
        const aiSendBtn = document.getElementById("aiSendBtn");
        if (askAiBtn) askAiBtn.addEventListener("click", () => {
          roiAiComposerOpen = !roiAiComposerOpen;
          renderFocus(roiCapturePayload, false);
        });
        if (aiProviderSelect) aiProviderSelect.addEventListener("change", () => {
          roiAiProvider = aiProviderSelect.value || "auto";
        });
        if (aiPromptInput) aiPromptInput.addEventListener("input", () => {
          roiAiDraftPrompt = aiPromptInput.value || "";
        });
        if (aiSendBtn) aiSendBtn.addEventListener("click", () => {
          if (roiAiBusy) return;
          if (aiProviderSelect) roiAiProvider = aiProviderSelect.value || "auto";
          if (aiPromptInput) roiAiDraftPrompt = aiPromptInput.value || "";
          roiAiBusy = true;
          roiAiErrorText = "";
          roiAiResultText = "";
          sendMessage({
            type: "ask_ai_roi",
            provider: roiAiProvider,
            prompt: roiAiDraftPrompt,
          });
          renderFocus(roiCapturePayload, false);
        });
        if (bboxBtn) bboxBtn.addEventListener("click", () => {
          showBboxOverlay = !showBboxOverlay;
          bboxBtn.classList.toggle("active", showBboxOverlay);
          drawFocusOverlays();
        });
        if (heatBtn) heatBtn.addEventListener("click", () => {
          showHeatmap = !showHeatmap;
          heatBtn.classList.toggle("active", showHeatmap);
          if (opacityRow) opacityRow.classList.toggle("visible", showHeatmap);
          drawFocusOverlays();
        });
        if (opacitySlider) opacitySlider.addEventListener("input", () => {
          heatmapOpacity = parseFloat(opacitySlider.value);
          if (opacityVal) opacityVal.textContent = Math.round(heatmapOpacity * 100) + "%";
          drawFocusOverlays();
        });
        // Draw overlays if already toggled on
        requestAnimationFrame(() => drawFocusOverlays());
      } else if (scan && Array.isArray(scan) && scan.length === 0) {
        lastScanResults = null;
        focusScan.style.display = "";
        let html = `<div class="focus-scan-header">[SCAN] region analysis</div><div class="focus-scan-empty">No objects detected in ROI</div>`;
        if (isRoiCapture) {
          html += `<div class="scan-toggles">`;
          html += `<button class="scan-toggle-btn ai${roiAiComposerOpen ? " active" : ""}" id="askAiToggleBtn" type="button">[AI] Ask AI</button>`;
          html += `</div>`;
          html += `<div class="ai-compose${roiAiComposerOpen ? " visible" : ""}" id="aiComposePanel">`;
          html += `<div class="ai-compose-row">`;
          html += `<select class="ai-provider-select" id="aiProviderSelect">`;
          html += `<option value="auto"${roiAiProvider === "auto" ? " selected" : ""}>Auto</option>`;
          html += `<option value="ollama"${roiAiProvider === "ollama" ? " selected" : ""}>Ollama</option>`;
          html += `<option value="openai"${roiAiProvider === "openai" ? " selected" : ""}>GPT API</option>`;
          html += `<option value="anthropic"${roiAiProvider === "anthropic" ? " selected" : ""}>Claude API</option>`;
          html += `</select>`;
          html += `<button class="ai-send-btn" id="aiSendBtn" type="button"${roiAiBusy ? " disabled" : ""}>${roiAiBusy ? "Sending..." : "Send"}</button>`;
          html += `</div>`;
          html += `<textarea class="ai-prompt-input" id="aiPromptInput" placeholder="Optional prompt. Leave empty for automatic analysis.">${esc(roiAiDraftPrompt)}</textarea>`;
          html += `</div>`;
          if (roiAiBusy || roiAiResultText || roiAiErrorText) {
            html += `<div class="ai-response">`;
            html += `<div class="ai-response-head">[AI] ${esc(roiAiProvider)} ${roiAiBusy ? " - running" : ""}</div>`;
            if (roiAiErrorText) {
              html += `<div class="ai-response-text ai-response-error">${escMultiline(roiAiErrorText)}</div>`;
            } else if (roiAiResultText) {
              html += `<div class="ai-response-text">${escMultiline(roiAiResultText)}</div>`;
            } else {
              html += `<div class="ai-response-text">Waiting for response...</div>`;
            }
            html += `</div>`;
          }
        }
        focusScan.innerHTML = html;
        const askAiBtn = document.getElementById("askAiToggleBtn");
        const aiProviderSelect = document.getElementById("aiProviderSelect");
        const aiPromptInput = document.getElementById("aiPromptInput");
        const aiSendBtn = document.getElementById("aiSendBtn");
        if (askAiBtn) askAiBtn.addEventListener("click", () => {
          roiAiComposerOpen = !roiAiComposerOpen;
          renderFocus(roiCapturePayload, false);
        });
        if (aiProviderSelect) aiProviderSelect.addEventListener("change", () => {
          roiAiProvider = aiProviderSelect.value || "auto";
        });
        if (aiPromptInput) aiPromptInput.addEventListener("input", () => {
          roiAiDraftPrompt = aiPromptInput.value || "";
        });
        if (aiSendBtn) aiSendBtn.addEventListener("click", () => {
          if (roiAiBusy) return;
          if (aiProviderSelect) roiAiProvider = aiProviderSelect.value || "auto";
          if (aiPromptInput) roiAiDraftPrompt = aiPromptInput.value || "";
          roiAiBusy = true;
          roiAiErrorText = "";
          roiAiResultText = "";
          sendMessage({
            type: "ask_ai_roi",
            provider: roiAiProvider,
            prompt: roiAiDraftPrompt,
          });
          renderFocus(roiCapturePayload, false);
        });
        drawFocusOverlays();
      } else {
        lastScanResults = null;
        focusScan.style.display = "none";
        focusScan.innerHTML = "";
        drawFocusOverlays();
      }
      renderFocusRuntimePanel();

      if (autoSwitch !== false) switchTab("focus");
    }

    function handleMessage(raw) {
      let payload = null;
      try {
        payload = JSON.parse(raw);
      } catch (_) {
        return;
      }
      if (!payload || !payload.type) return;

      if (payload.type === "source_switch") {
        if (payload.stage === "starting") {
          beginSourceSwap(payload.target || "video", payload.label || "source feed");
          setLoadingExact({ signal: 1, data: 1, video: 0.24, live: 0.08 });
        } else if (payload.stage === "prepared") {
          if (sourceSwapPending) {
            sourceSwapPending.prepared = true;
            setLoadingExact({ signal: 1, data: 1, video: 0.9, live: 0.64 });
            loadingState.textContent = "Ready To Swap";
            loadingCopy.textContent = `${sourceSwapPending.label} is warmed and ready. Confirm when you want to leave the current view.`;
            setLoadingActionsVisible(true);
          }
        } else if (payload.stage === "committing") {
          if (sourceSwapPending) {
            sourceSwapPending.committing = true;
            setLoadingActionsVisible(false);
            setLoadingExact({ signal: 1, data: 1, video: 0.98, live: 0.82 });
            loadingState.textContent = "Swapping";
            loadingCopy.textContent = `Moving to ${sourceSwapPending.label}. Waiting for the new live view to render.`;
          }
        } else if (payload.stage === "ready") {
          if (sourceSwapPending) {
            setLoadingExact({ signal: 1, data: 1, video: 1, live: 0.92 });
          }
        } else if (payload.stage === "cancelled") {
          sourceSwapPending = null;
          hideLoading(0);
        } else if (payload.stage === "failed") {
          sourceSwapPending = null;
          setLoadingActionsVisible(false);
          setLoadingExact({ signal: 1, data: 1, video: 0.12, live: 0.12 });
          showLoading("Swap Failed", payload.message || "The source could not be prepared.");
          blinkLoading("failure", 3, () => {
            window.setTimeout(() => {
              if (!sourceSwapPending) hideLoading(0);
            }, 180);
          });
        }
        return;
      }

      if (!hasReceivedLivePayload && payload.type !== "status") {
        hasReceivedLivePayload = true;
        setLoadingProgress("live", 1);
        loadingState.textContent = "Live";
        loadingCopy.textContent = "Telemetry online. Scene controls and analysis are active.";
        if (!sourceSwapPending && loadingProgress.signal >= 1 && loadingProgress.data >= 1 && loadingProgress.video >= 1) {
          hideLoading();
        }
      }

      if (payload.type === "state_update") {
        if (payload.hud) renderHud(payload.hud);
        if (payload.tiles) renderTiles(payload.tiles);
        if (payload.events) {
          detectionEvents = Array.isArray(payload.events) ? payload.events : [];
          renderEvents();
        }
        if (payload.focus && !historyFocusLock) {
          renderFocus(payload.focus, false);
        }
        if (payload.status) {
          showStatus(payload.status.message, payload.status.level);
        }
        if (payload.history) {
          applyHistoryMessage(payload.history);
        }
        return;
      }
      if (payload.type === "roi_capture") {
        // ROI screenshot — always render and lock focus
        historyFocusLock = true;
        renderFocus(payload.focus || null, true);
        return;
      }
      if (payload.type === "scenario_catalog") {
        scenarioCatalog = Array.isArray(payload.scenarios) ? payload.scenarios : [];
        if (payload.error) {
          showStatus(payload.error, "warn");
        }
        renderFocusRuntimePanel();
        return;
      }
      if (payload.type === "scenario_cell") {
        const rid = String(payload.run_id || "");
        const ts = payload.ts || Math.round(Date.now() / 1000);
        const cell = String(payload.cell_name || "cell");
        const stg = String(payload.cell_status || "");
        const out = String(payload.output || "");
        let text = `[${cell}] ${stg}`;
        if (out) text += ` — ${out.split("\n")[0].slice(0, 240)}`;
        appendRuntimeLog(rid, { ts, kind: "runtime_cell", tag: "RT CELL", text, stage: stg || "running" });
        renderEvents();
        renderFocusRuntimePanel();
        return;
      }
      if (payload.type === "scenario_log") {
        const rid = String(payload.run_id || "");
        const ts = payload.ts || Math.round(Date.now() / 1000);
        const phase = String(payload.phase || "").trim();
        const msg = String(payload.message || "");
        const tag = phase ? `RT ${phase.toUpperCase()}` : "RT LOG";
        const text = phase ? `[${phase}] ${msg}` : msg;
        appendRuntimeLog(rid, { ts, kind: "runtime_log", tag, text, stage: "running" });
        renderEvents();
        renderFocusRuntimePanel();
        return;
      }
      if (payload.type === "scenario_status") {
        upsertRuntimeEvent({
          run_id: payload.run_id,
          scenario: payload.scenario,
          source: payload.source,
          source_ref: payload.source_ref || {},
          stage: payload.stage,
          started_at: payload.started_at,
          finished_at: payload.finished_at,
          error: payload.error || "",
        });
        renderEvents();
        renderFocusRuntimePanel();
        return;
      }
      if (payload.type === "scenario_result") {
        const runId = String(payload.run_id || "");
        if (runId) {
          runtimeResultsByRunId.set(runId, payload);
          upsertRuntimeEvent({
            run_id: runId,
            scenario: payload.scenario,
            source: payload.source,
            source_ref: payload.source_ref || {},
            stage: payload.error ? "error" : "done",
            started_at: payload.started_at,
            finished_at: payload.finished_at || payload.ran_at,
            error: payload.error || "",
          });
          renderEvents();
          if (selectedScenarioRunId === runId) {
            renderFocusRuntimePanel();
          }
        }
        return;
      }
      if (payload.type === "roi_ai_status") {
        if (payload.stage === "started") {
          roiAiBusy = true;
          roiAiErrorText = "";
          roiAiResultText = "";
          roiAiProvider = payload.provider || roiAiProvider || "auto";
          if (
            roiCapturePayload &&
            (payload.captured_at === undefined || payload.captured_at === roiCapturePayload.captured_at)
          ) {
            renderFocus(roiCapturePayload, false);
          }
        }
        return;
      }
      if (payload.type === "roi_ai_result") {
        if (
          roiCapturePayload &&
          payload.captured_at !== undefined &&
          payload.captured_at !== roiCapturePayload.captured_at
        ) {
          return;
        }
        roiAiBusy = false;
        roiAiProvider = payload.provider || roiAiProvider || "auto";
        roiAiErrorText = payload.error || "";
        roiAiResultText = payload.text || "";
        if (roiCapturePayload) {
          renderFocus(roiCapturePayload, false);
        }
        return;
      }
      if (payload.type === "detection_history") {
        applyHistoryMessage(payload);
        return;
      }
      if (payload.type === "status") {
        showStatus(payload.message, payload.level);
      }
    }

    let reconnectAttempts = 0;
    const MAX_RECONNECT = 5;
    const RECONNECT_BASE_MS = 1000;
    let reconnectTimer = null;

    function scheduleReconnect() {
      if (reconnectAttempts >= MAX_RECONNECT) {
        showLoading("Link Lost", "Reconnect failed. Refresh the page to request a fresh session.");
        blinkLoading("failure", 3);
        showStatus("Connection lost. Please refresh.", "error");
        return;
      }
      const delay = RECONNECT_BASE_MS * Math.pow(2, reconnectAttempts);
      reconnectAttempts++;
      resetLoading(`Transport dropped. Re-establishing link in ${Math.round(delay / 1000)}s.`);
      showStatus(`Reconnecting (${reconnectAttempts}/${MAX_RECONNECT})...`, "warn");
      reconnectTimer = setTimeout(async () => {
        try {
          if (pc) { try { pc.close(); } catch (_) {} }
          await connect();
        } catch (e) {
          scheduleReconnect();
        }
      }, delay);
    }

    function insightOfferUrl() {
      return new URL("/offer", window.location.origin).href;
    }

    function canExitLoadingGate() {
      return (
        loadingProgress.video >= 1 &&
        loadingProgress.signal >= 1 &&
        (hasReceivedLivePayload || loadingProgress.data >= 1)
      );
    }

    function maybeHideLoadingWithFallback() {
      if (sourceSwapPending) return;
      if (canExitLoadingGate()) {
        clearLoadingFallbackTimer();
        hideLoading();
      }
    }

    async function connect() {
      resetLoading("Requesting signaling session and transport handshake.");
      clearLoadingFallbackTimer();
      loadingFallbackTimer = window.setTimeout(() => {
        // Fallback for localhost when data channel or telemetry are late:
        // show the video instead of appearing frozen on the loading screen.
        if (!sourceSwapPending && loadingProgress.signal >= 1 && loadingProgress.video >= 1) {
          hideLoading(1);
          showStatus("Live video ready; telemetry channel is still syncing", "warn");
        }
      }, 9000);
      pc = new RTCPeerConnection();
      setLoadingProgress("signal", 0.18);
      pc.addTransceiver("video", { direction: "recvonly" });
      dc = pc.createDataChannel("insight-hud");
      dc.onopen = () => {
        setLoadingProgress("data", 1);
        flushPendingMessages();
        sendMessage({ type: "client_ready" });
        sendMessage({ type: "list_scenarios" });
        loadingState.textContent = "Syncing";
        loadingCopy.textContent = "Control channel is open. Waiting for live video and scene state.";
        showStatus("ATLAS connected");
        maybeHideLoadingWithFallback();
      };
      dc.onmessage = (event) => handleMessage(event.data);
      dc.onclose = () => {
        showLoading("Data Link Lost", "The control channel closed. Attempting to restore session.");
        blinkLoading("failure", 3);
        showStatus("Data channel closed", "warn");
      };

      pc.onconnectionstatechange = () => {
        const state = pc.connectionState;
        if (state === "connected") {
          reconnectAttempts = 0;
          setLoadingProgress("signal", 1);
          maybeHideLoadingWithFallback();
        } else if (state === "connecting") {
          setLoadingProgress("signal", 0.72);
          loadingState.textContent = "Negotiating";
          loadingCopy.textContent = "Peer connection established. Finalizing session transport.";
        } else if (state === "failed" || state === "disconnected" || state === "closed") {
          scheduleReconnect();
        }
      };

      pc.oniceconnectionstatechange = () => {
        if (pc.iceConnectionState === "checking") {
          setLoadingProgress("signal", 0.52);
        } else if (pc.iceConnectionState === "connected" || pc.iceConnectionState === "completed") {
          setLoadingProgress("signal", 0.9);
        } else if (pc.iceConnectionState === "failed") {
          scheduleReconnect();
        }
      };

      pc.ontrack = (event) => {
        const stream = event.streams[0] || new MediaStream([event.track]);
        video.srcObject = stream;
        setLoadingProgress("video", 0.42);
        loadingState.textContent = "Buffering";
        loadingCopy.textContent = "Video stream attached. Waiting for frames to render.";
      };

      video.onloadedmetadata = () => {
        setLoadingProgress("video", 0.68);
      };
      video.oncanplay = () => {
        setLoadingProgress("video", 0.82);
      };
      video.onplaying = () => {
        setLoadingProgress("video", 1);
        maybeHideLoadingWithFallback();
      };

      const offer = await pc.createOffer();
      setLoadingProgress("signal", 0.28);
      await pc.setLocalDescription(offer);
      setLoadingProgress("signal", 0.4);

      const response = await fetch(insightOfferUrl(), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
        }),
      });
      if (!response.ok) {
        const hint =
          response.status === 404
            ? " (wrong URL or not the Insight server — open the page from the same host/port that runs main.py)"
            : "";
        throw new Error(`Server returned ${response.status}${hint}`);
      }
      setLoadingProgress("signal", 0.64);
      const answer = await response.json();
      await pc.setRemoteDescription(answer);
      setLoadingProgress("signal", 0.78);
      maybeHideLoadingWithFallback();
    }

    focusClear.addEventListener("click", () => {
      historyFocusLock = false;
      renderFocus(null);
      sendMessage({ type: "clear_focus" });
      switchTab("previews");
    });

    flightDismiss.addEventListener("click", () => {
      flightStrip.classList.add("dismissed");
      flightRestore.classList.add("visible");
      sidebarEl.style.top = "0";
    });
    flightRestore.addEventListener("click", () => {
      flightStrip.classList.remove("dismissed");
      flightRestore.classList.remove("visible");
      sidebarEl.style.top = "";
    });

    window.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        if (historyFocusLock || hasFocusContent) {
          historyFocusLock = false;
          renderFocus(null);
          sendMessage({ type: "clear_focus" });
          switchTab("previews");
          return;
        }
        if (roiActive) { toggleRoi(); return; }
        sendMessage({ type: "clear_focus" });
      }
      if (event.key.toLowerCase() === "v") {
        requestSourceSwitch();
      }
      if (event.key.toLowerCase() === "r") {
        toggleRoi();
      }
      if (event.key.toLowerCase() === "t") {
        toggleTimeline();
      }
    });

    /* ---- Audio subtitles via Web Speech API + Audio Visualization ---- */
    let subtitleTimeout = null;
    let vizAnimId = null;
    let vizAnalyser = null;
    let vizDataArray = null;

    function resizeVizCanvas() {
      const rect = subtitleBar.getBoundingClientRect();
      audioVizCanvas.width = rect.width * devicePixelRatio;
      audioVizCanvas.height = rect.height * devicePixelRatio;
    }

    let vizCtx = null;
    function drawAudioViz() {
      if (!vizAnalyser || !subtitleBar.classList.contains("visible")) {
        vizAnimId = null;
        return;
      }
      vizAnimId = requestAnimationFrame(drawAudioViz);
      vizAnalyser.getByteFrequencyData(vizDataArray);
      if (!vizCtx) vizCtx = audioVizCanvas.getContext("2d");
      const w = audioVizCanvas.width;
      const h = audioVizCanvas.height;
      vizCtx.clearRect(0, 0, w, h);
      const bins = vizAnalyser.frequencyBinCount;
      const barCount = Math.min(bins, 64);
      const step = Math.floor(bins / barCount);
      const barW = w / barCount;
      for (let i = 0; i < barCount; i++) {
        const val = vizDataArray[i * step] / 255;
        const barH = val * h * 0.9;
        const alpha = 0.15 + val * 0.35;
        vizCtx.fillStyle = `rgba(255, 42, 42, ${alpha})`;
        vizCtx.fillRect(i * barW, h - barH, barW - 1, barH);
      }
    }
    function startVizLoop() {
      if (vizAnimId === null && vizAnalyser) drawAudioViz();
    }
    function stopVizLoop() {
      if (vizAnimId !== null) {
        cancelAnimationFrame(vizAnimId);
        vizAnimId = null;
      }
    }

    let audioStream = null;
    function initAudioVisualization() {
      navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
        audioStream = stream;
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const source = audioCtx.createMediaStreamSource(stream);
        vizAnalyser = audioCtx.createAnalyser();
        vizAnalyser.fftSize = 256;
        vizAnalyser.smoothingTimeConstant = 0.7;
        source.connect(vizAnalyser);
        vizDataArray = new Uint8Array(vizAnalyser.frequencyBinCount);
        resizeVizCanvas();
        window.addEventListener("resize", resizeVizCanvas);
      }).catch((e) => {
        console.warn("[ATLAS] Could not access microphone for visualization:", e);
      });
    }

    let recognitionInstance = null;
    function initSpeechRecognition() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) {
        console.warn("[ATLAS] Speech recognition not supported in this browser");
        return;
      }
      const recognition = new SpeechRecognition();
      recognitionInstance = recognition;
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.lang = "en-US";

      recognition.onresult = (event) => {
        let final = "";
        let interim = "";
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const transcript = event.results[i][0].transcript;
          if (event.results[i].isFinal) {
            final += transcript;
          } else {
            interim += transcript;
          }
        }
        if (final || interim) {
          if (final) {
            subtitleText.textContent = final;
          } else {
            subtitleText.textContent = "";
            const sp = document.createElement("span");
            sp.className = "interim";
            sp.textContent = interim;
            subtitleText.appendChild(sp);
          }
          subtitleBar.classList.add("visible");
          resizeVizCanvas();
          startVizLoop();
          clearTimeout(subtitleTimeout);
          subtitleTimeout = setTimeout(() => {
            subtitleBar.classList.remove("visible");
            stopVizLoop();
          }, final ? 4000 : 8000);
        }
      };

      recognition.onerror = (event) => {
        if (event.error !== "no-speech") {
          console.warn("[ATLAS] Speech error:", event.error);
        }
      };

      recognition.onend = () => {
        try { recognition.start(); } catch (_) {}
      };

      try {
        recognition.start();
        showStatus("Subtitles active", "info");
      } catch (e) {
        console.warn("[ATLAS] Could not start speech recognition:", e);
      }
    }

    // Start speech recognition and audio visualization
    initSpeechRecognition();
    initAudioVisualization();

    // Resource cleanup on page unload
    window.addEventListener("beforeunload", () => {
      clearLoadingFallbackTimer();
      clearInterval(ttlIntervalId);
      stopVizLoop();
      if (audioStream) { audioStream.getTracks().forEach(t => t.stop()); }
      if (recognitionInstance) { try { recognitionInstance.stop(); } catch (_) {} }
      if (reconnectTimer) { clearTimeout(reconnectTimer); }
      if (pc) { try { pc.close(); } catch (_) {} }
    });

    connect().catch((error) => {
      console.error(error);
      showLoading("Connect Failed", `Session bootstrap failed: ${error.message}`);
      blinkLoading("failure", 3);
      showStatus(`Connection failed: ${error.message}`, "error");
    });
  </script>
</body>
</html>
"""


def resolve_model_path(model_value: str) -> Path:
    model_value = model_value.strip()
    if model_value in MODEL_CHOICES:
        return MODEL_CHOICES[model_value]
    return Path(model_value).expanduser().resolve()


def validate_config(config: RuntimeConfig) -> None:
    if not config.model_path.exists():
        raise SystemExit(f"Model file not found: {config.model_path}")
    if config.source == "video" and not config.video_path.exists():
        raise SystemExit(f"Video file not found: {config.video_path}")


def parse_args() -> RuntimeConfig:
    parser = argparse.ArgumentParser(description="Insight no-box tactical CV HUD")
    parser.add_argument("--source", choices=["camera", "video"], default="camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video-path", type=str, default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--model", type=str, default="yolo26n")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-cards", type=int, default=4)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = RuntimeConfig(
        source=args.source,
        camera_index=args.camera_index,
        video_path=Path(args.video_path).expanduser().resolve(),
        model_path=resolve_model_path(args.model),
        host=args.host,
        port=args.port,
        max_cards=max(3, min(6, args.max_cards)),
        debug=args.debug,
    )
    validate_config(config)
    return config


APP_CONFIG = RuntimeConfig(
    source="camera",
    camera_index=0,
    video_path=DEFAULT_VIDEO_PATH,
    model_path=DEFAULT_MODEL_PATH,
    host="0.0.0.0",
    port=8000,
    max_cards=4,
    debug=False,
)

PEER_CONNECTIONS: set[RTCPeerConnection] = set()
SESSIONS: dict[RTCPeerConnection, InsightSession] = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    for session in list(SESSIONS.values()):
        await session.close_async()
    for pc in list(PEER_CONNECTIONS):
        try:
            await pc.close()
        except Exception:
            pass
    SESSIONS.clear()
    PEER_CONNECTIONS.clear()


app = FastAPI(title="Insight Tactical HUD", lifespan=lifespan)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/health", response_class=JSONResponse)
async def health() -> JSONResponse:
    return JSONResponse(
        {
            "status": "ok",
            "source": APP_CONFIG.source,
            "video_path": str(APP_CONFIG.video_path),
            "model": str(APP_CONFIG.model_path),
            "active_sessions": len(SESSIONS),
        }
    )


@app.post("/offer")
async def offer(request: Request) -> JSONResponse:
    params = await request.json()
    offer_description = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    session = InsightSession(APP_CONFIG, loop=asyncio.get_running_loop())
    PEER_CONNECTIONS.add(pc)
    SESSIONS[pc] = session

    async def cleanup_pc() -> None:
        if pc in SESSIONS:
            await SESSIONS.pop(pc).close_async()
        if pc in PEER_CONNECTIONS:
            PEER_CONNECTIONS.discard(pc)
        try:
            await pc.close()
        except Exception:
            pass

    @pc.on("connectionstatechange")
    async def on_connectionstatechange() -> None:
        debug_print(APP_CONFIG.debug, f"[WebRTC] connection state: {pc.connectionState}")
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await cleanup_pc()

    @pc.on("datachannel")
    def on_datachannel(channel: Any) -> None:
        debug_print(APP_CONFIG.debug, f"[WebRTC] data channel: {channel.label}")
        session.register_channel(channel)

        @channel.on("open")
        def on_open() -> None:
            debug_print(APP_CONFIG.debug, f"[WebRTC] data channel open: {channel.label}")
            session.perception.publish_state(force=True)

        @channel.on("close")
        def on_close() -> None:
            session.unregister_channel(channel)

        @channel.on("message")
        def on_message(message: Any) -> None:
            if not isinstance(message, str):
                return
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                session.perception.set_status("Ignored non-JSON client message", "warn")
                return
            if channel not in session.channels:
                session.register_channel(channel)
            session.handle_client_message(data)

    try:
        pc.addTrack(session.video_track)
        await pc.setRemoteDescription(offer_description)
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        return JSONResponse({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})
    except Exception:
        await cleanup_pc()
        raise


def main() -> None:
    global APP_CONFIG
    APP_CONFIG = parse_args()
    debug_print(APP_CONFIG.debug, f"[Insight] source={APP_CONFIG.source}")
    debug_print(APP_CONFIG.debug, f"[Insight] model={APP_CONFIG.model_path}")
    debug_print(APP_CONFIG.debug, f"[Insight] video={APP_CONFIG.video_path}")
    uvicorn.run(app, host=APP_CONFIG.host, port=APP_CONFIG.port)


if __name__ == "__main__":
    main()
