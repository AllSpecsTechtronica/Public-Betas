from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import numpy as np


OperatingMode = Literal["boot", "full", "degraded_cv", "degraded_manual", "safe_idle", "shutting_down"]
SubsystemState = Literal["starting", "healthy", "degraded", "failed", "disabled"]


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
    recognized_identity: str = ""
    recognition_confidence: float = 0.0
    bbox_norm: tuple[float, float, float, float] | None = None


@dataclass
class HistoryEntry:
    entry_id: int
    track_id: int
    label: str
    confidence: float
    event_tag: str
    age_seconds: float
    score: float
    image: str
    captured_at: float
    recognized_identity: str = ""
    recognition_confidence: float = 0.0


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
    detection_mode: str = "boxes"
    segmentation_backend: str = "yolo"
    segmentation_resource: float = 1.0
    segmentation_ready: bool = True


@dataclass
class SubsystemHealth:
    name: str
    state: SubsystemState
    last_ok_ts: float = 0.0
    last_error: str = ""
    consecutive_failures: int = 0
    restart_count: int = 0
    impact: str = ""


@dataclass
class CapabilityReport:
    frame_source: bool = False
    detector: bool = False
    recognizer: bool = False
    gallery_db: bool = False
    local_ai: bool = False
    cloud_ai: bool = False


@dataclass
class TrackState:
    track_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    first_seen: float
    last_seen: float
    mask_norm: list[tuple[float, float]] = field(default_factory=list)
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
    recognized_identity: str = ""
    recognition_confidence: float = 0.0
    recognition_candidate: str = ""
    recognition_candidate_streak: int = 0
    recognition_miss_streak: int = 0
    recognition_last_update: float = 0.0
    last_recognition_request_ts: float = 0.0
    attendance_identity: str = ""
    # [ANTI-FLICKER] Confidence stabilization fields
    conf_history: deque = field(default_factory=lambda: deque(maxlen=10), repr=False)
    smoothed_conf: float = 0.0   # EWMA of raw detection confidence
    conf_tier: str = "lowest"    # "lowest" | "medium" | "alpha"
