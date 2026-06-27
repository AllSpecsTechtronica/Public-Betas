from __future__ import annotations
import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .runtime_profile import profile_runtime


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = Path(__file__).resolve().parents[2]
LOCAL_ASSETS_DIR = PACKAGE_DIR / "Insight_assets"
LEGACY_ASSETS_DIR = ROOT_DIR / "assets"
ASSETS_DIR = LOCAL_ASSETS_DIR if LOCAL_ASSETS_DIR.exists() else LEGACY_ASSETS_DIR
MODELS_DIR = ASSETS_DIR / "models"
VIDEOS_DIR = ASSETS_DIR / "videos"
CERTIFIED_IMAGE_SIZES = tuple(range(128, 641, 32))
LOCKED_INFERENCE_IMAGE_SIZE = 640
DETECTION_MODES = ("boxes", "segmentation")
SEGMENTATION_BACKENDS = ("yolo",)
TEXT_COLOR_CHOICES = ("black", "bright-cyan")
COLOR_SCHEME_CHOICES = ("aurora", "dark mode")
DETECTOR_MODEL_SUFFIXES = {".pt", ".onnx", ".engine", ".mlpackage"}

DEFAULT_MODEL_PATH = MODELS_DIR / "yolo26n.pt"
DEFAULT_SEG_MODEL_PATH = Path(
    os.environ.get("INSIGHT_SEG_MODEL_PATH", str(MODELS_DIR / "yolo26n-seg.pt"))
).expanduser().resolve()
SIMILARITY_MODEL_PATH = MODELS_DIR / "mobilenet_v3_small_imagenet.pth"
DEFAULT_VIDEO_PATH = VIDEOS_DIR / "frenchpeoplewalkinglong.mp4"
DEFAULT_STATE_DIR = ROOT_DIR / "state" / "insight_local"
OLLAMA_VISION_MODELS = (
    "llava:latest",
    "bakllava:latest",
    "llava:13b",
    "llava:7b",
    "moondream:latest",
    "minicpm-v:latest",
    "qwen2.5vl:latest",
    "granite3.2-vision:latest",
)


def is_apple_silicon() -> bool:
    runtime = profile_runtime()
    return runtime.is_apple_silicon


def _is_supported_model_candidate(candidate: Path) -> bool:
    suffix = candidate.suffix.lower()
    if suffix not in DETECTOR_MODEL_SUFFIXES:
        return False
    if suffix == ".mlpackage":
        return candidate.is_dir()
    return candidate.is_file()


def is_face_detection_model_path(model_path: str | Path) -> bool:
    name = Path(model_path).name.lower()
    return name.startswith("face_detection_yunet") and name.endswith(".onnx")


def _model_preference_key(candidate: Path) -> tuple[int, str]:
    priority = {
        suffix: index
        for index, suffix in enumerate(profile_runtime().preferred_model_suffixes)
    }
    return priority.get(candidate.suffix.lower(), 99), candidate.name.lower()


def discover_model_catalog(models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    catalog: dict[str, Path] = {}
    try:
        candidates = sorted(models_dir.iterdir(), key=lambda path: path.name.lower())
    except FileNotFoundError:
        candidates = []
    for candidate in candidates:
        if not _is_supported_model_candidate(candidate):
            continue
        stem = candidate.stem.lower()
        if not (stem.startswith("yolo") or is_face_detection_model_path(candidate)):
            continue
        catalog[candidate.name] = candidate.resolve()
    if not catalog and _is_supported_model_candidate(DEFAULT_MODEL_PATH):
        catalog[DEFAULT_MODEL_PATH.name] = DEFAULT_MODEL_PATH.resolve()
    return catalog


def normalize_image_size(value: object) -> int:
    # Current YOLO runtime/export is fixed to 640 inference input.
    return LOCKED_INFERENCE_IMAGE_SIZE


def normalize_detection_mode(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in DETECTION_MODES:
        return raw
    return "boxes"


def normalize_segmentation_backend(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in SEGMENTATION_BACKENDS:
        return raw
    return "yolo"


def normalize_text_color(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in TEXT_COLOR_CHOICES:
        return raw
    return "black"


def normalize_color_scheme(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw in COLOR_SCHEME_CHOICES:
        return raw
    # Aurora remains the default fallback for unknown or legacy values.
    return "aurora"


def normalize_fps(value: object) -> int:
    try:
        fps = int(value)
    except (TypeError, ValueError):
        return DEFAULT_FPS
    if fps <= 0:
        return 0
    return max(10, fps)


def discover_model_choices(models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    grouped_candidates: dict[str, list[Path]] = defaultdict(list)
    for candidate in discover_model_catalog(models_dir).values():
        grouped_candidates[candidate.stem].append(candidate.resolve())
    choices = {
        stem: min(paths, key=_model_preference_key)
        for stem, paths in sorted(grouped_candidates.items())
    }
    if not choices and DEFAULT_MODEL_PATH.exists():
        choices[DEFAULT_MODEL_PATH.stem] = DEFAULT_MODEL_PATH.resolve()
    return choices


def list_model_names(models_dir: Path = MODELS_DIR) -> list[str]:
    return list(discover_model_choices(models_dir).keys())


def list_model_catalog_names(models_dir: Path = MODELS_DIR) -> list[str]:
    return list(discover_model_catalog(models_dir).keys())


DEFAULT_CONFIDENCE = float(os.environ.get("INSIGHT_CONFIDENCE", "0.25"))
DEFAULT_IOU = float(os.environ.get("INSIGHT_IOU", "0.20"))
DEFAULT_IMG_SIZE = LOCKED_INFERENCE_IMAGE_SIZE
DEFAULT_MAX_DET = int(os.environ.get("INSIGHT_MAX_DET", "100"))
DEFAULT_FPS = int(os.environ.get("INSIGHT_FPS", "30"))
INSIGHT_SEGMENTATION_BACKEND = normalize_segmentation_backend(
    os.environ.get("INSIGHT_SEGMENTATION_BACKEND", "yolo")
)
TRACK_STALE_SECONDS = float(os.environ.get("INSIGHT_STALE_SEC", "1.4"))
TRACK_STALE_FRAMES = int(os.environ.get("INSIGHT_STALE_FRAMES", "8"))
NEW_TRACK_SECONDS = float(os.environ.get("INSIGHT_NEW_TRACK_SEC", "1.5"))
# -- Confidence hysteresis (anti-flicker) --
CONF_EWMA_ALPHA        = float(os.environ.get("INSIGHT_CONF_EWMA_ALPHA",       "0.35"))
CONF_TIER_MEDIUM       = float(os.environ.get("INSIGHT_CONF_TIER_MEDIUM",      "0.45"))
CONF_TIER_ALPHA_LOCK   = float(os.environ.get("INSIGHT_CONF_TIER_ALPHA",       "0.68"))
CONF_RAPID_DROP_PTS    = float(os.environ.get("INSIGHT_CONF_RAPID_DROP_PTS",   "0.15"))
CONF_RAPID_DROP_FRAMES = int(os.environ.get("INSIGHT_CONF_RAPID_DROP_FRAMES",  "5"))
CONF_GRACE_FRAMES_MULT = float(os.environ.get("INSIGHT_CONF_GRACE_MULT",       "3.0"))
CONF_GRACE_SECS_MULT   = float(os.environ.get("INSIGHT_CONF_GRACE_SEC_MULT",   "2.5"))
PERSISTENT_SECONDS = float(os.environ.get("INSIGHT_PERSISTENT_SEC", "4.0"))
PUBLISH_INTERVAL_SECONDS = float(os.environ.get("INSIGHT_PUBLISH_INTERVAL", "0.25"))
HISTORY_TTL_SECONDS = 30
HISTORY_PUBLISH_INTERVAL = float(os.environ.get("INSIGHT_HISTORY_PUB_INTERVAL", "1.0"))
PREVIEW_MAX_DIM = int(os.environ.get("INSIGHT_PREVIEW_DIM", "240"))
FOCUS_MAX_DIM = int(os.environ.get("INSIGHT_FOCUS_DIM", "420"))
PREVIEW_QUALITY = int(os.environ.get("INSIGHT_PREVIEW_QUALITY", "72"))
AI_HTTP_TIMEOUT_SECONDS = float(os.environ.get("INSIGHT_AI_TIMEOUT_SEC", "35"))
OFFLINE_ONLY = os.environ.get("INSIGHT_OFFLINE_ONLY", "1") != "0"
STATE_DIR = Path(os.environ.get("INSIGHT_STATE_DIR", str(DEFAULT_STATE_DIR))).expanduser().resolve()
BOOT_RETRY_BACKOFF_SEC = tuple(
    float(part.strip())
    for part in os.environ.get("INSIGHT_BOOT_BACKOFF_SEC", "1,5,30").split(",")
    if part.strip()
) or (1.0, 5.0, 30.0)
HEALTH_POLL_SECONDS = float(os.environ.get("INSIGHT_HEALTH_POLL_SEC", "5.0"))
SNAPSHOT_INTERVAL_SECONDS = float(os.environ.get("INSIGHT_SNAPSHOT_INTERVAL_SEC", "10.0"))
DETECTOR_ERROR_THRESHOLD = int(os.environ.get("INSIGHT_DETECTOR_ERROR_THRESHOLD", "3"))
FAULT_LATCH_THRESHOLD = int(os.environ.get("INSIGHT_FAULT_LATCH_THRESHOLD", "10"))

INSIGHT_OLLAMA_URL = os.environ.get("INSIGHT_OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
INSIGHT_OLLAMA_MODEL = os.environ.get("INSIGHT_OLLAMA_MODEL", "llava:latest")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
INSIGHT_OPENAI_MODEL = os.environ.get("INSIGHT_OPENAI_MODEL", "gpt-4.1-mini")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
INSIGHT_ANTHROPIC_MODEL = os.environ.get("INSIGHT_ANTHROPIC_MODEL", "claude-3-5-sonnet-latest")

GALLERY_DIR = ROOT_DIR / "gallery"
GALLERY_DB_PATH = GALLERY_DIR / "gallery.db"
RECOGNITION_THRESHOLD = float(os.environ.get("INSIGHT_RECOG_THRESHOLD", "0.42"))
RECOGNITION_TOP_K = int(os.environ.get("INSIGHT_RECOG_TOP_K", "5"))
RECOGNITION_AUTO = os.environ.get("INSIGHT_RECOG_AUTO", "1") == "1"
RECOGNITION_PERSON_ONLY = os.environ.get("INSIGHT_RECOG_PERSON_ONLY", "1") == "1"
RECOGNITION_MARGIN = float(os.environ.get("INSIGHT_RECOG_MARGIN", "0.045"))
RECOGNITION_CONFIRM_FRAMES = int(os.environ.get("INSIGHT_RECOG_CONFIRM_FRAMES", "3"))
RECOGNITION_MAX_MISSES = int(os.environ.get("INSIGHT_RECOG_MAX_MISSES", "4"))
RECOGNITION_REQUEST_INTERVAL = float(os.environ.get("INSIGHT_RECOG_REQUEST_INTERVAL", "0.45"))
ATTENDANCE_CHECKOUT_SECONDS = float(os.environ.get("INSIGHT_ATTENDANCE_CHECKOUT_SEC", "3.5"))
CVOPS_HOST = os.environ.get("INSIGHT_CVOPS_HOST", "127.0.0.1")
CVOPS_PORT = int(os.environ.get("INSIGHT_CVOPS_PORT", "8787"))
CVOPS_BASE_URL = os.environ.get("INSIGHT_CVOPS_URL", f"http://{CVOPS_HOST}:{CVOPS_PORT}")

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

HEATMAP_CATEGORIES: dict[str, set[str]] = {
    "human": {"person", "face"},
    "plant": {"potted plant"},
    "animal": {
        "dog", "cat", "bird", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe",
    },
    "tech": {
        "laptop", "cell phone", "keyboard", "mouse", "remote", "tv",
        "microwave", "oven", "toaster",
    },
}

_LABEL_TO_CATEGORY: dict[str, str] = {}
for _cat, _labels in HEATMAP_CATEGORIES.items():
    for _lbl in _labels:
        _LABEL_TO_CATEGORY[_lbl] = _cat


def heatmap_category(label: str) -> str:
    """Map a COCO label to a heat-map color category."""
    return _LABEL_TO_CATEGORY.get(label, "inorganic")


@dataclass
class RuntimeConfig:
    source: str = "camera"
    source_locked: bool = False
    camera_index: int = 0
    video_path: Path = DEFAULT_VIDEO_PATH
    model_path: Path = DEFAULT_MODEL_PATH
    state_dir: Path = STATE_DIR
    offline_only: bool = OFFLINE_ONLY
    boot_retry_backoff_sec: tuple[float, ...] = BOOT_RETRY_BACKOFF_SEC
    health_poll_sec: float = HEALTH_POLL_SECONDS
    snapshot_interval_sec: float = SNAPSHOT_INTERVAL_SECONDS
    detector_error_threshold: int = DETECTOR_ERROR_THRESHOLD
    fault_latch_threshold: int = FAULT_LATCH_THRESHOLD
    host: str = "0.0.0.0"
    port: int = 8000
    max_cards: int = 4
    debug: bool = False
    confidence: float = DEFAULT_CONFIDENCE
    iou: float = DEFAULT_IOU
    image_size: int = DEFAULT_IMG_SIZE
    max_det: int = DEFAULT_MAX_DET
    fps: int = DEFAULT_FPS
    detection_mode: str = "boxes"
    text_color: str = "black"
    color_scheme: str = "aurora"


def resolve_model_path(name_or_path: str) -> Path:
    model_catalog = discover_model_catalog()
    if name_or_path in model_catalog:
        return model_catalog[name_or_path]
    model_choices = discover_model_choices()
    if name_or_path in model_choices:
        return model_choices[name_or_path]
    return Path(name_or_path).expanduser().resolve()


def segmentation_pair_for_model(model_path: Path) -> Path:
    resolved = Path(model_path).expanduser().resolve()
    if resolved.stem.endswith("-seg"):
        return resolved
    paired = resolved.with_name(f"{resolved.stem}-seg{resolved.suffix}")
    if paired.exists():
        return paired
    return DEFAULT_SEG_MODEL_PATH


def model_choice_name(model_path: Path) -> str:
    resolved = Path(model_path).expanduser().resolve()
    for name, candidate in discover_model_catalog().items():
        if candidate.resolve() == resolved:
            return name
    for name, candidate in discover_model_choices().items():
        if candidate.resolve() == resolved:
            return name
    return str(resolved)


def validate_config(config: RuntimeConfig) -> None:
    config.image_size = normalize_image_size(config.image_size)
    config.detection_mode = normalize_detection_mode(config.detection_mode)
    config.text_color = normalize_text_color(config.text_color)
    config.color_scheme = normalize_color_scheme(config.color_scheme)
    config.fps = normalize_fps(config.fps)
    if not config.model_path.exists():
        raise SystemExit(f"Model file not found: {config.model_path}")
    if config.source == "video" and not config.video_path.exists():
        raise SystemExit(f"Video file not found: {config.video_path}")


def parse_args() -> RuntimeConfig:
    model_choices = discover_model_choices()
    default_model = DEFAULT_MODEL_PATH.stem
    if default_model not in model_choices and model_choices:
        default_model = next(iter(model_choices))
    parser = argparse.ArgumentParser(description="Insight Local - PyQt6 tactical CV HUD")
    parser.add_argument("--source", choices=["camera", "video"], default="camera")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--video-path", type=str, default=str(DEFAULT_VIDEO_PATH))
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--state-dir", type=str, default=str(STATE_DIR))
    parser.add_argument("--offline-only", action=argparse.BooleanOptionalAction, default=OFFLINE_ONLY)
    parser.add_argument("--max-cards", type=int, default=4)
    parser.add_argument("--text-color", choices=TEXT_COLOR_CHOICES, default="black")
    parser.add_argument("--color-scheme", choices=COLOR_SCHEME_CHOICES, default="aurora")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config = RuntimeConfig(
        source=args.source,
        camera_index=args.camera_index,
        video_path=Path(args.video_path).expanduser().resolve(),
        model_path=resolve_model_path(args.model),
        state_dir=Path(args.state_dir).expanduser().resolve(),
        offline_only=bool(args.offline_only),
        max_cards=max(3, min(6, args.max_cards)),
        text_color=args.text_color,
        color_scheme=args.color_scheme,
        debug=args.debug,
    )
    validate_config(config)
    return config


def debug_print(enabled: bool, *parts: object) -> None:
    if enabled:
        print(*parts, flush=True)
