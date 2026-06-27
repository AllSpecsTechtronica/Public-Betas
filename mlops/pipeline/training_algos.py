from __future__ import annotations

from typing import Any


_TRAINER_ALIASES = {
    "yolo": "ultralytics_yolo",
    "ultralytics": "ultralytics_yolo",
    "ultralytics_yolo": "ultralytics_yolo",
}


def normalize_trainer_name(value: Any) -> str:
    raw = str(value or "").strip().lower()
    if not raw:
        return "ultralytics_yolo"
    return _TRAINER_ALIASES.get(raw, raw)


def supported_trainers() -> list[str]:
    # Keep this explicit so unsupported names fail fast.
    return ["ultralytics_yolo"]


def validate_trainer_name(value: Any) -> str:
    trainer = normalize_trainer_name(value)
    if trainer not in supported_trainers():
        raise ValueError(
            f"Unsupported trainer '{value}'. Supported trainers: {', '.join(supported_trainers())}"
        )
    return trainer

