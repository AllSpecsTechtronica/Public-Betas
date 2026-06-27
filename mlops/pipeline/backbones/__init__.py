"""backbones — pluggable backbone registry for cvops.

Register a new backbone by adding its class to _REGISTRY.
"""
from __future__ import annotations

from typing import Any

from .yolo_detection import YoloDetectionBackbone
from .torch_tabular import TorchTabularBackbone
from .custom_code import CustomCodeBackbone
from .face_recognition import FaceRecognitionBackbone
from .audio_recognition import AudioRecognitionBackbone
from .llm_fine_tuning import LlmFineTuningBackbone
from .archival_ingestion import ArchivalIngestionBackbone
from ..backbone import BackboneBase

_REGISTRY: dict[str, type[BackboneBase]] = {
    "yolo_detection": YoloDetectionBackbone,
    "torch_tabular": TorchTabularBackbone,
    "custom_code": CustomCodeBackbone,
    "face_recognition": FaceRecognitionBackbone,
    "audio_recognition": AudioRecognitionBackbone,
    "llm_fine_tuning": LlmFineTuningBackbone,
    "archival_ingestion": ArchivalIngestionBackbone,
}

BACKBONE_LABELS: dict[str, str] = {
    "yolo_detection": "CV Detection (YOLO)",
    "torch_tabular": "ML / Tabular (PyTorch)",
    "custom_code": "Custom Code (Python Cells)",
    "face_recognition": "Face Recognition (Gallery)",
    "audio_recognition": "Audio Recognition",
    "llm_fine_tuning": "LLM Fine Tuning",
    "archival_ingestion": "Archival Ingestion",
}


def get_backbone(backbone_type: str, config: Any) -> BackboneBase:
    """Return an instantiated backbone for the given type and scenario config."""
    btype = str(backbone_type or "yolo_detection").strip().lower()
    cls = _REGISTRY.get(btype)
    if cls is None:
        raise ValueError(
            f"Unknown backbone_type '{btype}'. "
            f"Available: {', '.join(_REGISTRY)}"
        )
    return cls(config)


def list_backbone_types() -> list[dict[str, str]]:
    """Return metadata for all registered backbone types."""
    return [
        {"type": btype, "label": BACKBONE_LABELS.get(btype, btype)}
        for btype in _REGISTRY
    ]
