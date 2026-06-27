from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..config import is_face_detection_model_path
from .imaging import pick_device


class _ArrayProxy:
    def __init__(self, value: Any) -> None:
        self._value = np.asarray(value)

    def numpy(self) -> np.ndarray:
        return self._value

    def tolist(self) -> list:
        return self._value.tolist()


class _DetectionBoxes:
    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xyxy = _ArrayProxy(xyxy.astype(np.float32, copy=False).reshape(-1, 4))
        self.conf = _ArrayProxy(conf.astype(np.float32, copy=False).reshape(-1))
        self.cls = _ArrayProxy(cls.astype(np.int32, copy=False).reshape(-1))


class _DetectionResult:
    def __init__(self, detections: list[dict[str, Any]], names: dict[int, str]) -> None:
        xyxy: list[tuple[float, float, float, float]] = []
        conf: list[float] = []
        cls: list[int] = []
        for det in detections:
            xyxy.append((
                float(det.get("x1", 0.0)),
                float(det.get("y1", 0.0)),
                float(det.get("x2", 0.0)),
                float(det.get("y2", 0.0)),
            ))
            conf.append(float(det.get("conf", 0.0)))
            cls.append(0)
        self.boxes = _DetectionBoxes(
            np.asarray(xyxy, dtype=np.float32),
            np.asarray(conf, dtype=np.float32),
            np.asarray(cls, dtype=np.int32),
        )
        self.names = names


class YoloDetector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self.model_suffix = self.model_path.suffix.lower()
        self.device = pick_device()
        self._model: Any = None
        self._last_error = ""

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def last_error(self) -> str:
        return self._last_error

    def ensure_ready(self) -> bool:
        if self._model is not None:
            return True
        return self.reload()

    def reload(self) -> bool:
        try:
            from ultralytics import YOLO

            if not self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            model = YOLO(str(self.model_path))
            if self._supports_explicit_device():
                model.to(self.device)
            self._model = model
            self._last_error = ""
            return True
        except Exception as exc:
            self._model = None
            self._last_error = str(exc)
            return False

    def predict(
        self,
        frame: np.ndarray,
        *,
        image_size: int,
        confidence: float,
        iou: float,
        max_det: int,
    ) -> list[Any]:
        if not self.ensure_ready():
            raise RuntimeError(self._last_error or "Detector is not ready")
        assert self._model is not None
        # Always pass device so ONNX sessions get CoreMLExecutionProvider on
        # Apple Silicon (mps) instead of falling back to CPUExecutionProvider.
        predict_kwargs = dict(
            source=frame,
            imgsz=image_size,
            conf=confidence,
            iou=iou,
            max_det=max_det,
            verbose=False,
            stream=False,
            device=self.device,
        )
        return self._model.predict(**predict_kwargs)

    def _supports_explicit_device(self) -> bool:
        # model.to() only works for PyTorch (.pt) backends; ONNX/CoreML raise.
        return self.model_suffix == ".pt"


class YuNetFaceDetector:
    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self._backend: Any = None
        self._last_error = ""

    @property
    def is_ready(self) -> bool:
        return self._backend is not None

    @property
    def last_error(self) -> str:
        return self._last_error

    def ensure_ready(self) -> bool:
        if self._backend is not None:
            return True
        return self.reload()

    def reload(self) -> bool:
        try:
            from ..cvops.detection_backends import YuNetFaceDetectorBackend

            self._backend = YuNetFaceDetectorBackend(
                self.model_path,
                score_threshold=0.05,
            )
            self._last_error = ""
            return True
        except Exception as exc:
            self._backend = None
            self._last_error = str(exc)
            return False

    def predict(
        self,
        frame: np.ndarray,
        *,
        image_size: int,
        confidence: float,
        iou: float,
        max_det: int,
    ) -> list[Any]:
        if not self.ensure_ready():
            raise RuntimeError(self._last_error or "Face detector is not ready")
        assert self._backend is not None
        detections = [
            det for det in self._backend.predict(frame)
            if float(det.get("conf", 0.0)) >= float(confidence)
        ]
        detections = detections[: max(1, int(max_det))]
        return [_DetectionResult(detections, {0: "face"})]


def create_detector(model_path: Path) -> YoloDetector | YuNetFaceDetector:
    resolved = Path(model_path).expanduser().resolve()
    if is_face_detection_model_path(resolved):
        return YuNetFaceDetector(resolved)
    return YoloDetector(resolved)
