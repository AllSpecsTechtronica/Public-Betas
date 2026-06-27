"""Lazy Depth-Anything wrapper."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from mlops.image_to_3d.capability import (
    DEPTH_MODEL_ID,
    coreml_tmp_dir,
    coreml_depth_model_path,
    is_coreml_depth_model_package,
    normalize_coreml_depth_model_path,
)


DEPTH_BACKENDS = {"auto", "coreml", "transformers"}


class DepthModel:
    def __init__(
        self,
        model_id: str = DEPTH_MODEL_ID,
        device: str | None = None,
        *,
        backend: str = "auto",
        model_path: str | Path | None = None,
    ) -> None:
        self.model_id = model_id
        self.device = device or _default_device()
        self.requested_backend = _normalize_backend(backend)
        self.model_path = (
            normalize_coreml_depth_model_path(Path(model_path).expanduser()).resolve()
            if model_path
            else coreml_depth_model_path()
        )
        self.backend_used = ""
        self._processor: Any = None
        self._model: Any = None

    def _select_backend(self) -> str:
        if self.requested_backend != "auto":
            return self.requested_backend
        if is_coreml_depth_model_package(self.model_path):
            return "coreml"
        return "transformers"

    def _load_transformers(self) -> tuple[Any, Any]:
        if self._processor is not None and self._model is not None:
            return self._processor, self._model
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "Depth dependencies are missing. Install mlops/dashboard/requirements.txt."
            ) from exc
        self._processor = AutoImageProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
        self._model.to(self.device)
        self._model.eval()
        return self._processor, self._model

    def _load_coreml(self) -> Any:
        if self._model is not None:
            return self._model
        if not is_coreml_depth_model_package(self.model_path):
            raise RuntimeError(f"CoreML depth model package is missing or incomplete: {self.model_path}")
        _prepare_coreml_tmpdir()
        try:
            import coremltools as ct  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "CoreML depth dependencies are missing. Install coremltools or choose the transformers backend."
            ) from exc
        self._model = ct.models.MLModel(str(self.model_path), compute_units=ct.ComputeUnit.ALL)
        return self._model

    def predict(self, image_path: Path) -> np.ndarray:
        backend = self._select_backend()
        self.backend_used = backend
        if backend == "coreml":
            try:
                return self._predict_coreml(image_path)
            except Exception as exc:
                raise RuntimeError(self._coreml_failure_message(exc)) from exc
        return self._predict_transformers(image_path)

    def _coreml_failure_message(self, exc: Exception) -> str:
        if _is_missing_coremltools(exc):
            return _format_coreml_missing_dependency(self.model_path, exc)
        if _is_missing_model_package(exc):
            return _format_coreml_missing_package(self.model_path, exc)
        return _format_coreml_runtime_error(self.model_path, exc)

    def _predict_transformers(self, image_path: Path) -> np.ndarray:
        import torch  # type: ignore[import-not-found]
        import torch.nn.functional as F  # type: ignore[import-not-found]

        processor, model = self._load_transformers()
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            depth = outputs.predicted_depth
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        arr = depth.detach().float().cpu().numpy().astype("float32")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        lo = float(np.percentile(arr, 1))
        hi = float(np.percentile(arr, 99))
        if hi <= lo:
            return np.zeros_like(arr, dtype="float32")
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return arr.astype("float32")

    def _predict_coreml(self, image_path: Path) -> np.ndarray:
        model = self._load_coreml()
        image = Image.open(image_path).convert("RGB")
        input_name, input_shape = _coreml_input(model)
        height = int(input_shape[-2]) if len(input_shape) >= 2 else 518
        width = int(input_shape[-1]) if len(input_shape) >= 1 else 518
        resized = image.resize((width, height), Image.Resampling.BICUBIC)
        arr = np.asarray(resized, dtype="float32") / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        result = model.predict({input_name: arr.astype("float32", copy=False)})
        depth = _first_prediction_array(result)
        if depth is None:
            raise RuntimeError(f"CoreML depth model returned no array outputs: {list(result)}")
        depth = np.asarray(depth, dtype="float32").squeeze()
        if depth.ndim != 2:
            depth = depth.reshape(depth.shape[-2], depth.shape[-1])
        depth_image = Image.fromarray(_normalize_depth(depth), mode="F")
        depth_image = depth_image.resize(image.size, Image.Resampling.BICUBIC)
        return _normalize_depth(np.asarray(depth_image, dtype="float32"))


def _default_device() -> str:
    try:
        import torch  # type: ignore[import-not-found]

        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _normalize_backend(backend: str | None) -> str:
    value = str(backend or "auto").strip().lower()
    if value not in DEPTH_BACKENDS:
        raise ValueError(f"Unsupported depth backend: {backend!r}")
    return value


def _coreml_input(model: Any) -> tuple[str, tuple[int, ...]]:
    spec = model.get_spec()
    inputs = list(spec.description.input)
    if not inputs:
        raise RuntimeError("CoreML depth model has no inputs")
    feature = inputs[0]
    shape: tuple[int, ...] = (1, 3, 518, 518)
    multi_array = getattr(feature.type, "multiArrayType", None)
    if multi_array is not None and getattr(multi_array, "shape", None):
        shape = tuple(int(dim) for dim in multi_array.shape)
    image_type = getattr(feature.type, "imageType", None)
    if image_type is not None and getattr(image_type, "width", 0) and getattr(image_type, "height", 0):
        shape = (1, 3, int(image_type.height), int(image_type.width))
    return feature.name, shape


def _first_prediction_array(result: dict[str, Any]) -> np.ndarray | None:
    preferred = result.get("predicted_depth")
    if preferred is not None:
        return np.asarray(preferred)
    for value in result.values():
        try:
            arr = np.asarray(value)
        except Exception:
            continue
        if arr.size:
            return arr
    return None


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(depth, dtype="float32"), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros_like(arr, dtype="float32")
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype("float32")


def _prepare_coreml_tmpdir() -> None:
    tmpdir = coreml_tmp_dir()
    tmpdir.mkdir(parents=True, exist_ok=True)
    tmp = str(tmpdir)
    os.environ["TMPDIR"] = tmp
    os.environ["TMP"] = tmp
    os.environ["TEMP"] = tmp
    os.environ["TEMPDIR"] = tmp
    tempfile.tempdir = tmp


def _coreml_install_hint() -> str:
    return "Install `coremltools` in the dashboard environment or switch `depth_backend` to `transformers`."


def _coreml_path_hint(path: Path) -> str:
    return f"Local depth model: {path}"


def _exception_message(exc: Exception) -> str:
    text = str(exc).strip()
    return text or exc.__class__.__name__


def _is_missing_coremltools(exc: Exception) -> bool:
    text = _exception_message(exc).lower()
    return "coremltools" in text and "missing" in text


def _is_missing_model_package(exc: Exception) -> bool:
    text = _exception_message(exc).lower()
    return "mlpackage" in text and ("missing" in text or "incomplete" in text)


def _format_coreml_runtime_error(path: Path, exc: Exception) -> str:
    detail = _exception_message(exc)
    return f"CoreML depth inference failed: {detail}. {_coreml_path_hint(path)} {_coreml_install_hint()}"


def _format_coreml_missing_dependency(path: Path, exc: Exception) -> str:
    detail = _exception_message(exc)
    return f"{detail}. {_coreml_path_hint(path)} {_coreml_install_hint()}"


def _format_coreml_missing_package(path: Path, exc: Exception) -> str:
    detail = _exception_message(exc)
    return f"{detail}. {_coreml_path_hint(path)}"
