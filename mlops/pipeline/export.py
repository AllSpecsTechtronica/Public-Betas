from __future__ import annotations

from pathlib import Path
from typing import Any

from ultralytics import YOLO

_FORMAT_ALIASES: dict[str, str] = {
    "onnx": "onnx",
    "torchscript": "torchscript",
    "ts": "torchscript",
    "tensorrt": "engine",
    "trt": "engine",
    "engine": "engine",
}


def export_registered_run(
    weights_path: Path,
    export_format: str,
    *,
    imgsz: int = 640,
    out_dir: Path | None = None,
) -> Path:
    """Export a finished run ``weights.pt`` using Ultralytics ``YOLO.export``.

    Parameters
    ----------
    weights_path:
        Absolute path to ``weights.pt`` inside a scenario run directory.
    export_format:
        One of ``onnx``, ``torchscript`` / ``ts``, ``tensorrt`` / ``engine``.
    imgsz:
        Square image size passed through to the exporter.
    out_dir:
        Optional directory for exporter output; defaults to ``<run>/exports``.
    """
    key = str(export_format or "").strip().lower()
    ultralytics_fmt = _FORMAT_ALIASES.get(key)
    if ultralytics_fmt is None:
        allowed = ", ".join(sorted(set(_FORMAT_ALIASES.keys())))
        raise ValueError(f"unsupported export format {export_format!r}; allowed: {allowed}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"weights not found: {weights_path}")
    dest_root = out_dir or (weights_path.parent / "exports")
    dest_root.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights_path))
    export_kwargs: dict[str, Any] = {"format": ultralytics_fmt, "imgsz": int(imgsz)}
    out_path_str = model.export(**export_kwargs)
    return Path(str(out_path_str)).resolve()
