from __future__ import annotations

import argparse
import importlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
from ultralytics import YOLO

from .registry import ScenarioConfig, get_scenario_config, resolve_inference_target


_MODEL_CACHE: dict[str, YOLO] = {}
_MODEL_LOCK = threading.RLock()


def _load_postproc(target: str) -> Callable[[Any, Any, Any], dict[str, Any]]:
    if ":" not in target:
        raise ValueError(f"Invalid postproc target: {target}")
    module_name, func_name = target.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name, None)
    if fn is None or not callable(fn):
        raise ValueError(f"Postproc callable not found: {target}")
    return fn


def _normalize_detections(results: list[Any]) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    for result in results:
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box, conf_score, cls in zip(boxes.xyxy, boxes.conf, boxes.cls):
            coords = [float(v) for v in box.tolist()]
            x1, y1, x2, y2 = coords
            detections.append(
                {
                    "bbox": [x1, y1, x2, y2],
                    "label": str(names.get(int(cls), int(cls))),
                    "confidence": float(conf_score),
                }
            )
    return detections


def _get_model(weights_path: Path) -> YOLO:
    key = str(weights_path.resolve())
    with _MODEL_LOCK:
        model = _MODEL_CACHE.get(key)
        if model is None:
            model = YOLO(key)
            _MODEL_CACHE[key] = model
        return model


def _run_raw_inference(
    weights_path: Path,
    image_bgr: Any,
    *,
    conf: Optional[float] = None,
    iou: Optional[float] = None,
    max_det: Optional[int] = None,
) -> list[dict[str, Any]]:
    if not weights_path.exists() or weights_path.stat().st_size < 32:
        return []
    model = _get_model(weights_path)
    kwargs: dict[str, Any] = {"source": image_bgr, "verbose": False, "stream": False}
    if conf is not None:
        try:
            kwargs["conf"] = max(0.0, min(1.0, float(conf)))
        except Exception:
            pass
    if iou is not None:
        try:
            kwargs["iou"] = max(0.0, min(1.0, float(iou)))
        except Exception:
            pass
    if max_det is not None:
        try:
            kwargs["max_det"] = max(1, int(max_det))
        except Exception:
            pass
    results = model.predict(**kwargs)
    return _normalize_detections(results)


def run_scenario(
    name: str,
    image_bgr: Any,
    *,
    version: str = "",
    payload_extra: Optional[dict[str, Any]] = None,
    cell_callback: Optional[Callable[[Any], None]] = None,
    job_id: str = "",
) -> dict[str, Any]:
    ran_at = time.time()
    started = time.perf_counter()
    try:
        config = get_scenario_config(name)
        from .backbone import BackboneContext
        from .backbones import get_backbone

        backbone = get_backbone(config.backbone_type, config)
        payload = {"version": version}
        if isinstance(payload_extra, dict) and payload_extra:
            payload.update(dict(payload_extra))
        ctx = BackboneContext(
            scenario_config=config,
            job_id=job_id,
            job_type="infer",
            image_bgr=image_bgr,
            payload=payload,
            cell_callback=cell_callback or (lambda _: None),
        )
        result = backbone.run(ctx)
        result.setdefault("scenario", name)
        result.setdefault("ran_at", ran_at)
        if "elapsed_ms" not in result:
            result["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 2)
        try:
            from . import monitoring as _mon

            _mon.maybe_log_inference(scenario=config.name, image_bgr=image_bgr, result=result)
        except Exception:
            pass
        return result
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
        return {
            "scenario": name,
            "ran_at": ran_at,
            "elapsed_ms": elapsed_ms,
            "detections": [],
            "signal": {"flag": False, "summary": "", "metrics": {}},
            "overlay_image": "",
            "error": str(exc),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one CV scenario on an image")
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_path = Path(args.image)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Unable to read image: {image_path}")
    result = run_scenario(args.scenario, image_bgr)
    print(json.dumps(result, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
