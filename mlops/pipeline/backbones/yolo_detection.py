"""yolo_detection.py — CV backbone: YOLO detection via ultralytics.

Wraps the existing infer.py three-step pipeline as three BackboneCells:
  1. Resolve Weights  — locate and validate model weights
  2. Run Prediction   — YOLO.predict() + normalize detections
  3. Postprocess      — load postproc fn, build signal + overlay

Inference cell output is backward-compatible with the result dict shape
produced by the old mlops_infer.run_scenario() call.

Training jobs for yolo_detection delegate directly to run_training() (the
ultralytics YOLO training path) — no cell wrapping needed there.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..backbone import BackboneBase, BackboneCell, BackboneContext, CellResult
from ..infer import (
    _get_model,
    _load_postproc,
    _normalize_detections,
    _run_raw_inference,
)
from ..registry import resolve_inference_target


class _ResolveWeightsCell(BackboneCell):
    name = "Resolve Weights"
    description = "Locate trained model weights for the requested version"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        version = str(ctx.payload.get("version") or "")
        override_weights = str(ctx.payload.get("weights_path") or "").strip()
        target = None
        if override_weights:
            weights_path = Path(override_weights).resolve()
            target = {
                "scenario": ctx.scenario_config.name,
                "version": version,
                "weights_path": weights_path,
                "source": "payload_override",
            }
        else:
            target = resolve_inference_target(ctx.scenario_config.name, version)
            if target is None:
                raise RuntimeError(
                    f"Scenario '{ctx.scenario_config.name}' has no usable weights"
                    + (f" at version '{version}'" if version else "")
                )
            weights_path = Path(str(target.get("weights_path") or "")).resolve()
        if not weights_path.exists() or weights_path.stat().st_size < 32:
            raise RuntimeError(f"Weights file missing or empty: {weights_path}")
        model_version = str(target.get("version") or "")
        print(f"weights: {weights_path.name}  version: {model_version or 'latest'}")
        return CellResult(
            cell_name=self.name,
            status="done",
            output="",
            elapsed_ms=0,
            data={
                "weights_path": weights_path,
                "model_version": model_version,
                "target": target,
            },
        )


class _RunPredictionCell(BackboneCell):
    name = "Run Prediction"
    description = "Load YOLO model and run inference on the input image"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        weights_data = prev[0].data
        weights_path: Path = weights_data["weights_path"]

        if ctx.image_bgr is None:
            raise RuntimeError("No input image provided for prediction")

        overrides = ctx.payload.get("infer_overrides") if isinstance(ctx.payload, dict) else {}
        overrides = overrides if isinstance(overrides, dict) else {}
        raw_detections = _run_raw_inference(
            weights_path,
            ctx.image_bgr,
            conf=overrides.get("conf"),
            iou=overrides.get("iou"),
            max_det=overrides.get("max_det"),
        )
        n = len(raw_detections)
        labels = [d.get("label", "") for d in raw_detections]
        label_summary = ", ".join(
            f"{lbl}×{labels.count(lbl)}" for lbl in dict.fromkeys(labels)
        ) if labels else "none"
        print(f"detections: {n}  ({label_summary})")
        return CellResult(
            cell_name=self.name,
            status="done",
            output="",
            elapsed_ms=0,
            data={"raw_detections": raw_detections},
        )


class _PostprocessCell(BackboneCell):
    name = "Postprocess"
    description = "Apply scenario postproc function to build signal and overlay"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        raw_detections = prev[1].data["raw_detections"]
        postproc = _load_postproc(ctx.scenario_config.postproc)
        payload = postproc(ctx.image_bgr, raw_detections, ctx.scenario_config.raw)
        if not isinstance(payload, dict):
            raise RuntimeError("Postproc returned invalid payload (expected dict)")
        signal = payload.get("signal")
        if not isinstance(signal, dict):
            signal = {"flag": False, "summary": "no events", "metrics": {}}
        detections = payload.get("detections")
        if not isinstance(detections, list):
            detections = raw_detections
        flag = bool(signal.get("flag"))
        summary = str(signal.get("summary") or "")
        print(f"signal: {'FLAGGED' if flag else 'CLEAR'}  summary: {summary}")
        return CellResult(
            cell_name=self.name,
            status="done",
            output="",
            elapsed_ms=0,
            data={
                "signal": signal,
                "detections": detections,
                "overlay_image": str(payload.get("overlay_image", "") or ""),
            },
        )


class YoloDetectionBackbone(BackboneBase):
    backbone_type = "yolo_detection"

    _INFER_CELLS: list[BackboneCell] = [
        _ResolveWeightsCell(),
        _RunPredictionCell(),
        _PostprocessCell(),
    ]

    def __init__(self, config: Any) -> None:
        self._config = config

    @property
    def cells(self) -> list[BackboneCell]:
        # Training uses the existing run_training() path — no cells for train.
        return self._INFER_CELLS

    def _build_result(
        self,
        ctx: BackboneContext,
        cell_results: list[CellResult],
    ) -> dict[str, Any]:
        # Collect data from whichever cells completed.
        weights_data: dict = {}
        pred_data: dict = {}
        post_data: dict = {}
        for r in cell_results:
            if r.cell_name == "Resolve Weights":
                weights_data = r.data
            elif r.cell_name == "Run Prediction":
                pred_data = r.data
            elif r.cell_name == "Postprocess":
                post_data = r.data

        # Determine overall error from any failed cell.
        error = ""
        for r in cell_results:
            if r.status == "error":
                error = r.output or f"Cell '{r.cell_name}' failed"
                break

        signal = post_data.get("signal") or {"flag": False, "summary": "", "metrics": {}}
        return {
            "scenario": ctx.scenario_config.name,
            "model_version": str(weights_data.get("model_version") or ""),
            "weights": str(weights_data.get("weights_path") or ""),
            "summary": str(signal.get("summary") or ""),
            "detections": post_data.get("detections") or pred_data.get("raw_detections") or [],
            "elapsed_ms": sum(r.elapsed_ms for r in cell_results),
            "overlay_image": post_data.get("overlay_image") or "",
            "signal": signal,
            "error": error,
            "artifact_policy": "inline_overlay_optional",
        }
