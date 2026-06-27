"""audio_recognition.py - lightweight AudioFolder recognition backbone.

This backbone intentionally uses only the Python standard library for the
default path. It builds a nearest-centroid classifier over simple waveform/file
features, which gives CV Ops a local audio-recognition scenario without adding
heavy audio dependencies.
"""
from __future__ import annotations

import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import yaml

from ..audio_ops import FEATURE_SCHEMA, extract_feature_vector
from ..backbone import BackboneBase, BackboneCell, BackboneContext, CellResult
from ..registry import (
    DATASET_AUDIO_SUFFIXES,
    REPO_ROOT,
    list_audiofolder_entries_at,
)


def _resolve_repo_path(path_value: str) -> Path:
    p = Path(str(path_value or "").strip())
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def _next_run_dir(models_root: Path) -> Path:
    runs = [p for p in models_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    if not runs:
        return models_root / "v1"
    latest = max(int(p.name[1:]) for p in runs)
    return models_root / f"v{latest + 1}"


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _extract_features(path: Path) -> list[float]:
    """Return compact audio features for a sample.

    WAV/AIFF files get waveform-aware features. Other audio suffixes fall back
    to byte-level features so datasets can still be cataloged before an operator
    installs a richer decoder.
    """
    suffix = path.suffix.lower()
    try:
        size = float(path.stat().st_size)
    except Exception:
        size = 0.0
    if suffix == ".wav":
        feats = extract_feature_vector(path)
        if any(feats):
            return feats

    try:
        data = path.read_bytes()[:256000]
    except Exception:
        data = b""
    if not data:
        return [0.0] * 9
    vals = [float(b) / 255.0 for b in data]
    return [
        0.0,
        0.0,
        0.0,
        0.0,
        _mean(vals),
        float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        float(min(vals)),
        float(max(vals)),
        math.log1p(size),
    ]


def _dist(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    if n <= 0:
        return float("inf")
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(n)) / n)


def _update_scenario_weights_yaml(config_path: Path, weights_ref: str) -> None:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config must be a mapping: {config_path}")
    raw["weights"] = weights_ref
    config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


class _BuildAudioModelCell(BackboneCell):
    name = "Build Audio Model"
    description = "Extract audio features and build nearest-centroid class model"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        cfg = ctx.scenario_config
        items, split_counts, classes = list_audiofolder_entries_at(cfg.dataset_path)
        train_items = [it for it in items if str(it.get("split") or "") in {"train", "root"}]
        if not train_items:
            train_items = list(items)
        by_class: dict[str, list[list[float]]] = {}
        for item in train_items:
            label = str(item.get("classification_label") or "").strip()
            path = Path(str(item.get("path") or ""))
            if not label or not path.exists():
                continue
            by_class.setdefault(label, []).append(_extract_features(path))
        if not by_class:
            raise RuntimeError("No labeled audio files found for training")
        centroids: dict[str, list[float]] = {}
        for label, rows in sorted(by_class.items()):
            width = max(len(r) for r in rows)
            padded = [r + [0.0] * (width - len(r)) for r in rows]
            centroids[label] = [_mean([row[i] for row in padded]) for i in range(width)]

        models_root = (REPO_ROOT / "mlops" / "models" / cfg.name).resolve()
        run_dir = _next_run_dir(models_root)
        run_dir.mkdir(parents=True, exist_ok=False)
        model_path = run_dir / "model.json"
        metrics_path = run_dir / "metrics.json"
        model = {
            "backbone_type": "audio_recognition",
            "scenario": cfg.name,
            "classes": classes or sorted(centroids),
            "centroids": centroids,
            "feature_schema": FEATURE_SCHEMA,
            "trained_at": time.time(),
            "train_count": len(train_items),
            "split_counts": split_counts,
        }
        model_path.write_text(json.dumps(model, indent=2, ensure_ascii=True), encoding="utf-8")
        metrics = {
            "trained_at": model["trained_at"],
            "train_count": len(train_items),
            "class_count": len(centroids),
            "classes": sorted(centroids),
            "split_counts": split_counts,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
        rel_weights = model_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        _update_scenario_weights_yaml(Path(str(cfg.config_path)), rel_weights)
        print(f"Built audio model for {len(centroids)} class(es) from {len(train_items)} file(s).")
        return CellResult(
            cell_name=self.name,
            status="done",
            output=f"model: {rel_weights}",
            elapsed_ms=0,
            data={
                "weights": rel_weights,
                "result_path": str(run_dir),
                "model_version": run_dir.name,
                "signal": {
                    "flag": False,
                    "summary": f"audio model trained ({len(centroids)} classes)",
                    "metrics": metrics,
                },
            },
        )


class _RecognizeAudioCell(BackboneCell):
    name = "Recognize Audio"
    description = "Classify one audio file with the trained centroid model"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        cfg = ctx.scenario_config
        audio_path_raw = str(ctx.payload.get("audio_path") or ctx.payload.get("path") or "").strip()
        if not audio_path_raw:
            raise RuntimeError("audio_path is required for audio inference")
        audio_path = _resolve_repo_path(audio_path_raw)
        if not audio_path.exists() or audio_path.suffix.lower() not in DATASET_AUDIO_SUFFIXES:
            raise FileNotFoundError(f"Audio file not found or unsupported: {audio_path}")
        model_path = _resolve_repo_path(cfg.weights)
        model = json.loads(model_path.read_text(encoding="utf-8"))
        centroids = model.get("centroids") if isinstance(model, dict) else {}
        if not isinstance(centroids, dict) or not centroids:
            raise RuntimeError("Audio model has no centroids")
        feats = _extract_features(audio_path)
        scored: list[tuple[str, float]] = []
        for label, centroid in centroids.items():
            if isinstance(centroid, list):
                scored.append((str(label), _dist(feats, [float(v) for v in centroid])))
        if not scored:
            raise RuntimeError("Audio model has no valid classes")
        scored.sort(key=lambda x: x[1])
        best_label, best_dist = scored[0]
        confidence = 1.0 / (1.0 + max(0.0, best_dist))
        detections = [
            {
                "label": best_label,
                "confidence": confidence,
                "distance": best_dist,
                "audio_path": str(audio_path),
            }
        ]
        summary = f"audio recognized as {best_label} ({confidence:.2f})"
        print(summary)
        return CellResult(
            cell_name=self.name,
            status="done",
            output=summary,
            elapsed_ms=0,
            data={
                "weights": str(model_path),
                "detections": detections,
                "signal": {
                    "flag": False,
                    "summary": summary,
                    "metrics": {"confidence": confidence, "distance": best_dist},
                },
            },
        )


class AudioRecognitionBackbone(BackboneBase):
    backbone_type = "audio_recognition"

    def __init__(self, config: Any) -> None:
        self._config = config
        self._job_type = "infer"

    @property
    def cells(self) -> list[BackboneCell]:
        if self._job_type == "train":
            return [_BuildAudioModelCell()]
        return [_RecognizeAudioCell()]

    def run(self, ctx: BackboneContext) -> dict[str, Any]:
        self._job_type = ctx.job_type
        return super().run(ctx)

    def _build_result(self, ctx: BackboneContext, cell_results: list[CellResult]) -> dict[str, Any]:
        error = ""
        merged: dict[str, Any] = {}
        for result in cell_results:
            if isinstance(result.data, dict):
                merged.update(result.data)
            if result.status == "error" and not error:
                error = result.output or f"Cell '{result.cell_name}' failed"
        signal = merged.get("signal")
        if not isinstance(signal, dict):
            signal = {"flag": bool(error), "summary": error or "completed", "metrics": {}}
        return {
            "scenario": ctx.scenario_config.name,
            "model_version": str(merged.get("model_version") or ""),
            "weights": str(merged.get("weights") or ctx.scenario_config.weights or ""),
            "result_path": str(merged.get("result_path") or ""),
            "summary": str(signal.get("summary") or ""),
            "detections": merged.get("detections") or [],
            "elapsed_ms": sum(r.elapsed_ms for r in cell_results),
            "overlay_image": "",
            "signal": signal,
            "error": error,
            "artifact_policy": "path_only",
            "backbone_data": {k: v for k, v in merged.items() if k not in {"signal", "detections"}},
        }
