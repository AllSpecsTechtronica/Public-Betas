from __future__ import annotations

import copy
import json
import mimetypes
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .system_guard import build_training_guard


REPO_ROOT = Path(__file__).resolve().parents[2]
MLOPS_ROOT = REPO_ROOT / "mlops"
# Canonical dataset library root used by the CV Ops "Datasets" panel (`/database` endpoints).
# Keep it at repo-root so datasets can live outside `mlops/` (and `mlops/datasets` can be symlinks).
DATABASE_ROOT = REPO_ROOT / "database"
ML_AUDIO_ROOT = REPO_ROOT / "assets" / "ml_audio"
TABULAR_DATASETS_ROOT = MLOPS_ROOT / "datasets"
REGISTRY_PATH = MLOPS_ROOT / "registry.json"
MODEL_SEARCH_ROOTS = [
    REPO_ROOT / "Insight" / "insight_local" / "Insight_assets" / "models",
    REPO_ROOT / "assets" / "models",
    REPO_ROOT / "assets" / "models" / "ocr",
    REPO_ROOT / "Base_Cv_program",
]
MODEL_SUFFIXES = {".pt", ".torchscript", ".onnx", ".engine", ".mlmodel", ".mlpackage", ".tflite"}
DATASET_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATASET_AUDIO_SUFFIXES = {".wav", ".aiff", ".aif", ".flac", ".mp3", ".m4a", ".ogg"}
DATASET_TEXT_SUFFIXES = {".jsonl"}
DATASET_SPLITS = ("train", "val")
LIBRARY_DATASET_FORMAT_YOLO = "yolo_detection"
LIBRARY_DATASET_FORMAT_IMAGEFOLDER = "imagefolder_classification"
LIBRARY_DATASET_FORMAT_CSV = "csv_tabular"
LIBRARY_DATASET_FORMAT_FACE_CSV = "face_csv"
LIBRARY_DATASET_FORMAT_AUDIOFOLDER = "audiofolder_classification"
LIBRARY_DATASET_FORMAT_LLM_JSONL = "llm_instruction_jsonl"
LIBRARY_DATASET_FORMAT_UNKNOWN = "unknown"

DATASET_CATEGORY_IMAGE = "image"
DATASET_CATEGORY_TABULAR = "tabular"
DATASET_CATEGORY_AUDIO = "audio"
DATASET_CATEGORY_TEXT = "text"
YOLO_QUALITY_STOP_DEFAULTS: dict[str, Any] = {
    "quality_stop_enabled": True,
    "quality_stop_metric": "map50_95",
    "quality_stop_threshold": 0.90,
    "quality_stop_min_epochs": 5,
    "quality_stop_consecutive_epochs": 2,
    "quality_stop_rapid_clear_enabled": True,
    "quality_stop_rapid_clear_loss_ratio": 0.35,
    "quality_stop_rapid_clear_metric_margin": 0.03,
    "quality_regression_enabled": True,
    "quality_regression_abs_tolerance": 0.05,
    "quality_regression_rel_tolerance": 0.15,
    "quality_regression_consecutive_epochs": 1,
    "quality_stop_attempt_mode": False,
    "quality_stop_max_time_seconds": 0,
}


def dataset_category(fmt: str) -> str:
    """Return the broad category for a dataset format."""
    if fmt == LIBRARY_DATASET_FORMAT_AUDIOFOLDER:
        return DATASET_CATEGORY_AUDIO
    if fmt == LIBRARY_DATASET_FORMAT_LLM_JSONL:
        return DATASET_CATEGORY_TEXT
    if fmt == LIBRARY_DATASET_FORMAT_CSV:
        return DATASET_CATEGORY_TABULAR
    return DATASET_CATEGORY_IMAGE

# Common ImageFolder-style split directories.
_IMAGEFOLDER_SPLIT_DIRS: list[tuple[str, str]] = [
    ("train", "train"),
    ("valid", "val"),
    ("val", "val"),
    ("test", "test"),
]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    display_name: str
    description: str
    base_model: str
    dataset: str
    classes: list[str]
    postproc: str
    weights: str
    hyperparams: dict[str, Any]
    config_path: Path
    dataset_path: Path
    weights_path: Path
    raw: dict[str, Any]
    backbone_type: str = "yolo_detection"
    backbone_config: dict[str, Any] = field(default_factory=dict)


def _require_fields(data: dict[str, Any], fields: list[str], ctx: str) -> None:
    for field in fields:
        if field not in data:
            raise ValueError(f"Missing '{field}' in {ctx}")


def _resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _is_model_candidate(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix not in MODEL_SUFFIXES:
        return False
    if suffix == ".mlpackage":
        return path.is_dir()
    return path.is_file()


def list_available_models() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in MODEL_SEARCH_ROOTS:
        if not root.exists():
            continue
        for candidate in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not _is_model_candidate(candidate):
                continue
            try:
                rel = candidate.resolve().relative_to(REPO_ROOT).as_posix()
            except Exception:
                rel = str(candidate.resolve())
            if rel in seen:
                continue
            seen.add(rel)
            try:
                size = candidate.stat().st_size
            except Exception:
                size = 0
            out.append(
                {
                    "name": candidate.name,
                    "path": str(candidate.resolve()),
                    "value": rel,
                    "size_bytes": size,
                    "origin": root.name,
                }
            )

    # Append previously-trained models from the registry so they show up as
    # selectable base models alongside the asset/seed weights. Each entry is
    # surfaced as `<scenario>:<run_version>` — a symbolic identifier that
    # `resolve_model_reference` resolves back to the artifact path, and that
    # the training pipeline uses to record `parent_version_id` lineage.
    try:
        from . import model_registry as _mr

        payload = _mr._load_registry()  # type: ignore[attr-defined]
    except Exception:
        payload = {"models": {}}
    models_node = payload.get("models") if isinstance(payload, dict) else {}
    if isinstance(models_node, dict):
        for scenario, node in sorted(models_node.items()):
            if not isinstance(node, dict):
                continue
            versions = node.get("versions") or []
            if not isinstance(versions, list):
                continue
            sorted_versions = sorted(
                (v for v in versions if isinstance(v, dict)),
                key=lambda v: str(v.get("created_at") or ""),
                reverse=True,
            )
            for entry in sorted_versions:
                status = str(entry.get("status") or "active").lower()
                if status == "archived":
                    continue
                run_version = str(entry.get("run_version") or "").strip()
                if not run_version:
                    continue
                artifacts = entry.get("artifacts") if isinstance(entry.get("artifacts"), dict) else {}
                weights = str(artifacts.get("weights") or "")
                if not weights:
                    continue
                weights_path = Path(weights)
                if not weights_path.exists():
                    continue
                value = f"{scenario}:{run_version}"
                if value in seen:
                    continue
                seen.add(value)
                try:
                    size = weights_path.stat().st_size
                except Exception:
                    size = 0
                metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
                map50 = metrics.get("map50") if isinstance(metrics.get("map50"), (int, float)) else None
                forecast = metrics.get("forecast") if isinstance(metrics.get("forecast"), dict) else {}
                out.append(
                    {
                        "name": value,
                        "path": str(weights_path.resolve()),
                        "value": value,
                        "size_bytes": size,
                        "origin": f"trained:{scenario}",
                        "scenario": scenario,
                        "run_version": run_version,
                        "version_id": str(entry.get("version_id") or value),
                        "status": status,
                        "created_at": str(entry.get("created_at") or ""),
                        "map50": map50,
                        "forecast": forecast or {},
                    }
                )
    return out


def resolve_model_reference(model_ref: str) -> Path:
    value = str(model_ref or "").strip()
    if not value:
        raise ValueError("model reference is required")

    # Symbolic reference to a registered trained model: `scenario:run_version`.
    # We resolve via the registry rather than the filesystem so retrains of
    # retrains keep a stable identifier even if artifact paths move.
    if ":" in value and "/" not in value and "\\" not in value:
        scenario, run_version = value.split(":", 1)
        scenario = scenario.strip()
        run_version = run_version.strip()
        if scenario and run_version:
            try:
                from . import model_registry as _mr

                payload = _mr._load_registry()  # type: ignore[attr-defined]
            except Exception:
                payload = {"models": {}}
            node = (payload.get("models") or {}).get(scenario) if isinstance(payload, dict) else None
            if isinstance(node, dict):
                for entry in node.get("versions") or []:
                    if not isinstance(entry, dict):
                        continue
                    if str(entry.get("run_version") or "") != run_version:
                        continue
                    artifacts = entry.get("artifacts") if isinstance(entry.get("artifacts"), dict) else {}
                    weights = str(artifacts.get("weights") or "")
                    if weights:
                        weights_path = Path(weights)
                        if weights_path.exists() and _is_model_candidate(weights_path):
                            return weights_path.resolve()

    as_path = Path(value)
    if as_path.is_absolute() and as_path.exists() and _is_model_candidate(as_path):
        return as_path.resolve()

    rel = (REPO_ROOT / as_path).resolve()
    if rel.exists() and _is_model_candidate(rel):
        return rel

    matches: list[Path] = []
    for root in MODEL_SEARCH_ROOTS:
        candidate = (root / value).resolve()
        if candidate.exists() and _is_model_candidate(candidate):
            matches.append(candidate)
            continue
        by_name = (root / Path(value).name).resolve()
        if by_name.exists() and _is_model_candidate(by_name):
            matches.append(by_name)
    if matches:
        # Deterministic root-priority match.
        return matches[0]
    raise FileNotFoundError(f"Model not found in assets or repo: {value}")


def set_scenario_base_model(scenario: str, model_ref: str) -> dict[str, Any]:
    cfg = get_scenario_config(scenario)
    resolved = resolve_model_reference(model_ref)  # validates the reference
    raw_ref = str(model_ref or "").strip()
    if (
        ":" in raw_ref
        and "/" not in raw_ref
        and "\\" not in raw_ref
        and raw_ref.split(":", 1)[0]
        and raw_ref.split(":", 1)[1]
    ):
        # Preserve the `scenario:run_version` form so retrains record the
        # parent version_id in their lineage.
        model_value = raw_ref
    else:
        try:
            model_value = resolved.relative_to(REPO_ROOT).as_posix()
        except Exception:
            model_value = str(resolved)

    raw = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config must be a mapping: {cfg.config_path}")
    raw["base_model"] = model_value
    cfg.config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return get_scenario_status(scenario)


def set_scenario_dataset(scenario: str, dataset_ref: str) -> dict[str, Any]:
    cfg = get_scenario_config(scenario)
    raw = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config must be a mapping: {cfg.config_path}")

    btype = str(raw.get("backbone_type") or cfg.backbone_type or "yolo_detection").strip().lower()
    dataset_value = str(dataset_ref or "").strip()
    if not dataset_value:
        raise ValueError("Dataset is required.")

    if btype == "yolo_detection":
        dataset_slug = sanitize_library_dataset_slug(dataset_value)
        ds_root = resolve_library_dataset_path(dataset_slug)
        if detect_library_dataset_format(ds_root) != LIBRARY_DATASET_FORMAT_YOLO:
            raise ValueError("Selected dataset is not YOLO detection format. Convert it to YOLO first.")
        _ensure_mlops_dataset_link(dataset_slug)
        class_list = _read_dataset_classes_for_slug(dataset_slug)
        if class_list:
            raw["classes"] = class_list
        raw["dataset"] = dataset_slug
    elif btype == "face_recognition":
        dataset_slug = sanitize_library_dataset_slug(dataset_value)
        ds_root = resolve_library_dataset_path(dataset_slug)
        fmt = detect_library_dataset_format(ds_root)
        if fmt not in {LIBRARY_DATASET_FORMAT_IMAGEFOLDER, LIBRARY_DATASET_FORMAT_FACE_CSV}:
            raise ValueError("Selected dataset is not an ImageFolder (or face CSV) dataset.")
        _ensure_mlops_dataset_link(dataset_slug)
        raw["dataset"] = dataset_slug
    elif btype == "audio_recognition":
        dataset_slug = sanitize_library_dataset_slug(dataset_value)
        ds_root = resolve_audio_library_dataset_path(dataset_slug)
        if detect_library_dataset_format(ds_root) != LIBRARY_DATASET_FORMAT_AUDIOFOLDER:
            raise ValueError("Selected dataset is not AudioFolder classification format.")
        _ensure_mlops_dataset_link(dataset_slug)
        _items, _splits, class_list = list_audiofolder_entries_at(ds_root)
        if class_list:
            raw["classes"] = class_list
        raw["dataset"] = dataset_slug
    elif btype == "llm_fine_tuning":
        ds_path = resolve_llm_jsonl_dataset_path(dataset_value)
        if detect_library_dataset_format(ds_path) != LIBRARY_DATASET_FORMAT_LLM_JSONL:
            raise ValueError("Selected dataset is not JSONL instruction format.")
        try:
            raw["dataset"] = ds_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except Exception:
            raw["dataset"] = str(ds_path.resolve())
    elif btype in {"torch_tabular", "archival_ingestion"}:
        raise ValueError(f"Dataset selection is not supported for backbone_type '{btype}'.")
    else:
        dataset_slug = ""
        try:
            dataset_slug = sanitize_library_dataset_slug(dataset_value)
        except Exception:
            dataset_slug = ""
        raw["dataset"] = dataset_slug

    cfg.config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return get_scenario_status(scenario)


_SCENARIO_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def sanitize_scenario_name(name: str) -> str:
    value = str(name or "").strip()
    if not value or not _SCENARIO_NAME_RE.match(value):
        raise ValueError(
            "Invalid scenario name. Use letters/numbers plus '_' or '-' (max 64 chars)."
        )
    return value


def set_scenario_guard_profile(scenario: str, profile: str) -> dict[str, Any]:
    """Persist a training-guard profile under scenario hyperparams (guard_profile)."""
    cfg = get_scenario_config(scenario)
    raw = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config must be a mapping: {cfg.config_path}")
    hyper = raw.get("hyperparams")
    if not isinstance(hyper, dict):
        hyper = {}
    value = str(profile or "").strip().lower() or "balanced"
    if value not in {"balanced", "stable", "fast"}:
        raise ValueError("guard_profile must be one of: balanced, stable, fast")
    hyper["guard_profile"] = value
    raw["hyperparams"] = hyper
    cfg.config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return get_scenario_status(cfg.name)


# Schema for the full hyperparameter editor. Each entry is (type, validator|None).
# Types: 'int', 'float', 'bool', 'str_choices'. Validators: (min, max) or tuple of choices.
_HYPERPARAM_SCHEMA: dict[str, tuple[str, Any]] = {
    # schedule
    "epochs": ("int", (1, 100000)),
    "imgsz": ("int", (32, 4096)),
    "batch": ("int", (-1, 1024)),  # -1 = Ultralytics auto
    "workers": ("int", (0, 64)),
    "patience": ("int", (0, 100000)),
    "close_mosaic": ("int", (0, 1000)),
    "save_period": ("int", (-1, 100000)),  # -1 = end-of-training only
    # quality target early stop
    "quality_stop_enabled": ("bool", None),
    "quality_stop_metric": ("str_choices", ("map50_95", "map50", "precision", "recall")),
    "quality_stop_threshold": ("float", (0.0, 1.0)),
    "quality_stop_min_epochs": ("int", (1, 100000)),
    "quality_stop_consecutive_epochs": ("int", (1, 100000)),
    "quality_stop_rapid_clear_enabled": ("bool", None),
    "quality_stop_rapid_clear_loss_ratio": ("float", (0.0, 1.0)),
    "quality_stop_rapid_clear_metric_margin": ("float", (0.0, 1.0)),
    "quality_regression_enabled": ("bool", None),
    "quality_regression_abs_tolerance": ("float", (0.0, 1.0)),
    "quality_regression_rel_tolerance": ("float", (0.0, 1.0)),
    "quality_regression_consecutive_epochs": ("int", (1, 100000)),
    "quality_stop_attempt_mode": ("bool", None),
    "quality_stop_max_time_seconds": ("int", (0, 86400)),
    # optimizer
    "optimizer": ("str_choices", ("SGD", "Adam", "AdamW", "NAdam", "RAdam", "RMSProp", "auto")),
    "lr0": ("float", (1e-6, 1.0)),
    "lrf": ("float", (1e-6, 1.0)),
    "momentum": ("float", (0.0, 0.999)),
    "weight_decay": ("float", (0.0, 1.0)),
    "warmup_epochs": ("float", (0.0, 100.0)),
    "warmup_momentum": ("float", (0.0, 0.999)),
    "warmup_bias_lr": ("float", (0.0, 1.0)),
    "cos_lr": ("bool", None),
    "amp": ("bool", None),
    # regularization
    "dropout": ("float", (0.0, 1.0)),
    "label_smoothing": ("float", (0.0, 1.0)),
    # augmentation
    "hsv_h": ("float", (0.0, 1.0)),
    "hsv_s": ("float", (0.0, 1.0)),
    "hsv_v": ("float", (0.0, 1.0)),
    "degrees": ("float", (0.0, 180.0)),
    "translate": ("float", (0.0, 1.0)),
    "scale": ("float", (0.0, 1.0)),
    "shear": ("float", (0.0, 180.0)),
    "perspective": ("float", (0.0, 0.001)),
    "fliplr": ("float", (0.0, 1.0)),
    "flipud": ("float", (0.0, 1.0)),
    "mosaic": ("float", (0.0, 1.0)),
    "mixup": ("float", (0.0, 1.0)),
    "copy_paste": ("float", (0.0, 1.0)),
    "erasing": ("float", (0.0, 1.0)),
    # reproducibility
    "seed": ("int", (0, 2**31 - 1)),
    "deterministic": ("bool", None),
}


def hyperparam_schema() -> dict[str, tuple[str, Any]]:
    """Expose the hyperparam validation schema (used by the service endpoint)."""
    return dict(_HYPERPARAM_SCHEMA)


def _coerce_hyperparam_value(key: str, value: Any) -> Any:
    kind, validator = _HYPERPARAM_SCHEMA[key]
    if kind == "int":
        coerced = int(float(value))
        if isinstance(validator, tuple):
            lo, hi = validator
            if coerced < lo or coerced > hi:
                raise ValueError(f"{key}={coerced} outside allowed range [{lo}, {hi}]")
        return coerced
    if kind == "float":
        coerced = float(value)
        if isinstance(validator, tuple):
            lo, hi = validator
            if coerced < lo or coerced > hi:
                raise ValueError(f"{key}={coerced} outside allowed range [{lo}, {hi}]")
        return coerced
    if kind == "bool":
        if isinstance(value, bool):
            return value
        s = str(value).strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{key} must be boolean, got {value!r}")
    if kind == "str_choices":
        s = str(value)
        if isinstance(validator, tuple) and s not in validator:
            raise ValueError(f"{key}={s!r} not in {validator}")
        return s
    raise ValueError(f"unknown hyperparam kind for {key}")


def set_scenario_hyperparams(
    scenario: str,
    updates: Mapping[str, Any],
    *,
    reset: bool = False,
) -> dict[str, Any]:
    """Merge a partial hyperparams dict into the scenario YAML.

    Unknown keys are rejected. When `reset=True`, the incoming dict fully
    replaces the scenario hyperparams (except for `guard_profile`, which is
    preserved so the training-guard selection isn't silently lost).
    """
    cfg = get_scenario_config(scenario)
    raw = yaml.safe_load(cfg.config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config must be a mapping: {cfg.config_path}")
    hyper = raw.get("hyperparams")
    if not isinstance(hyper, dict):
        hyper = {}

    accepted: dict[str, Any] = {}
    rejected: dict[str, str] = {}
    for key, value in (updates or {}).items():
        if key == "guard_profile":
            # guard_profile has its own endpoint + validator; accept here too for
            # round-trip convenience.
            accepted[key] = str(value or "balanced").strip().lower() or "balanced"
            continue
        if key not in _HYPERPARAM_SCHEMA:
            rejected[key] = "unknown key"
            continue
        try:
            accepted[key] = _coerce_hyperparam_value(key, value)
        except Exception as exc:
            rejected[key] = str(exc)

    if reset:
        preserved = {k: hyper[k] for k in ("guard_profile",) if k in hyper}
        hyper = {**preserved, **accepted}
    else:
        hyper.update(accepted)

    raw["hyperparams"] = hyper
    cfg.config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    status = get_scenario_status(cfg.name)
    status["hyperparam_updates"] = {"accepted": accepted, "rejected": rejected}
    return status


def _read_dataset_classes_for_slug(dataset_slug: str) -> list[str]:
    """Best-effort class extraction for a YOLO dataset under database/<slug>."""
    ds_root = resolve_library_dataset_path(dataset_slug)
    if detect_library_dataset_format(ds_root) != LIBRARY_DATASET_FORMAT_YOLO:
        return []
    return _read_yolo_classes_at(ds_root)


def _read_yolo_classes_at(ds_root: Path) -> list[str]:
    ds_root = ds_root.resolve()
    classes_path = ds_root.resolve() / "classes.txt"
    if classes_path.exists():
        try:
            classes = [
                ln.strip()
                for ln in classes_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if ln.strip()
            ]
            if classes:
                return classes
        except Exception:
            pass

    data_yaml = ds_root / "data.yaml"
    if not data_yaml.exists():
        return []
    try:
        raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return []
    if not isinstance(raw, dict):
        return []
    names = raw.get("names")
    if isinstance(names, list):
        return [str(item).strip() for item in names if str(item).strip()]
    if isinstance(names, dict):
        parsed: list[tuple[int, str]] = []
        for k, v in names.items():
            try:
                idx = int(k)
            except Exception:
                try:
                    idx = int(float(k))
                except Exception:
                    continue
            name = str(v).strip()
            if name:
                parsed.append((idx, name))
        return [name for _idx, name in sorted(parsed, key=lambda item: item[0])]
    return []


def _ensure_mlops_dataset_link(dataset_slug: str) -> None:
    """Ensure `mlops/datasets/<slug>` exists and points at the library dataset when possible."""
    slug = sanitize_library_dataset_slug(dataset_slug)
    link_path = (MLOPS_ROOT / "datasets" / slug)
    # If already present, assume operator intentionally set it up.
    if link_path.exists():
        return
    try:
        ds_target = resolve_library_dataset_path(slug).resolve()
    except Exception:
        ds_target = (DATABASE_ROOT / slug).resolve()
    if not ds_target.is_dir():
        # Nothing to link to; caller may be using a dataset already living in mlops/datasets/.
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Use a relative symlink so the repo can move as a unit.
        import os

        rel_target = os.path.relpath(str(ds_target), start=str((MLOPS_ROOT / "datasets").resolve()))
        link_path.symlink_to(rel_target, target_is_directory=True)
        return
    except Exception:
        try:
            link_path.symlink_to(ds_target, target_is_directory=True)
            return
        except Exception as exc:
            raise ValueError(f"Could not create dataset link for '{slug}': {exc}") from exc


def create_scenario_profile(
    *,
    name: str,
    display_name: str,
    description: str,
    base_model: str = "",
    dataset: str = "",
    classes: list[str] | None = None,
    postproc: str = "mlops.pipeline.postproc.generic_detection:run",
    hyperparams: dict[str, Any] | None = None,
    guard_profile: str = "balanced",
    backbone_type: str = "yolo_detection",
    backbone_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a new scenario (profile): YAML config + registry entry.

    Supports these backbone types:
      - yolo_detection: requires a YOLO dataset, base_model, and classes
      - torch_tabular:  dataset/base_model/classes are optional; backbone_config
                        carries configured execution cells + ML-specific params
      - custom_code:    optional library dataset; draft/promoted Python cells in backbone_config
      - face_recognition: requires an ImageFolder dataset; produces a gallery.db artifact
      - audio_recognition: requires an AudioFolder dataset; produces model.json
      - llm_fine_tuning: requires JSONL instruction data; produces a LoRA adapter
      - archival_ingestion: managed archival corpus + phased snapshot pipeline
    """
    scenario_name = sanitize_scenario_name(name)
    disp = str(display_name or "").strip() or scenario_name
    desc = str(description or "").strip() or f"Scenario profile for {disp}."
    btype = str(backbone_type or "yolo_detection").strip().lower()
    is_tabular = btype == "torch_tabular"
    is_custom_code = btype == "custom_code"
    is_face = btype == "face_recognition"
    is_audio = btype == "audio_recognition"
    is_llm = btype == "llm_fine_tuning"
    is_archival = btype == "archival_ingestion"
    is_yolo = btype == "yolo_detection"
    if btype not in {
        "yolo_detection",
        "torch_tabular",
        "custom_code",
        "face_recognition",
        "audio_recognition",
        "llm_fine_tuning",
        "archival_ingestion",
    }:
        raise ValueError(
            "backbone_type must be one of: yolo_detection, torch_tabular, custom_code, "
            "face_recognition, audio_recognition, llm_fine_tuning, archival_ingestion"
        )

    if is_tabular or is_custom_code:
        # Tabular / custom_code: dataset slug is optional (invalid slugs are ignored).
        dataset_slug = ""
        if dataset:
            try:
                dataset_slug = sanitize_library_dataset_slug(dataset)
            except Exception:
                dataset_slug = ""
        if dataset_slug:
            _ensure_mlops_dataset_link(dataset_slug)
        class_list = [str(c).strip() for c in (classes or []) if str(c).strip()] if classes else []
        hp: dict[str, Any] = {}
    elif is_face:
        dataset_slug = sanitize_library_dataset_slug(dataset)
        ds_root = resolve_library_dataset_path(dataset_slug)
        fmt = detect_library_dataset_format(ds_root)
        if fmt not in {LIBRARY_DATASET_FORMAT_IMAGEFOLDER, LIBRARY_DATASET_FORMAT_FACE_CSV}:
            raise ValueError("Selected dataset is not an ImageFolder (or face CSV) dataset.")
        _ensure_mlops_dataset_link(dataset_slug)
        class_list = [str(c).strip() for c in (classes or []) if str(c).strip()]
        hp = {}
    elif is_audio:
        dataset_slug = sanitize_library_dataset_slug(dataset)
        ds_root = resolve_audio_library_dataset_path(dataset_slug)
        if detect_library_dataset_format(ds_root) != LIBRARY_DATASET_FORMAT_AUDIOFOLDER:
            raise ValueError("Selected dataset is not AudioFolder classification format.")
        _ensure_mlops_dataset_link(dataset_slug)
        class_list = [str(c).strip() for c in (classes or []) if str(c).strip()]
        if not class_list:
            _items, _splits, class_list = list_audiofolder_entries_at(ds_root)
        hp = {}
    elif is_llm:
        ds_path = resolve_llm_jsonl_dataset_path(dataset)
        if detect_library_dataset_format(ds_path) != LIBRARY_DATASET_FORMAT_LLM_JSONL:
            raise ValueError("Selected dataset is not JSONL instruction format.")
        try:
            dataset_slug = ds_path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
        except Exception:
            dataset_slug = str(ds_path.resolve())
        class_list = []
        hp = {}
    elif is_archival:
        dataset_slug = ""
        class_list = []
        hp = {}
    else:
        dataset_slug = sanitize_library_dataset_slug(dataset)
        # Require a YOLO dataset (train.py expects images/ + labels/).
        ds_root = resolve_library_dataset_path(dataset_slug)
        if detect_library_dataset_format(ds_root) != LIBRARY_DATASET_FORMAT_YOLO:
            raise ValueError("Selected dataset is not YOLO detection format. Convert it to YOLO first.")
        _ensure_mlops_dataset_link(dataset_slug)

        class_list = [str(c).strip() for c in (classes or []) if str(c).strip()]
        if not class_list:
            class_list = _read_dataset_classes_for_slug(dataset_slug)
        if not class_list:
            raise ValueError("Classes are required. Add classes.txt to the dataset or enter classes manually.")

        hp = dict(hyperparams or {})
        # Defaults to something sane; the system guard will clamp if needed.
        hp.setdefault("epochs", 20)
        hp.setdefault("imgsz", 640)
        for key, value in YOLO_QUALITY_STOP_DEFAULTS.items():
            hp.setdefault(key, value)
        gp = str(guard_profile or "").strip().lower() or "balanced"
        if gp not in {"balanced", "stable", "fast"}:
            raise ValueError("guard_profile must be one of: balanced, stable, fast")
        hp["guard_profile"] = gp

    cfg_dir = MLOPS_ROOT / "scenarios"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg_dir / f"{scenario_name}.yaml"
    if cfg_path.exists():
        raise ValueError(f"Scenario config already exists: {cfg_path}")

    weights_ref = ""
    if is_yolo:
        weights_ref = f"mlops/models/{scenario_name}/v1/weights.pt"
    elif is_face:
        weights_ref = f"mlops/models/{scenario_name}/v1/gallery.db"
    elif is_audio:
        weights_ref = f"mlops/models/{scenario_name}/v1/model.json"
    elif is_llm:
        weights_ref = f"mlops/models/{scenario_name}/v1/adapter/adapter_model.safetensors"
    raw: dict[str, Any] = {
        "name": scenario_name,
        "display_name": disp,
        "description": desc,
        "backbone_type": btype,
        "dataset": dataset_slug,
        "ci_cd": {
            "enabled": True,
            "metric": str(hp.get("quality_stop_metric") or "map50_95"),
            "threshold": float(hp.get("quality_stop_threshold") or 0.90),
            "regression_tolerance": 0.02,
            "promotion": "manual",
            "required_artifacts": ["weights", "metrics.json", "dataset_snapshot", "repro_manifest"],
        },
    }
    if is_yolo:
        raw["base_model"] = str(base_model or "").strip()
        raw["classes"] = class_list
        raw["postproc"] = str(postproc or "").strip() or "mlops.pipeline.postproc.generic_detection:run"
        raw["weights"] = weights_ref
        raw["hyperparams"] = hp
    elif is_tabular or is_custom_code:
        raw["base_model"] = ""
        raw["classes"] = class_list
        raw["postproc"] = ""
        raw["weights"] = ""
        raw["hyperparams"] = {}
    elif is_llm:
        llm_cfg = dict(backbone_config or {})
        base_ref = str(llm_cfg.get("base_model") or base_model or "").strip()
        if not base_ref:
            raise ValueError("LLM fine-tuning requires backbone_config.base_model.")
        llm_cfg.setdefault("base_model", base_ref)
        llm_cfg.setdefault("sources", ["jsonl", "feedback"])
        raw["base_model"] = base_ref
        raw["classes"] = []
        raw["postproc"] = ""
        raw["weights"] = weights_ref
        raw["hyperparams"] = {}
        backbone_config = llm_cfg
    elif is_archival:
        archival_cfg = dict(backbone_config or {})
        archival_cfg.setdefault("domain_profile", {})
        archival_cfg.setdefault("assembly_rules", {})
        archival_cfg.setdefault("providers", {})
        archival_cfg.setdefault("phase_defaults", {})
        archival_cfg.setdefault("archive_storage_root", "state/insight_local/cvops/archive_corpora")
        archival_cfg.setdefault("corpus_id", "")
        archival_cfg.setdefault("dataset_version_id", "")
        archival_cfg.setdefault("latest_snapshot_id", "")
        raw["base_model"] = ""
        raw["classes"] = []
        raw["postproc"] = ""
        raw["weights"] = ""
        raw["hyperparams"] = {}
        backbone_config = archival_cfg
    else:
        # gallery/audio backbones
        raw["base_model"] = ""
        raw["classes"] = class_list
        raw["postproc"] = ""
        raw["weights"] = weights_ref
        raw["hyperparams"] = {}
    if backbone_config:
        raw["backbone_config"] = dict(backbone_config)
    cfg_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    reg = load_registry()
    scenarios = reg.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Registry scenarios must be an array")
    if any(isinstance(item, dict) and str(item.get("name") or "") == scenario_name for item in scenarios):
        raise ValueError(f"Scenario already exists in registry: {scenario_name}")
    try:
        rel_cfg = cfg_path.resolve().relative_to(REPO_ROOT).as_posix()
    except Exception:
        rel_cfg = str(cfg_path.resolve())
    scenarios.append({"name": scenario_name, "config": rel_cfg, "enabled": True})
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    return get_scenario_status(scenario_name)


def load_registry(path: Path | None = None) -> dict[str, Any]:
    registry_path = path or REGISTRY_PATH
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    data = registry_path.read_text(encoding="utf-8")
    import json

    payload = json.loads(data)
    if not isinstance(payload, dict):
        raise ValueError("Registry must be a JSON object")
    if payload.get("version") != 1:
        raise ValueError("Registry version must be 1")
    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, list):
        raise ValueError("Registry scenarios must be an array")
    return payload


def list_enabled_scenarios() -> list[dict[str, str]]:
    payload = load_registry()
    out: list[dict[str, str]] = []
    for item in payload["scenarios"]:
        if not isinstance(item, dict):
            continue
        if not item.get("enabled", True):
            continue
        try:
            cfg = get_scenario_config(str(item.get("name", "")))
        except Exception:
            continue
        out.append(
            {
                "name": cfg.name,
                "display_name": cfg.display_name,
                "description": cfg.description,
            }
        )
    return out


def get_scenario_config(name: str) -> ScenarioConfig:
    if not name:
        raise ValueError("Scenario name is required")
    payload = load_registry()
    entries = payload["scenarios"]
    match: dict[str, Any] | None = None
    for item in entries:
        if not isinstance(item, dict):
            continue
        if str(item.get("name")) == name:
            match = item
            break
    if match is None:
        raise ValueError(f"Scenario '{name}' not found in registry")
    if not match.get("enabled", True):
        raise ValueError(f"Scenario '{name}' is disabled")

    config_ref = match.get("config")
    if not isinstance(config_ref, str) or not config_ref:
        raise ValueError(f"Scenario '{name}' has invalid config path")

    config_path = _resolve_repo_path(config_ref)
    if not config_path.exists():
        raise FileNotFoundError(f"Scenario config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Scenario config must be a mapping: {config_path}")

    backbone_type = str(raw.get("backbone_type") or "yolo_detection").strip().lower()
    requires_cv_fields = backbone_type == "yolo_detection"

    # For non-YOLO backbones base_model/weights/classes/postproc/hyperparams are optional —
    # the backbone implementation provides the logic.
    required = ["name", "display_name", "description", "dataset"]
    if requires_cv_fields:
        required += ["base_model", "classes", "postproc", "weights", "hyperparams"]
    _require_fields(raw, required, str(config_path))

    classes_raw = raw.get("classes", [])
    if not isinstance(classes_raw, list):
        classes_raw = []
    classes = [str(c) for c in classes_raw if isinstance(c, str)]

    hyperparams_raw = raw.get("hyperparams", {})
    hyperparams = dict(hyperparams_raw) if isinstance(hyperparams_raw, dict) else {}
    if requires_cv_fields:
        for key, value in YOLO_QUALITY_STOP_DEFAULTS.items():
            hyperparams.setdefault(key, value)

    if requires_cv_fields:
        if not classes:
            raise ValueError(f"Invalid classes in {config_path}")
        if not hyperparams:
            raise ValueError(f"Invalid hyperparams in {config_path}")

    backbone_config_raw = raw.get("backbone_config", {})
    backbone_config = dict(backbone_config_raw) if isinstance(backbone_config_raw, dict) else {}

    if backbone_type == "audio_recognition":
        dataset_path = ML_AUDIO_ROOT / str(raw["dataset"])
    elif backbone_type == "llm_fine_tuning":
        dataset_path = resolve_llm_jsonl_dataset_path(str(raw["dataset"]))
    elif backbone_type == "archival_ingestion":
        archive_root_ref = str(backbone_config.get("archive_storage_root") or "").strip()
        dataset_path = _resolve_repo_path(archive_root_ref) if archive_root_ref else REPO_ROOT
    else:
        dataset_path = MLOPS_ROOT / "datasets" / str(raw["dataset"])
    weights_ref = str(raw.get("weights") or "")
    weights_path = _resolve_repo_path(weights_ref) if weights_ref else Path("")

    return ScenarioConfig(
        name=str(raw["name"]),
        display_name=str(raw["display_name"]),
        description=str(raw["description"]),
        base_model=str(raw.get("base_model") or ""),
        dataset=str(raw["dataset"]),
        classes=classes,
        postproc=str(raw.get("postproc") or ""),
        weights=weights_ref,
        hyperparams=hyperparams,
        config_path=config_path,
        dataset_path=dataset_path,
        weights_path=weights_path,
        raw=dict(raw),
        backbone_type=backbone_type,
        backbone_config=backbone_config,
    )


def patch_scenario_backbone_config(scenario: str, patch: dict[str, Any]) -> dict[str, Any]:
    """Merge a patch dict into `backbone_config` and persist to the scenario YAML."""
    scen = sanitize_scenario_name(str(scenario or ""))
    cfg = get_scenario_config(scen)
    if not isinstance(patch, dict) or not patch:
        return get_scenario_status(scen)
    raw = dict(cfg.raw or {})
    existing = raw.get("backbone_config")
    backbone_cfg = dict(existing) if isinstance(existing, dict) else {}
    backbone_cfg.update(dict(patch))
    raw["backbone_config"] = backbone_cfg
    cfg.config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    return get_scenario_status(scen)


def get_scenario_ci_cd_policy(scenario: str) -> dict[str, Any]:
    cfg = get_scenario_config(scenario)
    from .ci_cd import normalize_policy

    raw_policy = cfg.raw.get("ci_cd") if isinstance(cfg.raw.get("ci_cd"), dict) else None
    return normalize_policy(
        raw_policy,
        hyperparams=cfg.hyperparams,
        legacy_default_enabled=False,
    )


def patch_scenario_ci_cd_policy(
    scenario: str,
    updates: Mapping[str, Any],
    *,
    reset: bool = False,
) -> dict[str, Any]:
    scen = sanitize_scenario_name(str(scenario or ""))
    cfg = get_scenario_config(scen)
    from .ci_cd import normalize_policy

    raw = dict(cfg.raw or {})
    existing = raw.get("ci_cd") if isinstance(raw.get("ci_cd"), dict) else {}
    merged = dict(updates or {}) if reset else {**dict(existing or {}), **dict(updates or {})}
    raw["ci_cd"] = normalize_policy(
        merged,
        hyperparams=cfg.hyperparams,
        legacy_default_enabled=False,
    )
    cfg.config_path.write_text(
        yaml.safe_dump(raw, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    try:
        with _SCENARIO_STATUS_CACHE_LOCK:
            _SCENARIO_STATUS_CACHE.pop(scen, None)
    except Exception:
        pass
    return get_scenario_status(scen)


# ---------- Status + dataset + verified helpers ----------


WEIGHTS_MIN_BYTES = 32


def _weights_ready(cfg: ScenarioConfig) -> bool:
    try:
        return cfg.weights_path.exists() and cfg.weights_path.stat().st_size >= WEIGHTS_MIN_BYTES
    except Exception:
        return False


def _normalize_dataset_split(split: str) -> str:
    value = str(split or "").strip().lower()
    if value not in DATASET_SPLITS:
        raise ValueError(f"Invalid dataset split '{split}'. Expected one of: {', '.join(DATASET_SPLITS)}")
    return value


def sanitize_library_dataset_slug(slug: str) -> str:
    name = str(slug or "").strip()
    if not name or ".." in name:
        raise ValueError("Invalid library dataset name.")
    if any(sep in name for sep in ("/", "\\")):
        raise ValueError("Invalid library dataset name.")
    return name


def ensure_database_root() -> Path:
    DATABASE_ROOT.mkdir(parents=True, exist_ok=True)
    return DATABASE_ROOT


def ensure_ml_audio_root() -> Path:
    ML_AUDIO_ROOT.mkdir(parents=True, exist_ok=True)
    return ML_AUDIO_ROOT


def is_ml_audio_dataset_path(path: Path) -> bool:
    try:
        path.resolve().relative_to(ML_AUDIO_ROOT.resolve())
        return True
    except Exception:
        return False


def _split_first_yolo_roots(base: Path) -> list[tuple[str, str, Path, Path, Path]]:
    """Return split-first YOLO roots.

    Handles datasets laid out as:
      train/images/*.jpg
      train/labels/*.txt
      valid/images/*.jpg
      valid/labels/*.txt
    """
    out: list[tuple[str, str, Path, Path, Path]] = []
    for split_dir, canon in _IMAGEFOLDER_SPLIT_DIRS:
        split_root = base / split_dir
        images_root = split_root / "images"
        labels_root = split_root / "labels"
        if not (images_root.is_dir() and labels_root.is_dir()):
            continue
        out.append((split_dir, canon, split_root, images_root, labels_root))
    return out


def detect_library_dataset_format(dataset_root: Path) -> str:
    """Detect dataset layout for the CV Ops dataset library.

    - `yolo_detection`: YOLO-style detection dataset (images/ + labels/, optionally split by train/val).
    - `imagefolder_classification`: torchvision ImageFolder-style classification dataset (split/class/image.jpg).
    - `csv_tabular`: tabular dataset — folder contains at least one .csv file at root level.
    - `face_csv`: face recognition dataset with a root-level CSV mapping filenames to identities.
    - `llm_instruction_jsonl`: JSONL instruction/chat rows for local LLM fine-tuning.
    """
    base = dataset_root.resolve()
    if base.is_file():
        if base.suffix.lower() in DATASET_TEXT_SUFFIXES:
            return LIBRARY_DATASET_FORMAT_LLM_JSONL
        if base.suffix.lower() == ".csv":
            return LIBRARY_DATASET_FORMAT_CSV
        return LIBRARY_DATASET_FORMAT_UNKNOWN
    if (base / "images").is_dir():
        return LIBRARY_DATASET_FORMAT_YOLO
    if _split_first_yolo_roots(base):
        return LIBRARY_DATASET_FORMAT_YOLO

    # Face CSV: root-level CSV with (id,label) plus an image folder such as Faces/.
    try:
        for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
            if not (p.is_file() and p.suffix.lower() == ".csv"):
                continue
            try:
                head = p.read_text(encoding="utf-8", errors="replace")[:2048]
            except Exception:
                head = ""
            header = (head.splitlines()[0] if head else "").strip().lower()
            if "id" in header and "label" in header:
                faces_dir = base / "Faces"
                originals_dir = base / "Original Images"
                if faces_dir.is_dir() or originals_dir.is_dir():
                    return LIBRARY_DATASET_FORMAT_FACE_CSV
    except Exception:
        pass

    # Split class folders: train/<class>/..., valid/<class>/..., etc.
    for split_dir, _canon in _IMAGEFOLDER_SPLIT_DIRS:
        root = base / split_dir
        if not root.is_dir():
            continue
        try:
            class_dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except Exception:
            return LIBRARY_DATASET_FORMAT_IMAGEFOLDER
        has_audio = False
        has_image = False
        for d in class_dirs[:50]:
            try:
                for p in d.iterdir():
                    if not p.is_file():
                        continue
                    suffix = p.suffix.lower()
                    has_audio = has_audio or suffix in DATASET_AUDIO_SUFFIXES
                    has_image = has_image or suffix in DATASET_IMAGE_SUFFIXES
                    if has_audio or has_image:
                        break
            except Exception:
                continue
        if has_audio and not has_image:
            return LIBRARY_DATASET_FORMAT_AUDIOFOLDER
        if class_dirs:
            return LIBRARY_DATASET_FORMAT_IMAGEFOLDER

    # Root class-folder fallback: <class>/sample.ext
    try:
        child_dirs = [p for p in base.iterdir() if p.is_dir() and not p.name.startswith(".")]
    except Exception:
        child_dirs = []
    for d in child_dirs[:50]:
        try:
            if any(p.is_file() and p.suffix.lower() in DATASET_IMAGE_SUFFIXES for p in d.iterdir()):
                return LIBRARY_DATASET_FORMAT_IMAGEFOLDER
        except Exception:
            continue
    for d in child_dirs[:50]:
        try:
            if any(p.is_file() and p.suffix.lower() in DATASET_AUDIO_SUFFIXES for p in d.iterdir()):
                return LIBRARY_DATASET_FORMAT_AUDIOFOLDER
        except Exception:
            continue

    # LLM instruction tuning: one or more JSONL files directly in the folder root.
    try:
        if any(p.suffix.lower() in DATASET_TEXT_SUFFIXES and p.is_file() for p in base.iterdir()):
            return LIBRARY_DATASET_FORMAT_LLM_JSONL
    except Exception:
        pass

    # CSV/tabular: any .csv file directly in the folder root.
    try:
        if any(p.suffix.lower() == ".csv" and p.is_file() for p in base.iterdir()):
            return LIBRARY_DATASET_FORMAT_CSV
    except Exception:
        pass

    return LIBRARY_DATASET_FORMAT_UNKNOWN


def list_library_dataset_names() -> list[str]:
    root = ensure_database_root()
    names: list[str] = []
    try:
        for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if p.is_dir() and not p.name.startswith("."):
                names.append(p.name)
    except Exception:
        return []
    audio_root = ensure_ml_audio_root()
    try:
        for p in sorted(audio_root.iterdir(), key=lambda x: x.name.lower()):
            if not (p.is_dir() and not p.name.startswith(".")):
                continue
            if p.name in names:
                continue
            names.append(p.name)
    except Exception:
        pass
    # Also include image-like datasets living directly under mlops/datasets/ (common when
    # operators unpack external datasets there without creating database/ entries).
    extra_root = (MLOPS_ROOT / "datasets").resolve()
    if extra_root.exists():
        try:
            for p in sorted(extra_root.iterdir(), key=lambda x: x.name.lower()):
                if not (p.is_dir() and not p.name.startswith(".")):
                    continue
                if p.name in names:
                    continue
                fmt = detect_library_dataset_format(p)
                if dataset_category(fmt) in {DATASET_CATEGORY_IMAGE, DATASET_CATEGORY_TEXT}:
                    names.append(p.name)
        except Exception:
            pass
    return names


def list_tabular_dataset_entries() -> list[dict[str, Any]]:
    """Return metadata for CSV files under mlops/datasets/.

    Each entry: {"name": str, "path": str, "size_bytes": int, "category": "tabular"}
    The "name" is the filename without extension; "path" is the path relative to REPO_ROOT.
    """
    entries: list[dict[str, Any]] = []
    root = TABULAR_DATASETS_ROOT
    if not root.exists():
        return entries
    try:
        for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file() and p.suffix.lower() == ".csv" and not p.name.startswith("."):
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                try:
                    rel = str(p.relative_to(REPO_ROOT))
                except Exception:
                    rel = str(p)
                entries.append({
                    "name": p.stem,
                    "filename": p.name,
                    "path": rel,
                    "size_bytes": size,
                    "category": DATASET_CATEGORY_TABULAR,
                    "format": LIBRARY_DATASET_FORMAT_CSV,
                })
    except Exception:
        pass
    return entries


def list_text_dataset_entries() -> list[dict[str, Any]]:
    """Return metadata for JSONL instruction files under mlops/datasets/."""
    entries: list[dict[str, Any]] = []
    root = TABULAR_DATASETS_ROOT
    if not root.exists():
        return entries
    try:
        for p in sorted(root.iterdir(), key=lambda x: x.name.lower()):
            if p.is_file() and p.suffix.lower() in DATASET_TEXT_SUFFIXES and not p.name.startswith("."):
                try:
                    size = p.stat().st_size
                except Exception:
                    size = 0
                try:
                    rel = str(p.relative_to(REPO_ROOT))
                except Exception:
                    rel = str(p)
                entries.append({
                    "name": p.stem,
                    "filename": p.name,
                    "path": rel,
                    "size_bytes": size,
                    "category": DATASET_CATEGORY_TEXT,
                    "format": LIBRARY_DATASET_FORMAT_LLM_JSONL,
                })
    except Exception:
        pass
    return entries


def resolve_library_dataset_path(slug: str) -> Path:
    name = sanitize_library_dataset_slug(slug)
    root = ensure_database_root().resolve()
    target = (DATABASE_ROOT / name).resolve()
    try:
        target.relative_to(root)
    except Exception as exc:
        raise ValueError("Invalid library dataset path.") from exc
    if target.is_dir():
        return target
    audio = (ensure_ml_audio_root() / name).resolve()
    try:
        audio.relative_to(ML_AUDIO_ROOT.resolve())
    except Exception:
        raise ValueError("Invalid library dataset path.") from None
    if audio.is_dir():
        return audio
    # Fallback: allow datasets that live in mlops/datasets/<slug> (folder) or
    # mlops/datasets/<slug>.csv (single-file tabular library drop).
    ds_root = (MLOPS_ROOT / "datasets").resolve()
    alt = (ds_root / name).resolve()
    try:
        alt.relative_to(REPO_ROOT.resolve())
    except Exception:
        raise ValueError(f"Library dataset '{name}' does not exist under database/.") from None
    if alt.is_dir():
        return alt
    csv_leaf = (ds_root / f"{name}.csv").resolve()
    try:
        csv_leaf.relative_to(ds_root)
    except Exception:
        raise ValueError(f"Library dataset '{name}' does not exist under database/.") from None
    if csv_leaf.is_file():
        return csv_leaf
    raise ValueError(f"Library dataset '{name}' does not exist under database/.")


def resolve_llm_jsonl_dataset_path(ref: str) -> Path:
    """Resolve a JSONL instruction dataset from a slug, repo path, or absolute path."""
    raw = str(ref or "").strip()
    if not raw:
        raise ValueError("JSONL instruction dataset is required.")
    candidates: list[Path] = []
    p = Path(raw).expanduser()
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            REPO_ROOT / p,
            MLOPS_ROOT / "datasets" / raw,
            DATABASE_ROOT / raw,
        ])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate
        if not resolved.exists():
            continue
        fmt = detect_library_dataset_format(resolved)
        if fmt == LIBRARY_DATASET_FORMAT_LLM_JSONL:
            return resolved
    raise ValueError(f"JSONL instruction dataset not found: {raw}")


def resolve_audio_library_dataset_path(slug: str) -> Path:
    """Resolve an AudioFolder dataset from the dedicated ML audio asset root."""
    name = sanitize_library_dataset_slug(slug)
    root = ensure_ml_audio_root().resolve()
    target = (ML_AUDIO_ROOT / name).resolve()
    try:
        target.relative_to(root)
    except Exception as exc:
        raise ValueError("Invalid audio dataset path.") from exc
    if not target.is_dir():
        raise ValueError(f"Audio dataset '{name}' does not exist under assets/ml_audio/.")
    return target


def pick_unique_library_dataset_slug(preferred: str) -> str:
    """Pick a non-existing dataset slug under `database/`."""
    base = sanitize_library_dataset_slug(preferred)
    root = ensure_database_root()
    audio_root = ensure_ml_audio_root()
    if not (root / base).exists() and not (audio_root / base).exists():
        return base
    for i in range(1, 1000):
        candidate = f"{base}-{i:02d}"
        if not (root / candidate).exists() and not (audio_root / candidate).exists():
            return candidate
    raise ValueError(f"Could not find an available dataset name for '{base}'.")


def pick_unique_audio_dataset_slug(preferred: str) -> str:
    """Pick a non-existing audio dataset slug under `assets/ml_audio/`."""
    base = sanitize_library_dataset_slug(preferred)
    root = ensure_ml_audio_root()
    database_root = ensure_database_root()
    if not (root / base).exists() and not (database_root / base).exists():
        return base
    for i in range(1, 1000):
        candidate = f"{base}-{i:02d}"
        if not (root / candidate).exists() and not (database_root / candidate).exists():
            return candidate
    raise ValueError(f"Could not find an available audio dataset name for '{base}'.")


def create_library_dataset_root(slug: str) -> Path:
    """Create a dataset directory under `database/` (must not already exist)."""
    name = sanitize_library_dataset_slug(slug)
    root = ensure_database_root()
    target = root / name
    if target.exists():
        raise ValueError(f"Library dataset '{name}' already exists under database/.")
    target.mkdir(parents=True, exist_ok=False)
    return target


def create_yolo_detection_dataset_template(
    preferred_slug: str,
    *,
    classes: list[str] | None = None,
    unique_slug: bool = True,
) -> dict[str, Any]:
    """Create an empty YOLO detection dataset under ``database/<slug>/``.

    Layout matches :func:`emit_yolo_dataset` and training expectations:
    ``images/{train,val}/``, ``labels/{train,val}/``, ``classes.txt``, ``data.yaml``.
    """
    raw_classes = [str(c).strip() for c in (classes or []) if str(c).strip()]
    if not raw_classes:
        raw_classes = ["object"]

    base = sanitize_library_dataset_slug(preferred_slug)
    if not base:
        raise ValueError("Dataset name is required.")
    if unique_slug:
        slug = pick_unique_library_dataset_slug(base)
    else:
        slug = base

    root = create_library_dataset_root(slug)
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)

    (root / "classes.txt").write_text("\n".join(raw_classes) + "\n", encoding="utf-8")

    names_block = "\n".join(f"  {i}: {nm}" for i, nm in enumerate(raw_classes))
    data_body = "\n".join(
        [
            f"path: {root.resolve().as_posix()}",
            "train: images/train",
            "val: images/val",
            "names:",
            names_block,
            "",
        ]
    )
    (root / "data.yaml").write_text(data_body, encoding="utf-8")

    try:
        _ensure_mlops_dataset_link(slug)
    except Exception:
        pass

    return {
        "slug": slug,
        "path": str(root.resolve()),
        "format": LIBRARY_DATASET_FORMAT_YOLO,
        "category": DATASET_CATEGORY_IMAGE,
        "classes": raw_classes,
    }


def _imagefolder_label_candidates(
    *,
    src: Path,
    img: Path,
    split_dir: str,
    canon_split: str,
    class_name: str,
    rel_under_class: Path,
) -> list[Path]:
    """Return existing-label candidates for an ImageFolder image.

    Supports simple sidecars:
      train/class/foo.jpg -> train/class/foo.txt

    And YOLO-style mirrors carried alongside an ImageFolder source:
      labels/train/class/foo.txt
      labels/valid/class/foo.txt
      labels/val/class/foo.txt
    """
    seen: set[Path] = set()
    out: list[Path] = []

    def _add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(path)

    _add(img.with_suffix(".txt"))
    _add((src / "labels" / split_dir / class_name / rel_under_class).with_suffix(".txt"))
    _add((src / "labels" / canon_split / class_name / rel_under_class).with_suffix(".txt"))
    return out


def _imagefolder_existing_label_path(
    *,
    src: Path,
    img: Path,
    split_dir: str,
    canon_split: str,
    class_name: str,
    rel_under_class: Path,
) -> Optional[Path]:
    for candidate in _imagefolder_label_candidates(
        src=src,
        img=img,
        split_dir=split_dir,
        canon_split=canon_split,
        class_name=class_name,
        rel_under_class=rel_under_class,
    ):
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _float_tokens(parts: list[str]) -> Optional[list[float]]:
    out: list[float] = []
    for part in parts:
        try:
            out.append(float(part))
        except Exception:
            return None
    return out


def _image_size(path: Path) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None


def _coord_values_to_yolo_xywh(values: list[float], *, image_path: Path) -> Optional[list[float]]:
    if len(values) != 4:
        return None
    # Most sidecar-only exports already store YOLO-normalized xc/yc/w/h.
    if all(0.0 <= v <= 1.0 for v in values):
        return values

    # Pixel xyxy fallback for tools that export raw box corners without class ids.
    size = _image_size(image_path)
    if size is None:
        return None
    w, h = size
    if w <= 0 or h <= 0:
        return None
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        return None
    xc = ((x1 + x2) / 2.0) / float(w)
    yc = ((y1 + y2) / 2.0) / float(h)
    bw = (x2 - x1) / float(w)
    bh = (y2 - y1) / float(h)
    return [xc, yc, bw, bh]


def _format_yolo_line(class_id: int, coords: list[float]) -> str:
    return f"{int(class_id)} " + " ".join(f"{float(v):.6g}" for v in coords[:4])


def _normalize_imported_label_text(
    raw: str,
    *,
    image_path: Path,
    class_name: str,
    class_to_id: dict[str, int],
) -> tuple[str, int, int]:
    """Normalize imported labels to YOLO detection lines.

    Accepts canonical YOLO lines, coord-only sidecars, and class-name-prefixed
    lines. Coord-only lines get the class id from the ImageFolder directory.
    Returns (text, normalized_coord_only_lines, skipped_invalid_lines).
    """
    class_lookup = {str(k).strip().lower(): int(v) for k, v in class_to_id.items()}
    folder_class_id = int(class_to_id.get(class_name, 0))
    out: list[str] = []
    normalized = 0
    skipped = 0

    for raw_line in str(raw or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 5:
            try:
                int(float(parts[0]))
                coords = _float_tokens(parts[1:5])
            except Exception:
                coords = None
            if coords is not None:
                out.append(" ".join(parts))
                continue

            label_key = str(parts[0]).strip().lower()
            coords = _float_tokens(parts[1:5])
            if coords is not None:
                class_id = class_lookup.get(label_key, folder_class_id)
                yolo_coords = _coord_values_to_yolo_xywh(coords, image_path=image_path)
                if yolo_coords is not None:
                    out.append(_format_yolo_line(class_id, yolo_coords))
                    normalized += 1
                    continue

        if len(parts) == 4:
            coords = _float_tokens(parts)
            if coords is not None:
                yolo_coords = _coord_values_to_yolo_xywh(coords, image_path=image_path)
                if yolo_coords is not None:
                    out.append(_format_yolo_line(folder_class_id, yolo_coords))
                    normalized += 1
                    continue

        skipped += 1

    return "\n".join(out) + ("\n" if out else ""), normalized, skipped


def convert_imagefolder_to_yolo(
    src_root: Path,
    dest_root: Path,
    *,
    mode: str = "full_frame",
    include_test: bool = True,
) -> dict[str, Any]:
    """Convert ImageFolder classification dataset into a YOLO-style detection dataset.

    Conversion writes:
      dest/images/{train,val}/<ClassName>/...<img>
      dest/labels/{train,val}/<ClassName>/...<img>.txt

    `mode`:
      - full_frame: one box per image (class_id 0.5 0.5 1.0 1.0)
      - empty: copy images only so the editor can create labels on save
      - import_labels: copy existing YOLO .txt labels from source sidecars or labels/ mirrors
    """
    import os
    import shutil

    src = src_root.resolve()
    dest = dest_root.resolve()
    if detect_library_dataset_format(src) != LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
        raise ValueError("Source dataset is not ImageFolder classification format.")
    if dest.exists():
        raise ValueError(f"Destination already exists: {dest}")
    mode_norm = str(mode or "").strip().lower()
    if mode_norm not in {"full_frame", "empty", "import_labels"}:
        raise ValueError("mode must be 'full_frame', 'empty', or 'import_labels'")

    # Gather classes from available splits.
    class_names: set[str] = set()
    for split_dir, _canon in _IMAGEFOLDER_SPLIT_DIRS:
        if split_dir == "test" and not include_test:
            continue
        root = src / split_dir
        if not root.is_dir():
            continue
        try:
            for p in root.iterdir():
                if p.is_dir() and not p.name.startswith("."):
                    class_names.add(p.name)
        except Exception:
            continue
    class_list = sorted(class_names, key=lambda s: s.lower())
    class_to_id = {name: idx for idx, name in enumerate(class_list)}

    dest.mkdir(parents=True, exist_ok=False)
    # Persist the class order so the editor (and future scenario configs) can
    # consistently map names -> IDs.
    try:
        (dest / "classes.txt").write_text("\n".join(class_list) + ("\n" if class_list else ""), encoding="utf-8")
    except Exception:
        pass
    images_out = dest / "images"
    labels_out = dest / "labels"
    (images_out / "train").mkdir(parents=True, exist_ok=True)
    (images_out / "val").mkdir(parents=True, exist_ok=True)
    (labels_out / "train").mkdir(parents=True, exist_ok=True)
    (labels_out / "val").mkdir(parents=True, exist_ok=True)

    converted = 0
    generated_labels = 0
    imported_labels = 0
    normalized_label_lines = 0
    invalid_label_lines = 0
    missing_labels = 0
    split_counts: dict[str, int] = {"train": 0, "val": 0}
    errors: list[str] = []

    def _write_label(
        label_path: Path,
        *,
        img: Path,
        split_dir: str,
        canon_split: str,
        class_name: str,
        rel_under_class: Path,
    ) -> str:
        nonlocal normalized_label_lines, invalid_label_lines
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if mode_norm == "empty":
            # For annotation workflows, avoid touching the filesystem for every image;
            # the editor will create (possibly empty) label files on save.
            return "empty"
        if mode_norm == "import_labels":
            source_label = _imagefolder_existing_label_path(
                src=src,
                img=img,
                split_dir=split_dir,
                canon_split=canon_split,
                class_name=class_name,
                rel_under_class=rel_under_class,
            )
            if source_label is None:
                return "missing"
            raw = source_label.read_text(encoding="utf-8", errors="replace")
            text, normalized, skipped = _normalize_imported_label_text(
                raw,
                image_path=img,
                class_name=class_name,
                class_to_id=class_to_id,
            )
            normalized_label_lines += normalized
            invalid_label_lines += skipped
            label_path.write_text(text, encoding="utf-8")
            return "imported"
        class_id = class_to_id.get(class_name, 0)
        label_path.write_text(f"{class_id} 0.5 0.5 1.0 1.0\n", encoding="utf-8")
        return "generated"

    def _link_or_copy(src_path: Path, dst_path: Path) -> None:
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src_path, dst_path)
            return
        except Exception:
            shutil.copy2(src_path, dst_path)

    for split_dir, canon in _IMAGEFOLDER_SPLIT_DIRS:
        if split_dir == "test" and not include_test:
            continue
        split_root = src / split_dir
        if not split_root.is_dir():
            continue
        # Map ImageFolder splits into YOLO splits.
        dest_split = "train" if canon == "train" else "val"
        try:
            class_dirs = [
                p for p in split_root.iterdir() if p.is_dir() and not p.name.startswith(".")
            ]
        except Exception:
            class_dirs = []
        for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
            class_name = class_dir.name
            for img in sorted(class_dir.rglob("*")):
                if not img.is_file() or img.suffix.lower() not in DATASET_IMAGE_SUFFIXES:
                    continue
                try:
                    rel_under_class = img.relative_to(class_dir)
                except Exception:
                    rel_under_class = Path(img.name)
                dst_img = images_out / dest_split / class_name / rel_under_class
                dst_label = (labels_out / dest_split / class_name / rel_under_class).with_suffix(".txt")
                try:
                    _link_or_copy(img, dst_img)
                    label_result = _write_label(
                        dst_label,
                        img=img,
                        split_dir=split_dir,
                        canon_split=canon,
                        class_name=class_name,
                        rel_under_class=rel_under_class,
                    )
                    if label_result == "generated":
                        generated_labels += 1
                    elif label_result == "imported":
                        imported_labels += 1
                    elif label_result == "missing":
                        missing_labels += 1
                    converted += 1
                    split_counts[dest_split] = split_counts.get(dest_split, 0) + 1
                except Exception as exc:
                    errors.append(f"{img}: {exc}")
                    if len(errors) >= 12:
                        break
            if len(errors) >= 12:
                break
        if len(errors) >= 12:
            break

    return {
        "mode": mode_norm,
        "converted": converted,
        "generated_labels": generated_labels,
        "imported_labels": imported_labels,
        "normalized_label_lines": normalized_label_lines,
        "invalid_label_lines": invalid_label_lines,
        "missing_labels": missing_labels,
        "split_counts": split_counts,
        "classes": class_list,
        "class_to_id": class_to_id,
        "errors": errors,
        "dest_root": str(dest),
    }


def list_dataset_entries_at(base: Path) -> list[dict[str, Any]]:
    base_resolved = base.resolve()
    if not base_resolved.is_dir():
        raise ValueError("Dataset root is not a directory.")
    images_root = base_resolved / "images"
    labels_root = base_resolved / "labels"
    out: list[dict[str, Any]] = []

    split_first_roots = _split_first_yolo_roots(base_resolved)
    if not images_root.exists() and split_first_roots:
        roots = [
            (canon, image_root, label_root)
            for _split_dir, canon, _split_root, image_root, label_root in split_first_roots
        ]
    else:
        # Some YOLO datasets include additional split folders (e.g. test/valid/custom). For preview/inspection,
        # show all top-level folders under images/ whenever the dataset looks "split-like".
        split_dirs: list[Path] = []
        try:
            split_dirs = [p for p in images_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except Exception:
            split_dirs = []
        split_dir_names = {p.name for p in split_dirs}
        split_markers = set(DATASET_SPLITS) | {"valid", "test"}
        has_split_layout = bool(split_dir_names & split_markers)

        if has_split_layout and split_dirs:
            roots = [
                (p.name, p, labels_root / p.name)
                for p in sorted(split_dirs, key=lambda x: x.name.lower())
            ]
        elif not images_root.exists():
            # Loose database folder: images live directly under this folder or
            # nested operator-created subfolders. Sidecar labels mirror the
            # image path with a .txt suffix in the same folder.
            roots = [("", base_resolved, base_resolved)]
        else:
            roots = [("", images_root, labels_root)]

    for split, image_root, label_root in roots:
        if not image_root.exists():
            continue
        split_name = split or "root"
        label_dir_exists = label_root.exists()
        label_count = 0
        if label_dir_exists:
            try:
                label_count = len([p for p in label_root.rglob("*.txt") if p.is_file()])
            except Exception:
                label_count = 0
        for p in sorted(image_root.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in DATASET_IMAGE_SUFFIXES:
                continue
            # Mirror nested paths: images/train/foo/bar.jpg -> labels/train/foo/bar.txt
            try:
                rel_under_split = p.relative_to(image_root)
            except Exception:
                rel_under_split = Path(p.name)
            label_path = (label_root / rel_under_split).with_suffix(".txt")
            has_label = label_path.exists()
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            relative_path = p.relative_to(base_resolved).as_posix()
            display_rel = rel_under_split.as_posix()
            out.append(
                {
                    "name": p.name,
                    "stem": p.stem,
                    "path": str(p),
                    "relative_path": relative_path,
                    "split": split_name,
                    "size": size,
                    "label_path": str(label_path) if has_label else "",
                    "label_relative_path": label_path.relative_to(base_resolved).as_posix()
                    if has_label
                    else "",
                    "has_label": has_label,
                    "display_name": f"{split_name}/{display_rel}" if split_name != "root" else display_rel,
                    "label_dir_exists": label_dir_exists,
                    "split_label_count": label_count,
                }
            )
    return out


def list_dataset_folders_at(base: Path) -> list[dict[str, Any]]:
    base_resolved = base.resolve()
    if not base_resolved.is_dir():
        raise ValueError("Dataset root is not a directory.")
    folders: list[dict[str, Any]] = []
    try:
        candidates = sorted(
            [p for p in base_resolved.rglob("*") if p.is_dir()],
            key=lambda p: p.relative_to(base_resolved).as_posix().lower(),
        )
    except Exception:
        candidates = []
    for p in candidates[:5000]:
        try:
            rel = p.relative_to(base_resolved).as_posix()
        except Exception:
            continue
        parts = Path(rel).parts
        if not parts or any(part.startswith(".") for part in parts):
            continue
        if any(part in {"__pycache__", "labels"} for part in parts):
            continue
        direct_image_count = 0
        descendant_image_count = 0
        try:
            for child in p.iterdir():
                if child.is_file() and child.suffix.lower() in DATASET_IMAGE_SUFFIXES:
                    direct_image_count += 1
        except Exception:
            direct_image_count = 0
        try:
            descendant_image_count = sum(
                1
                for child in p.rglob("*")
                if child.is_file() and child.suffix.lower() in DATASET_IMAGE_SUFFIXES
            )
        except Exception:
            descendant_image_count = direct_image_count
        folders.append(
            {
                "path": rel,
                "name": p.name,
                "direct_image_count": direct_image_count,
                "image_count": descendant_image_count,
            }
        )
    return folders


def list_imagefolder_entries_at(base: Path) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """List entries for ImageFolder-style classification datasets.

    Expected layout:
      train/<ClassName>/*.jpg
      valid/<ClassName>/*.jpg   (or val/)
      test/<ClassName>/*.jpg    (optional)
    """
    base_resolved = base.resolve()
    if not base_resolved.is_dir():
        raise ValueError("Dataset root is not a directory.")
    out: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    classes: set[str] = set()

    for split_dir, canon_split in _IMAGEFOLDER_SPLIT_DIRS:
        split_root = base_resolved / split_dir
        if not split_root.is_dir():
            continue
        try:
            class_dirs = [
                p for p in split_root.iterdir() if p.is_dir() and not p.name.startswith(".")
            ]
        except Exception:
            class_dirs = []
        for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
            classes.add(class_dir.name)
            for img in sorted(class_dir.rglob("*")):
                if not img.is_file() or img.suffix.lower() not in DATASET_IMAGE_SUFFIXES:
                    continue
                try:
                    size = img.stat().st_size
                except Exception:
                    size = 0
                rel = img.relative_to(base_resolved).as_posix()
                try:
                    rel_under_split = img.relative_to(split_root).as_posix()
                except Exception:
                    rel_under_split = img.name
                try:
                    rel_under_class = img.relative_to(class_dir)
                except Exception:
                    rel_under_class = Path(img.name)
                detection_label = _imagefolder_existing_label_path(
                    src=base_resolved,
                    img=img,
                    split_dir=split_dir,
                    canon_split=canon_split,
                    class_name=class_dir.name,
                    rel_under_class=rel_under_class,
                )
                display_name = f"{canon_split}/{rel_under_split}"
                out.append(
                    {
                        "name": img.name,
                        "stem": img.stem,
                        "path": str(img),
                        "relative_path": rel,
                        "split": canon_split,
                        "size": size,
                        "has_label": True,  # label is encoded by the folder name
                        "has_detection_label": detection_label is not None,
                        "detection_label_path": str(detection_label) if detection_label is not None else "",
                        "detection_label_relative_path": detection_label.relative_to(base_resolved).as_posix()
                        if detection_label is not None
                        else "",
                        "display_name": display_name,
                        "classification_label": class_dir.name,
                    }
                )
                split_counts[canon_split] = split_counts.get(canon_split, 0) + 1

    class_list = sorted(classes, key=lambda s: s.lower())
    return out, split_counts, class_list


def list_audiofolder_entries_at(base: Path) -> tuple[list[dict[str, Any]], dict[str, int], list[str]]:
    """List entries for AudioFolder-style classification datasets."""
    base_resolved = base.resolve()
    if not base_resolved.is_dir():
        raise ValueError("Dataset root is not a directory.")
    out: list[dict[str, Any]] = []
    split_counts: dict[str, int] = {}
    classes: set[str] = set()

    split_roots: list[tuple[str, Path]] = []
    for split_dir, canon_split in _IMAGEFOLDER_SPLIT_DIRS:
        split_root = base_resolved / split_dir
        if split_root.is_dir():
            split_roots.append((canon_split, split_root))
    if not split_roots:
        split_roots.append(("root", base_resolved))

    for canon_split, split_root in split_roots:
        try:
            class_dirs = [p for p in split_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except Exception:
            class_dirs = []
        for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
            classes.add(class_dir.name)
            for audio in sorted(class_dir.rglob("*")):
                if not audio.is_file() or audio.suffix.lower() not in DATASET_AUDIO_SUFFIXES:
                    continue
                try:
                    size = audio.stat().st_size
                except Exception:
                    size = 0
                try:
                    rel = audio.relative_to(base_resolved).as_posix()
                except Exception:
                    rel = audio.name
                try:
                    rel_under_split = audio.relative_to(split_root).as_posix()
                except Exception:
                    rel_under_split = audio.name
                out.append(
                    {
                        "name": audio.name,
                        "stem": audio.stem,
                        "path": str(audio),
                        "relative_path": rel,
                        "split": canon_split,
                        "size": size,
                        "has_label": True,
                        "display_name": f"{canon_split}/{rel_under_split}" if canon_split != "root" else rel_under_split,
                        "classification_label": class_dir.name,
                    }
                )
                split_counts[canon_split] = split_counts.get(canon_split, 0) + 1

    return out, split_counts, sorted(classes, key=lambda s: s.lower())


def list_llm_jsonl_entries_at(base: Path) -> tuple[list[dict[str, Any]], int]:
    """List JSONL instruction files and count non-empty rows."""
    root = base.resolve()
    if root.is_file():
        files = [root] if root.suffix.lower() in DATASET_TEXT_SUFFIXES else []
        rel_base = root.parent
    else:
        rel_base = root
        try:
            files = [
                p for p in sorted(root.iterdir(), key=lambda x: x.name.lower())
                if p.is_file() and p.suffix.lower() in DATASET_TEXT_SUFFIXES and not p.name.startswith(".")
            ]
        except Exception:
            files = []

    items: list[dict[str, Any]] = []
    total_rows = 0
    for path in files:
        rows = 0
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for raw in f:
                    if raw.strip():
                        rows += 1
                    if rows >= 1_000_000:
                        break
        except Exception:
            rows = 0
        total_rows += rows
        try:
            rel = path.relative_to(rel_base).as_posix()
        except Exception:
            rel = path.name
        try:
            size = path.stat().st_size
        except Exception:
            size = 0
        items.append({
            "name": path.name,
            "path": str(path),
            "relative_path": rel,
            "size": size,
            "row_count": rows,
            "training_ready": rows > 0,
        })
    return items, total_rows


def list_dataset_entries(scenario: str) -> list[dict[str, Any]]:
    cfg = get_scenario_config(scenario)
    return list_dataset_entries_at(cfg.dataset_path)


def list_dataset_images(scenario: str) -> list[Path]:
    return [Path(str(entry.get("path") or "")) for entry in list_dataset_entries(scenario)]


def resolve_dataset_image_path_at(dataset_root: Path, relative_path: str) -> Optional[Path]:
    root = dataset_root.resolve()
    target = str(relative_path or "").strip().lstrip("/")
    if not target:
        return None
    path = (root / target).resolve()
    try:
        path.relative_to(root)
    except Exception:
        return None
    if not path.exists() or not path.is_file() or path.suffix.lower() not in DATASET_IMAGE_SUFFIXES:
        return None
    return path


def resolve_dataset_image_path(scenario: str, relative_path: str) -> Optional[Path]:
    cfg = get_scenario_config(scenario)
    return resolve_dataset_image_path_at(cfg.dataset_path, relative_path)


def resolve_dataset_label_path(image_path: Path) -> Optional[Path]:
    """Resolve corresponding label path for an image within a YOLO-style dataset.

    Supports nested paths under split folders by mirroring the relative path under `images/`:
      images/train/foo/bar.jpg -> labels/train/foo/bar.txt
      images/foo/bar.jpg       -> labels/foo/bar.txt

    Some converted datasets also contain a class/folder named `images`, e.g.
      images/train/images/foo.jpg -> labels/train/images/foo.txt
    """
    ip = image_path.resolve()
    candidates: list[Path] = []
    for parent in ip.parents:
        if parent.name != "images":
            continue
        dataset_root = parent.parent
        try:
            rel_under_images = ip.relative_to(parent)
        except Exception:
            continue
        if not rel_under_images.parts:
            continue
        first = rel_under_images.parts[0]
        if first in DATASET_SPLITS:
            rel_rest = Path(*rel_under_images.parts[1:])
            if not rel_rest.parts:
                continue
            candidates.append((dataset_root / "labels" / first / rel_rest).with_suffix(".txt"))
        else:
            candidates.append((dataset_root / "labels" / rel_under_images).with_suffix(".txt"))

    if not candidates:
        return None
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Prefer the outermost images/ root for missing labels so converted
    # images/train/images/foo.jpg datasets write to labels/train/images/foo.txt.
    return candidates[-1]


_INVENTORY_EXT_NONE = "(none)"


def _safe_rel_under_root(root: Path, relative_dir: str) -> Path:
    base = root.resolve()
    rel = str(relative_dir or "").strip().lstrip("/")
    if not rel:
        return base
    target = (base / rel).resolve()
    try:
        target.relative_to(base)
    except Exception:
        raise ValueError("relative_dir escapes dataset root") from None
    return target


def inventory_folder_types_at(
    dataset_root: Path,
    *,
    relative_dir: str = "",
    include_hidden: bool = False,
    max_files: Optional[int] = None,
    examples_per_type: int = 3,
) -> dict[str, Any]:
    """Summarize a folder tree as a table of file types (by extension)."""
    root = dataset_root.resolve()
    start = _safe_rel_under_root(root, relative_dir)
    if not start.exists() or not start.is_dir():
        raise ValueError("relative_dir is not an existing directory")

    t0 = time.time()
    total_files = 0
    total_dirs = 0
    total_bytes = 0
    errors: list[str] = []
    truncated = False
    by_ext: dict[str, dict[str, Any]] = {}

    def should_skip_hidden(rel_path: Path) -> bool:
        if include_hidden:
            return False
        return any(part.startswith(".") for part in rel_path.parts)

    stack: list[Path] = [start]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        name = entry.name
                        if not include_hidden and name.startswith("."):
                            continue
                        abs_path = Path(entry.path)
                        try:
                            rel_to_root = abs_path.relative_to(root)
                        except Exception:
                            continue
                        if should_skip_hidden(rel_to_root):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            total_dirs += 1
                            stack.append(abs_path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue

                        total_files += 1
                        if max_files is not None and total_files > int(max_files):
                            truncated = True
                            stack.clear()
                            break

                        try:
                            st = entry.stat(follow_symlinks=False)
                            size = int(getattr(st, "st_size", 0) or 0)
                        except Exception:
                            size = 0

                        ext = abs_path.suffix.lower() or _INVENTORY_EXT_NONE
                        ctype, _enc = mimetypes.guess_type(abs_path.name)
                        mime = str(ctype or "")

                        total_bytes += size
                        bucket = by_ext.get(ext)
                        if bucket is None:
                            bucket = {"ext": ext, "mime": mime, "count": 0, "bytes": 0, "examples": []}
                            by_ext[ext] = bucket
                        bucket["count"] = int(bucket.get("count") or 0) + 1
                        bucket["bytes"] = int(bucket.get("bytes") or 0) + size
                        ex = bucket.get("examples")
                        if isinstance(ex, list) and len(ex) < max(0, int(examples_per_type)):
                            ex.append(str(rel_to_root))
                    except Exception as exc:
                        if len(errors) < 8:
                            errors.append(f"{current}: {exc}")
                        continue
        except Exception as exc:
            if len(errors) < 8:
                errors.append(f"{current}: {exc}")
            continue

    types = list(by_ext.values())
    types.sort(key=lambda r: (-int(r.get("count") or 0), -int(r.get("bytes") or 0), str(r.get("ext") or "")))
    elapsed_ms = int(round((time.time() - t0) * 1000.0))
    return {
        "root": str(root),
        "relative_dir": str(relative_dir or ""),
        "scanned_dir": str(start),
        "total_files": total_files,
        "total_dirs": total_dirs,
        "total_bytes": total_bytes,
        "types": types,
        "errors": errors,
        "truncated": truncated,
        "elapsed_ms": elapsed_ms,
    }


def move_files_by_extension_at(
    dataset_root: Path,
    *,
    ext: str,
    dest_relative_dir: str,
    relative_dir: str = "",
    include_hidden: bool = False,
    preserve_tree: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = dataset_root.resolve()
    start = _safe_rel_under_root(root, relative_dir)
    dest_dir = _safe_rel_under_root(root, dest_relative_dir)
    if not start.exists() or not start.is_dir():
        raise ValueError("relative_dir is not an existing directory")
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True, exist_ok=True)

    want_ext = str(ext or "").strip().lower()
    if not want_ext:
        raise ValueError("ext is required")

    moved = 0
    errors: list[str] = []

    def is_hidden(rel_path: Path) -> bool:
        if include_hidden:
            return False
        return any(part.startswith(".") for part in rel_path.parts)

    candidates: list[Path] = []
    stack: list[Path] = [start]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        abs_path = Path(entry.path)
                        try:
                            rel_to_root = abs_path.relative_to(root)
                        except Exception:
                            continue
                        if is_hidden(rel_to_root):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(abs_path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        ext_norm = abs_path.suffix.lower() or _INVENTORY_EXT_NONE
                        if ext_norm != want_ext:
                            continue
                        try:
                            abs_path.relative_to(dest_dir)
                            continue
                        except Exception:
                            pass
                        candidates.append(abs_path)
                    except Exception:
                        continue
        except Exception:
            continue

    for src in candidates:
        try:
            rel_to_root = src.relative_to(root)
        except Exception:
            continue
        if preserve_tree:
            dst = dest_dir / rel_to_root
        else:
            dst = dest_dir / src.name
        if dst.exists():
            stem = dst.stem
            suff = dst.suffix
            parent = dst.parent
            k = 2
            while True:
                candidate = parent / f"{stem}-{k}{suff}"
                if not candidate.exists():
                    dst = candidate
                    break
                k += 1
                if k >= 10000:
                    raise RuntimeError("too many name collisions")
        try:
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(src), str(dst))
            moved += 1
        except Exception as exc:
            if len(errors) < 8:
                errors.append(f"{src}: {exc}")
            continue

    return {"moved": moved, "candidate_count": len(candidates), "errors": errors}


def delete_files_by_extension_at(
    dataset_root: Path,
    *,
    ext: str,
    relative_dir: str = "",
    include_hidden: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    root = dataset_root.resolve()
    start = _safe_rel_under_root(root, relative_dir)
    if not start.exists() or not start.is_dir():
        raise ValueError("relative_dir is not an existing directory")

    want_ext = str(ext or "").strip().lower()
    if not want_ext:
        raise ValueError("ext is required")

    deleted = 0
    errors: list[str] = []

    def is_hidden(rel_path: Path) -> bool:
        if include_hidden:
            return False
        return any(part.startswith(".") for part in rel_path.parts)

    stack: list[Path] = [start]
    while stack:
        cur = stack.pop()
        try:
            with os.scandir(cur) as it:
                for entry in it:
                    try:
                        abs_path = Path(entry.path)
                        try:
                            rel_to_root = abs_path.relative_to(root)
                        except Exception:
                            continue
                        if is_hidden(rel_to_root):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(abs_path)
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        ext_norm = abs_path.suffix.lower() or _INVENTORY_EXT_NONE
                        if ext_norm != want_ext:
                            continue
                        if not dry_run:
                            abs_path.unlink(missing_ok=True)
                        deleted += 1
                    except Exception as exc:
                        if len(errors) < 8:
                            errors.append(f"{entry.path}: {exc}")
                        continue
        except Exception as exc:
            if len(errors) < 8:
                errors.append(f"{cur}: {exc}")
            continue

    return {"deleted": deleted, "errors": errors}


def inspect_library_dataset_at(base: Path) -> dict[str, Any]:
    base = base.resolve()
    fmt = detect_library_dataset_format(base)
    try:
        from .governance import infer_dataset_lineage

        lineage = infer_dataset_lineage(base)
    except Exception as exc:
        lineage = {"nodes": [], "edges": [], "dataset_root": str(base), "error": str(exc)}

    # Single root-level CSV file (e.g. mlops/datasets/metrics.csv) as its own dataset.
    if fmt == LIBRARY_DATASET_FORMAT_CSV and base.is_file():
        try:
            rel = str(base.relative_to(REPO_ROOT))
        except Exception:
            rel = str(base)
        try:
            size = int(base.stat().st_size)
        except OSError:
            size = 0
        csv_files = [
            {
                "name": base.name,
                "filename": base.name,
                "path": rel,
                "size": float(size),
                "size_bytes": size,
            }
        ]
        return {
            "format": fmt,
            "category": DATASET_CATEGORY_TABULAR,
            "images": [],
            "folders": [],
            "csv_files": csv_files,
            "split_counts": {},
            "classes": [],
            "count": 1,
            "lineage": lineage,
        }

    try:
        folders = list_dataset_folders_at(base)
    except Exception:
        folders = []
    if fmt == LIBRARY_DATASET_FORMAT_UNKNOWN and is_ml_audio_dataset_path(base):
        return {
            "format": LIBRARY_DATASET_FORMAT_AUDIOFOLDER,
            "category": DATASET_CATEGORY_AUDIO,
            "images": [],
            "audio_files": [],
            "folders": folders,
            "split_counts": {},
            "classes": [],
            "count": 0,
            "lineage": lineage,
        }
    if fmt == LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
        items, split_counts, classes = list_imagefolder_entries_at(base)
        detection_label_count = sum(1 for item in items if bool(item.get("has_detection_label")))
        return {
            "format": fmt,
            "images": items,
            "folders": folders,
            "split_counts": split_counts,
            "classes": classes,
            "count": len(items),
            "detection_label_count": detection_label_count,
            "missing_detection_label_count": max(0, len(items) - detection_label_count),
            "lineage": lineage,
        }
    if fmt == LIBRARY_DATASET_FORMAT_AUDIOFOLDER:
        items, split_counts, classes = list_audiofolder_entries_at(base)
        return {
            "format": fmt,
            "category": DATASET_CATEGORY_AUDIO,
            "images": [],
            "audio_files": items,
            "folders": folders,
            "split_counts": split_counts,
            "classes": classes,
            "count": len(items),
            "lineage": lineage,
        }
    if fmt == LIBRARY_DATASET_FORMAT_FACE_CSV:
        # Read CSV to build id(filename)->label map.
        csv_path: Optional[Path] = None
        try:
            for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
                if p.is_file() and p.suffix.lower() == ".csv" and not p.name.startswith("."):
                    csv_path = p
                    break
        except Exception:
            csv_path = None
        label_map: dict[str, str] = {}
        classes: set[str] = set()
        csv_row_count = 0
        if csv_path is not None and csv_path.exists():
            import csv as _csv
            try:
                with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = _csv.DictReader(f)
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        fid = str(row.get("id") or "").strip()
                        label = str(row.get("label") or "").strip()
                        if fid and label:
                            label_map[fid] = label
                            classes.add(label)
                        elif label:
                            classes.add(label)
                        csv_row_count += 1
                        if csv_row_count >= 500000:
                            break
            except Exception:
                pass

        # Enumerate face images from the candidate image directories.
        import os as _os_face
        base_resolved = base.resolve()
        face_img_dirs = [
            base_resolved / "Faces",
            base_resolved / "Faces" / "Faces",
            base_resolved / "Original Images",
            base_resolved / "Original Images" / "Original Images",
        ]
        face_images: list[dict[str, Any]] = []
        for face_dir in face_img_dirs:
            if not face_dir.is_dir():
                continue
            try:
                for root_s, _dirs, files in _os_face.walk(face_dir):
                    root_p = Path(root_s)
                    for fn in sorted(files):
                        low_fn = fn.lower()
                        if not any(low_fn.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                            continue
                        img_path = root_p / fn
                        try:
                            rel = img_path.relative_to(base_resolved).as_posix()
                        except Exception:
                            rel = fn
                        try:
                            size = img_path.stat().st_size
                        except Exception:
                            size = 0
                        identity = label_map.get(fn, "")
                        face_images.append({
                            "name": fn,
                            "stem": Path(fn).stem,
                            "path": str(img_path),
                            "relative_path": rel,
                            "split": "root",
                            "size": size,
                            "has_label": bool(identity),
                            "display_name": fn,
                            "classification_label": identity,
                        })
                        if len(face_images) >= 500000:
                            break
            except Exception:
                pass
            if len(face_images) >= 500000:
                break

        return {
            "format": fmt,
            "category": DATASET_CATEGORY_IMAGE,
            "images": face_images,
            "folders": folders,
            "csv_files": [{"name": csv_path.name, "size_bytes": csv_path.stat().st_size}] if csv_path else [],
            "split_counts": {"root": len(face_images)},
            "classes": sorted(classes, key=lambda s: s.lower()),
            "count": len(face_images),
            "lineage": lineage,
        }
    if fmt == LIBRARY_DATASET_FORMAT_YOLO:
        items = list_dataset_entries_at(base)
        split_counts = dataset_split_counts_at(base)
        classes = _read_yolo_classes_at(base)
        return {
            "format": fmt,
            "images": items,
            "folders": folders,
            "split_counts": split_counts,
            "classes": classes,
            "count": len(items),
            "lineage": lineage,
        }
    if fmt == LIBRARY_DATASET_FORMAT_LLM_JSONL:
        text_files, row_count = list_llm_jsonl_entries_at(base)
        return {
            "format": fmt,
            "category": DATASET_CATEGORY_TEXT,
            "images": [],
            "folders": folders,
            "text_files": text_files,
            "split_counts": {},
            "classes": [],
            "count": row_count,
            "lineage": lineage,
        }
    if fmt == LIBRARY_DATASET_FORMAT_CSV:
        # List .csv files at root as the dataset "items".
        csv_files: list[dict[str, Any]] = []
        try:
            for p in sorted(base.iterdir(), key=lambda x: x.name.lower()):
                if p.suffix.lower() == ".csv" and p.is_file():
                    try:
                        size = p.stat().st_size
                    except Exception:
                        size = 0
                    csv_files.append({"name": p.name, "size_bytes": size})
        except Exception:
            pass
        return {
            "format": fmt,
            "category": DATASET_CATEGORY_TABULAR,
            "images": [],
            "folders": folders,
            "csv_files": csv_files,
            "split_counts": {},
            "classes": [],
            "count": len(csv_files),
            "lineage": lineage,
        }

    # Unknown: best-effort (may be empty).
    items = list_dataset_entries_at(base)
    split_counts = dataset_split_counts_at(base)
    return {
        "format": fmt,
        "images": items,
        "folders": folders,
        "split_counts": split_counts,
        "classes": [],
        "lineage": lineage,
    }


def dataset_split_counts_at(base: Path) -> dict[str, int]:
    counts = {"train": 0, "val": 0, "root": 0}
    for entry in list_dataset_entries_at(base):
        split = str(entry.get("split") or "root")
        counts[split] = counts.get(split, 0) + 1
    return counts


def dataset_split_counts(scenario: str) -> dict[str, int]:
    cfg = get_scenario_config(scenario)
    return dataset_split_counts_at(cfg.dataset_path)


def _count_files_by_suffix(root: Path, suffixes) -> int:
    """Count files under ``root`` whose suffix is in ``suffixes``.

    Walks directory entries (one readdir per directory) and matches on the
    filename string only — no per-file ``stat()`` calls. This is the cheap
    counterpart to the entry listers, which build a dict (and stat each file)
    for every sample only to have the count taken from ``len()``. Hidden
    directories are skipped to mirror the listers.
    """
    count = 0
    try:
        for dirpath, dirnames, filenames in os.walk(str(root)):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for name in filenames:
                dot = name.rfind(".")
                if dot >= 0 and name[dot:].lower() in suffixes:
                    count += 1
    except Exception:
        return count
    return count


def _count_image_files_under(root: Path) -> int:
    return _count_files_by_suffix(root, DATASET_IMAGE_SUFFIXES)


def _count_yolo_images_at(base: Path) -> int:
    """Count YOLO dataset images. Mirrors root selection in
    :func:`list_dataset_entries_at` but only counts (no stat/label probes)."""
    base_resolved = base.resolve()
    if not base_resolved.is_dir():
        return 0
    images_root = base_resolved / "images"
    split_first_roots = _split_first_yolo_roots(base_resolved)
    if not images_root.exists() and split_first_roots:
        roots = [image_root for _s, _c, _sr, image_root, _lr in split_first_roots]
    else:
        try:
            split_dirs = [p for p in images_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except Exception:
            split_dirs = []
        split_dir_names = {p.name for p in split_dirs}
        split_markers = set(DATASET_SPLITS) | {"valid", "test"}
        has_split_layout = bool(split_dir_names & split_markers)
        if has_split_layout and split_dirs:
            roots = split_dirs
        elif not images_root.exists():
            roots = [base_resolved]
        else:
            roots = [images_root]
    total = 0
    for image_root in roots:
        if image_root.exists():
            total += _count_image_files_under(image_root)
    return total


def _count_imagefolder_images_at(base: Path) -> int:
    """Count ImageFolder dataset images. Mirrors
    :func:`list_imagefolder_entries_at` traversal but only counts."""
    base_resolved = base.resolve()
    if not base_resolved.is_dir():
        return 0
    total = 0
    for split_dir, _canon in _IMAGEFOLDER_SPLIT_DIRS:
        split_root = base_resolved / split_dir
        if not split_root.is_dir():
            continue
        try:
            class_dirs = [p for p in split_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except Exception:
            class_dirs = []
        for class_dir in class_dirs:
            total += _count_image_files_under(class_dir)
    return total


def _count_audiofolder_at(base: Path) -> int:
    """Count AudioFolder dataset samples. Mirrors
    :func:`list_audiofolder_entries_at` traversal but only counts."""
    base_resolved = base.resolve()
    if not base_resolved.is_dir():
        return 0
    split_roots: list[Path] = []
    for split_dir, _canon in _IMAGEFOLDER_SPLIT_DIRS:
        split_root = base_resolved / split_dir
        if split_root.is_dir():
            split_roots.append(split_root)
    if not split_roots:
        split_roots.append(base_resolved)
    total = 0
    for split_root in split_roots:
        try:
            class_dirs = [p for p in split_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        except Exception:
            class_dirs = []
        for class_dir in class_dirs:
            total += _count_files_by_suffix(class_dir, DATASET_AUDIO_SUFFIXES)
    return total


def dataset_count(scenario: str) -> int:
    try:
        cfg = get_scenario_config(scenario)
        if str(cfg.backbone_type or "") == "archival_ingestion":
            from Insight.insight_local.cvops.archive_store import ArchiveStore

            store = ArchiveStore(
                storage_root=_resolve_repo_path(
                    str((cfg.backbone_config or {}).get("archive_storage_root") or "state/insight_local/cvops/archive_corpora")
                )
            )
            try:
                corpus_id = str((cfg.backbone_config or {}).get("corpus_id") or "").strip()
                dataset_version_id = str((cfg.backbone_config or {}).get("dataset_version_id") or "").strip()
                if not corpus_id:
                    return 0
                if not dataset_version_id:
                    versions = store.list_dataset_versions(corpus_id)
                    if not versions:
                        return 0
                    dataset_version_id = str(versions[0].get("dataset_version_id") or "")
                if not dataset_version_id:
                    return 0
                version = store.get_dataset_version(dataset_version_id)
                files = version.get("files") if isinstance(version, dict) else []
                return len(files) if isinstance(files, list) else 0
            finally:
                store.close()
        base = cfg.dataset_path
        fmt = detect_library_dataset_format(base)
        if fmt == LIBRARY_DATASET_FORMAT_YOLO:
            return _count_yolo_images_at(base)
        if fmt == LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
            return _count_imagefolder_images_at(base)
        if fmt == LIBRARY_DATASET_FORMAT_AUDIOFOLDER:
            return _count_audiofolder_at(base)
        if fmt == LIBRARY_DATASET_FORMAT_LLM_JSONL:
            _items, rows = list_llm_jsonl_entries_at(base)
            return rows
        if fmt == LIBRARY_DATASET_FORMAT_CSV:
            inspected = inspect_library_dataset_at(base)
            csv_files = inspected.get("csv_files") if isinstance(inspected, dict) else []
            return len(csv_files) if isinstance(csv_files, list) else 0
        return 0
    except Exception:
        return 0


def scenario_names_for_dataset_folder(dataset_folder_name: str) -> list[str]:
    """Scenarios whose registry `dataset` field matches this folder name (e.g. fall_detection)."""
    want = str(dataset_folder_name or "").strip()
    if not want:
        return []
    payload = load_registry()
    out: list[str] = []
    for item in payload.get("scenarios") or []:
        if not isinstance(item, dict) or not item.get("enabled", True):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        try:
            cfg = get_scenario_config(name)
        except Exception:
            continue
        if cfg.dataset == want:
            out.append(name)
    return out


def _run_dirs(cfg: ScenarioConfig) -> list[Path]:
    model_root = MLOPS_ROOT / "models" / cfg.name
    if not model_root.exists():
        return []
    runs = [p for p in model_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    runs.sort(key=lambda p: int(p.name[1:]))
    return runs


def latest_run_dir(scenario: str) -> Optional[Path]:
    cfg = get_scenario_config(scenario)
    runs = _run_dirs(cfg)
    return runs[-1] if runs else None


def latest_run_metrics(scenario: str) -> Optional[dict[str, Any]]:
    run = latest_run_dir(scenario)
    if run is None:
        return None
    metrics_path = run / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        data = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    # Normalize common fields across CV and tabular runs:
    # - YOLO runs write top-level keys (map50, trained_at, ...)
    # - tabular template writes {"metrics": {...}, "history": [...]}
    if isinstance(data.get("metrics"), dict):
        merged = dict(data)
        merged.update(dict(data.get("metrics") or {}))
        data = merged
    data["_run"] = run.name
    return data


def _load_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _count_run_artifacts(run: Path) -> int:
    count = 0
    try:
        for item in run.rglob("*"):
            if item.is_file():
                count += 1
    except Exception:
        return count
    return count


def resolve_scenario_run_dir(scenario: str, version: str) -> Optional[Path]:
    cfg = get_scenario_config(scenario)
    runs = _run_dirs(cfg)
    if not runs:
        return None

    requested = str(version or "").strip()
    if not requested or requested == "latest":
        return runs[-1]
    if requested.startswith("v") and requested[1:].isdigit():
        match = next((run for run in runs if run.name == requested), None)
        if match is not None:
            return match
    return None


def list_scenario_runs(scenario: str) -> list[dict[str, Any]]:
    cfg = get_scenario_config(scenario)
    runs = _run_dirs(cfg)
    latest_name = runs[-1].name if runs else ""
    history: list[dict[str, Any]] = []
    try:
        from .model_registry import (
            get_model_version as _get_model_version,
            resolve_alias as _resolve_alias,
            version_id_for_run as _version_id_for_run,
        )

        prod_alias = _resolve_alias(cfg.name, "prod")
        candidate_alias = _resolve_alias(cfg.name, "candidate")
        prod_vid = str((prod_alias or {}).get("version_id") or "")
        candidate_vid = str((candidate_alias or {}).get("version_id") or "")
    except Exception:
        _version_id_for_run = lambda _s, _r: None  # type: ignore[assignment]
        _get_model_version = lambda _s, _vid: None  # type: ignore[assignment]
        prod_vid = ""
        candidate_vid = ""

    for run in runs:
        metrics_path = run / "metrics.json"
        metrics = _load_json_dict(metrics_path)
        if isinstance(metrics.get("metrics"), dict):
            merged = dict(metrics)
            merged.update(dict(metrics.get("metrics") or {}))
            metrics = merged
        status_marker = str(metrics.get("run_status") or metrics.get("status") or "").strip().lower()
        if status_marker == "cancelled":
            status_marker = "canceled"
        verified_path = _verified_path(run)
        verified_payload = _load_json_dict(verified_path)
        # CV backbones produce weights.pt; tabular backbones may produce weights.pth or model.pkl;
        # face_recognition produces gallery.db; LLM fine-tuning produces a LoRA adapter.
        interrupted_weight_candidates: list[Path] = []
        if status_marker in {"partial", "canceled", "interrupted", "error", "failed"}:
            for key in ("weights", "checkpoint", "last_checkpoint", "best_checkpoint"):
                raw_path = str(metrics.get(key) or "").strip()
                if not raw_path:
                    continue
                path = Path(raw_path)
                if not path.is_absolute():
                    path = (run / path).resolve()
                interrupted_weight_candidates.append(path)
            interrupted_weight_candidates.extend(
                [run / "weights" / "best.pt", run / "weights" / "last.pt"]
            )
        weights_candidates = [
            run / "weights.pt",
            *interrupted_weight_candidates,
            run / "weights.pth",
            run / "model.pkl",
            run / "gallery.db",
            run / "adapter" / "adapter_model.safetensors",
            run / "adapter_model.safetensors",
        ]
        weights_path = next((p for p in weights_candidates if p.exists()), run / "weights.pt")
        data_yaml_path = run / "data.generated.yaml"
        args_yaml_path = run / "args.yaml"
        weights_ready = False
        try:
            weights_ready = weights_path.exists() and weights_path.stat().st_size >= WEIGHTS_MIN_BYTES
        except Exception:
            weights_ready = False

        has_metrics = bool(metrics)
        verified = verified_path.exists()
        if cfg.backbone_type == "llm_fine_tuning":
            if weights_ready and has_metrics and verified:
                status = "ready"
            elif weights_ready and has_metrics:
                status = "trained"
            elif has_metrics:
                status = "metrics_only"
            else:
                status = "partial"
        elif weights_ready and verified:
            status = "ready"
        elif weights_ready:
            status = "trained"
        elif has_metrics:
            status = "metrics_only"
        else:
            status = "partial"
        if status_marker in {"partial", "canceled", "interrupted", "error"}:
            status = status_marker
        elif status_marker == "failed":
            status = "error"

        version_number = None
        if run.name.startswith("v") and run.name[1:].isdigit():
            version_number = int(run.name[1:])
        model_version_id = _version_id_for_run(cfg.name, run.name)
        model_version = _get_model_version(cfg.name, model_version_id) if model_version_id else None
        ci_cd_state = (
            dict(model_version.get("ci_cd") or {})
            if isinstance(model_version, dict) and isinstance(model_version.get("ci_cd"), dict)
            else {}
        )

        history.append(
            {
                "scenario": cfg.name,
                "version": run.name,
                "version_number": version_number,
                "model_version_id": model_version_id or "",
                "is_prod": bool(model_version_id and model_version_id == prod_vid),
                "is_candidate": bool(model_version_id and model_version_id == candidate_vid),
                "is_latest": run.name == latest_name,
                "status": status,
                "run_dir": str(run),
                "artifact_count": _count_run_artifacts(run),
                "base_model": str(metrics.get("base_model") or ""),
                "final_model_name": str(metrics.get("final_model_name") or ""),
                "final_model_file": str(metrics.get("final_model_file") or ""),
                "final_model_path": str(metrics.get("final_model_path") or ""),
                "trained_at": str(metrics.get("trained_at") or ""),
                "training_duration_seconds": metrics.get(
                    "training_duration_seconds",
                    metrics.get("duration_seconds", metrics.get("elapsed_seconds", "")),
                ),
                "map50": metrics.get("map50"),
                "task": str(metrics.get("task") or ""),
                "val_metric": metrics.get("val_metric", metrics.get("final_val_metric")),
                "error": str(metrics.get("error") or ""),
                "job_id": str(metrics.get("job_id") or ""),
                "stopped_at": str(metrics.get("stopped_at") or ""),
                "weights": str(weights_path) if weights_path.exists() else "",
                "weights_ready": weights_ready,
                "data_yaml": str(data_yaml_path) if data_yaml_path.exists() else "",
                "args_yaml": str(args_yaml_path) if args_yaml_path.exists() else "",
                "metrics_path": str(metrics_path) if metrics_path.exists() else "",
                "ci_cd": ci_cd_state,
                "ci_cd_gate_status": str(ci_cd_state.get("gate_status") or ""),
                "ci_cd_report_path": str(ci_cd_state.get("report_path") or ""),
                "has_metrics": has_metrics,
                "verified": verified,
                "verified_at": str(verified_payload.get("verified_at") or ""),
            }
        )
    return history


def get_scenario_run_record(scenario: str, version: str) -> Optional[dict[str, Any]]:
    target = str(version or "").strip()
    if not target:
        return None
    for entry in list_scenario_runs(scenario):
        if str(entry.get("version") or "") == target:
            return entry
    return None


def resolve_inference_target(scenario: str, version: str = "") -> Optional[dict[str, Any]]:
    requested = str(version or "").strip()
    if requested in {"candidate", "prod"}:
        try:
            from .model_registry import resolve_alias as _resolve_model_alias

            aliased = _resolve_model_alias(scenario, requested)
        except Exception:
            aliased = None
        if isinstance(aliased, dict):
            artifacts = aliased.get("artifacts") if isinstance(aliased.get("artifacts"), dict) else {}
            weights_path = Path(str(artifacts.get("weights") or "")).resolve()
            if weights_path.exists():
                return {
                    "scenario": scenario,
                    "version": str(aliased.get("run_version") or requested),
                    "weights_path": weights_path,
                    "source": f"model_registry_alias:{requested}",
                }
    if requested:
        entry = get_scenario_run_record(scenario, requested)
        if entry is None or not bool(entry.get("weights_ready")):
            return None
        weights_path = Path(str(entry.get("weights") or "")).resolve()
        return {
            "scenario": scenario,
            "version": str(entry.get("version") or requested),
            "weights_path": weights_path,
            "source": "run_history",
        }

    runs = list_scenario_runs(scenario)
    ready_runs = [entry for entry in runs if bool(entry.get("weights_ready"))]
    if ready_runs:
        ready_runs.sort(
            key=lambda entry: (
                int(entry.get("version_number") or -1),
                str(entry.get("trained_at") or ""),
                str(entry.get("version") or ""),
            ),
            reverse=True,
        )
        chosen = ready_runs[0]
        return {
            "scenario": scenario,
            "version": str(chosen.get("version") or ""),
            "weights_path": Path(str(chosen.get("weights") or "")).resolve(),
            "source": "run_history",
        }

    cfg = get_scenario_config(scenario)
    if _weights_ready(cfg):
        return {
            "scenario": scenario,
            "version": "",
            "weights_path": cfg.weights_path.resolve(),
            "source": "scenario_config",
        }
    return None


def _verified_path(run: Path) -> Path:
    return run / "verified.json"


def is_verified(scenario: str) -> bool:
    run = latest_run_dir(scenario)
    if run is None:
        return False
    return _verified_path(run).exists()


def mark_verified(scenario: str, *, note: str = "") -> dict[str, Any]:
    run = latest_run_dir(scenario)
    if run is None:
        raise ValueError(f"Scenario '{scenario}' has no training run to verify")
    payload = {
        "scenario": scenario,
        "run": run.name,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "note": note,
    }
    _verified_path(run).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        from .model_registry import set_alias as _set_alias, version_id_for_run as _version_id_for_run

        vid = _version_id_for_run(scenario, run.name)
        if vid:
            _set_alias(scenario, "prod", vid)
            payload["model_registry_prod"] = vid
    except Exception:
        pass
    return payload


def clear_verified(scenario: str) -> bool:
    run = latest_run_dir(scenario)
    if run is None:
        return False
    p = _verified_path(run)
    if p.exists():
        p.unlink()
        return True
    return False


_SCENARIO_STATUS_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}
_SCENARIO_STATUS_CACHE_LOCK = threading.Lock()


def _dir_tree_signature(root: Path) -> tuple:
    """Signature of a directory tree from per-directory mtimes only.

    Adding or removing a file in any directory bumps that directory's mtime,
    so this changes whenever dataset/run contents change — without statting
    every file. Returns ``()`` if the tree is missing or unreadable.
    """
    sig: list[tuple[str, int]] = []
    try:
        for dirpath, dirnames, _filenames in os.walk(str(root)):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            try:
                sig.append((dirpath, os.stat(dirpath).st_mtime_ns))
            except OSError:
                continue
    except Exception:
        return ()
    return tuple(sig)


def _scenario_status_signature(cfg: "ScenarioConfig") -> tuple:
    try:
        registry_mtime = REGISTRY_PATH.stat().st_mtime_ns
    except OSError:
        registry_mtime = 0
    return (
        registry_mtime,
        _dir_tree_signature(cfg.dataset_path),
        _dir_tree_signature(MLOPS_ROOT / "models" / cfg.name),
    )


def get_scenario_status(scenario: str) -> dict[str, Any]:
    """Cached wrapper around :func:`_compute_scenario_status`.

    The Train catalog calls this once per enabled scenario on every load. The
    underlying computation walks the dataset and run trees, so it is memoized
    by a cheap directory-mtime signature: a cache hit avoids the filesystem
    walk entirely, and the cache self-invalidates when dataset or run contents
    change. Archival scenarios are sqlite-backed and not cached here.
    """
    try:
        cfg = get_scenario_config(scenario)
    except Exception:
        return _compute_scenario_status(scenario)
    if str(cfg.backbone_type or "") == "archival_ingestion":
        return _compute_scenario_status(scenario)
    signature = _scenario_status_signature(cfg)
    with _SCENARIO_STATUS_CACHE_LOCK:
        cached = _SCENARIO_STATUS_CACHE.get(scenario)
        if cached is not None and cached[0] == signature:
            return copy.deepcopy(cached[1])
    payload = _compute_scenario_status(scenario)
    with _SCENARIO_STATUS_CACHE_LOCK:
        _SCENARIO_STATUS_CACHE[scenario] = (signature, payload)
    return copy.deepcopy(payload)


def _compute_scenario_status(scenario: str) -> dict[str, Any]:
    """Derive status purely from filesystem state. No DB required.

    Status values: empty | dataset | trained | ready | error
    (training is layered in by the service by inspecting the job queue)
    """
    try:
        cfg = get_scenario_config(scenario)
    except Exception as exc:
        return {
            "name": scenario,
            "status": "error",
            "error": str(exc),
            "backbone_type": "",
            "backbone_config": {},
            "dataset": "",
            "classes": [],
            "dataset_count": 0,
            "latest_run": None,
            "verified": False,
            "weights_ready": False,
        }

    ds_count = dataset_count(scenario)
    backbone = str(cfg.backbone_type or "yolo_detection")
    if backbone == "archival_ingestion":
        from Insight.insight_local.cvops.archive_store import ArchiveStore

        bcfg = dict(cfg.backbone_config or {})
        corpus_id = str(bcfg.get("corpus_id") or "").strip()
        dataset_version_id = str(bcfg.get("dataset_version_id") or "").strip()
        snapshot_id = str(bcfg.get("latest_snapshot_id") or "").strip()
        versions: list[dict[str, Any]] = []
        latest_snapshot: dict[str, Any] | None = None
        latest_version: dict[str, Any] | None = None
        archive_error = ""
        try:
            store = ArchiveStore(
                storage_root=_resolve_repo_path(
                    str(bcfg.get("archive_storage_root") or "state/insight_local/cvops/archive_corpora")
                )
            )
            try:
                if corpus_id:
                    versions = store.list_dataset_versions(corpus_id)
                    latest_version = next(
                        (
                            item for item in versions
                            if str(item.get("dataset_version_id") or "") == dataset_version_id
                        ),
                        versions[0] if versions else None,
                    )
                    if latest_version is not None:
                        dataset_version_id = str(latest_version.get("dataset_version_id") or "")
                        if snapshot_id:
                            try:
                                latest_snapshot = store.get_snapshot(snapshot_id)
                            except Exception:
                                latest_snapshot = None
                        if latest_snapshot is None:
                            latest_snapshot = store.latest_snapshot(corpus_id, dataset_version_id)
                            if latest_snapshot is not None:
                                snapshot_id = str(latest_snapshot.get("snapshot_id") or "")
                elif snapshot_id:
                    latest_snapshot = store.get_snapshot(snapshot_id)
                    corpus_id = str(latest_snapshot.get("corpus_id") or "")
                    dataset_version_id = str(latest_snapshot.get("dataset_version_id") or "")
                    versions = store.list_dataset_versions(corpus_id) if corpus_id else []
                    latest_version = next(
                        (
                            item for item in versions
                            if str(item.get("dataset_version_id") or "") == dataset_version_id
                        ),
                        None,
                    )
            finally:
                store.close()
        except Exception as exc:
            archive_error = str(exc)

        if latest_snapshot is not None:
            status = "trained"
        elif ds_count > 0:
            status = "dataset"
        else:
            status = "empty"

        latest_run = None
        if latest_snapshot is not None:
            latest_run = {
                "version": str(latest_snapshot.get("snapshot_id") or ""),
                "final_model_name": "",
                "final_model_file": "",
                "final_model_path": "",
                "map50": None,
                "trained_at": latest_snapshot.get("created_at"),
                "run_dir": str((latest_version or {}).get("raw_root") or ""),
                "weights": "",
                "verified": False,
            }

        return {
            "name": cfg.name,
            "display_name": cfg.display_name,
            "description": cfg.description,
            "dataset": cfg.dataset,
            "classes": [],
            "base_model": "",
            "backbone_type": backbone,
            "backbone_config": dict(cfg.backbone_config or {}),
            "base_model_exists": True,
            "base_model_resolved": "",
            "status": status,
            "dataset_count": ds_count,
            "latest_run": latest_run,
            "history_count": len(versions),
            "verified": False,
            "weights_ready": latest_snapshot is not None,
            "training_guard": {"status": "ok", "blocking_reasons": []},
            "archive_corpus_id": corpus_id,
            "archive_dataset_version_id": dataset_version_id,
            "archive_snapshot_id": snapshot_id,
            "ci_cd": get_scenario_ci_cd_policy(cfg.name),
            "error": archive_error,
        }

    weights_ok = _weights_ready(cfg)
    metrics = latest_run_metrics(scenario)
    verified = is_verified(scenario)
    runs = _run_dirs(cfg)
    latest_dir_for_status = latest_run_dir(scenario)
    if not weights_ok and latest_dir_for_status is not None:
        for candidate in (
            latest_dir_for_status / "weights.pt",
            latest_dir_for_status / "weights.pth",
            latest_dir_for_status / "model.pkl",
            latest_dir_for_status / "gallery.db",
            latest_dir_for_status / "adapter" / "adapter_model.safetensors",
            latest_dir_for_status / "adapter_model.safetensors",
        ):
            try:
                if candidate.exists() and candidate.stat().st_size >= WEIGHTS_MIN_BYTES:
                    weights_ok = True
                    break
            except Exception:
                continue
    cell_backbone_has_run = False
    if backbone in ("torch_tabular", "custom_code") and latest_dir_for_status is not None:
        if (latest_dir_for_status / "metrics.json").exists():
            cell_backbone_has_run = True
    elif backbone == "llm_fine_tuning" and latest_dir_for_status is not None:
        adapter_candidates = [
            latest_dir_for_status / "adapter" / "adapter_model.safetensors",
            latest_dir_for_status / "adapter_model.safetensors",
        ]
        adapter_ready = False
        for candidate in adapter_candidates:
            try:
                if candidate.exists() and candidate.stat().st_size >= WEIGHTS_MIN_BYTES:
                    adapter_ready = True
                    break
            except Exception:
                continue
        cell_backbone_has_run = adapter_ready and (latest_dir_for_status / "metrics.json").exists()
    try:
        resolved_model = resolve_model_reference(cfg.base_model)
        base_model_exists = True
        base_model_resolved = str(resolved_model)
    except Exception:
        base_model_exists = False
        base_model_resolved = ""

    if backbone == "llm_fine_tuning":
        if cell_backbone_has_run and verified:
            status = "ready"
        elif cell_backbone_has_run:
            status = "trained"
        elif ds_count > 0:
            status = "dataset"
        else:
            status = "empty"
    elif weights_ok and verified:
        status = "ready"
    elif weights_ok or cell_backbone_has_run:
        status = "trained"
    elif ds_count > 0:
        status = "dataset"
    else:
        status = "empty"

    latest_run = None
    if metrics is not None:
        latest_dir = latest_run_dir(scenario)
        latest_run = {
            "version": metrics.get("_run"),
            "final_model_name": metrics.get("final_model_name") or "",
            "final_model_file": metrics.get("final_model_file") or "",
            "final_model_path": metrics.get("final_model_path") or "",
            "map50": metrics.get("map50"),
            "trained_at": metrics.get("trained_at"),
            "run_dir": str(latest_dir) if latest_dir is not None else "",
            "weights": "",
            "verified": verified,
        }
        if latest_dir is not None:
            if (latest_dir / "weights.pt").exists():
                latest_run["weights"] = str(latest_dir / "weights.pt")
            elif (latest_dir / "weights.pth").exists():
                latest_run["weights"] = str(latest_dir / "weights.pth")
            elif (latest_dir / "gallery.db").exists():
                latest_run["weights"] = str(latest_dir / "gallery.db")
            elif (latest_dir / "model.json").exists():
                latest_run["weights"] = str(latest_dir / "model.json")
            elif (latest_dir / "adapter" / "adapter_model.safetensors").exists():
                latest_run["weights"] = str(latest_dir / "adapter" / "adapter_model.safetensors")
            elif (latest_dir / "adapter_model.safetensors").exists():
                latest_run["weights"] = str(latest_dir / "adapter_model.safetensors")

    return {
        "name": cfg.name,
        "display_name": cfg.display_name,
        "description": cfg.description,
        "dataset": cfg.dataset,
        "classes": list(cfg.classes),
        "base_model": cfg.base_model,
        "backbone_type": str(cfg.backbone_type or "yolo_detection"),
        "backbone_config": dict(cfg.backbone_config or {}),
        "base_model_exists": base_model_exists,
        "base_model_resolved": base_model_resolved,
        "status": status,
        "dataset_count": ds_count,
        "latest_run": latest_run,
        "history_count": len(runs),
        "verified": verified,
        "weights_ready": weights_ok,
        "training_guard": build_training_guard(cfg.base_model, cfg.hyperparams, scenario=cfg.name),
        "ci_cd": get_scenario_ci_cd_policy(cfg.name),
        "error": "",
    }
