from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from .training_algos import validate_trainer_name
from .registry import (
    MLOPS_ROOT,
    get_scenario_ci_cd_policy,
    get_scenario_config,
    list_available_models,
    resolve_model_reference,
)
from .governance import create_dataset_snapshot, persist_dataset_snapshot
from .model_registry import list_model_versions, register_model_version
from .reproducibility import apply_deterministic_policy, capture_environment_fingerprint, create_repro_manifest
from .system_guard import build_training_guard
from .verdict import TrainingVerdict, forecast_run, render_forecast


_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
YOLO: Any = None


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class _LineTee(io.TextIOBase):
    """A text stream that mirrors writes to an original stream and emits
    each complete line to a callback as a `log` training_progress event.

    Carriage-return-only updates (progress bars) are flushed once per line
    so the UI gets a snapshot without buffering indefinitely.
    """

    def __init__(
        self,
        original: Any,
        emit: Callable[[str, str], None],
        stream_name: str,
    ) -> None:
        super().__init__()
        self._original = original
        self._emit = emit
        self._stream_name = stream_name
        self._buffer = ""

    def writable(self) -> bool:
        return True

    def isatty(self) -> bool:
        try:
            return bool(self._original.isatty())
        except Exception:
            return False

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        try:
            self._original.write(s)
        except Exception:
            pass
        self._buffer += s
        # Normalize CR-only progress updates into \n so each state ships.
        normalized = self._buffer.replace("\r\n", "\n").replace("\r", "\n")
        parts = normalized.split("\n")
        # Last element is the unterminated tail.
        self._buffer = parts[-1]
        for part in parts[:-1]:
            clean = _strip_ansi(part).rstrip()
            if clean:
                try:
                    self._emit(clean, self._stream_name)
                except Exception:
                    pass
        return len(s)

    def flush(self) -> None:
        try:
            self._original.flush()
        except Exception:
            pass
        if self._buffer:
            clean = _strip_ansi(self._buffer).rstrip()
            self._buffer = ""
            if clean:
                try:
                    self._emit(clean, self._stream_name)
                except Exception:
                    pass


class _ProgressLogHandler(logging.Handler):
    """Route ultralytics logger records through the progress callback."""

    def __init__(self, emit: Callable[[str, str], None]) -> None:
        super().__init__(level=logging.INFO)
        self._emit = emit

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            return
        stream = "stderr" if record.levelno >= logging.WARNING else "stdout"
        for line in _strip_ansi(msg).splitlines():
            line = line.rstrip()
            if not line:
                continue
            try:
                self._emit(line, stream)
            except Exception:
                pass


@contextlib.contextmanager
def _capture_training_logs(
    progress_callback: Optional[Callable[[dict[str, Any]], None]],
    *,
    tee_streams: bool = True,
):
    """Context manager that tees stdout/stderr and attaches a log handler
    to the ultralytics logger. Every line captured is forwarded to
    progress_callback as a `log` event. If no callback is provided, this
    is a no-op passthrough.

    When `tee_streams` is False, skip the stdout/stderr redirect (the
    ultralytics logger handler is still installed). This is what the
    subprocess worker needs: its stdout is already the pipe to the parent,
    so teeing through the callback would cause infinite recursion when the
    callback itself writes structured events to that same stdout.
    """
    if progress_callback is None:
        yield
        return

    def _emit(line: str, stream: str) -> None:
        try:
            progress_callback(
                {
                    "event": "log",
                    "line": line,
                    "stream": stream,
                    "timestamp": time.time(),
                }
            )
        except Exception:
            pass

    handler = _ProgressLogHandler(_emit)
    handler.setFormatter(logging.Formatter("%(message)s"))
    ultra_logger = logging.getLogger("ultralytics")
    ultra_logger.addHandler(handler)
    prev_level = ultra_logger.level
    if prev_level > logging.INFO or prev_level == logging.NOTSET:
        ultra_logger.setLevel(logging.INFO)

    tee_out: Optional[_LineTee] = None
    tee_err: Optional[_LineTee] = None
    try:
        if tee_streams:
            tee_out = _LineTee(sys.stdout, _emit, "stdout")
            tee_err = _LineTee(sys.stderr, _emit, "stderr")
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                yield
        else:
            yield
    finally:
        if tee_out is not None:
            try:
                tee_out.flush()
            except Exception:
                pass
        if tee_err is not None:
            try:
                tee_err.flush()
            except Exception:
                pass
        ultra_logger.removeHandler(handler)
        if prev_level != ultra_logger.level:
            ultra_logger.setLevel(prev_level)


def _next_run_dir(models_root: Path) -> Path:
    runs = [p for p in models_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    if not runs:
        return models_root / "v1"
    latest = max(int(p.name[1:]) for p in runs)
    return models_root / f"v{latest + 1}"


def _next_run_dir_across(primary_root: Path, selected_root: Path) -> Path:
    roots = [primary_root]
    if selected_root.resolve() != primary_root.resolve():
        roots.append(selected_root)
    latest = 0
    for root in roots:
        try:
            runs = [p for p in root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
        except Exception:
            runs = []
        if runs:
            latest = max(latest, *(int(p.name[1:]) for p in runs))
    return selected_root / f"v{latest + 1}"


def _ensure_local_run_link(local_run_dir: Path, actual_run_dir: Path) -> None:
    try:
        if local_run_dir.resolve() == actual_run_dir.resolve():
            return
    except Exception:
        pass
    if local_run_dir.exists() or local_run_dir.is_symlink():
        return
    try:
        local_run_dir.parent.mkdir(parents=True, exist_ok=True)
        rel_target = os.path.relpath(str(actual_run_dir), start=str(local_run_dir.parent))
        local_run_dir.symlink_to(rel_target, target_is_directory=True)
    except Exception:
        pass


def _list_run_dirs(models_root: Path) -> list[Path]:
    runs = [p for p in models_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    return sorted(runs, key=lambda p: int(p.name[1:]), reverse=True)


def _find_latest_resume_checkpoint(models_root: Path) -> tuple[Path, Path] | None:
    for run_dir in _list_run_dirs(models_root):
        checkpoint = run_dir / "weights" / "last.pt"
        try:
            if checkpoint.exists() and checkpoint.stat().st_size > 1024:
                return run_dir, checkpoint
        except Exception:
            continue
    return None


def _is_completed_resume_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return (
        "nothing to resume" in text
        or ("training to" in text and "epochs is finished" in text)
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scenario with Ultralytics YOLO")
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--trainer",
        default=None,
        help="Training algorithm backend (default from hyperparams.trainer or ultralytics_yolo)",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override scenario base model for this run only",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override scenario hyperparams.epochs")
    parser.add_argument("--imgsz", type=int, default=None, help="Override scenario hyperparams.imgsz")
    parser.add_argument("--seed", type=int, default=None, help="Deterministic seed override")
    parser.add_argument("--final-model-name", default="", help="Human-friendly final model name for this run")
    parser.add_argument(
        "--non-deterministic",
        action="store_true",
        help="Disable deterministic training policy (not recommended for reproducibility)",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=None,
        help="Checkpoint save period in epochs (default: 1, i.e. every epoch)",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from latest available checkpoint (default).",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Always start a fresh run.",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def _lookup_registered_version(value: str, default_scenario: str | None) -> tuple[str, str] | None:
    """Try to resolve `value` as a registered model version.

    Accepts `scenario:run_version` (e.g. `fall_detection:v7`), or a bare
    `run_version` resolved against `default_scenario`. Returns
    `(weights_path, version_id)` on hit, or None if `value` does not
    name a registered version.
    """
    raw = str(value or "").strip()
    if not raw:
        return None
    if ":" in raw:
        scenario, run_version = raw.split(":", 1)
    else:
        if not default_scenario:
            return None
        scenario, run_version = default_scenario, raw
    scenario = scenario.strip()
    run_version = run_version.strip()
    if not scenario or not run_version:
        return None
    try:
        versions = list_model_versions(scenario)
    except Exception:
        return None
    for entry in versions:
        if str(entry.get("run_version") or "") != run_version:
            continue
        artifacts = entry.get("artifacts") if isinstance(entry.get("artifacts"), dict) else {}
        weights = str(artifacts.get("weights") or "")
        if weights and Path(weights).exists():
            return weights, str(entry.get("version_id") or "")
    return None


def _resolve_base_model(base_model: str, *, scenario: str | None = None) -> tuple[str, str | None]:
    """Resolve a base-model reference to (weights_path, parent_version_id_or_None).

    Resolution order:
      1. A registered model version (scenario:run_version, or bare run_version
         against the current scenario) — returns parent_version_id so the
         retrain run can be linked back to its source in the registry.
      2. An asset/default model (path or name in MODEL_SEARCH_ROOTS).
      3. Fallback to a known YOLO seed.
    """
    hit = _lookup_registered_version(base_model, scenario)
    if hit is not None:
        return hit
    try:
        return str(resolve_model_reference(base_model)), None
    except Exception:
        models = list_available_models()
        if not models:
            raise
        preferred = ("yolo11n.pt", "yolo26n.pt", "yolov10n.pt", "yolo26s.pt")
        by_name = {str(m.get("name") or ""): str(m.get("path") or "") for m in models}
        for name in preferred:
            path = by_name.get(name, "")
            if path:
                return path, None
        first = str(models[0].get("path") or "")
        if first:
            return first, None
        raise


def _existing_data_yaml_usable(data_yaml: Path) -> bool:
    try:
        raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return False
    if not isinstance(raw, dict):
        return False
    train_ref = raw.get("train")
    if not isinstance(train_ref, str) or not train_ref.strip():
        return False
    path_ref = raw.get("path")
    if isinstance(path_ref, str) and path_ref.strip():
        base = Path(path_ref).expanduser()
        if not base.is_absolute():
            base = (data_yaml.parent / base).resolve()
    else:
        base = data_yaml.parent
    train_path = Path(train_ref).expanduser()
    if not train_path.is_absolute():
        train_path = (base / train_path).resolve()
    return train_path.exists()


def _contains_dataset_images(path: Path) -> bool:
    try:
        for candidate in path.rglob("*"):
            if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                return True
    except Exception:
        return False
    return False


def _split_first_train_val_refs(dataset_path: Path) -> Optional[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []
    for split_dir, canon in (("train", "train"), ("valid", "val"), ("val", "val"), ("test", "test")):
        images = dataset_path / split_dir / "images"
        labels = dataset_path / split_dir / "labels"
        if images.exists() and labels.exists() and _contains_dataset_images(images):
            candidates.append((split_dir, canon))
    if not candidates:
        return None
    train = next((split_dir for split_dir, canon in candidates if canon == "train"), candidates[0][0])
    val = next((split_dir for split_dir, canon in candidates if canon == "val"), train)
    return f"{train}/images", f"{val}/images"


def _build_data_yaml(dataset_path: Path, classes: list[str], out_path: Path) -> Path:
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists() and _existing_data_yaml_usable(data_yaml):
        return data_yaml

    images_train = dataset_path / "images" / "train"
    labels_train = dataset_path / "labels" / "train"
    images_val = dataset_path / "images" / "val"
    labels_val = dataset_path / "labels" / "val"

    if images_train.exists() and labels_train.exists():
        train_ref = "images/train"
        if images_val.exists() and labels_val.exists() and _contains_dataset_images(images_val):
            val_ref = "images/val"
        else:
            val_ref = "images/train"
    else:
        split_first_refs = _split_first_train_val_refs(dataset_path)
        if split_first_refs is not None:
            train_ref, val_ref = split_first_refs
        else:
            images_any = dataset_path / "images"
            labels_any = dataset_path / "labels"
            if not (images_any.exists() and labels_any.exists()):
                raise ValueError(
                    f"Dataset format invalid for {dataset_path}. "
                    "Expected either data.yaml or images/ + labels/ (with train[/val] splits)."
                )
            train_ref = "images"
            val_ref = "images"

    names_map = {idx: cls for idx, cls in enumerate(classes)}
    body_lines = [
        f"path: {dataset_path.resolve().as_posix()}",
        f"train: {train_ref}",
        f"val: {val_ref}",
        "names:",
    ]
    for idx, cls in names_map.items():
        body_lines.append(f"  {idx}: {cls}")
    out_path.write_text("\n".join(body_lines) + "\n", encoding="utf-8")
    return out_path


def _extract_last_results_csv_row(save_dir: Path) -> dict[str, Any]:
    csv_path = save_dir / "results.csv"
    if not csv_path.exists():
        return {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if isinstance(row, dict)]
    except Exception:
        return {}
    if not rows:
        return {}
    clean: dict[str, Any] = {}
    for key, value in rows[-1].items():
        v = (value or "").strip()
        if not v:
            continue
        try:
            clean[key] = float(v)
        except Exception:
            clean[key] = v
    return clean


_METRIC_KEYS: dict[str, tuple[str, ...]] = {
    "map50": ("metrics/mAP50(B)", "map50", "mAP50"),
    "map50_95": ("metrics/mAP50-95(B)", "map50_95", "mAP50-95", "map50-95"),
    "precision": ("metrics/precision(B)", "precision"),
    "recall": ("metrics/recall(B)", "recall"),
}
_QUALITY_STOP_METRICS = ("map50_95", "map50", "precision", "recall")


def _metric_from_metrics(metrics: Mapping[str, Any], metric: str) -> float | None:
    for key in _METRIC_KEYS.get(metric, ()):
        value = metrics.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return None


def _map50_from_metrics(metrics: dict[str, Any]) -> float | None:
    # Keep the old fallback to mAP50-95 for legacy runs that only persisted one
    # mAP-like field, but prefer true mAP50 when available.
    value = _metric_from_metrics(metrics, "map50")
    if value is not None:
        return value
    return _metric_from_metrics(metrics, "map50_95")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _first_float(row: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "on"}:
        return True
    if s in {"0", "false", "no", "off"}:
        return False
    return default


def _quality_stop_config(hyperparams: Mapping[str, Any]) -> dict[str, Any]:
    metric = str(hyperparams.get("quality_stop_metric") or "map50_95").strip()
    if metric not in _QUALITY_STOP_METRICS:
        metric = "map50_95"
    threshold = _as_float(hyperparams.get("quality_stop_threshold"))
    if threshold is None:
        threshold = 0.90
    threshold = min(1.0, max(0.0, threshold))
    min_epochs = _as_int(hyperparams.get("quality_stop_min_epochs"))
    if min_epochs is None:
        min_epochs = 5
    consecutive_epochs = _as_int(hyperparams.get("quality_stop_consecutive_epochs"))
    if consecutive_epochs is None:
        consecutive_epochs = 2
    rapid_clear_loss_ratio = _as_float(hyperparams.get("quality_stop_rapid_clear_loss_ratio"))
    if rapid_clear_loss_ratio is None:
        rapid_clear_loss_ratio = 0.35
    rapid_clear_loss_ratio = min(1.0, max(0.0, rapid_clear_loss_ratio))
    rapid_clear_metric_margin = _as_float(hyperparams.get("quality_stop_rapid_clear_metric_margin"))
    if rapid_clear_metric_margin is None:
        rapid_clear_metric_margin = 0.03
    rapid_clear_metric_margin = min(1.0, max(0.0, rapid_clear_metric_margin))
    regression_abs_tolerance = _as_float(hyperparams.get("quality_regression_abs_tolerance"))
    if regression_abs_tolerance is None:
        regression_abs_tolerance = 0.05
    regression_abs_tolerance = min(1.0, max(0.0, regression_abs_tolerance))
    regression_rel_tolerance = _as_float(hyperparams.get("quality_regression_rel_tolerance"))
    if regression_rel_tolerance is None:
        regression_rel_tolerance = 0.15
    regression_rel_tolerance = min(1.0, max(0.0, regression_rel_tolerance))
    regression_consecutive_epochs = _as_int(hyperparams.get("quality_regression_consecutive_epochs"))
    if regression_consecutive_epochs is None:
        regression_consecutive_epochs = 1
    attempt_mode = _as_bool(hyperparams.get("quality_stop_attempt_mode"), False)
    max_time_seconds = _as_int(hyperparams.get("quality_stop_max_time_seconds")) or 0
    if max_time_seconds < 0:
        max_time_seconds = 0
    # Attempt mode is a one-switch feasibility probe: collapse min_epochs and
    # consecutive-epoch requirements so the moment the model clears threshold
    # we exit, and force the regression guard on with single-epoch sensitivity
    # so a noisy post-peak epoch pulls the ripcord. The operator's explicit
    # tolerance values are preserved.
    if attempt_mode:
        min_epochs = 1
        consecutive_epochs = 1
        regression_consecutive_epochs = 1
        rapid_clear_enabled_effective = True
        regression_enabled_effective = True
    else:
        rapid_clear_enabled_effective = _as_bool(
            hyperparams.get("quality_stop_rapid_clear_enabled"), True
        )
        regression_enabled_effective = _as_bool(
            hyperparams.get("quality_regression_enabled"), True
        )
    return {
        "enabled": _as_bool(hyperparams.get("quality_stop_enabled"), True),
        "metric": metric,
        "threshold": float(threshold),
        "min_epochs": max(1, int(min_epochs)),
        "consecutive_epochs": max(1, int(consecutive_epochs)),
        "rapid_clear_enabled": rapid_clear_enabled_effective,
        "rapid_clear_loss_ratio": float(rapid_clear_loss_ratio),
        "rapid_clear_metric_margin": float(rapid_clear_metric_margin),
        "regression_enabled": regression_enabled_effective,
        "regression_abs_tolerance": float(regression_abs_tolerance),
        "regression_rel_tolerance": float(regression_rel_tolerance),
        "regression_consecutive_epochs": max(1, int(regression_consecutive_epochs)),
        "attempt_mode": bool(attempt_mode),
        "max_time_seconds": int(max_time_seconds),
    }


def _evaluate_quality_stop(
    state: dict[str, Any],
    point: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    metric = str(config.get("metric") or "map50_95")
    epoch = _as_int(point.get("epoch"))
    value = _as_float(point.get(metric))
    train_loss = _as_float(point.get("train_loss"))
    threshold = _as_float(config.get("threshold"))
    if threshold is None:
        threshold = 0.90
    min_epochs = _as_int(config.get("min_epochs")) or 5
    required = _as_int(config.get("consecutive_epochs")) or 2
    rapid_clear_enabled = _as_bool(config.get("rapid_clear_enabled"), True)
    rapid_clear_loss_ratio = _as_float(config.get("rapid_clear_loss_ratio"))
    if rapid_clear_loss_ratio is None:
        rapid_clear_loss_ratio = 0.35
    rapid_clear_metric_margin = _as_float(config.get("rapid_clear_metric_margin"))
    if rapid_clear_metric_margin is None:
        rapid_clear_metric_margin = 0.03
    regression_enabled = _as_bool(config.get("regression_enabled"), True)
    regression_abs_tolerance = _as_float(config.get("regression_abs_tolerance"))
    if regression_abs_tolerance is None:
        regression_abs_tolerance = 0.05
    regression_rel_tolerance = _as_float(config.get("regression_rel_tolerance"))
    if regression_rel_tolerance is None:
        regression_rel_tolerance = 0.15
    regression_required = _as_int(config.get("regression_consecutive_epochs")) or 1
    if not _as_bool(config.get("enabled"), True):
        state["consecutive_epochs"] = 0
        state["regression_consecutive_epochs"] = 0
        return {
            "should_stop": False,
            "qualified": False,
            "consecutive_epochs": 0,
            "regression_consecutive_epochs": 0,
            "value": value,
            "mode": "disabled",
            "reason": "quality stop disabled",
        }
    if epoch is None or value is None:
        state["consecutive_epochs"] = 0
        state["regression_consecutive_epochs"] = 0
        return {
            "should_stop": False,
            "qualified": False,
            "consecutive_epochs": 0,
            "regression_consecutive_epochs": 0,
            "value": value,
            "mode": "waiting",
            "reason": f"{metric} not available",
        }

    if train_loss is not None and train_loss > 0 and _as_float(state.get("initial_train_loss")) is None:
        state["initial_train_loss"] = float(train_loss)

    previous_peak = _as_float(state.get("peak_value"))
    previous_peak_epoch = _as_int(state.get("peak_epoch"))
    if previous_peak is None or value > previous_peak:
        state["peak_value"] = float(value)
        state["peak_epoch"] = int(epoch)
        peak_value = float(value)
        peak_epoch = int(epoch)
    else:
        peak_value = float(previous_peak)
        peak_epoch = int(previous_peak_epoch if previous_peak_epoch is not None else epoch)

    max_time_seconds = _as_int(config.get("max_time_seconds")) or 0
    elapsed_seconds = _as_float(point.get("elapsed_seconds"))
    if (
        max_time_seconds > 0
        and elapsed_seconds is not None
        and elapsed_seconds >= float(max_time_seconds)
    ):
        reason = (
            f"time budget {max_time_seconds}s exhausted at epoch {epoch + 1}; "
            f"peak {peak_value:.4f} at epoch {peak_epoch + 1}"
        )
        return {
            "should_stop": True,
            "qualified": False,
            "consecutive_epochs": int(state.get("consecutive_epochs") or 0),
            "regression_consecutive_epochs": int(state.get("regression_consecutive_epochs") or 0),
            "value": value,
            "mode": "time_budget",
            "reason": reason,
            "peak_value": peak_value,
            "peak_epoch": peak_epoch,
            "recommended_max_epochs": peak_epoch + 1,
            "elapsed_seconds": float(elapsed_seconds),
            "max_time_seconds": int(max_time_seconds),
        }

    epoch_number = epoch + 1
    effective_min_epochs = min_epochs
    rapid_clear_triggered = False
    initial_train_loss = _as_float(state.get("initial_train_loss"))
    if (
        rapid_clear_enabled
        and epoch_number < min_epochs
        and train_loss is not None
        and initial_train_loss is not None
        and initial_train_loss > 0
        and value >= min(1.0, threshold + float(rapid_clear_metric_margin))
        and (train_loss / initial_train_loss) <= float(rapid_clear_loss_ratio)
    ):
        effective_min_epochs = 1
        rapid_clear_triggered = True

    qualified = epoch_number >= effective_min_epochs and value >= threshold
    if qualified:
        consecutive = int(state.get("consecutive_epochs") or 0) + 1
    else:
        consecutive = 0
    state["consecutive_epochs"] = consecutive
    state["last_epoch"] = epoch
    state["last_value"] = value
    threshold_stop = consecutive >= required

    regression_consecutive = 0
    regression_hit = False
    abs_drop = 0.0
    rel_drop = 0.0
    below_threshold = value < threshold
    if (
        regression_enabled
        and peak_epoch < epoch
        and epoch_number >= min_epochs
        and peak_value >= threshold
    ):
        abs_drop = max(0.0, peak_value - value)
        rel_drop = (abs_drop / peak_value) if peak_value > 0 else 0.0
        regression_hit = below_threshold and (
            abs_drop >= float(regression_abs_tolerance)
            or rel_drop >= float(regression_rel_tolerance)
        )
        regression_consecutive = int(state.get("regression_consecutive_epochs") or 0) + 1 if regression_hit else 0
    state["regression_consecutive_epochs"] = regression_consecutive
    regression_stop = regression_hit and regression_consecutive >= regression_required

    if threshold_stop:
        reason = (
            f"{metric} reached {value:.4f} >= {threshold:.4f} for "
            f"{consecutive} consecutive epochs after epoch {effective_min_epochs}"
        )
        if rapid_clear_triggered:
            reason += (
                f" (rapid-clear enabled: train_loss ratio {train_loss / initial_train_loss:.3f})"
            )
        return {
            "should_stop": True,
            "qualified": qualified,
            "consecutive_epochs": consecutive,
            "regression_consecutive_epochs": regression_consecutive,
            "value": value,
            "mode": "threshold",
            "reason": reason,
            "peak_value": peak_value,
            "peak_epoch": peak_epoch,
            "recommended_max_epochs": peak_epoch + 1,
            "effective_min_epochs": effective_min_epochs,
            "rapid_clear_triggered": rapid_clear_triggered,
        }

    if regression_stop:
        reason = (
            f"{metric} regressed from peak {peak_value:.4f} at epoch {peak_epoch + 1} "
            f"to {value:.4f} at epoch {epoch_number}; drop {abs_drop:.4f} "
            f"({rel_drop * 100.0:.1f}%) below guard tolerance"
        )
        return {
            "should_stop": True,
            "qualified": qualified,
            "consecutive_epochs": consecutive,
            "regression_consecutive_epochs": regression_consecutive,
            "value": value,
            "mode": "regression",
            "reason": reason,
            "peak_value": peak_value,
            "peak_epoch": peak_epoch,
            "recommended_max_epochs": peak_epoch + 1,
            "abs_drop": abs_drop,
            "rel_drop": rel_drop,
            "below_threshold": below_threshold,
            "effective_min_epochs": effective_min_epochs,
            "rapid_clear_triggered": rapid_clear_triggered,
        }

    if epoch_number < effective_min_epochs:
        reason = f"waiting for minimum epoch {effective_min_epochs}"
    elif value < threshold:
        reason = f"{metric} {value:.4f} below threshold {threshold:.4f}"
    elif rapid_clear_triggered:
        reason = f"{metric} qualified for {consecutive}/{required} consecutive epochs (rapid-clear active)"
    else:
        reason = f"{metric} qualified for {consecutive}/{required} consecutive epochs"
    return {
        "should_stop": False,
        "qualified": qualified,
        "consecutive_epochs": consecutive,
        "regression_consecutive_epochs": regression_consecutive,
        "value": value,
        "mode": "monitoring",
        "reason": reason,
        "peak_value": peak_value,
        "peak_epoch": peak_epoch,
        "recommended_max_epochs": peak_epoch + 1,
        "abs_drop": abs_drop,
        "rel_drop": rel_drop,
        "below_threshold": below_threshold,
        "effective_min_epochs": effective_min_epochs,
        "rapid_clear_triggered": rapid_clear_triggered,
    }


def _extract_training_point(
    row: dict[str, Any],
    *,
    epochs: int,
    run_dir: Path,
) -> dict[str, Any] | None:
    epoch = _as_int(row.get("epoch"))
    if epoch is None:
        return None
    map50 = _metric_from_metrics(row, "map50")
    map50_95 = _metric_from_metrics(row, "map50_95")
    if map50 is None:
        map50 = map50_95
    precision = _first_float(row, ("metrics/precision(B)", "precision"))
    recall = _first_float(row, ("metrics/recall(B)", "recall"))
    t_box = _first_float(row, ("train/box_loss",))
    t_cls = _first_float(row, ("train/cls_loss",))
    t_dfl = _first_float(row, ("train/dfl_loss",))
    v_box = _first_float(row, ("val/box_loss",))
    v_cls = _first_float(row, ("val/cls_loss",))
    v_dfl = _first_float(row, ("val/dfl_loss",))

    train_loss_parts = [v for v in (t_box, t_cls, t_dfl) if v is not None]
    val_loss_parts = [v for v in (v_box, v_cls, v_dfl) if v is not None]
    train_loss = float(sum(train_loss_parts)) if train_loss_parts else None
    val_loss = float(sum(val_loss_parts)) if val_loss_parts else None
    progress = min(100.0, max(0.0, ((epoch + 1) / max(1, epochs)) * 100.0))
    return {
        "event": "epoch",
        "epoch": epoch,
        "epochs": epochs,
        "progress": round(progress, 2),
        "map50": map50,
        "map50_95": map50_95,
        "precision": precision,
        "recall": recall,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "run_dir": str(run_dir),
        "timestamp": time.time(),
    }


def _monitor_training_progress(
    *,
    csv_path: Path,
    epochs: int,
    run_dir: Path,
    stop_event: threading.Event,
    callback: Callable[[dict[str, Any]], None],
) -> None:
    seen_epochs: set[int] = set()
    while not stop_event.is_set():
        try:
            rows = _extract_all_rows(csv_path)
            for row in rows:
                point = _extract_training_point(row, epochs=epochs, run_dir=run_dir)
                if point is None:
                    continue
                epoch = int(point.get("epoch", -1))
                if epoch in seen_epochs:
                    continue
                seen_epochs.add(epoch)
                callback(point)
        except Exception:
            # Training telemetry must not break training itself.
            pass
        stop_event.wait(0.45)

    # Final flush after training exits.
    try:
        rows = _extract_all_rows(csv_path)
        for row in rows:
            point = _extract_training_point(row, epochs=epochs, run_dir=run_dir)
            if point is None:
                continue
            epoch = int(point.get("epoch", -1))
            if epoch in seen_epochs:
                continue
            seen_epochs.add(epoch)
            callback(point)
    except Exception:
        return


def _extract_all_rows(csv_path: Path) -> list[dict[str, Any]]:
    if not csv_path.exists():
        return []
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            # YOLO pads column names with whitespace (e.g. "                   epoch").
            # Strip keys so lookups like row.get("epoch") work correctly.
            return [{k.strip(): v for k, v in row.items()} for row in reader if isinstance(row, dict)]
    except Exception:
        return []


def _resolve_training_device(
    system_specs: Optional[dict[str, Any]] = None,
    *,
    override: Optional[str] = None,
) -> str:
    """Return the explicit Ultralytics training device for this host.

    We always pass a concrete device into `model.train(...)`, even for resume
    runs, so Ultralytics does not reuse a stale `device=0` from an older
    checkpoint on a host that only has CPU or MPS available.

    A user-supplied ``override`` (e.g. "cpu", "mps", "0", "1") wins if it
    matches an available device on this host; otherwise we fall back to
    the auto-detected accelerator.
    """
    override_s = str(override or "").strip().lower()
    if override_s:
        if isinstance(system_specs, dict):
            accel = str(system_specs.get("accelerator") or "").strip().lower()
            try:
                gpu_count = int(system_specs.get("gpu_count") or 0)
            except Exception:
                gpu_count = 0
            if override_s == "cpu":
                return "cpu"
            if override_s == "mps" and accel == "mps":
                return "mps"
            if override_s.isdigit() and accel == "cuda" and int(override_s) < gpu_count:
                return override_s
        else:
            if override_s in ("cpu", "mps") or override_s.isdigit():
                return override_s

    if isinstance(system_specs, dict):
        accelerator = str(system_specs.get("accelerator") or "").strip().lower()
        try:
            gpu_count = int(system_specs.get("gpu_count") or 0)
        except Exception:
            gpu_count = 0
        if accelerator == "cuda" and gpu_count > 0:
            return "0"
        if accelerator == "mps":
            return "mps"
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available() and int(torch.cuda.device_count() or 0) > 0:
            return "0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _is_no_space_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    if "no space left on device" in text or "disk full" in text or "enospc" in text:
        return True
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, OSError) and getattr(current, "errno", None) == 28:
            return True
        current = current.__cause__ if isinstance(current.__cause__, BaseException) else None
    return False


def _pin_training_process_caches(asset_root: Path) -> dict[str, str]:
    """Route training temp/config/cache paths under the active asset root.

    Called before importing/constructing Ultralytics so hub weights, settings,
    temporary files, and generated kernels land on the same volume as
    checkpoints when possible, instead of silently filling the boot disk.
    """
    try:
        cache_root = (asset_root / ".cvlayer_train_process_cache").resolve()
        cache_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return {}
    torch_home = cache_root / "torch"
    xdg = cache_root / "xdg_cache"
    mpl = cache_root / "matplotlib"
    tmp = cache_root / "tmp"
    ultralytics_cfg = cache_root / "ultralytics"
    hf_home = cache_root / "huggingface"
    torch_extensions = cache_root / "torch_extensions"
    for p in (torch_home, xdg, mpl, tmp, ultralytics_cfg, hf_home, torch_extensions):
        try:
            p.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
    pinned = {
        "TORCH_HOME": str(torch_home),
        "XDG_CACHE_HOME": str(xdg),
        "MPLCONFIGDIR": str(mpl),
        "TMPDIR": str(tmp),
        "TEMP": str(tmp),
        "TMP": str(tmp),
        "YOLO_CONFIG_DIR": str(ultralytics_cfg),
        "ULTRALYTICS_CONFIG_DIR": str(ultralytics_cfg),
        "HF_HOME": str(hf_home),
        "TRANSFORMERS_CACHE": str(hf_home / "transformers"),
        "TORCH_EXTENSIONS_DIR": str(torch_extensions),
    }
    for key, value in pinned.items():
        os.environ[key] = value
    tempfile.tempdir = str(tmp)
    return pinned


def run_training(
    scenario: str,
    *,
    trainer_override: str | None = None,
    base_model_override: str | None = None,
    epochs_override: int | None = None,
    imgsz_override: int | None = None,
    checkpoint_period_override: int | None = None,
    seed_override: int | None = None,
    deterministic: bool = True,
    resume: bool = True,
    auto_fresh_on_completed_resume: bool = True,
    final_model_name: str = "",
    progress_callback: Optional[Callable[[dict[str, Any]], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
    hyperparams_overrides: Optional[Mapping[str, Any]] = None,
    _overflow_carryover: Optional[dict[str, str]] = None,
    _overflow_exclude_roots: Optional[list[str]] = None,
    _capture_streams: bool = True,
) -> dict[str, Any]:
    def _emit_status(line: str) -> None:
        # Surface pre-train setup phases so the operator sees activity in the
        # training console during the otherwise-blank cold-start window
        # (ultralytics import + model load + dataset scan).
        if progress_callback is None:
            return
        try:
            progress_callback(
                {
                    "event": "log",
                    "line": line,
                    "stream": "stdout",
                    "timestamp": time.time(),
                }
            )
        except Exception:
            pass

    training_started_at = datetime.now(timezone.utc)
    training_started_monotonic = time.monotonic()
    _emit_status(f"[trainer] starting scenario={scenario}")
    cfg = get_scenario_config(scenario)
    try:
        ci_cd_policy = get_scenario_ci_cd_policy(cfg.name)
    except Exception:
        ci_cd_policy = {"enabled": False}
    ci_cd_enabled = bool(ci_cd_policy.get("enabled"))
    _emit_status(
        f"[trainer] dataset: {cfg.dataset} ({len(cfg.classes)} classes)"
    )
    trainer = validate_trainer_name(
        trainer_override if trainer_override is not None else cfg.hyperparams.get("trainer", "ultralytics_yolo")
    )
    _emit_status(f"[trainer] backend={trainer}")
    _emit_status("[trainer] resolving base model (~1s)")
    try:
        has_existing_weights = cfg.weights_path.exists() and cfg.weights_path.stat().st_size > 1024
    except Exception:
        has_existing_weights = False
    # If the scenario config was written after the weights were produced, the
    # user changed settings (most likely base_model via Apply Model) since the
    # last training run — honour cfg.base_model instead of resuming from the
    # old trained weights.
    try:
        _cfg_newer_than_weights = (
            has_existing_weights
            and cfg.config_path.stat().st_mtime > cfg.weights_path.stat().st_mtime
        )
    except Exception:
        _cfg_newer_than_weights = False
    parent_version_id: str | None = None
    if base_model_override is not None and str(base_model_override).strip():
        base_model, parent_version_id = _resolve_base_model(
            str(base_model_override).strip(), scenario=cfg.name
        )
    elif has_existing_weights and not _cfg_newer_than_weights:
        base_model = str(cfg.weights_path)
    else:
        base_model, parent_version_id = _resolve_base_model(cfg.base_model, scenario=cfg.name)
    _emit_status(f"[trainer] base model: {base_model}")

    hyperparams = dict(cfg.hyperparams)
    if isinstance(hyperparams_overrides, Mapping):
        for key, value in hyperparams_overrides.items():
            key_s = str(key or "").strip()
            if not key_s:
                continue
            if value is None:
                continue
            if isinstance(value, str) and not value.strip():
                continue
            hyperparams[key_s] = value
    epochs = int(epochs_override if epochs_override is not None else hyperparams.get("epochs", 20))
    imgsz = int(imgsz_override if imgsz_override is not None else hyperparams.get("imgsz", 640))
    checkpoint_period = int(
        checkpoint_period_override
        if checkpoint_period_override is not None
        else hyperparams.get("save_period", 1)
    )
    seed = int(seed_override if seed_override is not None else hyperparams.get("seed", 42))
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if imgsz <= 0:
        raise ValueError(f"imgsz must be positive, got {imgsz}")
    if checkpoint_period <= 0:
        raise ValueError(f"save_period must be positive, got {checkpoint_period}")
    hyperparams["epochs"] = epochs
    hyperparams["imgsz"] = imgsz
    hyperparams["save_period"] = checkpoint_period
    hyperparams["seed"] = seed
    hyperparams["deterministic"] = bool(deterministic)
    _emit_status(
        f"[trainer] schedule: epochs={epochs} imgsz={imgsz} "
        f"batch={hyperparams.get('batch', 'auto')} save_period={checkpoint_period}"
    )

    _emit_status("[trainer] snapshotting dataset (~2-10s, scales with file count)")
    dataset_snapshot = create_dataset_snapshot(cfg.dataset, cfg.dataset_path, cfg.classes)
    snapshot_path = persist_dataset_snapshot(dataset_snapshot)
    _emit_status(
        f"[trainer] dataset snapshot ready (id={dataset_snapshot.get('id') or 'n/a'})"
    )
    contract = dataset_snapshot.get("contract") if isinstance(dataset_snapshot.get("contract"), dict) else {}
    if str(contract.get("status") or "") == "failed":
        issues = contract.get("issues") if isinstance(contract.get("issues"), list) else []
        detail = "; ".join(str(i) for i in issues) if issues else "dataset contract failed"
        raise ValueError(f"Dataset contract validation failed: {detail}")
    _emit_status("[trainer] running system guard probe (~1-3s: disk, RAM, GPU)")
    training_guard = build_training_guard(
        base_model,
        hyperparams,
        scenario=cfg.name,
        exclude_asset_roots=_overflow_exclude_roots,
    )
    _emit_status("[trainer] guard cleared")
    overflow_protocol = (
        training_guard.get("overflow_protocol") if isinstance(training_guard.get("overflow_protocol"), dict) else {}
    )
    if overflow_protocol.get("status") == "no_space":
        drive_lines = []
        for drive in overflow_protocol.get("drives") or []:
            if not isinstance(drive, dict):
                continue
            drive_lines.append(
                f"{drive.get('label')}: {drive.get('free_gb')} GB free / {drive.get('total_gb')} GB total "
                f"at {drive.get('asset_root')}"
            )
        for vol in overflow_protocol.get("volume_inventory") or []:
            if not isinstance(vol, dict):
                continue
            drive_lines.append(
                f"{vol.get('probe')}: {vol.get('free_gb')} GB free / {vol.get('total_gb')} GB total "
                f"(used {vol.get('used_pct')}%)"
            )
        detail = "\n".join(drive_lines)
        raise RuntimeError(f"{overflow_protocol.get('message')}\n{detail}".strip())

    primary_model_root = MLOPS_ROOT / "models" / cfg.name
    active_asset_root = str(overflow_protocol.get("active_asset_root") or "").strip()
    model_root = Path(active_asset_root) if active_asset_root else primary_model_root

    pinned_cache_paths = _pin_training_process_caches(model_root)

    carry = dict(_overflow_carryover or {})
    run_name = str(carry.get("run_name") or "").strip()
    weights_src = Path(str(carry.get("weights_pt") or ""))
    carry_ok = bool(run_name and weights_src.is_file())

    resume_checkpoint_info: tuple[Path, Path] | None = None
    resumed_from = ""
    run_dir: Path
    resume_checkpoint: Path | None = None

    if carry_ok:
        run_dir = model_root / run_name
        dest_ckpt = run_dir / "weights" / "last.pt"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            shutil.copy2(weights_src, dest_ckpt)
            src_run = weights_src.parent.parent
            for extra in ("args.yaml", "results.csv"):
                src_f = src_run / extra
                if src_f.is_file():
                    shutil.copy2(src_f, run_dir / extra)
            resume_checkpoint = dest_ckpt
            resume_checkpoint_info = (run_dir, dest_ckpt)
            resumed_from = f"{weights_src} -> {dest_ckpt}"
        except Exception as exc:
            resume_checkpoint_info = None
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "event": "log",
                            "line": f"[overflow-protocol] checkpoint relocation failed: {exc}",
                            "stream": "stderr",
                            "timestamp": time.time(),
                        }
                    )
                except Exception:
                    pass

    if resume_checkpoint_info is None:
        resume_checkpoint_info = _find_latest_resume_checkpoint(model_root) if resume else None
        if resume_checkpoint_info is not None:
            run_dir, resume_checkpoint = resume_checkpoint_info
            resumed_from = str(resume_checkpoint)
        else:
            run_dir = _next_run_dir_across(primary_model_root, model_root)
            resume_checkpoint = None
    run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_local_run_link(primary_model_root / run_dir.name, run_dir)
    _emit_status(
        f"[trainer] run directory ready: {run_dir.name}"
        + (f" (resuming from {resumed_from})" if resumed_from else " (fresh run)")
    )

    _emit_status(f"[trainer] preparing dataset yaml ({cfg.dataset_path})")
    data_yaml = _build_data_yaml(cfg.dataset_path, cfg.classes, run_dir / "data.generated.yaml")
    effective_hyperparams = dict(training_guard.get("effective_hyperparams") or {})
    runtime_device = _resolve_training_device(
        training_guard.get("system_specs") if isinstance(training_guard, dict) else None,
        override=str(hyperparams.get("device") or ""),
    )
    _emit_status(f"[trainer] device={runtime_device}")
    epochs = int(effective_hyperparams.get("epochs") or epochs)
    imgsz = int(effective_hyperparams.get("imgsz") or imgsz)
    quality_stop = _quality_stop_config(hyperparams)
    quality_stop_state: dict[str, Any] = {
        "consecutive_epochs": 0,
        "start_time": time.time(),
    }
    quality_stop_result: dict[str, Any] = {
        "enabled": bool(quality_stop.get("enabled")),
        "triggered": False,
        "config": dict(quality_stop),
        "attempt_mode": bool(quality_stop.get("attempt_mode")),
    }

    global YOLO
    if YOLO is None:
        _emit_status("[trainer] importing ultralytics (first run can take ~10-30s)")
        from ultralytics import YOLO as _YOLO

        YOLO = _YOLO
        _emit_status("[trainer] ultralytics ready")

    # Resume must load the model FROM the checkpoint, not from the base model.
    # Ultralytics ignores any resume=<path> we pass and overwrites it with the
    # loaded model's own ckpt_path (engine/model.py: `args["resume"] =
    # self.ckpt_path`). If we load the base model here, it resumes from the base
    # weights instead of last.pt -- and the published yolov8n.pt carries stale
    # COCO train_args (v5loader/image_weights/fl_gamma, epochs=500, project=YOLOv8)
    # that current Ultralytics rejects in get_cfg, crashing every resume and (when
    # it did not crash) writing runs to the wrong directory. The canonical resume
    # is `YOLO(last.pt).train(resume=True)`, so load from the resume checkpoint.
    if resume_checkpoint is not None:
        _emit_status(f"[trainer] loading resume checkpoint: {resume_checkpoint}")
        model = YOLO(str(resume_checkpoint))
        _emit_status("[trainer] resume checkpoint loaded")
    else:
        _emit_status(f"[trainer] loading base model: {base_model}")
        model = YOLO(base_model)
        _emit_status("[trainer] base model loaded")
    if cancel_check is not None:
        # Ultralytics only consults `trainer.stop` at epoch boundaries, so a
        # batch-end set won't abort mid-epoch. To make Stop responsive during
        # iterative testing we raise from every cheap callback Ultralytics
        # fires — including val batches and pretrain hooks — so cancellation
        # lands at the next Python bytecode boundary rather than the next
        # epoch boundary. The exception propagates through model.train(...)
        # and is normalized to "training cancelled by operator" by the
        # post-run cancel check.
        def _stop_if_cancelled(trainer: Any) -> None:
            try:
                cancelled = bool(cancel_check())
            except Exception:
                cancelled = True
            if cancelled:
                try:
                    trainer.stop = True
                except Exception:
                    pass
                raise RuntimeError("training cancelled by operator")

        for _evt in (
            "on_pretrain_routine_start",
            "on_pretrain_routine_end",
            "on_train_epoch_start",
            "on_train_batch_start",
            "on_train_batch_end",
            "on_train_epoch_end",
            "on_val_start",
            "on_val_batch_start",
            "on_val_batch_end",
            "on_val_end",
            "on_fit_epoch_end",
        ):
            model.add_callback(_evt, _stop_if_cancelled)

    if bool(quality_stop.get("enabled")):
        def _stop_if_quality_target(trainer: Any) -> None:
            if bool(quality_stop_result.get("triggered")):
                try:
                    trainer.stop = True
                except Exception:
                    pass
                return
            metrics = getattr(trainer, "metrics", None)
            if not isinstance(metrics, Mapping):
                return
            epoch = _as_int(getattr(trainer, "epoch", None))
            if epoch is None:
                return
            point = _extract_training_point(
                {"epoch": epoch, **dict(metrics)},
                epochs=epochs,
                run_dir=run_dir,
            )
            if point is None:
                return
            start_time = _as_float(quality_stop_state.get("start_time"))
            if start_time is not None:
                point = {**point, "elapsed_seconds": max(0.0, time.time() - start_time)}
            decision = _evaluate_quality_stop(quality_stop_state, point, quality_stop)
            quality_stop_result.update(
                {
                    "last_epoch": epoch,
                    "last_value": decision.get("value"),
                    "consecutive_epochs": decision.get("consecutive_epochs"),
                    "regression_consecutive_epochs": decision.get("regression_consecutive_epochs"),
                    "reason": decision.get("reason"),
                    "mode": decision.get("mode"),
                    "peak_epoch": decision.get("peak_epoch"),
                    "peak_value": decision.get("peak_value"),
                    "recommended_max_epochs": decision.get("recommended_max_epochs"),
                }
            )
            if not bool(decision.get("should_stop")):
                return
            try:
                trainer.stop = True
            except Exception:
                pass
            reason = str(decision.get("reason") or "quality target reached")
            event = {
                "event": "quality_stop",
                "mode": str(decision.get("mode") or "threshold"),
                "metric": str(quality_stop.get("metric") or "map50_95"),
                "value": decision.get("value"),
                "threshold": float(quality_stop.get("threshold") or 0.90),
                "epoch": epoch,
                "epochs": epochs,
                "progress": point.get("progress"),
                "consecutive_epochs": int(decision.get("consecutive_epochs") or 0),
                "regression_consecutive_epochs": int(decision.get("regression_consecutive_epochs") or 0),
                "peak_epoch": decision.get("peak_epoch"),
                "peak_value": decision.get("peak_value"),
                "recommended_max_epochs": decision.get("recommended_max_epochs"),
                "reason": reason,
                "run_dir": str(run_dir),
                "timestamp": time.time(),
            }
            quality_stop_result.update(
                {
                    "triggered": True,
                    "event": event,
                    "mode": event["mode"],
                    "metric": event["metric"],
                    "value": event["value"],
                    "threshold": event["threshold"],
                    "epoch": epoch,
                    "reason": reason,
                    "peak_epoch": event["peak_epoch"],
                    "peak_value": event["peak_value"],
                    "recommended_max_epochs": event["recommended_max_epochs"],
                }
            )
            if progress_callback is not None:
                try:
                    progress_callback(event)
                    progress_callback(
                        {
                            "event": "log",
                            "line": f"[quality-stop] {reason}",
                            "stream": "stdout",
                            "timestamp": time.time(),
                        }
                    )
                except Exception:
                    pass

        model.add_callback("on_fit_epoch_end", _stop_if_quality_target)

    # Dataloader-stall instrumentation. We time the gap between the end of
    # one train step and the start of the next — that gap is dominated by
    # waiting on the dataloader. The ratio stall / (stall + step) tells the
    # user how much of wall-time the GPU spent idle waiting for data.
    if progress_callback is not None:
        stall_state: dict[str, Any] = {
            "last_end": None,
            "last_start": None,
            "step_count": 0,
            "data_sum": 0.0,
            "step_sum": 0.0,
            "emit_every": 20,
        }

        def _on_batch_start(trainer: Any) -> None:
            now = time.perf_counter()
            last_end = stall_state.get("last_end")
            if isinstance(last_end, float):
                stall_state["data_sum"] = float(stall_state["data_sum"]) + max(0.0, now - last_end)
            stall_state["last_start"] = now

        def _on_batch_end(trainer: Any) -> None:
            now = time.perf_counter()
            last_start = stall_state.get("last_start")
            if isinstance(last_start, float):
                stall_state["step_sum"] = float(stall_state["step_sum"]) + max(0.0, now - last_start)
            stall_state["last_end"] = now
            stall_state["step_count"] = int(stall_state["step_count"]) + 1
            emit_every = int(stall_state.get("emit_every") or 20)
            if stall_state["step_count"] % emit_every != 0:
                return
            data_sum = float(stall_state["data_sum"])
            step_sum = float(stall_state["step_sum"])
            total = data_sum + step_sum
            stall_pct = (data_sum / total * 100.0) if total > 0 else 0.0
            step_ms = (step_sum / max(1, emit_every)) * 1000.0
            data_ms = (data_sum / max(1, emit_every)) * 1000.0
            batch_size = 0
            try:
                batch_size = int(getattr(trainer.args, "batch", 0) or 0)
            except Exception:
                batch_size = 0
            samples_per_sec = 0.0
            if total > 0 and batch_size > 0:
                samples_per_sec = (emit_every * batch_size) / total
            try:
                progress_callback(
                    {
                        "event": "batch_metrics",
                        "epoch": int(getattr(trainer, "epoch", 0) or 0),
                        "window": emit_every,
                        "stall_pct": round(stall_pct, 1),
                        "step_time_ms": round(step_ms, 1),
                        "data_time_ms": round(data_ms, 1),
                        "samples_per_sec": round(samples_per_sec, 1),
                        "timestamp": time.time(),
                    }
                )
            except Exception:
                pass
            # Reset the rolling window so each emission reflects the last N steps.
            stall_state["data_sum"] = 0.0
            stall_state["step_sum"] = 0.0

        model.add_callback("on_train_batch_start", _on_batch_start)
        model.add_callback("on_train_batch_end", _on_batch_end)
    deterministic_state: dict[str, Any] = {}
    if deterministic:
        deterministic_state = apply_deterministic_policy(seed)
    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml),
        "project": str(model_root),
        "name": run_dir.name,
        "exist_ok": True,
        "epochs": epochs,
        "imgsz": imgsz,
        "save": True,
        "save_period": checkpoint_period,
        "seed": seed,
        "deterministic": bool(deterministic),
        "verbose": False,
        "device": runtime_device,
    }
    # Pass through numeric hyperparameters when available.
    _NUMERIC_KEYS = (
        # schedule
        "batch", "patience", "workers", "close_mosaic",
        # optimizer
        "lr0", "lrf", "momentum", "weight_decay",
        "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
        # regularization
        "dropout", "label_smoothing",
        # augmentation
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale",
        "shear", "perspective", "fliplr", "flipud",
        "mosaic", "mixup", "copy_paste", "erasing",
    )
    _BOOL_KEYS = ("cos_lr", "amp")
    _STR_KEYS = ("optimizer",)
    for key in _NUMERIC_KEYS:
        value = hyperparams.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            train_kwargs[key] = value
    for key in _BOOL_KEYS:
        value = hyperparams.get(key)
        if isinstance(value, bool):
            train_kwargs[key] = value
    for key in _STR_KEYS:
        value = hyperparams.get(key)
        if isinstance(value, str) and value:
            train_kwargs[key] = value
    # save_period from hyperparams overrides the per-call checkpoint_period when set.
    if isinstance(hyperparams.get("save_period"), int):
        train_kwargs["save_period"] = int(hyperparams["save_period"])
    for key in ("batch", "workers"):
        value = effective_hyperparams.get(key)
        if isinstance(value, int):
            train_kwargs[key] = value
    if resume_checkpoint is not None:
        train_kwargs["resume"] = str(resume_checkpoint)
    if progress_callback is not None:
        try:
            progress_callback(
                {
                    "event": "start",
                    "epoch": -1,
                    "epochs": epochs,
                    "progress": 0.0,
                    "run_dir": str(run_dir),
                    "asset_root": str(model_root),
                    "resume": bool(resume_checkpoint is not None),
                    "resumed_from": resumed_from,
                    "save_period": checkpoint_period,
                    "seed": seed,
                    "deterministic": bool(deterministic),
                    "device": runtime_device,
                    "dataset_snapshot_id": str(dataset_snapshot.get("snapshot_id") or ""),
                    "dataset_snapshot_path": str(snapshot_path),
                    "trainer": trainer,
                    "quality_stop": quality_stop,
                    "pinned_cache_paths": pinned_cache_paths,
                    "training_guard": training_guard,
                    "requested_hyperparams": training_guard.get("requested_hyperparams"),
                    "effective_hyperparams": effective_hyperparams,
                    "timestamp": time.time(),
                }
            )
        except Exception:
            pass
        for preamble in (
            f"[system-guard] {training_guard.get('summary') or ''}",
            f"[system-guard] status={training_guard.get('status')}  model_scale={training_guard.get('model_scale')}",
            f"[overflow-protocol] {overflow_protocol.get('message') or ''}",
            f"[overflow-protocol] asset_root={model_root}",
            f"[overflow-protocol] tmpdir={pinned_cache_paths.get('TMPDIR', os.environ.get('TMPDIR', ''))}",
            f"[overflow-protocol] yolo_config={pinned_cache_paths.get('YOLO_CONFIG_DIR', os.environ.get('YOLO_CONFIG_DIR', ''))}",
            f"[overflow-protocol] torch_home={pinned_cache_paths.get('TORCH_HOME', os.environ.get('TORCH_HOME', ''))}",
            f"[system-guard] run_dir={run_dir}",
            (
                f"[train] resume checkpoint={resume_checkpoint}"
                if resume_checkpoint is not None
                else "[train] resume checkpoint=none (starting fresh run)"
            ),
            f"[train] checkpoint save_period={checkpoint_period}",
            f"[train] seed={seed} deterministic={bool(deterministic)}",
            f"[train] device={runtime_device}",
            f"[train] dataset_snapshot={dataset_snapshot.get('snapshot_id')}",
            f"[train] trainer={trainer}",
        ):
            try:
                progress_callback(
                    {
                        "event": "log",
                        "line": preamble,
                        "stream": "stdout",
                        "timestamp": time.time(),
                    }
                )
            except Exception:
                pass
        for adj in (training_guard.get("adjustments") or []):
            try:
                progress_callback(
                    {
                        "event": "log",
                        "line": f"[system-guard] adjust: {adj}",
                        "stream": "stdout",
                        "timestamp": time.time(),
                    }
                )
            except Exception:
                pass

    verdict_tracker = TrainingVerdict()

    def _verdict_wrap(cb: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
        def _wrapped(point: dict[str, Any]) -> None:
            try:
                if isinstance(point, dict) and point.get("event") == "epoch":
                    v = verdict_tracker.update(point)
                    point["verdict"] = v.get("label")
                    point["verdict_reason"] = v.get("reason")
                    cb(point)
                    cb(v)
                    return
            except Exception:
                pass
            cb(point)
        return _wrapped

    monitor_stop = threading.Event()
    monitor_thread: Optional[threading.Thread] = None
    if progress_callback is not None:
        monitor_thread = threading.Thread(
            target=_monitor_training_progress,
            kwargs={
                "csv_path": run_dir / "results.csv",
                "epochs": epochs,
                "run_dir": run_dir,
                "stop_event": monitor_stop,
                "callback": _verdict_wrap(progress_callback),
            },
            daemon=True,
            name=f"TrainProgress-{scenario}",
        )
        monitor_thread.start()

    _emit_status(
        f"[trainer] launching ultralytics training "
        f"(epochs={epochs}, imgsz={imgsz}, device={runtime_device})"
    )
    train_exc: BaseException | None = None
    try:
        capture_ctx = (
            _capture_training_logs(progress_callback)
            if _capture_streams
            else _capture_training_logs(progress_callback, tee_streams=False)
        )
        with capture_ctx:
            results = model.train(**train_kwargs)
    except BaseException as exc:
        train_exc = exc
    finally:
        monitor_stop.set()
        if monitor_thread is not None and monitor_thread.is_alive():
            monitor_thread.join(timeout=1.2)
    if train_exc is not None:
        if (
            auto_fresh_on_completed_resume
            and resume_checkpoint is not None
            and _is_completed_resume_error(train_exc)
        ):
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "event": "log",
                            "line": (
                                "[train] resume checkpoint is already complete; "
                                "clearing resume state and starting a fresh run"
                            ),
                            "stream": "stdout",
                            "timestamp": time.time(),
                        }
                    )
                    progress_callback(
                        {
                            "event": "restart_fresh",
                            "epoch": -1,
                            "epochs": epochs,
                            "progress": 0.0,
                            "run_dir": str(run_dir),
                            "resumed_from": resumed_from,
                            "error": str(train_exc),
                            "timestamp": time.time(),
                        }
                    )
                except Exception:
                    pass
            return run_training(
                scenario,
                trainer_override=trainer_override,
                base_model_override=base_model_override,
                epochs_override=epochs_override,
                imgsz_override=imgsz_override,
                checkpoint_period_override=checkpoint_period_override,
                seed_override=seed_override,
                deterministic=deterministic,
                resume=False,
                auto_fresh_on_completed_resume=False,
                final_model_name=final_model_name,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                hyperparams_overrides=hyperparams_overrides,
                _overflow_carryover=None,
                _overflow_exclude_roots=_overflow_exclude_roots,
                _capture_streams=_capture_streams,
            )
        if _is_no_space_error(train_exc):
            excluded = list(_overflow_exclude_roots or [])
            try:
                excluded.append(str(model_root.resolve()))
            except Exception:
                excluded.append(str(model_root))
            carry_weights: Path | None = None
            for fname in ("last.pt", "best.pt"):
                cand = run_dir / "weights" / fname
                try:
                    if cand.is_file() and cand.stat().st_size > 1024:
                        carry_weights = cand
                        break
                except Exception:
                    continue
            carry_payload: dict[str, str] | None = None
            resume_after_overflow = False
            if carry_weights is not None:
                carry_payload = {"run_name": run_dir.name, "weights_pt": str(carry_weights)}
                resume_after_overflow = True
            if progress_callback is not None:
                try:
                    progress_callback(
                        {
                            "event": "overflow_switch",
                            "epoch": -1,
                            "epochs": epochs,
                            "progress": 0.0,
                            "run_dir": str(run_dir),
                            "failed_asset_root": str(model_root),
                            "message": (
                                "Disk full during training; overflow protocol is relocating checkpoints "
                                "to the next volume with free space and resuming if a partial "
                                f"checkpoint was saved ({carry_weights or 'none found'})."
                            ),
                            "carry_checkpoint": str(carry_weights) if carry_weights else "",
                            "timestamp": time.time(),
                        }
                    )
                    progress_callback(
                        {
                            "event": "log",
                            "line": (
                                "[overflow-protocol] data overflow encountered; "
                                "retrying training on the next storage location with room"
                            ),
                            "stream": "stderr",
                            "timestamp": time.time(),
                        }
                    )
                except Exception:
                    pass
            return run_training(
                scenario,
                trainer_override=trainer_override,
                base_model_override=base_model_override,
                epochs_override=epochs_override,
                imgsz_override=imgsz_override,
                checkpoint_period_override=checkpoint_period_override,
                seed_override=seed_override,
                deterministic=deterministic,
                resume=resume_after_overflow,
                auto_fresh_on_completed_resume=auto_fresh_on_completed_resume,
                final_model_name=final_model_name,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                hyperparams_overrides=hyperparams_overrides,
                _overflow_carryover=carry_payload,
                _overflow_exclude_roots=excluded,
                _capture_streams=_capture_streams,
            )
        raise train_exc

    if cancel_check is not None:
        try:
            if cancel_check():
                raise RuntimeError("training cancelled by operator")
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError("training cancelled by operator") from exc

    save_dir = Path(getattr(results, "save_dir", run_dir) or run_dir)
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"
    if best_pt.exists():
        source_weights = best_pt
    elif last_pt.exists():
        source_weights = last_pt
    else:
        raise RuntimeError(f"Training finished without output weights in {save_dir / 'weights'}")

    run_weights = run_dir / "weights.pt"
    if source_weights.resolve() != run_weights.resolve():
        shutil.copy2(source_weights, run_weights)
    cfg.weights_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        not ci_cd_enabled
        and cfg.weights_path.resolve() != run_weights.resolve()
        and not bool(overflow_protocol.get("overflowed"))
    ):
        shutil.copy2(run_weights, cfg.weights_path)
    final_model_name = str(final_model_name or "").strip()
    final_model_file = ""
    final_model_path = ""
    if final_model_name:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", final_model_name).strip("._-")
        if safe_name:
            final_model_file = f"{safe_name}{run_weights.suffix or '.pt'}"
            named_weights = run_dir / final_model_file
            if named_weights.resolve() != run_weights.resolve():
                shutil.copy2(run_weights, named_weights)
            final_model_path = str(named_weights)

    results_dict = getattr(results, "results_dict", None)
    metrics = dict(results_dict) if isinstance(results_dict, dict) else {}
    csv_last = _extract_last_results_csv_row(save_dir)
    for key, value in csv_last.items():
        metrics.setdefault(key, value)
    map50 = _map50_from_metrics(metrics)
    map50_95 = _metric_from_metrics(metrics, "map50_95")
    final_epoch = _as_int(metrics.get("epoch"))
    if final_epoch is None:
        final_epoch = epochs - 1
    quality_stop_result.setdefault("last_epoch", final_epoch)
    if quality_stop_result.get("last_value") is None:
        quality_stop_result["last_value"] = _metric_from_metrics(
            metrics,
            str(quality_stop.get("metric") or "map50_95"),
        )
    quality_stop_result.setdefault("peak_epoch", quality_stop_state.get("peak_epoch"))
    quality_stop_result.setdefault("peak_value", quality_stop_state.get("peak_value"))
    quality_stop_result.setdefault(
        "recommended_max_epochs",
        (int(quality_stop_result["peak_epoch"]) + 1)
        if _as_int(quality_stop_result.get("peak_epoch")) is not None
        else None,
    )
    if bool(quality_stop_result.get("triggered")) and str(quality_stop_result.get("mode") or "") == "regression":
        quality_stop_result["promoted_checkpoint"] = str(best_pt if best_pt.exists() else source_weights)
        quality_stop_result["reverted_to_best"] = True

    # Feasibility verdict: did this run ever clear the configured threshold?
    # Computed unconditionally so downstream callers can surface it whenever
    # quality_stop is enabled; attempt mode is the primary consumer.
    _peak_for_verdict = _as_float(quality_stop_result.get("peak_value"))
    if _peak_for_verdict is None:
        _peak_for_verdict = _as_float(
            _metric_from_metrics(metrics, str(quality_stop.get("metric") or "map50_95"))
        )
    _threshold_for_verdict = _as_float(quality_stop.get("threshold"))
    if _threshold_for_verdict is not None and _peak_for_verdict is not None:
        quality_stop_result["verdict"] = (
            "viable" if _peak_for_verdict >= _threshold_for_verdict else "unreachable"
        )
        quality_stop_result["verdict_peak_value"] = float(_peak_for_verdict)
        quality_stop_result["verdict_threshold"] = float(_threshold_for_verdict)
        quality_stop_result["verdict_metric"] = str(quality_stop.get("metric") or "map50_95")

    # Build forecast from results.csv so this works whether or not the
    # progress_callback / monitor thread was active during the run.
    forecast_payload: dict[str, Any] = {}
    try:
        csv_rows = _extract_all_rows(run_dir / "results.csv")
        history_points: list[dict[str, Any]] = []
        for row in csv_rows:
            point = _extract_training_point(row, epochs=epochs, run_dir=run_dir)
            if point is not None:
                history_points.append(point)
        forecast_payload = forecast_run(history_points)
    except Exception:
        forecast_payload = {}
    forecast_text = render_forecast(forecast_payload) if forecast_payload else ""
    if forecast_text:
        print(forecast_text)
    training_finished_at = datetime.now(timezone.utc)
    training_duration_seconds = max(0.0, time.monotonic() - training_started_monotonic)

    if progress_callback is not None:
        try:
            progress_callback(
                {
                    "event": "completed",
                    "epoch": final_epoch,
                    "epochs": epochs,
                    "requested_epochs": epochs,
                    "progress": 100.0,
                    "map50": map50,
                    "map50_95": map50_95,
                    "quality_stop": quality_stop_result,
                    "run_dir": str(run_dir),
                    "asset_root": str(model_root),
                    "overflow_protocol": overflow_protocol,
                    "overflow_message": str(overflow_protocol.get("message") or ""),
                    "resume": bool(resume_checkpoint is not None),
                    "resumed_from": resumed_from,
                    "elapsed_seconds": training_duration_seconds,
                    "timestamp": time.time(),
                }
            )
            if forecast_payload:
                forecast_payload.setdefault("run_dir", str(run_dir))
                forecast_payload.setdefault("timestamp", time.time())
                progress_callback(forecast_payload)
                for line in forecast_text.splitlines():
                    progress_callback(
                        {
                            "event": "log",
                            "line": line,
                            "stream": "stdout",
                            "timestamp": time.time(),
                        }
                    )
        except Exception:
            pass

    metrics_payload: dict[str, Any] = {
        "scenario": cfg.name,
        "trainer": trainer,
        "base_model": base_model,
        "dataset": cfg.dataset,
        "trained_at": training_finished_at.isoformat(),
        "training_started_at": training_started_at.isoformat(),
        "training_finished_at": training_finished_at.isoformat(),
        "training_duration_seconds": training_duration_seconds,
        "hyperparams": train_kwargs,
        "training_guard": training_guard,
        "map50": map50,
        "map50_95": map50_95,
        "quality_stop": quality_stop_result,
        "weights": str(run_weights),
        "final_model_name": final_model_name,
        "final_model_file": final_model_file,
        "final_model_path": final_model_path,
        "source_weights": str(source_weights),
        "resumed_from": resumed_from,
        "save_period": checkpoint_period,
        "data_yaml": str(data_yaml),
        "dataset_snapshot_id": str(dataset_snapshot.get("snapshot_id") or ""),
        "dataset_snapshot_path": str(snapshot_path),
        "dataset_contract": dataset_snapshot.get("contract"),
        "dataset_quality": dataset_snapshot.get("quality"),
        "deterministic": bool(deterministic),
        "seed": seed,
        "metrics": metrics,
        "forecast": forecast_payload,
        "ci_cd": {"enabled": ci_cd_enabled, "policy": ci_cd_policy},
    }
    env_fingerprint = capture_environment_fingerprint(run_dir)
    metrics_payload["environment"] = env_fingerprint
    repro_manifest = create_repro_manifest(
        run_dir=run_dir,
        scenario=cfg.name,
        base_model=base_model,
        data_yaml=str(data_yaml),
        dataset_snapshot_id=str(dataset_snapshot.get("snapshot_id") or ""),
        hyperparams=train_kwargs,
        env=env_fingerprint,
    )
    metrics_payload["repro_manifest"] = str(run_dir / "repro_manifest.json")
    metrics_payload["replay_command"] = str(repro_manifest.get("replay_command") or "")
    metrics_payload["deterministic_state"] = deterministic_state
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=True, default=str),
        encoding="utf-8",
    )
    run_version = run_dir.name
    _ = register_model_version(
        scenario=cfg.name,
        run_version=run_version,
        artifacts={
            "run_dir": str(run_dir),
            "weights": str(run_weights),
            "final_model_name": final_model_name,
            "final_model_file": final_model_file,
            "final_model_path": final_model_path,
            "metrics_path": str(run_dir / "metrics.json"),
            "data_yaml": str(data_yaml),
        },
        lineage={
            "dataset_snapshot_id": str(dataset_snapshot.get("snapshot_id") or ""),
            "dataset_snapshot_path": str(snapshot_path),
            "base_model": base_model,
            "parent_version_id": parent_version_id or "",
            "source_weights": str(source_weights),
            "environment": env_fingerprint,
            "repro_manifest": str(run_dir / "repro_manifest.json"),
        },
        metrics={
            "map50": map50,
            "map50_95": map50_95,
            "quality_stop": quality_stop_result,
            "raw": metrics,
            "forecast": forecast_payload,
            "training_duration_seconds": training_duration_seconds,
        },
        ci_cd={
            "gate_status": "pending",
            "policy": ci_cd_policy,
        } if ci_cd_enabled else {},
        set_candidate=True,
    )
    return {
        "scenario": cfg.name,
        "run_version": run_version,
        "trainer": trainer,
        "config": str(cfg.config_path),
        "output": str(run_dir),
        "weights": str(run_weights),
        "final_model_name": final_model_name,
        "final_model_file": final_model_file,
        "final_model_path": final_model_path,
        "map50": "" if map50 is None else f"{map50:.4f}",
        "map50_95": "" if map50_95 is None else f"{map50_95:.4f}",
        "quality_stop": quality_stop_result,
        "data_yaml": str(data_yaml),
        "resumed_from": resumed_from,
        "save_period": str(checkpoint_period),
        "training_duration_seconds": f"{training_duration_seconds:.3f}",
        "seed": str(seed),
        "deterministic": str(bool(deterministic)).lower(),
        "dataset_snapshot_id": str(dataset_snapshot.get("snapshot_id") or ""),
        "replay_command": str(repro_manifest.get("replay_command") or ""),
        "training_guard": training_guard.get("summary", ""),
    }


def main() -> int:
    args = _parse_args()
    summary = run_training(
        args.scenario,
        trainer_override=args.trainer,
        base_model_override=args.base_model,
        epochs_override=args.epochs,
        imgsz_override=args.imgsz,
        checkpoint_period_override=args.save_period,
        seed_override=args.seed,
        deterministic=not bool(args.non_deterministic),
        resume=bool(args.resume),
        final_model_name=str(args.final_model_name or ""),
    )
    print(f"[train] scenario={summary['scenario']}")
    print(f"[train] trainer={summary['trainer']}")
    print(f"[train] config={summary['config']}")
    print(f"[train] output={summary['output']}")
    print(f"[train] resumed_from={summary['resumed_from'] or 'none'}")
    print(f"[train] checkpoint_save_period={summary['save_period']}")
    print(f"[train] seed={summary['seed']} deterministic={summary['deterministic']}")
    print(f"[train] dataset_snapshot_id={summary['dataset_snapshot_id']}")
    print(f"[train] replay={summary['replay_command']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
