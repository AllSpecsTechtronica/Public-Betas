from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from . import model_registry
from . import registry as mlops_registry


DEFAULT_REQUIRED_ARTIFACTS = ["weights", "metrics.json", "dataset_snapshot", "repro_manifest"]
VALID_GATE_STATUSES = {"pending", "passed", "failed", "overridden"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return default


def normalize_policy(
    raw_policy: Mapping[str, Any] | None,
    *,
    hyperparams: Mapping[str, Any] | None = None,
    legacy_default_enabled: bool = False,
) -> dict[str, Any]:
    """Return a normalized per-scenario CI/CD policy.

    Existing scenarios that lack a ``ci_cd`` key remain legacy-compatible by
    passing ``legacy_default_enabled=False``. New scenario creation writes an
    explicit policy with ``enabled=True``.
    """
    raw = dict(raw_policy or {})
    hp = dict(hyperparams or {})
    has_policy = bool(raw_policy)
    metric = str(raw.get("metric") or hp.get("quality_stop_metric") or "map50_95").strip() or "map50_95"
    threshold = _safe_float(raw.get("threshold"))
    if threshold is None:
        threshold = _safe_float(hp.get("quality_stop_threshold"))
    if threshold is None:
        threshold = 0.90
    regression_tolerance = _safe_float(raw.get("regression_tolerance"))
    if regression_tolerance is None:
        regression_tolerance = 0.02
    promotion = str(raw.get("promotion") or "manual").strip().lower()
    if promotion not in {"manual", "auto"}:
        promotion = "manual"
    required = raw.get("required_artifacts")
    if not isinstance(required, list):
        required = list(DEFAULT_REQUIRED_ARTIFACTS)
    required_artifacts = []
    seen: set[str] = set()
    for item in required:
        name = str(item or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        required_artifacts.append(name)
    if not required_artifacts:
        required_artifacts = list(DEFAULT_REQUIRED_ARTIFACTS)
    return {
        "enabled": _safe_bool(raw.get("enabled"), legacy_default_enabled if not has_policy else False),
        "metric": metric,
        "threshold": float(threshold),
        "regression_tolerance": float(regression_tolerance),
        "promotion": promotion,
        "required_artifacts": required_artifacts,
    }


def scenario_policy(scenario: str) -> dict[str, Any]:
    cfg = mlops_registry.get_scenario_config(scenario)
    raw = getattr(cfg, "raw", {}) if cfg is not None else {}
    if not isinstance(raw, dict):
        raw = {}
    policy_raw = raw.get("ci_cd") if isinstance(raw.get("ci_cd"), dict) else None
    return normalize_policy(
        policy_raw,
        hyperparams=getattr(cfg, "hyperparams", {}) if cfg is not None else {},
        legacy_default_enabled=False,
    )


def _metric_value(metrics: Mapping[str, Any], metric: str) -> float | None:
    keys = [
        metric,
        metric.replace("map50_95", "map50-95"),
        f"metrics/{metric}",
        f"metrics/{metric}(B)",
    ]
    nested = metrics.get("metrics") if isinstance(metrics.get("metrics"), dict) else {}
    for source in (metrics, nested):
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = _safe_float(source.get(key))
            if value is not None:
                return value
    if metric == "map50_95":
        for key in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map50_95"):
            value = _safe_float(metrics.get(key))
            if value is not None:
                return value
    if metric == "map50":
        for key in ("metrics/mAP50(B)", "metrics/mAP50", "map50"):
            value = _safe_float(metrics.get(key))
            if value is not None:
                return value
    return None


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _weights_path(run_dir: Path, metrics: Mapping[str, Any], version: dict[str, Any] | None) -> Path | None:
    candidates: list[Path] = []
    for raw in (
        metrics.get("weights"),
        metrics.get("final_model_path"),
        ((version or {}).get("artifacts") or {}).get("weights") if isinstance((version or {}).get("artifacts"), dict) else "",
    ):
        text = str(raw or "").strip()
        if not text:
            continue
        path = Path(text)
        candidates.append(path if path.is_absolute() else (run_dir / path).resolve())
    candidates.extend(
        [
            run_dir / "weights.pt",
            run_dir / "weights.pth",
            run_dir / "model.pkl",
            run_dir / "gallery.db",
            run_dir / "adapter" / "adapter_model.safetensors",
            run_dir / "adapter_model.safetensors",
            run_dir / "weights" / "best.pt",
            run_dir / "weights" / "last.pt",
        ]
    )
    for path in candidates:
        try:
            if path.exists() and path.is_file() and path.stat().st_size > 0:
                return path
        except Exception:
            continue
    return None


def _baseline_metric(scenario: str, metric: str) -> tuple[str, float | None]:
    prod = model_registry.resolve_alias(scenario, "prod")
    if not isinstance(prod, dict):
        return "", None
    baseline_metrics = prod.get("metrics") if isinstance(prod.get("metrics"), dict) else {}
    return str(prod.get("version_id") or ""), _metric_value(baseline_metrics, metric)


def _version_for_run(scenario: str, run_version: str) -> dict[str, Any] | None:
    version_id = model_registry.version_id_for_run(scenario, run_version)
    if not version_id:
        return None
    return model_registry.get_model_version(scenario, version_id)


def evaluate_run_gate(
    scenario: str,
    run_version: str,
    *,
    policy: Mapping[str, Any] | None = None,
    update_registry: bool = True,
) -> dict[str, Any]:
    cfg = mlops_registry.get_scenario_config(scenario)
    normalized = normalize_policy(
        policy if policy is not None else (getattr(cfg, "raw", {}) or {}).get("ci_cd"),
        hyperparams=getattr(cfg, "hyperparams", {}),
        legacy_default_enabled=False,
    )
    run_dir = mlops_registry.resolve_scenario_run_dir(scenario, run_version)
    if run_dir is None:
        raise FileNotFoundError(f"run not found for {scenario}:{run_version}")
    run_version = run_dir.name
    metrics_path = run_dir / "metrics.json"
    metrics = _read_json(metrics_path)
    version = _version_for_run(scenario, run_version)
    version_id = str((version or {}).get("version_id") or f"{scenario}:{run_version}")
    weights = _weights_path(run_dir, metrics, version)

    artifacts: dict[str, Any] = {
        "run_dir": str(run_dir),
        "weights": str(weights or ""),
        "metrics_json": str(metrics_path) if metrics_path.is_file() else "",
        "dataset_snapshot_id": str(metrics.get("dataset_snapshot_id") or ""),
        "dataset_snapshot_path": str(metrics.get("dataset_snapshot_path") or ""),
        "repro_manifest": str(metrics.get("repro_manifest") or ""),
    }
    failures: list[str] = []
    warnings: list[str] = []

    required = set(str(item) for item in normalized.get("required_artifacts") or DEFAULT_REQUIRED_ARTIFACTS)
    if "weights" in required and weights is None:
        failures.append("missing weights artifact")
    if "metrics.json" in required and not metrics_path.is_file():
        failures.append("missing metrics.json")
    if "dataset_snapshot" in required:
        if not artifacts["dataset_snapshot_id"]:
            failures.append("missing dataset snapshot id")
        snap_path = Path(artifacts["dataset_snapshot_path"]) if artifacts["dataset_snapshot_path"] else None
        if snap_path is not None and not snap_path.is_file():
            warnings.append("dataset snapshot path is not readable")
    if "repro_manifest" in required:
        repro = Path(artifacts["repro_manifest"]) if artifacts["repro_manifest"] else run_dir / "repro_manifest.json"
        if not repro.is_file():
            failures.append("missing replay manifest")
        else:
            artifacts["repro_manifest"] = str(repro)

    contract = metrics.get("dataset_contract") if isinstance(metrics.get("dataset_contract"), dict) else {}
    if str(contract.get("status") or "").lower() == "failed":
        issues = contract.get("issues") if isinstance(contract.get("issues"), list) else []
        detail = "; ".join(str(item) for item in issues) if issues else "dataset contract failed"
        failures.append(detail)

    metric_name = str(normalized.get("metric") or "map50_95")
    metric_value = _metric_value(metrics, metric_name)
    threshold = _safe_float(normalized.get("threshold"))
    if threshold is None:
        threshold = 0.90
    if metric_value is None:
        failures.append(f"metric {metric_name!r} is missing")
    elif metric_value < threshold:
        failures.append(f"{metric_name} {metric_value:.4f} is below threshold {threshold:.4f}")

    quality_stop = metrics.get("quality_stop") if isinstance(metrics.get("quality_stop"), dict) else {}
    if str(quality_stop.get("verdict") or "").lower() == "unreachable":
        failures.append("quality-stop verdict is unreachable")

    baseline_version_id, baseline_value = _baseline_metric(scenario, metric_name)
    regression_tolerance = _safe_float(normalized.get("regression_tolerance"))
    if regression_tolerance is None:
        regression_tolerance = 0.02
    if baseline_version_id and baseline_value is None:
        warnings.append(f"baseline {baseline_version_id} has no {metric_name} metric")
    if baseline_value is not None and metric_value is not None:
        min_allowed = baseline_value - regression_tolerance
        if metric_value < min_allowed:
            failures.append(
                f"{metric_name} regressed from baseline {baseline_value:.4f} "
                f"to {metric_value:.4f} beyond tolerance {regression_tolerance:.4f}"
            )

    gate_status = "passed" if not failures else "failed"
    report = {
        "scenario": scenario,
        "run_version": run_version,
        "version_id": version_id,
        "generated_at": _utc_now(),
        "policy": normalized,
        "gate_status": gate_status,
        "passed": gate_status == "passed",
        "failures": failures,
        "warnings": warnings,
        "artifacts": artifacts,
        "metrics": {
            "metric": metric_name,
            "value": metric_value,
            "threshold": threshold,
            "baseline_version_id": baseline_version_id,
            "baseline_value": baseline_value,
            "regression_tolerance": regression_tolerance,
        },
        "dataset_contract": contract,
        "dataset_quality": metrics.get("dataset_quality") if isinstance(metrics.get("dataset_quality"), dict) else {},
        "quality_stop": quality_stop,
    }
    report_path = run_dir / "ci_cd_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    report["report_path"] = str(report_path)

    if update_registry and version is not None:
        model_registry.update_version_ci_cd(
            scenario,
            version_id,
            {
                "gate_status": gate_status,
                "report_path": str(report_path),
                "baseline_version_id": baseline_version_id,
                "metric": metric_name,
                "metric_value": metric_value,
                "threshold": threshold,
                "evaluated_at": report["generated_at"],
                "failures": failures,
                "warnings": warnings,
            },
            lifecycle_event={
                "at": report["generated_at"],
                "status": f"gate_{gate_status}",
                "reason": "; ".join(failures) if failures else "CI/CD gate passed",
            },
        )
    return report


def load_gate_report(scenario: str, run_version: str) -> dict[str, Any]:
    run_dir = mlops_registry.resolve_scenario_run_dir(scenario, run_version)
    if run_dir is None:
        raise FileNotFoundError(f"run not found for {scenario}:{run_version}")
    report_path = run_dir / "ci_cd_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"CI/CD report not found: {report_path}")
    report = _read_json(report_path)
    if not report:
        raise ValueError(f"invalid CI/CD report: {report_path}")
    report.setdefault("report_path", str(report_path))
    return report


def promote_run(
    scenario: str,
    run_version: str,
    *,
    target_alias: str = "prod",
    actor: str = "cvops",
    reason: str = "",
    override: bool = False,
) -> dict[str, Any]:
    target_alias = str(target_alias or "prod").strip().lower()
    cfg = mlops_registry.get_scenario_config(scenario)
    report: dict[str, Any]
    try:
        report = load_gate_report(scenario, run_version)
    except FileNotFoundError:
        report = evaluate_run_gate(scenario, run_version, update_registry=True)
    passed = bool(report.get("passed")) and str(report.get("gate_status") or "") == "passed"
    if not passed and not override:
        failures = report.get("failures") if isinstance(report.get("failures"), list) else []
        raise ValueError("candidate did not pass CI/CD gate: " + ("; ".join(str(f) for f in failures) or "unknown failure"))

    run_version = str(report.get("run_version") or run_version)
    version_id = str(report.get("version_id") or model_registry.version_id_for_run(scenario, run_version) or "")
    if not version_id:
        raise ValueError(f"model registry version not found for {scenario}:{run_version}")
    version = model_registry.get_model_version(scenario, version_id)
    if version is None:
        raise ValueError(f"model registry version not found: {version_id}")
    artifacts = version.get("artifacts") if isinstance(version.get("artifacts"), dict) else {}
    weights = Path(str((report.get("artifacts") or {}).get("weights") or artifacts.get("weights") or ""))
    if not weights.is_absolute():
        run_dir = mlops_registry.resolve_scenario_run_dir(scenario, run_version)
        if run_dir is not None:
            weights = (run_dir / weights).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"weights artifact not found: {weights}")
    # Only a prod promotion overwrites the live serving weights. Staging is a
    # challenger tier: the weights stay in the run dir until a human promotes
    # the challenger to prod.
    target = Path(getattr(cfg, "weights_path", Path("")))
    copy_to_live = target_alias == "prod"
    if copy_to_live and str(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            same = weights.resolve() == target.resolve()
        except Exception:
            same = False
        if not same:
            shutil.copy2(weights, target)

    now = _utc_now()
    ci_cd_update = {
        "gate_status": "overridden" if override and not passed else "passed",
        "report_path": str(report.get("report_path") or ""),
        "baseline_version_id": str((report.get("metrics") or {}).get("baseline_version_id") or ""),
        "promoted_at": now,
        "promoted_by": str(actor or "cvops"),
        "promotion_alias": target_alias,
        "promotion_reason": str(reason or ("override" if override and not passed else "gate passed")),
    }
    entry = model_registry.promote_version(
        scenario,
        version_id,
        alias=target_alias,
        actor=str(actor or "cvops"),
        reason=str(reason or ""),
        ci_cd=ci_cd_update,
    )
    result = {
        "ok": True,
        "scenario": scenario,
        "run_version": run_version,
        "version_id": version_id,
        "alias": target_alias,
        target_alias: entry,
        "weights_path": str(target if (copy_to_live and str(target)) else weights),
        "report": report,
        "override": bool(override),
    }
    return result

