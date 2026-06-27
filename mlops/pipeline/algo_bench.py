from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .registry import MLOPS_ROOT, get_scenario_config, list_available_models
from .train import run_training
from .training_algos import supported_trainers, validate_trainer_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark trainer/backbone combinations for one scenario")
    parser.add_argument("--scenario", required=True)
    parser.add_argument(
        "--trainers",
        default="ultralytics_yolo",
        help="Comma-separated trainers (default: ultralytics_yolo)",
    )
    parser.add_argument(
        "--base-models",
        default="",
        help="Comma-separated base model refs/paths. Empty = scenario base model only",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Allow resume behavior for each trial")
    parser.add_argument(
        "--promote-candidate",
        action="store_true",
        help="Set candidate alias to the best-performing trial version",
    )
    return parser.parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_csv(value: str) -> list[str]:
    return [p.strip() for p in str(value or "").split(",") if p.strip()]


def _resolve_default_base_model(cfg: Any) -> list[str]:
    base = str(getattr(cfg, "base_model", "") or "").strip()
    return [base] if base else []


def _top_model_candidates(limit: int = 3) -> list[str]:
    out: list[str] = []
    for item in list_available_models():
        if not isinstance(item, dict):
            continue
        value = str(item.get("value") or item.get("path") or "").strip()
        if not value:
            continue
        out.append(value)
        if len(out) >= limit:
            break
    return out


def main() -> int:
    args = _parse_args()
    cfg = get_scenario_config(args.scenario)
    trainers = [validate_trainer_name(t) for t in _parse_csv(args.trainers)] or ["ultralytics_yolo"]
    models = _parse_csv(args.base_models)
    if not models:
        models = _resolve_default_base_model(cfg)
    if not models:
        models = _top_model_candidates(limit=1)
    if not models:
        raise SystemExit("No base model candidates available")

    trials: list[dict[str, Any]] = []
    best_idx = -1
    best_score = float("-inf")

    for trainer in trainers:
        if trainer not in supported_trainers():
            raise SystemExit(f"Unsupported trainer in bench: {trainer}")
        for base_model in models:
            trial_spec = {"trainer": trainer, "base_model": base_model}
            started = _now()
            try:
                summary = run_training(
                    cfg.name,
                    trainer_override=trainer,
                    base_model_override=base_model,
                    epochs_override=args.epochs,
                    imgsz_override=args.imgsz,
                    checkpoint_period_override=args.save_period,
                    seed_override=args.seed,
                    deterministic=True,
                    resume=bool(args.resume),
                )
                map50 = float(str(summary.get("map50") or "0") or 0.0)
                trial = {
                    **trial_spec,
                    "started_at": started,
                    "finished_at": _now(),
                    "status": "ok",
                    "map50": map50,
                    "summary": summary,
                }
                if map50 > best_score:
                    best_score = map50
                    best_idx = len(trials)
            except Exception as exc:
                trial = {
                    **trial_spec,
                    "started_at": started,
                    "finished_at": _now(),
                    "status": "error",
                    "error": str(exc),
                    "map50": None,
                }
            trials.append(trial)

    report = {
        "version": 1,
        "scenario": cfg.name,
        "created_at": _now(),
        "trials": trials,
        "best_index": best_idx,
        "best_trial": trials[best_idx] if 0 <= best_idx < len(trials) else None,
    }

    out_dir = MLOPS_ROOT / "models" / cfg.name / "algo_benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"bench_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True, default=str), encoding="utf-8")

    if args.promote_candidate and report.get("best_trial"):
        from .model_registry import set_alias, version_id_for_run

        best_summary = report["best_trial"].get("summary") if isinstance(report["best_trial"], dict) else {}
        run_dir = str((best_summary or {}).get("output") or "")
        run_version = Path(run_dir).name if run_dir else ""
        vid = version_id_for_run(cfg.name, run_version) if run_version else None
        if vid:
            set_alias(cfg.name, "candidate", vid)
            report["candidate_promoted_to"] = vid
            out_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=True, default=str),
                encoding="utf-8",
            )

    print(json.dumps({"report": str(out_path), **report}, ensure_ascii=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

