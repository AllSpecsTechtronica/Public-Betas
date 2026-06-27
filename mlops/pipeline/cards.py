from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .governance import load_dataset_snapshot
from .registry import get_scenario_config, get_scenario_status, latest_run_dir


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def build_dataset_card_markdown(scenario: str) -> str:
    cfg = get_scenario_config(scenario)
    status = get_scenario_status(scenario)
    snap_id = ""
    run_dir = latest_run_dir(scenario)
    metrics: dict[str, Any] = {}
    if run_dir is not None:
        metrics = _read_json(run_dir / "metrics.json")
        snap_id = str(metrics.get("dataset_snapshot_id") or "")
    snap = load_dataset_snapshot(snap_id) if snap_id else None
    lines = [
        f"## Dataset card — `{cfg.name}`",
        "",
        f"- **Library dataset**: `{cfg.dataset}`",
        f"- **Labelled images (scenario view)**: {status.get('dataset_count', 0)}",
        f"- **Classes ({len(cfg.classes)})**: {', '.join(cfg.classes) if cfg.classes else '[none]'}",
        "",
    ]
    if isinstance(snap, dict):
        q = snap.get("quality") if isinstance(snap.get("quality"), dict) else {}
        ma = q.get("media_audit") if isinstance(q.get("media_audit"), dict) else {}
        lines += [
            "### Snapshot",
            f"- **snapshot_id**: `{snap.get('snapshot_id', '')}`",
            f"- **files hashed**: {snap.get('total_files', 0)}  |  **bytes**: {snap.get('total_bytes', 0)}",
            f"- **fingerprint**: `{snap.get('fingerprint_mode', 'legacy')}` via `content_sha256` per file row",
            "",
        ]
        lin = snap.get("lineage") if isinstance(snap.get("lineage"), dict) else {}
        edges = lin.get("edges") if isinstance(lin.get("edges"), list) else []
        if edges:
            lines.append("### Lineage (heuristic)")
            for e in edges:
                if isinstance(e, dict):
                    lines.append(f"- {e.get('from')} → {e.get('to')}: {e.get('evidence', '')}")
            lines.append("")
        if ma:
            lines += [
                "### Data validation",
                f"- **images scanned**: {ma.get('images_scanned', 0)}",
                f"- **PIL verify failures**: {ma.get('corrupt_count', 0)}",
                f"- **duplicate clusters (8×8 avg-hash)**: {ma.get('duplicate_cluster_count', 0)}",
                f"- **images with GPS EXIF**: {ma.get('gps_exif_image_count', 0)}",
                "",
            ]
            if ma.get("corrupt_count"):
                sample = ma.get("corrupt_images") if isinstance(ma.get("corrupt_images"), list) else []
                if sample:
                    lines.append("**Corrupt paths (sample)**:")
                    for rel in sample[:12]:
                        lines.append(f"- `{rel}`")
                    lines.append("")
    else:
        lines.append("_No dataset snapshot linked on the latest run._")
        lines.append("")
    lines.append("### Intended use")
    lines.append(f"- **Operator description**: {cfg.description or '[not set]'}")
    lines.append("")
    return "\n".join(lines)


def build_model_card_markdown(scenario: str) -> str:
    cfg = get_scenario_config(scenario)
    status = get_scenario_status(scenario)
    run_dir = latest_run_dir(scenario)
    metrics = _read_json(run_dir / "metrics.json") if run_dir else {}
    eval_report = _read_json(run_dir / "eval_report.json") if run_dir else {}
    guard = metrics.get("training_guard") if isinstance(metrics.get("training_guard"), dict) else {}
    lines = [
        f"## Model card — `{cfg.name}`",
        "",
        f"- **Status**: `{status.get('status', '')}`  |  **weights_ready**: {status.get('weights_ready')}",
        f"- **Base model**: `{cfg.base_model}`",
        f"- **Post-processing**: `{cfg.postproc}`",
        "",
        "### Training snapshot",
        f"- **Latest run dir**: `{run_dir or ''}`",
        f"- **map50 (metrics.json)**: `{metrics.get('map50', '')}`",
        f"- **trainer**: `{metrics.get('trainer', '')}`",
        "",
    ]
    if guard:
        lines += [
            "### System guard",
            f"- **profile summary**: {guard.get('summary', '')}",
            f"- **status**: `{guard.get('status', '')}`",
            "",
        ]
        adj = guard.get("adjustments") if isinstance(guard.get("adjustments"), list) else []
        if adj:
            lines.append("**Adjustments**:")
            for a in adj[:10]:
                lines.append(f"- {a}")
            lines.append("")
    hp = metrics.get("hyperparams") if isinstance(metrics.get("hyperparams"), dict) else {}
    if hp:
        lines.append("### Key hyperparameters")
        for key in ("epochs", "imgsz", "batch", "lr0", "patience", "save_period"):
            if key in hp:
                lines.append(f"- **{key}**: `{hp.get(key)}`")
        lines.append("")
    if eval_report:
        lines += [
            "### Eval / known failure signals",
            f"- **eval status**: `{eval_report.get('status', '')}`",
        ]
        tc = eval_report.get("threshold_checks") if isinstance(eval_report.get("threshold_checks"), dict) else {}
        for name, payload in tc.items():
            if isinstance(payload, dict):
                lines.append(f"- **{name}**: passed={payload.get('passed')} actual={payload.get('actual')}")
        lines.append("")
    else:
        lines.append("_No `eval_report.json` on latest run; run `mlops.pipeline.eval` with `--save` for slice checks._")
        lines.append("")
    lines.append("### Failure modes (operator)")
    lines.append("- Review **eval robustness** and **dataset drift** sections after `--save` eval runs.")
    lines.append("- Watch **media_audit** duplicate clusters before scaling ingest.")
    lines.append("")
    return "\n".join(lines)


def build_scenario_cards(scenario: str) -> dict[str, str]:
    scenario = str(scenario or "").strip()
    if not scenario:
        return {"dataset_card": "", "model_card": ""}
    return {
        "dataset_card": build_dataset_card_markdown(scenario),
        "model_card": build_model_card_markdown(scenario),
    }
