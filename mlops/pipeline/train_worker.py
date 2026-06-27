"""Subprocess entry point for YOLO training runs.

Spawned by the cvops service so a Stop click can SIGKILL the entire process
group instead of waiting for a Python callback to fire. The worker reads its
inputs from a JSON file, invokes `run_training`, tunnels progress events back
to the parent as `__CVOPS_EVT__{json}` lines on stdout, and writes the final
summary dict to an output JSON file the parent reads after the process exits.

Plain stdout/stderr lines that are not event lines (e.g. ultralytics logs)
flow through unchanged so the parent can route them as log events.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

EVENT_MARKER = "__CVOPS_EVT__"

# Snapshot the real stdout at module load so events bypass any later redirect
# (run_training installs a stdout tee internally; without this, every event
# we emit would feed back through the tee and trigger infinite recursion).
_REAL_STDOUT = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout


def _emit_event(payload: dict[str, Any]) -> None:
    try:
        line = EVENT_MARKER + json.dumps(payload, default=str)
    except Exception:
        line = EVENT_MARKER + json.dumps(
            {"event": "log", "line": "[worker] event serialization failed", "stream": "stderr"}
        )
    try:
        _REAL_STDOUT.write(line + "\n")
        _REAL_STDOUT.flush()
    except Exception:
        pass


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cvops YOLO training subprocess worker")
    parser.add_argument("--input-json", required=True, help="Path to a JSON file with run_training kwargs")
    parser.add_argument("--output-json", required=True, help="Path the worker writes the final summary to")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)

    try:
        with input_path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
    except Exception as exc:
        _emit_event({"event": "log", "line": f"[worker] failed to read input json: {exc}", "stream": "stderr"})
        output_path.write_text(json.dumps({"error": f"worker input read failed: {exc}"}), encoding="utf-8")
        return 1

    scenario = str(spec.get("scenario") or "").strip()
    if not scenario:
        _emit_event({"event": "log", "line": "[worker] missing scenario in input", "stream": "stderr"})
        output_path.write_text(json.dumps({"error": "missing scenario"}), encoding="utf-8")
        return 1

    # Import inside main() so import cost is part of the spawned-process startup
    # (and tunneled through our event protocol, not the parent's import).
    try:
        from mlops.pipeline.train import run_training
    except Exception as exc:
        _emit_event({"event": "log", "line": f"[worker] import failed: {exc}", "stream": "stderr"})
        output_path.write_text(json.dumps({"error": f"worker import failed: {exc}"}), encoding="utf-8")
        return 1

    def _progress(payload: dict[str, Any]) -> None:
        _emit_event(payload)

    kwargs: dict[str, Any] = {
        "scenario": scenario,
        "progress_callback": _progress,
        # Cancel is enforced by the parent via SIGKILL on the process group,
        # so the worker passes no cancel_check — Ultralytics callbacks won't
        # try to raise from inside, which keeps the in-process state simple.
        "cancel_check": None,
        # In subprocess mode the worker's stdout is already the parent's pipe.
        # Skip run_training's internal stdout tee so the structured events we
        # write to _REAL_STDOUT don't get captured + re-emitted recursively.
        "_capture_streams": False,
    }
    for key in (
        "trainer_override",
        "base_model_override",
        "epochs_override",
        "imgsz_override",
        "checkpoint_period_override",
        "seed_override",
        "deterministic",
        "resume",
        "auto_fresh_on_completed_resume",
        "final_model_name",
        "hyperparams_overrides",
    ):
        if key in spec and spec[key] is not None:
            kwargs[key] = spec[key]

    try:
        summary = run_training(**kwargs)
    except BaseException as exc:
        tb = traceback.format_exc()
        _emit_event(
            {
                "event": "log",
                "line": f"[worker] training raised: {exc}",
                "stream": "stderr",
            }
        )
        output_path.write_text(
            json.dumps({"error": str(exc) or exc.__class__.__name__, "traceback": tb}),
            encoding="utf-8",
        )
        return 1

    try:
        output_path.write_text(json.dumps(summary, default=str), encoding="utf-8")
    except Exception as exc:
        _emit_event(
            {
                "event": "log",
                "line": f"[worker] failed to write summary json: {exc}",
                "stream": "stderr",
            }
        )
        output_path.write_text(json.dumps({"error": f"summary write failed: {exc}"}), encoding="utf-8")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
