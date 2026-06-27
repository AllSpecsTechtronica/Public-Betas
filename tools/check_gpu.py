"""Standalone system guard probe — vendored from mlops/pipeline/system_guard.py.

Copy this file to any machine and run (Python 3.10+ with torch recommended):

  python system_guard_isolated.py
  python system_guard_isolated.py --json
  MLOPS_MODELS_ROOT=/path/to/models python system_guard_isolated.py

Optional: pip install psutil (finer CPU/RAM; guard works without it but RAM
may read as 0 and trigger false "blocked" for minimum RAM). Set
SYSTEM_GUARD_MIN_RAM_GB=0 to relax the hard RAM floor for smoke tests.

Keep this file aligned with mlops/pipeline/system_guard.py when guard logic changes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

try:
    import psutil
except Exception:  # pragma: no cover - fallback only used when psutil is absent
    psutil = None


_GIB = 1024 ** 3


def _min_safe_ram_bytes() -> int:
    env = str(os.environ.get("SYSTEM_GUARD_MIN_RAM_GB", "") or "").strip()
    if env:
        try:
            return max(0, int(float(env) * _GIB))
        except Exception:
            pass
    return 4 * _GIB


def _min_safe_disk_bytes() -> int:
    env = str(os.environ.get("SYSTEM_GUARD_MIN_FREE_DISK_GB", "") or "").strip()
    if env:
        try:
            return max(0, int(float(env) * _GIB))
        except Exception:
            pass
    return 1 * _GIB


_GUARD_PROFILES = ("balanced", "stable", "fast")


@dataclass(frozen=True)
class SystemSpecs:
    system: str
    machine: str
    cpu_logical_cores: int
    cpu_physical_cores: int
    cpu_brand: str
    total_memory_bytes: int
    accelerator: str
    gpu_name: str
    gpu_memory_bytes: int | None
    gpus: tuple[dict[str, Any], ...] = field(default_factory=tuple)

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory_bytes / float(_GIB) if self.total_memory_bytes > 0 else 0.0

    @property
    def gpu_memory_gb(self) -> float | None:
        if self.gpu_memory_bytes is None or self.gpu_memory_bytes <= 0:
            return None
        return self.gpu_memory_bytes / float(_GIB)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)


def _detect_cpu_brand(system: str) -> str:
    try:
        if system == "Darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=1.0,
            )
            brand = (out.stdout or "").strip()
            if brand:
                return brand
        elif system == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as fh:
                    for line in fh:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except Exception:
                pass
        elif system == "Windows":
            brand = platform.processor()
            if brand:
                return brand
    except Exception:
        pass
    return platform.processor() or "Unknown CPU"


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except Exception:
        return None


@lru_cache(maxsize=1)
def detect_system_specs() -> SystemSpecs:
    system = platform.system()
    machine = platform.machine()
    cpu_logical = os.cpu_count() or 1
    cpu_physical = cpu_logical
    total_memory = 0
    if psutil is not None:
        try:
            cpu_physical = psutil.cpu_count(logical=False) or cpu_logical
        except Exception:
            cpu_physical = cpu_logical
        try:
            total_memory = int(psutil.virtual_memory().total)
        except Exception:
            total_memory = 0

    accelerator = "cpu"
    gpu_name = "CPU"
    gpu_memory_bytes: int | None = None
    gpus: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            accelerator = "cuda"
            count = int(torch.cuda.device_count() or 0)
            for idx in range(count):
                try:
                    name = str(torch.cuda.get_device_name(idx) or f"CUDA GPU {idx}")
                except Exception:
                    name = f"CUDA GPU {idx}"
                try:
                    mem_bytes = int(torch.cuda.get_device_properties(idx).total_memory)
                except Exception:
                    mem_bytes = None
                gpus.append({
                    "index": idx,
                    "name": name,
                    "memory_bytes": mem_bytes,
                    "memory_gb": None if not mem_bytes else round(mem_bytes / float(_GIB), 2),
                    "backend": "cuda",
                })
            if gpus:
                gpu_name = str(gpus[0]["name"])
                gpu_memory_bytes = gpus[0].get("memory_bytes")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            accelerator = "mps"
            if system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
                gpu_name = "Apple Silicon GPU"
            else:
                gpu_name = "Metal GPU"
            gpus.append({
                "index": 0,
                "name": gpu_name,
                "memory_bytes": None,
                "memory_gb": None,
                "backend": "mps",
            })
    except Exception:
        pass
    if gpu_name == "CPU" and system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
        gpu_name = "Apple Silicon GPU"
    if not gpus and accelerator == "cpu":
        gpus.append({
            "index": 0,
            "name": gpu_name,
            "memory_bytes": None,
            "memory_gb": None,
            "backend": "cpu",
        })

    return SystemSpecs(
        system=system,
        machine=machine,
        cpu_logical_cores=cpu_logical,
        cpu_physical_cores=cpu_physical,
        cpu_brand=_detect_cpu_brand(system),
        total_memory_bytes=total_memory,
        accelerator=accelerator,
        gpu_name=gpu_name,
        gpu_memory_bytes=gpu_memory_bytes,
        gpus=tuple(gpus),
    )


def _infer_model_scale(model_ref: str) -> str:
    name = Path(str(model_ref or "")).name.lower()
    match = re.search(r"yolo[a-z0-9]*?([nslmx])(?=[._-]|$)", name)
    if match:
        return match.group(1)
    return "n"


def _round_down_multiple(value: int, step: int) -> int:
    if step <= 1:
        return max(1, value)
    return max(step, int(math.floor(value / float(step))) * step)


def _base_limits(specs: SystemSpecs) -> tuple[int, int, int, str]:
    """Return (max_imgsz, max_batch, max_workers, rung_label).

    rung_label is a short human description of which branch fired — surfaced
    in the derivation trail so the UI can explain *why* these limits were
    chosen.
    """
    mem_gb = specs.total_memory_gb
    if specs.accelerator == "mps":
        if mem_gb <= 8.5:
            return 512, 2, 0, f"mps, unified RAM {mem_gb:.1f} <= 8.5 GB"
        if mem_gb <= 16.5:
            return 640, 4, 1, f"mps, unified RAM {mem_gb:.1f} <= 16.5 GB"
        return 768, 8, 2, f"mps, unified RAM {mem_gb:.1f} > 16.5 GB"
    if specs.accelerator == "cuda":
        gpu_gb = specs.gpu_memory_gb or 0.0
        if gpu_gb and gpu_gb <= 6.5:
            return 640, 4, 2, f"cuda, VRAM {gpu_gb:.1f} <= 6.5 GB"
        if gpu_gb and gpu_gb <= 8.5:
            return 640, 8, 4, f"cuda, VRAM {gpu_gb:.1f} <= 8.5 GB"
        if gpu_gb and gpu_gb <= 12.5:
            return 768, 12, 6, f"cuda, VRAM {gpu_gb:.1f} <= 12.5 GB"
        return 1024, 16, 8, f"cuda, VRAM {gpu_gb:.1f} > 12.5 GB"
    if mem_gb <= 8.5:
        return 512, 2, 0, f"cpu, RAM {mem_gb:.1f} <= 8.5 GB"
    if mem_gb <= 16.5:
        return 640, 4, 1, f"cpu, RAM {mem_gb:.1f} <= 16.5 GB"
    return 640, 8, 2, f"cpu, RAM {mem_gb:.1f} > 16.5 GB"


def _guard_profile(name: Any) -> str:
    value = str(name or "").strip().lower()
    return value if value in _GUARD_PROFILES else "balanced"


def _apply_profile(max_imgsz: int, max_batch: int, max_workers: int, profile: str) -> tuple[int, int, int, str]:
    """Tune base limits by a user-selected guard profile.

    Returns (imgsz, batch, workers, profile_note). The note is a short phrase
    describing the multiplier applied, for the derivation trail.

    - stable: bias toward fewer workers/smaller batch to reduce OOM + UI thrash.
    - balanced: current auto heuristic.
    - fast: allow slightly higher batch/workers when the system can handle it.
    """
    p = _guard_profile(profile)
    if p == "stable":
        return (
            max(320, _round_down_multiple(int(max_imgsz * 0.90), 32)),
            max(1, int(round(max_batch * 0.75))),
            max(0, int(round(max_workers * 0.70))),
            "stable: imgsz x0.90, batch x0.75, workers x0.70",
        )
    if p == "fast":
        return (
            min(1280, _round_down_multiple(int(max_imgsz * 1.10), 32)),
            min(64, max_batch + 2),
            min(16, max_workers + 2),
            "fast: imgsz x1.10, batch +2, workers +2",
        )
    return max_imgsz, max_batch, max_workers, "balanced: no delta"


def _free_disk_bytes(target: Path | None = None) -> int:
    """Return free disk bytes. Uses MLOPS_MODELS_ROOT if set, else cwd."""
    probe: Path
    if target is not None:
        probe = target
    else:
        root = str(os.environ.get("MLOPS_MODELS_ROOT", "") or "").strip()
        if root:
            probe = Path(root)
        else:
            probe = Path.cwd()
    try:
        while not probe.exists():
            if probe.parent == probe:
                break
            probe = probe.parent
        return int(shutil.disk_usage(probe).free)
    except Exception:
        return 0


def _apply_model_scale(max_imgsz: int, max_batch: int, model_scale: str) -> tuple[int, int, str]:
    """Clamp limits further based on the detected YOLO scale.

    Returns (imgsz, batch, scale_note). The note spells out the rule applied
    so the UI can show why the effective limits tightened.
    """
    if model_scale == "s":
        return max_imgsz, max(1, max_batch - 1), "s: batch -1"
    if model_scale == "m":
        return min(max_imgsz, 640), max(1, max_batch // 2), "m: clamp imgsz<=640, batch//=2"
    if model_scale == "l":
        return min(max_imgsz, 512), max(1, max_batch // 3), "l: clamp imgsz<=512, batch//=3"
    if model_scale == "x":
        return min(max_imgsz, 512), 1, "x: clamp imgsz<=512, batch=1"
    return max_imgsz, max_batch, "n: no clamp"


# Rough YOLOv8/v11 scale footprint. Parameter counts in millions taken from
# the Ultralytics model cards; activation coefficients are empirical GB per
# (batch=1, imgsz=640) observed during training with AMP on. Used only for a
# user-facing projection — training monitors the real peak separately.
_YOLO_SCALE_FOOTPRINT: dict[str, tuple[float, float]] = {
    "n": (3.2, 0.25),
    "s": (11.2, 0.50),
    "m": (25.9, 0.90),
    "l": (43.7, 1.30),
    "x": (68.2, 1.80),
}


def _project_training_vram(
    model_scale: str,
    imgsz: int,
    batch: int,
    specs: SystemSpecs,
) -> dict[str, Any]:
    """Heuristic projection of peak training VRAM/RAM for a YOLO run.

    The model copy during training carries 4x its weight footprint on top
    of activations (weights + grads + Adam first/second moments). Activations
    scale with batch x (imgsz/640)^2. This is a rule-of-thumb — the training
    process still enforces the real peak at runtime.
    """
    params_m, act_coef = _YOLO_SCALE_FOOTPRINT.get(model_scale, _YOLO_SCALE_FOOTPRINT["n"])
    # fp32 weight bytes: params (millions) * 4 bytes / 1024^3 GB
    weights_gb = (params_m * 1e6 * 4.0) / float(_GIB)
    # Adam state (m/v) + grads + weights = 4x
    optim_gb = weights_gb * 4.0
    size_scale = (float(imgsz) / 640.0) ** 2
    activations_gb = float(batch) * size_scale * act_coef
    # Workspace (cuDNN scratch, kernel autotuner) — small but non-zero.
    workspace_gb = 0.5
    peak_gb = optim_gb + activations_gb + workspace_gb

    if specs.accelerator == "cuda" and specs.gpu_memory_gb:
        budget_gb = float(specs.gpu_memory_gb)
        target = "VRAM"
    elif specs.accelerator == "mps":
        budget_gb = max(0.0, specs.total_memory_gb - 2.0)
        target = "unified RAM (macOS)"
    else:
        budget_gb = max(0.0, specs.total_memory_gb - 2.0)
        target = "system RAM"
    headroom_pct = (peak_gb / budget_gb * 100.0) if budget_gb > 0 else None
    risk = "ok"
    if headroom_pct is None:
        risk = "unknown"
    elif headroom_pct >= 95.0:
        risk = "over"
    elif headroom_pct >= 80.0:
        risk = "tight"

    return {
        "params_m": round(params_m, 2),
        "weights_gb": round(weights_gb, 2),
        "optim_state_gb": round(optim_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "workspace_gb": round(workspace_gb, 2),
        "peak_gb": round(peak_gb, 2),
        "budget_gb": round(budget_gb, 2),
        "headroom_pct": None if headroom_pct is None else round(headroom_pct, 1),
        "target": target,
        "risk": risk,
        "formula": (
            f"peak = 4 x weights({weights_gb:.2f} GB) + batch({batch}) x "
            f"(imgsz/640)^2 x act_coef({act_coef:.2f} GB) + workspace(0.5 GB)"
        ),
    }


def build_training_guard(base_model: str, hyperparams: Mapping[str, Any]) -> dict[str, Any]:
    specs = detect_system_specs()
    model_scale = _infer_model_scale(base_model)
    profile = _guard_profile(hyperparams.get("guard_profile") if isinstance(hyperparams, Mapping) else "")

    derivation: list[dict[str, Any]] = []
    base_imgsz, base_batch, base_workers, base_note = _base_limits(specs)
    derivation.append({
        "step": "base",
        "note": base_note,
        "imgsz": base_imgsz,
        "batch": base_batch,
        "workers": base_workers,
    })

    prof_imgsz, prof_batch, prof_workers, prof_note = _apply_profile(
        base_imgsz, base_batch, base_workers, profile
    )
    derivation.append({
        "step": f"profile={profile}",
        "note": prof_note,
        "imgsz": prof_imgsz,
        "batch": prof_batch,
        "workers": prof_workers,
    })

    scale_imgsz, scale_batch, scale_note = _apply_model_scale(prof_imgsz, prof_batch, model_scale)
    derivation.append({
        "step": f"model_scale={model_scale}",
        "note": scale_note,
        "imgsz": scale_imgsz,
        "batch": scale_batch,
        "workers": prof_workers,
    })

    max_imgsz, max_batch, max_workers = scale_imgsz, scale_batch, prof_workers

    requested_epochs = max(1, _coerce_int(hyperparams.get("epochs")) or 20)
    requested_imgsz = max(32, _coerce_int(hyperparams.get("imgsz")) or 640)
    requested_batch = _coerce_int(hyperparams.get("batch"))
    requested_workers = _coerce_int(hyperparams.get("workers"))

    effective_imgsz = min(requested_imgsz, max_imgsz)
    effective_imgsz = _round_down_multiple(effective_imgsz, 32)
    effective_batch = max(1, min(requested_batch if requested_batch is not None else max_batch, max_batch))
    effective_workers = max(0, min(requested_workers if requested_workers is not None else max_workers, max_workers))

    clamp_bits: list[str] = []
    if requested_imgsz != effective_imgsz:
        clamp_bits.append(f"imgsz {requested_imgsz}->{effective_imgsz}")
    if requested_batch is not None and requested_batch != effective_batch:
        clamp_bits.append(f"batch {requested_batch}->{effective_batch}")
    if requested_workers is not None and requested_workers != effective_workers:
        clamp_bits.append(f"workers {requested_workers}->{effective_workers}")
    derivation.append({
        "step": "clamp",
        "note": (", ".join(clamp_bits) if clamp_bits else "request fit within limits"),
        "imgsz": effective_imgsz,
        "batch": effective_batch,
        "workers": effective_workers,
    })

    projection = _project_training_vram(model_scale, effective_imgsz, effective_batch, specs)

    adjustments: list[str] = []
    if requested_imgsz > effective_imgsz:
        adjustments.append(f"imgsz reduced from {requested_imgsz} to {effective_imgsz}")
    if requested_batch is None:
        adjustments.append(f"batch set to {effective_batch} instead of Ultralytics auto/default")
    elif requested_batch > effective_batch:
        adjustments.append(f"batch reduced from {requested_batch} to {effective_batch}")
    if requested_workers is None:
        adjustments.append(f"workers set to {effective_workers} for local stability")
    elif requested_workers > effective_workers:
        adjustments.append(f"workers reduced from {requested_workers} to {effective_workers}")

    host_bits = [
        f"{specs.system} {specs.machine}",
        f"{specs.total_memory_gb:.1f} GB RAM",
        specs.gpu_name,
        f"runtime={specs.accelerator.upper()}",
    ]
    host_label = " | ".join(host_bits)

    effective = {
        "epochs": requested_epochs,
        "imgsz": effective_imgsz,
        "batch": effective_batch,
        "workers": effective_workers,
    }
    requested = {
        "epochs": requested_epochs,
        "imgsz": requested_imgsz,
        "batch": requested_batch,
        "workers": requested_workers,
    }

    free_disk = _free_disk_bytes()
    blocking_reasons: list[str] = []
    min_ram = _min_safe_ram_bytes()
    if min_ram and specs.total_memory_bytes and specs.total_memory_bytes < min_ram:
        blocking_reasons.append(
            f"only {specs.total_memory_gb:.1f} GB RAM detected; training requires at least "
            f"{min_ram / _GIB:.0f} GB"
        )
    min_disk = _min_safe_disk_bytes()
    if min_disk and free_disk and free_disk < min_disk:
        blocking_reasons.append(
            f"only {free_disk / _GIB:.1f} GB free disk on models volume; training requires at least "
            f"{min_disk / _GIB:.0f} GB"
        )

    if blocking_reasons:
        for reason in blocking_reasons:
            adjustments.insert(0, f"[BLOCKED] {reason}")
        summary = f"{host_label}; [BLOCKED] " + "; ".join(blocking_reasons)
        status = "blocked"
    elif adjustments:
        summary = (
            f"{host_label}; safeguarded training params -> "
            f"imgsz={effective_imgsz}, batch={effective_batch}, workers={effective_workers}"
        )
        status = "adjusted"
    else:
        summary = (
            f"{host_label}; current training params fit within safeguard -> "
            f"imgsz={effective_imgsz}, batch={effective_batch}, workers={effective_workers}"
        )
        status = "ok"

    return {
        "profile": profile,
        "status": status,
        "summary": summary,
        "blocking_reasons": blocking_reasons,
        "model_scale": model_scale,
        "system_specs": {
            "system": specs.system,
            "machine": specs.machine,
            "cpu_logical_cores": specs.cpu_logical_cores,
            "cpu_physical_cores": specs.cpu_physical_cores,
            "cpu_brand": specs.cpu_brand,
            "total_memory_bytes": specs.total_memory_bytes,
            "total_memory_gb": round(specs.total_memory_gb, 2),
            "accelerator": specs.accelerator,
            "gpu_name": specs.gpu_name,
            "gpu_memory_bytes": specs.gpu_memory_bytes,
            "gpu_memory_gb": None if specs.gpu_memory_gb is None else round(specs.gpu_memory_gb, 2),
            "gpu_count": specs.gpu_count,
            "gpus": [dict(g) for g in specs.gpus],
            "free_disk_bytes": free_disk,
            "free_disk_gb": round(free_disk / float(_GIB), 2) if free_disk else 0.0,
        },
        "requested_hyperparams": requested,
        "effective_hyperparams": effective,
        "limits": {
            "max_imgsz": max_imgsz,
            "max_batch": max_batch,
            "max_workers": max_workers,
        },
        "derivation": derivation,
        "projections": {
            "vram": projection,
        },
        "adjustments": adjustments,
    }


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the same training system guard as cvLayer without the full repo.",
    )
    p.add_argument(
        "--base-model",
        default="yolo11n.pt",
        help="Model filename/path for YOLO scale inference (e.g. yolo11m.pt).",
    )
    p.add_argument("--guard-profile", default="balanced", choices=_GUARD_PROFILES)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=None, help="Omit to mimic unset / Ultralytics default.")
    p.add_argument("--workers", type=int, default=None, help="Omit to mimic unset.")
    p.add_argument(
        "--json",
        action="store_true",
        help="Print the full guard dict as JSON (for scripts and logs).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the verdict block (readiness, can_run, memory risk, and two summary lines).",
    )
    p.add_argument(
        "--strict-exit",
        action="store_true",
        help="Exit 0=ok, 2=adjusted, 1=blocked. Default: exit 1 only if blocked, else 0.",
    )
    return p.parse_args(argv)


def _build_verdict(out: dict[str, Any]) -> dict[str, Any]:
    st = str(out.get("status") or "ok").lower()
    vram = (out.get("projections") or {}).get("vram")
    vram = vram if isinstance(vram, dict) else {}
    risk = str(vram.get("risk") or "unknown")
    can_run = st != "blocked"
    if st == "blocked":
        readiness = "blocked"
        primary = (
            "This system cannot run the workload: training is blocked by the system guard. "
            "Address the items under blocking_reasons (typically RAM, free disk, or both)."
        )
    elif st == "adjusted":
        readiness = "ready_with_safeguards"
        primary = (
            "This system can handle the load using the safeguarded effective parameters. "
            "Values were reduced from the request to fit CPU/GPU and profile limits; use those to train."
        )
    else:
        readiness = "ready"
        primary = (
            "This system can handle the load for the requested model, profile, and guard limits without "
            "further clamping of imgsz, batch, or workers."
        )

    if risk == "over":
        mem_note = (
            "Memory projection: estimated peak is at or over the available budget. Training may OOM; "
            "reduce batch or imgsz if runs fail."
        )
    elif risk == "tight":
        mem_note = (
            "Memory projection: headroom is tight. Training will likely work but is close to the budget."
        )
    elif risk == "ok":
        mem_note = (
            "Memory projection: estimated peak fits the budget with comfortable headroom (heuristic; "
            "actual peak is workload-dependent)."
        )
    else:
        mem_note = (
            "Memory projection: not fully assessed (e.g. missing psutil for RAM, or no CUDA VRAM read). "
            "Install psutil and a GPU build of PyTorch for a full projection on this host."
        )

    return {
        "readiness": readiness,
        "can_run_workload": can_run,
        "memory_risk": risk,
        "lines": (primary, mem_note),
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    hyperparams: dict[str, Any] = {
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "guard_profile": args.guard_profile,
    }
    out = build_training_guard(str(args.base_model or ""), hyperparams)
    verdict = _build_verdict(out)
    if args.json:
        payload: dict[str, Any] = dict(out)
        payload["verdict"] = {k: v for k, v in verdict.items() if k != "lines"}
        payload["verdict"]["summary_lines"] = list(verdict["lines"])
        print(json.dumps(payload, indent=2))
    elif args.quiet:
        print(
            f"readiness={verdict['readiness']}  can_run_workload={verdict['can_run_workload']}  "
            f"memory_risk={verdict['memory_risk']}"
        )
        for line in verdict["lines"]:
            print(line)
    else:
        print(out.get("summary", ""))
        print()
        st = str(out.get("status") or "ok")
        print(f"status={st}  model_scale={out.get('model_scale')}  profile={out.get('profile')}")
        br = out.get("blocking_reasons") or []
        if br:
            print("blocking_reasons:")
            for r in br:
                print(f"  - {r}")
        lim = out.get("limits") or {}
        if isinstance(lim, dict) and lim:
            print("limits (max):", lim)
        eff = out.get("effective_hyperparams")
        if isinstance(eff, dict) and eff:
            print("effective:", eff)
        adj = out.get("adjustments") or []
        if adj:
            print("adjustments:")
            for a in adj:
                print(f"  - {a}")
        print()
        print("--- Verdict ---")
        print(
            f"Readiness: {verdict['readiness']}  |  can_run_workload: {verdict['can_run_workload']}  |  "
            f"memory_risk: {verdict['memory_risk']}"
        )
        for line in verdict["lines"]:
            print(f"  {line}")
    status = str(out.get("status") or "ok").lower()
    if not args.strict_exit:
        return 0 if status != "blocked" else 1
    if status == "blocked":
        return 1
    if status == "adjusted":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
