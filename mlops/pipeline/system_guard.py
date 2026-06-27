from __future__ import annotations

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
_MIN_SAFE_RAM_BYTES = 4 * _GIB
_MIN_SAFE_DISK_BYTES = 1 * _GIB
_OVERFLOW_MIN_FREE_BYTES = 2 * _GIB
# Prefer a roomier mounted volume over the internal models root when the
# primary location is below this free threshold so long runs do not exhaust
# the system disk while an external SSD still has space.
_PRIMARY_COMFORT_BYTES = 8 * _GIB


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
    """Return free disk bytes for the mlops models root (falls back to cwd)."""
    probe: Path
    if target is not None:
        probe = target
    else:
        try:
            from .registry import MLOPS_ROOT
            probe = Path(MLOPS_ROOT)
        except Exception:
            probe = Path.cwd()
    try:
        while not probe.exists():
            if probe.parent == probe:
                break
            probe = probe.parent
        return int(shutil.disk_usage(probe).free)
    except Exception:
        return 0


def _disk_state(path: Path) -> dict[str, Any]:
    probe = path
    try:
        while not probe.exists():
            if probe.parent == probe:
                break
            probe = probe.parent
        usage = shutil.disk_usage(probe)
        free = int(usage.free)
        total = int(usage.total)
        used = int(usage.used)
        return {
            "path": str(path),
            "probe": str(probe),
            "total_bytes": total,
            "used_bytes": used,
            "free_bytes": free,
            "total_gb": round(total / float(_GIB), 2) if total else 0.0,
            "used_gb": round(used / float(_GIB), 2) if used else 0.0,
            "free_gb": round(free / float(_GIB), 2) if free else 0.0,
            "used_pct": round((used / total) * 100.0, 1) if total else None,
            "available": free >= _OVERFLOW_MIN_FREE_BYTES,
        }
    except Exception as exc:
        return {
            "path": str(path),
            "probe": str(probe),
            "total_bytes": 0,
            "used_bytes": 0,
            "free_bytes": 0,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
            "used_pct": None,
            "available": False,
            "error": str(exc),
        }


def _mounted_storage_roots() -> list[Path]:
    roots: list[Path] = [Path("/")]
    if psutil is not None:
        try:
            for part in psutil.disk_partitions(all=False):
                mount = Path(str(part.mountpoint or ""))
                if not str(mount):
                    continue
                opts = str(getattr(part, "opts", "") or "").lower().split(",")
                if "ro" in opts:
                    continue
                roots.append(mount)
        except Exception:
            pass
    volumes = Path("/Volumes")
    try:
        if volumes.exists():
            roots.extend(p for p in volumes.iterdir() if p.is_dir())
    except Exception:
        pass

    out: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        out.append(root)
    return out


def _preferred_asset_root(hyperparams: Mapping[str, Any]) -> Path | None:
    for key in ("training_assets_root", "asset_save_root", "save_root"):
        value = str(hyperparams.get(key) or "").strip()
        if value:
            return Path(value).expanduser()
    value = str(os.environ.get("CVLAYER_TRAINING_ASSETS_ROOT") or "").strip()
    if value:
        return Path(value).expanduser()
    return None


def _default_model_root(scenario: str | None) -> Path:
    name = str(scenario or "").strip()
    try:
        from .registry import MLOPS_ROOT

        root = Path(MLOPS_ROOT) / "models"
    except Exception:
        root = Path.cwd() / "mlops" / "models"
    return root / name if name else root


def _overflow_candidate(label: str, kind: str, asset_root: Path, disk_path: Path | None = None) -> dict[str, Any]:
    state = _disk_state(disk_path or asset_root)
    state.update({
        "label": label,
        "kind": kind,
        "asset_root": str(asset_root),
        "min_required_bytes": _OVERFLOW_MIN_FREE_BYTES,
        "min_required_gb": round(_OVERFLOW_MIN_FREE_BYTES / float(_GIB), 2),
    })
    return state


def _volume_inventory() -> list[dict[str, Any]]:
    """All local writable mount points with usage (for operator diagnostics)."""
    out: list[dict[str, Any]] = []
    if psutil is None:
        return out
    try:
        for part in psutil.disk_partitions(all=False):
            mount = Path(str(part.mountpoint or ""))
            if not str(mount):
                continue
            opts = str(getattr(part, "opts", "") or "").lower().split(",")
            if "ro" in opts:
                continue
            st = _disk_state(mount)
            st["fstype"] = str(getattr(part, "fstype", "") or "")
            st["device"] = str(getattr(part, "device", "") or "")
            out.append(st)
    except Exception:
        return out
    return out


def _select_overflow_active(deduped: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick where training checkpoints should live.

    Order: explicit preferred root; else internal primary if it has enough
    headroom; otherwise the candidate with the most free space (typically an
    external volume discovered under /Volumes).
    """
    if not deduped:
        return None
    eligible = [c for c in deduped if int(c.get("free_bytes") or 0) >= _OVERFLOW_MIN_FREE_BYTES]
    pool = eligible if eligible else list(deduped)
    if not pool:
        return None
    preferred = next((c for c in pool if c.get("kind") == "preferred"), None)
    if preferred is not None:
        return preferred
    primary = next((c for c in pool if c.get("kind") == "primary"), None)
    primary_free = int(primary.get("free_bytes") or 0) if primary else 0
    if primary is not None and primary_free >= _PRIMARY_COMFORT_BYTES:
        return primary
    return max(pool, key=lambda c: int(c.get("free_bytes") or 0))


def build_overflow_protocol(
    hyperparams: Mapping[str, Any],
    *,
    scenario: str | None = None,
    exclude_roots: list[str] | None = None,
) -> dict[str, Any]:
    """Choose where training assets should be written.

    Overflow protocol is intentionally always on. The primary location is the
    normal cvLayer models directory unless the user supplies a save root via
    `training_assets_root`, `asset_save_root`, `save_root`, or the
    `CVLAYER_TRAINING_ASSETS_ROOT` environment variable.

    When the internal models volume drops below ~8 GiB free but another volume
    still has ample space, checkpoints are steered to the roomiest candidate so
    training does not silently consume the rest of the system disk.
    """
    default_root = _default_model_root(scenario)
    preferred_root = _preferred_asset_root(hyperparams)
    excluded = {str(Path(p).expanduser()) for p in (exclude_roots or []) if str(p or "").strip()}

    candidates: list[dict[str, Any]] = []
    if preferred_root is not None:
        asset_root = preferred_root / str(scenario) if scenario and preferred_root.name != str(scenario) else preferred_root
        candidates.append(_overflow_candidate("preferred save location", "preferred", asset_root))
    candidates.append(_overflow_candidate("primary cvLayer models", "primary", default_root))

    for mount in _mounted_storage_roots():
        if mount == Path("/"):
            asset_root = mount / "cvLayer_overflow" / "mlops" / "models"
        else:
            asset_root = mount / "cvLayer_overflow" / "mlops" / "models"
        if scenario:
            asset_root = asset_root / str(scenario)
        candidates.append(_overflow_candidate(f"mounted volume {mount}", "overflow", asset_root, disk_path=mount))

    deduped: list[dict[str, Any]] = []
    seen_assets: set[str] = set()
    for candidate in candidates:
        key = str(candidate.get("asset_root") or "")
        if not key or key in seen_assets or key in excluded:
            continue
        seen_assets.add(key)
        deduped.append(candidate)

    active = _select_overflow_active(deduped)
    if active is None:
        nonzero = [c for c in deduped if int(c.get("free_bytes") or 0) > 0]
        if nonzero:
            active = max(nonzero, key=lambda c: int(c.get("free_bytes") or 0))

    overflowed = bool(active and active.get("kind") not in {"primary", "preferred"})
    no_space = not bool(active and active.get("available"))
    active_root = str(active.get("asset_root") or "") if active else ""
    message = ""
    if no_space:
        message = (
            "Overflow protocol could not find enough free space on any detected drive. "
            "Training is blocked to avoid filling the system disk."
        )
    elif overflowed:
        message = (
            "Overflow protocol encountered a data overflow risk on the primary location "
            f"and is sending this training batch to {active_root}."
        )
    else:
        message = f"Overflow protocol is active; training assets will be saved to {active_root}."

    for candidate in deduped:
        candidate["active"] = bool(active_root and candidate.get("asset_root") == active_root)

    return {
        "enabled": True,
        "status": "no_space" if no_space else ("overflow" if overflowed else "ok"),
        "message": message,
        "active_asset_root": active_root,
        "active_label": str(active.get("label") or "") if active else "",
        "overflowed": overflowed,
        "min_required_bytes": _OVERFLOW_MIN_FREE_BYTES,
        "min_required_gb": round(_OVERFLOW_MIN_FREE_BYTES / float(_GIB), 2),
        "primary_comfort_bytes": _PRIMARY_COMFORT_BYTES,
        "primary_comfort_gb": round(_PRIMARY_COMFORT_BYTES / float(_GIB), 2),
        "drives": deduped,
        "volume_inventory": _volume_inventory(),
    }


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


def build_training_guard(
    base_model: str,
    hyperparams: Mapping[str, Any],
    *,
    scenario: str | None = None,
    exclude_asset_roots: list[str] | None = None,
) -> dict[str, Any]:
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

    overflow_protocol = build_overflow_protocol(
        hyperparams,
        scenario=scenario,
        exclude_roots=exclude_asset_roots,
    )
    active_asset_root = str(overflow_protocol.get("active_asset_root") or "")
    free_disk = _free_disk_bytes(Path(active_asset_root) if active_asset_root else None)
    blocking_reasons: list[str] = []
    if specs.total_memory_bytes and specs.total_memory_bytes < _MIN_SAFE_RAM_BYTES:
        blocking_reasons.append(
            f"only {specs.total_memory_gb:.1f} GB RAM detected; training requires at least "
            f"{_MIN_SAFE_RAM_BYTES / _GIB:.0f} GB"
        )
    if overflow_protocol.get("status") == "no_space":
        blocking_reasons.append(str(overflow_protocol.get("message") or "no drive has enough free space"))
    elif free_disk and free_disk < _MIN_SAFE_DISK_BYTES:
        blocking_reasons.append(
            f"only {free_disk / _GIB:.1f} GB free disk on models volume; training requires at least "
            f"{_MIN_SAFE_DISK_BYTES / _GIB:.0f} GB"
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
        "overflow_protocol": overflow_protocol,
        "adjustments": adjustments,
    }
