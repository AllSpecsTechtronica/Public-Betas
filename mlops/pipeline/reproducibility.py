from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_deterministic_policy(seed: int) -> dict[str, Any]:
    seed = int(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch_state: dict[str, Any] = {"available": False}
    try:
        import torch

        torch_state["available"] = True
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Best-effort determinism; avoid crashing when unsupported.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
        torch_state["version"] = str(getattr(torch, "__version__", ""))
        torch_state["cuda_available"] = bool(torch.cuda.is_available())
        torch_state["cuda"] = str(getattr(torch.version, "cuda", "") or "")
    except Exception:
        pass
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    return {"seed": seed, "torch": torch_state}


def _pip_freeze() -> list[str]:
    cmd = [sys.executable, "-m", "pip", "freeze"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=45)
    except Exception:
        return []
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    return sorted(lines)


def _git_sha() -> str:
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=8)
    except Exception:
        return ""
    return str(out or "").strip()


def capture_environment_fingerprint(run_dir: Path) -> dict[str, Any]:
    requirements = _pip_freeze()
    lock_path = run_dir / "env.requirements.lock"
    if requirements:
        lock_path.write_text("\n".join(requirements) + "\n", encoding="utf-8")
    env = {
        "captured_at": _utc_now(),
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "git_sha": _git_sha(),
        "requirements_lock_path": str(lock_path) if lock_path.exists() else "",
        "requirements_count": len(requirements),
        "cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
    }
    return env


def create_repro_manifest(
    *,
    run_dir: Path,
    scenario: str,
    base_model: str,
    data_yaml: str,
    dataset_snapshot_id: str,
    hyperparams: dict[str, Any],
    env: dict[str, Any],
) -> dict[str, Any]:
    replay_cmd = (
        f"python -m mlops.pipeline.replay --manifest \"{(run_dir / 'repro_manifest.json').resolve()}\""
    )
    manifest = {
        "version": 1,
        "created_at": _utc_now(),
        "scenario": scenario,
        "run_dir": str(run_dir.resolve()),
        "base_model": base_model,
        "data_yaml": data_yaml,
        "dataset_snapshot_id": dataset_snapshot_id,
        "hyperparams": dict(hyperparams or {}),
        "environment": dict(env or {}),
        "replay_command": replay_cmd,
    }
    path = run_dir / "repro_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, default=str), encoding="utf-8")
    return manifest

