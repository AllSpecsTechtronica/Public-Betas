"""Host capability detection for the 3D Generation panel.

Mirrors the Go gateway's capability probe: the cloud (Hugging Face) backend is
always available; the local engine is only viable on Linux/Windows hosts with
an NVIDIA CUDA GPU. On macOS we always fall back to cloud.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from dataclasses import dataclass


@dataclass
class GPU:
    nvidia: bool = False
    cuda: bool = False
    name: str = ""


@dataclass
class BackendStatus:
    available: bool
    reason: str = ""


@dataclass
class Capabilities:
    os: str
    gpu: GPU
    cloud: BackendStatus
    local: BackendStatus
    default_backend: str


def _probe_gpu() -> GPU:
    if shutil.which("nvidia-smi") is None:
        return GPU()
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return GPU()
    if out.returncode != 0 or not out.stdout.strip():
        return GPU()
    first = out.stdout.strip().splitlines()[0]
    name = first.split(",", 1)[0].strip() if "," in first else first.strip()
    return GPU(nvidia=True, cuda=True, name=name)


def detect() -> Capabilities:
    os_name = platform.system().lower()  # "darwin", "windows", "linux"
    gpu = _probe_gpu()

    cloud = BackendStatus(available=True)

    if os_name == "darwin":
        local = BackendStatus(
            available=False,
            reason="Local TRELLIS.2 requires Linux or Windows with an NVIDIA CUDA GPU.",
        )
        default = "cloud"
    elif os_name in {"windows", "linux"}:
        if gpu.nvidia and gpu.cuda:
            local = BackendStatus(available=True)
            default = "local"
        else:
            local = BackendStatus(
                available=False,
                reason="No NVIDIA CUDA GPU detected.",
            )
            default = "cloud"
    else:
        local = BackendStatus(available=False, reason="Unsupported OS for local engine.")
        default = "cloud"

    return Capabilities(os=os_name, gpu=gpu, cloud=cloud, local=local, default_backend=default)
