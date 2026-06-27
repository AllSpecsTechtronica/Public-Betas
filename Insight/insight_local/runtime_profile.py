from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class RuntimeProfile:
    system: str
    machine: str
    torch_device: str
    accelerator: str
    has_cuda: bool
    has_mps: bool
    onnx_providers: tuple[str, ...]
    preferred_model_suffixes: tuple[str, ...]

    @property
    def is_apple_silicon(self) -> bool:
        return self.system == "Darwin" and self.machine.lower() in {"arm64", "aarch64"}

    @property
    def is_windows(self) -> bool:
        return self.system == "Windows"

    @property
    def is_linux(self) -> bool:
        return self.system == "Linux"

    @property
    def has_nvidia_gpu(self) -> bool:
        return self.has_cuda


def _detect_torch_device() -> tuple[str, bool, bool]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", True, False
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps", False, True
    except ImportError:
        pass
    return "cpu", False, False


def _detect_onnx_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort

        return tuple(str(provider) for provider in ort.get_available_providers())
    except Exception:
        return ()


def _preferred_model_suffixes(system: str, machine: str, has_cuda: bool) -> tuple[str, ...]:
    if system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
        return (".mlpackage", ".onnx", ".pt", ".engine")
    if system in {"Windows", "Linux"} and has_cuda:
        return (".engine", ".onnx", ".pt", ".mlpackage")
    return (".pt", ".engine", ".onnx", ".mlpackage")


@lru_cache(maxsize=1)
def profile_runtime() -> RuntimeProfile:
    system = platform.system()
    machine = platform.machine()
    torch_device, has_cuda, has_mps = _detect_torch_device()
    onnx_providers = _detect_onnx_providers()

    if system == "Darwin" and machine.lower() in {"arm64", "aarch64"}:
        accelerator = "apple-silicon"
    elif system in {"Windows", "Linux"} and has_cuda:
        accelerator = "nvidia-cuda"
    else:
        accelerator = "cpu"

    return RuntimeProfile(
        system=system,
        machine=machine,
        torch_device=torch_device,
        accelerator=accelerator,
        has_cuda=has_cuda,
        has_mps=has_mps,
        onnx_providers=onnx_providers,
        preferred_model_suffixes=_preferred_model_suffixes(system, machine, has_cuda),
    )


def pick_torch_device() -> str:
    return profile_runtime().torch_device


def pick_onnx_providers() -> list[str]:
    """Return the preferred ONNX Runtime execution-provider list for this host.

    Priority order:
      Apple Silicon  -> CoreMLExecutionProvider > CPUExecutionProvider
      NVIDIA CUDA    -> CUDAExecutionProvider   > CPUExecutionProvider
      Everything else-> CPUExecutionProvider
    """
    rp = profile_runtime()
    available = set(rp.onnx_providers)
    if rp.is_apple_silicon and "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    if rp.has_cuda and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]
