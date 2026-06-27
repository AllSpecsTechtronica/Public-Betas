"""Host capability checks for the owned image-to-3D pipeline."""

from __future__ import annotations

import importlib.util
import os
import platform
import socket
from dataclasses import asdict, dataclass
from pathlib import Path


DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
COREML_DEPTH_PACKAGE_NAME = "DepthAnythingSmallF16.mlpackage"
REPO_ROOT = Path(__file__).resolve().parents[2]
INSIGHT_DEPTH_MODEL_DIR = (
    REPO_ROOT
    / "Insight"
    / "insight_local"
    / "Insight_assets"
    / "models"
    / "DepthAnythingModelSmall"
)


@dataclass
class CapabilityItem:
    available: bool
    reason: str = ""


@dataclass
class Capabilities:
    os: str
    torch: CapabilityItem
    mps: CapabilityItem
    depth_model: CapabilityItem
    coreml_depth: CapabilityItem
    transformers_depth: CapabilityItem
    trellis_cloud: CapabilityItem
    default_device: str

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)


def detect(check_trellis: bool = False) -> Capabilities:
    os_name = platform.system().lower()
    torch_item = CapabilityItem(importlib.util.find_spec("torch") is not None)
    mps_item = CapabilityItem(False, "torch is not installed")
    default_device = "cpu"
    if torch_item.available:
        try:
            import torch  # type: ignore[import-not-found]

            if bool(torch.backends.mps.is_available()):
                mps_item = CapabilityItem(True)
                default_device = "mps"
            else:
                mps_item = CapabilityItem(False, "MPS is not available on this host")
        except Exception as exc:
            mps_item = CapabilityItem(False, f"torch probe failed: {exc}")

    coreml_item = _coreml_depth_status()
    transformers_item = _transformers_depth_status()
    depth_item = coreml_item if coreml_item.available else transformers_item
    trellis_item = _trellis_status() if check_trellis else CapabilityItem(True, "not checked")
    return Capabilities(
        os=os_name,
        torch=torch_item,
        mps=mps_item,
        depth_model=depth_item,
        coreml_depth=coreml_item,
        transformers_depth=transformers_item,
        trellis_cloud=trellis_item,
        default_device=default_device,
    )


def models_dir() -> Path:
    return Path(os.environ.get("CVLAYER_MODELS_DIR", str(REPO_ROOT / "models"))).expanduser().resolve()


def coreml_tmp_dir() -> Path:
    override = os.environ.get("IMAGE_TO_3D_COREML_TMPDIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return (REPO_ROOT / ".cvlayer_coreml_tmp").resolve()


def coreml_depth_model_path() -> Path:
    override = os.environ.get("IMAGE_TO_3D_DEPTH_MODEL_PATH", "").strip()
    if override:
        return normalize_coreml_depth_model_path(Path(override).expanduser()).resolve()
    candidates = [
        models_dir() / COREML_DEPTH_PACKAGE_NAME,
        INSIGHT_DEPTH_MODEL_DIR / COREML_DEPTH_PACKAGE_NAME,
    ]
    for candidate in candidates:
        if is_coreml_depth_model_package(candidate):
            return candidate.resolve()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _coreml_depth_status() -> CapabilityItem:
    path = coreml_depth_model_path()
    if not path.exists():
        return CapabilityItem(False, f"missing: {path}")
    if not path.is_dir():
        return CapabilityItem(False, f"not an .mlpackage directory: {path}")
    if not is_coreml_depth_model_package(path):
        return CapabilityItem(False, f"incomplete .mlpackage: {path}")
    if importlib.util.find_spec("coremltools") is None:
        return CapabilityItem(False, f"coremltools is not installed; local model at {path}")
    return CapabilityItem(True, str(path))


def is_coreml_depth_model_package(path: Path) -> bool:
    root = normalize_coreml_depth_model_path(Path(path).expanduser())
    return (root / "Manifest.json").exists() and (
        _coreml_payload_exists(root / "Data" / "com.apple.CoreML")
        or _coreml_payload_exists(root / "com.apple.CoreML")
    )


def normalize_coreml_depth_model_path(path: Path) -> Path:
    if path.suffix == ".mlpackage":
        return path
    nested = path / COREML_DEPTH_PACKAGE_NAME
    return nested if nested.exists() else path


def _coreml_payload_exists(payload_root: Path) -> bool:
    return (payload_root / "model.mlmodel").exists() and (payload_root / "weights").exists()


def _transformers_depth_status() -> CapabilityItem:
    if importlib.util.find_spec("transformers") is None:
        return CapabilityItem(False, "transformers is not installed")
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / ("models--" + DEPTH_MODEL_ID.replace("/", "--"))
    if model_dir.exists():
        return CapabilityItem(True, "cached")
    return CapabilityItem(True, "will download on first run if network is available")


def _trellis_status() -> CapabilityItem:
    try:
        with socket.create_connection(("microsoft-trellis-2.hf.space", 443), timeout=3):
            return CapabilityItem(True)
    except Exception as exc:
        return CapabilityItem(False, f"unreachable: {exc}")
