"""On-device depth → mesh GLB using the bundled DepthAnything Core ML package.

Expects ``Insight/insight_local/Insight_assets/models/DepthAnythingModelSmall/*.mlpackage``
(or override via ``INSIGHT_DEPTH_ANYTHING_MLPACKAGE``).

Requires: Pillow, numpy, coremltools (prediction), trimesh (GLB export).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .client import SamplingParams, Trellis2Error

StatusCallback = Callable[[str, str, float], None]


def _noop(_stage: str, _message: str, _progress: float = -1.0) -> None:
    return None


def _repo_root() -> Path:
    # mlops/trellis2/<this> -> parents[2] == workspace root (cvLayer)
    return Path(__file__).resolve().parents[2]


def resolve_depth_mlpackage() -> Optional[Path]:
    """Return path to ``*.mlpackage`` if present."""
    env = os.environ.get("INSIGHT_DEPTH_ANYTHING_MLPACKAGE", "").strip()
    if env:
        p = Path(env).expanduser().resolve()
        if p.suffix == ".mlpackage" and p.is_dir():
            return p
    bundle_dir = (
        _repo_root()
        / "Insight"
        / "insight_local"
        / "Insight_assets"
        / "models"
        / "DepthAnythingModelSmall"
    )
    if not bundle_dir.is_dir():
        return None
    packs = sorted(bundle_dir.glob("*.mlpackage"))
    return packs[0] if packs else None


def depth_bundle_available() -> tuple[bool, str]:
    p = resolve_depth_mlpackage()
    if p is None:
        return False, (
            "No DepthAnything .mlpackage found. "
            "Place one under Insight_assets/models/DepthAnythingModelSmall/ "
            "or set INSIGHT_DEPTH_ANYTHING_MLPACKAGE."
        )
    return True, str(p)


def _depth_from_ml_output(raw: Any) -> np.ndarray:
    if isinstance(raw, np.ndarray):
        arr = raw.astype(np.float32).squeeze()
        return arr
    if hasattr(raw, "numpy") and callable(raw.numpy):
        arr = np.asarray(raw.numpy()).astype(np.float32).squeeze()
        return arr
    return _depth_to_numpy(_pil_from_depth_output(raw))


def _pil_from_depth_output(raw: Any) -> Any:
    from PIL import Image

    if isinstance(raw, Image.Image):
        return raw
    if hasattr(raw, "astype"):
        arr = np.asarray(raw)
        return Image.fromarray(arr)
    raise Trellis2Error(f"unexpected depth output type: {type(raw)!r}")


def _depth_to_numpy(depth_img: Any) -> np.ndarray:
    from PIL import Image

    if isinstance(depth_img, Image.Image):
        arr = np.asarray(depth_img.convert("F"))
        return arr.astype(np.float32)
    arr = np.asarray(depth_img)
    return arr.astype(np.float32)


def _run_coreml_depth(model_path: Path, image_path: Path) -> np.ndarray:
    try:
        import coremltools as ct  # noqa: PLC0415
    except Exception as exc:
        raise Trellis2Error(
            "coremltools is required for the local DepthAnything backend. "
            "Install with: pip install coremltools"
        ) from exc

    from PIL import Image  # noqa: PLC0415

    try:
        ml = ct.models.MLModel(str(model_path))
    except Exception as exc:
        raise Trellis2Error(f"failed to load Core ML model at {model_path}: {exc}") from exc

    spec = ml.get_spec()
    if not spec.description.input:
        raise Trellis2Error("Core ML model has no inputs")
    inp = spec.description.input[0]
    if inp.type.WhichOneof("Type") != "imageType":
        raise Trellis2Error("Expected image input for DepthAnything Core ML model")
    it = inp.type.imageType
    target_w = int(it.width)
    target_h = int(it.height)

    img = Image.open(image_path).convert("RGB").resize(
        (target_w, target_h),
        Image.Resampling.LANCZOS,
    )

    try:
        out = ml.predict({"image": img})
    except Exception as exc:
        raise Trellis2Error(
            "Core ML prediction failed. On Apple Silicon this usually runs on-device; "
            f"details: {exc}"
        ) from exc

    if "depth" not in out:
        raise Trellis2Error(f"model output missing 'depth' key; got {list(out.keys())}")

    depth = _depth_from_ml_output(out["depth"])
    if depth.ndim != 2:
        raise Trellis2Error(f"expected HxW depth map, got shape {depth.shape}")
    return depth


def _subsample_depth(depth: np.ndarray, max_faces: int) -> np.ndarray:
    """Reduce grid resolution so triangle count stays under ``max_faces``."""
    H, W = depth.shape
    stride = 1
    while True:
        hs = max(2, (H + stride - 1) // stride)
        ws = max(2, (W + stride - 1) // stride)
        fc = (hs - 1) * (ws - 1) * 2
        if fc <= max_faces:
            return depth[::stride, ::stride]
        stride += 1
        if stride > max(H, W, 1):
            return depth[:: max(H // 2, 1), :: max(W // 2, 1)]


def _build_mesh(depth: np.ndarray, scale_z: float = 1.85) -> tuple[np.ndarray, np.ndarray]:
    H, W = depth.shape
    d = depth.astype(np.float64)
    d = d - np.nanmin(d)
    d = d / (np.nanmax(d) - np.nanmin(d) + 1e-8)
    # Larger metric = farther in many monocular models — pull closer vertices forward.
    Z = (1.0 - d) * scale_z
    aspect = float(W) / float(H)
    ys = np.linspace(-1.0, 1.0, H, dtype=np.float64)
    xs = np.linspace(-aspect, aspect, W, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys)
    verts = np.column_stack([XX.ravel(), YY.ravel(), Z.ravel()])
    faces: list[list[int]] = []
    for i in range(H - 1):
        for j in range(W - 1):
            v00 = i * W + j
            v01 = i * W + (j + 1)
            v10 = (i + 1) * W + j
            v11 = (i + 1) * W + (j + 1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    farr = np.asarray(faces, dtype=np.int64)
    return verts, farr


def _export_glb(verts: np.ndarray, faces: np.ndarray, dest: Path) -> None:
    try:
        import trimesh  # noqa: PLC0415
    except Exception as exc:
        raise Trellis2Error(
            "trimesh is required to export GLB. Install with: pip install trimesh"
        ) from exc

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mesh.export(dest, file_type="glb")


def _write_preview_png(depth: np.ndarray, dest: Path) -> None:
    from PIL import Image  # noqa: PLC0415

    d = depth.astype(np.float32)
    d = d - float(np.min(d))
    d = d / (float(np.max(d)) - float(np.min(d)) + 1e-8)
    u8 = (d * 255.0).clip(0, 255).astype(np.uint8)
    Image.fromarray(u8, mode="L").save(dest)


def generate_depth_glb(
    *,
    mlpackage_path: Path,
    image_path: Path,
    out_dir: Path,
    params: SamplingParams,
    on_status: StatusCallback = _noop,
) -> dict[str, str]:
    """Run DepthAnything and build a heightfield mesh GLB (no TRELLIS texture pipeline)."""
    out_dir.mkdir(parents=True, exist_ok=True)

    on_status("depth", "running DepthAnything (Core ML)", 0.15)
    depth = _run_coreml_depth(mlpackage_path, image_path)
    on_status("depth", "building mesh from depth map", 0.55)

    max_faces = max(5_000, int(params.decimation_target))
    depth_work = _subsample_depth(depth, max_faces=max_faces)

    verts, faces = _build_mesh(depth_work)
    glb_path = out_dir / "output.glb"
    _export_glb(verts, faces, glb_path)

    preview_path = out_dir / "preview.png"
    _write_preview_png(depth, preview_path)

    preview_html_path = out_dir / "preview.html"
    preview_html_path.write_text(
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/>"
        "<title>Depth preview</title></head>"
        '<body style="margin:0;background:#111;color:#ccc;font-family:sans-serif;">'
        f'<p style="padding:8px;">DepthAnything heightfield (local). '
        f"<code>{mlpackage_path.name}</code></p>"
        '<img src="preview.png" style="max-width:100%;height:auto;display:block;"/>'
        "</body></html>\n",
        encoding="utf-8",
    )

    on_status("done", "complete", 1.0)
    return {
        "preview_path": str(preview_path),
        "preview_html_path": str(preview_html_path),
        "glb_path": str(glb_path),
        "seed": "",
    }
