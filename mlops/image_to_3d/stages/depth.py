from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mlops.image_to_3d.models.depth import DepthModel
from mlops.image_to_3d.scene import Scene


def run(
    scene: Scene,
    image_path: Path,
    *,
    model_id: str,
    device: str | None = None,
    backend: str = "auto",
    model_path: str | Path | None = None,
) -> tuple[Path, Path]:
    model = DepthModel(model_id=model_id, device=device, backend=backend, model_path=model_path)
    depth = model.predict(image_path)
    depth_path = Path(scene.root) / "depth.npy"
    np.save(depth_path, depth)
    vis_path = Path(scene.root) / "depth_vis.png"
    _save_depth_vis(depth, vis_path)
    source = f"mono_depth_{model.backend_used or backend}"
    scene.add_artifact(
        depth_path,
        kind="depth",
        source=source,
        confidence=0.62,
        stage_version=f"{model.model_path if model.backend_used == 'coreml' else model_id}@v1",
        depends_on=[Path(image_path).name],
    )
    scene.add_artifact(
        vis_path,
        kind="image",
        source=source,
        confidence=0.62,
        stage_version=f"{model.model_path if model.backend_used == 'coreml' else model_id}@vis-v1",
        depends_on=["depth.npy"],
    )
    return depth_path, vis_path


def _save_depth_vis(depth: np.ndarray, path: Path) -> None:
    arr = np.asarray(depth, dtype="float32")
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    r = (arr * 255).astype("uint8")
    g = ((1.0 - abs(arr - 0.5) * 2.0) * 255).clip(0, 255).astype("uint8")
    b = ((1.0 - arr) * 255).astype("uint8")
    rgb = np.dstack([r, g, b])
    Image.fromarray(rgb, mode="RGB").save(path)
