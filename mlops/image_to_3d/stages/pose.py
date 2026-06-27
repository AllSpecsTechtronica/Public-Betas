from __future__ import annotations

from pathlib import Path

from PIL import Image

from mlops.image_to_3d.intrinsics import default_intrinsics
from mlops.image_to_3d.scene import Scene


def run(scene: Scene, image_path: Path, *, fov_degrees: float = 50.0) -> Path:
    with Image.open(image_path) as image:
        intr = default_intrinsics(image.width, image.height, fov_degrees=fov_degrees)
    out = Path(scene.root) / "intrinsics.json"
    intr.save(out)
    scene.add_artifact(
        out,
        kind="intrinsics",
        source=intr.source,
        confidence=0.65,
        stage_version="single-image-pinhole@v1",
        depends_on=[Path(image_path).name],
    )
    return out
