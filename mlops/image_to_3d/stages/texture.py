from __future__ import annotations

from pathlib import Path

from PIL import Image

from mlops.image_to_3d.scene import Scene


def run(scene: Scene, image_path: Path) -> Path:
    out = Path(scene.root) / "texture.png"
    Image.open(image_path).convert("RGB").save(out)
    scene.add_artifact(
        out,
        kind="texture",
        source="user",
        confidence=0.8,
        stage_version="projected-image-texture@v1",
        depends_on=[Path(image_path).name],
    )
    return out
