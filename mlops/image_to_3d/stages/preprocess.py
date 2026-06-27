from __future__ import annotations

from pathlib import Path

from PIL import Image

from mlops.image_to_3d.scene import Scene


def run(scene: Scene, input_path: Path, *, max_size: int = 768) -> Path:
    image = Image.open(input_path).convert("RGBA")
    image.thumbnail((int(max_size), int(max_size)))
    out = Path(scene.root) / "preprocessed.png"
    image.save(out)
    scene.add_artifact(
        out,
        kind="image",
        source="user",
        confidence=1.0,
        stage_version="preprocess@v1",
        depends_on=[Path(input_path).name],
    )
    return out
