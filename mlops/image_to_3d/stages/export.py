from __future__ import annotations

import shutil
from pathlib import Path

from mlops.image_to_3d.scene import Scene


def run(scene: Scene, mesh_path: Path) -> Path:
    out = Path(scene.root) / "scene.glb"
    if Path(mesh_path).resolve() != out.resolve():
        shutil.copyfile(mesh_path, out)
    scene.add_artifact(
        out,
        kind="scene",
        source="image_depth",
        confidence=0.52,
        stage_version="scene-export@v1",
        depends_on=["mesh.glb", "texture.png", "floor_plane.json"],
    )
    scene.save()
    return out
