from __future__ import annotations

import shutil
from pathlib import Path
from typing import Callable

from mlops.image_to_3d.models.segment import segment_foreground
from mlops.image_to_3d.scene import Scene
from mlops.trellis2 import DEFAULT_PARAMS, SamplingParams, Trellis2Client


StatusCallback = Callable[[str, str, float], None]


def run(
    scene: Scene,
    image_path: Path,
    *,
    params: dict[str, object] | None = None,
    on_status: StatusCallback | None = None,
) -> list[Path]:
    objects_root = Path(scene.root) / "objects"
    segments = segment_foreground(image_path, objects_root / "0", max_objects=1)
    if not segments:
        scene.add_provenance(
            "objects",
            source="trellis",
            confidence=0.0,
            stage_version="trellis-enhance@v1",
            depends_on=[Path(image_path).name],
            status="skipped",
            message="no foreground segment found",
        )
        return []

    client = Trellis2Client()
    trellis_params = _sampling_params(params or {})
    outputs: list[Path] = []
    for idx, segment in enumerate(segments):
        obj_dir = objects_root / str(idx)
        obj_dir.mkdir(parents=True, exist_ok=True)
        crop_path = obj_dir / "crop.png"
        if segment.crop_path != crop_path:
            shutil.copyfile(segment.crop_path, crop_path)
        scene.add_artifact(
            crop_path,
            kind="image",
            source="image_depth",
            confidence=segment.confidence,
            stage_version="foreground-segment@v1",
            depends_on=[Path(image_path).name],
        )
        result = client.generate(
            image_path=crop_path,
            params=trellis_params,
            out_dir=obj_dir,
            on_status=on_status or (lambda _s, _m, _p=-1.0: None),
        )
        glb_value = str(result.get("glb_path") or "")
        src = Path(glb_value) if glb_value else Path()
        if glb_value and src.is_file():
            dest = obj_dir / "asset.glb"
            if src.resolve() != dest.resolve():
                shutil.copyfile(src, dest)
            outputs.append(dest)
            scene.add_artifact(
                dest,
                kind="mesh",
                source="trellis",
                confidence=0.7,
                stage_version="trellis2-cloud@v1",
                depends_on=[str(crop_path.relative_to(Path(scene.root)))],
            )
    return outputs


def record_failure(scene: Scene, message: str) -> None:
    scene.add_provenance(
        "objects",
        source="trellis",
        confidence=0.0,
        stage_version="trellis-enhance@v1",
        depends_on=["preprocessed.png"],
        status="failed",
        message=message,
    )


def _sampling_params(raw: dict[str, object]) -> SamplingParams:
    base = DEFAULT_PARAMS.as_dict()
    for key, value in raw.items():
        if key in base:
            base[key] = value
    return SamplingParams(**base)
