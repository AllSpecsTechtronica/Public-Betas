from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mlops.image_to_3d.scene import Scene


def run(scene: Scene, depth_path: Path, *, floor_percentile: float = 85.0) -> Path:
    depth = np.load(depth_path).astype("float32")
    lower_band = depth[int(depth.shape[0] * 0.66) :, :]
    threshold = float(np.percentile(lower_band, float(floor_percentile))) if lower_band.size else 0.0
    confidence = 0.35
    if lower_band.size:
        near_floor = lower_band >= threshold
        confidence = max(0.2, min(0.65, float(near_floor.mean()) * 2.0))
    data = {
        "source": "ransac_stub",
        "plane": {"normal": [0.0, 1.0, 0.0], "offset": 0.0},
        "confidence": confidence,
        "notes": "Single-image v1 emits a conservative floor-plane estimate for downstream collision work.",
    }
    out = Path(scene.root) / "floor_plane.json"
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    scene.add_artifact(
        out,
        kind="walkability",
        source="ransac",
        confidence=confidence,
        stage_version="floor-plane-stub@v1",
        depends_on=["depth.npy"],
    )
    return out
