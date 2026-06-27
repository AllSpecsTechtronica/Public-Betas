"""Single-image camera intrinsics helpers."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fov_degrees: float = 50.0
    source: str = "default"
    distortion: list[float] | None = None

    def to_jsonable(self) -> dict[str, object]:
        return asdict(self)

    def save(self, path: Path) -> Path:
        path.write_text(json.dumps(self.to_jsonable(), indent=2), encoding="utf-8")
        return path


def default_intrinsics(width: int, height: int, fov_degrees: float = 50.0) -> Intrinsics:
    fov = math.radians(float(fov_degrees))
    focal = (float(width) * 0.5) / math.tan(fov * 0.5)
    return Intrinsics(
        width=int(width),
        height=int(height),
        fx=focal,
        fy=focal,
        cx=(float(width) - 1.0) * 0.5,
        cy=(float(height) - 1.0) * 0.5,
        fov_degrees=float(fov_degrees),
        source="default",
        distortion=[0.0, 0.0, 0.0, 0.0],
    )
