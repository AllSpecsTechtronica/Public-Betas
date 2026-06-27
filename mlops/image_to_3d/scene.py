"""Canonical scene and provenance records for the image-to-3D pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Artifact:
    path: str
    kind: str
    source: str
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Provenance:
    artifact: str
    source: str
    confidence: float
    stage_version: str
    depends_on: list[str] = field(default_factory=list)
    status: str = "ok"
    message: str = ""


@dataclass
class Scene:
    job_id: str
    root: str
    coordinate_system: str = "single_image_camera"
    artifacts: list[Artifact] = field(default_factory=list)
    provenance: list[Provenance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_artifact(
        self,
        path: Path | str,
        *,
        kind: str,
        source: str,
        confidence: float = 1.0,
        metadata: dict[str, Any] | None = None,
        stage_version: str = "",
        depends_on: list[str] | None = None,
        status: str = "ok",
        message: str = "",
    ) -> None:
        artifact_path = _relpath(path, Path(self.root))
        confidence = max(0.0, min(1.0, float(confidence)))
        self.artifacts.append(
            Artifact(
                path=artifact_path,
                kind=kind,
                source=source,
                confidence=confidence,
                metadata=dict(metadata or {}),
            )
        )
        self.provenance.append(
            Provenance(
                artifact=artifact_path,
                source=source,
                confidence=confidence,
                stage_version=stage_version,
                depends_on=list(depends_on or []),
                status=status,
                message=message,
            )
        )

    def add_provenance(
        self,
        artifact: str,
        *,
        source: str,
        confidence: float,
        stage_version: str,
        depends_on: list[str] | None = None,
        status: str = "ok",
        message: str = "",
    ) -> None:
        self.provenance.append(
            Provenance(
                artifact=artifact,
                source=source,
                confidence=max(0.0, min(1.0, float(confidence))),
                stage_version=stage_version,
                depends_on=list(depends_on or []),
                status=status,
                message=message,
            )
        )

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    def save(self) -> Path:
        path = Path(self.root) / "provenance.json"
        path.write_text(json.dumps(self.to_jsonable(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> "Scene":
        data = json.loads(path.read_text(encoding="utf-8"))
        scene = cls(
            job_id=str(data["job_id"]),
            root=str(data["root"]),
            coordinate_system=str(data.get("coordinate_system") or "single_image_camera"),
            metadata=dict(data.get("metadata") or {}),
        )
        scene.artifacts = [Artifact(**item) for item in data.get("artifacts", [])]
        scene.provenance = [Provenance(**item) for item in data.get("provenance", [])]
        return scene


def _relpath(path: Path | str, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)
