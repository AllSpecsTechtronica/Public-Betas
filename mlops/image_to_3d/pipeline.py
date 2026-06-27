"""Stage orchestrator for the owned image-to-3D pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from mlops.image_to_3d.capability import DEPTH_MODEL_ID
from mlops.image_to_3d.jobs import Job, JobStatus, JobStore
from mlops.image_to_3d.scene import Scene
from mlops.image_to_3d.stages import depth, enhance_trellis, export, geometry, pose, preprocess, texture, walkability


StatusCallback = Callable[[str, str, float], None]


@dataclass
class PipelineConfig:
    max_image_size: int = 768
    fov_degrees: float = 50.0
    depth_model_id: str = DEPTH_MODEL_ID
    depth_backend: str = "auto"
    depth_model_path: str = ""
    device: str | None = None
    mesh_stride: int = 2
    max_depth: float = 4.0
    floor_percentile: float = 85.0
    enhance_trellis: bool = False
    trellis_params: dict[str, object] = field(default_factory=dict)
    stages: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object] | None) -> "PipelineConfig":
        raw = dict(data or {})
        allowed = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in raw.items() if k in allowed})

    def stage_enabled(self, name: str) -> bool:
        if not self.stages:
            return True
        return bool(self.stages.get(name, True))


def run_pipeline(
    *,
    job: Job,
    store: JobStore,
    input_path: Path,
    config: PipelineConfig,
    on_status: StatusCallback | None = None,
) -> Job:
    scene = Scene(job_id=job.job_id, root=str(store.dir(job.job_id)))

    def status(stage: str, message: str, progress: float) -> None:
        fresh = store.load(job.job_id) or job
        fresh.status = JobStatus.RUNNING
        fresh.stage = stage
        fresh.message = message
        fresh.progress = max(0.0, min(1.0, float(progress)))
        fresh.stage_status[stage] = "running"
        store.save(fresh)
        if on_status:
            on_status(stage, message, fresh.progress)

    def mark(stage: str, state: str) -> None:
        fresh = store.load(job.job_id) or job
        fresh.stage_status[stage] = state
        store.save(fresh)

    try:
        scene.add_artifact(
            input_path,
            kind="image",
            source="user",
            confidence=1.0,
            stage_version="upload@v1",
        )
        current_image = input_path
        status("preprocess", "preprocessing input image", 0.03)
        if config.stage_enabled("preprocess"):
            current_image = preprocess.run(scene, input_path, max_size=config.max_image_size)
        mark("preprocess", "ok")

        status("pose", "creating default camera intrinsics", 0.12)
        intrinsics_path = pose.run(scene, current_image, fov_degrees=config.fov_degrees)
        mark("pose", "ok")

        status("depth", "estimating monocular depth", 0.20)
        depth_path, depth_vis_path = depth.run(
            scene,
            current_image,
            model_id=config.depth_model_id,
            device=config.device,
            backend=config.depth_backend,
            model_path=config.depth_model_path or None,
        )
        _update_paths(store, job.job_id, depth_vis_path=depth_vis_path)
        mark("depth", "ok")

        status("geometry", "building RGBD point cloud and mesh", 0.48)
        points_path, mesh_path = geometry.run(
            scene,
            current_image,
            depth_path,
            intrinsics_path,
            mesh_stride=config.mesh_stride,
            max_depth=config.max_depth,
        )
        _update_paths(store, job.job_id, points_path=points_path, mesh_path=mesh_path)
        mark("geometry", "ok")

        status("texture", "saving projected texture", 0.66)
        texture.run(scene, current_image)
        mark("texture", "ok")

        status("walkability", "estimating floor plane", 0.74)
        walkability.run(scene, depth_path, floor_percentile=config.floor_percentile)
        mark("walkability", "ok")

        if config.enhance_trellis and config.stage_enabled("enhance_trellis"):
            status("enhance_trellis", "enhancing foreground object with TRELLIS", 0.80)
            try:
                enhance_trellis.run(
                    scene,
                    current_image,
                    params=config.trellis_params,
                    on_status=lambda s, m, p=-1.0: status(
                        "enhance_trellis",
                        f"{s}: {m}",
                        0.80 + 0.14 * max(0.0, min(1.0, p if p >= 0 else 0.0)),
                    ),
                )
                mark("enhance_trellis", "ok")
            except Exception as exc:
                enhance_trellis.record_failure(scene, str(exc))
                mark("enhance_trellis", "failed")

        status("export", "assembling scene export", 0.96)
        scene_path = export.run(scene, mesh_path)
        _update_paths(store, job.job_id, scene_path=scene_path, provenance_path=Path(scene.root) / "provenance.json")
        mark("export", "ok")

        fresh = store.load(job.job_id) or job
        fresh.status = JobStatus.COMPLETED
        fresh.stage = "done"
        fresh.message = "complete"
        fresh.progress = 1.0
        store.save(fresh)
        return fresh
    except Exception as exc:
        scene.add_provenance(
            "pipeline",
            source="image_to_3d",
            confidence=0.0,
            stage_version="pipeline@v1",
            status="failed",
            message=str(exc),
        )
        scene.save()
        fresh = store.load(job.job_id) or job
        fresh.status = JobStatus.FAILED
        fresh.stage = "error"
        fresh.error = str(exc)
        fresh.provenance_path = str(Path(scene.root) / "provenance.json")
        store.save(fresh)
        return fresh


def _update_paths(store: JobStore, job_id: str, **paths: Path) -> None:
    fresh = store.load(job_id)
    if fresh is None:
        return
    for key, value in paths.items():
        setattr(fresh, key, str(value))
    store.save(fresh)
