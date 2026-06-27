"""Disk-backed job state for the owned image-to-3D pipeline."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    stage: str = ""
    message: str = ""
    progress: float = 0.0
    error: str = ""
    input_path: str = ""
    depth_vis_path: str = ""
    points_path: str = ""
    mesh_path: str = ""
    scene_path: str = ""
    provenance_path: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    stage_status: dict[str, str] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_jsonable(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        data["progress"] = max(0.0, min(1.0, float(self.progress)))
        return data


class JobStore:
    def __init__(self, root: Path | None = None) -> None:
        self.root = (root or Path.home() / ".cvlayer" / "image_to_3d" / "jobs").expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create(self, params: dict[str, Any]) -> Job:
        job_id = self.new_id()
        self.dir(job_id).mkdir(parents=True, exist_ok=True)
        now = time.time()
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            params=dict(params),
            created_at=now,
            updated_at=now,
        )
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        job.updated_at = time.time()
        path = self.dir(job.job_id) / "status.json"
        path.write_text(json.dumps(job.to_jsonable(), indent=2), encoding="utf-8")

    def load(self, job_id: str) -> Job | None:
        path = self.dir(job_id) / "status.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            status = JobStatus(data.get("status", "queued"))
        except Exception:
            return None
        return Job(
            job_id=str(data.get("job_id") or job_id),
            status=status,
            stage=str(data.get("stage") or ""),
            message=str(data.get("message") or ""),
            progress=float(data.get("progress") or 0.0),
            error=str(data.get("error") or ""),
            input_path=str(data.get("input_path") or ""),
            depth_vis_path=str(data.get("depth_vis_path") or ""),
            points_path=str(data.get("points_path") or ""),
            mesh_path=str(data.get("mesh_path") or ""),
            scene_path=str(data.get("scene_path") or ""),
            provenance_path=str(data.get("provenance_path") or ""),
            params=dict(data.get("params") or {}),
            stage_status=dict(data.get("stage_status") or {}),
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
        )

    def list_recent(self, limit: int = 10) -> list[Job]:
        jobs: list[Job] = []
        if not self.root.exists():
            return jobs
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            job = self.load(d.name)
            if job is not None:
                jobs.append(job)
            if len(jobs) >= limit:
                break
        return jobs
