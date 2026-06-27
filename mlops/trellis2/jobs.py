"""On-disk job state for the 3D Generation panel.

Each job lives at ``~/.trellis2/jobs/<job_id>/`` and writes ``status.json`` so
Streamlit reruns can read live progress without holding thread state in
session_state.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    job_id: str
    status: JobStatus
    backend: str
    stage: str = ""
    message: str = ""
    progress: float = 0.0
    error: str = ""
    input_path: str = ""
    preview_path: str = ""
    preview_html_path: str = ""
    glb_path: str = ""
    seed: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    def to_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class JobStore:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = (root or Path.home() / ".trellis2" / "jobs").expanduser()
        self.root.mkdir(parents=True, exist_ok=True)

    def new_id(self) -> str:
        return f"job_{int(time.time())}_{uuid.uuid4().hex[:6]}"

    def dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create(self, backend: str, params: dict[str, Any]) -> Job:
        job_id = self.new_id()
        d = self.dir(job_id)
        d.mkdir(parents=True, exist_ok=True)
        now = time.time()
        job = Job(
            job_id=job_id,
            status=JobStatus.QUEUED,
            backend=backend,
            params=params,
            created_at=now,
            updated_at=now,
        )
        self.save(job)
        return job

    def save(self, job: Job) -> None:
        job.updated_at = time.time()
        path = self.dir(job.job_id) / "status.json"
        path.write_text(json.dumps(job.to_jsonable(), indent=2), encoding="utf-8")

    def load(self, job_id: str) -> Optional[Job]:
        path = self.dir(job_id) / "status.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        return Job(
            job_id=data.get("job_id", job_id),
            status=JobStatus(data.get("status", "queued")),
            backend=data.get("backend", "cloud"),
            stage=data.get("stage", ""),
            message=data.get("message", ""),
            progress=float(data.get("progress") or 0.0),
            error=data.get("error", ""),
            input_path=data.get("input_path", ""),
            preview_path=data.get("preview_path", ""),
            preview_html_path=data.get("preview_html_path", ""),
            glb_path=data.get("glb_path", ""),
            seed=data.get("seed", ""),
            params=data.get("params") or {},
            created_at=float(data.get("created_at") or 0.0),
            updated_at=float(data.get("updated_at") or 0.0),
        )

    def list_recent(self, limit: int = 10) -> list[Job]:
        jobs: list[Job] = []
        for d in sorted(self.root.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if not d.is_dir():
                continue
            j = self.load(d.name)
            if j is not None:
                jobs.append(j)
            if len(jobs) >= limit:
                break
        return jobs
