from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mlops.pipeline import registry as reg

JOB_STATES = ("pending", "scraping", "staged", "labeling", "emitted", "error")


@dataclass
class JobState:
    slug: str
    topic: str
    target_count: int
    state: str = "pending"
    message: str = ""
    raw_count: int = 0
    staged_count: int = 0
    classes: list[str] = field(default_factory=list)
    labels: dict[str, list[list[float]]] = field(default_factory=dict)
    """Map staged-image filename -> list of [class_idx, cx, cy, w, h] (normalized)."""
    processing_log: list[str] = field(default_factory=list)
    """Append-only pipeline trace (UI + disk); truncated in ``append_log``."""
    scrape_paused: bool = False
    """Operator pause — downloader thread sleeps while True."""
    scrape_generation: int = 0
    """Bumped when a new worker is started — previous workers treat as cancelled."""
    last_scrape_query: str = ""
    """Query string last used / to use for Continue or Restart downloads."""
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class JobStore:
    """Per-slug job state with disk persistence at `database/<slug>/scrap.json`.

    Streamlit reruns the script on every interaction, so all in-memory state
    must round-trip through the JSON file. A process-local lock guards
    read/modify/write cycles within one Streamlit worker.

    Use an RLock: ``update`` / ``append_log`` / ``bump_scrape_generation`` hold
    the lock while calling ``save``, which also acquires the same lock.
    A plain ``Lock`` would deadlock on those paths (UI freeze on Pause, etc.).
    """

    _lock = threading.RLock()

    @staticmethod
    def state_path(slug: str) -> Path:
        return reg.resolve_library_dataset_path(slug) / "scrap.json"

    @classmethod
    def exists(cls, slug: str) -> bool:
        try:
            return cls.state_path(slug).exists()
        except Exception:
            return False

    @classmethod
    def load(cls, slug: str) -> JobState | None:
        try:
            path = cls.state_path(slug)
        except Exception:
            return None
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        raw_log = data.get("processing_log")
        if isinstance(raw_log, list):
            proc_log = [str(x) for x in raw_log]
        else:
            proc_log = []
        return JobState(
            slug=str(data.get("slug") or slug),
            topic=str(data.get("topic") or ""),
            target_count=int(data.get("target_count") or 0),
            state=str(data.get("state") or "pending"),
            message=str(data.get("message") or ""),
            raw_count=int(data.get("raw_count") or 0),
            staged_count=int(data.get("staged_count") or 0),
            classes=list(data.get("classes") or []),
            labels={str(k): [list(map(float, b)) for b in v] for k, v in (data.get("labels") or {}).items()},
            processing_log=proc_log,
            scrape_paused=bool(data.get("scrape_paused")),
            scrape_generation=int(data.get("scrape_generation") or 0),
            last_scrape_query=str(data.get("last_scrape_query") or ""),
            created_at=float(data.get("created_at") or time.time()),
            updated_at=float(data.get("updated_at") or time.time()),
        )

    @classmethod
    def save(cls, job: JobState) -> None:
        path = cls.state_path(job.slug)
        path.parent.mkdir(parents=True, exist_ok=True)
        job.updated_at = time.time()
        payload: dict[str, Any] = {
            "slug": job.slug,
            "topic": job.topic,
            "target_count": job.target_count,
            "state": job.state,
            "message": job.message,
            "raw_count": job.raw_count,
            "staged_count": job.staged_count,
            "classes": job.classes,
            "labels": job.labels,
            "processing_log": job.processing_log,
            "scrape_paused": job.scrape_paused,
            "scrape_generation": job.scrape_generation,
            "last_scrape_query": job.last_scrape_query,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        with cls._lock:
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, path)

    @classmethod
    def update(cls, slug: str, **fields: Any) -> JobState | None:
        with cls._lock:
            job = cls.load(slug)
            if job is None:
                return None
            for k, v in fields.items():
                if hasattr(job, k):
                    setattr(job, k, v)
            cls.save(job)
            return job

    @classmethod
    def append_log(cls, slug: str, line: str, *, max_lines: int = 500) -> None:
        """Append one timestamped line to ``processing_log`` (thread-safe)."""
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {line}"
        with cls._lock:
            job = cls.load(slug)
            if job is None:
                return
            merged = list(job.processing_log) + [entry]
            job.processing_log = merged[-max_lines:]
            cls.save(job)

    @classmethod
    def bump_scrape_generation(cls, slug: str) -> int | None:
        """Increment ``scrape_generation`` so concurrent workers halt at pause checkpoints."""
        with cls._lock:
            job = cls.load(slug)
            if job is None:
                return None
            job.scrape_generation = int(job.scrape_generation or 0) + 1
            cls.save(job)
            return job.scrape_generation
