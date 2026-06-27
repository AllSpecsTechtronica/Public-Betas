from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    scenario TEXT NOT NULL,
    job_type TEXT NOT NULL,
    state TEXT NOT NULL,
    source TEXT NOT NULL,
    image_path TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    error TEXT,
    result_ref TEXT,
    payload_json TEXT NOT NULL,
    cancel_requested INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS job_results (
    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
    result_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_state_created ON jobs(state, created_at);
"""


@dataclass
class JobRecord:
    job_id: str
    scenario: str
    job_type: str
    state: str
    source: str
    image_path: str
    created_at: float
    started_at: Optional[float]
    finished_at: Optional[float]
    error: str
    result_ref: str
    payload: dict[str, Any]
    cancel_requested: bool = False

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "JobRecord":
        payload_raw = row["payload_json"]
        try:
            payload = json.loads(payload_raw) if payload_raw else {}
        except Exception:
            payload = {}
        try:
            cancel = bool(row["cancel_requested"])
        except Exception:
            cancel = False
        return cls(
            job_id=str(row["job_id"]),
            scenario=str(row["scenario"]),
            job_type=str(row["job_type"]),
            state=str(row["state"]),
            source=str(row["source"]),
            image_path=str(row["image_path"]),
            created_at=float(row["created_at"]),
            started_at=float(row["started_at"]) if row["started_at"] is not None else None,
            finished_at=float(row["finished_at"]) if row["finished_at"] is not None else None,
            error=str(row["error"] or ""),
            result_ref=str(row["result_ref"] or ""),
            payload=payload,
            cancel_requested=cancel,
        )

    def to_dict(self) -> dict[str, Any]:
        now = time.time()
        end = self.finished_at
        if end is None and self.state in {"queued", "running"}:
            end = now
        if end is None:
            end = self.started_at if self.started_at is not None else self.created_at
        queue_end = self.started_at if self.started_at is not None else end
        queue_duration = max(0.0, float(queue_end) - self.created_at) if queue_end is not None else None
        run_duration = (
            max(0.0, float(end) - float(self.started_at))
            if self.started_at is not None and end is not None
            else None
        )
        total_duration = max(0.0, float(end) - self.created_at) if end is not None else None
        return {
            "job_id": self.job_id,
            "scenario": self.scenario,
            "job_type": self.job_type,
            "state": self.state,
            "source": self.source,
            "image_path": self.image_path,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "result_ref": self.result_ref,
            "payload": self.payload,
            "cancel_requested": self.cancel_requested,
            "queue_duration_seconds": queue_duration,
            "run_duration_seconds": run_duration,
            "total_duration_seconds": total_duration,
        }


class JobStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(jobs)").fetchall()}
        if "cancel_requested" not in cols:
            self._conn.execute(
                "ALTER TABLE jobs ADD COLUMN cancel_requested INTEGER NOT NULL DEFAULT 0"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def create_job(
        self,
        *,
        job_id: str,
        scenario: str,
        job_type: str,
        source: str,
        image_path: str,
        payload: dict[str, Any],
    ) -> JobRecord:
        created_at = time.time()
        payload_json = json.dumps(payload, ensure_ascii=True)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (job_id, scenario, job_type, state, source, image_path, created_at, payload_json)
                VALUES (?, ?, ?, 'queued', ?, ?, ?, ?)
                """,
                (job_id, scenario, job_type, source, image_path, created_at, payload_json),
            )
            self._conn.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str) -> JobRecord:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return JobRecord.from_row(row)

    def list_jobs(self, limit: int = 200) -> list[JobRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                (max(1, min(limit, 1000)),),
            ).fetchall()
        return [JobRecord.from_row(row) for row in rows]

    def list_running_train_scenarios(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT scenario FROM jobs WHERE state = 'running' AND job_type = 'train'"
            ).fetchall()
        return {str(r[0]) for r in rows if r and r[0]}

    def claim_next_queued_job(
        self,
        *,
        job_types: Optional[tuple[str, ...]] = None,
        exclude_scenarios_with_running: Optional[set[str]] = None,
    ) -> Optional[JobRecord]:
        """Claim the oldest queued job, optionally restricted to given job_types.

        If exclude_scenarios_with_running is provided, any job whose scenario is
        currently running (in that set) will be skipped. Used to ensure only one
        training job per scenario runs at a time while still allowing parallel
        inference.
        """
        with self._lock:
            if job_types:
                placeholders = ",".join("?" * len(job_types))
                rows = self._conn.execute(
                    f"SELECT * FROM jobs WHERE state = 'queued' AND job_type IN ({placeholders}) "
                    "ORDER BY created_at ASC",
                    tuple(job_types),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM jobs WHERE state = 'queued' ORDER BY created_at ASC"
                ).fetchall()
            pick = None
            for row in rows:
                if exclude_scenarios_with_running and str(row["scenario"]) in exclude_scenarios_with_running:
                    continue
                pick = row
                break
            if pick is None:
                return None
            job_id = str(pick["job_id"])
            started_at = time.time()
            self._conn.execute(
                "UPDATE jobs SET state = 'running', started_at = ? WHERE job_id = ? AND state = 'queued'",
                (started_at, job_id),
            )
            self._conn.commit()
        return self.get_job(job_id)

    def request_cancel(self, job_id: str) -> JobRecord:
        """Mark a job as cancel_requested. If queued, immediately move to error.

        For running jobs, the worker polls cancel_requested via was_cancel_requested()
        and aborts at the next safe point.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            state = str(row["state"])
            self._conn.execute(
                "UPDATE jobs SET cancel_requested = 1 WHERE job_id = ?", (job_id,)
            )
            if state == "queued":
                finished = time.time()
                self._conn.execute(
                    "UPDATE jobs SET state = 'error', error = ?, finished_at = ? WHERE job_id = ?",
                    ("cancelled before start", finished, job_id),
                )
            self._conn.commit()
        return self.get_job(job_id)

    def was_cancel_requested(self, job_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT cancel_requested FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        if row is None:
            return False
        try:
            return bool(row["cancel_requested"])
        except Exception:
            return False

    def set_job_state(
        self,
        job_id: str,
        state: str,
        *,
        error: str = "",
        result_ref: str = "",
        finished_at: Optional[float] = None,
    ) -> JobRecord:
        with self._lock:
            if finished_at is None and state in {"done", "error"}:
                finished_at = time.time()
            clear_cancel = 1 if state in {"done", "error"} else 0
            self._conn.execute(
                """
                UPDATE jobs
                SET state = ?, error = ?, result_ref = ?, finished_at = COALESCE(?, finished_at),
                    cancel_requested = CASE WHEN ? = 1 THEN 0 ELSE cancel_requested END
                WHERE job_id = ?
                """,
                (state, error, result_ref, finished_at, clear_cancel, job_id),
            )
            self._conn.commit()
        return self.get_job(job_id)

    def enqueue_retry(self, job_id: str) -> JobRecord:
        """Create a new queued job cloned from a failed job."""
        old = self.get_job(job_id)
        if old.state != "error":
            raise ValueError("only failed jobs can be retried")
        new_id = f"job-{uuid.uuid4().hex[:12]}"
        payload = dict(old.payload) if isinstance(old.payload, dict) else {}
        payload.setdefault("retried_from", job_id)
        return self.create_job(
            job_id=new_id,
            scenario=old.scenario,
            job_type=old.job_type,
            source=str(old.source or "cvops"),
            image_path=old.image_path,
            payload=payload,
        )

    def write_result(self, job_id: str, result: dict[str, Any]) -> None:
        raw = json.dumps(result, ensure_ascii=True)
        with self._lock:
            self._conn.execute(
                "INSERT INTO job_results(job_id, result_json) VALUES (?, ?) "
                "ON CONFLICT(job_id) DO UPDATE SET result_json = excluded.result_json",
                (job_id, raw),
            )
            self._conn.commit()

    def get_result(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT result_json FROM job_results WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        if row is None:
            return None
        raw = row["result_json"]
        try:
            return json.loads(raw) if raw else None
        except Exception:
            return None

    def batch_results(self, job_ids: list[str]) -> dict[str, dict]:
        """Return {job_id: result_dict} for the given job IDs in one query.

        Missing or unparseable results are omitted from the returned dict.
        """
        if not job_ids:
            return {}
        placeholders = ",".join("?" * len(job_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT job_id, result_json FROM job_results WHERE job_id IN ({placeholders})",
                job_ids,
            ).fetchall()
        out: dict[str, dict] = {}
        for row in rows:
            try:
                parsed = json.loads(row["result_json"] or "")
                if isinstance(parsed, dict):
                    out[str(row["job_id"])] = parsed
            except Exception:
                pass
        return out
