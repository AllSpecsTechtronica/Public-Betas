from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


VALID_RANGE_MODES = {"single", "compare", "lineage_sweep"}
VALID_DRIFT_KINDS = {"label_noise", "covariate_shift", "corruption", "imbalance"}
VALID_THRESHOLD_TYPES = {"absolute", "delta_from_baseline", "delta_from_prev"}
VALID_GATE_ACTIONS = {"warn", "block"}

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS ranges (
    range_id           TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    sector_id          TEXT NOT NULL,
    sector_path        TEXT NOT NULL,
    description        TEXT NOT NULL DEFAULT '',
    mode               TEXT NOT NULL,
    config_json        TEXT NOT NULL DEFAULT '{}',
    tags_json          TEXT NOT NULL DEFAULT '[]',
    metadata_json      TEXT NOT NULL DEFAULT '{}',
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL,
    last_run_at        REAL
);

CREATE INDEX IF NOT EXISTS idx_range_sector ON ranges(sector_path);

CREATE TABLE IF NOT EXISTS range_subjects (
    range_id           TEXT NOT NULL REFERENCES ranges(range_id) ON DELETE CASCADE,
    snapshot_id        TEXT NOT NULL,
    label              TEXT NOT NULL,
    added_at           REAL NOT NULL,
    PRIMARY KEY (range_id, snapshot_id)
);

CREATE INDEX IF NOT EXISTS idx_range_subjects_snap ON range_subjects(snapshot_id);

CREATE TABLE IF NOT EXISTS golden_sets (
    golden_id          TEXT PRIMARY KEY,
    range_id           TEXT NOT NULL REFERENCES ranges(range_id) ON DELETE CASCADE,
    name               TEXT NOT NULL,
    description        TEXT NOT NULL DEFAULT '',
    split_spec_json    TEXT NOT NULL,
    storage_uri        TEXT NOT NULL,
    row_count          INTEGER NOT NULL,
    content_sha256     TEXT NOT NULL,
    sealed_at          REAL NOT NULL,
    UNIQUE(range_id, name)
);

CREATE INDEX IF NOT EXISTS idx_golden_range ON golden_sets(range_id);

CREATE TABLE IF NOT EXISTS drift_scenarios (
    drift_id           TEXT PRIMARY KEY,
    range_id           TEXT NOT NULL REFERENCES ranges(range_id) ON DELETE CASCADE,
    name               TEXT NOT NULL,
    kind               TEXT NOT NULL,
    params_json        TEXT NOT NULL DEFAULT '{}',
    created_at         REAL NOT NULL,
    UNIQUE(range_id, name)
);

CREATE INDEX IF NOT EXISTS idx_drift_range ON drift_scenarios(range_id);

CREATE TABLE IF NOT EXISTS evaluations (
    eval_id            TEXT PRIMARY KEY,
    range_id           TEXT NOT NULL REFERENCES ranges(range_id) ON DELETE CASCADE,
    snapshot_id        TEXT NOT NULL,
    golden_id          TEXT NOT NULL REFERENCES golden_sets(golden_id) ON DELETE CASCADE,
    drift_id           TEXT REFERENCES drift_scenarios(drift_id) ON DELETE SET NULL,
    metrics_json       TEXT NOT NULL DEFAULT '{}',
    predictions_uri    TEXT,
    ran_at             REAL NOT NULL,
    duration_ms        INTEGER NOT NULL DEFAULT 0,
    UNIQUE(snapshot_id, golden_id, drift_id)
);

CREATE INDEX IF NOT EXISTS idx_eval_range    ON evaluations(range_id);
CREATE INDEX IF NOT EXISTS idx_eval_snap     ON evaluations(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_eval_golden   ON evaluations(golden_id);
CREATE INDEX IF NOT EXISTS idx_eval_drift    ON evaluations(drift_id);

CREATE TABLE IF NOT EXISTS regression_gates (
    gate_id            TEXT PRIMARY KEY,
    range_id           TEXT NOT NULL REFERENCES ranges(range_id) ON DELETE CASCADE,
    golden_id          TEXT REFERENCES golden_sets(golden_id) ON DELETE CASCADE,
    metric             TEXT NOT NULL,
    threshold_type     TEXT NOT NULL,
    threshold_value    REAL NOT NULL,
    baseline_snapshot_id TEXT,
    action             TEXT NOT NULL DEFAULT 'warn',
    created_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gate_range ON regression_gates(range_id);
"""


# ---------------------------------------------------------------------------
# Record dataclasses
# ---------------------------------------------------------------------------

def _load_list(raw: Any) -> list[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except Exception:
        return []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        token = str(item or "").strip()
        if not token:
            continue
        norm = token.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append(token)
    return out


def _load_obj(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class RangeRecord:
    range_id: str
    name: str
    sector_id: str
    sector_path: str
    description: str
    mode: str
    config: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0
    last_run_at: Optional[float] = None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RangeRecord":
        last = row["last_run_at"]
        return cls(
            range_id=str(row["range_id"]),
            name=str(row["name"]),
            sector_id=str(row["sector_id"]),
            sector_path=str(row["sector_path"]),
            description=str(row["description"] or ""),
            mode=str(row["mode"]),
            config=_load_obj(row["config_json"]),
            tags=_load_list(row["tags_json"]),
            metadata=_load_obj(row["metadata_json"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_run_at=float(last) if last is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "range_id": self.range_id,
            "name": self.name,
            "sector_id": self.sector_id,
            "sector_path": self.sector_path,
            "description": self.description,
            "mode": self.mode,
            "config": dict(self.config),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_run_at": self.last_run_at,
        }


@dataclass(frozen=True)
class SubjectRecord:
    range_id: str
    snapshot_id: str
    label: str
    added_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SubjectRecord":
        return cls(
            range_id=str(row["range_id"]),
            snapshot_id=str(row["snapshot_id"]),
            label=str(row["label"]),
            added_at=float(row["added_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "range_id": self.range_id,
            "snapshot_id": self.snapshot_id,
            "label": self.label,
            "added_at": self.added_at,
        }


@dataclass(frozen=True)
class GoldenSetRecord:
    golden_id: str
    range_id: str
    name: str
    description: str
    split_spec: dict[str, Any] = field(default_factory=dict)
    storage_uri: str = ""
    row_count: int = 0
    content_sha256: str = ""
    sealed_at: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "GoldenSetRecord":
        return cls(
            golden_id=str(row["golden_id"]),
            range_id=str(row["range_id"]),
            name=str(row["name"]),
            description=str(row["description"] or ""),
            split_spec=_load_obj(row["split_spec_json"]),
            storage_uri=str(row["storage_uri"]),
            row_count=int(row["row_count"] or 0),
            content_sha256=str(row["content_sha256"] or ""),
            sealed_at=float(row["sealed_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "golden_id": self.golden_id,
            "range_id": self.range_id,
            "name": self.name,
            "description": self.description,
            "split_spec": dict(self.split_spec),
            "storage_uri": self.storage_uri,
            "row_count": self.row_count,
            "content_sha256": self.content_sha256,
            "sealed_at": self.sealed_at,
        }


@dataclass(frozen=True)
class DriftScenarioRecord:
    drift_id: str
    range_id: str
    name: str
    kind: str
    params: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DriftScenarioRecord":
        return cls(
            drift_id=str(row["drift_id"]),
            range_id=str(row["range_id"]),
            name=str(row["name"]),
            kind=str(row["kind"]),
            params=_load_obj(row["params_json"]),
            created_at=float(row["created_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "drift_id": self.drift_id,
            "range_id": self.range_id,
            "name": self.name,
            "kind": self.kind,
            "params": dict(self.params),
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class EvaluationRecord:
    eval_id: str
    range_id: str
    snapshot_id: str
    golden_id: str
    drift_id: str
    metrics: dict[str, Any] = field(default_factory=dict)
    predictions_uri: str = ""
    ran_at: float = 0.0
    duration_ms: int = 0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "EvaluationRecord":
        return cls(
            eval_id=str(row["eval_id"]),
            range_id=str(row["range_id"]),
            snapshot_id=str(row["snapshot_id"]),
            golden_id=str(row["golden_id"]),
            drift_id=str(row["drift_id"] or ""),
            metrics=_load_obj(row["metrics_json"]),
            predictions_uri=str(row["predictions_uri"] or ""),
            ran_at=float(row["ran_at"]),
            duration_ms=int(row["duration_ms"] or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "eval_id": self.eval_id,
            "range_id": self.range_id,
            "snapshot_id": self.snapshot_id,
            "golden_id": self.golden_id,
            "drift_id": self.drift_id,
            "metrics": dict(self.metrics),
            "predictions_uri": self.predictions_uri,
            "ran_at": self.ran_at,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class RegressionGateRecord:
    gate_id: str
    range_id: str
    golden_id: str
    metric: str
    threshold_type: str
    threshold_value: float
    baseline_snapshot_id: str
    action: str
    created_at: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "RegressionGateRecord":
        return cls(
            gate_id=str(row["gate_id"]),
            range_id=str(row["range_id"]),
            golden_id=str(row["golden_id"] or ""),
            metric=str(row["metric"]),
            threshold_type=str(row["threshold_type"]),
            threshold_value=float(row["threshold_value"]),
            baseline_snapshot_id=str(row["baseline_snapshot_id"] or ""),
            action=str(row["action"]),
            created_at=float(row["created_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "range_id": self.range_id,
            "golden_id": self.golden_id,
            "metric": self.metric,
            "threshold_type": self.threshold_type,
            "threshold_value": self.threshold_value,
            "baseline_snapshot_id": self.baseline_snapshot_id,
            "action": self.action,
            "created_at": self.created_at,
        }


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class RangeStore:
    """Range catalog: ranges + subjects + golden sets + drift scenarios
    + evaluations + regression gates.

    Ranges never produce snapshots (that's the LineageStore's job). A range
    pulls in already-trained snapshots as subjects and records how they
    perform against sealed golden sets under optional drift scenarios.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def _validate_mode(value: str) -> str:
        v = str(value or "").strip().lower()
        if v not in VALID_RANGE_MODES:
            raise ValueError(f"mode must be one of {sorted(VALID_RANGE_MODES)}")
        return v

    @staticmethod
    def _validate_drift_kind(value: str) -> str:
        v = str(value or "").strip().lower()
        if v not in VALID_DRIFT_KINDS:
            raise ValueError(f"drift kind must be one of {sorted(VALID_DRIFT_KINDS)}")
        return v

    @staticmethod
    def _validate_threshold_type(value: str) -> str:
        v = str(value or "").strip().lower()
        if v not in VALID_THRESHOLD_TYPES:
            raise ValueError(
                f"threshold_type must be one of {sorted(VALID_THRESHOLD_TYPES)}"
            )
        return v

    @staticmethod
    def _validate_gate_action(value: str) -> str:
        v = str(value or "").strip().lower()
        if v not in VALID_GATE_ACTIONS:
            raise ValueError(f"action must be one of {sorted(VALID_GATE_ACTIONS)}")
        return v

    # ----- Ranges -----

    def create_range(
        self,
        *,
        name: str,
        sector_id: str,
        sector_path: str,
        mode: str = "single",
        description: str = "",
        config: Optional[dict[str, Any]] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> RangeRecord:
        nm = str(name or "").strip()
        sid = str(sector_id or "").strip()
        spath = str(sector_path or "").strip()
        if not nm or not sid or not spath:
            raise ValueError("name, sector_id and sector_path are required")
        mv = self._validate_mode(mode)
        range_id = f"range-{uuid.uuid4().hex[:12]}"
        now = time.time()
        tags_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO ranges (
                    range_id, name, sector_id, sector_path, description,
                    mode, config_json, tags_json, metadata_json,
                    created_at, updated_at, last_run_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    range_id, nm, sid, spath, str(description or ""),
                    mv, json.dumps(dict(config or {})),
                    json.dumps(tags_list), json.dumps(dict(metadata or {})),
                    now, now,
                ),
            )
            self._conn.commit()
        rec = self.get_range(range_id)
        assert rec is not None
        return rec

    def get_range(self, range_id: str) -> Optional[RangeRecord]:
        rid = str(range_id or "").strip()
        if not rid:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM ranges WHERE range_id = ?", (rid,)
            ).fetchone()
        return RangeRecord.from_row(row) if row is not None else None

    def list_ranges(
        self,
        *,
        sector_path: Optional[str] = None,
        include_subtree: bool = True,
    ) -> list[RangeRecord]:
        sql = "SELECT * FROM ranges WHERE 1=1"
        args: list[Any] = []
        if sector_path:
            spath = str(sector_path).strip()
            if include_subtree and spath != "/":
                sql += " AND (sector_path = ? OR sector_path LIKE ?)"
                args.extend([spath, f"{spath}/%"])
            elif include_subtree and spath == "/":
                pass
            else:
                sql += " AND sector_path = ?"
                args.append(spath)
        sql += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [RangeRecord.from_row(r) for r in rows]

    def rename_range(self, range_id: str, name: str) -> RangeRecord:
        rid = str(range_id or "").strip()
        nm = str(name or "").strip()
        if not rid or not nm:
            raise ValueError("range_id and name are required")
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE ranges SET name = ?, updated_at = ? WHERE range_id = ?",
                (nm, now, rid),
            )
            if cur.rowcount == 0:
                raise KeyError(f"range not found: {rid}")
            self._conn.commit()
        rec = self.get_range(rid)
        assert rec is not None
        return rec

    def mark_run(self, range_id: str, ran_at: Optional[float] = None) -> RangeRecord:
        rid = str(range_id or "").strip()
        if not rid:
            raise ValueError("range_id is required")
        ts = float(ran_at) if ran_at is not None else time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE ranges SET last_run_at = ?, updated_at = ? WHERE range_id = ?",
                (ts, ts, rid),
            )
            if cur.rowcount == 0:
                raise KeyError(f"range not found: {rid}")
            self._conn.commit()
        rec = self.get_range(rid)
        assert rec is not None
        return rec

    def delete_range(self, range_id: str) -> bool:
        rid = str(range_id or "").strip()
        if not rid:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM ranges WHERE range_id = ?", (rid,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    # ----- Subjects -----

    def attach_subject(
        self,
        *,
        range_id: str,
        snapshot_id: str,
        label: str = "",
    ) -> SubjectRecord:
        rid = str(range_id or "").strip()
        sid = str(snapshot_id or "").strip()
        if not rid or not sid:
            raise ValueError("range_id and snapshot_id are required")
        lbl = str(label or "").strip() or sid
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO range_subjects (range_id, snapshot_id, label, added_at)
                VALUES (?, ?, ?, ?)
                """,
                (rid, sid, lbl, now),
            )
            self._conn.commit()
        return SubjectRecord(range_id=rid, snapshot_id=sid, label=lbl, added_at=now)

    def remove_subject(self, range_id: str, snapshot_id: str) -> bool:
        rid = str(range_id or "").strip()
        sid = str(snapshot_id or "").strip()
        if not rid or not sid:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM range_subjects WHERE range_id = ? AND snapshot_id = ?",
                (rid, sid),
            )
            self._conn.commit()
        return cur.rowcount > 0

    def list_subjects(self, range_id: str) -> list[SubjectRecord]:
        rid = str(range_id or "").strip()
        if not rid:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM range_subjects WHERE range_id = ? "
                "ORDER BY added_at ASC",
                (rid,),
            ).fetchall()
        return [SubjectRecord.from_row(r) for r in rows]

    # ----- Golden sets -----

    def seal_golden_set(
        self,
        *,
        range_id: str,
        name: str,
        split_spec: dict[str, Any],
        storage_uri: str,
        row_count: int,
        content_sha256: str,
        description: str = "",
    ) -> GoldenSetRecord:
        rid = str(range_id or "").strip()
        nm = str(name or "").strip()
        uri = str(storage_uri or "").strip()
        sha = str(content_sha256 or "").strip().lower()
        if not rid or not nm or not uri or not sha:
            raise ValueError(
                "range_id, name, storage_uri and content_sha256 are required"
            )
        golden_id = f"gold-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO golden_sets (
                    golden_id, range_id, name, description, split_spec_json,
                    storage_uri, row_count, content_sha256, sealed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    golden_id, rid, nm, str(description or ""),
                    json.dumps(dict(split_spec or {})),
                    uri, int(row_count), sha, now,
                ),
            )
            self._conn.commit()
        rec = self.get_golden_set(golden_id)
        assert rec is not None
        return rec

    def get_golden_set(self, golden_id: str) -> Optional[GoldenSetRecord]:
        gid = str(golden_id or "").strip()
        if not gid:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM golden_sets WHERE golden_id = ?", (gid,)
            ).fetchone()
        return GoldenSetRecord.from_row(row) if row is not None else None

    def list_golden_sets(self, range_id: str) -> list[GoldenSetRecord]:
        rid = str(range_id or "").strip()
        if not rid:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM golden_sets WHERE range_id = ? ORDER BY sealed_at ASC",
                (rid,),
            ).fetchall()
        return [GoldenSetRecord.from_row(r) for r in rows]

    def delete_golden_set(self, golden_id: str) -> bool:
        gid = str(golden_id or "").strip()
        if not gid:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM golden_sets WHERE golden_id = ?", (gid,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    # ----- Drift scenarios -----

    def add_drift(
        self,
        *,
        range_id: str,
        name: str,
        kind: str,
        params: Optional[dict[str, Any]] = None,
    ) -> DriftScenarioRecord:
        rid = str(range_id or "").strip()
        nm = str(name or "").strip()
        if not rid or not nm:
            raise ValueError("range_id and name are required")
        kv = self._validate_drift_kind(kind)
        drift_id = f"drift-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO drift_scenarios (
                    drift_id, range_id, name, kind, params_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (drift_id, rid, nm, kv, json.dumps(dict(params or {})), now),
            )
            self._conn.commit()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM drift_scenarios WHERE drift_id = ?", (drift_id,)
            ).fetchone()
        assert row is not None
        return DriftScenarioRecord.from_row(row)

    def list_drifts(self, range_id: str) -> list[DriftScenarioRecord]:
        rid = str(range_id or "").strip()
        if not rid:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM drift_scenarios WHERE range_id = ? "
                "ORDER BY created_at ASC",
                (rid,),
            ).fetchall()
        return [DriftScenarioRecord.from_row(r) for r in rows]

    def delete_drift(self, drift_id: str) -> bool:
        did = str(drift_id or "").strip()
        if not did:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM drift_scenarios WHERE drift_id = ?", (did,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    # ----- Evaluations -----

    def record_evaluation(
        self,
        *,
        range_id: str,
        snapshot_id: str,
        golden_id: str,
        metrics: dict[str, Any],
        drift_id: Optional[str] = None,
        predictions_uri: str = "",
        ran_at: Optional[float] = None,
        duration_ms: int = 0,
        upsert: bool = True,
    ) -> EvaluationRecord:
        rid = str(range_id or "").strip()
        sid = str(snapshot_id or "").strip()
        gid = str(golden_id or "").strip()
        if not rid or not sid or not gid:
            raise ValueError("range_id, snapshot_id and golden_id are required")
        drift_key = (str(drift_id).strip() if drift_id else None) or None
        ts = float(ran_at) if ran_at is not None else time.time()

        with self._lock:
            existing = self._conn.execute(
                """
                SELECT eval_id FROM evaluations
                 WHERE snapshot_id = ? AND golden_id = ?
                   AND (drift_id IS ? OR drift_id = ?)
                """,
                (sid, gid, drift_key, drift_key or ""),
            ).fetchone()

            if existing is not None and upsert:
                eval_id = str(existing["eval_id"])
                self._conn.execute(
                    """
                    UPDATE evaluations SET
                        range_id = ?, metrics_json = ?, predictions_uri = ?,
                        ran_at = ?, duration_ms = ?, drift_id = ?
                     WHERE eval_id = ?
                    """,
                    (
                        rid, json.dumps(dict(metrics or {})),
                        str(predictions_uri or ""),
                        ts, int(duration_ms), drift_key, eval_id,
                    ),
                )
            elif existing is not None and not upsert:
                raise ValueError(
                    "evaluation already exists (snapshot, golden, drift) and upsert=False"
                )
            else:
                eval_id = f"eval-{uuid.uuid4().hex[:12]}"
                self._conn.execute(
                    """
                    INSERT INTO evaluations (
                        eval_id, range_id, snapshot_id, golden_id, drift_id,
                        metrics_json, predictions_uri, ran_at, duration_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        eval_id, rid, sid, gid, drift_key,
                        json.dumps(dict(metrics or {})),
                        str(predictions_uri or ""),
                        ts, int(duration_ms),
                    ),
                )
            self._conn.execute(
                "UPDATE ranges SET last_run_at = ?, updated_at = ? WHERE range_id = ?",
                (ts, ts, rid),
            )
            self._conn.commit()

        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evaluations WHERE eval_id = ?", (eval_id,)
            ).fetchone()
        assert row is not None
        return EvaluationRecord.from_row(row)

    def list_evaluations(
        self,
        *,
        range_id: Optional[str] = None,
        snapshot_id: Optional[str] = None,
        golden_id: Optional[str] = None,
        drift_id: Optional[str] = None,
    ) -> list[EvaluationRecord]:
        sql = "SELECT * FROM evaluations WHERE 1=1"
        args: list[Any] = []
        if range_id:
            sql += " AND range_id = ?"
            args.append(str(range_id))
        if snapshot_id:
            sql += " AND snapshot_id = ?"
            args.append(str(snapshot_id))
        if golden_id:
            sql += " AND golden_id = ?"
            args.append(str(golden_id))
        if drift_id is not None:
            if drift_id == "":
                sql += " AND drift_id IS NULL"
            else:
                sql += " AND drift_id = ?"
                args.append(str(drift_id))
        sql += " ORDER BY ran_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [EvaluationRecord.from_row(r) for r in rows]

    def get_evaluation(self, eval_id: str) -> Optional[EvaluationRecord]:
        eid = str(eval_id or "").strip()
        if not eid:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM evaluations WHERE eval_id = ?", (eid,)
            ).fetchone()
        return EvaluationRecord.from_row(row) if row is not None else None

    def delete_evaluation(self, eval_id: str) -> bool:
        eid = str(eval_id or "").strip()
        if not eid:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM evaluations WHERE eval_id = ?", (eid,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    # ----- Regression gates -----

    def add_gate(
        self,
        *,
        range_id: str,
        metric: str,
        threshold_type: str,
        threshold_value: float,
        golden_id: Optional[str] = None,
        baseline_snapshot_id: Optional[str] = None,
        action: str = "warn",
    ) -> RegressionGateRecord:
        rid = str(range_id or "").strip()
        m = str(metric or "").strip()
        if not rid or not m:
            raise ValueError("range_id and metric are required")
        tt = self._validate_threshold_type(threshold_type)
        act = self._validate_gate_action(action)
        if tt == "delta_from_baseline" and not (baseline_snapshot_id or "").strip():
            raise ValueError(
                "baseline_snapshot_id is required for threshold_type 'delta_from_baseline'"
            )
        gate_id = f"gate-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO regression_gates (
                    gate_id, range_id, golden_id, metric,
                    threshold_type, threshold_value,
                    baseline_snapshot_id, action, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    gate_id, rid, (golden_id or None), m,
                    tt, float(threshold_value),
                    (baseline_snapshot_id or None), act, now,
                ),
            )
            self._conn.commit()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM regression_gates WHERE gate_id = ?", (gate_id,)
            ).fetchone()
        assert row is not None
        return RegressionGateRecord.from_row(row)

    def list_gates(self, range_id: str) -> list[RegressionGateRecord]:
        rid = str(range_id or "").strip()
        if not rid:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM regression_gates WHERE range_id = ? "
                "ORDER BY created_at ASC",
                (rid,),
            ).fetchall()
        return [RegressionGateRecord.from_row(r) for r in rows]

    def delete_gate(self, gate_id: str) -> bool:
        gid = str(gate_id or "").strip()
        if not gid:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM regression_gates WHERE gate_id = ?", (gid,)
            )
            self._conn.commit()
        return cur.rowcount > 0
