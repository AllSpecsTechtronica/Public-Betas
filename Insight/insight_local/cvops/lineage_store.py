from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


VALID_UPDATE_STRATEGIES = {"full", "head_only", "lora", "replay_mixed"}
VALID_LINEAGE_STATES = {"active", "frozen", "archived"}

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS lineages (
    lineage_id         TEXT PRIMARY KEY,
    name               TEXT NOT NULL,
    sector_id          TEXT NOT NULL,
    sector_path        TEXT NOT NULL,
    description        TEXT NOT NULL DEFAULT '',
    base_snapshot_id   TEXT NOT NULL,
    head_snapshot_id   TEXT NOT NULL,
    update_strategy    TEXT NOT NULL,
    replay_config_json TEXT NOT NULL DEFAULT '{}',
    state              TEXT NOT NULL DEFAULT 'active',
    tags_json          TEXT NOT NULL DEFAULT '[]',
    metadata_json      TEXT NOT NULL DEFAULT '{}',
    created_at         REAL NOT NULL,
    updated_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lineage_sector ON lineages(sector_path);
CREATE INDEX IF NOT EXISTS idx_lineage_state  ON lineages(state);

CREATE TABLE IF NOT EXISTS drops (
    drop_id            TEXT PRIMARY KEY,
    lineage_id         TEXT NOT NULL REFERENCES lineages(lineage_id) ON DELETE CASCADE,
    drop_index         INTEGER NOT NULL,
    snapshot_id        TEXT NOT NULL,
    parent_drop_id     TEXT REFERENCES drops(drop_id) ON DELETE SET NULL,
    source_json        TEXT NOT NULL,
    replay_json        TEXT NOT NULL DEFAULT '{}',
    training_delta_json TEXT NOT NULL DEFAULT '{}',
    sample_count       INTEGER NOT NULL DEFAULT 0,
    data_sha256        TEXT NOT NULL DEFAULT '',
    started_at         REAL NOT NULL,
    finished_at        REAL,
    duration_ms        INTEGER,
    notes              TEXT NOT NULL DEFAULT '',
    UNIQUE(lineage_id, drop_index)
);

CREATE INDEX IF NOT EXISTS idx_drops_lineage_idx ON drops(lineage_id, drop_index);
CREATE INDEX IF NOT EXISTS idx_drops_snapshot    ON drops(snapshot_id);
"""


@dataclass(frozen=True)
class LineageRecord:
    lineage_id: str
    name: str
    sector_id: str
    sector_path: str
    description: str
    base_snapshot_id: str
    head_snapshot_id: str
    update_strategy: str
    replay_config: dict[str, Any] = field(default_factory=dict)
    state: str = "active"
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0
    updated_at: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LineageRecord":
        return cls(
            lineage_id=str(row["lineage_id"]),
            name=str(row["name"]),
            sector_id=str(row["sector_id"]),
            sector_path=str(row["sector_path"]),
            description=str(row["description"] or ""),
            base_snapshot_id=str(row["base_snapshot_id"]),
            head_snapshot_id=str(row["head_snapshot_id"]),
            update_strategy=str(row["update_strategy"]),
            replay_config=_load_obj(row["replay_config_json"]),
            state=str(row["state"]),
            tags=_load_list(row["tags_json"]),
            metadata=_load_obj(row["metadata_json"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_id": self.lineage_id,
            "name": self.name,
            "sector_id": self.sector_id,
            "sector_path": self.sector_path,
            "description": self.description,
            "base_snapshot_id": self.base_snapshot_id,
            "head_snapshot_id": self.head_snapshot_id,
            "update_strategy": self.update_strategy,
            "replay_config": dict(self.replay_config),
            "state": self.state,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class DropRecord:
    drop_id: str
    lineage_id: str
    drop_index: int
    snapshot_id: str
    parent_drop_id: str
    source: dict[str, Any] = field(default_factory=dict)
    replay: dict[str, Any] = field(default_factory=dict)
    training_delta: dict[str, Any] = field(default_factory=dict)
    sample_count: int = 0
    data_sha256: str = ""
    started_at: float = 0.0
    finished_at: Optional[float] = None
    duration_ms: Optional[int] = None
    notes: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "DropRecord":
        finished = row["finished_at"]
        dur = row["duration_ms"]
        return cls(
            drop_id=str(row["drop_id"]),
            lineage_id=str(row["lineage_id"]),
            drop_index=int(row["drop_index"]),
            snapshot_id=str(row["snapshot_id"]),
            parent_drop_id=str(row["parent_drop_id"] or ""),
            source=_load_obj(row["source_json"]),
            replay=_load_obj(row["replay_json"]),
            training_delta=_load_obj(row["training_delta_json"]),
            sample_count=int(row["sample_count"] or 0),
            data_sha256=str(row["data_sha256"] or ""),
            started_at=float(row["started_at"]),
            finished_at=float(finished) if finished is not None else None,
            duration_ms=int(dur) if dur is not None else None,
            notes=str(row["notes"] or ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "drop_id": self.drop_id,
            "lineage_id": self.lineage_id,
            "drop_index": self.drop_index,
            "snapshot_id": self.snapshot_id,
            "parent_drop_id": self.parent_drop_id,
            "source": dict(self.source),
            "replay": dict(self.replay),
            "training_delta": dict(self.training_delta),
            "sample_count": self.sample_count,
            "data_sha256": self.data_sha256,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": self.duration_ms,
            "notes": self.notes,
        }


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


class LineageStore:
    """Continuous Learning Catalog: lineages (model + append-only drop chain).

    A *lineage* is a model's evolution over time: it starts from a base
    snapshot and grows by one drop at a time. Each drop feeds new data into
    the head snapshot and produces the next snapshot, which becomes the new
    head.

    This store persists lineage metadata and the ordered drop chain; the
    actual weights live in the SnapshotStore. Sector identifiers are stored
    as strings (shared sector tree lives in the catalog db) — validation
    happens at the service layer.
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
    def _validate_update_strategy(value: str) -> str:
        v = str(value or "").strip().lower()
        if v not in VALID_UPDATE_STRATEGIES:
            raise ValueError(
                f"update_strategy must be one of {sorted(VALID_UPDATE_STRATEGIES)}"
            )
        return v

    @staticmethod
    def _validate_state(value: str) -> str:
        v = str(value or "").strip().lower()
        if v not in VALID_LINEAGE_STATES:
            raise ValueError(
                f"state must be one of {sorted(VALID_LINEAGE_STATES)}"
            )
        return v

    # ----- Lineages -----

    def create_lineage(
        self,
        *,
        name: str,
        sector_id: str,
        sector_path: str,
        base_snapshot_id: str,
        update_strategy: str = "head_only",
        replay_config: Optional[dict[str, Any]] = None,
        description: str = "",
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> LineageRecord:
        name_v = str(name or "").strip()
        if not name_v:
            raise ValueError("lineage name is required")
        sid = str(sector_id or "").strip()
        spath = str(sector_path or "").strip()
        if not sid or not spath:
            raise ValueError("sector_id and sector_path are required")
        base = str(base_snapshot_id or "").strip()
        if not base:
            raise ValueError("base_snapshot_id is required")
        strategy = self._validate_update_strategy(update_strategy)
        replay = dict(replay_config or {})
        tags_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        meta = dict(metadata or {})

        lineage_id = f"line-{uuid.uuid4().hex[:12]}"
        now = time.time()
        drop_id = f"drop-{uuid.uuid4().hex[:12]}"

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO lineages (
                    lineage_id, name, sector_id, sector_path, description,
                    base_snapshot_id, head_snapshot_id, update_strategy,
                    replay_config_json, state, tags_json, metadata_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)
                """,
                (
                    lineage_id, name_v, sid, spath, str(description or ""),
                    base, base, strategy,
                    json.dumps(replay), json.dumps(tags_list),
                    json.dumps(meta),
                    now, now,
                ),
            )
            # drop_index=0 is the base; represents the starting point.
            self._conn.execute(
                """
                INSERT INTO drops (
                    drop_id, lineage_id, drop_index, snapshot_id,
                    parent_drop_id, source_json, replay_json,
                    training_delta_json, sample_count, data_sha256,
                    started_at, finished_at, duration_ms, notes
                ) VALUES (?, ?, 0, ?, NULL, ?, '{}', '{}', 0, '', ?, ?, 0, ?)
                """,
                (
                    drop_id, lineage_id, base,
                    json.dumps({"kind": "base"}),
                    now, now, "base snapshot",
                ),
            )
            self._conn.commit()

        rec = self.get_lineage(lineage_id)
        assert rec is not None
        return rec

    def get_lineage(self, lineage_id: str) -> Optional[LineageRecord]:
        lid = str(lineage_id or "").strip()
        if not lid:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM lineages WHERE lineage_id = ?", (lid,)
            ).fetchone()
        return LineageRecord.from_row(row) if row is not None else None

    def list_lineages(
        self,
        *,
        sector_path: Optional[str] = None,
        state: Optional[str] = None,
        include_subtree: bool = True,
    ) -> list[LineageRecord]:
        sql = "SELECT * FROM lineages WHERE 1=1"
        args: list[Any] = []
        if sector_path:
            spath = str(sector_path).strip()
            if include_subtree and spath != "/":
                sql += " AND (sector_path = ? OR sector_path LIKE ?)"
                args.extend([spath, f"{spath}/%"])
            elif include_subtree and spath == "/":
                pass  # all
            else:
                sql += " AND sector_path = ?"
                args.append(spath)
        if state is not None:
            sql += " AND state = ?"
            args.append(self._validate_state(state))
        sql += " ORDER BY updated_at DESC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [LineageRecord.from_row(r) for r in rows]

    def rename_lineage(self, lineage_id: str, name: str) -> LineageRecord:
        lid = str(lineage_id or "").strip()
        nm = str(name or "").strip()
        if not lid or not nm:
            raise ValueError("lineage_id and name are required")
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE lineages SET name = ?, updated_at = ? WHERE lineage_id = ?",
                (nm, now, lid),
            )
            if cur.rowcount == 0:
                raise KeyError(f"lineage not found: {lid}")
            self._conn.commit()
        rec = self.get_lineage(lid)
        assert rec is not None
        return rec

    def set_state(self, lineage_id: str, state: str) -> LineageRecord:
        lid = str(lineage_id or "").strip()
        if not lid:
            raise ValueError("lineage_id is required")
        new_state = self._validate_state(state)
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE lineages SET state = ?, updated_at = ? WHERE lineage_id = ?",
                (new_state, now, lid),
            )
            if cur.rowcount == 0:
                raise KeyError(f"lineage not found: {lid}")
            self._conn.commit()
        rec = self.get_lineage(lid)
        assert rec is not None
        return rec

    def set_replay_config(
        self,
        lineage_id: str,
        replay_config: dict[str, Any],
    ) -> LineageRecord:
        lid = str(lineage_id or "").strip()
        if not lid:
            raise ValueError("lineage_id is required")
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE lineages SET replay_config_json = ?, updated_at = ? "
                "WHERE lineage_id = ?",
                (json.dumps(dict(replay_config or {})), now, lid),
            )
            if cur.rowcount == 0:
                raise KeyError(f"lineage not found: {lid}")
            self._conn.commit()
        rec = self.get_lineage(lid)
        assert rec is not None
        return rec

    def delete_lineage(self, lineage_id: str) -> bool:
        lid = str(lineage_id or "").strip()
        if not lid:
            return False
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM lineages WHERE lineage_id = ?", (lid,)
            )
            self._conn.commit()
        return cur.rowcount > 0

    # ----- Drops -----

    def add_drop(
        self,
        *,
        lineage_id: str,
        snapshot_id: str,
        source: dict[str, Any],
        training_delta: Optional[dict[str, Any]] = None,
        sample_count: int = 0,
        data_sha256: str = "",
        replay: Optional[dict[str, Any]] = None,
        started_at: Optional[float] = None,
        finished_at: Optional[float] = None,
        notes: str = "",
    ) -> DropRecord:
        """Append a drop to a lineage. Advances head_snapshot_id to the new
        snapshot and increments drop_index.

        Caller is responsible for having produced and registered the snapshot
        via SnapshotStore before calling this method (origin='lineage').
        """
        lid = str(lineage_id or "").strip()
        sid = str(snapshot_id or "").strip()
        if not lid or not sid:
            raise ValueError("lineage_id and snapshot_id are required")

        now = time.time()
        start_ts = float(started_at) if started_at is not None else now
        finish_ts = float(finished_at) if finished_at is not None else now
        dur_ms = max(0, int((finish_ts - start_ts) * 1000))
        drop_id = f"drop-{uuid.uuid4().hex[:12]}"

        with self._lock:
            lineage_row = self._conn.execute(
                "SELECT state, head_snapshot_id FROM lineages WHERE lineage_id = ?",
                (lid,),
            ).fetchone()
            if lineage_row is None:
                raise KeyError(f"lineage not found: {lid}")
            if str(lineage_row["state"]) != "active":
                raise ValueError(
                    f"cannot add drop to lineage in state '{lineage_row['state']}'"
                )

            last = self._conn.execute(
                "SELECT drop_id, drop_index FROM drops "
                "WHERE lineage_id = ? ORDER BY drop_index DESC LIMIT 1",
                (lid,),
            ).fetchone()
            next_index = int(last["drop_index"]) + 1 if last else 1
            parent_drop_id = str(last["drop_id"]) if last else None

            self._conn.execute(
                """
                INSERT INTO drops (
                    drop_id, lineage_id, drop_index, snapshot_id,
                    parent_drop_id, source_json, replay_json,
                    training_delta_json, sample_count, data_sha256,
                    started_at, finished_at, duration_ms, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    drop_id, lid, next_index, sid, parent_drop_id,
                    json.dumps(dict(source or {})),
                    json.dumps(dict(replay or {})),
                    json.dumps(dict(training_delta or {})),
                    int(sample_count),
                    str(data_sha256 or ""),
                    start_ts, finish_ts, dur_ms, str(notes or ""),
                ),
            )
            self._conn.execute(
                "UPDATE lineages SET head_snapshot_id = ?, updated_at = ? "
                "WHERE lineage_id = ?",
                (sid, now, lid),
            )
            self._conn.commit()

        row = self._conn.execute(
            "SELECT * FROM drops WHERE drop_id = ?", (drop_id,)
        ).fetchone()
        assert row is not None
        return DropRecord.from_row(row)

    def list_drops(self, lineage_id: str) -> list[DropRecord]:
        lid = str(lineage_id or "").strip()
        if not lid:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM drops WHERE lineage_id = ? ORDER BY drop_index ASC",
                (lid,),
            ).fetchall()
        return [DropRecord.from_row(r) for r in rows]

    def get_drop(self, drop_id: str) -> Optional[DropRecord]:
        did = str(drop_id or "").strip()
        if not did:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM drops WHERE drop_id = ?", (did,)
            ).fetchone()
        return DropRecord.from_row(row) if row is not None else None

    def get_drop_at(self, lineage_id: str, drop_index: int) -> Optional[DropRecord]:
        lid = str(lineage_id or "").strip()
        if not lid:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM drops WHERE lineage_id = ? AND drop_index = ?",
                (lid, int(drop_index)),
            ).fetchone()
        return DropRecord.from_row(row) if row is not None else None

    def fork_lineage(
        self,
        *,
        source_lineage_id: str,
        at_drop_index: int,
        new_name: str,
        description: str = "",
        update_strategy: Optional[str] = None,
        replay_config: Optional[dict[str, Any]] = None,
    ) -> LineageRecord:
        """Create a new lineage whose base snapshot is the snapshot produced
        by ``source_lineage_id`` at ``at_drop_index``. The old lineage is not
        modified; the new lineage's drop_index=0 references the shared
        snapshot (no duplicate weights)."""
        src_id = str(source_lineage_id or "").strip()
        if not src_id:
            raise ValueError("source_lineage_id is required")
        parent = self.get_lineage(src_id)
        if parent is None:
            raise KeyError(f"source lineage not found: {src_id}")
        drop = self.get_drop_at(src_id, int(at_drop_index))
        if drop is None:
            raise KeyError(
                f"drop_index {at_drop_index} not found in lineage {src_id}"
            )
        strategy = update_strategy if update_strategy is not None else parent.update_strategy
        replay = replay_config if replay_config is not None else parent.replay_config

        meta = dict(parent.metadata)
        meta.setdefault("forked_from", {})
        meta["forked_from"] = {
            "lineage_id": parent.lineage_id,
            "drop_index": int(at_drop_index),
            "drop_id": drop.drop_id,
            "snapshot_id": drop.snapshot_id,
            "forked_at": time.time(),
        }

        return self.create_lineage(
            name=new_name,
            sector_id=parent.sector_id,
            sector_path=parent.sector_path,
            base_snapshot_id=drop.snapshot_id,
            update_strategy=strategy,
            replay_config=replay,
            description=description or f"Forked from {parent.name} @ drop {at_drop_index}",
            tags=list(parent.tags),
            metadata=meta,
        )

    def count_drops(self, lineage_id: str) -> int:
        lid = str(lineage_id or "").strip()
        if not lid:
            return 0
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM drops WHERE lineage_id = ?", (lid,)
            ).fetchone()
        return int(row[0] or 0)
