from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


VALID_ORIGINS = {"imported", "lineage", "range_fork"}
VALID_STORAGE_MODES = {"managed_copy", "reference"}

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id        TEXT PRIMARY KEY,
    lineage_id         TEXT,
    parent_snapshot_id TEXT REFERENCES snapshots(snapshot_id) ON DELETE SET NULL,
    origin             TEXT NOT NULL,
    model_type         TEXT NOT NULL,
    adapter_only       INTEGER NOT NULL DEFAULT 0,
    storage_mode       TEXT NOT NULL,
    weights_uri        TEXT NOT NULL,
    weights_sha256     TEXT NOT NULL,
    size_bytes         INTEGER NOT NULL,
    tags_json          TEXT NOT NULL DEFAULT '[]',
    metadata_json      TEXT NOT NULL DEFAULT '{}',
    created_at         REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snap_lineage ON snapshots(lineage_id);
CREATE INDEX IF NOT EXISTS idx_snap_sha     ON snapshots(weights_sha256);
CREATE INDEX IF NOT EXISTS idx_snap_origin  ON snapshots(origin);
"""


@dataclass(frozen=True)
class SnapshotRecord:
    snapshot_id: str
    lineage_id: str
    parent_snapshot_id: str
    origin: str
    model_type: str
    adapter_only: bool
    storage_mode: str
    weights_uri: str
    weights_sha256: str
    size_bytes: int
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = 0.0

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SnapshotRecord":
        def _load_list(key: str) -> list[str]:
            try:
                value = json.loads(str(row[key] or "[]"))
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

        def _load_obj(key: str) -> dict[str, Any]:
            try:
                value = json.loads(str(row[key] or "{}"))
            except Exception:
                return {}
            return value if isinstance(value, dict) else {}

        return cls(
            snapshot_id=str(row["snapshot_id"]),
            lineage_id=str(row["lineage_id"] or ""),
            parent_snapshot_id=str(row["parent_snapshot_id"] or ""),
            origin=str(row["origin"]),
            model_type=str(row["model_type"]),
            adapter_only=bool(row["adapter_only"]),
            storage_mode=str(row["storage_mode"]),
            weights_uri=str(row["weights_uri"]),
            weights_sha256=str(row["weights_sha256"]),
            size_bytes=int(row["size_bytes"] or 0),
            tags=_load_list("tags_json"),
            metadata=_load_obj("metadata_json"),
            created_at=float(row["created_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "lineage_id": self.lineage_id,
            "parent_snapshot_id": self.parent_snapshot_id,
            "origin": self.origin,
            "model_type": self.model_type,
            "adapter_only": self.adapter_only,
            "storage_mode": self.storage_mode,
            "weights_uri": self.weights_uri,
            "weights_sha256": self.weights_sha256,
            "size_bytes": self.size_bytes,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
        }


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


class SnapshotStore:
    """Registry of model snapshots shared by the Continuous Learning Catalog
    and the Range.

    Weights are either copied into a managed directory (storage_mode=
    'managed_copy') or referenced in place (storage_mode='reference').
    Dedup is based on the file's sha256.
    """

    def __init__(self, db_path: Path, weights_root: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._weights_root = Path(weights_root).resolve()
        self._weights_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def weights_root(self) -> Path:
        return self._weights_root

    @staticmethod
    def _validate_origin(origin: str) -> str:
        value = str(origin or "").strip().lower()
        if value not in VALID_ORIGINS:
            raise ValueError(
                f"origin must be one of {sorted(VALID_ORIGINS)}, got '{origin}'"
            )
        return value

    @staticmethod
    def _validate_storage_mode(mode: str) -> str:
        value = str(mode or "").strip().lower()
        if value not in VALID_STORAGE_MODES:
            raise ValueError(
                f"storage_mode must be one of {sorted(VALID_STORAGE_MODES)}, got '{mode}'"
            )
        return value

    def _managed_path_for(self, snapshot_id: str, src: Path) -> Path:
        suffix = "".join(src.suffixes) or ""
        return self._weights_root / f"{snapshot_id}{suffix}"

    def register(
        self,
        *,
        weights_path: Path,
        model_type: str,
        storage_mode: str = "managed_copy",
        lineage_id: Optional[str] = None,
        parent_snapshot_id: Optional[str] = None,
        origin: str = "imported",
        adapter_only: bool = False,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        dedup: bool = True,
    ) -> SnapshotRecord:
        src = Path(weights_path).expanduser().resolve()
        if not src.is_file():
            raise FileNotFoundError(f"weights file not found: {src}")

        mode = self._validate_storage_mode(storage_mode)
        origin_v = self._validate_origin(origin)
        model_type_v = str(model_type or "").strip()
        if not model_type_v:
            raise ValueError("model_type is required")

        sha256, size = _sha256_file(src)

        if dedup:
            existing = self.get_by_sha(sha256)
            if existing is not None:
                return existing

        snapshot_id = f"snap-{uuid.uuid4().hex[:12]}"
        if mode == "managed_copy":
            dst = self._managed_path_for(snapshot_id, src)
            shutil.copy2(src, dst)
            final_uri = str(dst)
        else:
            final_uri = str(src)

        now = time.time()
        tags_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        metadata_obj = dict(metadata or {})

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO snapshots (
                    snapshot_id, lineage_id, parent_snapshot_id, origin,
                    model_type, adapter_only, storage_mode,
                    weights_uri, weights_sha256, size_bytes,
                    tags_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    (lineage_id or None),
                    (parent_snapshot_id or None),
                    origin_v,
                    model_type_v,
                    1 if adapter_only else 0,
                    mode,
                    final_uri,
                    sha256,
                    int(size),
                    json.dumps(tags_list),
                    json.dumps(metadata_obj),
                    now,
                ),
            )
            self._conn.commit()

        record = self.get(snapshot_id)
        assert record is not None  # just inserted
        return record

    def get(self, snapshot_id: str) -> Optional[SnapshotRecord]:
        sid = str(snapshot_id or "").strip()
        if not sid:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM snapshots WHERE snapshot_id = ?", (sid,)
            ).fetchone()
        return SnapshotRecord.from_row(row) if row is not None else None

    def get_by_sha(self, sha256: str) -> Optional[SnapshotRecord]:
        value = str(sha256 or "").strip().lower()
        if not value:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM snapshots WHERE weights_sha256 = ? "
                "ORDER BY created_at ASC LIMIT 1",
                (value,),
            ).fetchone()
        return SnapshotRecord.from_row(row) if row is not None else None

    def list(
        self,
        *,
        lineage_id: Optional[str] = None,
        origin: Optional[str] = None,
        tag: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[SnapshotRecord]:
        sql = "SELECT * FROM snapshots WHERE 1=1"
        args: list[Any] = []
        if lineage_id is not None:
            if lineage_id == "":
                sql += " AND lineage_id IS NULL"
            else:
                sql += " AND lineage_id = ?"
                args.append(str(lineage_id))
        if origin is not None:
            sql += " AND origin = ?"
            args.append(self._validate_origin(origin))
        if tag:
            sql += " AND tags_json LIKE ?"
            args.append(f'%"{str(tag).strip()}"%')
        sql += " ORDER BY created_at DESC"
        if limit is not None and limit > 0:
            sql += " LIMIT ?"
            args.append(int(limit))
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [SnapshotRecord.from_row(r) for r in rows]

    def set_tags(self, snapshot_id: str, tags: list[str]) -> SnapshotRecord:
        sid = str(snapshot_id or "").strip()
        if not sid:
            raise ValueError("snapshot_id is required")
        tags_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        now = time.time()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE snapshots SET tags_json = ? WHERE snapshot_id = ?",
                (json.dumps(tags_list), sid),
            )
            if cur.rowcount == 0:
                raise KeyError(f"snapshot not found: {sid}")
            self._conn.commit()
        _ = now
        rec = self.get(sid)
        assert rec is not None
        return rec

    def update_metadata(
        self,
        snapshot_id: str,
        patch: dict[str, Any],
        *,
        replace: bool = False,
    ) -> SnapshotRecord:
        sid = str(snapshot_id or "").strip()
        if not sid:
            raise ValueError("snapshot_id is required")
        with self._lock:
            row = self._conn.execute(
                "SELECT metadata_json FROM snapshots WHERE snapshot_id = ?",
                (sid,),
            ).fetchone()
            if row is None:
                raise KeyError(f"snapshot not found: {sid}")
            if replace:
                merged = dict(patch or {})
            else:
                try:
                    current = json.loads(str(row["metadata_json"] or "{}"))
                    if not isinstance(current, dict):
                        current = {}
                except Exception:
                    current = {}
                current.update(patch or {})
                merged = current
            self._conn.execute(
                "UPDATE snapshots SET metadata_json = ? WHERE snapshot_id = ?",
                (json.dumps(merged), sid),
            )
            self._conn.commit()
        rec = self.get(sid)
        assert rec is not None
        return rec

    def attach_to_lineage(
        self,
        snapshot_id: str,
        *,
        lineage_id: str,
        parent_snapshot_id: Optional[str] = None,
    ) -> SnapshotRecord:
        sid = str(snapshot_id or "").strip()
        lid = str(lineage_id or "").strip()
        if not sid or not lid:
            raise ValueError("snapshot_id and lineage_id are required")
        with self._lock:
            cur = self._conn.execute(
                """
                UPDATE snapshots
                   SET lineage_id = ?,
                       parent_snapshot_id = ?,
                       origin = CASE WHEN origin = 'imported' THEN 'lineage' ELSE origin END
                 WHERE snapshot_id = ?
                """,
                (lid, (parent_snapshot_id or None), sid),
            )
            if cur.rowcount == 0:
                raise KeyError(f"snapshot not found: {sid}")
            self._conn.commit()
        rec = self.get(sid)
        assert rec is not None
        return rec

    def delete(self, snapshot_id: str, *, delete_weights: bool = True) -> bool:
        sid = str(snapshot_id or "").strip()
        if not sid:
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT storage_mode, weights_uri FROM snapshots WHERE snapshot_id = ?",
                (sid,),
            ).fetchone()
            if row is None:
                return False
            storage_mode = str(row["storage_mode"])
            weights_uri = str(row["weights_uri"])
            self._conn.execute(
                "DELETE FROM snapshots WHERE snapshot_id = ?", (sid,)
            )
            self._conn.commit()

        if delete_weights and storage_mode == "managed_copy":
            try:
                Path(weights_uri).unlink(missing_ok=True)
            except Exception:
                pass
        return True

    def count(self, *, lineage_id: Optional[str] = None) -> int:
        sql = "SELECT COUNT(*) FROM snapshots"
        args: tuple = ()
        if lineage_id is not None:
            if lineage_id == "":
                sql += " WHERE lineage_id IS NULL"
            else:
                sql += " WHERE lineage_id = ?"
                args = (str(lineage_id),)
        with self._lock:
            row = self._conn.execute(sql, args).fetchone()
        return int(row[0] or 0)
