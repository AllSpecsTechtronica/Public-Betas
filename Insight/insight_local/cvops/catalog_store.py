from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


ROOT_SECTOR_ID = "sector-root"
ROOT_SECTOR_PATH = "/"
_SECTOR_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS sectors (
    sector_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id TEXT REFERENCES sectors(sector_id) ON DELETE CASCADE,
    path TEXT NOT NULL UNIQUE,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS collections (
    collection_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    sector_id TEXT NOT NULL REFERENCES sectors(sector_id) ON DELETE RESTRICT,
    sector_path TEXT NOT NULL,
    description TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS assets (
    asset_id TEXT PRIMARY KEY,
    collection_id TEXT REFERENCES collections(collection_id) ON DELETE SET NULL,
    name TEXT NOT NULL,
    source_type TEXT NOT NULL,
    storage_mode TEXT NOT NULL,
    sector_id TEXT NOT NULL REFERENCES sectors(sector_id) ON DELETE RESTRICT,
    sector_path TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    managed_path TEXT NOT NULL,
    status TEXT NOT NULL,
    schema_status TEXT NOT NULL,
    extraction_status TEXT NOT NULL,
    availability_status TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    tags_json TEXT NOT NULL,
    keywords_json TEXT NOT NULL,
    lineage_json TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_checked_at REAL
);

CREATE INDEX IF NOT EXISTS idx_assets_sector_path ON assets(sector_path);
CREATE INDEX IF NOT EXISTS idx_assets_source_type ON assets(source_type);
CREATE INDEX IF NOT EXISTS idx_assets_status ON assets(status);
CREATE INDEX IF NOT EXISTS idx_assets_created ON assets(created_at DESC);
"""


@dataclass(frozen=True)
class SectorRecord:
    sector_id: str
    name: str
    parent_id: str
    path: str
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "SectorRecord":
        return cls(
            sector_id=str(row["sector_id"]),
            name=str(row["name"]),
            parent_id=str(row["parent_id"] or ""),
            path=str(row["path"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sector_id": self.sector_id,
            "name": self.name,
            "parent_id": self.parent_id,
            "path": self.path,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class AssetRecord:
    asset_id: str
    collection_id: str
    name: str
    source_type: str
    storage_mode: str
    sector_id: str
    sector_path: str
    source_uri: str
    managed_path: str
    status: str
    schema_status: str
    extraction_status: str
    availability_status: str
    size_bytes: int
    tags: list[str]
    keywords: list[str]
    lineage: dict[str, Any]
    metadata: dict[str, Any]
    created_at: float
    updated_at: float
    last_checked_at: float | None

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "AssetRecord":
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
                key_norm = token.lower()
                if key_norm in seen:
                    continue
                seen.add(key_norm)
                out.append(token)
            return out

        def _load_obj(key: str) -> dict[str, Any]:
            try:
                value = json.loads(str(row[key] or "{}"))
            except Exception:
                return {}
            return value if isinstance(value, dict) else {}

        last_checked = row["last_checked_at"]
        return cls(
            asset_id=str(row["asset_id"]),
            collection_id=str(row["collection_id"] or ""),
            name=str(row["name"]),
            source_type=str(row["source_type"]),
            storage_mode=str(row["storage_mode"]),
            sector_id=str(row["sector_id"]),
            sector_path=str(row["sector_path"]),
            source_uri=str(row["source_uri"] or ""),
            managed_path=str(row["managed_path"] or ""),
            status=str(row["status"]),
            schema_status=str(row["schema_status"]),
            extraction_status=str(row["extraction_status"]),
            availability_status=str(row["availability_status"]),
            size_bytes=int(row["size_bytes"] or 0),
            tags=_load_list("tags_json"),
            keywords=_load_list("keywords_json"),
            lineage=_load_obj("lineage_json"),
            metadata=_load_obj("metadata_json"),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            last_checked_at=(float(last_checked) if last_checked is not None else None),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "collection_id": self.collection_id,
            "name": self.name,
            "source_type": self.source_type,
            "storage_mode": self.storage_mode,
            "sector_id": self.sector_id,
            "sector_path": self.sector_path,
            "source_uri": self.source_uri,
            "managed_path": self.managed_path,
            "status": self.status,
            "schema_status": self.schema_status,
            "extraction_status": self.extraction_status,
            "availability_status": self.availability_status,
            "size_bytes": self.size_bytes,
            "tags": list(self.tags),
            "keywords": list(self.keywords),
            "lineage": dict(self.lineage),
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_checked_at": self.last_checked_at,
        }


class CatalogStore:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._ensure_root_sector()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @staticmethod
    def sanitize_sector_name(name: str) -> str:
        value = str(name or "").strip()
        if not value or not _SECTOR_NAME_RE.match(value):
            raise ValueError(
                "Invalid sector name. Use letters/numbers with spaces, '-' or '_' (max 64 chars)."
            )
        return value

    @staticmethod
    def normalize_storage_mode(mode: str) -> str:
        value = str(mode or "").strip().lower()
        if value not in {"managed_copy", "reference"}:
            raise ValueError("storage_mode must be 'managed_copy' or 'reference'")
        return value

    def _ensure_root_sector(self) -> None:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT sector_id FROM sectors WHERE sector_id = ?",
                (ROOT_SECTOR_ID,),
            ).fetchone()
            if row is not None:
                return
            self._conn.execute(
                """
                INSERT INTO sectors (sector_id, name, parent_id, path, created_at, updated_at)
                VALUES (?, ?, NULL, ?, ?, ?)
                """,
                (ROOT_SECTOR_ID, "root", ROOT_SECTOR_PATH, now, now),
            )
            self._conn.commit()

    @staticmethod
    def _compose_sector_path(parent_path: str, child_name: str) -> str:
        if parent_path == ROOT_SECTOR_PATH:
            return f"/{child_name}"
        return f"{parent_path}/{child_name}"

    @staticmethod
    def _subtree_match_clause(path: str) -> tuple[str, tuple[Any, ...]]:
        if path == ROOT_SECTOR_PATH:
            return "1=1", ()
        return "(path = ? OR path LIKE ?)", (path, f"{path}/%")

    def get_sector(self, *, sector_id: str = "", sector_path: str = "") -> SectorRecord:
        sid = str(sector_id or "").strip()
        spath = str(sector_path or "").strip()
        if not sid and not spath:
            raise ValueError("sector_id or sector_path is required")
        with self._lock:
            if sid:
                row = self._conn.execute(
                    "SELECT * FROM sectors WHERE sector_id = ?",
                    (sid,),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT * FROM sectors WHERE path = ?",
                    (spath or ROOT_SECTOR_PATH,),
                ).fetchone()
        if row is None:
            raise KeyError(sid or spath)
        return SectorRecord.from_row(row)

    def create_sector(
        self,
        *,
        name: str,
        parent_id: str = ROOT_SECTOR_ID,
        parent_path: str = "",
    ) -> SectorRecord:
        clean = self.sanitize_sector_name(name)
        parent: SectorRecord
        if parent_path:
            parent = self.get_sector(sector_path=parent_path)
        else:
            parent = self.get_sector(sector_id=parent_id or ROOT_SECTOR_ID)
        new_path = self._compose_sector_path(parent.path, clean)
        now = time.time()
        sector_id = f"sector-{uuid.uuid4().hex[:12]}"
        with self._lock:
            exists = self._conn.execute(
                "SELECT sector_id FROM sectors WHERE path = ?",
                (new_path,),
            ).fetchone()
            if exists is not None:
                raise ValueError(f"Sector path already exists: {new_path}")
            self._conn.execute(
                """
                INSERT INTO sectors (sector_id, name, parent_id, path, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (sector_id, clean, parent.sector_id, new_path, now, now),
            )
            self._conn.commit()
        return self.get_sector(sector_id=sector_id)

    def rename_sector(self, sector_id: str, new_name: str) -> SectorRecord:
        target = self.get_sector(sector_id=sector_id)
        if target.sector_id == ROOT_SECTOR_ID:
            raise ValueError("Root sector cannot be renamed")
        clean = self.sanitize_sector_name(new_name)
        parent = self.get_sector(sector_id=target.parent_id)
        old_path = target.path
        new_path = self._compose_sector_path(parent.path, clean)
        if new_path == old_path:
            return target
        now = time.time()
        with self._lock:
            collision = self._conn.execute(
                "SELECT sector_id FROM sectors WHERE path = ? AND sector_id != ?",
                (new_path, target.sector_id),
            ).fetchone()
            if collision is not None:
                raise ValueError(f"Sector path already exists: {new_path}")
            self._conn.execute(
                "UPDATE sectors SET name = ?, path = ?, updated_at = ? WHERE sector_id = ?",
                (clean, new_path, now, target.sector_id),
            )
            old_prefix = f"{old_path}/"
            rows = self._conn.execute(
                "SELECT sector_id, path FROM sectors WHERE path LIKE ?",
                (f"{old_prefix}%",),
            ).fetchall()
            for row in rows:
                child_id = str(row["sector_id"])
                child_path = str(row["path"])
                suffix = child_path[len(old_path):]
                self._conn.execute(
                    "UPDATE sectors SET path = ?, updated_at = ? WHERE sector_id = ?",
                    (f"{new_path}{suffix}", now, child_id),
                )
            self._sync_asset_sector_paths(now=now)
            self._conn.commit()
        return self.get_sector(sector_id=sector_id)

    def move_sector(self, sector_id: str, new_parent_id: str) -> SectorRecord:
        target = self.get_sector(sector_id=sector_id)
        if target.sector_id == ROOT_SECTOR_ID:
            raise ValueError("Root sector cannot be moved")
        new_parent = self.get_sector(sector_id=new_parent_id or ROOT_SECTOR_ID)
        if new_parent.sector_id == target.sector_id:
            raise ValueError("Sector cannot be moved under itself")
        if new_parent.path == target.path or new_parent.path.startswith(f"{target.path}/"):
            raise ValueError("Sector cannot be moved under its own subtree")
        old_path = target.path
        new_path = self._compose_sector_path(new_parent.path, target.name)
        now = time.time()
        with self._lock:
            collision = self._conn.execute(
                "SELECT sector_id FROM sectors WHERE path = ? AND sector_id != ?",
                (new_path, target.sector_id),
            ).fetchone()
            if collision is not None:
                raise ValueError(f"Sector path already exists: {new_path}")
            self._conn.execute(
                "UPDATE sectors SET parent_id = ?, path = ?, updated_at = ? WHERE sector_id = ?",
                (new_parent.sector_id, new_path, now, target.sector_id),
            )
            old_prefix = f"{old_path}/"
            rows = self._conn.execute(
                "SELECT sector_id, path FROM sectors WHERE path LIKE ?",
                (f"{old_prefix}%",),
            ).fetchall()
            for row in rows:
                child_id = str(row["sector_id"])
                child_path = str(row["path"])
                suffix = child_path[len(old_path):]
                self._conn.execute(
                    "UPDATE sectors SET path = ?, updated_at = ? WHERE sector_id = ?",
                    (f"{new_path}{suffix}", now, child_id),
                )
            self._sync_asset_sector_paths(now=now)
            self._conn.commit()
        return self.get_sector(sector_id=sector_id)

    def _sync_asset_sector_paths(self, *, now: float) -> None:
        self._conn.execute(
            """
            UPDATE assets
               SET sector_path = (
                    SELECT path FROM sectors WHERE sectors.sector_id = assets.sector_id
               ),
                   updated_at = ?
            """,
            (now,),
        )
        self._conn.execute(
            """
            UPDATE collections
               SET sector_path = (
                    SELECT path FROM sectors WHERE sectors.sector_id = collections.sector_id
               ),
                   updated_at = ?
            """,
            (now,),
        )

    def list_sectors(self) -> list[SectorRecord]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sectors ORDER BY path ASC",
            ).fetchall()
        return [SectorRecord.from_row(row) for row in rows]

    def create_collection(
        self,
        *,
        name: str,
        source_type: str,
        sector_id: str,
        description: str = "",
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        sector = self.get_sector(sector_id=sector_id)
        now = time.time()
        coll_id = f"collection-{uuid.uuid4().hex[:12]}"
        payload = json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO collections
                    (collection_id, name, source_type, sector_id, sector_path, description, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    coll_id,
                    str(name or "").strip() or "collection",
                    str(source_type or "").strip(),
                    sector.sector_id,
                    sector.path,
                    str(description or ""),
                    payload,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_collection(coll_id)

    def list_collections(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM collections ORDER BY sector_path ASC, name ASC"
            ).fetchall()
        result = []
        for row in rows:
            try:
                metadata = json.loads(str(row["metadata_json"] or "{}"))
            except Exception:
                metadata = {}
            result.append({
                "collection_id": str(row["collection_id"]),
                "name": str(row["name"]),
                "source_type": str(row["source_type"]),
                "sector_id": str(row["sector_id"]),
                "sector_path": str(row["sector_path"]),
                "description": str(row["description"] or ""),
                "metadata": metadata if isinstance(metadata, dict) else {},
                "created_at": float(row["created_at"]),
            })
        return result

    def get_collection(self, collection_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM collections WHERE collection_id = ?",
                (str(collection_id or "").strip(),),
            ).fetchone()
        if row is None:
            raise KeyError(collection_id)
        try:
            metadata = json.loads(str(row["metadata_json"] or "{}"))
        except Exception:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        return {
            "collection_id": str(row["collection_id"]),
            "name": str(row["name"]),
            "source_type": str(row["source_type"]),
            "sector_id": str(row["sector_id"]),
            "sector_path": str(row["sector_path"]),
            "description": str(row["description"] or ""),
            "metadata": metadata,
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def create_asset(
        self,
        *,
        name: str,
        source_type: str,
        storage_mode: str,
        sector_id: str,
        source_uri: str = "",
        managed_path: str = "",
        status: str = "ingested",
        schema_status: str = "pending",
        extraction_status: str = "pending",
        availability_status: str = "unknown",
        size_bytes: int = 0,
        tags: Optional[list[str]] = None,
        keywords: Optional[list[str]] = None,
        lineage: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
        collection_id: str = "",
    ) -> AssetRecord:
        sector = self.get_sector(sector_id=sector_id)
        mode = self.normalize_storage_mode(storage_mode)
        now = time.time()
        asset_id = f"asset-{uuid.uuid4().hex[:12]}"
        norm_name = str(name or "").strip() or asset_id
        tags_norm = self._normalize_token_list(tags or [])
        keywords_norm = self._normalize_token_list(keywords or [])
        lineage_norm = dict(lineage or {})
        metadata_norm = dict(metadata or {})
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO assets (
                    asset_id, collection_id, name, source_type, storage_mode, sector_id, sector_path,
                    source_uri, managed_path, status, schema_status, extraction_status, availability_status,
                    size_bytes, tags_json, keywords_json, lineage_json, metadata_json,
                    created_at, updated_at, last_checked_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    asset_id,
                    str(collection_id or "").strip() or None,
                    norm_name,
                    str(source_type or "").strip().lower(),
                    mode,
                    sector.sector_id,
                    sector.path,
                    str(source_uri or "").strip(),
                    str(managed_path or "").strip(),
                    str(status or "ingested").strip().lower(),
                    str(schema_status or "pending").strip().lower(),
                    str(extraction_status or "pending").strip().lower(),
                    str(availability_status or "unknown").strip().lower(),
                    max(0, int(size_bytes or 0)),
                    json.dumps(tags_norm, ensure_ascii=True),
                    json.dumps(keywords_norm, ensure_ascii=True),
                    json.dumps(lineage_norm, ensure_ascii=True, sort_keys=True),
                    json.dumps(metadata_norm, ensure_ascii=True, sort_keys=True),
                    now,
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return self.get_asset(asset_id)

    @staticmethod
    def _normalize_token_list(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            token = str(item or "").strip()
            if not token:
                continue
            key = token.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(token[:128])
            if len(out) >= 256:
                break
        return out

    def get_asset(self, asset_id: str) -> AssetRecord:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM assets WHERE asset_id = ?",
                (str(asset_id or "").strip(),),
            ).fetchone()
        if row is None:
            raise KeyError(asset_id)
        return AssetRecord.from_row(row)

    def update_asset_availability(
        self,
        asset_id: str,
        *,
        availability_status: str,
        metadata_patch: Optional[dict[str, Any]] = None,
    ) -> AssetRecord:
        current = self.get_asset(asset_id)
        metadata = dict(current.metadata)
        if isinstance(metadata_patch, dict) and metadata_patch:
            metadata.update(metadata_patch)
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                UPDATE assets
                   SET availability_status = ?, metadata_json = ?, updated_at = ?, last_checked_at = ?
                 WHERE asset_id = ?
                """,
                (
                    str(availability_status or "unknown").strip().lower(),
                    json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                    now,
                    now,
                    current.asset_id,
                ),
            )
            self._conn.commit()
        return self.get_asset(current.asset_id)

    def assign_asset_sector(
        self,
        asset_id: str,
        *,
        sector_id: str = "",
        sector_path: str = "",
    ) -> AssetRecord:
        asset = self.get_asset(asset_id)
        if sector_path:
            sector = self.get_sector(sector_path=sector_path)
        else:
            sector = self.get_sector(sector_id=sector_id)
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                UPDATE assets
                   SET sector_id = ?, sector_path = ?, updated_at = ?
                 WHERE asset_id = ?
                """,
                (sector.sector_id, sector.path, now, asset.asset_id),
            )
            self._conn.commit()
        return self.get_asset(asset.asset_id)

    def search_assets(
        self,
        *,
        query: str = "",
        source_type: str = "",
        status: str = "",
        storage_mode: str = "",
        sector_path: str = "",
        include_descendants: bool = True,
        limit: int = 100,
    ) -> list[AssetRecord]:
        where: list[str] = []
        params: list[Any] = []
        if query:
            q = f"%{query.strip().lower()}%"
            where.append(
                """
                (
                    lower(name) LIKE ? OR
                    lower(source_uri) LIKE ? OR
                    lower(tags_json) LIKE ? OR
                    lower(keywords_json) LIKE ?
                )
                """
            )
            params.extend([q, q, q, q])
        if source_type:
            where.append("source_type = ?")
            params.append(str(source_type).strip().lower())
        if status:
            where.append("status = ?")
            params.append(str(status).strip().lower())
        if storage_mode:
            where.append("storage_mode = ?")
            params.append(self.normalize_storage_mode(storage_mode))
        if sector_path:
            if include_descendants:
                where.append("(sector_path = ? OR sector_path LIKE ?)")
                params.extend([sector_path, f"{sector_path}/%"])
            else:
                where.append("sector_path = ?")
                params.append(sector_path)
        sql = "SELECT * FROM assets"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, asset_id ASC LIMIT ?"
        params.append(max(1, min(int(limit or 100), 1000)))
        with self._lock:
            rows = self._conn.execute(sql, tuple(params)).fetchall()
        return [AssetRecord.from_row(row) for row in rows]

    def sector_summary(self, sector_path: str) -> dict[str, Any]:
        sector = self.get_sector(sector_path=sector_path or ROOT_SECTOR_PATH)
        if sector.path == ROOT_SECTOR_PATH:
            clause = "1=1"
            params: tuple[Any, ...] = ()
        else:
            clause = "(sector_path = ? OR sector_path LIKE ?)"
            params = (sector.path, f"{sector.path}/%")
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT source_type, status, extraction_status, availability_status, storage_mode
                  FROM assets
                 WHERE {clause}
                """,
                params,
            ).fetchall()
        total = len(rows)
        by_type: dict[str, int] = {}
        health: dict[str, int] = {}
        unprocessed = 0
        stale_refs = 0
        for row in rows:
            stype = str(row["source_type"] or "")
            by_type[stype] = int(by_type.get(stype, 0)) + 1
            avail = str(row["availability_status"] or "unknown")
            health[avail] = int(health.get(avail, 0)) + 1
            extract_status = str(row["extraction_status"] or "")
            if extract_status != "complete":
                unprocessed += 1
            mode = str(row["storage_mode"] or "")
            if mode == "reference" and avail in {"missing", "stale"}:
                stale_refs += 1
        return {
            "sector": sector.to_dict(),
            "assets_total": total,
            "counts_by_type": by_type,
            "ingest_health": health,
            "unprocessed_assets": unprocessed,
            "stale_references": stale_refs,
        }
