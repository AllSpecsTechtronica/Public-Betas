from __future__ import annotations

import concurrent.futures
import hashlib
import json
import mimetypes
import os
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..config import ROOT_DIR


CVOPS_ARCHIVE_STATE_DIR = ROOT_DIR / "state" / "insight_local" / "cvops"
CVOPS_ARCHIVE_DB_PATH = CVOPS_ARCHIVE_STATE_DIR / "archives.db"
CVOPS_ARCHIVE_STORAGE_ROOT = CVOPS_ARCHIVE_STATE_DIR / "archive_corpora"

PROCESSABLE_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
PROCESSABLE_DOCUMENT_SUFFIXES = {".pdf", ".txt", ".md", ".docx", ".rtf"}
PROCESSABLE_AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aiff", ".aif", ".ogg"}
PROCESSABLE_SUFFIXES = (
    PROCESSABLE_IMAGE_SUFFIXES
    | PROCESSABLE_DOCUMENT_SUFFIXES
    | PROCESSABLE_AUDIO_SUFFIXES
)
RETAINED_UNSUPPORTED_SUFFIXES = {".psd", ".aup3", ".doc", ".cwk", ".ico", ".inf", ".ai", ".odt", ".mdb"}
NOISE_FILENAMES = {".ds_store", "thumbs.db"}

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS archive_corpora (
    corpus_id TEXT PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    source_root TEXT NOT NULL,
    managed_root TEXT NOT NULL,
    collection_id TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS archive_dataset_versions (
    dataset_version_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES archive_corpora(corpus_id) ON DELETE CASCADE,
    version_index INTEGER NOT NULL,
    label TEXT NOT NULL,
    source_path TEXT NOT NULL,
    raw_root TEXT NOT NULL,
    catalog_asset_id TEXT NOT NULL DEFAULT '',
    manifest_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(corpus_id, version_index)
);

CREATE TABLE IF NOT EXISTS archive_files (
    file_id TEXT PRIMARY KEY,
    dataset_version_id TEXT NOT NULL REFERENCES archive_dataset_versions(dataset_version_id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL,
    original_path TEXT NOT NULL,
    stored_path TEXT NOT NULL,
    current_path TEXT NOT NULL DEFAULT '',
    basename TEXT NOT NULL,
    extension TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    media_family TEXT NOT NULL,
    size_bytes INTEGER NOT NULL DEFAULT 0,
    checksum_sha256 TEXT NOT NULL,
    ingest_status TEXT NOT NULL,
    processable INTEGER NOT NULL DEFAULT 0,
    file_exists INTEGER NOT NULL DEFAULT 1,
    file_moved INTEGER NOT NULL DEFAULT 0,
    last_verified_at REAL,
    lost_detected_at REAL,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    UNIQUE(dataset_version_id, relative_path)
);

CREATE TABLE IF NOT EXISTS archive_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES archive_corpora(corpus_id) ON DELETE CASCADE,
    dataset_version_id TEXT NOT NULL REFERENCES archive_dataset_versions(dataset_version_id) ON DELETE CASCADE,
    parent_snapshot_id TEXT REFERENCES archive_snapshots(snapshot_id) ON DELETE SET NULL,
    phase TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS archive_runs (
    run_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES archive_corpora(corpus_id) ON DELETE CASCADE,
    dataset_version_id TEXT NOT NULL REFERENCES archive_dataset_versions(dataset_version_id) ON DELETE CASCADE,
    snapshot_id TEXT REFERENCES archive_snapshots(snapshot_id) ON DELETE SET NULL,
    phase TEXT NOT NULL,
    status TEXT NOT NULL,
    backbone_version TEXT NOT NULL,
    job_id TEXT NOT NULL,
    started_at REAL NOT NULL,
    finished_at REAL,
    metrics_json TEXT NOT NULL DEFAULT '{}',
    error TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS archive_objects (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    object_id TEXT NOT NULL,
    dataset_version_id TEXT NOT NULL REFERENCES archive_dataset_versions(dataset_version_id) ON DELETE CASCADE,
    object_key TEXT NOT NULL,
    object_type TEXT NOT NULL,
    title TEXT NOT NULL,
    assembly_method TEXT NOT NULL,
    assembly_confidence REAL NOT NULL DEFAULT 0.0,
    status TEXT NOT NULL DEFAULT 'pending',
    earliest TEXT NOT NULL DEFAULT '',
    latest TEXT NOT NULL DEFAULT '',
    era_bucket TEXT NOT NULL DEFAULT '',
    media_family TEXT NOT NULL DEFAULT '',
    content_complexity TEXT NOT NULL DEFAULT '',
    unresolved_reason TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, object_id)
);

CREATE TABLE IF NOT EXISTS archive_object_files (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    object_id TEXT NOT NULL,
    file_id TEXT NOT NULL REFERENCES archive_files(file_id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (snapshot_id, object_id, file_id),
    FOREIGN KEY (snapshot_id, object_id) REFERENCES archive_objects(snapshot_id, object_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS archive_temporal_anchors (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    anchor_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    type TEXT NOT NULL,
    earliest TEXT NOT NULL DEFAULT '',
    latest TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 0.0,
    source TEXT NOT NULL,
    is_publication_date INTEGER NOT NULL DEFAULT 0,
    raw_expression TEXT NOT NULL DEFAULT '',
    resolved INTEGER NOT NULL DEFAULT 0,
    resolution_requires TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, anchor_id),
    FOREIGN KEY (snapshot_id, object_id) REFERENCES archive_objects(snapshot_id, object_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS archive_entities (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases_json TEXT NOT NULL DEFAULT '[]',
    confidence REAL NOT NULL DEFAULT 0.0,
    known_facts_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, entity_id)
);

CREATE TABLE IF NOT EXISTS archive_entity_mentions (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    mention_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    text_span TEXT NOT NULL DEFAULT '',
    mention_text TEXT NOT NULL DEFAULT '',
    mention_confidence REAL NOT NULL DEFAULT 0.0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, mention_id),
    FOREIGN KEY (snapshot_id, entity_id) REFERENCES archive_entities(snapshot_id, entity_id) ON DELETE CASCADE,
    FOREIGN KEY (snapshot_id, object_id) REFERENCES archive_objects(snapshot_id, object_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS archive_relationships (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    relationship_id TEXT NOT NULL,
    source_entity_id TEXT NOT NULL,
    target_entity_id TEXT NOT NULL DEFAULT '',
    object_id TEXT NOT NULL,
    relationship_type TEXT NOT NULL,
    attributes_json TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, relationship_id),
    FOREIGN KEY (snapshot_id, object_id) REFERENCES archive_objects(snapshot_id, object_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS archive_semantic_clusters (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    cluster_id TEXT NOT NULL,
    label TEXT NOT NULL,
    object_ids_json TEXT NOT NULL DEFAULT '[]',
    earliest TEXT NOT NULL DEFAULT '',
    latest TEXT NOT NULL DEFAULT '',
    dominant_entities_json TEXT NOT NULL DEFAULT '[]',
    centroid_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, cluster_id)
);

CREATE TABLE IF NOT EXISTS archive_assembly_overrides (
    override_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES archive_corpora(corpus_id) ON DELETE CASCADE,
    dataset_version_id TEXT NOT NULL REFERENCES archive_dataset_versions(dataset_version_id) ON DELETE CASCADE,
    scope_key TEXT NOT NULL,
    action TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS archive_resolution_overrides (
    override_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES archive_corpora(corpus_id) ON DELETE CASCADE,
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    action TEXT NOT NULL,
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS archive_file_observations (
    observation_id TEXT PRIMARY KEY,
    file_id TEXT NOT NULL REFERENCES archive_files(file_id) ON DELETE CASCADE,
    observed_path TEXT NOT NULL DEFAULT '',
    file_exists INTEGER NOT NULL DEFAULT 0,
    file_moved INTEGER NOT NULL DEFAULT 0,
    observed_at REAL NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS archive_assertions (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    assertion_id TEXT NOT NULL,
    object_id TEXT NOT NULL,
    field TEXT NOT NULL,
    raw_extraction TEXT NOT NULL DEFAULT '',
    current_value TEXT NOT NULL DEFAULT '',
    current_confidence REAL NOT NULL DEFAULT 0.0,
    extraction_model TEXT NOT NULL DEFAULT '',
    extraction_run_id TEXT NOT NULL DEFAULT '',
    extraction_timestamp REAL,
    source_file_id TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL DEFAULT '',
    raw_region_json TEXT NOT NULL DEFAULT '{}',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, assertion_id),
    FOREIGN KEY (snapshot_id, object_id) REFERENCES archive_objects(snapshot_id, object_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS archive_assertion_edits (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    edit_id TEXT NOT NULL,
    assertion_id TEXT NOT NULL,
    previous_value TEXT NOT NULL DEFAULT '',
    new_value TEXT NOT NULL DEFAULT '',
    editor TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, edit_id),
    FOREIGN KEY (snapshot_id, assertion_id) REFERENCES archive_assertions(snapshot_id, assertion_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_archive_files_dataset ON archive_files(dataset_version_id);
CREATE INDEX IF NOT EXISTS idx_archive_snapshots_dataset ON archive_snapshots(dataset_version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_archive_objects_snapshot ON archive_objects(snapshot_id, earliest, latest);
CREATE INDEX IF NOT EXISTS idx_archive_overrides_dataset ON archive_assembly_overrides(dataset_version_id, created_at);
CREATE INDEX IF NOT EXISTS idx_archive_resolve_snapshot ON archive_resolution_overrides(snapshot_id, created_at);
CREATE INDEX IF NOT EXISTS idx_archive_file_obs_file ON archive_file_observations(file_id, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_archive_assertions_snapshot_object ON archive_assertions(snapshot_id, object_id, created_at ASC);
CREATE INDEX IF NOT EXISTS idx_archive_assertion_edits_snapshot_assertion ON archive_assertion_edits(snapshot_id, assertion_id, created_at ASC);

CREATE TABLE IF NOT EXISTS archive_proposals (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    proposal_id TEXT NOT NULL,
    proposal_type TEXT NOT NULL,                          -- entity_merge | temporal_propagation | anchor_resolution | relationship | cluster_membership
    target_kind TEXT NOT NULL,                            -- entity | temporal_anchor | relationship | object | cluster
    target_id TEXT NOT NULL DEFAULT '',                   -- node the proposal acts on or would create
    subject_id TEXT NOT NULL DEFAULT '',                  -- primary subject (e.g. source entity for a merge)
    related_id TEXT NOT NULL DEFAULT '',                  -- secondary subject (e.g. target entity for a merge)
    proposed_value_json TEXT NOT NULL DEFAULT '{}',       -- the change it would apply (date range, merged record, etc.)
    confidence REAL NOT NULL DEFAULT 0.0,
    signature TEXT NOT NULL DEFAULT '',                   -- stable semantic hash for cross-snapshot decision memory
    status TEXT NOT NULL DEFAULT 'proposed',              -- proposed | confirmed | rejected | superseded | auto_suppressed
    review_bucket TEXT NOT NULL DEFAULT '',               -- queue grouping for operator review
    generator TEXT NOT NULL DEFAULT '',                   -- model / heuristic that produced it
    generator_run_id TEXT NOT NULL DEFAULT '',
    cascade_source_proposal_id TEXT NOT NULL DEFAULT '',  -- proposal whose confirmation spawned this one
    resulting_assertion_id TEXT NOT NULL DEFAULT '',      -- append-only assertion created on confirm
    decided_at REAL,
    decided_by TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, proposal_id)
);

CREATE TABLE IF NOT EXISTS archive_proposal_evidence (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    evidence_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL,                          -- name_similarity | co_occurrence | shared_attribute | temporal_constraint | spatial_overlap | provenance
    description TEXT NOT NULL DEFAULT '',
    weight REAL NOT NULL DEFAULT 0.0,
    supporting_refs_json TEXT NOT NULL DEFAULT '{}',      -- object_ids / assertion_ids / anchor_ids / file_ids backing the claim
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, evidence_id),
    FOREIGN KEY (snapshot_id, proposal_id) REFERENCES archive_proposals(snapshot_id, proposal_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS archive_proposal_decisions (
    snapshot_id TEXT NOT NULL REFERENCES archive_snapshots(snapshot_id) ON DELETE CASCADE,
    decision_id TEXT NOT NULL,
    proposal_id TEXT NOT NULL,
    decision TEXT NOT NULL,                               -- confirm | reject | defer | undo
    decided_by TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    cascade_emitted_json TEXT NOT NULL DEFAULT '[]',      -- proposal_ids spawned by this decision
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (snapshot_id, decision_id),
    FOREIGN KEY (snapshot_id, proposal_id) REFERENCES archive_proposals(snapshot_id, proposal_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS archive_decision_memory (
    memory_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES archive_corpora(corpus_id) ON DELETE CASCADE,
    dataset_version_id TEXT NOT NULL REFERENCES archive_dataset_versions(dataset_version_id) ON DELETE CASCADE,
    proposal_type TEXT NOT NULL,
    signature TEXT NOT NULL,                              -- recomputed on rerun to re-match prior judgments
    decision TEXT NOT NULL,                               -- confirm | reject
    canonical_subject_json TEXT NOT NULL DEFAULT '{}',    -- human-readable subject description for re-matching/debug
    decided_by TEXT NOT NULL DEFAULT '',
    reason TEXT NOT NULL DEFAULT '',
    source_snapshot_id TEXT NOT NULL DEFAULT '',          -- provenance only; intentionally not a cascading FK
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_archive_proposals_review ON archive_proposals(snapshot_id, status, review_bucket, confidence DESC);
CREATE INDEX IF NOT EXISTS idx_archive_proposals_type ON archive_proposals(snapshot_id, proposal_type, status);
CREATE INDEX IF NOT EXISTS idx_archive_proposals_cascade ON archive_proposals(snapshot_id, cascade_source_proposal_id);
CREATE INDEX IF NOT EXISTS idx_archive_proposal_evidence_proposal ON archive_proposal_evidence(snapshot_id, proposal_id);
CREATE INDEX IF NOT EXISTS idx_archive_proposal_decisions_proposal ON archive_proposal_decisions(snapshot_id, proposal_id, created_at ASC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_archive_decision_memory_sig ON archive_decision_memory(dataset_version_id, proposal_type, signature);
"""


def _json_loads_dict(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _json_loads_list(raw: Any) -> list[Any]:
    try:
        value = json.loads(str(raw or "[]"))
    except Exception:
        return []
    return value if isinstance(value, list) else []


def normalize_archive_slug(name: str) -> str:
    token = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(name or "").strip())
    while "--" in token:
        token = token.replace("--", "-")
    token = token.strip("-")
    return token[:80] or f"archive-{uuid.uuid4().hex[:8]}"


def is_noise_file(path: Path) -> bool:
    name = path.name.lower()
    return (
        name in NOISE_FILENAMES
        or name.startswith("._")
        or any(part.startswith(".") and part not in {".", ".."} for part in path.parts)
    )


def classify_media_family(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in PROCESSABLE_IMAGE_SUFFIXES:
        return "image"
    if suffix in PROCESSABLE_AUDIO_SUFFIXES:
        return "audio"
    if suffix in PROCESSABLE_DOCUMENT_SUFFIXES or suffix in RETAINED_UNSUPPORTED_SUFFIXES:
        return "document"
    return "binary"


def is_processable_suffix(path: Path) -> bool:
    return path.suffix.lower() in PROCESSABLE_SUFFIXES


def file_ingest_status(path: Path) -> str:
    if is_noise_file(path):
        return "ignored_noise"
    if is_processable_suffix(path):
        return "ready"
    if path.suffix.lower() in RETAINED_UNSUPPORTED_SUFFIXES:
        return "retained_unsupported"
    return "retained_unknown"


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


def _copy_and_hash(src: Path, dest: Path, chunk_size: int = 1 << 22) -> tuple[str, int]:
    """Copy src to dest while computing SHA256 in one pass — eliminates the second read."""
    h = hashlib.sha256()
    size = 0
    with src.open("rb") as src_fh, dest.open("wb") as dst_fh:
        while True:
            chunk = src_fh.read(chunk_size)
            if not chunk:
                break
            dst_fh.write(chunk)
            h.update(chunk)
            size += len(chunk)
    shutil.copystat(str(src), str(dest))
    return h.hexdigest(), size


def summarize_file_metadata(path: Path) -> dict[str, Any]:
    mime_type = mimetypes.guess_type(str(path))[0] or ""
    return {
        "suffix": path.suffix.lower(),
        "mime_type": mime_type,
        "name": path.name,
        "stem": path.stem,
        "parent": path.parent.name,
    }


@dataclass(frozen=True)
class ArchiveCorpusRecord:
    corpus_id: str
    slug: str
    name: str
    description: str
    source_root: str
    managed_root: str
    collection_id: str
    metadata: dict[str, Any]
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ArchiveCorpusRecord":
        return cls(
            corpus_id=str(row["corpus_id"]),
            slug=str(row["slug"]),
            name=str(row["name"]),
            description=str(row["description"] or ""),
            source_root=str(row["source_root"] or ""),
            managed_root=str(row["managed_root"] or ""),
            collection_id=str(row["collection_id"] or ""),
            metadata=_json_loads_dict(row["metadata_json"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "source_root": self.source_root,
            "managed_root": self.managed_root,
            "collection_id": self.collection_id,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class ArchiveDatasetVersionRecord:
    dataset_version_id: str
    corpus_id: str
    version_index: int
    label: str
    source_path: str
    raw_root: str
    catalog_asset_id: str
    manifest: dict[str, Any]
    created_at: float
    updated_at: float

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "ArchiveDatasetVersionRecord":
        return cls(
            dataset_version_id=str(row["dataset_version_id"]),
            corpus_id=str(row["corpus_id"]),
            version_index=int(row["version_index"] or 0),
            label=str(row["label"] or ""),
            source_path=str(row["source_path"] or ""),
            raw_root=str(row["raw_root"] or ""),
            catalog_asset_id=str(row["catalog_asset_id"] or ""),
            manifest=_json_loads_dict(row["manifest_json"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_version_id": self.dataset_version_id,
            "corpus_id": self.corpus_id,
            "version_index": self.version_index,
            "label": self.label,
            "source_path": self.source_path,
            "raw_root": self.raw_root,
            "catalog_asset_id": self.catalog_asset_id,
            "manifest": dict(self.manifest),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ArchiveStore:
    def __init__(self, db_path: Path = CVOPS_ARCHIVE_DB_PATH, storage_root: Path = CVOPS_ARCHIVE_STORAGE_ROOT) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._storage_root = Path(storage_root).resolve()
        self._storage_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_schema()
        self._conn.commit()

    def _column_names(self, table_name: str) -> set[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        return {str(row["name"]) for row in rows}

    def _ensure_column(self, table_name: str, column_name: str, ddl: str) -> None:
        if column_name in self._column_names(table_name):
            return
        self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {ddl}")

    def _migrate_schema(self) -> None:
        self._ensure_column("archive_files", "current_path", "current_path TEXT NOT NULL DEFAULT ''")
        self._ensure_column("archive_files", "file_exists", "file_exists INTEGER NOT NULL DEFAULT 1")
        self._ensure_column("archive_files", "file_moved", "file_moved INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("archive_files", "last_verified_at", "last_verified_at REAL")
        self._ensure_column("archive_files", "lost_detected_at", "lost_detected_at REAL")

    @staticmethod
    def _first_existing_path(paths: Iterable[str]) -> str:
        for token in paths:
            candidate = str(token or "").strip()
            if not candidate:
                continue
            try:
                if Path(candidate).exists():
                    return candidate
            except Exception:
                continue
        return ""

    def _record_file_observation(
        self,
        *,
        file_id: str,
        observed_path: str,
        exists: bool,
        moved: bool,
        observed_at: float,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO archive_file_observations (
                observation_id, file_id, observed_path, file_exists, file_moved, observed_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"fobs-{uuid.uuid4().hex[:12]}",
                str(file_id or "").strip(),
                str(observed_path or ""),
                1 if exists else 0,
                1 if moved else 0,
                float(observed_at or time.time()),
                json.dumps(dict(metadata or {}), ensure_ascii=True, sort_keys=True),
            ),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    @property
    def storage_root(self) -> Path:
        return self._storage_root

    def _next_version_index(self, corpus_id: str) -> int:
        row = self._conn.execute(
            "SELECT COALESCE(MAX(version_index), 0) AS mx FROM archive_dataset_versions WHERE corpus_id = ?",
            (corpus_id,),
        ).fetchone()
        return int((row["mx"] or 0) if row is not None else 0) + 1

    def _unique_slug(self, preferred: str) -> str:
        base = normalize_archive_slug(preferred)
        candidate = base
        for index in range(1, 1000):
            row = self._conn.execute(
                "SELECT corpus_id FROM archive_corpora WHERE slug = ?",
                (candidate,),
            ).fetchone()
            if row is None:
                return candidate
            candidate = f"{base}-{index:02d}"
        raise ValueError(f"Could not find unique archive slug for '{preferred}'")

    def list_corpora(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM archive_corpora ORDER BY updated_at DESC, created_at DESC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = ArchiveCorpusRecord.from_row(row).to_dict()
            versions = self.list_dataset_versions(item["corpus_id"])
            item["version_count"] = len(versions)
            item["latest_dataset_version_id"] = versions[0]["dataset_version_id"] if versions else ""
            item["latest_snapshot_id"] = ""
            if versions:
                latest_snapshot = self.latest_snapshot(item["corpus_id"], versions[0]["dataset_version_id"])
                if latest_snapshot:
                    item["latest_snapshot_id"] = str(latest_snapshot.get("snapshot_id") or "")
            out.append(item)
        return out

    def get_corpus(self, corpus_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM archive_corpora WHERE corpus_id = ?",
                (str(corpus_id or "").strip(),),
            ).fetchone()
        if row is None:
            raise KeyError(corpus_id)
        payload = ArchiveCorpusRecord.from_row(row).to_dict()
        payload["versions"] = self.list_dataset_versions(payload["corpus_id"])
        return payload

    def list_dataset_versions(self, corpus_id: str) -> list[dict[str, Any]]:
        cid = str(corpus_id or "").strip()
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM archive_dataset_versions WHERE corpus_id = ? ORDER BY version_index DESC",
                (cid,),
            ).fetchall()
            counts = {
                str(r["dataset_version_id"]): (int(r["total"] or 0), int(r["proc"] or 0))
                for r in self._conn.execute(
                    """
                    SELECT af.dataset_version_id,
                           COUNT(*) AS total,
                           SUM(CASE WHEN af.processable=1 THEN 1 ELSE 0 END) AS proc
                    FROM archive_files af
                    WHERE af.dataset_version_id IN (
                        SELECT dataset_version_id FROM archive_dataset_versions WHERE corpus_id = ?
                    )
                    GROUP BY af.dataset_version_id
                    """,
                    (cid,),
                ).fetchall()
            }
        out = []
        for row in rows:
            rec = ArchiveDatasetVersionRecord.from_row(row).to_dict()
            dvid = rec["dataset_version_id"]
            fc, pc = counts.get(dvid, (0, 0))
            rec["file_count"] = fc
            rec["processable_count"] = pc
            snapshots = self.list_snapshots(rec["corpus_id"], dvid)
            rec["snapshot_count"] = len(snapshots)
            rec["latest_snapshot_id"] = snapshots[0]["snapshot_id"] if snapshots else ""
            out.append(rec)
        return out

    def get_dataset_version(self, dataset_version_id: str) -> dict[str, Any]:
        dvid = str(dataset_version_id or "").strip()
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM archive_dataset_versions WHERE dataset_version_id = ?",
                (dvid,),
            ).fetchone()
            if row is None:
                raise KeyError(dataset_version_id)
            counts = self._conn.execute(
                """
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN processable = 1 THEN 1 ELSE 0 END) AS proc_cnt,
                    SUM(size_bytes) AS total_bytes
                FROM archive_files WHERE dataset_version_id = ?
                """,
                (dvid,),
            ).fetchone()
        payload = ArchiveDatasetVersionRecord.from_row(row).to_dict()
        payload["file_count"] = int(counts["total"] or 0)
        payload["processable_count"] = int(counts["proc_cnt"] or 0)
        payload["total_size_bytes"] = int(counts["total_bytes"] or 0)
        payload["files"] = []  # full listing available via list_files()
        payload["snapshots"] = self.list_snapshots(payload["corpus_id"], payload["dataset_version_id"])
        return payload

    def set_corpus_collection(self, corpus_id: str, collection_id: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE archive_corpora SET collection_id = ?, updated_at = ? WHERE corpus_id = ?",
                (str(collection_id or "").strip(), now, str(corpus_id or "").strip()),
            )
            self._conn.commit()

    def set_dataset_version_catalog_asset(self, dataset_version_id: str, asset_id: str) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                "UPDATE archive_dataset_versions SET catalog_asset_id = ?, updated_at = ? WHERE dataset_version_id = ?",
                (str(asset_id or "").strip(), now, str(dataset_version_id or "").strip()),
            )
            self._conn.commit()

    def import_paths(
        self,
        *,
        source_paths: list[Path],
        name: str = "",
        description: str = "",
        metadata: Optional[dict[str, Any]] = None,
        corpus_id: str = "",
        progress_cb: Optional[Callable[[dict[str, Any]], None]] = None,
    ) -> dict[str, Any]:
        cleaned = [Path(p).expanduser() for p in source_paths if str(p)]
        if not cleaned:
            raise ValueError("source_paths is required")
        for path in cleaned:
            if not path.exists():
                raise FileNotFoundError(f"source path does not exist: {path}")
        now = time.time()
        existing_corpus_id = str(corpus_id or "").strip()
        is_new_corpus = not bool(existing_corpus_id)

        reserved: set[str] = set()

        def _reserve(rel_path: Path) -> Path:
            raw = rel_path.as_posix()
            if raw not in reserved:
                reserved.add(raw)
                return rel_path
            stem = rel_path.stem
            suffix = "".join(rel_path.suffixes)
            parent = rel_path.parent
            for i in range(1, 1000):
                candidate = parent / f"{stem}-{i:02d}{suffix}"
                token = candidate.as_posix()
                if token not in reserved:
                    reserved.add(token)
                    return candidate
            raise ValueError(f"could not make unique path for {rel_path}")

        # --- Phase 1: DB setup + path pre-reservation (inside lock, no disk I/O) ---
        with self._lock:
            if existing_corpus_id:
                corpus = self.get_corpus(existing_corpus_id)
                corpus_rec = ArchiveCorpusRecord(
                    corpus_id=corpus["corpus_id"],
                    slug=corpus["slug"],
                    name=corpus["name"],
                    description=corpus["description"],
                    source_root=corpus["source_root"],
                    managed_root=corpus["managed_root"],
                    collection_id=corpus.get("collection_id", ""),
                    metadata=dict(corpus.get("metadata") or {}),
                    created_at=float(corpus["created_at"]),
                    updated_at=float(corpus["updated_at"]),
                )
            else:
                preferred_name = str(name or cleaned[0].stem or cleaned[0].name or "archive").strip()
                slug = self._unique_slug(preferred_name)
                corpus_id = f"corpus-{uuid.uuid4().hex[:12]}"
                managed_root = self._storage_root / slug
                managed_root.mkdir(parents=True, exist_ok=True)
                corpus_rec = ArchiveCorpusRecord(
                    corpus_id=corpus_id,
                    slug=slug,
                    name=preferred_name,
                    description=str(description or ""),
                    source_root=str(cleaned[0].resolve()),
                    managed_root=str(managed_root),
                    collection_id="",
                    metadata=dict(metadata or {}),
                    created_at=now,
                    updated_at=now,
                )
                self._conn.execute(
                    """
                    INSERT INTO archive_corpora (
                        corpus_id, slug, name, description, source_root, managed_root,
                        collection_id, metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        corpus_rec.corpus_id,
                        corpus_rec.slug,
                        corpus_rec.name,
                        corpus_rec.description,
                        corpus_rec.source_root,
                        corpus_rec.managed_root,
                        "",
                        json.dumps(corpus_rec.metadata, ensure_ascii=True, sort_keys=True),
                        corpus_rec.created_at,
                        corpus_rec.updated_at,
                    ),
                )
            version_index = self._next_version_index(corpus_rec.corpus_id)
            label = f"v{version_index:04d}"
            dataset_version_id = f"dataset-{uuid.uuid4().hex[:12]}"
            raw_root = Path(corpus_rec.managed_root) / "datasets" / label / "raw"
            raw_root.mkdir(parents=True, exist_ok=True)

            if progress_cb:
                progress_cb({"phase": "scanning", "current": 0, "total": 0, "message": "Scanning source paths…"})

            _scan_queue: list[tuple[Path, Path]] = []
            for src in cleaned:
                src_resolved = src.resolve()
                if src_resolved.is_dir():
                    for file_path in sorted(
                        (p for p in src_resolved.rglob("*") if p.is_file()),
                        key=lambda p: p.relative_to(src_resolved).as_posix().lower(),
                    ):
                        _scan_queue.append((file_path, file_path.relative_to(src_resolved)))
                else:
                    _scan_queue.append((src_resolved, Path(src_resolved.name)))

            # pre-reserve all destination paths and pre-create dirs — serial, before workers start
            _work_items: list[tuple[Path, Path, Path]] = []
            _seen_dirs: set[Path] = set()
            for _csrc, _crel in _scan_queue:
                _target_rel = _reserve(_crel)
                _dest = raw_root / _target_rel
                _work_items.append((_csrc, _target_rel, _dest))
                _dpar = _dest.parent
                if _dpar not in _seen_dirs:
                    _dpar.mkdir(parents=True, exist_ok=True)
                    _seen_dirs.add(_dpar)

            _total_copy = len(_work_items)
            _n_workers = max(1, min(
                int(os.environ.get("CVOPS_IMPORT_WORKERS", "4")),
                _total_copy or 1,
            ))

            if progress_cb:
                progress_cb({
                    "phase": "copying",
                    "current": 0,
                    "total": _total_copy,
                    "workers": _n_workers,
                    "current_filename": "",
                    "elapsed_seconds": 0.0,
                    "eta_seconds": None,
                    "message": f"Found {_total_copy} file(s) — starting {_n_workers} parallel worker(s)…",
                })

            manifest_payload = {
                "source_paths": [str(p.resolve()) for p in cleaned],
                "file_count": _total_copy,
                "copied_at": now,
            }
            self._conn.execute(
                """
                INSERT INTO archive_dataset_versions (
                    dataset_version_id, corpus_id, version_index, label, source_path,
                    raw_root, catalog_asset_id, manifest_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, ?)
                """,
                (
                    dataset_version_id,
                    corpus_rec.corpus_id,
                    version_index,
                    label,
                    json.dumps([str(p.resolve()) for p in cleaned], ensure_ascii=True),
                    str(raw_root.resolve()),
                    json.dumps(manifest_payload, ensure_ascii=True, sort_keys=True),
                    now,
                    now,
                ),
            )
            # commit corpus + version before releasing lock so DB isn't held open during I/O
            self._conn.commit()

        # --- Phase 2: parallel copy+hash (outside lock — pure file I/O) ---
        _results: list[tuple[str, int]] = [("", 0)] * len(_work_items)
        _state: dict[str, Any] = {"done": 0}
        _progress_lock = threading.Lock()
        _t_copy = time.monotonic()

        def _do_one(args: tuple) -> None:  # type: ignore[type-arg]
            idx, src, _rel, dest = args
            checksum, size_bytes = _copy_and_hash(src, dest)
            _results[idx] = (checksum, size_bytes)
            with _progress_lock:
                _state["done"] += 1
                _n = _state["done"]
                if progress_cb:
                    _elapsed = time.monotonic() - _t_copy
                    _eta = (_elapsed / _n * (_total_copy - _n)) if _n < _total_copy else 0.0
                    progress_cb({
                        "phase": "copying",
                        "current": _n,
                        "total": _total_copy,
                        "current_filename": src.name,
                        "file_size_bytes": size_bytes,
                        "hash_prefix": checksum[:16],
                        "elapsed_seconds": _elapsed,
                        "eta_seconds": _eta,
                    })

        if _work_items:
            with concurrent.futures.ThreadPoolExecutor(max_workers=_n_workers) as _pool:
                list(_pool.map(
                    _do_one,
                    ((i, src, rel, dest) for i, (src, rel, dest) in enumerate(_work_items)),
                ))

        manifest: list[dict[str, Any]] = [
            {
                "original_path": str(src.resolve()),
                "relative_path": rel.as_posix(),
                "stored_path": str(dest.resolve()),
                "checksum": checksum,
                "size_bytes": size_bytes,
            }
            for (src, rel, dest), (checksum, size_bytes) in zip(_work_items, _results)
        ]

        # --- Phase 3: insert file records + commit (inside lock) ---
        with self._lock:
            file_rows = []
            noise_count = 0
            processable_count = 0
            if progress_cb:
                progress_cb({
                    "phase": "indexing",
                    "current": len(manifest),
                    "total": len(manifest),
                    "message": f"Writing {len(manifest)} file record(s) to database…",
                })
            for item in manifest:
                stored = Path(item["stored_path"]).resolve()
                relative_path = str(item["relative_path"])
                checksum = item["checksum"]
                size_bytes = int(item["size_bytes"])
                status = file_ingest_status(Path(relative_path))
                if status == "ignored_noise":
                    noise_count += 1
                if is_processable_suffix(stored) and status != "ignored_noise":
                    processable_count += 1
                meta = summarize_file_metadata(stored)
                file_rows.append(
                    (
                        f"file-{uuid.uuid4().hex[:12]}",
                        dataset_version_id,
                        relative_path,
                        str(item["original_path"]),
                        str(stored),
                        str(stored),
                        stored.name,
                        stored.suffix.lower(),
                        str(meta.get("mime_type") or ""),
                        classify_media_family(stored),
                        size_bytes,
                        checksum,
                        status,
                        1 if is_processable_suffix(stored) and status != "ignored_noise" else 0,
                        1,
                        0,
                        now,
                        None,
                        json.dumps(meta, ensure_ascii=True, sort_keys=True),
                        now,
                    )
                )
            self._conn.executemany(
                """
                INSERT INTO archive_files (
                    file_id, dataset_version_id, relative_path, original_path, stored_path, current_path,
                    basename, extension, mime_type, media_family, size_bytes,
                    checksum_sha256, ingest_status, processable, file_exists, file_moved, last_verified_at,
                    lost_detected_at, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                file_rows,
            )
            self._conn.execute(
                "UPDATE archive_corpora SET updated_at = ? WHERE corpus_id = ?",
                (time.time(), corpus_rec.corpus_id),
            )
            self._conn.commit()

        if progress_cb:
            progress_cb({"phase": "done", "current": len(manifest), "total": len(manifest), "message": f"Import complete — {len(manifest)} file(s) indexed"})
        payload = self.get_dataset_version(dataset_version_id)
        payload["corpus"] = self.get_corpus(corpus_rec.corpus_id)
        payload["noise_file_count"] = noise_count
        payload["processable_file_count"] = processable_count
        payload["is_new_corpus"] = is_new_corpus
        return payload

    def list_files(self, dataset_version_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM archive_files WHERE dataset_version_id = ? ORDER BY relative_path ASC",
                (str(dataset_version_id or "").strip(),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            payload = {
                "file_id": str(row["file_id"]),
                "dataset_version_id": str(row["dataset_version_id"]),
                "relative_path": str(row["relative_path"]),
                "original_path": str(row["original_path"]),
                "stored_path": str(row["stored_path"]),
                "current_path": str(row["current_path"] or ""),
                "basename": str(row["basename"]),
                "extension": str(row["extension"]),
                "mime_type": str(row["mime_type"]),
                "media_family": str(row["media_family"]),
                "size_bytes": int(row["size_bytes"] or 0),
                "checksum_sha256": str(row["checksum_sha256"]),
                "ingest_status": str(row["ingest_status"]),
                "processable": bool(row["processable"]),
                "exists": bool(row["file_exists"]),
                "moved": bool(row["file_moved"]),
                "last_verified_at": float(row["last_verified_at"] or 0.0),
                "lost_detected_at": float(row["lost_detected_at"] or 0.0),
                "metadata": _json_loads_dict(row["metadata_json"]),
                "created_at": float(row["created_at"]),
            }
            out.append(payload)
        return out

    def refresh_file_states(
        self,
        dataset_version_id: str,
        *,
        file_ids: Optional[Iterable[str]] = None,
        force: bool = False,
    ) -> dict[str, dict[str, Any]]:
        wanted = {str(item).strip() for item in (file_ids or []) if str(item).strip()}
        now = time.time()
        changed = False
        out: dict[str, dict[str, Any]] = {}
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM archive_files WHERE dataset_version_id = ? ORDER BY relative_path ASC",
                (str(dataset_version_id or "").strip(),),
            ).fetchall()
            for row in rows:
                file_id = str(row["file_id"])
                if wanted and file_id not in wanted:
                    continue
                prior_current = str(row["current_path"] or "")
                stored_path = str(row["stored_path"] or "")
                original_path = str(row["original_path"] or "")
                prior_exists = bool(row["file_exists"])
                prior_moved = bool(row["file_moved"])
                prior_lost = float(row["lost_detected_at"] or 0.0)
                last_verified = float(row["last_verified_at"] or 0.0)
                if not force and last_verified and now - last_verified < 5.0:
                    out[file_id] = {
                        "current_path": prior_current,
                        "exists": prior_exists,
                        "moved": prior_moved,
                        "last_verified_at": last_verified,
                        "lost_detected_at": prior_lost,
                    }
                    continue

                observed_path = self._first_existing_path((prior_current, stored_path, original_path))
                exists = bool(observed_path)
                current_path = observed_path or prior_current or stored_path or original_path
                moved = bool(exists and stored_path and current_path and Path(current_path) != Path(stored_path))
                lost_detected_at = prior_lost
                if exists:
                    lost_detected_at = 0.0
                elif not prior_lost:
                    lost_detected_at = now

                if (
                    current_path != prior_current
                    or exists != prior_exists
                    or moved != prior_moved
                    or float(lost_detected_at or 0.0) != float(prior_lost or 0.0)
                ):
                    self._conn.execute(
                        """
                        UPDATE archive_files
                           SET current_path = ?, file_exists = ?, file_moved = ?, last_verified_at = ?, lost_detected_at = ?
                         WHERE file_id = ?
                        """,
                        (
                            str(current_path or ""),
                            1 if exists else 0,
                            1 if moved else 0,
                            now,
                            float(lost_detected_at or 0.0) if lost_detected_at else None,
                            file_id,
                        ),
                    )
                    self._record_file_observation(
                        file_id=file_id,
                        observed_path=current_path,
                        exists=exists,
                        moved=moved,
                        observed_at=now,
                        metadata={"stored_path": stored_path, "original_path": original_path},
                    )
                    changed = True
                else:
                    self._conn.execute(
                        "UPDATE archive_files SET last_verified_at = ? WHERE file_id = ?",
                        (now, file_id),
                    )
                out[file_id] = {
                    "current_path": current_path,
                    "exists": exists,
                    "moved": moved,
                    "last_verified_at": now,
                    "lost_detected_at": float(lost_detected_at or 0.0),
                }
            if changed or wanted or force:
                self._conn.commit()
        return out

    def create_snapshot(
        self,
        *,
        corpus_id: str,
        dataset_version_id: str,
        phase: str,
        label: str,
        parent_snapshot_id: str = "",
        status: str = "complete",
        metrics: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        snapshot_id = f"snapshot-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO archive_snapshots (
                    snapshot_id, corpus_id, dataset_version_id, parent_snapshot_id,
                    phase, label, status, metrics_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    str(corpus_id or "").strip(),
                    str(dataset_version_id or "").strip(),
                    str(parent_snapshot_id or "").strip() or None,
                    str(phase or "").strip(),
                    str(label or phase or ""),
                    str(status or "complete"),
                    json.dumps(dict(metrics or {}), ensure_ascii=True, sort_keys=True),
                    json.dumps(dict(metadata or {}), ensure_ascii=True, sort_keys=True),
                    now,
                ),
            )
            self._conn.commit()
        return self.get_snapshot(snapshot_id)

    def list_snapshots(self, corpus_id: str, dataset_version_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM archive_snapshots
                 WHERE corpus_id = ? AND dataset_version_id = ?
                 ORDER BY created_at DESC
                """,
                (str(corpus_id or "").strip(), str(dataset_version_id or "").strip()),
            ).fetchall()
        out = []
        for row in rows:
            out.append(
                {
                    "snapshot_id": str(row["snapshot_id"]),
                    "corpus_id": str(row["corpus_id"]),
                    "dataset_version_id": str(row["dataset_version_id"]),
                    "parent_snapshot_id": str(row["parent_snapshot_id"] or ""),
                    "phase": str(row["phase"]),
                    "label": str(row["label"]),
                    "status": str(row["status"]),
                    "metrics": _json_loads_dict(row["metrics_json"]),
                    "metadata": _json_loads_dict(row["metadata_json"]),
                    "created_at": float(row["created_at"]),
                }
            )
        return out

    def latest_snapshot(self, corpus_id: str, dataset_version_id: str) -> Optional[dict[str, Any]]:
        items = self.list_snapshots(corpus_id, dataset_version_id)
        return items[0] if items else None

    def get_snapshot(self, snapshot_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM archive_snapshots WHERE snapshot_id = ?",
                (str(snapshot_id or "").strip(),),
            ).fetchone()
        if row is None:
            raise KeyError(snapshot_id)
        return {
            "snapshot_id": str(row["snapshot_id"]),
            "corpus_id": str(row["corpus_id"]),
            "dataset_version_id": str(row["dataset_version_id"]),
            "parent_snapshot_id": str(row["parent_snapshot_id"] or ""),
            "phase": str(row["phase"]),
            "label": str(row["label"]),
            "status": str(row["status"]),
            "metrics": _json_loads_dict(row["metrics_json"]),
            "metadata": _json_loads_dict(row["metadata_json"]),
            "created_at": float(row["created_at"]),
        }

    def begin_run(
        self,
        *,
        corpus_id: str,
        dataset_version_id: str,
        snapshot_id: str,
        phase: str,
        job_id: str,
        backbone_version: str,
    ) -> str:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO archive_runs (
                    run_id, corpus_id, dataset_version_id, snapshot_id, phase,
                    status, backbone_version, job_id, started_at, metrics_json, error
                ) VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, '{}', '')
                """,
                (
                    run_id,
                    corpus_id,
                    dataset_version_id,
                    str(snapshot_id or "").strip() or None,
                    phase,
                    backbone_version,
                    job_id,
                    now,
                ),
            )
            self._conn.commit()
        return run_id

    def finish_run(
        self,
        run_id: str,
        *,
        status: str,
        metrics: Optional[dict[str, Any]] = None,
        error: str = "",
    ) -> None:
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                UPDATE archive_runs
                   SET status = ?, finished_at = ?, metrics_json = ?, error = ?
                 WHERE run_id = ?
                """,
                (
                    str(status or "complete"),
                    now,
                    json.dumps(dict(metrics or {}), ensure_ascii=True, sort_keys=True),
                    str(error or ""),
                    str(run_id or "").strip(),
                ),
            )
            self._conn.commit()

    def replace_snapshot_state(self, snapshot_id: str, state: dict[str, Any]) -> None:
        snap = self.get_snapshot(snapshot_id)
        dataset_version_id = str(snap["dataset_version_id"])
        objects = [dict(item) for item in (state.get("objects") or []) if isinstance(item, dict)]
        anchors = [dict(item) for item in (state.get("anchors") or []) if isinstance(item, dict)]
        entities = [dict(item) for item in (state.get("entities") or []) if isinstance(item, dict)]
        mentions = [dict(item) for item in (state.get("mentions") or []) if isinstance(item, dict)]
        relationships = [dict(item) for item in (state.get("relationships") or []) if isinstance(item, dict)]
        clusters = [dict(item) for item in (state.get("clusters") or []) if isinstance(item, dict)]
        assertions = [dict(item) for item in (state.get("assertions") or []) if isinstance(item, dict)]
        assertion_edits = [dict(item) for item in (state.get("assertion_edits") or []) if isinstance(item, dict)]
        now = time.time()
        with self._lock:
            for table in (
                "archive_assertion_edits",
                "archive_assertions",
                "archive_semantic_clusters",
                "archive_relationships",
                "archive_entity_mentions",
                "archive_entities",
                "archive_temporal_anchors",
                "archive_object_files",
                "archive_objects",
            ):
                self._conn.execute(f"DELETE FROM {table} WHERE snapshot_id = ?", (snapshot_id,))

            for obj in objects:
                metadata = dict(obj.get("metadata") or {})
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO archive_objects (
                        snapshot_id, object_id, dataset_version_id, object_key, object_type, title,
                        assembly_method, assembly_confidence, status, earliest, latest,
                        era_bucket, media_family, content_complexity, unresolved_reason,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(obj.get("object_id") or ""),
                        dataset_version_id,
                        str(obj.get("object_key") or ""),
                        str(obj.get("object_type") or "unknown"),
                        str(obj.get("title") or ""),
                        str(obj.get("assembly_method") or "heuristic"),
                        float(obj.get("assembly_confidence") or 0.0),
                        str(obj.get("status") or "pending"),
                        str(obj.get("earliest") or ""),
                        str(obj.get("latest") or ""),
                        str(obj.get("era_bucket") or ""),
                        str(obj.get("media_family") or ""),
                        str(obj.get("content_complexity") or ""),
                        str(obj.get("unresolved_reason") or ""),
                        json.dumps(metadata, ensure_ascii=True, sort_keys=True),
                        float(obj.get("created_at") or now),
                        float(obj.get("updated_at") or now),
                    ),
                )
                for idx, file_ref in enumerate(obj.get("files") or []):
                    if not isinstance(file_ref, dict):
                        continue
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO archive_object_files (
                            snapshot_id, object_id, file_id, role, ordinal, confidence
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            str(obj.get("object_id") or ""),
                            str(file_ref.get("file_id") or ""),
                            str(file_ref.get("role") or "component"),
                            int(file_ref.get("ordinal") or idx),
                            float(file_ref.get("confidence") or 0.0),
                        ),
                    )

            for anchor in anchors:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO archive_temporal_anchors (
                        snapshot_id, anchor_id, object_id, type, earliest, latest, confidence,
                        source, is_publication_date, raw_expression, resolved,
                        resolution_requires, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(anchor.get("anchor_id") or ""),
                        str(anchor.get("object_id") or ""),
                        str(anchor.get("type") or ""),
                        str(anchor.get("earliest") or ""),
                        str(anchor.get("latest") or ""),
                        float(anchor.get("confidence") or 0.0),
                        str(anchor.get("source") or ""),
                        1 if bool(anchor.get("is_publication_date")) else 0,
                        str(anchor.get("raw_expression") or ""),
                        1 if bool(anchor.get("resolved")) else 0,
                        str(anchor.get("resolution_requires") or ""),
                        json.dumps(dict(anchor.get("metadata") or {}), ensure_ascii=True, sort_keys=True),
                        float(anchor.get("created_at") or now),
                    ),
                )

            for entity in entities:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO archive_entities (
                        snapshot_id, entity_id, canonical_name, entity_type, aliases_json,
                        confidence, known_facts_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(entity.get("entity_id") or ""),
                        str(entity.get("canonical_name") or ""),
                        str(entity.get("entity_type") or "unknown"),
                        json.dumps(list(entity.get("aliases") or []), ensure_ascii=True),
                        float(entity.get("confidence") or 0.0),
                        json.dumps(dict(entity.get("known_facts") or {}), ensure_ascii=True, sort_keys=True),
                        float(entity.get("created_at") or now),
                        float(entity.get("updated_at") or now),
                    ),
                )

            for mention in mentions:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO archive_entity_mentions (
                        snapshot_id, mention_id, entity_id, object_id, text_span, mention_text,
                        mention_confidence, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(mention.get("mention_id") or ""),
                        str(mention.get("entity_id") or ""),
                        str(mention.get("object_id") or ""),
                        str(mention.get("text_span") or ""),
                        str(mention.get("mention_text") or ""),
                        float(mention.get("mention_confidence") or 0.0),
                        json.dumps(dict(mention.get("metadata") or {}), ensure_ascii=True, sort_keys=True),
                        float(mention.get("created_at") or now),
                    ),
                )

            for rel in relationships:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO archive_relationships (
                        snapshot_id, relationship_id, source_entity_id, target_entity_id,
                        object_id, relationship_type, attributes_json, confidence, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(rel.get("relationship_id") or ""),
                        str(rel.get("source_entity_id") or ""),
                        str(rel.get("target_entity_id") or ""),
                        str(rel.get("object_id") or ""),
                        str(rel.get("relationship_type") or ""),
                        json.dumps(dict(rel.get("attributes") or {}), ensure_ascii=True, sort_keys=True),
                        float(rel.get("confidence") or 0.0),
                        float(rel.get("created_at") or now),
                    ),
                )

            for cluster in clusters:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO archive_semantic_clusters (
                        snapshot_id, cluster_id, label, object_ids_json, earliest, latest,
                        dominant_entities_json, centroid_json, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(cluster.get("cluster_id") or ""),
                        str(cluster.get("label") or ""),
                        json.dumps(list(cluster.get("object_ids") or []), ensure_ascii=True),
                        str(cluster.get("earliest") or ""),
                        str(cluster.get("latest") or ""),
                        json.dumps(list(cluster.get("dominant_entities") or []), ensure_ascii=True),
                        json.dumps(list(cluster.get("embedding_centroid") or []), ensure_ascii=True),
                        json.dumps(dict(cluster.get("metadata") or {}), ensure_ascii=True, sort_keys=True),
                        float(cluster.get("created_at") or now),
                    ),
                )

            for assertion in assertions:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO archive_assertions (
                        snapshot_id, assertion_id, object_id, field, raw_extraction, current_value,
                        current_confidence, extraction_model, extraction_run_id, extraction_timestamp,
                        source_file_id, source_type, raw_region_json, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(assertion.get("assertion_id") or ""),
                        str(assertion.get("object_id") or ""),
                        str(assertion.get("field") or ""),
                        str(assertion.get("raw_extraction") or ""),
                        str(assertion.get("current_value") or ""),
                        float(assertion.get("current_confidence") or 0.0),
                        str(assertion.get("extraction_model") or ""),
                        str(assertion.get("extraction_run_id") or assertion.get("extraction_run") or ""),
                        float(assertion.get("extraction_timestamp") or assertion.get("extraction_ts") or 0.0) or None,
                        str(assertion.get("source_file_id") or assertion.get("source_file") or ""),
                        str(assertion.get("source_type") or ""),
                        json.dumps(dict(assertion.get("raw_region") or {}), ensure_ascii=True, sort_keys=True),
                        json.dumps(dict(assertion.get("metadata") or {}), ensure_ascii=True, sort_keys=True),
                        float(assertion.get("created_at") or now),
                    ),
                )

            for edit in assertion_edits:
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO archive_assertion_edits (
                        snapshot_id, edit_id, assertion_id, previous_value, new_value,
                        editor, reason, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        snapshot_id,
                        str(edit.get("edit_id") or ""),
                        str(edit.get("assertion_id") or ""),
                        str(edit.get("previous_value") or edit.get("previous") or ""),
                        str(edit.get("new_value") or ""),
                        str(edit.get("editor") or ""),
                        str(edit.get("reason") or ""),
                        json.dumps(dict(edit.get("metadata") or {}), ensure_ascii=True, sort_keys=True),
                        float(edit.get("created_at") or edit.get("timestamp") or now),
                    ),
                )

            self._conn.commit()

    def load_snapshot_state(self, snapshot_id: str) -> dict[str, Any]:
        return {
            "snapshot": self.get_snapshot(snapshot_id),
            "objects": self.list_objects(snapshot_id),
            "anchors": self.list_anchors(snapshot_id),
            "entities": self.list_entities(snapshot_id),
            "mentions": self.list_mentions(snapshot_id),
            "relationships": self.list_relationships(snapshot_id),
            "clusters": self.list_clusters(snapshot_id),
            "assertions": self.list_assertions(snapshot_id),
            "assertion_edits": self.list_assertion_edits(snapshot_id),
        }

    def list_objects(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM archive_objects WHERE snapshot_id = ? ORDER BY COALESCE(NULLIF(earliest, ''), '9999-12-31') ASC, object_id ASC",
                (str(snapshot_id or "").strip(),),
            ).fetchall()
            file_rows = self._conn.execute(
                "SELECT * FROM archive_object_files WHERE snapshot_id = ? ORDER BY object_id ASC, ordinal ASC",
                (str(snapshot_id or "").strip(),),
            ).fetchall()
        files_by_object: dict[str, list[dict[str, Any]]] = {}
        file_lookup = {item["file_id"]: item for item in self.list_files(self.get_snapshot(snapshot_id)["dataset_version_id"])}
        for row in file_rows:
            object_id = str(row["object_id"])
            file_id = str(row["file_id"])
            entry = {
                "file_id": file_id,
                "role": str(row["role"]),
                "ordinal": int(row["ordinal"] or 0),
                "confidence": float(row["confidence"] or 0.0),
            }
            file_meta = file_lookup.get(file_id)
            if isinstance(file_meta, dict):
                entry["file"] = dict(file_meta)
            files_by_object.setdefault(object_id, []).append(entry)
        out = []
        for row in rows:
            object_id = str(row["object_id"])
            out.append(
                {
                    "snapshot_id": str(row["snapshot_id"]),
                    "object_id": object_id,
                    "dataset_version_id": str(row["dataset_version_id"]),
                    "object_key": str(row["object_key"]),
                    "object_type": str(row["object_type"]),
                    "title": str(row["title"]),
                    "assembly_method": str(row["assembly_method"]),
                    "assembly_confidence": float(row["assembly_confidence"] or 0.0),
                    "status": str(row["status"] or ""),
                    "earliest": str(row["earliest"] or ""),
                    "latest": str(row["latest"] or ""),
                    "era_bucket": str(row["era_bucket"] or ""),
                    "media_family": str(row["media_family"] or ""),
                    "content_complexity": str(row["content_complexity"] or ""),
                    "unresolved_reason": str(row["unresolved_reason"] or ""),
                    "metadata": _json_loads_dict(row["metadata_json"]),
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                    "files": files_by_object.get(object_id, []),
                }
            )
        return out

    def get_object(self, snapshot_id: str, object_id: str) -> dict[str, Any]:
        for item in self.list_objects(snapshot_id):
            if str(item.get("object_id") or "") == str(object_id or ""):
                return item
        raise KeyError(object_id)

    def list_anchors(self, snapshot_id: str, *, object_id: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_temporal_anchors WHERE snapshot_id = ?"
        args: list[Any] = [str(snapshot_id or "").strip()]
        if object_id:
            sql += " AND object_id = ?"
            args.append(str(object_id or "").strip())
        sql += " ORDER BY object_id ASC, created_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "anchor_id": str(row["anchor_id"]),
                "object_id": str(row["object_id"]),
                "type": str(row["type"]),
                "earliest": str(row["earliest"] or ""),
                "latest": str(row["latest"] or ""),
                "confidence": float(row["confidence"] or 0.0),
                "source": str(row["source"] or ""),
                "is_publication_date": bool(row["is_publication_date"]),
                "raw_expression": str(row["raw_expression"] or ""),
                "resolved": bool(row["resolved"]),
                "resolution_requires": str(row["resolution_requires"] or ""),
                "metadata": _json_loads_dict(row["metadata_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def list_entities(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM archive_entities WHERE snapshot_id = ? ORDER BY canonical_name ASC",
                (str(snapshot_id or "").strip(),),
            ).fetchall()
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "entity_id": str(row["entity_id"]),
                "canonical_name": str(row["canonical_name"]),
                "entity_type": str(row["entity_type"]),
                "aliases": _json_loads_list(row["aliases_json"]),
                "confidence": float(row["confidence"] or 0.0),
                "known_facts": _json_loads_dict(row["known_facts_json"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    def get_entity(self, snapshot_id: str, entity_id: str) -> dict[str, Any]:
        for item in self.list_entities(snapshot_id):
            if str(item.get("entity_id") or "") == str(entity_id or ""):
                return item
        raise KeyError(entity_id)

    def list_mentions(self, snapshot_id: str, *, object_id: str = "", entity_id: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_entity_mentions WHERE snapshot_id = ?"
        args: list[Any] = [str(snapshot_id or "").strip()]
        if object_id:
            sql += " AND object_id = ?"
            args.append(str(object_id or "").strip())
        if entity_id:
            sql += " AND entity_id = ?"
            args.append(str(entity_id or "").strip())
        sql += " ORDER BY object_id ASC, created_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "mention_id": str(row["mention_id"]),
                "entity_id": str(row["entity_id"]),
                "object_id": str(row["object_id"]),
                "text_span": str(row["text_span"] or ""),
                "mention_text": str(row["mention_text"] or ""),
                "mention_confidence": float(row["mention_confidence"] or 0.0),
                "metadata": _json_loads_dict(row["metadata_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def list_relationships(self, snapshot_id: str, *, object_id: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_relationships WHERE snapshot_id = ?"
        args: list[Any] = [str(snapshot_id or "").strip()]
        if object_id:
            sql += " AND object_id = ?"
            args.append(str(object_id or "").strip())
        sql += " ORDER BY object_id ASC, created_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "relationship_id": str(row["relationship_id"]),
                "source_entity_id": str(row["source_entity_id"]),
                "target_entity_id": str(row["target_entity_id"] or ""),
                "object_id": str(row["object_id"]),
                "relationship_type": str(row["relationship_type"]),
                "attributes": _json_loads_dict(row["attributes_json"]),
                "confidence": float(row["confidence"] or 0.0),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def list_clusters(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM archive_semantic_clusters WHERE snapshot_id = ? ORDER BY label ASC",
                (str(snapshot_id or "").strip(),),
            ).fetchall()
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "cluster_id": str(row["cluster_id"]),
                "label": str(row["label"]),
                "object_ids": _json_loads_list(row["object_ids_json"]),
                "earliest": str(row["earliest"] or ""),
                "latest": str(row["latest"] or ""),
                "dominant_entities": _json_loads_list(row["dominant_entities_json"]),
                "embedding_centroid": _json_loads_list(row["centroid_json"]),
                "metadata": _json_loads_dict(row["metadata_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def list_assertion_edits(self, snapshot_id: str, *, assertion_id: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_assertion_edits WHERE snapshot_id = ?"
        args: list[Any] = [str(snapshot_id or "").strip()]
        if assertion_id:
            sql += " AND assertion_id = ?"
            args.append(str(assertion_id or "").strip())
        sql += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "edit_id": str(row["edit_id"]),
                "assertion_id": str(row["assertion_id"]),
                "previous_value": str(row["previous_value"] or ""),
                "new_value": str(row["new_value"] or ""),
                "editor": str(row["editor"] or ""),
                "reason": str(row["reason"] or ""),
                "metadata": _json_loads_dict(row["metadata_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def list_assertions(self, snapshot_id: str, *, object_id: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_assertions WHERE snapshot_id = ?"
        args: list[Any] = [str(snapshot_id or "").strip()]
        if object_id:
            sql += " AND object_id = ?"
            args.append(str(object_id or "").strip())
        sql += " ORDER BY created_at ASC, assertion_id ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        edits_by_assertion: dict[str, list[dict[str, Any]]] = {}
        for edit in self.list_assertion_edits(snapshot_id):
            edits_by_assertion.setdefault(str(edit.get("assertion_id") or ""), []).append(edit)
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "assertion_id": str(row["assertion_id"]),
                "object_id": str(row["object_id"]),
                "field": str(row["field"] or ""),
                "raw_extraction": str(row["raw_extraction"] or ""),
                "current_value": str(row["current_value"] or ""),
                "current_confidence": float(row["current_confidence"] or 0.0),
                "extraction_model": str(row["extraction_model"] or ""),
                "extraction_run_id": str(row["extraction_run_id"] or ""),
                "extraction_run": str(row["extraction_run_id"] or ""),
                "extraction_timestamp": float(row["extraction_timestamp"] or 0.0),
                "extraction_ts": float(row["extraction_timestamp"] or 0.0),
                "source_file_id": str(row["source_file_id"] or ""),
                "source_file": str(row["source_file_id"] or ""),
                "source_type": str(row["source_type"] or ""),
                "raw_region": _json_loads_dict(row["raw_region_json"]),
                "metadata": _json_loads_dict(row["metadata_json"]),
                "created_at": float(row["created_at"]),
                "edits": edits_by_assertion.get(str(row["assertion_id"]), []),
            }
            for row in rows
        ]

    def add_assembly_override(
        self,
        *,
        corpus_id: str,
        dataset_version_id: str,
        scope_key: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        override_id = f"asm-{uuid.uuid4().hex[:12]}"
        now = time.time()
        record = {
            "override_id": override_id,
            "corpus_id": str(corpus_id or "").strip(),
            "dataset_version_id": str(dataset_version_id or "").strip(),
            "scope_key": str(scope_key or "").strip(),
            "action": str(action or "").strip(),
            "payload": dict(payload or {}),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO archive_assembly_overrides (
                    override_id, corpus_id, dataset_version_id, scope_key, action,
                    payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["override_id"],
                    record["corpus_id"],
                    record["dataset_version_id"],
                    record["scope_key"],
                    record["action"],
                    json.dumps(record["payload"], ensure_ascii=True, sort_keys=True),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return record

    def list_assembly_overrides(self, dataset_version_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM archive_assembly_overrides
                 WHERE dataset_version_id = ?
                 ORDER BY created_at ASC
                """,
                (str(dataset_version_id or "").strip(),),
            ).fetchall()
        return [
            {
                "override_id": str(row["override_id"]),
                "corpus_id": str(row["corpus_id"]),
                "dataset_version_id": str(row["dataset_version_id"]),
                "scope_key": str(row["scope_key"]),
                "action": str(row["action"]),
                "payload": _json_loads_dict(row["payload_json"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    def add_resolution_override(
        self,
        *,
        corpus_id: str,
        snapshot_id: str,
        target_type: str,
        target_id: str,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        override_id = f"res-{uuid.uuid4().hex[:12]}"
        now = time.time()
        record = {
            "override_id": override_id,
            "corpus_id": str(corpus_id or "").strip(),
            "snapshot_id": str(snapshot_id or "").strip(),
            "target_type": str(target_type or "").strip(),
            "target_id": str(target_id or "").strip(),
            "action": str(action or "").strip(),
            "payload": dict(payload or {}),
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO archive_resolution_overrides (
                    override_id, corpus_id, snapshot_id, target_type, target_id,
                    action, payload_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["override_id"],
                    record["corpus_id"],
                    record["snapshot_id"],
                    record["target_type"],
                    record["target_id"],
                    record["action"],
                    json.dumps(record["payload"], ensure_ascii=True, sort_keys=True),
                    now,
                    now,
                ),
            )
            self._conn.commit()
        return record

    def list_resolution_overrides(self, snapshot_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM archive_resolution_overrides
                 WHERE snapshot_id = ?
                 ORDER BY created_at ASC
                """,
                (str(snapshot_id or "").strip(),),
            ).fetchall()
        return [
            {
                "override_id": str(row["override_id"]),
                "corpus_id": str(row["corpus_id"]),
                "snapshot_id": str(row["snapshot_id"]),
                "target_type": str(row["target_type"]),
                "target_id": str(row["target_id"]),
                "action": str(row["action"]),
                "payload": _json_loads_dict(row["payload_json"]),
                "created_at": float(row["created_at"]),
                "updated_at": float(row["updated_at"]),
            }
            for row in rows
        ]

    # --- Phase 5 propose-then-confirm -------------------------------------

    def _insert_proposal_rows(
        self,
        snapshot_id: str,
        proposals: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        now: float,
    ) -> None:
        for proposal in proposals:
            decided_at = proposal.get("decided_at")
            self._conn.execute(
                """
                INSERT OR REPLACE INTO archive_proposals (
                    snapshot_id, proposal_id, proposal_type, target_kind, target_id,
                    subject_id, related_id, proposed_value_json, confidence, signature,
                    status, review_bucket, generator, generator_run_id,
                    cascade_source_proposal_id, resulting_assertion_id, decided_at,
                    decided_by, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    str(proposal.get("proposal_id") or ""),
                    str(proposal.get("proposal_type") or ""),
                    str(proposal.get("target_kind") or ""),
                    str(proposal.get("target_id") or ""),
                    str(proposal.get("subject_id") or ""),
                    str(proposal.get("related_id") or ""),
                    json.dumps(dict(proposal.get("proposed_value") or {}), ensure_ascii=True, sort_keys=True),
                    float(proposal.get("confidence") or 0.0),
                    str(proposal.get("signature") or ""),
                    str(proposal.get("status") or "proposed"),
                    str(proposal.get("review_bucket") or ""),
                    str(proposal.get("generator") or ""),
                    str(proposal.get("generator_run_id") or ""),
                    str(proposal.get("cascade_source_proposal_id") or ""),
                    str(proposal.get("resulting_assertion_id") or ""),
                    (float(decided_at) if decided_at not in (None, "") else None),
                    str(proposal.get("decided_by") or ""),
                    json.dumps(dict(proposal.get("metadata") or {}), ensure_ascii=True, sort_keys=True),
                    float(proposal.get("created_at") or now),
                ),
            )
        for item in evidence:
            self._conn.execute(
                """
                INSERT OR REPLACE INTO archive_proposal_evidence (
                    snapshot_id, evidence_id, proposal_id, evidence_type, description,
                    weight, supporting_refs_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    str(item.get("evidence_id") or ""),
                    str(item.get("proposal_id") or ""),
                    str(item.get("evidence_type") or ""),
                    str(item.get("description") or ""),
                    float(item.get("weight") or 0.0),
                    json.dumps(dict(item.get("supporting_refs") or {}), ensure_ascii=True, sort_keys=True),
                    json.dumps(dict(item.get("metadata") or {}), ensure_ascii=True, sort_keys=True),
                    float(item.get("created_at") or now),
                ),
            )

    def replace_proposals(
        self,
        snapshot_id: str,
        proposals: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> None:
        snapshot_id = str(snapshot_id or "").strip()
        now = time.time()
        with self._lock:
            self._conn.execute("DELETE FROM archive_proposals WHERE snapshot_id = ?", (snapshot_id,))
            self._insert_proposal_rows(snapshot_id, list(proposals or []), list(evidence or []), now)
            self._conn.commit()

    def add_proposals(
        self,
        snapshot_id: str,
        proposals: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
    ) -> None:
        snapshot_id = str(snapshot_id or "").strip()
        now = time.time()
        with self._lock:
            self._insert_proposal_rows(snapshot_id, list(proposals or []), list(evidence or []), now)
            self._conn.commit()

    @staticmethod
    def _proposal_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "snapshot_id": str(row["snapshot_id"]),
            "proposal_id": str(row["proposal_id"]),
            "proposal_type": str(row["proposal_type"]),
            "target_kind": str(row["target_kind"]),
            "target_id": str(row["target_id"] or ""),
            "subject_id": str(row["subject_id"] or ""),
            "related_id": str(row["related_id"] or ""),
            "proposed_value": _json_loads_dict(row["proposed_value_json"]),
            "confidence": float(row["confidence"] or 0.0),
            "signature": str(row["signature"] or ""),
            "status": str(row["status"] or "proposed"),
            "review_bucket": str(row["review_bucket"] or ""),
            "generator": str(row["generator"] or ""),
            "generator_run_id": str(row["generator_run_id"] or ""),
            "cascade_source_proposal_id": str(row["cascade_source_proposal_id"] or ""),
            "resulting_assertion_id": str(row["resulting_assertion_id"] or ""),
            "decided_at": (float(row["decided_at"]) if row["decided_at"] is not None else None),
            "decided_by": str(row["decided_by"] or ""),
            "metadata": _json_loads_dict(row["metadata_json"]),
            "created_at": float(row["created_at"]),
        }

    def list_proposals(
        self,
        snapshot_id: str,
        *,
        status: str = "",
        review_bucket: str = "",
        proposal_type: str = "",
    ) -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_proposals WHERE snapshot_id = ?"
        args: list[Any] = [str(snapshot_id or "").strip()]
        if status:
            sql += " AND status = ?"
            args.append(str(status))
        if review_bucket:
            sql += " AND review_bucket = ?"
            args.append(str(review_bucket))
        if proposal_type:
            sql += " AND proposal_type = ?"
            args.append(str(proposal_type))
        sql += " ORDER BY confidence DESC, proposal_id ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [self._proposal_row_to_dict(row) for row in rows]

    def get_proposal(self, snapshot_id: str, proposal_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM archive_proposals WHERE snapshot_id = ? AND proposal_id = ?",
                (str(snapshot_id or "").strip(), str(proposal_id or "").strip()),
            ).fetchone()
        if row is None:
            raise KeyError(proposal_id)
        return self._proposal_row_to_dict(row)

    def list_proposal_evidence(self, snapshot_id: str, *, proposal_id: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_proposal_evidence WHERE snapshot_id = ?"
        args: list[Any] = [str(snapshot_id or "").strip()]
        if proposal_id:
            sql += " AND proposal_id = ?"
            args.append(str(proposal_id or "").strip())
        sql += " ORDER BY weight DESC, evidence_id ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "evidence_id": str(row["evidence_id"]),
                "proposal_id": str(row["proposal_id"]),
                "evidence_type": str(row["evidence_type"]),
                "description": str(row["description"] or ""),
                "weight": float(row["weight"] or 0.0),
                "supporting_refs": _json_loads_dict(row["supporting_refs_json"]),
                "metadata": _json_loads_dict(row["metadata_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    def add_assertion(self, snapshot_id: str, assertion: dict[str, Any]) -> dict[str, Any]:
        """Append a single immutable assertion row (e.g. a proposal confirmation)."""
        snapshot_id = str(snapshot_id or "").strip()
        now = time.time()
        record = dict(assertion or {})
        assertion_id = str(record.get("assertion_id") or f"assert-{uuid.uuid4().hex[:12]}")
        record["assertion_id"] = assertion_id
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO archive_assertions (
                    snapshot_id, assertion_id, object_id, field, raw_extraction, current_value,
                    current_confidence, extraction_model, extraction_run_id, extraction_timestamp,
                    source_file_id, source_type, raw_region_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    assertion_id,
                    str(record.get("object_id") or ""),
                    str(record.get("field") or ""),
                    str(record.get("raw_extraction") or ""),
                    str(record.get("current_value") or ""),
                    float(record.get("current_confidence") or 0.0),
                    str(record.get("extraction_model") or ""),
                    str(record.get("extraction_run_id") or ""),
                    float(record.get("extraction_timestamp") or 0.0) or None,
                    str(record.get("source_file_id") or ""),
                    str(record.get("source_type") or ""),
                    json.dumps(dict(record.get("raw_region") or {}), ensure_ascii=True, sort_keys=True),
                    json.dumps(dict(record.get("metadata") or {}), ensure_ascii=True, sort_keys=True),
                    float(record.get("created_at") or now),
                ),
            )
            self._conn.commit()
        return record

    def record_proposal_decision(
        self,
        snapshot_id: str,
        proposal_id: str,
        decision: str,
        *,
        decided_by: str = "",
        reason: str = "",
        cascade_emitted: Optional[list[str]] = None,
        resulting_assertion_id: str = "",
        new_status: str = "",
    ) -> dict[str, Any]:
        snapshot_id = str(snapshot_id or "").strip()
        proposal_id = str(proposal_id or "").strip()
        decision_id = f"dec-{uuid.uuid4().hex[:12]}"
        now = time.time()
        cascade = [str(item) for item in (cascade_emitted or []) if str(item)]
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO archive_proposal_decisions (
                    snapshot_id, decision_id, proposal_id, decision, decided_by, reason,
                    cascade_emitted_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_id,
                    decision_id,
                    proposal_id,
                    str(decision or ""),
                    str(decided_by or ""),
                    str(reason or ""),
                    json.dumps(cascade, ensure_ascii=True),
                    json.dumps({}, ensure_ascii=True),
                    now,
                ),
            )
            if new_status:
                self._conn.execute(
                    """
                    UPDATE archive_proposals
                       SET status = ?, decided_at = ?, decided_by = ?, resulting_assertion_id = ?
                     WHERE snapshot_id = ? AND proposal_id = ?
                    """,
                    (str(new_status), now, str(decided_by or ""), str(resulting_assertion_id or ""), snapshot_id, proposal_id),
                )
            self._conn.commit()
        return {
            "decision_id": decision_id,
            "proposal_id": proposal_id,
            "decision": str(decision or ""),
            "cascade_emitted": cascade,
            "created_at": now,
        }

    def list_proposal_decisions(self, snapshot_id: str, *, proposal_id: str = "") -> list[dict[str, Any]]:
        sql = "SELECT * FROM archive_proposal_decisions WHERE snapshot_id = ?"
        args: list[Any] = [str(snapshot_id or "").strip()]
        if proposal_id:
            sql += " AND proposal_id = ?"
            args.append(str(proposal_id or "").strip())
        sql += " ORDER BY created_at ASC"
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        return [
            {
                "snapshot_id": str(row["snapshot_id"]),
                "decision_id": str(row["decision_id"]),
                "proposal_id": str(row["proposal_id"]),
                "decision": str(row["decision"]),
                "decided_by": str(row["decided_by"] or ""),
                "reason": str(row["reason"] or ""),
                "cascade_emitted": _json_loads_list(row["cascade_emitted_json"]),
                "metadata": _json_loads_dict(row["metadata_json"]),
                "created_at": float(row["created_at"]),
            }
            for row in rows
        ]

    @staticmethod
    def _decision_memory_row_to_dict(row: Any) -> dict[str, Any]:
        return {
            "memory_id": str(row["memory_id"]),
            "corpus_id": str(row["corpus_id"]),
            "dataset_version_id": str(row["dataset_version_id"]),
            "proposal_type": str(row["proposal_type"]),
            "signature": str(row["signature"]),
            "decision": str(row["decision"]),
            "canonical_subject": _json_loads_dict(row["canonical_subject_json"]),
            "decided_by": str(row["decided_by"] or ""),
            "reason": str(row["reason"] or ""),
            "source_snapshot_id": str(row["source_snapshot_id"] or ""),
            "metadata": _json_loads_dict(row["metadata_json"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def upsert_decision_memory(
        self,
        *,
        corpus_id: str,
        dataset_version_id: str,
        proposal_type: str,
        signature: str,
        decision: str,
        canonical_subject: Optional[dict[str, Any]] = None,
        decided_by: str = "",
        reason: str = "",
        source_snapshot_id: str = "",
    ) -> dict[str, Any]:
        now = time.time()
        dvid = str(dataset_version_id or "").strip()
        ptype = str(proposal_type or "").strip()
        sig = str(signature or "").strip()
        subject_json = json.dumps(dict(canonical_subject or {}), ensure_ascii=True, sort_keys=True)
        with self._lock:
            existing = self._conn.execute(
                "SELECT memory_id FROM archive_decision_memory WHERE dataset_version_id = ? AND proposal_type = ? AND signature = ?",
                (dvid, ptype, sig),
            ).fetchone()
            if existing is not None:
                memory_id = str(existing["memory_id"])
                self._conn.execute(
                    """
                    UPDATE archive_decision_memory
                       SET decision = ?, canonical_subject_json = ?, decided_by = ?, reason = ?,
                           source_snapshot_id = ?, updated_at = ?
                     WHERE memory_id = ?
                    """,
                    (str(decision), subject_json, str(decided_by or ""), str(reason or ""), str(source_snapshot_id or ""), now, memory_id),
                )
            else:
                memory_id = f"mem-{uuid.uuid4().hex[:12]}"
                self._conn.execute(
                    """
                    INSERT INTO archive_decision_memory (
                        memory_id, corpus_id, dataset_version_id, proposal_type, signature,
                        decision, canonical_subject_json, decided_by, reason, source_snapshot_id,
                        metadata_json, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        memory_id, str(corpus_id or ""), dvid, ptype, sig, str(decision),
                        subject_json, str(decided_by or ""), str(reason or ""), str(source_snapshot_id or ""),
                        json.dumps({}, ensure_ascii=True), now, now,
                    ),
                )
            self._conn.commit()
        return {"memory_id": memory_id, "dataset_version_id": dvid, "proposal_type": ptype, "signature": sig, "decision": str(decision)}

    def delete_decision_memory(self, dataset_version_id: str, proposal_type: str, signature: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM archive_decision_memory WHERE dataset_version_id = ? AND proposal_type = ? AND signature = ?",
                (str(dataset_version_id or "").strip(), str(proposal_type or "").strip(), str(signature or "").strip()),
            )
            self._conn.commit()

    def match_decision_memory(self, dataset_version_id: str, proposal_type: str, signature: str) -> Optional[dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM archive_decision_memory WHERE dataset_version_id = ? AND proposal_type = ? AND signature = ?",
                (str(dataset_version_id or "").strip(), str(proposal_type or "").strip(), str(signature or "").strip()),
            ).fetchone()
        return self._decision_memory_row_to_dict(row) if row is not None else None

    def list_decision_memory(self, dataset_version_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM archive_decision_memory WHERE dataset_version_id = ? ORDER BY updated_at DESC",
                (str(dataset_version_id or "").strip(),),
            ).fetchall()
        return [self._decision_memory_row_to_dict(row) for row in rows]

    @staticmethod
    def _synthetic_title_assertion(obj: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(obj.get("metadata") or {})
        title_meta = dict(metadata.get("title_provenance") or {})
        files = list(obj.get("files") or [])
        first_file = files[0] if files else {}
        file_meta = dict(first_file.get("file") or {}) if isinstance(first_file, dict) else {}
        raw_title = str(
            title_meta.get("raw")
            or file_meta.get("basename")
            or obj.get("object_key")
            or obj.get("title")
            or ""
        )
        current_value = str(obj.get("title") or raw_title)
        return {
            "assertion_id": f"assert-title-{str(obj.get('object_id') or '')}",
            "object_id": str(obj.get("object_id") or ""),
            "field": "title",
            "raw_extraction": raw_title,
            "current_value": current_value,
            "current_confidence": 1.0,
            "extraction_model": str(obj.get("assembly_method") or "assembly"),
            "extraction_run": "",
            "extraction_run_id": "",
            "extraction_timestamp": float(obj.get("updated_at") or obj.get("created_at") or 0.0),
            "source_file_id": str(first_file.get("file_id") or ""),
            "source_file": str(first_file.get("file_id") or ""),
            "source_type": "assembly",
            "raw_region": {},
            "metadata": {"title_provenance": title_meta},
            "created_at": float(obj.get("created_at") or 0.0),
            "edits": [],
        }

    @staticmethod
    def _preview_payload(obj: dict[str, Any], text_blocks: list[dict[str, Any]]) -> dict[str, Any]:
        files = [dict(ref) for ref in (obj.get("files") or []) if isinstance(ref, dict)]
        first_image = next(
            (
                ref for ref in files
                if isinstance(ref.get("file"), dict)
                and str((ref.get("file") or {}).get("media_family") or "") == "image"
            ),
            None,
        )
        if isinstance(first_image, dict):
            file_meta = dict(first_image.get("file") or {})
            return {
                "kind": "image",
                "source_file_id": str(first_image.get("file_id") or ""),
                "path": str(file_meta.get("current_path") or file_meta.get("stored_path") or ""),
                "label": str(file_meta.get("relative_path") or file_meta.get("basename") or ""),
                "role": str(first_image.get("role") or ""),
                "exists": bool(file_meta.get("exists")),
            }

        first_pdf = next(
            (
                ref for ref in files
                if isinstance(ref.get("file"), dict)
                and str((ref.get("file") or {}).get("extension") or "").lower() == ".pdf"
            ),
            None,
        )
        if isinstance(first_pdf, dict):
            file_meta = dict(first_pdf.get("file") or {})
            page_indexes = sorted(
                {
                    int(block.get("page_index"))
                    for block in text_blocks
                    if isinstance(block, dict)
                    and str(block.get("source_file_id") or "") == str(first_pdf.get("file_id") or "")
                    and isinstance(block.get("page_index"), int)
                }
            )
            return {
                "kind": "pdf_page",
                "source_file_id": str(first_pdf.get("file_id") or ""),
                "path": str(file_meta.get("current_path") or file_meta.get("stored_path") or ""),
                "label": str(file_meta.get("relative_path") or file_meta.get("basename") or ""),
                "page_index": page_indexes[0] if page_indexes else 0,
                "exists": bool(file_meta.get("exists")),
            }

        first_audio = next(
            (
                ref for ref in files
                if isinstance(ref.get("file"), dict)
                and str((ref.get("file") or {}).get("media_family") or "") == "audio"
            ),
            None,
        )
        if isinstance(first_audio, dict):
            file_meta = dict(first_audio.get("file") or {})
            segments = [
                {
                    "segment_id": str(block.get("segment_id") or ""),
                    "text": str(block.get("text") or ""),
                    "start_sec": float((block.get("metadata") or {}).get("start_sec") or (block.get("raw_region") or {}).get("start_sec") or 0.0),
                    "end_sec": float((block.get("metadata") or {}).get("end_sec") or (block.get("raw_region") or {}).get("end_sec") or 0.0),
                }
                for block in text_blocks
                if isinstance(block, dict)
                and str(block.get("block_kind") or "") == "audio_transcript"
                and str(block.get("source_file_id") or "") == str(first_audio.get("file_id") or "")
            ]
            return {
                "kind": "audio",
                "source_file_id": str(first_audio.get("file_id") or ""),
                "path": str(file_meta.get("current_path") or file_meta.get("stored_path") or ""),
                "label": str(file_meta.get("relative_path") or file_meta.get("basename") or ""),
                "segment_count": len(segments),
                "segments": segments,
                "exists": bool(file_meta.get("exists")),
            }

        return {
            "kind": "none",
            "label": str(obj.get("title") or "No preview available"),
            "path": "",
        }

    def build_object_detail(self, snapshot_id: str, object_id: str) -> dict[str, Any]:
        obj = self.get_object(snapshot_id, object_id)
        dataset_version_id = str(obj.get("dataset_version_id") or self.get_snapshot(snapshot_id).get("dataset_version_id") or "")
        if dataset_version_id:
            file_ids = [str(ref.get("file_id") or "") for ref in (obj.get("files") or []) if isinstance(ref, dict)]
            self.refresh_file_states(dataset_version_id, file_ids=file_ids)
            obj = self.get_object(snapshot_id, object_id)
        anchors = self.list_anchors(snapshot_id, object_id=object_id)
        mentions = self.list_mentions(snapshot_id, object_id=object_id)
        relationships = self.list_relationships(snapshot_id, object_id=object_id)
        clusters = [
            item
            for item in self.list_clusters(snapshot_id)
            if object_id in {str(v) for v in (item.get("object_ids") or [])}
        ]
        assertions = self.list_assertions(snapshot_id, object_id=object_id)
        if not assertions:
            assertions = [self._synthetic_title_assertion(obj)]
        metadata = dict(obj.get("metadata") or {})
        text_blocks = [dict(item) for item in (metadata.get("text_blocks") or []) if isinstance(item, dict)]
        segmentation = dict(metadata.get("segmentation") or {})
        extraction_summary = dict(metadata.get("extraction_summary") or {})
        preview = self._preview_payload(obj, text_blocks)
        all_mentions = self.list_mentions(snapshot_id)
        entity_ids = {str(item.get("entity_id") or "") for item in mentions}
        object_cluster_ids = {str(item.get("cluster_id") or "") for item in clusters}
        related_entities: list[dict[str, Any]] = []
        related_clusters: list[dict[str, Any]] = []
        related_classification: list[dict[str, Any]] = []
        object_type = str(obj.get("object_type") or "").strip()
        era_bucket = str(obj.get("era_bucket") or "").strip()
        content_complexity = str(obj.get("content_complexity") or "").strip()
        for candidate in self.list_objects(snapshot_id):
            candidate_id = str(candidate.get("object_id") or "")
            if candidate_id == str(object_id or ""):
                continue
            candidate_mentions = [item for item in all_mentions if str(item.get("object_id") or "") == candidate_id]
            candidate_entity_ids = {str(item.get("entity_id") or "") for item in candidate_mentions}
            if entity_ids and candidate_entity_ids & entity_ids:
                related_entities.append(candidate)
                continue
            candidate_cluster_ids = {
                str(item.get("cluster_id") or "")
                for item in self.list_clusters(snapshot_id)
                if candidate_id in {str(v) for v in (item.get("object_ids") or [])}
            }
            if object_cluster_ids and candidate_cluster_ids & object_cluster_ids:
                related_clusters.append(candidate)
            same_type = object_type and str(candidate.get("object_type") or "").strip() == object_type
            same_era = era_bucket and str(candidate.get("era_bucket") or "").strip() == era_bucket
            same_complexity = content_complexity and str(candidate.get("content_complexity") or "").strip() == content_complexity
            if same_type or same_era or same_complexity:
                related_classification.append(candidate)
        health = {
            "file_count": len(obj.get("files") or []),
            "missing_file_count": sum(1 for ref in (obj.get("files") or []) if isinstance(ref, dict) and not bool((ref.get("file") or {}).get("exists"))),
            "assertion_count": len(assertions),
            "edited_assertion_count": sum(1 for item in assertions if list(item.get("edits") or [])),
            "resolved_anchor_count": sum(1 for item in anchors if bool(item.get("resolved"))),
        }
        return {
            "object": obj,
            "anchors": anchors,
            "mentions": mentions,
            "relationships": relationships,
            "clusters": clusters,
            "assertions": assertions,
            "preview": preview,
            "extraction_summary": extraction_summary,
            "text_blocks": text_blocks,
            "segmentation": segmentation,
            "related": {
                "by_entity": related_entities,
                "by_cluster": related_clusters,
                "by_classification": related_classification,
            },
            "health": health,
        }
