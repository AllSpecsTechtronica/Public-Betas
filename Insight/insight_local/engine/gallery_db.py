from __future__ import annotations

import sqlite3
import threading
import time
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import SIMILARITY_MODEL_PATH
from .face_biometrics import FaceBiometricsEngine, encode_face_png
from .recognizer import MobileNetV3Embedder, similarity_search

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS identities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    group_name  TEXT NOT NULL DEFAULT '',
    source_path TEXT NOT NULL,
    added_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    identity_id  INTEGER NOT NULL REFERENCES identities(id) ON DELETE CASCADE,
    embedding    BLOB NOT NULL,
    quality      REAL NOT NULL DEFAULT 1.0,
    face_png     BLOB,
    sample_type  TEXT NOT NULL DEFAULT 'generic'
);

CREATE INDEX IF NOT EXISTS idx_identities_name ON identities(name);
CREATE INDEX IF NOT EXISTS idx_embeddings_identity ON embeddings(identity_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(sample_type);

CREATE TABLE IF NOT EXISTS similarity_items (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    display_name TEXT NOT NULL,
    batch_label  TEXT NOT NULL DEFAULT '',
    source_path  TEXT NOT NULL,
    thumb_png    BLOB,
    added_at     REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS similarity_embeddings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id    INTEGER NOT NULL REFERENCES similarity_items(id) ON DELETE CASCADE,
    embedding BLOB NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_similarity_items_name ON similarity_items(display_name);
CREATE INDEX IF NOT EXISTS idx_similarity_embeddings_item ON similarity_embeddings(item_id);
"""


@dataclass
class GalleryStats:
    identity_count: int
    image_count: int
    group_names: list[str]
    last_rebuild: float
    similarity_item_count: int = 0


@dataclass
class GalleryEntry:
    identity_id: int
    name: str
    group_name: str
    source_path: str
    embedding_count: int


@dataclass
class IdentityProfile:
    name: str
    group_name: str
    source_path: str
    embedding: np.ndarray
    sample_count: int
    mean_quality: float


@dataclass
class SimilarityItem:
    item_id: int
    display_name: str
    batch_label: str
    source_path: str
    thumb_png: bytes


class GalleryDB:
    """
    Persistent SQLite store for face-gallery identities and their face features.
    """

    def __init__(self, db_path: Path, embedder=None, read_only: bool = False) -> None:
        self._path = db_path
        self._read_only = bool(read_only)
        self._embedder = embedder or MobileNetV3Embedder(SIMILARITY_MODEL_PATH)
        self._lock = threading.RLock()
        self._face_engine = FaceBiometricsEngine()
        if not self._read_only:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._matrix: Optional[np.ndarray] = None
        self._matrix_labels: list[str] = []
        self._matrix_groups: list[str] = []
        self._matrix_sources: list[str] = []
        self._profile_matrix: Optional[np.ndarray] = None
        self._profile_labels: list[str] = []
        self._profile_groups: list[str] = []
        self._profile_sources: list[str] = []
        self._profile_sample_counts: list[int] = []
        self._similarity_matrix: Optional[np.ndarray] = None
        self._similarity_item_ids: list[int] = []
        self._similarity_names: list[str] = []
        self._similarity_batches: list[str] = []
        self._similarity_sources: list[str] = []
        self._last_rebuild: float = 0.0
        self._has_sample_type = False
        self._has_similarity_tables = False
        self._connect()
        self._detect_schema_capabilities()
        if not self._read_only:
            self._init_schema()
            self._detect_schema_capabilities()
        self.build_matrix()

    @staticmethod
    def verify_integrity(db_path: Path) -> tuple[bool, str]:
        if not db_path.exists():
            return True, ""
        conn = sqlite3.connect(str(db_path))
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
        message = str(result[0]) if result else "unknown integrity result"
        return (message.lower() == "ok", message)

    @staticmethod
    def quarantine_database(db_path: Path, reason: str = "") -> Optional[Path]:
        if not db_path.exists():
            return None
        stamp = time.strftime("%Y%m%d%H%M%S")
        suffix = db_path.suffix or ".db"
        quarantine = db_path.with_name(f"{db_path.stem}.corrupt.{stamp}{suffix}")
        db_path.rename(quarantine)
        for extra_suffix in ("-wal", "-shm"):
            extra = Path(str(db_path) + extra_suffix)
            if extra.exists():
                extra.rename(Path(str(quarantine) + extra_suffix))
        return quarantine

    @property
    def _db_path(self) -> Path:
        return self._path

    def _connect(self) -> None:
        if self._read_only:
            uri = f"file:{self._db_path}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        else:
            self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def _require_writable(self) -> None:
        if self._read_only:
            raise RuntimeError("gallery is read-only")

    def _detect_schema_capabilities(self) -> None:
        try:
            emb_cols = {
                str(row["name"])
                for row in self._conn.execute("PRAGMA table_info(embeddings)").fetchall()
            }
        except sqlite3.OperationalError:
            emb_cols = set()
        self._has_sample_type = "sample_type" in emb_cols

        try:
            tbl_rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            tables = {str(row["name"]) for row in tbl_rows}
        except sqlite3.OperationalError:
            tables = set()
        self._has_similarity_tables = (
            "similarity_items" in tables and "similarity_embeddings" in tables
        )

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._migrate_schema_locked()
            self._conn.commit()

    def _migrate_schema_locked(self) -> None:
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(embeddings)").fetchall()
        }
        if "face_png" not in columns:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN face_png BLOB")
        if "sample_type" not in columns:
            self._conn.execute("ALTER TABLE embeddings ADD COLUMN sample_type TEXT NOT NULL DEFAULT 'generic'")

    def ensure_face_backend(self) -> bool:
        return self._face_engine.ensure_loaded()

    @property
    def face_backend_error(self) -> str:
        return self._face_engine.load_error or ""

    def ensure_similarity_backend(self) -> bool:
        return self._embedder.ensure_loaded()

    @property
    def similarity_backend_error(self) -> str:
        return getattr(self._embedder, "load_error", "") or ""

    def extract_face(self, bgr: np.ndarray):
        return self._face_engine.extract_face(bgr)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def ingest_folder(
        self,
        folder: Path,
        identity_name: str,
        group_name: str = "",
        progress_cb=None,
    ) -> tuple[int, list[str]]:
        self._require_writable()
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in _SUPPORTED_EXTS)
        if not images:
            return 0, [f"No supported images found in {folder}"]
        if not self.ensure_face_backend():
            return 0, [self.face_backend_error or "Face recognition backend unavailable"]

        errors: list[str] = []
        added = 0
        for idx, img_path in enumerate(images):
            if progress_cb:
                progress_cb(idx, len(images), str(img_path.name))
            try:
                bgr = cv2.imread(str(img_path))
                if bgr is None:
                    errors.append(f"Cannot read: {img_path.name}")
                    continue
                ok, message = self._ingest_face_sample(
                    bgr=bgr,
                    identity_name=identity_name,
                    group_name=group_name,
                    source_path=str(img_path),
                )
                if ok:
                    added += 1
                else:
                    errors.append(f"{img_path.name}: {message}")
            except Exception as exc:
                errors.append(f"{img_path.name}: {exc}")
        return added, errors

    def ingest_single(
        self,
        image_path: Path,
        identity_name: str,
        group_name: str = "",
    ) -> tuple[bool, str]:
        self._require_writable()
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return False, f"Cannot read image: {image_path}"
        return self._ingest_face_sample(
            bgr=bgr,
            identity_name=identity_name,
            group_name=group_name,
            source_path=str(image_path),
        )

    def ingest_bgr(
        self,
        bgr,
        identity_name: str,
        group_name: str = "",
        source_label: str = "crop",
    ) -> tuple[bool, str]:
        self._require_writable()
        if bgr is None or bgr.size == 0:
            return False, "Empty image"
        return self._ingest_face_sample(
            bgr=bgr,
            identity_name=identity_name,
            group_name=group_name,
            source_path=source_label,
        )

    def _ingest_face_sample(
        self,
        bgr: np.ndarray,
        identity_name: str,
        group_name: str,
        source_path: str,
    ) -> tuple[bool, str]:
        if not self.ensure_face_backend():
            return False, self.face_backend_error or "Face recognition backend unavailable"
        sample = self._face_engine.extract_face(bgr)
        if sample is None:
            return False, "No usable face detected"

        face_png = encode_face_png(sample.aligned_bgr)
        if not face_png:
            return False, "Could not encode aligned face"

        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO identities (name, group_name, source_path, added_at) VALUES (?,?,?,?)",
                (identity_name, group_name or "", source_path, time.time()),
            )
            identity_id = cur.lastrowid
            self._conn.execute(
                """
                INSERT INTO embeddings (identity_id, embedding, quality, face_png, sample_type)
                VALUES (?,?,?,?,?)
                """,
                (
                    identity_id,
                    sample.feature.astype(np.float32).tobytes(),
                    float(sample.quality),
                    face_png,
                    "face",
                ),
            )
            self._conn.commit()
        return True, ""

    def ingest_similarity_image(
        self,
        image_path: Path,
        batch_label: str = "",
    ) -> tuple[bool, str]:
        self._require_writable()
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            return False, f"Cannot read image: {image_path}"
        display_name = image_path.stem.replace("_", " ").replace("-", " ").strip() or image_path.stem
        batch = batch_label.strip() or image_path.parent.name
        return self._ingest_similarity_sample(
            bgr=bgr,
            display_name=display_name,
            batch_label=batch,
            source_path=str(image_path),
        )

    def ingest_similarity_folder(
        self,
        folder: Path,
        progress_cb=None,
    ) -> tuple[int, list[str]]:
        self._require_writable()
        images = sorted(
            p for p in folder.rglob("*")
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTS
        )
        if not images:
            return 0, [f"No supported images found in {folder}"]
        if not self.ensure_similarity_backend():
            return 0, [self.similarity_backend_error or "Similarity search backend unavailable"]

        errors: list[str] = []
        added = 0
        batch_label = folder.name
        for idx, img_path in enumerate(images):
            if progress_cb:
                progress_cb(idx, len(images), str(img_path.name))
            ok, message = self.ingest_similarity_image(img_path, batch_label=batch_label)
            if ok:
                added += 1
            else:
                errors.append(f"{img_path.name}: {message}")
        return added, errors

    def _ingest_similarity_sample(
        self,
        bgr: np.ndarray,
        display_name: str,
        batch_label: str,
        source_path: str,
    ) -> tuple[bool, str]:
        if not self.ensure_similarity_backend():
            return False, self.similarity_backend_error or "Similarity search backend unavailable"
        embedding = self._embedder.embed(bgr)
        if embedding is None:
            return False, "Could not create similarity features"
        thumb_png = self._encode_thumb_png(bgr)
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO similarity_items (display_name, batch_label, source_path, thumb_png, added_at)
                VALUES (?,?,?,?,?)
                """,
                (display_name, batch_label, source_path, thumb_png, time.time()),
            )
            item_id = int(cur.lastrowid)
            self._conn.execute(
                "INSERT INTO similarity_embeddings (item_id, embedding) VALUES (?,?)",
                (item_id, embedding.astype(np.float32).tobytes()),
            )
            self._conn.commit()
        return True, ""

    # ------------------------------------------------------------------
    # Matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_thumb_png(bgr: np.ndarray, max_dim: int = 144) -> bytes:
        if bgr is None or bgr.size == 0:
            return b""
        height, width = bgr.shape[:2]
        scale = min(1.0, float(max_dim) / max(1, max(height, width)))
        if scale < 1.0:
            resized = cv2.resize(
                bgr,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            resized = bgr
        ok, encoded = cv2.imencode(".png", resized)
        return encoded.tobytes() if ok else b""

    def build_matrix(self) -> int:
        if not self.ensure_face_backend():
            self._clear_index()
        else:
            if not self._read_only:
                self._upgrade_legacy_rows()
            where_face = "WHERE e.sample_type = 'face'" if self._has_sample_type else ""
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT i.name, i.group_name, i.source_path, e.embedding, e.quality
                    FROM embeddings e
                    JOIN identities i ON e.identity_id = i.id
                    {where_face}
                    ORDER BY e.id
                    """.format(where_face=where_face)
                ).fetchall()

            if not rows:
                self._clear_face_index()
            else:
                vectors: list[np.ndarray] = []
                labels: list[str] = []
                groups: list[str] = []
                sources: list[str] = []
                identity_groups: dict[tuple[str, str], list[tuple[np.ndarray, float, str]]] = {}
                feature_dim = 0

                for row in rows:
                    vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
                    if vec.size == 0:
                        continue
                    if feature_dim == 0:
                        feature_dim = int(vec.shape[0])
                    if vec.shape[0] != feature_dim:
                        continue
                    norm = float(np.linalg.norm(vec))
                    if norm < 1e-6:
                        continue
                    vec = vec / norm
                    vectors.append(vec)
                    labels.append(str(row["name"]))
                    groups.append(str(row["group_name"]))
                    sources.append(str(row["source_path"]))
                    key = (str(row["name"]), str(row["group_name"]))
                    quality = float(row["quality"])
                    identity_groups.setdefault(key, []).append((vec, max(0.05, quality), str(row["source_path"])))

                if not vectors:
                    self._clear_face_index()
                else:
                    self._matrix = np.stack(vectors, axis=0).astype(np.float32)
                    self._matrix_labels = labels
                    self._matrix_groups = groups
                    self._matrix_sources = sources

                    profile_vectors: list[np.ndarray] = []
                    profile_labels: list[str] = []
                    profile_groups: list[str] = []
                    profile_sources: list[str] = []
                    profile_sample_counts: list[int] = []

                    for (name, group_name), samples in sorted(identity_groups.items()):
                        sample_vectors = np.stack([item[0] for item in samples], axis=0).astype(np.float32)
                        weights = np.array([item[1] for item in samples], dtype=np.float32)
                        centroid = np.sum(sample_vectors * weights[:, None], axis=0)
                        norm = float(np.linalg.norm(centroid))
                        if norm < 1e-6:
                            continue
                        profile_vectors.append((centroid / norm).astype(np.float32))
                        profile_labels.append(name)
                        profile_groups.append(group_name)
                        profile_sources.append(samples[0][2])
                        profile_sample_counts.append(len(samples))

                    self._profile_matrix = (
                        np.stack(profile_vectors, axis=0).astype(np.float32)
                        if profile_vectors
                        else np.zeros((0, feature_dim), dtype=np.float32)
                    )
                    self._profile_labels = profile_labels
                    self._profile_groups = profile_groups
                    self._profile_sources = profile_sources
                    self._profile_sample_counts = profile_sample_counts

        self._build_similarity_index()
        self._last_rebuild = time.time()
        return int(self._matrix.shape[0] if self._matrix is not None else 0)

    def _clear_face_index(self) -> None:
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._matrix_labels = []
        self._matrix_groups = []
        self._matrix_sources = []
        self._profile_matrix = np.zeros((0, 0), dtype=np.float32)
        self._profile_labels = []
        self._profile_groups = []
        self._profile_sources = []
        self._profile_sample_counts = []

    def _clear_similarity_index(self) -> None:
        self._similarity_matrix = np.zeros((0, 0), dtype=np.float32)
        self._similarity_item_ids = []
        self._similarity_names = []
        self._similarity_batches = []
        self._similarity_sources = []

    def _clear_index(self) -> None:
        self._clear_face_index()
        self._clear_similarity_index()

    def _build_similarity_index(self) -> None:
        if not self._has_similarity_tables:
            self._clear_similarity_index()
            return
        if not self.ensure_similarity_backend():
            self._clear_similarity_index()
            return
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT i.id, i.display_name, i.batch_label, i.source_path, e.embedding
                    FROM similarity_items i
                    JOIN similarity_embeddings e ON e.item_id = i.id
                    ORDER BY i.id
                    """
                ).fetchall()
            except sqlite3.OperationalError:
                self._clear_similarity_index()
                return
        if not rows:
            self._clear_similarity_index()
            return
        vectors: list[np.ndarray] = []
        item_ids: list[int] = []
        display_names: list[str] = []
        batch_labels: list[str] = []
        source_paths: list[str] = []
        feature_dim = 0
        for row in rows:
            vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if vec.size == 0:
                continue
            if feature_dim == 0:
                feature_dim = int(vec.shape[0])
            if vec.shape[0] != feature_dim:
                continue
            norm = float(np.linalg.norm(vec))
            if norm < 1e-6:
                continue
            vectors.append((vec / norm).astype(np.float32))
            item_ids.append(int(row["id"]))
            display_names.append(str(row["display_name"]))
            batch_labels.append(str(row["batch_label"]))
            source_paths.append(str(row["source_path"]))
        if not vectors:
            self._clear_similarity_index()
            return
        self._similarity_matrix = np.stack(vectors, axis=0).astype(np.float32)
        self._similarity_item_ids = item_ids
        self._similarity_names = display_names
        self._similarity_batches = batch_labels
        self._similarity_sources = source_paths

    def _upgrade_legacy_rows(self) -> None:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT e.id AS embedding_id, i.source_path
                FROM embeddings e
                JOIN identities i ON e.identity_id = i.id
                WHERE (e.sample_type != 'face' OR e.face_png IS NULL OR length(e.embedding) = 0)
                ORDER BY e.id
                """
            ).fetchall()

        updates: list[tuple[bytes, float, bytes, int]] = []
        for row in rows:
            source_path = str(row["source_path"] or "")
            if not source_path or not Path(source_path).exists():
                continue
            bgr = cv2.imread(source_path)
            if bgr is None:
                continue
            sample = self._face_engine.extract_face(bgr)
            if sample is None:
                continue
            face_png = encode_face_png(sample.aligned_bgr)
            if not face_png:
                continue
            updates.append(
                (
                    sample.feature.astype(np.float32).tobytes(),
                    float(sample.quality),
                    face_png,
                    int(row["embedding_id"]),
                )
            )

        if not updates:
            return

        with self._lock:
            self._conn.executemany(
                """
                UPDATE embeddings
                SET embedding = ?, quality = ?, face_png = ?, sample_type = 'face'
                WHERE id = ?
                """,
                updates,
            )
            self._conn.commit()

    @property
    def matrix(self) -> Optional[np.ndarray]:
        return self._matrix

    @property
    def matrix_labels(self) -> list[str]:
        return self._matrix_labels

    @property
    def matrix_groups(self) -> list[str]:
        return self._matrix_groups

    @property
    def matrix_sources(self) -> list[str]:
        return self._matrix_sources

    @property
    def has_gallery(self) -> bool:
        return self._matrix is not None and self._matrix.size > 0 and self._matrix.shape[0] > 0

    @property
    def profile_matrix(self) -> Optional[np.ndarray]:
        return self._profile_matrix

    @property
    def profile_labels(self) -> list[str]:
        return self._profile_labels

    @property
    def profile_groups(self) -> list[str]:
        return self._profile_groups

    @property
    def profile_sources(self) -> list[str]:
        return self._profile_sources

    @property
    def profile_sample_counts(self) -> list[int]:
        return self._profile_sample_counts

    @property
    def similarity_matrix(self) -> Optional[np.ndarray]:
        return self._similarity_matrix

    @property
    def similarity_item_ids(self) -> list[int]:
        return self._similarity_item_ids

    @property
    def similarity_display_names(self) -> list[str]:
        return self._similarity_names

    @property
    def similarity_batch_labels(self) -> list[str]:
        return self._similarity_batches

    @property
    def similarity_source_paths(self) -> list[str]:
        return self._similarity_sources

    @property
    def has_similarity_items(self) -> bool:
        return (
            self._similarity_matrix is not None
            and self._similarity_matrix.size > 0
            and self._similarity_matrix.shape[0] > 0
        )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_stats(self) -> GalleryStats:
        face_where = "WHERE e.sample_type = 'face'" if self._has_sample_type else ""
        face_and = "AND e.sample_type = 'face'" if self._has_sample_type else ""
        with self._lock:
            identity_count = self._conn.execute(
                """
                SELECT COUNT(DISTINCT i.name)
                FROM identities i
                JOIN embeddings e ON e.identity_id = i.id
                {face_where}
                """
                .format(face_where=face_where)
            ).fetchone()[0]
            image_count = self._conn.execute(
                "SELECT COUNT(*) FROM embeddings " + ("WHERE sample_type = 'face'" if self._has_sample_type else "")
            ).fetchone()[0]
            groups_raw = self._conn.execute(
                """
                SELECT DISTINCT i.group_name
                FROM identities i
                JOIN embeddings e ON e.identity_id = i.id
                WHERE i.group_name != '' {face_and}
                ORDER BY i.group_name
                """
                .format(face_and=face_and)
            ).fetchall()
            if self._has_similarity_tables:
                try:
                    similarity_item_count = self._conn.execute(
                        "SELECT COUNT(*) FROM similarity_items"
                    ).fetchone()[0]
                except sqlite3.OperationalError:
                    similarity_item_count = 0
            else:
                similarity_item_count = 0
        return GalleryStats(
            identity_count=int(identity_count or 0),
            image_count=int(image_count or 0),
            group_names=[r[0] for r in groups_raw],
            last_rebuild=self._last_rebuild,
            similarity_item_count=int(similarity_item_count or 0),
        )

    def list_identities(self, group_filter: str = "") -> list[GalleryEntry]:
        face_and = "AND e.sample_type = 'face'" if self._has_sample_type else ""
        with self._lock:
            if group_filter:
                rows = self._conn.execute(
                    """
                    SELECT MIN(i.id) AS id, i.name, i.group_name, MIN(i.source_path) AS source_path,
                           COUNT(e.id) AS emb_count
                    FROM identities i
                    JOIN embeddings e ON e.identity_id = i.id
                    WHERE i.group_name = ? {face_and}
                    GROUP BY i.name, i.group_name
                    ORDER BY i.name
                    """.format(face_and=face_and),
                    (group_filter,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT MIN(i.id) AS id, i.name, i.group_name, MIN(i.source_path) AS source_path,
                           COUNT(e.id) AS emb_count
                    FROM identities i
                    JOIN embeddings e ON e.identity_id = i.id
                    {where_face}
                    GROUP BY i.name, i.group_name
                    ORDER BY i.name
                    """
                    .format(
                        face_and=face_and,
                        where_face=("WHERE e.sample_type = 'face'" if self._has_sample_type else ""),
                    )
                ).fetchall()
        return [
            GalleryEntry(
                identity_id=int(r["id"]),
                name=str(r["name"]),
                group_name=str(r["group_name"]),
                source_path=str(r["source_path"]),
                embedding_count=int(r["emb_count"]),
            )
            for r in rows
        ]

    def get_all_source_paths(self) -> set[str]:
        """Return all source paths already ingested (used for incremental update deduplication)."""
        with self._lock:
            rows = self._conn.execute("SELECT source_path FROM identities").fetchall()
        return {str(r["source_path"]) for r in rows}

    def get_identity_images(self, identity_name: str) -> list[str]:
        face_and = "AND e.sample_type = 'face'" if self._has_sample_type else ""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT i.source_path
                FROM identities i
                JOIN embeddings e ON e.identity_id = i.id
                WHERE i.name = ? {face_and}
                ORDER BY i.added_at
                """.format(face_and=face_and),
                (identity_name,),
            ).fetchall()
        return [str(r["source_path"]) for r in rows]

    def list_similarity_items(self) -> list[SimilarityItem]:
        if not self._has_similarity_tables:
            return []
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT id, display_name, batch_label, source_path, thumb_png
                    FROM similarity_items
                    ORDER BY added_at DESC, id DESC
                    """
                ).fetchall()
            except sqlite3.OperationalError:
                return []
        return [
            SimilarityItem(
                item_id=int(row["id"]),
                display_name=str(row["display_name"]),
                batch_label=str(row["batch_label"]),
                source_path=str(row["source_path"]),
                thumb_png=bytes(row["thumb_png"] or b""),
            )
            for row in rows
        ]

    def get_similarity_item_path(self, item_id: int) -> str:
        if not self._has_similarity_tables:
            return ""
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT source_path FROM similarity_items WHERE id = ?",
                    (item_id,),
                ).fetchone()
            except sqlite3.OperationalError:
                return ""
        return str(row["source_path"]) if row else ""

    def find_similar_items(self, item_id: int, top_k: int = 12) -> list[dict[str, object]]:
        if not self.has_similarity_items:
            return []
        try:
            index = self._similarity_item_ids.index(int(item_id))
        except ValueError:
            return []
        query = self._similarity_matrix[index]
        results = similarity_search(
            query=query,
            gallery_matrix=self._similarity_matrix,
            item_ids=self._similarity_item_ids,
            display_names=self._similarity_names,
            batch_labels=self._similarity_batches,
            source_paths=self._similarity_sources,
            exclude_item_id=int(item_id),
            top_k=top_k,
        )
        return [
            {
                "item_id": match.item_id,
                "display_name": match.display_name,
                "batch_label": match.batch_label,
                "similarity": match.similarity,
                "source_path": match.source_path,
            }
            for match in results
        ]

    def delete_identity(self, identity_name: str) -> int:
        self._require_writable()
        with self._lock:
            ids = self._conn.execute(
                """
                SELECT DISTINCT i.id
                FROM identities i
                JOIN embeddings e ON e.identity_id = i.id
                WHERE i.name = ? AND e.sample_type = 'face'
                """,
                (identity_name,),
            ).fetchall()
            if not ids:
                return 0
            id_list = [int(r["id"]) for r in ids]
            placeholders = ",".join("?" * len(id_list))
            self._conn.execute(
                f"DELETE FROM identities WHERE id IN ({placeholders})",
                id_list,
            )
            self._conn.commit()
        return len(id_list)

    def delete_similarity_item(self, item_id: int) -> int:
        self._require_writable()
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM similarity_items WHERE id = ?",
                (int(item_id),),
            )
            self._conn.commit()
        return int(cur.rowcount or 0)

    def rename_identity(self, old_name: str, new_name: str) -> None:
        self._require_writable()
        with self._lock:
            self._conn.execute(
                "UPDATE identities SET name = ? WHERE name = ?",
                (new_name, old_name),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
