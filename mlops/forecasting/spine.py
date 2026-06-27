from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from ..pipeline.registry import REPO_ROOT


SPINE_DB_PATH = REPO_ROOT / "state" / "insight_local" / "forecasting" / "signals.db"

_SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS signals (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id     TEXT NOT NULL,
    ts            REAL NOT NULL,
    value         REAL NOT NULL,
    source        TEXT NOT NULL DEFAULT '',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_sig_id_ts ON signals(signal_id, ts);

CREATE TABLE IF NOT EXISTS signal_registry (
    signal_id     TEXT PRIMARY KEY,
    description   TEXT NOT NULL DEFAULT '',
    unit          TEXT NOT NULL DEFAULT '',
    source        TEXT NOT NULL DEFAULT '',
    config_json   TEXT NOT NULL DEFAULT '{}',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS projections (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    signal_id     TEXT NOT NULL,
    arm_name      TEXT NOT NULL,
    ts_issued     REAL NOT NULL,
    ts_target     REAL NOT NULL,
    value         REAL NOT NULL,
    ci_low        REAL NOT NULL,
    ci_high       REAL NOT NULL,
    status        TEXT NOT NULL DEFAULT 'ok',
    metadata_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_proj_sig_issued
    ON projections(signal_id, ts_issued DESC);
CREATE INDEX IF NOT EXISTS idx_proj_sig_arm_issued
    ON projections(signal_id, arm_name, ts_issued DESC);
"""


@dataclass(frozen=True)
class Sample:
    signal_id: str
    ts: float
    value: float
    source: str = ""
    metadata: dict[str, Any] | None = None

    def to_row(self) -> tuple:
        return (
            self.signal_id,
            float(self.ts),
            float(self.value),
            self.source or "",
            json.dumps(self.metadata or {}, ensure_ascii=True),
        )


class SignalSpine:
    """Append-only time-series spine shared by every forecaster arm."""

    def __init__(self, db_path: Path | str = SPINE_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._subscribers: list[Callable[[Sample], None]] = []

    # ---- subscription ----------------------------------------------------

    def subscribe(self, callback: Callable[[Sample], None]) -> Callable[[], None]:
        self._subscribers.append(callback)

        def _unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _fanout(self, sample: Sample) -> None:
        for cb in list(self._subscribers):
            try:
                cb(sample)
            except Exception:
                pass

    # ---- registry --------------------------------------------------------

    def register_signal(
        self,
        signal_id: str,
        *,
        description: str = "",
        unit: str = "",
        source: str = "",
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sid = str(signal_id or "").strip()
        if not sid:
            raise ValueError("signal_id is required")
        now = time.time()
        cfg_json = json.dumps(config or {}, ensure_ascii=True)
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO signal_registry
                    (signal_id, description, unit, source, config_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(signal_id) DO UPDATE SET
                    description = excluded.description,
                    unit = excluded.unit,
                    source = excluded.source,
                    config_json = excluded.config_json,
                    updated_at = excluded.updated_at
                """,
                (sid, description, unit, source, cfg_json, now, now),
            )
            self._conn.commit()
        return self.get_signal(sid) or {}

    def get_signal(self, signal_id: str) -> Optional[dict[str, Any]]:
        sid = str(signal_id or "").strip()
        if not sid:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM signal_registry WHERE signal_id = ?", (sid,)
            ).fetchone()
        if row is None:
            return None
        return {
            "signal_id": row["signal_id"],
            "description": row["description"],
            "unit": row["unit"],
            "source": row["source"],
            "config": _loads_obj(row["config_json"]),
            "created_at": float(row["created_at"]),
            "updated_at": float(row["updated_at"]),
        }

    def list_signals(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM signal_registry ORDER BY signal_id ASC"
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "signal_id": row["signal_id"],
                    "description": row["description"],
                    "unit": row["unit"],
                    "source": row["source"],
                    "config": _loads_obj(row["config_json"]),
                    "created_at": float(row["created_at"]),
                    "updated_at": float(row["updated_at"]),
                }
            )
        return out

    # ---- samples ---------------------------------------------------------

    def append(self, sample: Sample) -> None:
        sid = str(sample.signal_id or "").strip()
        if not sid:
            raise ValueError("sample.signal_id is required")
        normalized = Sample(
            signal_id=sid,
            ts=float(sample.ts),
            value=float(sample.value),
            source=sample.source or "",
            metadata=dict(sample.metadata or {}),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO signals (signal_id, ts, value, source, metadata_json) "
                "VALUES (?, ?, ?, ?, ?)",
                normalized.to_row(),
            )
            existing = self._conn.execute(
                "SELECT 1 FROM signal_registry WHERE signal_id = ?", (sid,)
            ).fetchone()
            if existing is None:
                now = time.time()
                self._conn.execute(
                    """
                    INSERT INTO signal_registry
                        (signal_id, description, unit, source, config_json, created_at, updated_at)
                    VALUES (?, '', '', ?, '{}', ?, ?)
                    """,
                    (sid, normalized.source, now, now),
                )
            self._conn.commit()
        self._fanout(normalized)

    def append_many(self, samples: Iterable[Sample]) -> int:
        count = 0
        for s in samples:
            self.append(s)
            count += 1
        return count

    def recent(
        self,
        signal_id: str,
        *,
        limit: int = 512,
        since_ts: Optional[float] = None,
    ) -> list[Sample]:
        sid = str(signal_id or "").strip()
        if not sid:
            return []
        sql = "SELECT signal_id, ts, value, source, metadata_json FROM signals WHERE signal_id = ?"
        args: list[Any] = [sid]
        if since_ts is not None:
            sql += " AND ts >= ?"
            args.append(float(since_ts))
        sql += " ORDER BY ts DESC LIMIT ?"
        args.append(int(max(1, limit)))
        with self._lock:
            rows = self._conn.execute(sql, tuple(args)).fetchall()
        out = [
            Sample(
                signal_id=r["signal_id"],
                ts=float(r["ts"]),
                value=float(r["value"]),
                source=str(r["source"] or ""),
                metadata=_loads_obj(r["metadata_json"]),
            )
            for r in rows
        ]
        out.reverse()  # chronological ascending
        return out

    def count(self, signal_id: str) -> int:
        sid = str(signal_id or "").strip()
        if not sid:
            return 0
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM signals WHERE signal_id = ?", (sid,)
            ).fetchone()
        return int(row[0] or 0)

    # ---- projections -----------------------------------------------------

    def write_projection(
        self,
        *,
        signal_id: str,
        arm_name: str,
        ts_issued: float,
        points: list[tuple[float, float, float, float]],
        status: str = "ok",
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Points = list of (ts_target, value, ci_low, ci_high). Returns row count inserted."""
        sid = str(signal_id or "").strip()
        arm = str(arm_name or "").strip()
        if not sid or not arm:
            raise ValueError("signal_id and arm_name are required")
        meta_json = json.dumps(metadata or {}, ensure_ascii=True)
        status_v = str(status or "ok").strip() or "ok"
        rows = [
            (sid, arm, float(ts_issued), float(t), float(v), float(lo), float(hi), status_v, meta_json)
            for (t, v, lo, hi) in points
        ]
        if not rows:
            return 0
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO projections
                    (signal_id, arm_name, ts_issued, ts_target, value, ci_low, ci_high, status, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()
        return len(rows)

    def latest_projection(
        self,
        signal_id: str,
        *,
        arm_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        sid = str(signal_id or "").strip()
        if not sid:
            return []
        with self._lock:
            if arm_name:
                row = self._conn.execute(
                    """
                    SELECT MAX(ts_issued) AS mx FROM projections
                    WHERE signal_id = ? AND arm_name = ?
                    """,
                    (sid, arm_name),
                ).fetchone()
            else:
                row = self._conn.execute(
                    "SELECT MAX(ts_issued) AS mx FROM projections WHERE signal_id = ?",
                    (sid,),
                ).fetchone()
            if row is None or row["mx"] is None:
                return []
            mx = float(row["mx"])
            if arm_name:
                rows = self._conn.execute(
                    """
                    SELECT * FROM projections
                    WHERE signal_id = ? AND arm_name = ? AND ts_issued = ?
                    ORDER BY ts_target ASC
                    """,
                    (sid, arm_name, mx),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM projections
                    WHERE signal_id = ? AND ts_issued >= ?
                    ORDER BY arm_name ASC, ts_target ASC
                    """,
                    (sid, mx - 1e-6),
                ).fetchall()
        return [_projection_row_to_dict(r) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _loads_obj(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _projection_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "signal_id": row["signal_id"],
        "arm_name": row["arm_name"],
        "ts_issued": float(row["ts_issued"]),
        "ts_target": float(row["ts_target"]),
        "value": float(row["value"]),
        "ci_low": float(row["ci_low"]),
        "ci_high": float(row["ci_high"]),
        "status": str(row["status"]),
        "metadata": _loads_obj(row["metadata_json"]),
    }


_DEFAULT_SPINE: Optional[SignalSpine] = None
_DEFAULT_LOCK = threading.Lock()


def get_spine(db_path: Path | str | None = None) -> SignalSpine:
    """Process-global default spine, created on first use."""
    global _DEFAULT_SPINE
    with _DEFAULT_LOCK:
        if _DEFAULT_SPINE is None:
            _DEFAULT_SPINE = SignalSpine(db_path or SPINE_DB_PATH)
        return _DEFAULT_SPINE
