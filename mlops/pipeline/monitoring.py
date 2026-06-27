from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .registry import REPO_ROOT

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

INFERENCE_DB_PATH = REPO_ROOT / "state" / "insight_local" / "cvops" / "inference_log.db"
_LOCK = threading.RLock()

_SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS inference_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    scenario TEXT NOT NULL,
    model_version TEXT NOT NULL,
    detection_count INTEGER NOT NULL,
    signal_flag INTEGER NOT NULL,
    summary TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    drift_score REAL,
    drift_alert INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_inf_scen_ts ON inference_samples(scenario, ts);
"""


def _ensure_db() -> None:
    INFERENCE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        conn = sqlite3.connect(str(INFERENCE_DB_PATH), check_same_thread=False)
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


def _embedding_from_bgr(image_bgr: Any) -> list[float]:
    if cv2 is None or image_bgr is None:
        return []
    try:
        small = cv2.resize(image_bgr, (32, 32), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        vec = gray.flatten()
        n = float(np.linalg.norm(vec) + 1e-9)
        return [float(x / n) for x in vec.tolist()]
    except Exception:
        return []


def _recent_mean_embedding(conn: sqlite3.Connection, scenario: str, version: str, limit: int = 48) -> Optional[list[float]]:
    rows = conn.execute(
        """
        SELECT embedding_json FROM inference_samples
        WHERE scenario = ? AND model_version = ?
        ORDER BY id DESC LIMIT ?
        """,
        (scenario, version, limit),
    ).fetchall()
    vectors: list[list[float]] = []
    for (raw,) in rows:
        try:
            vec = json.loads(raw)
        except Exception:
            continue
        if isinstance(vec, list) and len(vec) == 32 * 32:
            vectors.append([float(x) for x in vec])
    if not vectors:
        return None
    arr = np.array(vectors, dtype=np.float64)
    mean = arr.mean(axis=0)
    n = float(np.linalg.norm(mean) + 1e-9)
    return [float(x / n) for x in mean.tolist()]


def _l2(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    diff = np.array(a, dtype=np.float64) - np.array(b, dtype=np.float64)
    return float(np.linalg.norm(diff))


def maybe_log_inference(
    *,
    scenario: str,
    image_bgr: Any,
    result: dict[str, Any],
) -> None:
    """Sampled inference telemetry with PII-safe embeddings (no raw pixels in DB)."""
    rate = float(os.environ.get("CVOPS_INFERENCE_LOG_SAMPLE_RATE", "0") or 0.0)
    if rate <= 0.0:
        return
    if random.random() > min(1.0, max(0.0, rate)):
        return
    if os.environ.get("CVOPS_INFERENCE_LOG_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return
    opt_out = os.environ.get("CVOPS_INFERENCE_PII_OPT_OUT", "1").strip().lower() in {"1", "true", "yes"}
    if not opt_out:
        # Default strict: never store raw imagery; only structured counts.
        pass
    _ensure_db()
    scenario = str(scenario or "").strip()
    version = str(result.get("model_version") or "")
    dets = result.get("detections") if isinstance(result.get("detections"), list) else []
    signal = result.get("signal") if isinstance(result.get("signal"), dict) else {}
    summary = str(signal.get("summary") or "")[:2000]
    flag = 1 if bool(signal.get("flag")) else 0
    emb = _embedding_from_bgr(image_bgr)
    emb_json = json.dumps(emb, ensure_ascii=True)
    drift_score = 0.0
    alert = 0
    with _LOCK:
        conn = sqlite3.connect(str(INFERENCE_DB_PATH), check_same_thread=False)
        try:
            mean = _recent_mean_embedding(conn, scenario, version, limit=64)
            if mean is not None and emb:
                drift_score = _l2(emb, mean)
                thr = float(os.environ.get("CVOPS_INFERENCE_DRIFT_L2", "0.35") or 0.35)
                if drift_score > thr:
                    alert = 1
            conn.execute(
                """
                INSERT INTO inference_samples
                (ts, scenario, model_version, detection_count, signal_flag, summary, embedding_json, drift_score, drift_alert)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    time.time(),
                    scenario,
                    version,
                    len(dets),
                    flag,
                    summary,
                    emb_json,
                    drift_score,
                    alert,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    _tap_forecasting_spine(scenario=scenario, version=version, dets=dets, drift_score=drift_score)
    if alert:
        try:
            from .integration import append_integration_event

            append_integration_event(
                {
                    "type": "inference_drift_alert",
                    "scenario": scenario,
                    "model_version": version,
                    "drift_score": drift_score,
                    "threshold": float(os.environ.get("CVOPS_INFERENCE_DRIFT_L2", "0.35") or 0.35),
                    "emitted_at": time.time(),
                }
            )
        except Exception:
            pass


def _tap_forecasting_spine(*, scenario: str, version: str, dets: list[Any], drift_score: float) -> None:
    """Mirror inference telemetry into the signal spine.

    This is a best-effort fan-out — the forecasting package is optional and
    any failure here must not affect the inference path.
    """
    if os.environ.get("CVOPS_FORECASTING_DISABLE", "").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        from ..forecasting.runtime import get_runtime
        from ..forecasting.spine import Sample
    except Exception:
        return
    try:
        rt = get_runtime()
        now = time.time()
        scen = scenario or "global"
        samples = [
            Sample(
                signal_id=f"cv/{scen}/detection_count",
                ts=now,
                value=float(len(dets)),
                source="monitoring",
                metadata={"model_version": version},
            ),
            Sample(
                signal_id=f"cv/{scen}/drift_score",
                ts=now,
                value=float(drift_score),
                source="monitoring",
                metadata={"model_version": version},
            ),
        ]
        for s in samples:
            rt.spine.append(s)
            rt.register_signal(s.signal_id)
    except Exception:
        pass
