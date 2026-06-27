from __future__ import annotations

import csv
import io
import json
import time
from typing import Any, Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from pydantic import BaseModel

from .arms.drift_slope import DriftSlopeArm
from .runtime import ForecastingRuntime, get_runtime
from .spine import Sample


def build_router(runtime: Optional[ForecastingRuntime] = None) -> APIRouter:
    rt = runtime or get_runtime()
    router = APIRouter(prefix="")

    # ---- signal registry & ingest ---------------------------------------

    @router.get("/forecasting/signals")
    async def list_signals() -> dict[str, Any]:
        signals = rt.spine.list_signals()
        for sig in signals:
            sig["sample_count"] = rt.spine.count(sig["signal_id"])
            sig["arms"] = rt.octopus.arm_names(sig["signal_id"])
        return {"signals": signals}

    @router.get("/forecasting/signals/{signal_id}")
    async def get_signal(signal_id: str) -> dict[str, Any]:
        info = rt.spine.get_signal(signal_id)
        if info is None:
            raise HTTPException(status_code=404, detail="signal not found")
        info["sample_count"] = rt.spine.count(signal_id)
        info["arms"] = rt.octopus.arm_names(signal_id)
        info["health"] = {
            name: {
                "status": h.status,
                "last_error": h.last_error,
                "last_mse": h.last_mse,
                "consecutive_failures": h.consecutive_failures,
                "last_update_ts": h.last_update_ts,
            }
            for name, h in rt.octopus.health(signal_id).items()
        }
        return info

    class RegisterSignalRequest(BaseModel):
        signal_id: str
        description: str = ""
        unit: str = ""
        source: str = ""
        config: dict[str, Any] = {}
        min_interval: Optional[float] = None

    @router.post("/forecasting/signals")
    async def register_signal(req: RegisterSignalRequest) -> dict[str, Any]:
        try:
            info = rt.spine.register_signal(
                req.signal_id,
                description=req.description,
                unit=req.unit,
                source=req.source,
                config=req.config,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        rt.register_signal(req.signal_id, min_interval=req.min_interval)
        return info

    class AppendSampleRequest(BaseModel):
        signal_id: str
        ts: Optional[float] = None
        value: float
        source: str = ""
        metadata: dict[str, Any] = {}

    @router.post("/forecasting/samples")
    async def append_sample(req: AppendSampleRequest) -> dict[str, Any]:
        sample = Sample(
            signal_id=req.signal_id,
            ts=float(req.ts if req.ts is not None else time.time()),
            value=float(req.value),
            source=req.source,
            metadata=req.metadata,
        )
        try:
            rt.spine.append(sample)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        rt.register_signal(req.signal_id)
        return {"ok": True, "signal_id": sample.signal_id, "ts": sample.ts}

    @router.post("/ingest/signals")
    async def ingest_signals(
        file: UploadFile = File(...),
        ts_column: str = Form("ts"),
        source: str = Form("uploaded"),
        signal_prefix: str = Form(""),
    ) -> dict[str, Any]:
        """Accepts CSV or JSONL with a ts column + one or more numeric value columns.
        Each value column becomes a signal_id (prefixed with signal_prefix/)."""
        try:
            raw = await file.read()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"could not read upload: {e}")
        if not raw:
            raise HTTPException(status_code=400, detail="empty upload")

        filename = (file.filename or "").lower()
        samples = _parse_rows(raw, filename, ts_column)
        if not samples:
            raise HTTPException(status_code=400, detail="no rows parsed from upload")

        prefix = (signal_prefix or "").strip().strip("/")
        ingested: dict[str, int] = {}
        registered: set[str] = set()
        for row_ts, row_values in samples:
            for col, val in row_values.items():
                signal_id = f"{prefix}/{col}" if prefix else col
                rt.spine.append(
                    Sample(
                        signal_id=signal_id,
                        ts=float(row_ts),
                        value=float(val),
                        source=source,
                    )
                )
                ingested[signal_id] = ingested.get(signal_id, 0) + 1
                if signal_id not in registered:
                    rt.register_signal(signal_id)
                    registered.add(signal_id)
        return {"ok": True, "signals_ingested": ingested}

    # ---- projection queries ---------------------------------------------

    @router.get("/forecasting/project/{signal_id}")
    async def project(
        signal_id: str,
        horizon: int = Query(12, ge=1, le=512),
        arm: str = Query("ensemble"),
    ) -> dict[str, Any]:
        if rt.spine.get_signal(signal_id) is None:
            raise HTTPException(status_code=404, detail="signal not found")
        composite = await rt.octopus.tick(signal_id)
        if arm == "ensemble":
            points = composite.ensemble[:horizon]
            return {
                "signal_id": signal_id,
                "arm": "ensemble",
                "ts_issued": composite.ts_issued,
                "points": [
                    {
                        "ts_target": p.ts_target,
                        "value": p.value,
                        "ci_low": p.ci_low,
                        "ci_high": p.ci_high,
                    }
                    for p in points
                ],
                "arm_weights": composite.arm_weights,
            }
        proj = composite.per_arm.get(arm)
        if proj is None:
            raise HTTPException(status_code=404, detail=f"arm not found: {arm}")
        return {
            "signal_id": signal_id,
            "arm": arm,
            "ts_issued": composite.ts_issued,
            "status": proj.status,
            "points": [
                {
                    "ts_target": p.ts_target,
                    "value": p.value,
                    "ci_low": p.ci_low,
                    "ci_high": p.ci_high,
                }
                for p in proj.points[:horizon]
            ],
            "metadata": proj.metadata,
        }

    @router.get("/forecasting/trend/{signal_id}")
    async def trend(
        signal_id: str,
        window: int = Query(64, ge=2, le=4096),
    ) -> dict[str, Any]:
        if rt.spine.get_signal(signal_id) is None:
            raise HTTPException(status_code=404, detail="signal not found")
        samples = rt.spine.recent(signal_id, limit=window)
        if len(samples) < 2:
            return {"status": "cold", "signal_id": signal_id, "n": len(samples)}
        # Use a throwaway drift-slope arm — same regression, no interference
        # with persistent per-signal arm state.
        tmp = DriftSlopeArm(window=len(samples))
        tmp.update(samples)
        slope_info = tmp.time_to_cross(samples[-1].value)  # threshold==current → eta=0; used for slope
        slope = float(slope_info.get("slope", 0.0))
        first = samples[0].value
        last = samples[-1].value
        magnitude = last - first
        direction = "rising" if slope > 0 else "falling" if slope < 0 else "flat"
        # Confidence from normalized R-like measure: 1 - resid/|range|.
        values = [s.value for s in samples]
        v_range = max(values) - min(values) or 1.0
        # crude residual proxy: stdev of first differences / range
        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        noise = (sum(d * d for d in diffs) / len(diffs)) ** 0.5 if diffs else 0.0
        confidence = max(0.0, min(1.0, 1.0 - noise / v_range))
        return {
            "signal_id": signal_id,
            "n": len(samples),
            "direction": direction,
            "slope": slope,
            "magnitude": magnitude,
            "first_value": first,
            "last_value": last,
            "confidence": confidence,
            "window_seconds": samples[-1].ts - samples[0].ts,
        }

    class CrossRequest(BaseModel):
        signal_id: str
        threshold: float

    @router.post("/forecasting/cross")
    async def cross(req: CrossRequest) -> dict[str, Any]:
        if rt.spine.get_signal(req.signal_id) is None:
            raise HTTPException(status_code=404, detail="signal not found")
        # Use the persistent drift_slope arm if it's registered for this signal
        # so the result reflects the same rolling window the continuous loop uses.
        arm = rt.octopus.get_arm(req.signal_id, "drift_slope")
        if not isinstance(arm, DriftSlopeArm) or not arm.ready():
            # Fall back to a one-shot fit.
            samples = rt.spine.recent(req.signal_id, limit=128)
            if len(samples) < 2:
                return {"status": "cold", "signal_id": req.signal_id}
            arm = DriftSlopeArm(window=len(samples))
            arm.update(samples)
        result = arm.time_to_cross(req.threshold)
        result["signal_id"] = req.signal_id
        return result

    @router.get("/forecasting/health/{signal_id}")
    async def health(signal_id: str) -> dict[str, Any]:
        return {
            "signal_id": signal_id,
            "health": {
                name: {
                    "status": h.status,
                    "last_error": h.last_error,
                    "last_mse": h.last_mse,
                    "consecutive_failures": h.consecutive_failures,
                    "last_update_ts": h.last_update_ts,
                }
                for name, h in rt.octopus.health(signal_id).items()
            },
        }

    return router


# ---- helpers ------------------------------------------------------------

def _parse_rows(raw: bytes, filename: str, ts_column: str) -> list[tuple[float, dict[str, float]]]:
    if filename.endswith(".jsonl") or filename.endswith(".ndjson"):
        return _parse_jsonl(raw, ts_column)
    # Default: CSV.
    return _parse_csv(raw, ts_column)


def _parse_csv(raw: bytes, ts_column: str) -> list[tuple[float, dict[str, float]]]:
    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        return []
    reader = csv.DictReader(io.StringIO(text))
    out: list[tuple[float, dict[str, float]]] = []
    for row in reader:
        ts = _coerce_ts(row.get(ts_column))
        if ts is None:
            continue
        values: dict[str, float] = {}
        for k, v in row.items():
            if k == ts_column or k is None:
                continue
            try:
                values[k] = float(v)
            except (TypeError, ValueError):
                continue
        if values:
            out.append((ts, values))
    return out


def _parse_jsonl(raw: bytes, ts_column: str) -> list[tuple[float, dict[str, float]]]:
    out: list[tuple[float, dict[str, float]]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        ts = _coerce_ts(obj.get(ts_column))
        if ts is None:
            continue
        values: dict[str, float] = {}
        for k, v in obj.items():
            if k == ts_column:
                continue
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values[k] = float(v)
        if values:
            out.append((ts, values))
    return out


def _coerce_ts(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    # ISO-ish fallback.
    try:
        from datetime import datetime

        s = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None
