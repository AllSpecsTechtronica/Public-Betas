from __future__ import annotations

import math
import time
from collections import deque
from typing import Any, Deque

from ..spine import Sample
from .base import ForecasterArm, Projection, ProjectionPoint


class DriftSlopeArm(ForecasterArm):
    """Linear regression over a rolling window. Specialized for threshold-cross
    prediction: exposes time-to-cross in metadata for direct consumption by
    monitoring.py's drift early-warning path."""

    name = "drift_slope"

    def __init__(self, *, window: int = 64, ci_k: float = 1.96) -> None:
        self._window = int(max(4, window))
        self._ci_k = float(ci_k)
        self._buf: Deque[Sample] = deque(maxlen=self._window)

    def update(self, samples: list[Sample]) -> None:
        for s in samples:
            self._buf.append(s)

    def project(self, *, horizon_steps: int, step_dt: float, ts_now: float) -> Projection:
        issued = float(ts_now or time.time())
        if len(self._buf) < 2:
            return Projection(arm_name=self.name, ts_issued=issued, points=[], status="cold")
        slope, intercept, sigma = self._fit()
        points: list[ProjectionPoint] = []
        for i in range(1, int(max(1, horizon_steps)) + 1):
            target = issued + i * float(step_dt)
            y = intercept + slope * target
            width = self._ci_k * sigma * math.sqrt(i)
            points.append(
                ProjectionPoint(ts_target=target, value=y, ci_low=y - width, ci_high=y + width)
            )
        return Projection(
            arm_name=self.name,
            ts_issued=issued,
            points=points,
            status="ok",
            metadata={
                "slope": slope,
                "intercept": intercept,
                "sigma": sigma,
                "window_size": len(self._buf),
            },
        )

    def ready(self) -> bool:
        return len(self._buf) >= 2

    def time_to_cross(self, threshold: float, *, ts_now: float | None = None) -> dict[str, Any]:
        """Estimate seconds-to-cross for the current slope. Returns direction
        and confidence band. This is the drift early-warning surface."""
        now = float(ts_now or time.time())
        if len(self._buf) < 2:
            return {"status": "cold"}
        slope, intercept, _sigma = self._fit()
        # Use the most recent value as the anchor rather than intercept (which
        # could extrapolate across a long baseline).
        last = self._buf[-1]
        if abs(slope) < 1e-12:
            return {
                "status": "flat",
                "current_value": float(last.value),
                "slope": slope,
                "threshold": float(threshold),
            }
        eta_seconds = (float(threshold) - float(last.value)) / slope
        direction = "rising" if slope > 0 else "falling"
        will_cross = eta_seconds >= 0
        return {
            "status": "ok",
            "current_value": float(last.value),
            "slope": float(slope),
            "threshold": float(threshold),
            "eta_seconds": float(eta_seconds),
            "ts_cross": float(now + eta_seconds) if will_cross else None,
            "direction": direction,
            "will_cross": bool(will_cross),
        }

    def state(self) -> dict[str, Any]:
        return {
            "window": self._window,
            "ci_k": self._ci_k,
            "buf": [(s.signal_id, s.ts, s.value, s.source) for s in self._buf],
        }

    def load(self, state: dict[str, Any]) -> None:
        self._window = int(state.get("window", self._window))
        self._ci_k = float(state.get("ci_k", self._ci_k))
        self._buf = deque(maxlen=self._window)
        for (sid, ts, val, src) in state.get("buf", []):
            self._buf.append(Sample(signal_id=sid, ts=float(ts), value=float(val), source=src or ""))

    def _fit(self) -> tuple[float, float, float]:
        n = len(self._buf)
        if n < 2:
            return 0.0, 0.0, 0.0
        xs = [float(s.ts) for s in self._buf]
        ys = [float(s.value) for s in self._buf]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
        den = sum((x - mean_x) ** 2 for x in xs)
        if den <= 0:
            return 0.0, mean_y, 0.0
        slope = num / den
        intercept = mean_y - slope * mean_x
        resid_sq = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
        sigma = math.sqrt(resid_sq / max(1, n - 2))
        return slope, intercept, sigma
