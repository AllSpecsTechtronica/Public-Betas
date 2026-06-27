from __future__ import annotations

import math
import time
from typing import Any

from ..spine import Sample
from .base import ForecasterArm, Projection, ProjectionPoint


class EwmaArm(ForecasterArm):
    """Holt's linear (double-exponential) smoothing with online update.

    Level + trend, no seasonality. Cheap, O(1) per sample. Serves as the
    always-available baseline when heavier arms are still fitting or have failed.
    """

    name = "ewma"

    def __init__(self, *, alpha: float = 0.35, beta: float = 0.12, ci_k: float = 1.96) -> None:
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._ci_k = float(ci_k)
        self._level: float | None = None
        self._trend: float = 0.0
        self._last_ts: float | None = None
        self._resid_sq_sum: float = 0.0
        self._resid_count: int = 0
        self._seen: int = 0

    def update(self, samples: list[Sample]) -> None:
        for s in samples:
            v = float(s.value)
            if self._level is None:
                self._level = v
                self._trend = 0.0
                self._last_ts = float(s.ts)
                self._seen = 1
                continue
            forecast = self._level + self._trend
            resid = v - forecast
            self._resid_sq_sum += resid * resid
            self._resid_count += 1
            prev_level = self._level
            self._level = self._alpha * v + (1.0 - self._alpha) * forecast
            self._trend = self._beta * (self._level - prev_level) + (1.0 - self._beta) * self._trend
            self._last_ts = float(s.ts)
            self._seen += 1

    def project(self, *, horizon_steps: int, step_dt: float, ts_now: float) -> Projection:
        issued = float(ts_now or time.time())
        if self._level is None:
            return Projection(arm_name=self.name, ts_issued=issued, points=[], status="cold")
        sigma = self._residual_sigma()
        points: list[ProjectionPoint] = []
        for i in range(1, int(max(1, horizon_steps)) + 1):
            y = self._level + self._trend * i
            # CI widens with horizon (random-walk approximation).
            width = self._ci_k * sigma * math.sqrt(i)
            points.append(
                ProjectionPoint(
                    ts_target=issued + i * float(step_dt),
                    value=y,
                    ci_low=y - width,
                    ci_high=y + width,
                )
            )
        return Projection(
            arm_name=self.name,
            ts_issued=issued,
            points=points,
            status="ok",
            metadata={
                "level": self._level,
                "trend": self._trend,
                "sigma": sigma,
                "samples_seen": self._seen,
            },
        )

    def ready(self) -> bool:
        return self._level is not None and self._seen >= 2

    def state(self) -> dict[str, Any]:
        return {
            "alpha": self._alpha,
            "beta": self._beta,
            "ci_k": self._ci_k,
            "level": self._level,
            "trend": self._trend,
            "last_ts": self._last_ts,
            "resid_sq_sum": self._resid_sq_sum,
            "resid_count": self._resid_count,
            "seen": self._seen,
        }

    def load(self, state: dict[str, Any]) -> None:
        self._alpha = float(state.get("alpha", self._alpha))
        self._beta = float(state.get("beta", self._beta))
        self._ci_k = float(state.get("ci_k", self._ci_k))
        self._level = state.get("level")
        self._trend = float(state.get("trend", 0.0))
        self._last_ts = state.get("last_ts")
        self._resid_sq_sum = float(state.get("resid_sq_sum", 0.0))
        self._resid_count = int(state.get("resid_count", 0))
        self._seen = int(state.get("seen", 0))

    def _residual_sigma(self) -> float:
        if self._resid_count <= 0:
            return 0.0
        return math.sqrt(self._resid_sq_sum / float(self._resid_count))
