from __future__ import annotations

import asyncio
import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .arms.base import ForecasterArm, Projection, ProjectionPoint
from .spine import Sample, SignalSpine


@dataclass
class ArmHealth:
    name: str
    status: str = "ok"  # ok | cold | failed
    last_error: str = ""
    last_mse: float | None = None
    consecutive_failures: int = 0
    last_update_ts: float = 0.0


@dataclass
class CompositeProjection:
    signal_id: str
    ts_issued: float
    horizon_steps: int
    step_dt: float
    per_arm: dict[str, Projection] = field(default_factory=dict)
    ensemble: list[ProjectionPoint] = field(default_factory=list)
    arm_weights: dict[str, float] = field(default_factory=dict)
    health: dict[str, ArmHealth] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "ts_issued": self.ts_issued,
            "horizon_steps": self.horizon_steps,
            "step_dt": self.step_dt,
            "per_arm": {
                name: {
                    "status": p.status,
                    "points": [
                        {
                            "ts_target": pt.ts_target,
                            "value": pt.value,
                            "ci_low": pt.ci_low,
                            "ci_high": pt.ci_high,
                        }
                        for pt in p.points
                    ],
                    "metadata": p.metadata,
                }
                for name, p in self.per_arm.items()
            },
            "ensemble": [
                {
                    "ts_target": pt.ts_target,
                    "value": pt.value,
                    "ci_low": pt.ci_low,
                    "ci_high": pt.ci_high,
                }
                for pt in self.ensemble
            ],
            "arm_weights": self.arm_weights,
            "health": {
                name: {
                    "status": h.status,
                    "last_error": h.last_error,
                    "last_mse": h.last_mse,
                    "consecutive_failures": h.consecutive_failures,
                    "last_update_ts": h.last_update_ts,
                }
                for name, h in self.health.items()
            },
        }


class OctopusForecaster:
    """The head. Holds N arms per signal; updates and projects them in parallel.

    Fault isolation is load-bearing: one arm raising must not stop the others.
    Each signal owns an independent set of arm instances so arms can keep
    per-signal state without cross-talk.
    """

    def __init__(
        self,
        *,
        spine: SignalSpine,
        arm_factories: list[Callable[[], ForecasterArm]],
        horizon_steps: int = 12,
        step_dt: float = 1.0,
        mse_eval_window: int = 32,
    ) -> None:
        self._spine = spine
        self._arm_factories = list(arm_factories)
        self._horizon = int(horizon_steps)
        self._step_dt = float(step_dt)
        self._mse_window = int(mse_eval_window)
        self._lock = threading.RLock()
        # signal_id -> {arm_name: ForecasterArm}
        self._arms_by_signal: dict[str, dict[str, ForecasterArm]] = {}
        # signal_id -> {arm_name: ArmHealth}
        self._health_by_signal: dict[str, dict[str, ArmHealth]] = {}
        # signal_id -> last seen ts (to feed only unseen samples)
        self._last_seen_ts: dict[str, float] = {}
        # signal_id -> {arm_name: list[(ts_target, predicted)]} buffered for MSE
        self._pending_preds: dict[str, dict[str, list[tuple[float, float]]]] = {}

    def register_signal(self, signal_id: str) -> None:
        sid = str(signal_id or "").strip()
        if not sid:
            raise ValueError("signal_id required")
        with self._lock:
            if sid in self._arms_by_signal:
                return
            self._arms_by_signal[sid] = {}
            self._health_by_signal[sid] = {}
            self._pending_preds[sid] = {}
            for factory in self._arm_factories:
                arm = factory()
                self._arms_by_signal[sid][arm.name] = arm
                self._health_by_signal[sid][arm.name] = ArmHealth(name=arm.name, status="cold")
                self._pending_preds[sid][arm.name] = []

    def list_signals(self) -> list[str]:
        with self._lock:
            return list(self._arms_by_signal.keys())

    def arm_names(self, signal_id: str) -> list[str]:
        with self._lock:
            arms = self._arms_by_signal.get(signal_id, {})
            return list(arms.keys())

    def health(self, signal_id: str) -> dict[str, ArmHealth]:
        with self._lock:
            return dict(self._health_by_signal.get(signal_id, {}))

    def get_arm(self, signal_id: str, arm_name: str) -> ForecasterArm | None:
        with self._lock:
            return self._arms_by_signal.get(signal_id, {}).get(arm_name)

    # ---- parallel update + project --------------------------------------

    async def tick(self, signal_id: str, *, ts_now: float | None = None) -> CompositeProjection:
        """Pull new samples, update all arms in parallel, project all arms in parallel."""
        sid = str(signal_id or "").strip()
        if not sid:
            raise ValueError("signal_id required")
        self.register_signal(sid)
        ts = float(ts_now or time.time())

        new_samples = self._pull_new_samples(sid)
        self._score_previous_predictions(sid, new_samples)

        with self._lock:
            arms = dict(self._arms_by_signal[sid])

        # Update arms concurrently. Each update is wrapped so one arm's
        # failure cannot cascade.
        update_tasks = [self._safe_update(sid, name, arm, new_samples) for name, arm in arms.items()]
        if update_tasks:
            await asyncio.gather(*update_tasks, return_exceptions=False)

        # Project arms concurrently.
        project_tasks = [self._safe_project(sid, name, arm, ts) for name, arm in arms.items()]
        projections_list: list[tuple[str, Projection]] = []
        if project_tasks:
            projections_list = await asyncio.gather(*project_tasks, return_exceptions=False)

        per_arm: dict[str, Projection] = {name: proj for name, proj in projections_list}
        weights = self._compute_weights(sid, per_arm)
        ensemble = self._ensemble(per_arm, weights)

        self._buffer_predictions(sid, per_arm)
        self._persist(sid, ts, per_arm)

        return CompositeProjection(
            signal_id=sid,
            ts_issued=ts,
            horizon_steps=self._horizon,
            step_dt=self._step_dt,
            per_arm=per_arm,
            ensemble=ensemble,
            arm_weights=weights,
            health=self.health(sid),
        )

    # ---- internals ------------------------------------------------------

    def _pull_new_samples(self, sid: str) -> list[Sample]:
        with self._lock:
            last_ts = self._last_seen_ts.get(sid)
        # Always pull a bounded recent window; arms can ignore duplicates via ts.
        recent = self._spine.recent(sid, limit=self._mse_window * 8, since_ts=last_ts)
        if not recent:
            return []
        new_samples = [s for s in recent if last_ts is None or s.ts > last_ts]
        if new_samples:
            with self._lock:
                self._last_seen_ts[sid] = max(s.ts for s in new_samples)
        return new_samples

    async def _safe_update(
        self, sid: str, name: str, arm: ForecasterArm, samples: list[Sample]
    ) -> None:
        if not samples:
            return
        try:
            await asyncio.to_thread(arm.update, samples)
        except Exception as e:
            self._mark_failed(sid, name, f"update: {e!r}")

    async def _safe_project(
        self, sid: str, name: str, arm: ForecasterArm, ts_now: float
    ) -> tuple[str, Projection]:
        try:
            proj = await asyncio.to_thread(
                arm.project,
                horizon_steps=self._horizon,
                step_dt=self._step_dt,
                ts_now=ts_now,
            )
            self._mark_ok(sid, name)
            return name, proj
        except Exception as e:
            self._mark_failed(sid, name, f"project: {e!r}")
            return name, Projection(arm_name=name, ts_issued=ts_now, points=[], status="failed",
                                    metadata={"error": str(e)})

    def _mark_ok(self, sid: str, name: str) -> None:
        with self._lock:
            h = self._health_by_signal.setdefault(sid, {}).setdefault(name, ArmHealth(name=name))
            h.status = "ok"
            h.last_error = ""
            h.consecutive_failures = 0
            h.last_update_ts = time.time()

    def _mark_failed(self, sid: str, name: str, err: str) -> None:
        with self._lock:
            h = self._health_by_signal.setdefault(sid, {}).setdefault(name, ArmHealth(name=name))
            h.status = "failed"
            h.last_error = err
            h.consecutive_failures += 1
            h.last_update_ts = time.time()

    def _compute_weights(self, sid: str, per_arm: dict[str, Projection]) -> dict[str, float]:
        """Inverse-MSE weighting. Arms without a measured MSE get the median weight.
        Failed/cold arms are excluded. Weights sum to 1.0."""
        with self._lock:
            health = dict(self._health_by_signal.get(sid, {}))

        viable: list[str] = []
        mses: list[float] = []
        for name, proj in per_arm.items():
            if proj.status != "ok" or not proj.points:
                continue
            mse = health.get(name).last_mse if name in health else None
            viable.append(name)
            mses.append(mse if (mse is not None and mse >= 0) else float("nan"))

        if not viable:
            return {}

        known = [m for m in mses if not math.isnan(m)]
        if known:
            fallback = sorted(known)[len(known) // 2]
        else:
            fallback = 1.0

        raw: dict[str, float] = {}
        for name, mse in zip(viable, mses):
            m = fallback if math.isnan(mse) else mse
            raw[name] = 1.0 / (m + 1e-6)

        total = sum(raw.values())
        if total <= 0:
            n = len(viable)
            return {name: 1.0 / n for name in viable}
        return {name: w / total for name, w in raw.items()}

    def _ensemble(
        self,
        per_arm: dict[str, Projection],
        weights: dict[str, float],
    ) -> list[ProjectionPoint]:
        if not weights:
            return []
        viable = [(name, per_arm[name]) for name in weights if per_arm.get(name) and per_arm[name].points]
        if not viable:
            return []
        horizon = min(len(p.points) for _, p in viable)
        out: list[ProjectionPoint] = []
        for i in range(horizon):
            ts_target = viable[0][1].points[i].ts_target
            value = 0.0
            ci_low_sq = 0.0
            ci_high_sq = 0.0
            for name, proj in viable:
                w = weights[name]
                pt = proj.points[i]
                value += w * pt.value
                # Combine CIs by weighted RSS — treats arms as independent.
                ci_low_sq += (w * (pt.value - pt.ci_low)) ** 2
                ci_high_sq += (w * (pt.ci_high - pt.value)) ** 2
            out.append(
                ProjectionPoint(
                    ts_target=ts_target,
                    value=value,
                    ci_low=value - math.sqrt(ci_low_sq),
                    ci_high=value + math.sqrt(ci_high_sq),
                )
            )
        return out

    def _buffer_predictions(self, sid: str, per_arm: dict[str, Projection]) -> None:
        """Remember what each arm predicted so we can score it when reality catches up."""
        with self._lock:
            bucket = self._pending_preds.setdefault(sid, {})
            for name, proj in per_arm.items():
                if proj.status != "ok":
                    continue
                buf = bucket.setdefault(name, [])
                for pt in proj.points:
                    buf.append((pt.ts_target, pt.value))
                # Cap buffer length to avoid unbounded growth.
                cap = self._mse_window * 16
                if len(buf) > cap:
                    del buf[: len(buf) - cap]

    def _score_previous_predictions(self, sid: str, new_samples: list[Sample]) -> None:
        if not new_samples:
            return
        with self._lock:
            bucket = self._pending_preds.get(sid, {})
            for name, buf in bucket.items():
                if not buf:
                    continue
                # For each new observed sample, find the nearest buffered prediction
                # and collect the squared error.
                errs: list[float] = []
                remaining: list[tuple[float, float]] = []
                preds = list(buf)
                for sample in new_samples:
                    best_idx = -1
                    best_dt = float("inf")
                    for idx, (t, _v) in enumerate(preds):
                        dt = abs(t - sample.ts)
                        if dt < best_dt:
                            best_dt = dt
                            best_idx = idx
                    if best_idx >= 0 and best_dt < self._step_dt * 1.5:
                        _, pred_v = preds[best_idx]
                        errs.append((pred_v - sample.value) ** 2)
                        preds.pop(best_idx)
                # Keep only predictions still in the future (or within one step of now).
                now_cutoff = max(s.ts for s in new_samples)
                remaining = [(t, v) for (t, v) in preds if t >= now_cutoff - self._step_dt]
                bucket[name] = remaining

                if errs:
                    h = self._health_by_signal.setdefault(sid, {}).setdefault(
                        name, ArmHealth(name=name)
                    )
                    # Rolling MSE via exponential smoothing so recent performance dominates.
                    new_mse = sum(errs) / len(errs)
                    if h.last_mse is None:
                        h.last_mse = new_mse
                    else:
                        h.last_mse = 0.7 * h.last_mse + 0.3 * new_mse

    def _persist(self, sid: str, ts_issued: float, per_arm: dict[str, Projection]) -> None:
        for name, proj in per_arm.items():
            if not proj.points:
                continue
            points = [(p.ts_target, p.value, p.ci_low, p.ci_high) for p in proj.points]
            try:
                self._spine.write_projection(
                    signal_id=sid,
                    arm_name=name,
                    ts_issued=ts_issued,
                    points=points,
                    status=proj.status,
                    metadata=proj.metadata,
                )
            except Exception:
                pass
