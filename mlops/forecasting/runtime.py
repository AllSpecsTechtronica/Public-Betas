from __future__ import annotations

import threading
from typing import Callable, Optional

from .arms.base import ForecasterArm
from .arms.drift_slope import DriftSlopeArm
from .arms.ewma import EwmaArm
from .continuous import ContinuousForecaster
from .octopus import OctopusForecaster
from .spine import SignalSpine, get_spine


def default_arm_factories() -> list[Callable[[], ForecasterArm]]:
    """Baseline arms — always available, no heavy deps."""
    return [lambda: EwmaArm(), lambda: DriftSlopeArm()]


def optional_arm_factories() -> list[Callable[[], ForecasterArm]]:
    """Heavy arms — only loaded if their dependencies are installed."""
    out: list[Callable[[], ForecasterArm]] = []
    try:
        from .arms.arima import ArimaArm  # type: ignore[attr-defined]

        out.append(lambda: ArimaArm())
    except Exception:
        pass
    try:
        from .arms.lstm import LstmArm  # type: ignore[attr-defined]

        out.append(lambda: LstmArm())
    except Exception:
        pass
    return out


class ForecastingRuntime:
    """Bundles spine + octopus + continuous loop. One per process."""

    def __init__(
        self,
        *,
        spine: Optional[SignalSpine] = None,
        arm_factories: Optional[list[Callable[[], ForecasterArm]]] = None,
        horizon_steps: int = 12,
        step_dt: float = 1.0,
    ) -> None:
        self.spine = spine or get_spine()
        factories = arm_factories if arm_factories is not None else (
            default_arm_factories() + optional_arm_factories()
        )
        self.octopus = OctopusForecaster(
            spine=self.spine,
            arm_factories=factories,
            horizon_steps=horizon_steps,
            step_dt=step_dt,
        )
        self.continuous = ContinuousForecaster(spine=self.spine, octopus=self.octopus)
        # Default cadence: heavy arms rate-limited. Cheap arms are essentially
        # free, so we only rate-limit the whole tick if heavy arms are present.
        if any("lstm" in f.__qualname__.lower() or "arima" in f.__qualname__.lower()
               for f in factories):
            # applied per-signal as it's registered
            self._default_min_interval = 1.0
        else:
            self._default_min_interval = 0.0
        self._started = False
        self._lock = threading.Lock()

    def register_signal(self, signal_id: str, *, min_interval: float | None = None) -> None:
        self.octopus.register_signal(signal_id)
        interval = self._default_min_interval if min_interval is None else float(min_interval)
        self.continuous.set_cadence(signal_id, min_interval=interval)

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self.continuous.start_in_thread()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            self.continuous.stop()
            self._started = False


_DEFAULT_RUNTIME: Optional[ForecastingRuntime] = None
_DEFAULT_LOCK = threading.Lock()


def get_runtime() -> ForecastingRuntime:
    global _DEFAULT_RUNTIME
    with _DEFAULT_LOCK:
        if _DEFAULT_RUNTIME is None:
            _DEFAULT_RUNTIME = ForecastingRuntime()
        return _DEFAULT_RUNTIME
