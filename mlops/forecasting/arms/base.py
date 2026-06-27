from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..spine import Sample


@dataclass(frozen=True)
class ProjectionPoint:
    ts_target: float
    value: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class Projection:
    arm_name: str
    ts_issued: float
    points: list[ProjectionPoint]
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)


class ForecasterArm:
    """Base interface. Each arm owns its own state and updates online."""

    name: str = "base"

    def update(self, samples: list[Sample]) -> None:
        raise NotImplementedError

    def project(self, *, horizon_steps: int, step_dt: float, ts_now: float) -> Projection:
        raise NotImplementedError

    def state(self) -> dict[str, Any]:
        return {}

    def load(self, state: dict[str, Any]) -> None:
        return None

    def ready(self) -> bool:
        """True when the arm has enough data to produce meaningful projections."""
        return True
