"""Live training verdict + post-training accuracy forecast.

Pure consumer of the per-epoch points already emitted by `train.py` —
does not touch the training loop itself. Two surfaces:

* `TrainingVerdict` — fed each epoch point, returns a state-machine label
  (`improving`, `good`, `regressing`, `failing`, `absolute_loss`, `pending`).
* `forecast_run` — given the full epoch history, returns the predicted
  production mAP50 with a confidence band, reliability tier, and dominant
  risk mode.

Thresholds are hardcoded conservative defaults tuned for YOLO-shaped runs.
They are normalized by the first epoch's losses so they survive scale
differences across model sizes.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Optional

# Window size for slope calculations. Five epochs is enough to smooth YOLO's
# noisy per-epoch jitter without lagging too far behind a real regression.
WINDOW = 5

# Slope thresholds (per-epoch fractional change of the normalized signal).
# A val_slope of +0.005 means val_loss is rising ~0.5% of its initial value
# per epoch over the window — a real upward trend, not noise.
FLAT_BAND = 0.003
SLOPE_EPS = 0.005

# Divergence: train_loss has climbed back above its starting value by 5%.
DIVERGE_RATIO = 1.05

# Convergence: val_loss within this fraction of its running minimum AND
# the minimum has been stable for at least this many epochs.
CONVERGED_EPS = 0.02
CONVERGED_AGE = 3

# Forecast: how many trailing epochs feed the production estimate.
FORECAST_TAIL = 5

# Generalization-gap penalty cap (mAP50 points).
MAX_GAP_PENALTY = 0.15


def _is_finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _slope(values: list[float]) -> float:
    """Least-squares slope of `values` against epoch index 0..n-1."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((xs[i] - mean_x) * (values[i] - mean_y) for i in range(n))
    den = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if den == 0:
        return 0.0
    return num / den


def _stdev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    m = sum(values) / n
    return math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))


class TrainingVerdict:
    """Rolling-window state machine over per-epoch training points."""

    def __init__(self, *, window: int = WINDOW) -> None:
        self._window = max(2, int(window))
        self._history: list[dict[str, Any]] = []
        self._initial_train_loss: Optional[float] = None
        self._initial_val_loss: Optional[float] = None
        self._val_min: Optional[float] = None
        self._val_min_epoch: int = -1
        self._diverged: bool = False
        self._last_label: str = "pending"
        self._last_reason: str = "warming up"

    def history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def update(self, point: dict[str, Any]) -> dict[str, Any]:
        """Feed an epoch point. Returns a verdict dict."""
        self._history.append(dict(point))

        train_loss = point.get("train_loss")
        val_loss = point.get("val_loss")
        epoch = int(point.get("epoch") or 0)

        if not _is_finite(train_loss) or not _is_finite(val_loss):
            # Either truly NaN/Inf (divergence) or just unreported this epoch.
            train_raw = point.get("train_loss")
            val_raw = point.get("val_loss")
            train_nan = train_raw is not None and not _is_finite(train_raw)
            val_nan = val_raw is not None and not _is_finite(val_raw)
            if train_nan or val_nan:
                self._diverged = True
                return self._set("absolute_loss", "non-finite loss detected", point, epoch)
            return self._set(self._last_label, self._last_reason, point, epoch)

        tl = float(train_loss)
        vl = float(val_loss)
        if self._initial_train_loss is None and tl > 0:
            self._initial_train_loss = tl
        if self._initial_val_loss is None and vl > 0:
            self._initial_val_loss = vl
        if self._val_min is None or vl < self._val_min:
            self._val_min = vl
            self._val_min_epoch = epoch

        if self._diverged:
            return self._set("absolute_loss", "diverged earlier in run", point, epoch)
        if (
            self._initial_train_loss is not None
            and epoch >= 3
            and tl >= self._initial_train_loss * DIVERGE_RATIO
        ):
            self._diverged = True
            return self._set(
                "absolute_loss",
                f"train_loss {tl:.3f} exceeded {DIVERGE_RATIO:.2f}x initial",
                point,
                epoch,
            )

        # Need a full window to call regressing/failing/converged.
        recent = self._history[-self._window :]
        if len(recent) < self._window:
            label = "improving" if vl < (self._val_min or vl) + 1e-9 else "pending"
            return self._set(label, f"window {len(recent)}/{self._window}", point, epoch)

        train_series = [float(p.get("train_loss") or 0.0) for p in recent]
        val_series = [float(p.get("val_loss") or 0.0) for p in recent]
        map_series = [float(p.get("map50")) for p in recent if _is_finite(p.get("map50"))]

        scale = float(self._initial_train_loss or 1.0) or 1.0
        t_slope = _slope(train_series) / scale
        v_slope = _slope(val_series) / scale
        m_slope = _slope(map_series) if len(map_series) >= 2 else 0.0

        # Regressing first: val getting worse (val_loss up, or map50 down).
        # This must beat both `good` and `failing` — a model that's making val
        # worse is not converged and not merely stuck.
        if v_slope > SLOPE_EPS or (len(map_series) >= 2 and m_slope < -SLOPE_EPS):
            why = (
                f"val_slope {v_slope:+.4f}"
                if v_slope > SLOPE_EPS
                else f"map50_slope {m_slope:+.4f}"
            )
            return self._set(
                "regressing", why, point, epoch,
                t_slope=t_slope, v_slope=v_slope, m_slope=m_slope,
            )

        # Converged before failing: a plateau AT the running minimum is `good`,
        # not `failing`. Failing is reserved for plateaus that never reached a
        # useful loss floor — so require real descent from the initial val_loss.
        val_min = self._val_min if self._val_min is not None else vl
        near_min = abs(vl - val_min) <= max(CONVERGED_EPS * abs(val_min), 1e-6)
        age = epoch - self._val_min_epoch
        initial_val = self._initial_val_loss or vl
        descended = initial_val > 0 and val_min <= initial_val * 0.95
        if abs(v_slope) < FLAT_BAND and near_min and age >= CONVERGED_AGE and descended:
            return self._set(
                "good",
                f"converged (val within {CONVERGED_EPS*100:.0f}% of min, age {age})",
                point, epoch,
                t_slope=t_slope, v_slope=v_slope, m_slope=m_slope,
            )

        # Failing: train_loss not falling at all (or rising) across the window,
        # and we're not at a useful plateau.
        if epoch >= 3 and t_slope >= -FLAT_BAND:
            return self._set(
                "failing",
                f"train_slope {t_slope:+.4f} (loss not decreasing)",
                point,
                epoch,
                t_slope=t_slope,
                v_slope=v_slope,
                m_slope=m_slope,
            )

        # Improving: val trending down (or map50 trending up).
        if v_slope < -SLOPE_EPS or (len(map_series) >= 2 and m_slope > SLOPE_EPS):
            why = (
                f"val_slope {v_slope:+.4f}"
                if v_slope < -SLOPE_EPS
                else f"map50_slope {m_slope:+.4f}"
            )
            return self._set(
                "improving", why, point, epoch,
                t_slope=t_slope, v_slope=v_slope, m_slope=m_slope,
            )

        # In-band, not yet converged: hold previous label or call it improving.
        return self._set(
            "improving", "slow progress", point, epoch,
            t_slope=t_slope, v_slope=v_slope, m_slope=m_slope,
        )

    def _set(
        self,
        label: str,
        reason: str,
        point: dict[str, Any],
        epoch: int,
        **extras: float,
    ) -> dict[str, Any]:
        self._last_label = label
        self._last_reason = reason
        out: dict[str, Any] = {
            "event": "verdict",
            "label": label,
            "reason": reason,
            "epoch": epoch,
            "epochs": int(point.get("epochs") or 0),
            "train_loss": point.get("train_loss"),
            "val_loss": point.get("val_loss"),
            "map50": point.get("map50"),
            "val_min": self._val_min,
        }
        out.update({k: round(v, 6) for k, v in extras.items()})
        return out


def forecast_run(history: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Predict production performance from the completed epoch history."""
    points = [p for p in history if isinstance(p, dict)]
    if not points:
        return {
            "event": "forecast",
            "expected_map50": None,
            "reliability": "UNKNOWN",
            "risk": "no data",
        }

    tail = points[-FORECAST_TAIL:]
    val_maps = [float(p["map50"]) for p in tail if _is_finite(p.get("map50"))]
    last = points[-1]
    train_loss = float(last.get("train_loss") or 0.0)
    val_loss = float(last.get("val_loss") or 0.0)

    val_map_mean = sum(val_maps) / len(val_maps) if val_maps else 0.0
    val_map_std = _stdev(val_maps) if len(val_maps) >= 2 else 0.0

    if val_loss > 0:
        gap = max(0.0, val_loss - train_loss) / val_loss
    else:
        gap = 0.0
    gap_penalty = min(MAX_GAP_PENALTY, gap * 0.3)
    expected = max(0.0, val_map_mean - gap_penalty)
    ci = 1.96 * val_map_std

    if val_map_std < 0.01 and gap < 0.10:
        reliability = "HIGH"
    elif val_map_std < 0.03 and gap < 0.25:
        reliability = "MEDIUM"
    else:
        reliability = "LOW"

    if not _is_finite(train_loss) or not _is_finite(val_loss):
        risk = "diverged"
    elif gap > 0.30:
        risk = "overfit"
    elif val_map_mean < 0.30:
        risk = "underfit or data-starved"
    elif val_map_std > 0.05:
        risk = "unstable"
    else:
        risk = "none"

    return {
        "event": "forecast",
        "expected_map50": round(expected, 4),
        "ci_half_width": round(ci, 4),
        "reliability": reliability,
        "risk": risk,
        "val_map50_final": round(val_maps[-1], 4) if val_maps else None,
        "val_map50_mean_tail": round(val_map_mean, 4),
        "val_map50_std_tail": round(val_map_std, 4),
        "train_val_gap": round(gap, 4),
        "gap_penalty": round(gap_penalty, 4),
        "tail_window": len(val_maps),
    }


def render_forecast(fc: dict[str, Any]) -> str:
    """One-line + multi-line summary suitable for stdout."""
    em = fc.get("expected_map50")
    ci = fc.get("ci_half_width") or 0.0
    rel = fc.get("reliability") or "UNKNOWN"
    risk = fc.get("risk") or "none"
    head = (
        f"[FORECAST] expected mAP50 {em:.3f} +/-{ci:.3f} | reliability {rel} | risk: {risk}"
        if isinstance(em, (int, float))
        else f"[FORECAST] expected mAP50 unknown | reliability {rel} | risk: {risk}"
    )
    body = (
        f"          tail window={fc.get('tail_window')}  "
        f"val_map50_mean={fc.get('val_map50_mean_tail')}  "
        f"val_map50_std={fc.get('val_map50_std_tail')}  "
        f"train_val_gap={fc.get('train_val_gap')}  "
        f"gap_penalty={fc.get('gap_penalty')}"
    )
    return head + "\n" + body
