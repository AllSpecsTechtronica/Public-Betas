"""Global event pulse strip — always-visible, scrolling monospace event log.

Subscribes to CvOpsMemoryClient.rawEvent and renders a fixed-height horizontal
strip at the bottom of CvOpsWindow showing the last N events color-coded by
type. Gives the scientist a continuous sense of system breathing without
requiring any panel focus.

Color semantics (strict — nothing else should use these values):
    Lime    #C5FF46  system heartbeat        service alive / breathing
    Cyan    #2fa4b5  active system process   job events
    White   #c8c8c8  neutral system event    scenario / dataset events
    Red     #d4381e  human attention needed  error / failed states
    Amber   #e8b84b  degraded-operational    warning / cancelled (Aurora: cancelled uses alert red)
    Iron    #6a6a6a  low-signal              all other / unknown
"""
from __future__ import annotations

from datetime import datetime
from collections import deque
from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ...ui.theme import current_color_scheme, is_aurora_family_scheme

from .cvops_theme import cvops_color
from .time_format import format_clock_timestamp, time_pattern

_MAX_EVENTS = 200
_STRIP_HEIGHT = 28
_SEPARATOR = "  |  "
_NOTIFICATION_RED = "#dc322f"
_NOTIFICATION_GREEN = "#859900"

_TYPE_COLOR_ROLE: dict[str, str] = {
    "heartbeat":         _NOTIFICATION_GREEN,
    "job_status":        _NOTIFICATION_GREEN,
    "job_result":        _NOTIFICATION_GREEN,
    "training_progress": _NOTIFICATION_GREEN,
    "cell_progress":     _NOTIFICATION_GREEN,
    "scenario_updated":  "text_signal",
    "dataset_event":     "text_signal",
    "socket_state":      "text_signal",
    "toast":             "text_signal",
}


def _event_color_role(payload: dict[str, Any]) -> str:
    mtype = str(payload.get("type") or "")
    state = str(
        payload.get("state")
        or payload.get("status")
        or payload.get("cell_status")
        or payload.get("event")
        or ""
    ).strip().lower()

    if mtype.endswith("_error") or mtype == "local_error":
        return _NOTIFICATION_RED
    if state in {"error", "failed"}:
        return _NOTIFICATION_RED
    if is_aurora_family_scheme(current_color_scheme()) and state in {"canceled", "cancelled"}:
        return _NOTIFICATION_RED
    if state in {"cancelled", "canceled", "warning", "warn", "degraded"}:
        return _NOTIFICATION_RED
    if state in {
        "ok",
        "ready",
        "healthy",
        "live",
        "connected",
        "open",
        "queued",
        "running",
        "done",
        "completed",
        "complete",
        "succeeded",
        "success",
    }:
        return _NOTIFICATION_GREEN
    if mtype.startswith("asset_") or mtype.startswith("sector_"):
        return "text_signal"
    return _TYPE_COLOR_ROLE.get(mtype, "text_iron")


def _event_color(payload: dict[str, Any]) -> str:
    """Compatibility helper used by the grouped notifications panel."""
    return cvops_color(_event_color_role(payload))


def _event_timestamp(payload: dict[str, Any]) -> str:
    raw = payload.get("emitted_at")
    if raw is None:
        raw = payload.get("timestamp")
    if raw is None:
        raw = payload.get("t")
    shown = format_clock_timestamp(raw, seconds=True, empty="")
    if shown:
        return shown
    return datetime.now().strftime(time_pattern(seconds=True))


def _event_scope(payload: dict[str, Any]) -> str:
    nested = payload.get("payload")
    payload_body = nested if isinstance(nested, dict) else {}
    nested_status = payload.get("status_payload")
    status_body = nested_status if isinstance(nested_status, dict) else {}
    candidates = (
        payload.get("scope"),
        payload.get("scenario"),
        payload.get("job_id"),
        payload.get("name"),
        payload_body.get("name"),
        status_body.get("name"),
        payload.get("asset_id"),
        payload_body.get("asset_id"),
        payload.get("sector_path"),
        payload_body.get("sector_path"),
        payload.get("sector_id"),
        payload_body.get("sector_id"),
        payload.get("service"),
        payload.get("source"),
    )
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _event_summary(payload: dict[str, Any]) -> str:
    nested_payload = payload.get("payload")
    payload_body = nested_payload if isinstance(nested_payload, dict) else {}
    nested_status = payload.get("status_payload")
    status_body = nested_status if isinstance(nested_status, dict) else {}
    nested_result = payload.get("result")
    result_body = nested_result if isinstance(nested_result, dict) else {}

    mtype = str(payload.get("type") or "unknown").strip().lower()
    state = str(
        payload.get("state")
        or payload.get("status")
        or status_body.get("status")
        or payload.get("cell_status")
        or ""
    ).strip()

    if mtype == "heartbeat":
        return (
            f"{int(payload.get('queued') or 0)} queued / "
            f"{int(payload.get('running') or 0)} running / "
            f"{int(payload.get('done') or 0)} done / "
            f"{int(payload.get('error') or 0)} error"
        )

    summary_keys = ("message", "error", "detail", "summary", "note")
    for key in summary_keys:
        for body in (payload, result_body, payload_body, status_body):
            text = str(body.get(key) or "").strip() if isinstance(body, dict) else ""
            if text:
                return text

    cell_name = str(payload.get("cell_name") or "").strip()
    if cell_name and state:
        return f"{cell_name} -> {state}"
    if cell_name:
        return cell_name

    event_name = str(payload.get("event") or "").strip()
    if event_name and state and event_name.lower() != state.lower():
        return f"{event_name} -> {state}"
    if event_name:
        return event_name

    name = str(payload.get("name") or payload_body.get("name") or status_body.get("name") or "").strip()
    if name and state:
        return f"{name} -> {state}"
    if name:
        return name

    if state:
        return state

    progress = payload.get("progress")
    if progress is not None:
        try:
            return f"{float(progress) * 100:.0f}%"
        except Exception:
            return ""
    return ""


def _event_label(payload: dict[str, Any]) -> str:
    mtype = str(payload.get("type") or "unknown")
    ts = _event_timestamp(payload)
    scope = _event_scope(payload)
    summary = _event_summary(payload)
    parts = [f"[{ts}]", f"[{mtype.upper()[:12]}]"]
    if scope:
        parts.append(scope)
    if summary:
        parts.append(f":: {summary}")

    return " ".join(parts)


class EventPulseWidget(QWidget):
    """Fixed-height horizontal event log strip.

    Usage::

        pulse = EventPulseWidget(parent=self)
        self._ws.rawEvent.connect(pulse.ingest)
        layout.addWidget(pulse)

    The strip auto-scrolls right to newest; hovering pauses scrolling.
    Clicking the strip emits ``eventClicked(dict)`` with the payload.
    Double-clicking the strip emits ``openNotificationsRequested``.
    """

    eventClicked = pyqtSignal(dict)
    openNotificationsRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_STRIP_HEIGHT)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setObjectName("eventPulse")

        self._events: deque[tuple[str, str, dict[str, Any]]] = deque(maxlen=_MAX_EVENTS)
        # (label, dynamic_color_role, raw_payload)

        self._scroll_offset = 0.0
        self._paused = False
        self._total_text_width = 0

        font = QFont("JetBrains Mono, Courier New, monospace")
        font.setPixelSize(11)
        self.setFont(font)

        self._tick = QTimer(self)
        self._tick.setInterval(50)
        self._tick.timeout.connect(self._auto_scroll)
        self._tick.start()

        self.setMouseTracking(True)

    # ------------------------------------------------------------------

    def ingest(self, payload: dict[str, Any]) -> None:
        label = _event_label(payload)
        color_role = _event_color_role(payload)
        self._events.append((label, color_role, payload))
        self._rebuild_text_width()
        self.update()

    def _rebuild_text_width(self) -> None:
        fm = self.fontMetrics()
        items = list(self._events)
        total = sum(fm.horizontalAdvance(lbl + _SEPARATOR) for lbl, _, _ in items)
        self._total_text_width = total

    def _auto_scroll(self) -> None:
        if self._paused:
            return
        w = self.width()
        max_offset = max(0, self._total_text_width - w + 20)
        if self._scroll_offset < max_offset:
            self._scroll_offset += 1.0
            self.update()

    # ------------------------------------------------------------------
    # Qt overrides

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(cvops_color("bg_void")))

        fm = painter.fontMetrics()
        # Top-align so the strip does not read as a half-empty bar under the log line.
        baseline_y = fm.ascent() + 3

        x = int(4 - self._scroll_offset)
        painter.setFont(self.font())

        for label, color_role, _ in self._events:
            painter.setPen(QColor(cvops_color(color_role)))
            painter.drawText(x, baseline_y, label)
            x += fm.horizontalAdvance(label)

            painter.setPen(QColor(cvops_color("line_light")))
            painter.drawText(x, baseline_y, _SEPARATOR)
            x += fm.horizontalAdvance(_SEPARATOR)

            if x > self.width() + 200:
                break

        painter.end()

    def mouseMoveEvent(self, _event) -> None:  # type: ignore[override]
        self._paused = True

    def leaveEvent(self, _event) -> None:  # type: ignore[override]
        self._paused = False

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        x_click = event.pos().x() + self._scroll_offset - 4
        fm = self.fontMetrics()
        cursor = 0.0
        for label, _color, payload in self._events:
            w = fm.horizontalAdvance(label + _SEPARATOR)
            if cursor <= x_click < cursor + w:
                self.eventClicked.emit(payload)
                break
            cursor += w
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        self.openNotificationsRequested.emit()
        super().mouseDoubleClickEvent(event)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._rebuild_text_width()
