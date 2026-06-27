from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from .cvops_theme import cvops_color, cvops_rgba
from .event_pulse_widget import _event_color_role, _event_scope, _event_summary, _event_timestamp

_CARD_ERROR = "#dc322f"
_CARD_SUCCESS = "#859900"
_CARD_PROCESS = "#2fa4b5"
_CARD_WARNING = "#e8b84b"


def _rgba_css(color: str, alpha: int) -> str:
    swatch = QColor(color if str(color).startswith("#") else cvops_color(color))
    return f"rgba({swatch.red()}, {swatch.green()}, {swatch.blue()}, {max(0, min(255, alpha))})"


def _pretty_event_type(payload: dict[str, Any]) -> str:
    raw = str(payload.get("type") or "unknown").strip().replace("-", "_")
    parts = [part for part in raw.split("_") if part]
    if not parts:
        return "Unknown"
    return " ".join(part.capitalize() for part in parts)


def _event_state(payload: dict[str, Any]) -> str:
    return str(
        payload.get("state")
        or payload.get("status")
        or payload.get("cell_status")
        or payload.get("event")
        or ""
    ).strip().lower()


def _notification_card_color_role(payload: dict[str, Any]) -> str:
    mtype = str(payload.get("type") or "").strip().lower()
    state = _event_state(payload)

    if mtype.endswith("_error") or mtype == "local_error":
        return _CARD_ERROR
    if state in {"error", "failed", "failure"}:
        return _CARD_ERROR
    if state in {"cancelled", "canceled", "warning", "warn", "degraded"}:
        return _CARD_WARNING
    if state in {"done", "completed", "complete", "succeeded", "success", "healthy", "live"}:
        return _CARD_SUCCESS
    if state in {"queued", "running", "pending", "scraping", "open", "connected", "ready"}:
        return _CARD_PROCESS

    if mtype in {"job_status", "training_progress", "cell_progress", "socket_state"}:
        return _CARD_PROCESS
    if mtype in {"job_result", "heartbeat"}:
        return _CARD_SUCCESS
    if mtype in {"scenario_updated", "dataset_event"}:
        return "accent_active" if mtype == "scenario_updated" else "accent_select"
    if mtype.startswith("asset_") or mtype.startswith("sector_"):
        return "accent_select"
    return _event_color_role(payload)


def _notification_card_identity(payload: dict[str, Any]) -> str:
    nested = payload.get("payload")
    payload_body = nested if isinstance(nested, dict) else {}
    nested_status = payload.get("status_payload")
    status_body = nested_status if isinstance(nested_status, dict) else {}
    nested_result = payload.get("result")
    result_body = nested_result if isinstance(nested_result, dict) else {}

    for value in (
        payload.get("job_id"),
        payload.get("scenario"),
        payload.get("scope"),
        payload.get("cell_name"),
        payload.get("name"),
        payload_body.get("name"),
        status_body.get("name"),
        result_body.get("name"),
        payload.get("asset_id"),
        payload_body.get("asset_id"),
        payload.get("sector_path"),
        payload_body.get("sector_path"),
        payload.get("sector_id"),
        payload_body.get("sector_id"),
        payload.get("service"),
        payload.get("source"),
        payload.get("action"),
    ):
        text = str(value or "").strip()
        if text:
            return text.lower()
    return ""


def notification_card_key(payload: dict[str, Any]) -> str:
    mtype = str(payload.get("type") or "unknown").strip().lower() or "unknown"
    identity = _notification_card_identity(payload)
    return f"{mtype}|{identity}" if identity else mtype


def notification_card_title(payload: dict[str, Any]) -> str:
    title = _pretty_event_type(payload)
    scope = _event_scope(payload)
    if scope:
        return f"{title} · {scope}"
    return title


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


class HeartbeatNotificationGate:
    """Only surface heartbeat cards when the health signature changes meaningfully."""

    def __init__(self) -> None:
        self._last_signature: tuple[str, int, int, int] | None = None

    def should_emit(self, payload: dict[str, Any]) -> bool:
        signature = self._signature(payload)
        previous = self._last_signature
        self._last_signature = signature

        if previous == signature:
            return False
        if self._is_interesting(signature):
            return True
        if previous is None:
            return False
        previous_state = previous[0]
        current_state = signature[0]
        return previous_state != current_state and (previous_state != "live" or current_state != "live")

    @staticmethod
    def _signature(payload: dict[str, Any]) -> tuple[str, int, int, int]:
        return (
            str(payload.get("state") or "unknown").strip().lower() or "unknown",
            _safe_int(payload.get("queued")),
            _safe_int(payload.get("running")),
            _safe_int(payload.get("error")),
        )

    @staticmethod
    def _is_interesting(signature: tuple[str, int, int, int]) -> bool:
        state, queued, running, error = signature
        return state != "live" or queued > 0 or running > 0 or error > 0


def should_show_notification_card(
    payload: dict[str, Any],
    heartbeat_gate: Optional[HeartbeatNotificationGate] = None,
) -> bool:
    mtype = str(payload.get("type") or "").strip().lower()
    if not mtype:
        return False
    if mtype == "heartbeat":
        return heartbeat_gate.should_emit(payload) if heartbeat_gate is not None else False
    if mtype in {"training_progress", "cell_progress", "scenario_updated", "dataset_event"}:
        return False
    if mtype == "toast":
        return False
    if mtype.endswith("_error"):
        return True

    state = _event_state(payload)
    if state in {"error", "failed", "degraded"}:
        return True
    if mtype in {"local_error", "job_result", "socket_state"}:
        return True
    if mtype == "job_status":
        return state in {
            "queued",
            "running",
            "done",
            "completed",
            "complete",
            "succeeded",
            "success",
            "error",
            "failed",
            "cancelled",
            "canceled",
        }
    return False


class NotificationCard(QFrame):
    activated = pyqtSignal(dict)
    dismissRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = ""
        self._payload: dict[str, Any] = {}
        self._color_role = "text_signal"

        self.setObjectName("cvOpsNotificationCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setMinimumWidth(280)
        self.setMaximumWidth(380)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(6)

        self._title = QLabel("", self)
        self._title.setWordWrap(False)
        header.addWidget(self._title, stretch=1)

        self._count_badge = QLabel("1", self)
        self._count_badge.setObjectName("countBadge")
        self._count_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_badge.setMinimumWidth(22)
        self._count_badge.setVisible(False)
        header.addWidget(self._count_badge)

        self._dismiss_btn = QPushButton("x", self)
        self._dismiss_btn.setObjectName("dismissButton")
        self._dismiss_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dismiss_btn.setFixedSize(20, 20)
        self._dismiss_btn.clicked.connect(self._on_dismiss_clicked)
        header.addWidget(self._dismiss_btn)

        root.addLayout(header)

        self._summary = QLabel("", self)
        self._summary.setWordWrap(True)
        root.addWidget(self._summary)

        self._meta = QLabel("", self)
        root.addWidget(self._meta)

        self._apply_style(self._color_role)

    def configure(self, key: str, payload: dict[str, Any], *, count: int = 1) -> None:
        self._key = str(key or "")
        self._payload = dict(payload)
        self._color_role = _notification_card_color_role(payload)

        self._title.setText(notification_card_title(payload))
        self._summary.setText(_event_summary(payload) or _event_scope(payload) or self._title.text())
        self._meta.setText(self._meta_text(payload))
        self._count_badge.setText(str(max(1, int(count))))
        self._count_badge.setVisible(int(count) > 1)
        self._apply_style(self._color_role)

    def summary_text(self) -> str:
        return str(self._summary.text() or "")

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if self._payload:
            self.activated.emit(dict(self._payload))
        event.accept()

    def _meta_text(self, payload: dict[str, Any]) -> str:
        parts = [_event_timestamp(payload)]
        action = str(payload.get("action") or "").strip().lower()
        if action == "open_test_range":
            parts.append("open Range")
        else:
            parts.append("open notifications")
        return " · ".join(part for part in parts if part)

    def _apply_style(self, color_role: str) -> None:
        accent = _rgba_css(color_role, 215)
        accent_hot = _rgba_css(color_role, 238)
        fill_top = cvops_rgba("bg_panel", 0.88)
        fill_bottom = cvops_rgba("bg_void", 0.92)
        hover_fill = cvops_rgba("line_light", 0.22)
        border = cvops_rgba("line_light", 0.36)
        border_soft = cvops_rgba("line_light", 0.24)
        control = cvops_rgba("line_light", 0.18)
        control_hover = cvops_rgba("line_light", 0.28)
        text_bright = cvops_color("text_bright")
        text_soft = cvops_color("text_signal")
        text_muted = cvops_rgba("text_iron", 0.88)
        self.setStyleSheet(
            "QFrame#cvOpsNotificationCard {"
            f" background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 {fill_top}, stop:1 {fill_bottom});"
            f" border: 1px solid {border};"
            f" border-left: 3px solid {accent};"
            " border-radius: 8px;"
            "}"
            "QFrame#cvOpsNotificationCard:hover {"
            f" background: {hover_fill};"
            "}"
            "QLabel {"
            f" color: {text_bright};"
            " border: none;"
            "}"
            "QLabel#countBadge {"
            f" color: {text_soft};"
            f" background: {control};"
            f" border: 1px solid {border_soft};"
            " border-radius: 8px;"
            " font-size: 10px;"
            " font-weight: 700;"
            " padding: 0 4px;"
            "}"
            "QPushButton#dismissButton {"
            f" background: {control};"
            f" border: 1px solid {border_soft};"
            " border-radius: 6px;"
            f" color: {text_muted};"
            " font-size: 11px;"
            " font-weight: 700;"
            "}"
            "QPushButton#dismissButton:hover {"
            f" background: {control_hover};"
            f" border-color: {accent_hot};"
            f" color: {text_bright};"
            "}"
        )
        self._title.setStyleSheet(
            f"color: {text_bright}; font-size: 11px; font-weight: 800; border: none;"
        )
        self._summary.setStyleSheet(
            f"color: {text_soft}; font-size: 10px; border: none;"
        )
        self._meta.setStyleSheet(
            f"color: {text_muted}; font-size: 10px; border: none;"
        )

    def _on_dismiss_clicked(self) -> None:
        if self._key:
            self.dismissRequested.emit(self._key)

    def refresh_theme_styles(self) -> None:
        self._apply_style(self._color_role)


class NotificationCardTray(QWidget):
    notificationActivated = pyqtSignal(dict)
    trayLayoutChanged = pyqtSignal()

    def __init__(
        self,
        *,
        max_cards: int = 3,
        ttl_ms: int = 7000,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("notificationCardTray")
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        self._max_cards = max(1, int(max_cards))
        self._ttl_ms = max(1500, int(ttl_ms))
        self._cards: dict[str, NotificationCard] = {}
        self._counts: dict[str, int] = {}
        self._timers: dict[str, QTimer] = {}
        self._order: list[str] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        root.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)

        self.setVisible(False)

    def push(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        key = notification_card_key(payload)
        if not key:
            return

        card = self._cards.get(key)
        if card is None:
            while len(self._order) >= self._max_cards:
                self._remove_card(self._order[-1])
            card = NotificationCard(self)
            card.activated.connect(self.notificationActivated.emit)
            card.dismissRequested.connect(self._remove_card)
            self.layout().insertWidget(0, card, 0, Qt.AlignmentFlag.AlignRight)  # type: ignore[union-attr]
            self._cards[key] = card
            self._counts[key] = 1
            self._order.insert(0, key)
        else:
            self._counts[key] = self._counts.get(key, 1) + 1
            self._move_to_top(key)

        card.configure(key, payload, count=self._counts[key])
        self._restart_timer(key)
        self.adjustSize()
        self._position_near_parent_right()
        self.raise_()
        self.setVisible(True)
        self.trayLayoutChanged.emit()

    def clear(self) -> None:
        for key in list(self._order):
            self._remove_card(key)
        self.trayLayoutChanged.emit()

    def card_count(self) -> int:
        return len(self._order)

    def refresh_theme_styles(self) -> None:
        for card in self._cards.values():
            card.refresh_theme_styles()

    def _move_to_top(self, key: str) -> None:
        card = self._cards.get(key)
        if card is None:
            return
        self.layout().removeWidget(card)  # type: ignore[union-attr]
        self.layout().insertWidget(0, card, 0, Qt.AlignmentFlag.AlignRight)  # type: ignore[union-attr]
        if key in self._order:
            self._order.remove(key)
        self._order.insert(0, key)

    def _restart_timer(self, key: str) -> None:
        timer = self._timers.get(key)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.timeout.connect(lambda key=key: self._remove_card(key))
            self._timers[key] = timer
        timer.stop()
        timer.start(self._ttl_ms)

    def _remove_card(self, key: str) -> None:
        timer = self._timers.pop(key, None)
        if timer is not None:
            timer.stop()
            timer.deleteLater()

        card = self._cards.pop(key, None)
        if card is not None:
            self.layout().removeWidget(card)  # type: ignore[union-attr]
            card.deleteLater()

        self._counts.pop(key, None)
        if key in self._order:
            self._order.remove(key)
        self.adjustSize()
        self.setVisible(bool(self._order))
        self.trayLayoutChanged.emit()

    def _position_near_parent_right(self, margin: int = 12) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        hint = self.sizeHint().expandedTo(self.minimumSizeHint())
        width = self.width() if self.width() > 0 else hint.width()
        height = self.height() if self.height() > 0 else hint.height()
        if width <= 0 or height <= 0:
            return
        try:
            rect = parent.contentsRect()
        except Exception:
            rect = parent.rect()
        max_width = max(240, rect.width() - (margin * 2))
        width = min(width, max_width)
        x = max(rect.left() + margin, rect.right() - width - margin)
        y = rect.top() + margin
        self.resize(width, height)
        self.move(x, y)
