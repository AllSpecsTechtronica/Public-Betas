"""Grouped notification feed for CvOps system activity.

Mirrors the same payloads as EventPulseWidget (job_status, training_progress,
scenario_updated, etc.) plus local UI/system alerts in a grouped card stack so
repeated events do not flood the operator view. Each card counts related
notifications and can be expanded to inspect the full event history.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .cvops_theme import cvops_color, cvops_rgba, set_cvops_stylesheet
from .event_pulse_widget import (
    _event_color_role,
    _event_label,
    _event_scope,
    _event_summary,
    _event_timestamp,
)

_MAX_ROWS = 2000
_TOOLTIP_MAX = 6000


@dataclass(slots=True)
class _NotificationRow:
    label: str
    color: str
    payload: dict[str, Any]


@dataclass(slots=True)
class _NotificationGroup:
    key: str
    title: str
    latest_timestamp: str
    latest_summary: str
    color: str
    rows: list[_NotificationRow] = field(default_factory=list)


def _payload_matches_filter(payload: dict[str, Any], label: str, needle: str) -> bool:
    if not needle:
        return True
    if needle in label.lower():
        return True
    try:
        blob = json.dumps(payload, default=str).lower()
    except Exception:
        blob = str(payload).lower()
    return needle in blob


def _pretty_event_type(payload: dict[str, Any]) -> str:
    raw = str(payload.get("type") or "unknown").strip().replace("-", "_")
    parts = [part for part in raw.split("_") if part]
    if not parts:
        return "Unknown"
    return " ".join(part.capitalize() for part in parts)


def _notification_group_identity(payload: dict[str, Any]) -> str:
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


def _notification_group_key(payload: dict[str, Any]) -> str:
    mtype = str(payload.get("type") or "unknown").strip().lower() or "unknown"
    identity = _notification_group_identity(payload)
    return f"{mtype}|{identity}" if identity else mtype


def _notification_group_title(payload: dict[str, Any]) -> str:
    title = _pretty_event_type(payload)
    scope = _event_scope(payload)
    if scope:
        return f"{title} · {scope}"
    return title


def _payload_tooltip(payload: dict[str, Any]) -> str:
    try:
        tip = json.dumps(payload, indent=2, default=str)
    except Exception:
        tip = str(payload)
    if len(tip) > _TOOLTIP_MAX:
        tip = tip[:_TOOLTIP_MAX] + "\n…"
    return tip


def _build_notification_groups(
    rows: deque[tuple[str, str, dict[str, Any]]],
    needle: str = "",
) -> list[_NotificationGroup]:
    groups: dict[str, _NotificationGroup] = {}
    order: list[str] = []
    for label, color, payload in rows:
        if not _payload_matches_filter(payload, label, needle):
            continue
        key = _notification_group_key(payload)
        row = _NotificationRow(label=label, color=color, payload=payload)
        group = groups.get(key)
        if group is None:
            group = _NotificationGroup(
                key=key,
                title=_notification_group_title(payload),
                latest_timestamp=_event_timestamp(payload),
                latest_summary=_event_summary(payload) or label,
                color=color,
                rows=[row],
            )
            groups[key] = group
            order.append(key)
            continue
        group.rows.append(row)
    return [groups[key] for key in order]


def _rgba_css(color: str, alpha: int) -> str:
    swatch = QColor(color if str(color).startswith("#") else cvops_color(color))
    return f"rgba({swatch.red()}, {swatch.green()}, {swatch.blue()}, {max(0, min(255, alpha))})"


class NotificationGroupCard(QFrame):
    expansionChanged = pyqtSignal(str, bool)
    dismissRequested = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._group_key = ""
        self._latest_payload: dict[str, Any] | None = None
        self._latest_color = "line_light"

        self.setObjectName("notificationGroupCard")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        # Cap the card width and let it hug content instead of stretching to the
        # window edge (full-width cards across a wide window look tacky).
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.setMaximumWidth(760)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 7, 10, 7)
        root.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self._toggle = QToolButton(self)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(False)
        self._toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.RightArrow)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.toggled.connect(self._on_toggled)
        header.addWidget(self._toggle, stretch=1)

        self._dismiss_btn = QPushButton("DISMISS x", self)
        self._dismiss_btn.setObjectName("notificationDismissBtn")
        self._dismiss_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._dismiss_btn.setToolTip("Remove this notification stack from the feed")
        self._dismiss_btn.setFixedHeight(22)
        self._dismiss_btn.clicked.connect(self._on_dismiss_clicked)
        header.addWidget(self._dismiss_btn)

        self._count_badge = QLabel("0", self)
        self._count_badge.setObjectName("countBadge")
        self._count_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count_badge.setMinimumWidth(32)
        header.addWidget(self._count_badge)

        self._timestamp = QLabel("", self)
        self._timestamp.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self._timestamp)

        root.addLayout(header)

        self._preview = QLabel("", self)
        self._preview.setObjectName("preview")
        self._preview.setWordWrap(True)
        self._preview.setMaximumHeight(42)
        root.addWidget(self._preview)

        self._details = QListWidget(self)
        self._details.setObjectName("notificationGroupDetails")
        self._details.setAlternatingRowColors(False)
        self._details.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        details_font = QFont("JetBrains Mono, Courier New, monospace")
        details_font.setPixelSize(11)
        self._details.setFont(details_font)
        self._details.setVisible(False)
        root.addWidget(self._details)

        self._apply_color(self._latest_color)

    def configure(self, group: _NotificationGroup, *, expanded: bool = False) -> None:
        self._group_key = group.key
        self._latest_payload = group.rows[0].payload if group.rows else None
        self._latest_color = group.color or "line_light"

        self._toggle.blockSignals(True)
        self._toggle.setText(group.title)
        self._toggle.setChecked(bool(expanded))
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.blockSignals(False)

        self._count_badge.setText(str(len(group.rows)))
        self._timestamp.setText(group.latest_timestamp)
        self._preview.setText(group.latest_summary)

        self._details.clear()
        for row in group.rows:
            item = QListWidgetItem(row.label)
            item.setData(Qt.ItemDataRole.UserRole, row.payload)
            item.setToolTip(_payload_tooltip(row.payload))
            self._details.addItem(item)

        self._details.setVisible(bool(expanded))
        self._update_details_height()
        self._apply_color(self._latest_color)
        self.updateGeometry()

    def is_expanded(self) -> bool:
        return self._toggle.isChecked()

    def selected_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for item in self._details.selectedItems():
            payload = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def all_payloads(self) -> list[dict[str, Any]]:
        payloads: list[dict[str, Any]] = []
        for idx in range(self._details.count()):
            item = self._details.item(idx)
            payload = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(payload, dict):
                payloads.append(payload)
        return payloads

    def latest_payload(self) -> dict[str, Any] | None:
        return dict(self._latest_payload) if isinstance(self._latest_payload, dict) else None

    def _apply_color(self, color: str) -> None:
        accent = _rgba_css(color, 230)
        accent_hot = _rgba_css(color, 238)
        accent_wash = _rgba_css(color, 58)
        accent_badge = _rgba_css(color, 50)
        accent_badge_border = _rgba_css(color, 150)
        panel_top = cvops_rgba("bg_panel", 0.82)
        panel_bottom = cvops_rgba("bg_void", 0.90)
        panel_hover = cvops_rgba("line_light", 0.18)
        border = cvops_rgba("line_light", 0.34)
        border_soft = cvops_rgba("line_light", 0.22)
        control = cvops_rgba("line_light", 0.18)
        control_hover = cvops_rgba("line_light", 0.28)
        text_bright = cvops_color("text_bright")
        text_signal = cvops_color("text_signal")
        text_muted = cvops_rgba("text_iron", 0.88)
        self.setStyleSheet(
            "QFrame#notificationGroupCard {"
            f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent_wash}, stop:0.16 {panel_top}, stop:1 {panel_bottom});"
            f" border: 1px solid {border};"
            f" border-left: 5px solid {accent};"
            " border-radius: 8px;"
            "}"
            "QFrame#notificationGroupCard:hover {"
            f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent_badge}, stop:0.18 {panel_hover}, stop:1 {panel_bottom});"
            "}"
            "QToolButton {"
            " background: transparent;"
            " border: none;"
            f" color: {text_bright};"
            " font-size: 11px;"
            " font-weight: 800;"
            " text-align: left;"
            " padding: 1px 0px;"
            "}"
            "QToolButton:hover {"
            f" color: {text_bright};"
            "}"
            "QLabel {"
            f" color: {text_muted};"
            " font-size: 10px;"
            "}"
            "QLabel#preview {"
            f" color: {text_signal};"
            " font-size: 10px;"
            "}"
            "QLabel#countBadge {"
            f" color: {text_bright};"
            f" background: {accent_badge};"
            f" border: 1px solid {accent_badge_border};"
            " border-radius: 8px;"
            " font-size: 10px;"
            " font-weight: 700;"
            " padding: 1px 6px;"
            "}"
            "QListWidget#notificationGroupDetails {"
            f" background: {cvops_rgba('bg_void', 0.52)};"
            f" color: {text_signal};"
            f" border: 1px solid {border_soft};"
            " border-radius: 6px;"
            " outline: none;"
            "}"
            "QListWidget#notificationGroupDetails::item {"
            " background: transparent;"
            " border: none;"
            f" color: {text_signal};"
            " padding: 2px 6px;"
            "}"
            "QListWidget#notificationGroupDetails::item:hover {"
            f" background: {accent_wash};"
            f" color: {text_bright};"
            "}"
            "QListWidget#notificationGroupDetails::item:selected {"
            f" background: {cvops_rgba('selection_active', 0.88)};"
            f" color: {cvops_color('selection_text')};"
            "}"
            "QPushButton#notificationDismissBtn {"
            f" background: {control};"
            f" border: 1px solid {border_soft};"
            " border-radius: 6px;"
            f" color: {text_bright};"
            " font-size: 10px; font-weight: 800; padding: 1px 8px;"
            "}"
            "QPushButton#notificationDismissBtn:hover {"
            f" background: {control_hover};"
            f" border-color: {accent_hot};"
            f" color: {text_bright};"
            "}"
        )

    def refresh_theme_styles(self) -> None:
        self._apply_color(self._latest_color)

    def _update_details_height(self) -> None:
        if self._details.count() <= 0:
            self._details.setMinimumHeight(0)
            self._details.setMaximumHeight(0)
            return
        row_height = max(18, self._details.sizeHintForRow(0))
        visible_rows = min(4, self._details.count())
        frame = self._details.frameWidth() * 2
        height = (row_height * visible_rows) + frame + 6
        self._details.setMinimumHeight(height)
        self._details.setMaximumHeight(height)

    def _on_toggled(self, checked: bool) -> None:
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        self._details.setVisible(bool(checked))
        self._update_details_height()
        self.updateGeometry()
        self.expansionChanged.emit(self._group_key, bool(checked))

    def _on_dismiss_clicked(self) -> None:
        if self._group_key:
            self.dismissRequested.emit(self._group_key)


class NotificationsPanel(QWidget):
    """Grouped stack of system and local notification events."""

    def __init__(
        self,
        *,
        stream_hint: str = "",
        parent: Optional[QWidget] = None,
        top_toolbar_layout: Optional[QHBoxLayout] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("notificationsPanel")
        self._toolbar_in_window = top_toolbar_layout is not None
        self._rows: deque[tuple[str, str, dict[str, Any]]] = deque(maxlen=_MAX_ROWS)
        self._group_cards: dict[str, NotificationGroupCard] = {}
        self._group_expanded: dict[str, bool] = {}
        self._group_order: list[str] = []
        self._visible_notification_count = 0
        self._empty_label: QLabel | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QFrame(self)
        header.setObjectName("notificationsHeader")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 7, 10, 7)
        header_layout.setSpacing(10)

        title_block = QVBoxLayout()
        title_block.setContentsMargins(0, 0, 0, 0)
        title_block.setSpacing(2)
        title = QLabel("Notification Center", header)
        title.setProperty("isTitle", True)
        title_block.addWidget(title)
        header_layout.addLayout(title_block, stretch=1)

        if stream_hint:
            sub = QLabel(stream_hint, header)
            sub.setWordWrap(False)
            self._sub_hint = sub
            sub.setObjectName("streamHint")
            header_layout.addWidget(sub)
        else:
            self._sub_hint = None
        layout.addWidget(header)

        self._count = QLabel("0 stacks", self)
        self._count.setObjectName("notificationCount")

        self._filter = QLineEdit(self)
        self._filter.setPlaceholderText("filter by text in summary or raw JSON…")
        self._filter.setClearButtonEnabled(True)
        self._filter.setMinimumWidth(120)
        self._filter.textChanged.connect(self._on_filter_changed)

        self._pause = QCheckBox("Pause", self)
        self._pause.toggled.connect(self._on_pause_toggled)

        self._follow = QCheckBox("Follow", self)
        self._follow.setChecked(True)

        self._clear_btn = QPushButton("Clear", self)
        self._clear_btn.setObjectName("notificationUtilityButton")
        self._clear_btn.clicked.connect(self._clear)

        self._copy_btn = QPushButton("Copy selected", self)
        self._copy_btn.setObjectName("notificationUtilityButton")
        self._copy_btn.clicked.connect(self._copy_selected)

        if top_toolbar_layout is not None:
            bar_host = top_toolbar_layout
            bar_host.addWidget(self._count)
            bar_host.addWidget(self._filter, stretch=1)
            bar_host.addWidget(self._pause)
            bar_host.addWidget(self._follow)
            bar_host.addWidget(self._clear_btn)
            bar_host.addWidget(self._copy_btn)
        else:
            toolbar = QFrame(self)
            toolbar.setObjectName("notificationsToolbar")
            bar = QHBoxLayout()
            bar.setContentsMargins(8, 5, 8, 5)
            bar.setSpacing(6)
            toolbar.setLayout(bar)
            bar.addWidget(self._count)
            bar.addWidget(self._filter, stretch=1)
            bar.addWidget(self._pause)
            bar.addWidget(self._follow)
            bar.addWidget(self._clear_btn)
            bar.addWidget(self._copy_btn)
            layout.addWidget(toolbar)

        self._scroll = QScrollArea(self)
        self._scroll.setObjectName("notificationsScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._cards_host = QWidget(self._scroll)
        self._cards_layout = QVBoxLayout(self._cards_host)
        self._cards_layout.setContentsMargins(2, 2, 2, 2)
        self._cards_layout.setSpacing(6)
        self._cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._scroll.setWidget(self._cards_host)
        layout.addWidget(self._scroll, stretch=1)

        self._apply_panel_styles()
        self._apply_toolbar_chrome()
        self._rebuild_cards()

    def _apply_toolbar_chrome(self) -> None:
        set_cvops_stylesheet(
            self._filter,
            lambda: (
                f"QLineEdit {{ background: {cvops_rgba('bg_panel', 0.66)};"
                f" color: {cvops_color('text_signal')};"
                f" border: 1px solid {cvops_rgba('line_light', 0.34)};"
                " border-radius: 6px; padding: 4px 8px;"
                " font-size: 10px; font-family: 'JetBrains Mono','Courier New',monospace; }"
                f"QLineEdit:focus {{ border-color: {cvops_rgba('text_signal', 0.36)}; }}"
            ),
        )
        count_ss = (
            "QLabel#notificationCount {"
            f"color: {cvops_rgba('text_signal', 0.92)};"
            " font-size: 10px; font-weight: 700; padding-right: 8px;"
            "}"
        )
        self._count.setStyleSheet(count_ss)
        chk = (
            "QCheckBox {"
            f"color: {cvops_color('text_signal')};"
            " font-size: 10px; spacing: 4px;"
            "}"
        )
        self._pause.setStyleSheet(chk)
        self._follow.setStyleSheet(chk)
        btn = (
            "QPushButton#notificationUtilityButton {"
            f" background: {cvops_rgba('line_light', 0.16)};"
            f" border: 1px solid {cvops_rgba('line_light', 0.30)};"
            " border-radius: 6px;"
            f" color: {cvops_color('text_signal')};"
            " min-height: 20px; padding: 2px 8px; font-size: 10px; font-weight: 700;"
            "}"
            "QPushButton#notificationUtilityButton:hover {"
            f" background: {cvops_rgba('line_light', 0.26)};"
            "}"
        )
        self._clear_btn.setStyleSheet(btn)
        self._copy_btn.setStyleSheet(btn)

    def _apply_panel_styles(self) -> None:
        parts = [
            "QWidget#notificationsPanel {"
            f" background: {cvops_rgba('bg_void', 0.16)};"
            "}",
        ]
        if not self._toolbar_in_window:
            parts.append(
                "QFrame#notificationsToolbar {"
                f" background: {cvops_rgba('bg_panel', 0.52)};"
                f" border: 1px solid {cvops_rgba('line_light', 0.28)};"
                " border-radius: 8px;"
                "}"
            )
        parts.extend(
            [
                "QFrame#notificationsHeader {"
                f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {cvops_rgba('bg_panel', 0.72)}, stop:1 {cvops_rgba('bg_void', 0.74)});"
                f" border: 1px solid {cvops_rgba('line_light', 0.30)};"
                " border-radius: 8px;"
                "}",
                "QFrame#notificationsHeader QLabel[isTitle=\"true\"] {"
                " background: transparent;"
                " border: none;"
                f" color: {cvops_color('text_bright')};"
                " padding: 0px;"
                " font-size: 13px;"
                " font-weight: 800;"
                "}",
                "QFrame#notificationsHeader QLabel {"
                f" color: {cvops_rgba('text_iron', 0.86)};"
                " font-size: 10px;"
                "}",
                "QLabel#streamHint {"
                f" color: {cvops_rgba('text_iron', 0.82)};"
                " font-size: 10px;"
                " font-family: 'JetBrains Mono','Courier New',monospace;"
                "}",
            ]
        )
        if not self._toolbar_in_window:
            parts.extend(
                [
                    "QLabel#notificationCount {"
                    f" color: {cvops_color('text_signal')};"
                    " font-size: 10px;"
                    " font-weight: 700;"
                    " padding-right: 8px;"
                    "}",
                    "QFrame#notificationsToolbar QCheckBox {"
                    f" color: {cvops_color('text_signal')};"
                    " font-size: 10px;"
                    " spacing: 4px;"
                    "}",
                    "QFrame#notificationsToolbar QPushButton#notificationUtilityButton {"
                    f" background: {cvops_rgba('line_light', 0.16)};"
                    f" border: 1px solid {cvops_rgba('line_light', 0.30)};"
                    " border-radius: 6px;"
                    f" color: {cvops_color('text_signal')};"
                    " min-height: 20px;"
                    " padding: 2px 8px;"
                    " font-size: 10px;"
                    "}",
                ]
            )
        parts.extend(
            [
                "QScrollArea#notificationsScroll {"
                f" border: 1px solid {cvops_rgba('line_light', 0.24)};"
                " border-radius: 8px;"
                f" background: {cvops_rgba('bg_void', 0.18)};"
                "}",
                "QScrollArea#notificationsScroll QWidget#qt_scrollarea_viewport {"
                " background: transparent;"
                "}",
            ]
        )
        self.setStyleSheet("".join(parts))

    def ingest(self, payload: dict[str, Any]) -> None:
        label = _event_label(payload)
        color = _event_color_role(payload)
        overflowed = len(self._rows) == self._rows.maxlen
        self._rows.appendleft((label, color, payload))

        if self._pause.isChecked():
            self._update_count_only()
            return

        if overflowed or self._active_needle():
            self._rebuild_cards()
            return

        self._ingest_visible_row(label, color, payload)
        self._update_count_only()
        if self._follow.isChecked():
            self._scroll.verticalScrollBar().setValue(0)

    def _active_needle(self) -> str:
        return str(self._filter.text() or "").strip().lower()

    def _ingest_visible_row(self, label: str, color: str, payload: dict[str, Any]) -> None:
        key = _notification_group_key(payload)
        card = self._group_cards.get(key)
        row = _NotificationRow(label=label, color=color, payload=payload)
        if card is None:
            group = _NotificationGroup(
                key=key,
                title=_notification_group_title(payload),
                latest_timestamp=_event_timestamp(payload),
                latest_summary=_event_summary(payload) or label,
                color=color,
                rows=[row],
            )
            self._insert_group_card(group)
            return

        card_group = _NotificationGroup(
            key=key,
            title=_notification_group_title(payload),
            latest_timestamp=_event_timestamp(payload),
            latest_summary=_event_summary(payload) or label,
            color=color,
            rows=[row],
        )
        existing_rows = card.all_payloads()
        for old_payload in existing_rows:
            old_label = _event_label(old_payload)
            old_color = _event_color_role(old_payload)
            card_group.rows.append(
                _NotificationRow(label=old_label, color=old_color, payload=old_payload)
            )
        expanded = self._group_expanded.get(key, False)
        card.configure(card_group, expanded=expanded)
        self._move_card_to_top(key)
        if key in self._group_order:
            self._group_order.remove(key)
        self._group_order.insert(0, key)
        self._visible_notification_count += 1

    def _insert_group_card(self, group: _NotificationGroup) -> None:
        if self._empty_label is not None:
            self._clear_cards_layout()
            self._empty_label = None
        card = NotificationGroupCard(self._cards_host)
        card.expansionChanged.connect(self._on_card_expansion_changed)
        card.dismissRequested.connect(self._on_card_dismiss)
        card.configure(group, expanded=self._group_expanded.get(group.key, False))
        self._cards_layout.insertWidget(0, card)
        self._group_cards[group.key] = card
        self._group_order.insert(0, group.key)
        self._visible_notification_count += len(group.rows)

    def _move_card_to_top(self, key: str) -> None:
        card = self._group_cards.get(key)
        if card is None:
            return
        self._cards_layout.removeWidget(card)
        self._cards_layout.insertWidget(0, card)

    def _on_filter_changed(self, _text: str) -> None:
        if self._pause.isChecked():
            self._update_count_only()
            return
        self._rebuild_cards()

    def _on_pause_toggled(self, paused: bool) -> None:
        if not paused:
            self._rebuild_cards()
            return
        self._update_count_only()

    def _rebuild_cards(self) -> None:
        groups = _build_notification_groups(self._rows, self._active_needle())

        self._group_cards.clear()
        self._group_order.clear()
        self._visible_notification_count = sum(len(group.rows) for group in groups)
        self._empty_label = None

        self._clear_cards_layout()

        if not groups:
            empty = QLabel("No notifications yet.", self._cards_host)
            empty.setStyleSheet(f"color: {cvops_color('text_iron')}; font-size: 11px;")
            self._cards_layout.addWidget(empty)
            self._empty_label = empty
            self._update_count_only()
            return

        for group in groups:
            card = NotificationGroupCard(self._cards_host)
            card.expansionChanged.connect(self._on_card_expansion_changed)
            card.dismissRequested.connect(self._on_card_dismiss)
            card.configure(group, expanded=self._group_expanded.get(group.key, False))
            self._cards_layout.addWidget(card)
            self._group_cards[group.key] = card
            self._group_order.append(group.key)

        self._update_count_only()
        if self._follow.isChecked():
            self._scroll.verticalScrollBar().setValue(0)

    def _clear_cards_layout(self) -> None:
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _update_count_only(self) -> None:
        if self._pause.isChecked():
            self._count.setText(f"{len(self._rows)} buffered (paused)")
            return
        if self._active_needle():
            self._count.setText(
                f"{len(self._group_order)} stacks · {self._visible_notification_count} shown · {len(self._rows)} total"
            )
            return
        self._count.setText(
            f"{len(self._group_order)} stacks · {self._visible_notification_count} notifications"
        )

    def refresh_theme_styles(self) -> None:
        self._apply_panel_styles()
        self._apply_toolbar_chrome()
        if self._empty_label is not None:
            self._empty_label.setStyleSheet(f"color: {cvops_color('text_iron')}; font-size: 11px;")
        for card in self._iter_cards():
            card.refresh_theme_styles()

    def _clear(self) -> None:
        self._rows.clear()
        self._group_expanded.clear()
        self._rebuild_cards()

    def _copy_selected(self) -> None:
        payloads: list[dict[str, Any]] = []
        for card in self._iter_cards():
            payloads.extend(card.selected_payloads())

        if not payloads:
            for card in self._iter_cards():
                if card.is_expanded():
                    payloads = card.all_payloads()
                    break

        if not payloads:
            for card in self._iter_cards():
                latest = card.latest_payload()
                if isinstance(latest, dict):
                    payloads = [latest]
                    break

        if not payloads:
            return

        blocks = []
        for payload in payloads:
            try:
                blocks.append(json.dumps(payload, indent=2, default=str))
            except Exception:
                blocks.append(str(payload))
        QApplication.clipboard().setText("\n\n".join(blocks))

    def _iter_cards(self) -> list[NotificationGroupCard]:
        cards: list[NotificationGroupCard] = []
        for key in self._group_order:
            card = self._group_cards.get(key)
            if isinstance(card, NotificationGroupCard):
                cards.append(card)
        return cards

    def _on_card_expansion_changed(self, key: str, expanded: bool) -> None:
        if key:
            self._group_expanded[key] = bool(expanded)

    def _on_card_dismiss(self, group_key: str) -> None:
        """Remove one notification stack (all payloads sharing its group key)."""
        if not group_key:
            return
        self._rows = deque(
            (r for r in self._rows if _notification_group_key(r[2]) != group_key),
            maxlen=_MAX_ROWS,
        )
        self._group_expanded.pop(group_key, None)
        self._rebuild_cards()
