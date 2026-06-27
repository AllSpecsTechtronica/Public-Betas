from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

from PyQt6.QtCore import QPoint, QPointF, QRectF, Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QFontMetrics, QIcon, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import QWidget

from .theme import _scheme_rgb, current_color_scheme


def _hexagon_poly(center: QPointF, radius: float) -> QPolygonF:
    return QPolygonF([
        QPointF(center.x() + radius * math.cos(math.pi / 3 * i - math.pi / 2),
                center.y() + radius * math.sin(math.pi / 3 * i - math.pi / 2))
        for i in range(6)
    ])


def _tint_icon_pixmap(icon: QIcon, size: QSize, color: QColor, mode: QIcon.Mode) -> QPixmap:
    pixmap = icon.pixmap(size, mode)
    if pixmap.isNull():
        return pixmap
    tinted = pixmap.copy()
    painter = QPainter(tinted)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), color)
    painter.end()
    return tinted


@dataclass
class RadialAction:
    key: str
    label: str
    trigger: Callable[[], None]
    icon: Optional[QIcon] = None
    checked: bool = False
    enabled: bool = True
    status: str = "none"


class RadialMenuOverlay(QWidget):
    """Transient circular-node radial launcher for HUD actions."""

    action_triggered = pyqtSignal(str)

    _PER_RING = 10
    _CENTER_RADIUS = 20.0
    _CENTER_ACTION_Y_OFFSET_RATIO = 0.45
    _NODE_RADIUS = 15.0
    _RING_START = 200.0
    _RING_STEP = 52.0
    _DISMISS_KEYS = {"dismiss_hud", "dismiss", "close"}
    _STATUS_COLORS = {
        "running": QColor(43, 196, 217, 245),   # ns-cyan
        "loading": QColor(232, 163, 23, 245),   # ns-amber
        "error": QColor(230, 62, 43, 245),      # ns-vermillion
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("radialMenuOverlay")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.hide()
        self._actions: list[RadialAction] = []
        self._center_action: RadialAction | None = None
        self._center = QPointF(0.0, 0.0)
        self._hover_index = -1
        self._hover_center = False
        self._opened = False
        self._nodes: list[tuple[int, QPointF, float]] = []

    def open_at(self, center_in_parent: QPoint, actions: list[RadialAction]) -> None:
        if not actions:
            self.hide()
            return
        self._center_action, self._actions = self._split_center_action(actions)
        self._center = QPointF(float(center_in_parent.x()), float(center_in_parent.y()))
        self._hover_index = -1
        self._hover_center = False
        self._opened = True
        self._rebuild_nodes()
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.PopupFocusReason)
        self.update()

    def sync_actions(self, actions: list[RadialAction]) -> None:
        """Update live action state while the menu is open."""
        if not self._opened:
            return
        if not actions:
            self.close_menu()
            return

        next_center, next_ring = self._split_center_action(actions)
        current_keys = [a.key for a in self._actions]
        next_keys = [a.key for a in next_ring]
        layout_changed = current_keys != next_keys

        if layout_changed:
            self._center_action = next_center
            self._actions = next_ring
            self._rebuild_nodes()
            self.update()
            return

        for idx, source in enumerate(next_ring):
            target = self._actions[idx]
            target.label = source.label
            target.trigger = source.trigger
            target.icon = source.icon
            target.checked = source.checked
            target.enabled = source.enabled
            target.status = source.status

        if self._center_action is None and next_center is not None:
            self._center_action = next_center
        elif self._center_action is not None and next_center is None:
            self._center_action = None
        elif self._center_action is not None and next_center is not None:
            self._center_action.label = next_center.label
            self._center_action.trigger = next_center.trigger
            self._center_action.icon = next_center.icon
            self._center_action.checked = next_center.checked
            self._center_action.enabled = next_center.enabled
            self._center_action.status = next_center.status

        self.update()

    def close_menu(self) -> None:
        self._opened = False
        self._hover_index = -1
        self._hover_center = False
        self._nodes = []
        self.hide()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._opened:
            self._rebuild_nodes()

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Tab):
            self.close_menu()
            event.accept()
            return
        super().keyPressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        kind, idx = self._hit_test(event.position())
        self._hover_center = kind == "center"
        self._hover_index = idx if kind == "ring" else -1
        self.update()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            event.accept()
            return
        kind, idx = self._hit_test(event.position())
        self._hover_center = kind == "center"
        self._hover_index = idx if kind == "ring" else -1
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            event.accept()
            return
        kind, idx = self._hit_test(event.position())
        if kind == "center":
            action = self._center_action
            if action is not None and action.enabled:
                action.trigger()
                self.action_triggered.emit(action.key)
            self.close_menu()
            event.accept()
            return
        if kind == "ring" and 0 <= idx < len(self._actions):
            action = self._actions[idx]
            if action.enabled:
                action.trigger()
                self.action_triggered.emit(action.key)
                if action.key in self._DISMISS_KEYS:
                    self.close_menu()
                    event.accept()
                    return
        kind, idx = self._hit_test(event.position())
        self._hover_center = kind == "center"
        self._hover_index = idx if kind == "ring" else -1
        self._rebuild_nodes()
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:
        if not self._opened or not self._actions:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        # No full-screen dim / vignette — paint only the diamond nodes so video stays unobscured.
        icon_font = painter.font()
        label_font = painter.font()
        label_font.setPointSizeF(max(7.0, label_font.pointSizeF() - 2.0))
        label_font.setBold(True)
        label_metrics = QFontMetrics(label_font)
        ar, ag, ab = _scheme_rgb("accent_dark")
        aux_r, aux_g, aux_b = (255, 210, 53) if current_color_scheme() == "beacon" else (ar, ag, ab)
        icon_color = QColor(255, 255, 255, 245)
        disabled_icon_color = QColor(255, 255, 255, 110)

        for idx, node_center, node_radius in self._nodes:
            action = self._actions[idx]

            if not action.enabled:
                bg = QColor(ar, ag, ab, 82)
                border = QColor(ar, ag, ab, 60)
            elif action.checked:
                bg = QColor(ar, ag, ab, 238)
                border = QColor(255, 255, 255, 160)
            elif idx == self._hover_index:
                bg = QColor(ar, ag, ab, 224)
                border = QColor(255, 255, 255, 132)
            else:
                bg = QColor(ar, ag, ab, 202)
                border = QColor(ar, ag, ab, 150)

            painter.setPen(QPen(border, 1.2))
            painter.setBrush(bg)
            painter.drawPolygon(_hexagon_poly(node_center, node_radius))

            # Status indicator: small filled diamond in the top-right corner.
            status_color = self._STATUS_COLORS.get(str(action.status or "").strip().lower())
            if status_color is not None:
                tip = QPointF(node_center.x() + node_radius * 0.62, node_center.y() - node_radius * 0.62)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(status_color)
                painter.drawPolygon(_hexagon_poly(tip, 4.0))

            icon = action.icon
            if icon is not None and not icon.isNull():
                icon_px = _tint_icon_pixmap(
                    icon,
                    QSize(18, 18),
                    icon_color if action.enabled else disabled_icon_color,
                    QIcon.Mode.Normal if action.enabled else QIcon.Mode.Disabled,
                )
                if not icon_px.isNull():
                    logical = icon_px.deviceIndependentSize()
                    painter.drawPixmap(
                        int(node_center.x() - (logical.width() / 2.0)),
                        int(node_center.y() - (logical.height() / 2.0)),
                        icon_px,
                    )
            else:
                glyph = (action.label or action.key or "?").strip()[:2].upper()
                text_col = icon_color if action.enabled else disabled_icon_color
                painter.setPen(text_col)
                painter.setFont(icon_font)
                node_rect = QRectF(
                    node_center.x() - node_radius,
                    node_center.y() - node_radius,
                    node_radius * 2.0,
                    node_radius * 2.0,
                )
                painter.drawText(node_rect, int(Qt.AlignmentFlag.AlignCenter), glyph)

            label_text = label_metrics.elidedText(
                action.label,
                Qt.TextElideMode.ElideRight,
                78,
            )
            if label_text:
                label_side = -1.0 if node_center.x() < self._center.x() else 1.0
                label_col = QColor(aux_r, aux_g, aux_b, 235 if action.enabled else 92)
                if idx == self._hover_index:
                    label_col = QColor(aux_r, aux_g, aux_b, 255)
                painter.setFont(label_font)
                painter.setPen(label_col)
                label_w = 82.0
                label_rect = QRectF(
                    node_center.x() + (node_radius + 6.0) if label_side > 0 else node_center.x() - node_radius - 6.0 - label_w,
                    node_center.y() - ((label_metrics.height() + 2) / 2.0),
                    label_w,
                    float(label_metrics.height() + 2),
                )
                painter.drawText(
                    label_rect,
                    int(
                        (Qt.AlignmentFlag.AlignLeft if label_side > 0 else Qt.AlignmentFlag.AlignRight)
                        | Qt.AlignmentFlag.AlignVCenter
                    ),
                    label_text,
                )
                painter.setFont(icon_font)

        center_action = self._center_action
        center_enabled = center_action.enabled if center_action is not None else True
        center_point = self._center_action_point()
        if self._hover_center and center_enabled:
            c_bg = QColor(ar, ag, ab, 224)
            c_border = QColor(255, 255, 255, 132)
        else:
            c_bg = QColor(ar, ag, ab, 202 if center_enabled else 82)
            c_border = QColor(ar, ag, ab, 175 if center_enabled else 72)

        painter.setPen(QPen(c_border, 1.3))
        painter.setBrush(c_bg)
        painter.drawPolygon(_hexagon_poly(center_point, self._CENTER_RADIUS))

        center_rect = QRectF(
            center_point.x() - self._CENTER_RADIUS,
            center_point.y() - self._CENTER_RADIUS,
            self._CENTER_RADIUS * 2.0,
            self._CENTER_RADIUS * 2.0,
        )
        center_icon = center_action.icon if center_action is not None else None
        center_text = center_action.label if center_action is not None else "X"

        if center_icon is not None and not center_icon.isNull():
            icon_px = _tint_icon_pixmap(
                center_icon,
                QSize(20, 20),
                icon_color if center_enabled else disabled_icon_color,
                QIcon.Mode.Normal if center_enabled else QIcon.Mode.Disabled,
            )
            if not icon_px.isNull():
                logical = icon_px.deviceIndependentSize()
                painter.drawPixmap(
                    int(center_point.x() - (logical.width() / 2.0)),
                    int(center_point.y() - (logical.height() / 2.0)),
                    icon_px,
                )
            else:
                painter.setPen(icon_color if center_enabled else disabled_icon_color)
                center_glyph = (center_text or "X").strip()[:2].upper()
                painter.drawText(center_rect, int(Qt.AlignmentFlag.AlignCenter), center_glyph)
        else:
            painter.setPen(icon_color if center_enabled else disabled_icon_color)
            center_glyph = (center_text or "X").strip()[:2].upper()
            painter.drawText(center_rect, int(Qt.AlignmentFlag.AlignCenter), center_glyph)

        if center_action is not None and center_text:
            painter.setFont(label_font)
            painter.setPen(QColor(aux_r, aux_g, aux_b, 235 if center_enabled else 92))
            center_label = label_metrics.elidedText(center_text, Qt.TextElideMode.ElideRight, 76)
            center_label_rect = QRectF(
                center_point.x() + self._CENTER_RADIUS + 7.0,
                center_point.y() - ((label_metrics.height() + 2) / 2.0),
                80.0,
                float(label_metrics.height() + 2),
            )
            painter.drawText(
                center_label_rect,
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                center_label,
            )
            painter.setFont(icon_font)

    def _hit_test(self, point: QPointF) -> tuple[str, int]:
        if self._distance(point, self._center_action_point()) <= self._CENTER_RADIUS:
            return "center", -1
        for idx, node_center, node_radius in self._nodes:
            if self._distance(point, node_center) <= node_radius:
                return "ring", idx
        return "none", -1

    def _center_action_point(self) -> QPointF:
        offset = self._RING_START * self._CENTER_ACTION_Y_OFFSET_RATIO
        y = self._center.y() + offset
        margin = self._CENTER_RADIUS + 4.0
        if self.height() > 0:
            y = max(margin, min(float(self.height()) - margin, y))
        return QPointF(self._center.x(), y)

    def _rebuild_nodes(self) -> None:
        # Nodes are distributed across two vertical half-arcs (left and right),
        # leaving the center of view clear. Left side spans 180deg--360deg (top-left
        # to bottom-left), right side spans 0deg--180deg (top-right to bottom-right).
        self._nodes = []
        total = len(self._actions)
        if total <= 0:
            return

        left_half = self._actions[: (total + 1) // 2]
        right_half = self._actions[(total + 1) // 2 :]

        ring_count = max(1, (len(left_half) + self._PER_RING - 1) // self._PER_RING)

        def _place_half(items: list, side: str, global_offset: int) -> None:
            total_items = len(items)
            if total_items == 0:
                return
            per_ring = max(1, (total_items + ring_count - 1) // ring_count)
            for ring in range(ring_count):
                start = ring * per_ring
                count = min(per_ring, total_items - start)
                if count <= 0:
                    continue
                orbit = self._RING_START + (ring * self._RING_STEP)
                # Narrow arc keeps nodes in a vertical column on each side
                if count == 1:
                    angles = [0.0]
                else:
                    angles = [-50.0 + (100.0 / (count - 1)) * i for i in range(count)]
                for idx_in_ring, arc_angle in enumerate(angles):
                    global_idx = global_offset + start + idx_in_ring
                    if side == "left":
                        # Left arc: mirror around 180deg (pointing left)
                        deg = 180.0 + arc_angle
                    else:
                        # Right arc: around 0deg (pointing right)
                        deg = 0.0 + arc_angle
                    self._nodes.append((global_idx, self._point_for_angle(orbit, deg), self._NODE_RADIUS))

        _place_half(left_half, "left", 0)
        _place_half(right_half, "right", len(left_half))

    def _split_center_action(self, actions: list[RadialAction]) -> tuple[RadialAction | None, list[RadialAction]]:
        center: RadialAction | None = None
        ring: list[RadialAction] = []
        for action in actions:
            if center is None and action.key in self._DISMISS_KEYS:
                center = action
                continue
            ring.append(action)
        return center, ring

    @staticmethod
    def _distance(a: QPointF, b: QPointF) -> float:
        dx = a.x() - b.x()
        dy = a.y() - b.y()
        return math.hypot(dx, dy)

    def _point_for_angle(self, radius: float, deg: float) -> QPointF:
        rad = math.radians(deg)
        return QPointF(
            self._center.x() + math.cos(rad) * radius,
            self._center.y() + math.sin(rad) * radius,
        )
