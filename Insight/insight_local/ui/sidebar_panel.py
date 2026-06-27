from __future__ import annotations

import math

from PyQt6.QtCore import QEasingCurve, QEvent, QPointF, Qt, QVariantAnimation, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from .theme import _scheme_rgb, current_color_scheme, get_hud_strip_bg_css, get_hud_strip_border_css, text_css, theme_metallic, theme_rgba

# [HEX MESH] Pre-computed hex grid constants (pointy-top orientation)
_HEX_R = 18
_HEX_H_STEP = math.sqrt(3) * _HEX_R   # center-to-center horizontal
_HEX_V_STEP = _HEX_R * 1.5            # center-to-center vertical
_HEX_ANGLES = [math.radians(30 + 60 * i) for i in range(6)]

# [SQUARE MESH] Alternating checker-grid look (draw one square, skip one).
_SQUARE_SIZE = 16
_SQUARE_STEP = _SQUARE_SIZE * 2
_PANEL_BG_STYLE_HEX = "hexagons"
_PANEL_BG_STYLE_SQUARES = "squares"
_PANEL_BG_STYLE_ALIASES = {
    "hex": _PANEL_BG_STYLE_HEX,
    "hexagon": _PANEL_BG_STYLE_HEX,
    "hexagons": _PANEL_BG_STYLE_HEX,
    "hexigon": _PANEL_BG_STYLE_HEX,
    "hexigons": _PANEL_BG_STYLE_HEX,
    "square": _PANEL_BG_STYLE_SQUARES,
    "squares": _PANEL_BG_STYLE_SQUARES,
}


def _hex_chrome_rgb() -> tuple[int, int, int]:
    # [BEACON] Mesh lines use graphite-5 (cool mid-grey) not vermillion — hairlines
    # should read as structural, not chromatic, on the dark navy substrate.
    if current_color_scheme() == "beacon":
        return _scheme_rgb("graphite_5")
    return _scheme_rgb("accent_dark")


class SidebarPanel(QWidget):
    """Right rail with stylized wireframe background and drag/resize behavior."""

    visibility_changed = pyqtSignal(bool)
    width_changed = pyqtSignal(int)
    moved = pyqtSignal(int)  # emits new x after horizontal drag

    def __init__(self, content: QWidget, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Pattern-only backing: the cached pixmap carries the hex/square lines
        # without a solid fill, so the rail does not add a grey slab over video.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, False)
        self._saved_width = 320
        self._min_w = 240
        self._max_w = 560
        self._handle_side = "left"
        self._dragging: tuple[int, int] | None = None
        self._moving: tuple[int, int] | None = None
        self._anim: QVariantAnimation | None = None
        self._background_style = _PANEL_BG_STYLE_HEX
        self._mesh_cache: QPixmap | None = None

        self._root = QHBoxLayout(self)
        self._root.setContentsMargins(0, 0, 0, 0)
        self._root.setSpacing(0)

        self._handle = QWidget()
        self._handle.setFixedWidth(8)
        self._handle.setCursor(Qt.CursorShape.SizeHorCursor)
        self._handle.setStyleSheet("background: transparent;")
        self._handle.setMouseTracking(True)

        self._chrome = QFrame()
        self._chrome.setObjectName("sidebarChrome")
        # [NO GRAPHICS EFFECT] — QGraphicsEffect forces the entire widget subtree
        # into an off-screen pixmap on every repaint, which tanks FPS when the
        # video feed is live. Shadow is achieved without the effect.

        chrome_vbox = QVBoxLayout(self._chrome)
        chrome_vbox.setContentsMargins(0, 0, 0, 0)
        chrome_vbox.setSpacing(0)

        self._move_bar = QWidget()
        self._move_bar.setFixedHeight(14)
        self._move_bar.setCursor(Qt.CursorShape.OpenHandCursor)
        self._move_bar.installEventFilter(self)
        chrome_vbox.addWidget(self._move_bar)
        chrome_vbox.addWidget(content, stretch=1)

        self._apply_handle_side_layout()

        self.setFixedWidth(self._saved_width)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.refresh_theme()

    def set_handle_side(self, side: str) -> None:
        normalized = "right" if str(side or "").strip().lower() == "right" else "left"
        if normalized == self._handle_side:
            return
        self._handle_side = normalized
        self._apply_handle_side_layout()

    def _apply_handle_side_layout(self) -> None:
        while self._root.count():
            item = self._root.takeAt(0)
            if item is not None:
                item.widget()
        if self._handle_side == "right":
            self._root.addWidget(self._chrome, stretch=1)
            self._root.addWidget(self._handle)
        else:
            self._root.addWidget(self._handle)
            self._root.addWidget(self._chrome, stretch=1)

    @staticmethod
    def normalize_background_style(style: str) -> str:
        return _PANEL_BG_STYLE_ALIASES.get(str(style or "").strip().lower(), _PANEL_BG_STYLE_HEX)

    def set_background_style(self, style: str) -> None:
        normalized = self.normalize_background_style(style)
        if normalized == self._background_style:
            return
        self._background_style = normalized
        self._mesh_cache = None  # invalidate: mesh pattern changed
        self.update()

    def content_frame(self) -> QFrame:
        return self._chrome

    def refresh_theme(self) -> None:
        # Force every child widget transparent so only the chrome mesh shows.
        r, g, b = _hex_chrome_rgb()
        ad = f"rgba({r},{g},{b}"
        if current_color_scheme() == "beacon":
            # [BEACON] Tabs: graphite hairlines, vermillion bottom bar on selected.
            vr, vg, vb = _scheme_rgb("accent_dark")
            verm = f"rgba({vr},{vg},{vb}"
            self._chrome.setStyleSheet(
                "QFrame#sidebarChrome, QFrame#sidebarChrome * { background: transparent; }"
                " QTabWidget::pane { background: transparent; border: none; }"
                " QTabBar { background: transparent; }"
                f" QTabBar::tab {{ background: transparent; "
                f"border: 1px solid {ad},0.30); color: {ad},1.0); padding: 4px 10px; font-size: 10px; }}"
                f" QTabBar::tab:selected {{ background: transparent; "
                f"border: 1px solid {ad},0.50); border-bottom: 2px solid {verm},1.0); color: {verm},1.0); }}"
                f" QTabBar::tab:hover:!selected {{ background: transparent; color: {ad},1.0); }}"
                " QStackedWidget { background: transparent; }"
            )
        else:
            self._chrome.setStyleSheet(
                "QFrame#sidebarChrome, QFrame#sidebarChrome * { background: transparent; }"
                " QTabWidget::pane { background: transparent; border: none; }"
                " QTabBar { background: transparent; }"
                f" QTabBar::tab {{ background: transparent; "
                f"border: 1px dotted {ad},0.28); color: {ad},1.0); padding: 4px 10px; font-size: 10px; }}"
                f" QTabBar::tab:selected {{ background: transparent; "
                f"border: 1px dotted {ad},0.55); border-bottom: 2px solid {ad},0.72); color: {ad},1.0); }}"
                f" QTabBar::tab:hover:!selected {{ background: transparent; color: {ad},1.0); }}"
                " QStackedWidget { background: transparent; }"
            )
        self._move_bar.setStyleSheet(
            f"background: transparent; border-bottom: 1px solid {get_hud_strip_border_css()};"
        )
        self._mesh_cache = None  # invalidate: theme color changed
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._mesh_cache = None  # invalidate: dimensions changed

    def paintEvent(self, event) -> None:
        """Blit the pre-rendered mesh pixmap — O(1) GPU memcopy per frame."""
        if self._mesh_cache is None or self._mesh_cache.size() != self.size():
            self._rebuild_mesh_cache()
        if self._mesh_cache is not None:
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self._mesh_cache)
            painter.end()

    def _rebuild_mesh_cache(self) -> None:
        """Render the mesh once into a QPixmap so paintEvent is a cheap blit."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            self._mesh_cache = None
            return
        px = QPixmap(w, h)
        px.fill(Qt.GlobalColor.transparent)
        painter = QPainter(px)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        r, g, b = _hex_chrome_rgb()
        pen = QPen(QColor(r, g, b, 55))
        pen.setWidthF(0.7)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if self._background_style == _PANEL_BG_STYLE_SQUARES:
            self._paint_square_mesh(painter)
        else:
            self._paint_hex_mesh(painter)
        painter.end()
        self._mesh_cache = px

    def _paint_hex_mesh(self, painter: QPainter) -> None:
        rct = self.rect()
        row = 0
        cy = _HEX_R * 0.5
        while cy - _HEX_R < rct.height() + _HEX_R:
            offset = _HEX_H_STEP * 0.5 if row % 2 else 0.0
            cx = offset
            while cx - _HEX_R < rct.width() + _HEX_R:
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(cx + _HEX_R * math.cos(a), cy + _HEX_R * math.sin(a))
                            for a in _HEX_ANGLES
                        ]
                    )
                )
                cx += _HEX_H_STEP
            cy += _HEX_V_STEP
            row += 1

    def _paint_square_mesh(self, painter: QPainter) -> None:
        rct = self.rect()
        y = -_SQUARE_SIZE
        row = 0
        while y < rct.height() + _SQUARE_SIZE:
            x = -_SQUARE_SIZE if row % 2 == 0 else 0
            while x < rct.width() + _SQUARE_SIZE:
                painter.drawRect(int(x), int(y), _SQUARE_SIZE, _SQUARE_SIZE)
                x += _SQUARE_STEP
            y += _SQUARE_STEP
            row += 1

    def _stop_anim(self) -> None:
        if self._anim is not None:
            self._anim.stop()
            self._anim = None

    def eventFilter(self, obj: object, event: QEvent) -> bool:
        if obj is self._move_bar:
            t = event.type()
            if t == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    self._moving = (int(event.globalPosition().x()), self.x())
                    self._move_bar.setCursor(Qt.CursorShape.ClosedHandCursor)
                    return True
            elif t == QEvent.Type.MouseMove and self._moving is not None:
                gx = int(event.globalPosition().x())
                dx = gx - self._moving[0]
                new_x = self._moving[1] + dx
                p = self.parentWidget()
                if p:
                    new_x = max(0, min(p.width() - self.width(), new_x))
                self.move(new_x, self.y())
                self.moved.emit(new_x)
                return True
            elif t == QEvent.Type.MouseButtonRelease and self._moving is not None:
                self._moving = None
                self._move_bar.setCursor(Qt.CursorShape.OpenHandCursor)
                return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            ch = self.childAt(event.position().toPoint())
            if ch is self._handle:
                self._dragging = (int(event.globalPosition().x()), self.width())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._dragging is not None:
            gx = int(event.globalPosition().x())
            if self._handle_side == "right":
                dx = gx - self._dragging[0]
            else:
                dx = self._dragging[0] - gx
            nw = max(self._min_w, min(self._max_w, self._dragging[1] + dx))
            self.setFixedWidth(nw)
            self._saved_width = nw
            self.width_changed.emit(nw)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._dragging = None
        super().mouseReleaseEvent(event)

    def dismiss_animated(self) -> None:
        self._stop_anim()
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(160)
        self._anim.setStartValue(self.width())
        self._anim.setEndValue(0)
        self._anim.setEasingCurve(QEasingCurve.Type.InCubic)
        self._anim.valueChanged.connect(lambda v: self.setFixedWidth(max(0, int(v))))
        self._anim.finished.connect(self._on_dismiss_done)
        self._anim.start()

    def _on_dismiss_done(self) -> None:
        self._anim = None
        self.hide()
        self.setFixedWidth(self._saved_width)
        self.visibility_changed.emit(False)

    def restore(self) -> None:
        self._stop_anim()
        self.setFixedWidth(0)
        self.show()
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(220)
        self._anim.setStartValue(0)
        self._anim.setEndValue(self._saved_width)
        self._anim.setEasingCurve(QEasingCurve.Type.OutBack)
        self._anim.valueChanged.connect(lambda v: self.setFixedWidth(max(0, int(v))))
        self._anim.finished.connect(self._on_restore_done)
        self._anim.start()

    def _on_restore_done(self) -> None:
        self._anim = None
        self.setFixedWidth(self._saved_width)
        self.visibility_changed.emit(True)
