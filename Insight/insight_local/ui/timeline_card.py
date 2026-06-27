from __future__ import annotations
import time
from typing import Any, Callable, Optional

from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QBrush, QColor, QLinearGradient, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .media_utils import pixmap_from_b64_jpeg
from .theme import _scheme_rgb, current_color_scheme, detection_label_text, text_css, text_hex, theme_rgba

# Pre-rendered card backgrounds (shared across all instances)
_card_bg_normal: Optional[QPixmap] = None
_card_bg_hover: Optional[QPixmap] = None
_card_bg_key: Optional[tuple[int, int, str]] = None

_LEGACY_PANEL_RGB = (198, 40, 40)
_LEGACY_ACCENT_RGB = (138, 20, 20)
_LEGACY_TEXT_RGB = (20, 8, 8)
_LEGACY_WARN_RGB = (160, 18, 18)
_LEGACY_CAUTION_RGB = (181, 137, 0)


def _is_fire() -> bool:
    return current_color_scheme() == "fire"


def _shade(rgb: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    return tuple(max(0, min(255, channel + delta)) for channel in rgb)  # type: ignore[return-value]


def _legacy_rgba(rgb: tuple[int, int, int], alpha: float) -> str:
    return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha:.2f})"


def _card_surface_colors(*, hover: bool) -> tuple[QColor, QColor, QColor]:
    if _is_fire():
        panel = _scheme_rgb("panel")
        accent = _scheme_rgb("accent_dark")
        top = _shade(panel, 10 if hover else 6)
        bottom = _shade(panel, -8 if hover else -12)
        border_alpha = 96 if hover else 74
        return (
            QColor(top[0], top[1], top[2], 226 if hover else 212),
            QColor(bottom[0], bottom[1], bottom[2], 240 if hover else 230),
            QColor(accent[0], accent[1], accent[2], border_alpha),
        )
    if hover:
        return (
            QColor(_LEGACY_ACCENT_RGB[0], _LEGACY_ACCENT_RGB[1], _LEGACY_ACCENT_RGB[2], 242),
            QColor(_LEGACY_ACCENT_RGB[0], _LEGACY_ACCENT_RGB[1], _LEGACY_ACCENT_RGB[2], 248),
            QColor(_LEGACY_ACCENT_RGB[0], _LEGACY_ACCENT_RGB[1], _LEGACY_ACCENT_RGB[2], 110),
        )
    return (
        QColor(_LEGACY_PANEL_RGB[0], _LEGACY_PANEL_RGB[1], _LEGACY_PANEL_RGB[2], 226),
        QColor(_LEGACY_PANEL_RGB[0], _LEGACY_PANEL_RGB[1], _LEGACY_PANEL_RGB[2], 236),
        QColor(_LEGACY_ACCENT_RGB[0], _LEGACY_ACCENT_RGB[1], _LEGACY_ACCENT_RGB[2], 72),
    )


def _format_confidence_percent(value: object, *, normalized: bool = True) -> str:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    pct = int(round(raw * 100.0)) if normalized else int(round(raw))
    pct = max(0, min(100, pct))
    prefix = "~" if pct < 50 else ("+" if pct > 65 else "")
    return f"{prefix}{pct}%"


def _ensure_card_backgrounds(w: int, h: int) -> None:
    global _card_bg_normal, _card_bg_hover, _card_bg_key
    key = (w, h, current_color_scheme())
    if _card_bg_key == key and _card_bg_normal is not None and _card_bg_hover is not None:
        return
    _card_bg_key = key
    for hover in (False, True):
        pm = QPixmap(w, h)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        grad = QLinearGradient(0, 0, w, h)
        grad_start, grad_end, border = _card_surface_colors(hover=hover)
        grad.setColorAt(0.0, grad_start)
        grad.setColorAt(1.0, grad_end)
        p.setPen(QPen(border, 1))
        p.setBrush(QBrush(grad))
        p.drawRect(0, 0, w - 1, h - 1)
        p.end()
        if hover:
            _card_bg_hover = pm
        else:
            _card_bg_normal = pm


class TimelineCardWidget(QWidget):
    """History card with TTL ring and delete control."""

    @classmethod
    def refresh_theme(cls) -> None:
        global _card_bg_normal, _card_bg_hover, _card_bg_key
        _card_bg_normal = None
        _card_bg_hover = None
        _card_bg_key = None

    def __init__(
        self,
        entry: dict[str, Any],
        on_open: Callable[[], None],
        on_delete: Callable[[], None],
        ttl_seconds: float = 40.0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._captured_at = float(entry.get("captured_at", time.time()))
        self._ttl = ttl_seconds
        self._on_open = on_open
        self._on_delete = on_delete
        self._hover = False
        self._label_text = detection_label_text(entry.get("label", ""))
        self.setFixedSize(132, 156)
        _ensure_card_backgrounds(132, 156)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._shadow = QGraphicsDropShadowEffect(self)
        self._shadow.setBlurRadius(22)
        self._shadow.setOffset(0, 6)
        self.setGraphicsEffect(self._shadow)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self._thumb = QLabel()
        self._thumb.setFixedHeight(80)
        pix = pixmap_from_b64_jpeg(str(entry.get("image", "")))
        self._thumb.setPixmap(
            pix.scaled(116, 80, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        )
        self._thumb.setScaledContents(True)
        layout.addWidget(self._thumb)

        row = QHBoxLayout()
        self._label = QLabel(self._label_text)
        self._label.setTextFormat(Qt.TextFormat.PlainText)
        self._meta = QLabel(_format_confidence_percent(entry.get("confidence", 0)))
        row.addWidget(self._label)
        row.addStretch(1)
        row.addWidget(self._meta)
        layout.addLayout(row)
        self._ts = QLabel(time.strftime("%H:%M:%S", time.localtime(self._captured_at)))
        layout.addWidget(self._ts)

        self._del = QPushButton(chr(0x00D7))
        self._del.setFixedSize(20, 20)
        self._del.setParent(self)
        self._del.raise_()
        self._del.hide()
        self._del.clicked.connect(lambda: self._on_delete())
        self.apply_theme()

        self._tick = QTimer(self)
        self._tick.timeout.connect(self.update)
        self._tick.start(500)

    def apply_theme(self) -> None:
        _ensure_card_backgrounds(self.width(), self.height())
        if _is_fire():
            panel = _scheme_rgb("panel")
            accent = _scheme_rgb("accent_dark")
            shadow_rgb = _shade(panel, -12)
            self._shadow.setColor(QColor(shadow_rgb[0], shadow_rgb[1], shadow_rgb[2], 44))
            self._thumb.setStyleSheet(
                f"border: 1px solid {theme_rgba('accent_dark', 0.28)}; "
                f"background: {theme_rgba('panel', 0.34)};"
            )
            self._label.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {text_hex()};")
            self._meta.setStyleSheet(f"font-size: 9px; color: {text_css(0.70)};")
            self._ts.setStyleSheet(f"font-size: 9px; color: {text_css(0.58)};")
            self._del.setStyleSheet(
                f"QPushButton {{ border: 1px solid {theme_rgba('accent_dark', 0.34)}; "
                f"background: {theme_rgba('panel', 0.84)}; color: {text_css(0.92)}; font-size: 12px; padding: 0; }}"
                f"QPushButton:hover {{ border-color: {theme_rgba('accent_dark', 0.58)}; "
                f"background: {theme_rgba('accent_dark', 0.42)}; color: #1a1d22; }}"
                f"QPushButton:pressed {{ background: {theme_rgba('accent_dark', 0.58)}; color: #1a1d22; }}"
            )
            self.update()
            return

        self._shadow.setColor(QColor(_LEGACY_ACCENT_RGB[0], _LEGACY_ACCENT_RGB[1], _LEGACY_ACCENT_RGB[2], 30))
        self._thumb.setStyleSheet(
            f"border: 1px solid {_legacy_rgba(_LEGACY_ACCENT_RGB, 0.40)}; "
            f"background: {_legacy_rgba(_LEGACY_PANEL_RGB, 0.18)};"
        )
        self._label.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {_legacy_rgba(_LEGACY_TEXT_RGB, 1.00)};")
        self._meta.setStyleSheet(f"font-size: 9px; color: {_legacy_rgba(_LEGACY_TEXT_RGB, 0.70)};")
        self._ts.setStyleSheet(f"font-size: 9px; color: {_legacy_rgba(_LEGACY_TEXT_RGB, 0.58)};")
        self._del.setStyleSheet(
            f"QPushButton {{ border: 1px solid {_legacy_rgba(_LEGACY_ACCENT_RGB, 0.42)}; "
            f"background: {_legacy_rgba(_LEGACY_PANEL_RGB, 0.92)}; "
            f"color: {_legacy_rgba(_LEGACY_TEXT_RGB, 1.00)}; font-size: 12px; padding: 0; }}"
        )
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._del.move(self.width() - 22, 4)

    def enterEvent(self, event) -> None:
        self._hover = True
        self._del.show()
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hover = False
        self._del.hide()
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        p = event.position().toPoint()
        if self._del.isVisible() and self._del.geometry().contains(p):
            return super().mousePressEvent(event)
        self._on_open()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        # Blit pre-rendered background instead of rebuilding gradient
        bg = _card_bg_hover if self._hover else _card_bg_normal
        if bg is not None:
            painter.drawPixmap(0, 0, bg)
        super().paintEvent(event)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        now = time.time()
        elapsed = now - self._captured_at
        remaining = max(0.0, min(1.0, 1.0 - elapsed / self._ttl))
        cx, cy, r = 12, 12, 7.0
        rect = QRect(int(cx - r), int(cy - r), int(2 * r), int(2 * r))
        if _is_fire():
            accent = _scheme_rgb("accent_dark")
            warn = _scheme_rgb("privacy_warn")
            pen_bg = QPen(QColor(accent[0], accent[1], accent[2], 112))
        else:
            pen_bg = QPen(QColor(_LEGACY_ACCENT_RGB[0], _LEGACY_ACCENT_RGB[1], _LEGACY_ACCENT_RGB[2], 130))
        pen_bg.setWidthF(2.5)
        painter.setPen(pen_bg)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(rect)
        if _is_fire():
            col = QColor(accent[0], accent[1], accent[2])
            if remaining <= 0.2:
                col = QColor(warn[0], warn[1], warn[2])
            elif remaining <= 0.45:
                muted = _shade(accent, -24)
                col = QColor(muted[0], muted[1], muted[2])
        else:
            col = QColor(_LEGACY_TEXT_RGB[0], _LEGACY_TEXT_RGB[1], _LEGACY_TEXT_RGB[2])
            if remaining <= 0.2:
                col = QColor(_LEGACY_WARN_RGB[0], _LEGACY_WARN_RGB[1], _LEGACY_WARN_RGB[2])
            elif remaining <= 0.45:
                col = QColor(_LEGACY_CAUTION_RGB[0], _LEGACY_CAUTION_RGB[1], _LEGACY_CAUTION_RGB[2])
        pen_fg = QPen(col)
        pen_fg.setWidthF(2.5)
        pen_fg.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(pen_fg)
        span = int(remaining * 360 * 16)
        painter.drawArc(rect, 90 * 16, -span)
        painter.end()
