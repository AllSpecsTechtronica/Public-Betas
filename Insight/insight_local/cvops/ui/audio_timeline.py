"""Shared interactive audio-waveform timeline widget.

Imported by both VideoTestPanel (passive waveform + seek) and
AudioWaveformPlayer (standalone player for the dataset catalog).
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QFont, QGuiApplication, QPainter, QPen
from PyQt6.QtWidgets import QMenu, QSizePolicy, QWidget

from ...ui.theme import current_color_scheme, is_aurora_family_scheme, text_qcolor, theme_hex


def _fmt_ms(ms: int) -> str:
    if ms < 0:
        ms = 0
    total = ms // 1000
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _theme_qcolor(role: str, alpha: int = 255) -> QColor:
    color = QColor(theme_hex(role))
    color.setAlpha(max(0, min(255, int(alpha))))
    return color


class AudioTimeline(QWidget):
    """Interactive waveform timeline with scrubbing, zoom, scroll, and region selection.

    Interaction:
      - Click or drag          : seek playback to that position (emits seek_requested)
      - Shift + click/drag     : draw a selection region (emits selection_changed)
      - Ctrl + scroll wheel    : zoom in/out (1× – 20×), keeps the point under the
                                 mouse fixed in time
      - Scroll wheel           : pan left/right (only meaningful when zoomed)
      - Right-click            : context menu — clear selection, copy times, reset zoom
    """

    seek_requested = pyqtSignal(int)          # position_ms
    selection_changed = pyqtSignal(int, int)  # start_ms, end_ms
    selection_cleared = pyqtSignal()

    _BASE_HEIGHT = 72
    _BASE_BOTTOM_H = 22  # pixels reserved below the waveform for tick + label row

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMouseTracking(True)
        self._layout_scale: float = 1.0
        self._apply_layout_scale()

        self._duration_ms: int = 0
        self._cursor_ms: int = 0
        self._levels: list[float] = []
        self._state: str = "empty"  # empty | analyzing | ready | muted | unavailable
        self._message: str = "No audio loaded."
        self._aurora_waveform_cyan: bool = False

        # zoom / scroll
        self._zoom: float = 1.0
        self._view_start_ms: int = 0

        # selection
        self._sel_anchor_ms: Optional[int] = None
        self._sel_start_ms: Optional[int] = None
        self._sel_end_ms: Optional[int] = None

        # interaction state
        self._drag_mode: str = "none"  # "seek" | "select"

    def _scaled_px(self, px: int, *, minimum: int = 1) -> int:
        return max(minimum, int(round(px * self._layout_scale)))

    def _apply_layout_scale(self) -> None:
        height = self._scaled_px(self._BASE_HEIGHT, minimum=32)
        self.setMinimumHeight(height)
        self.setMaximumHeight(height)
        self.updateGeometry()

    def set_visual_scale(self, scale: float) -> None:
        self._layout_scale = max(0.5, min(1.5, float(scale)))
        self._apply_layout_scale()
        self.update()

    def set_aurora_waveform_cyan(self, enabled: bool) -> None:
        self._aurora_waveform_cyan = bool(enabled)
        self.update()

    def _waveform_qcolor(self, alpha: int = 255) -> QColor:
        if self._aurora_waveform_cyan and is_aurora_family_scheme(current_color_scheme()):
            color = QColor("#2BC4D9")
            color.setAlpha(max(0, min(255, int(alpha))))
            return color
        return _theme_qcolor("accent_dark", alpha)

    # ------------------------------------------------------------------
    # Public state setters
    # ------------------------------------------------------------------

    def set_duration(self, duration_ms: int) -> None:
        self._duration_ms = max(0, int(duration_ms))
        self._clamp_view()
        self.update()

    def set_cursor(self, position_ms: int) -> None:
        self._cursor_ms = max(0, int(position_ms))
        self.update()

    def reset(self) -> None:
        self._levels.clear()
        self._state = "empty"
        self._message = "No audio loaded."
        self._cursor_ms = 0
        self._zoom = 1.0
        self._view_start_ms = 0
        self._sel_anchor_ms = None
        self._sel_start_ms = None
        self._sel_end_ms = None
        self._drag_mode = "none"
        self.update()

    def set_analyzing(self) -> None:
        self._levels.clear()
        self._state = "analyzing"
        self._message = "Analyzing audio..."
        self.update()

    def set_levels(self, levels: list[float]) -> None:
        cleaned: list[float] = []
        for v in levels:
            try:
                x = float(v)
            except (TypeError, ValueError):
                continue
            if x != x or x == float("inf") or x == float("-inf"):  # NaN / non-finite
                continue
            cleaned.append(max(0.0, min(1.0, x)))
        self._levels = cleaned
        if not self._levels:
            self.set_muted()
            return
        peak = max(self._levels)
        if peak > 1e-9:
            self._state = "ready"
            self._message = ""
            self.update()
            return
        self.set_muted()

    def set_muted(self) -> None:
        self._levels.clear()
        self._state = "muted"
        self._message = "Silent or no audio content."
        self.update()

    def set_unavailable(self, message: str) -> None:
        self._levels.clear()
        self._state = "unavailable"
        self._message = message
        self.update()

    def set_selection(self, start_ms: int, end_ms: int) -> None:
        """Programmatic selection; does NOT emit selection_changed."""
        self._sel_start_ms = max(0, int(start_ms))
        self._sel_end_ms = max(0, int(end_ms))
        self._sel_anchor_ms = self._sel_start_ms
        self.update()

    def clear_selection(self) -> None:
        self._sel_start_ms = None
        self._sel_end_ms = None
        self._sel_anchor_ms = None
        self.selection_cleared.emit()
        self.update()

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------

    def _visible_ms(self) -> float:
        if self._duration_ms <= 0:
            return 1.0
        return self._duration_ms / max(1.0, self._zoom)

    def _clamp_view(self) -> None:
        if self._duration_ms <= 0:
            self._view_start_ms = 0
            return
        max_start = max(0, self._duration_ms - int(self._visible_ms()))
        self._view_start_ms = max(0, min(max_start, self._view_start_ms))

    def _ms_from_x(self, x: float) -> int:
        width = max(1, self.width())
        ms = self._view_start_ms + (x / width) * self._visible_ms()
        return max(0, min(self._duration_ms, int(ms)))

    def _x_from_ms(self, ms: int) -> int:
        width = max(1, self.width())
        visible = self._visible_ms()
        if visible <= 0:
            return 0
        return int((ms - self._view_start_ms) / visible * width)

    # ------------------------------------------------------------------
    # Mouse interaction
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            ms = self._ms_from_x(event.position().x())
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                self._drag_mode = "select"
                self._sel_anchor_ms = ms
                self._sel_start_ms = ms
                self._sel_end_ms = ms
            else:
                self._drag_mode = "seek"
                if self._duration_ms > 0:
                    self.seek_requested.emit(ms)
            self.update()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() & Qt.MouseButton.LeftButton:
            ms = self._ms_from_x(event.position().x())
            if self._drag_mode == "seek":
                if self._duration_ms > 0:
                    self.seek_requested.emit(ms)
            elif self._drag_mode == "select" and self._sel_anchor_ms is not None:
                self._sel_start_ms = min(self._sel_anchor_ms, ms)
                self._sel_end_ms = max(self._sel_anchor_ms, ms)
                self.update()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            if self._drag_mode == "select":
                if (
                    self._sel_start_ms is not None
                    and self._sel_end_ms is not None
                    and self._sel_end_ms > self._sel_start_ms
                ):
                    self.selection_changed.emit(self._sel_start_ms, self._sel_end_ms)
                else:
                    self._sel_start_ms = None
                    self._sel_end_ms = None
                    self._sel_anchor_ms = None
                    self.selection_cleared.emit()
            self._drag_mode = "none"
            self.update()
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if self._duration_ms <= 0:
            event.accept()
            return
        delta = event.angleDelta().y()
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            x = event.position().x()
            ms_under = self._ms_from_x(x)
            factor = 1.18 if delta > 0 else (1.0 / 1.18)
            self._zoom = max(1.0, min(20.0, self._zoom * factor))
            visible = self._visible_ms()
            width = max(1, self.width())
            self._view_start_ms = int(ms_under - (x / width) * visible)
            self._clamp_view()
        else:
            visible = int(self._visible_ms())
            step = max(1, visible // 10)
            self._view_start_ms += -step if delta > 0 else step
            self._clamp_view()
        event.accept()
        self.update()

    def contextMenuEvent(self, event) -> None:  # type: ignore[override]
        menu = QMenu(self)
        has_sel = self._sel_start_ms is not None and self._sel_end_ms is not None
        if has_sel:
            copy_act = QAction(
                f"Copy times  {_fmt_ms(self._sel_start_ms)} \u2013 {_fmt_ms(self._sel_end_ms)}",
                self,
            )
            copy_act.triggered.connect(self._copy_selection_times)
            menu.addAction(copy_act)
            clear_act = QAction("Clear selection", self)
            clear_act.triggered.connect(self.clear_selection)
            menu.addAction(clear_act)
            menu.addSeparator()
        zoom_reset = QAction("Reset zoom / scroll", self)
        zoom_reset.triggered.connect(self._reset_zoom)
        zoom_reset.setEnabled(self._zoom > 1.0 or self._view_start_ms > 0)
        menu.addAction(zoom_reset)
        menu.exec(event.globalPos())

    def _copy_selection_times(self) -> None:
        if self._sel_start_ms is not None and self._sel_end_ms is not None:
            QGuiApplication.clipboard().setText(
                f"{self._sel_start_ms}\u2013{self._sel_end_ms}"
            )

    def _reset_zoom(self) -> None:
        self._zoom = 1.0
        self._view_start_ms = 0
        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def sizeHint(self) -> QSize:
        return QSize(400, self._scaled_px(self._BASE_HEIGHT, minimum=32))

    def paintEvent(self, _event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        rect = self.rect()
        font = QFont(painter.font())
        font.setPixelSize(self._scaled_px(10, minimum=8))
        painter.setFont(font)
        painter.fillRect(rect, _theme_qcolor("input_fill"))

        show_ticks = rect.width() >= 200 and self._duration_ms > 0
        bottom_h = self._scaled_px(self._BASE_BOTTOM_H, minimum=12) if show_ticks else 2
        track_top = rect.top() + self._scaled_px(5, minimum=3)
        track_bottom = rect.bottom() - bottom_h
        track_h = max(4, track_bottom - track_top)
        mid_y = track_top + track_h // 2
        width = max(1, rect.width())
        visible = self._visible_ms()

        painter.setPen(QPen(_theme_qcolor("hover"), 1))
        painter.drawRect(rect.left(), track_top, width - 1, track_h)
        painter.setPen(QPen(_theme_qcolor("accent_dark", 130), 1))
        painter.drawLine(rect.left() + 1, mid_y, rect.right() - 1, mid_y)

        if self._state == "ready" and self._levels:
            count = len(self._levels)

            # Selection background
            if self._sel_start_ms is not None and self._sel_end_ms is not None:
                sx = max(0, self._x_from_ms(self._sel_start_ms))
                ex = min(width, self._x_from_ms(self._sel_end_ms))
                sel_bg = _theme_qcolor("accent_dark")
                sel_bg.setAlpha(45)
                painter.fillRect(sx, track_top + 1, max(1, ex - sx), track_h - 1, sel_bg)

            # Played-region dim overlay (left of playhead)
            if self._duration_ms > 0 and self._cursor_ms > self._view_start_ms:
                cx = max(0, min(width, self._x_from_ms(self._cursor_ms)))
                dim = QColor(0, 0, 0, 55)
                painter.fillRect(rect.left() + 1, track_top + 1, cx - 1, track_h - 1, dim)

            # Waveform bars — only samples that fall inside the visible window
            bar_colour = self._waveform_qcolor()
            painter.setPen(QPen(bar_colour, 1))
            for x in range(width):
                if self._duration_ms > 0:
                    ms_left = self._view_start_ms + (x / width) * visible
                    ms_right = self._view_start_ms + ((x + 1) / width) * visible
                    i0 = max(0, min(count - 1, int((ms_left / self._duration_ms) * count)))
                    i1 = max(0, min(count, int((ms_right / self._duration_ms) * count)))
                else:
                    i0 = max(0, min(count - 1, int((x / width) * count)))
                    i1 = max(0, min(count, int(((x + 1) / width) * count)))
                if i1 <= i0:
                    i1 = min(count, i0 + 1)
                if i0 >= count:
                    continue
                level = max(self._levels[i0:i1]) if i1 <= count else self._levels[i0]
                half_h = max(1, int(level * (track_h / 2)))
                painter.drawLine(x, mid_y - half_h, x, mid_y + half_h)

            # Selection edge lines and time labels
            if self._sel_start_ms is not None and self._sel_end_ms is not None:
                sx = self._x_from_ms(self._sel_start_ms)
                ex = self._x_from_ms(self._sel_end_ms)
                painter.setPen(QPen(_theme_qcolor("strip_soft"), 1))
                if 0 <= sx <= width:
                    painter.drawLine(sx, track_top, sx, track_bottom)
                if 0 <= ex <= width:
                    painter.drawLine(ex, track_top, ex, track_bottom)
                label_y = track_bottom + self._scaled_px(11, minimum=8)
                painter.setPen(QPen(_theme_qcolor("strip_soft"), 1))
                if 0 <= sx <= width:
                    painter.drawText(sx + 2, label_y, _fmt_ms(self._sel_start_ms))
                if 0 <= ex <= width and ex - sx > 36:
                    painter.drawText(ex + 2, label_y, _fmt_ms(self._sel_end_ms))

        else:
            painter.setPen(QPen(text_qcolor(0.58), 1))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self._message)

        # Playhead
        if self._duration_ms > 0:
            cx = self._x_from_ms(self._cursor_ms)
            if 0 <= cx <= width:
                painter.setPen(QPen(_theme_qcolor("privacy_warn"), 2))
                painter.drawLine(cx, rect.top(), cx, rect.bottom())

        # Tick marks and time labels — omit when the widget is too narrow to fit them
        if show_ticks:
            painter.setPen(QPen(text_qcolor(0.50), 1))
            ticks = 6
            for i in range(ticks + 1):
                frac = i / ticks
                tick_ms = int(self._view_start_ms + frac * visible)
                tick_ms = max(0, min(self._duration_ms, tick_ms))
                tx = int(frac * (width - 1))
                painter.drawLine(
                    tx,
                    track_bottom,
                    tx,
                    track_bottom + self._scaled_px(4, minimum=3),
                )
                painter.drawText(tx + 2, rect.bottom() - 2, _fmt_ms(tick_ms))

        # Zoom indicator (top-right corner, only when zoomed)
        if self._zoom > 1.05:
            painter.setPen(QPen(text_qcolor(0.42), 1))
            painter.drawText(
                rect.right() - self._scaled_px(52, minimum=34),
                rect.top() + self._scaled_px(12, minimum=10),
                f"{self._zoom:.1f}\u00d7",
            )
