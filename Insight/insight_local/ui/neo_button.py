from __future__ import annotations

from PyQt6.QtCore import QPoint, QSize, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPolygon, QRegion
from PyQt6.QtWidgets import QAbstractButton, QSizePolicy, QWidget

from .theme import _scheme_rgb, contrast_text_hex


_SKEW_PX = 8


class ParallelogramButton(QAbstractButton):
    """
    Interactive button drawn as a left-leaning parallelogram.

    Shape (skew leans left side down, right side up):
        (skew, 0) __________ (w, 0)
       (0, h) ____________ (w-skew, h)

    variant="primary" — filled accent background (CTA, ON/checked state).
    variant="ghost"   — panel background with accent border and text.
    """

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
        *,
        variant: str = "ghost",
    ) -> None:
        super().__init__(parent)
        self._variant = variant
        self._hovered = False
        self.setText(text)
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)

    # ---- Qt overrides ----

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._rebuild_mask()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._rebuild_mask()

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        tw = fm.horizontalAdvance(self.text() or " ")
        th = fm.height()
        return QSize(tw + 20 + _SKEW_PX * 2, th + 12)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        poly = _para_poly(w, h)
        bg, border_col, text_col = self._resolve_colors()

        painter.setBrush(bg)
        painter.setPen(QPen(border_col, 1.0))
        painter.drawPolygon(poly)

        font = painter.font()
        font.setPixelSize(10)
        font.setWeight(QFont.Weight.DemiBold)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.8)
        painter.setFont(font)
        painter.setPen(text_col)
        painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), self.text().upper())
        painter.end()

    # ---- internals ----

    def _rebuild_mask(self) -> None:
        w, h = self.width(), self.height()
        if w > 0 and h > 0:
            self.setMask(QRegion(_para_poly(w, h)))

    def _resolve_colors(self) -> tuple[QColor, QColor, QColor]:
        ar, ag, ab = _scheme_rgb("accent_dark")
        pr, pg, pb = _scheme_rgb("panel")
        hr, hg, hb = _scheme_rgb("hover")
        enabled = self.isEnabled()
        checked = self.isChecked()
        pressed = self.isDown()
        is_primary = self._variant == "primary" or checked

        if not enabled:
            return (
                QColor(pr, pg, pb, 90),
                QColor(ar, ag, ab, 45),
                QColor(ar, ag, ab, 65),
            )

        if is_primary:
            alpha = 230 if pressed else (248 if self._hovered else 210)
            return (
                QColor(ar, ag, ab, alpha),
                QColor(ar, ag, ab, min(255, alpha + 20)),
                QColor(contrast_text_hex("accent_dark")),
            )

        # ghost variant
        if pressed:
            fill = QColor(hr, hg, hb, 210)
            b_alpha = 130
        elif self._hovered:
            fill = QColor(hr, hg, hb, 150)
            b_alpha = 105
        else:
            fill = QColor(pr, pg, pb, 150)
            b_alpha = 72
        return (
            fill,
            QColor(ar, ag, ab, b_alpha),
            QColor(ar, ag, ab, 190),
        )


def _para_poly(w: int, h: int) -> QPolygon:
    return QPolygon([
        QPoint(_SKEW_PX, 0),
        QPoint(w, 0),
        QPoint(w - _SKEW_PX, h),
        QPoint(0, h),
    ])
