"""Early-load patch: parallelogram chrome for push/tool buttons and tab bars.

Qt Stylesheets cannot skew widgets; region masks clip chrome so only horizontal
border segments survive. We subclass before application panels import widgets,
clip styled painting to a QPainterPath, then stroke the full perimeter.

Tab strips use a per-QTabBar ``QProxyStyle`` so ``CE_TabBarTab*`` paints inside a
parallelogram clip (scroll buttons still use the base style unchanged).

Import this module before other cvops modules import ``PyQt6.QtWidgets``.
"""
from __future__ import annotations

import PyQt6.QtWidgets as _qt_widgets

from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QProxyStyle,
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QStyleOptionTab,
    QStyleOptionToolButton,
    QStylePainter,
    QTabBar,
    QTabWidget,
    QToolButton,
)

_BasePushButton = QPushButton
_BaseTabBar = QTabBar
_BaseTabWidget = QTabWidget
_BaseToolButton = QToolButton
_BUTTON_SHAPE = "parallelogram"


def normalize_cvops_button_shape(shape: object) -> str:
    value = str(shape or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "parallelogram",
        "default": "parallelogram",
        "skew": "parallelogram",
        "skewed": "parallelogram",
        "off": "none",
        "no": "none",
        "normal": "none",
        "rectangle": "none",
        "rect": "none",
        "square": "none",
        "rounded": "radial",
        "round": "radial",
        "soft": "radial",
        "soft_round": "radial",
        "octogon": "octagon",
        "wide_octagon": "octagon",
    }
    value = aliases.get(value, value)
    if value in {"none", "radial", "parallelogram", "octagon"}:
        return value
    return "parallelogram"


def set_cvops_button_shape(shape: object) -> str:
    global _BUTTON_SHAPE
    _BUTTON_SHAPE = normalize_cvops_button_shape(shape)
    app = QApplication.instance()
    if app is not None:
        for widget in app.allWidgets():
            if isinstance(widget, (_BasePushButton, _BaseToolButton, _BaseTabBar)):
                widget.updateGeometry()
                widget.update()
    return _BUTTON_SHAPE


def cvops_button_shape() -> str:
    return _BUTTON_SHAPE


def _uses_native_button_chrome() -> bool:
    return _BUTTON_SHAPE in {"none", "radial"}


def _skew_px(w: int, h: int) -> int:
    if w < 10 or h < 10:
        return 4
    return max(4, min(max(w, h) // 14, 20))


def _parallelogram_path(width: int, height: int) -> QPainterPath:
    k = _skew_px(width, height)
    path = QPainterPath()
    path.moveTo(float(k), 0.0)
    path.lineTo(float(width), 0.0)
    path.lineTo(float(width - k), float(height))
    path.lineTo(0.0, float(height))
    path.closeSubpath()
    return path


def _octagon_path(width: int, height: int) -> QPainterPath:
    c = max(6, min(22, width // 10, height // 2))
    path = QPainterPath()
    path.moveTo(float(c), 0.0)
    path.lineTo(float(width - c), 0.0)
    path.lineTo(float(width), float(c))
    path.lineTo(float(width), float(height - c))
    path.lineTo(float(width - c), float(height))
    path.lineTo(float(c), float(height))
    path.lineTo(0.0, float(height - c))
    path.lineTo(0.0, float(c))
    path.closeSubpath()
    return path


def _button_path(width: int, height: int) -> QPainterPath:
    if _BUTTON_SHAPE == "octagon":
        return _octagon_path(width, height)
    return _parallelogram_path(width, height)


def _border_pen_for_push(opt: QStyleOptionButton) -> QPen:
    from .cvops_theme import cvops_color

    accent = cvops_color("accent_active")
    line = cvops_color("line_light")

    st = opt.state
    if not (st & QStyle.StateFlag.State_Enabled):
        return QPen(QColor(line), 1.0)
    if st & QStyle.StateFlag.State_MouseOver:
        return QPen(QColor(accent), 1.35)
    if st & QStyle.StateFlag.State_HasFocus:
        return QPen(QColor(accent), 1.25)
    if st & QStyle.StateFlag.State_On:
        return QPen(QColor(accent), 1.25)
    return QPen(QColor(line), 1.15)


def _border_pen_for_tab(opt: QStyleOptionTab) -> QPen:
    from .cvops_theme import cvops_color

    accent = cvops_color("accent_active")
    line = cvops_color("line_light")

    st = opt.state
    if not (st & QStyle.StateFlag.State_Enabled):
        return QPen(QColor(line), 1.0)
    if st & QStyle.StateFlag.State_Selected:
        return QPen(QColor(accent), 1.25)
    if st & QStyle.StateFlag.State_MouseOver:
        return QPen(QColor(accent), 1.35)
    if st & QStyle.StateFlag.State_HasFocus:
        return QPen(QColor(accent), 1.2)
    return QPen(QColor(line), 1.15)


class _CvOpsParallelogramTabProxyStyle(QProxyStyle):
    """Clip Mac/Fusion tab painting to the skewed quadrilateral (no stroke here)."""

    def drawControl(self, element, opt, painter, widget=None):  # type: ignore[override]
        if isinstance(opt, QStyleOptionTab) and widget is not None:
            if widget.property("cvopsNoSkew") is True or _uses_native_button_chrome():
                return super().drawControl(element, opt, painter, widget)
            if element in (
                QStyle.ControlElement.CE_TabBarTab,
                QStyle.ControlElement.CE_TabBarTabShape,
                QStyle.ControlElement.CE_TabBarTabLabel,
            ):
                r = opt.rect
                path = _button_path(r.width(), r.height())
                path.translate(float(r.x()), float(r.y()))
                painter.save()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                painter.setClipPath(path)
                super().drawControl(element, opt, painter, widget)
                painter.restore()
                return
        super().drawControl(element, opt, painter, widget)


def _border_pen_for_tool(opt: QStyleOptionToolButton) -> QPen:
    from .cvops_theme import cvops_color

    accent = cvops_color("accent_active")
    line = cvops_color("line_light")

    st = opt.state
    if not (st & QStyle.StateFlag.State_Enabled):
        return QPen(QColor(line), 1.0)
    if st & QStyle.StateFlag.State_MouseOver:
        return QPen(QColor(accent), 1.35)
    if st & QStyle.StateFlag.State_HasFocus:
        return QPen(QColor(accent), 1.25)
    if (st & QStyle.StateFlag.State_On) or (st & QStyle.StateFlag.State_Sunken):
        return QPen(QColor(accent), 1.25)
    return QPen(QColor(line), 1.15)


class CvOpsParallelogramPushButton(_BasePushButton):
    def hitButton(self, pos: QPoint) -> bool:
        if self.property("cvopsNoSkew") is True or _uses_native_button_chrome():
            return super().hitButton(pos)
        return _button_path(self.width(), self.height()).contains(QPointF(pos))

    def paintEvent(self, event) -> None:
        if self.property("cvopsNoSkew") is True or _uses_native_button_chrome():
            super().paintEvent(event)
            return
        path = _button_path(self.width(), self.height())
        painter = QStylePainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipPath(path)
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        painter.drawControl(QStyle.ControlElement.CE_PushButton, opt)
        painter.setClipping(False)

        pen = _border_pen_for_push(opt)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.strokePath(path, pen)


class CvOpsParallelogramToolButton(_BaseToolButton):
    def hitButton(self, pos: QPoint) -> bool:
        if self.property("cvopsNoSkew") is True or _uses_native_button_chrome():
            return super().hitButton(pos)
        if self.property("isTitle") is True:
            return super().hitButton(pos)
        return _button_path(self.width(), self.height()).contains(QPointF(pos))

    def paintEvent(self, event) -> None:
        if (
            self.property("cvopsNoSkew") is True
            or self.property("isTitle") is True
            or _uses_native_button_chrome()
        ):
            super().paintEvent(event)
            return
        path = _button_path(self.width(), self.height())
        painter = QStylePainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setClipPath(path)
        opt = QStyleOptionToolButton()
        self.initStyleOption(opt)
        painter.drawComplexControl(QStyle.ComplexControl.CC_ToolButton, opt)
        painter.setClipping(False)

        pen = _border_pen_for_tool(opt)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.strokePath(path, pen)


class CvOpsParallelogramTabBar(_BaseTabBar):
    """Skewed tab shape + AA outline; hit-testing matches the parallelogram."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        app = QApplication.instance()
        base_style = app.style() if app is not None else self.style()
        self.setStyle(_CvOpsParallelogramTabProxyStyle(base_style))

    def tabAt(self, pos: QPoint) -> int:
        if self.property("cvopsNoSkew") is True or _uses_native_button_chrome():
            return super().tabAt(pos)
        for i in range(self.count()):
            rect = super().tabRect(i)
            if not rect.isValid():
                continue
            plg = _button_path(rect.width(), rect.height())
            plg.translate(float(rect.x()), float(rect.y()))
            if plg.contains(QPointF(pos)):
                return i
        return -1

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.property("cvopsNoSkew") is True or _uses_native_button_chrome():
            return
        painter = QStylePainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        opt = QStyleOptionTab()
        for i in range(self.count()):
            self.initStyleOption(opt, i)
            rect = self.tabRect(i)
            if not rect.isValid():
                continue
            path = _button_path(rect.width(), rect.height())
            path.translate(float(rect.x()), float(rect.y()))
            pen = _border_pen_for_tab(opt)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.strokePath(path, pen)


class CvOpsParallelogramTabWidget(_BaseTabWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setTabBar(CvOpsParallelogramTabBar(self))


def _apply() -> None:
    if getattr(_qt_widgets, "_cvopsParallelogramPatched", False):
        return
    _qt_widgets.QPushButton = CvOpsParallelogramPushButton
    _qt_widgets.QToolButton = CvOpsParallelogramToolButton
    _qt_widgets.QTabWidget = CvOpsParallelogramTabWidget
    _qt_widgets._cvopsParallelogramPatched = True


_apply()
