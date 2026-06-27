from __future__ import annotations

import os
import re
from weakref import WeakKeyDictionary
from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QEvent, QObject, QSize, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLabel,
    QLayout,
    QLineEdit,
    QProgressBar,
    QSizePolicy,
    QStyle,
    QStyleOptionComboBox,
    QStylePainter,
    QToolButton,
    QWidget,
)

from ...ui.theme import (
    current_color_scheme,
    is_aurora_family_scheme,
    text_css,
    text_qcolor,
    theme_hex,
    theme_rgba,
    themed_css,
)

from .backdrop_blend import WorkspaceBackdropBlend, scale_rgba_string
from .patch_parallelogram_buttons import cvops_button_shape

_WIDGET_SIZE_MAX = 16_777_215


# Preferred design families (often not installed) -> the first few that DO exist
# on this machine. Sans/mono are resolved separately so monospace requests land
# on a real fixed-pitch face rather than the proportional system font.
_CVOPS_WANTED_SANS = ("IBM Plex Sans", "Roboto", "Segoe UI", "Inter")
_CVOPS_WANTED_MONO = (
    "JetBrains Mono", "IBM Plex Mono", "SFMono-Regular", "SF Mono",
    "Cascadia Code", "Consolas", "Fira Code",
)
_CVOPS_SANS_FALLBACK_ORDER = (
    "SF Pro Text", "SF Pro", "Helvetica Neue", "Helvetica",
    "Segoe UI", "Roboto", "Arial",
)
_CVOPS_MONO_FALLBACK_ORDER = (
    "SF Mono", "SFMono-Regular", "Menlo", "Monaco",
    "JetBrains Mono", "Cascadia Code", "Consolas", "Courier New",
)


def install_cvops_font_substitutions() -> str:
    """Route the design's preferred (frequently un-installed) font families onto
    concrete faces that exist on this machine, and return the chosen base UI family.

    Why this matters for crispness: the QSS specifies families like ``JetBrains
    Mono`` / ``IBM Plex Sans`` that aren't installed (e.g. on macOS). Qt's QSS
    parser only honours the FIRST family in a ``font-family`` comma list, so every
    such request collapses onto the hidden ``.AppleSystemUIFont`` — a proportional
    system font Qt renders less cleanly than a concrete family, and never the
    intended monospace. Registering substitutions (consulted during font matching,
    including QSS) sends those names to real installed faces — Menlo for monospace
    (fixed-pitch), SF Pro Text for sans — so text renders sharply everywhere
    without editing hundreds of stylesheet rules. No-ops on platforms where the
    requested families already exist (e.g. Segoe UI on Windows).
    """
    try:
        available = set(QFontDatabase.families())
    except Exception:
        return ""

    sans_fallbacks = [f for f in _CVOPS_SANS_FALLBACK_ORDER if f in available]
    mono_fallbacks = [f for f in _CVOPS_MONO_FALLBACK_ORDER if f in available]

    for fam in _CVOPS_WANTED_SANS:
        if fam not in available and sans_fallbacks:
            QFont.insertSubstitutions(fam, sans_fallbacks)
    for fam in _CVOPS_WANTED_MONO:
        if fam not in available and mono_fallbacks:
            QFont.insertSubstitutions(fam, mono_fallbacks)

    # Concrete base family for the application default font. Prefer an installed
    # design sans; otherwise keep IBM Plex Sans (its substitution will resolve it).
    return sans_fallbacks[0] if sans_fallbacks else ""


_UI_SCALE_MIN_PCT = 70
_UI_SCALE_MAX_PCT = 140
_UI_SCALE_BASE_WIDTH = 1360.0
_UI_SCALE_BASE_HEIGHT = 900.0
_QSS_PX_RE = re.compile(r"(?P<sign>-?)(?P<value>\d+(?:\.\d+)?)px")
_STYLE_FACTORIES: WeakKeyDictionary[QWidget, Callable[[], str]] = WeakKeyDictionary()
_ORIGINAL_WIDGET_SET_STYLESHEET: Optional[Callable[..., None]] = None
_STYLESHEET_NORMALIZER_ACTIVE = False
_CVOPS_RUNTIME_ROLE_OVERRIDES: dict[str, str] = {}


# ---------------------------------------------------------------------------
# QLabel text-selectability filter
# ---------------------------------------------------------------------------

class _SelectableLabelsFilter(QObject):
    """Application-level event filter that makes every QLabel text-selectable.

    Intercepts the Polish event (fired when a widget's style is applied) so
    newly created labels are caught automatically without requiring individual
    call-sites to set TextInteractionFlags.
    """

    _SELECT_FLAGS = (
        Qt.TextInteractionFlag.TextSelectableByMouse
        | Qt.TextInteractionFlag.TextSelectableByKeyboard
    )

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if isinstance(obj, QLabel) and event.type() == QEvent.Type.Polish:
            existing = obj.textInteractionFlags()
            obj.setTextInteractionFlags(existing | self._SELECT_FLAGS)
        return False  # never consume the event


_SELECTABLE_FILTER: Optional[_SelectableLabelsFilter] = None


def _plain_label_text(widget: QWidget) -> str:
    text_getter = getattr(widget, "text", None)
    if not callable(text_getter):
        return ""
    try:
        text = str(text_getter() or "")
    except Exception:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    return text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")


_TITLE_OBJECT_NAMES = {"cvOpsSplitPaneTitle"}


def _is_title_widget(widget: Optional[QWidget]) -> bool:
    if widget is None:
        return False
    return widget.property("isTitle") is True or widget.objectName() in _TITLE_OBJECT_NAMES


def fit_title_widget_to_text(widget: Optional[QWidget]) -> None:
    """Keep title chrome wide enough for the rendered label and Qt button chrome."""
    if not _is_title_widget(widget):
        return
    text = _plain_label_text(widget).strip()
    if not text:
        return
    parent = widget.parentWidget()
    anchor_width = parent.width() if parent is not None and parent.width() > 0 else widget.width()
    pad_x = max(4, int(round(max(80, anchor_width) * 0.01)))
    try:
        metrics = widget.fontMetrics()
        # Beacon title QSS renders uppercase with letter spacing; measure that
        # variant too so title chips do not elide at narrow splitter widths.
        text_width = max(metrics.horizontalAdvance(text), metrics.horizontalAdvance(text.upper()))
    except Exception:
        return
    if isinstance(widget, QLabel):
        widget.setContentsMargins(pad_x, 0, pad_x, 0)
    try:
        hint_width = max(1, widget.sizeHint().width())
    except Exception:
        hint_width = 1
    if isinstance(widget, QToolButton):
        # QToolButton.sizeHint() already accounts for arrow/icon chrome and
        # style padding. Add only a narrow anti-elide guard.
        target_width = max(1, hint_width + 4)
    else:
        target_width = max(1, text_width + (2 * pad_x) + 2, hint_width + 2)
    policy = widget.sizePolicy()
    if policy.horizontalPolicy() != QSizePolicy.Policy.Fixed:
        policy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        widget.setSizePolicy(policy)
    widget.setMinimumWidth(target_width)
    widget.setMaximumWidth(target_width)


class _TitleFitFilter(QObject):
    """Application-level title sizing for all title tag widgets."""

    _EVENTS = {
        QEvent.Type.Polish,
        QEvent.Type.Resize,
        QEvent.Type.FontChange,
        QEvent.Type.StyleChange,
        QEvent.Type.ParentChange,
        QEvent.Type.DynamicPropertyChange,
        QEvent.Type.LayoutRequest,
    }

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if isinstance(obj, QWidget) and event.type() in self._EVENTS:
            fit_title_widget_to_text(obj)
            if event.type() == QEvent.Type.Resize:
                for child in obj.findChildren(QWidget):
                    if _is_title_widget(child):
                        fit_title_widget_to_text(child)
        return False


_TITLE_FIT_FILTER: Optional[_TitleFitFilter] = None


def _combo_current_icon(combo: QComboBox) -> QIcon:
    """QComboBox.currentIcon exists in Qt6 C++ but is not always bound in PyQt6; use itemIcon fallback."""
    getter = getattr(combo, "currentIcon", None)
    if callable(getter):
        try:
            return getter()
        except Exception:
            pass
    idx = combo.currentIndex()
    if idx >= 0:
        return combo.itemIcon(idx)
    return QIcon()


class _ChamferComboPaintFilter(QObject):
    """App-level event filter: replaces each QComboBox paint event with chamfered rendering.

    QProxyStyle cannot reliably intercept QComboBox painting when app.setStyleSheet()
    is active (Qt routes through QStyleSheetStyle internally). Intercepting the Paint
    event directly and owning the full draw sequence is the only crash-safe approach.
    """

    @staticmethod
    def _chamfer_path(rect, c: int) -> QPainterPath:
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        c = max(1, min(c, max(1, w // 2), max(1, h // 2)))
        path = QPainterPath()
        path.moveTo(float(x), float(y))
        path.lineTo(float(x + w), float(y))
        path.lineTo(float(x + w), float(y + h - c))
        path.lineTo(float(x + w - c), float(y + h))
        path.lineTo(float(x), float(y + h))
        path.closeSubpath()
        return path

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if not isinstance(obj, QComboBox):
            return False
        if event.type() != QEvent.Type.Paint:
            return False
        self._paint_chamfer(obj)
        return True  # consumed — widget's own paintEvent is skipped

    def _paint_chamfer(self, combo: QComboBox) -> None:
        # Do not call QComboBox.initStyleOption — it is a *protected* C++ hook.
        # PyQt raises RuntimeError for some combo instances (e.g. proxied widgets
        # during labeling). Populate QStyleOptionComboBox via the public API only.
        opt = QStyleOptionComboBox()
        opt.initFrom(combo)
        opt.editable = combo.isEditable()
        opt.frame = True
        opt.currentText = combo.currentText()
        opt.currentIcon = _combo_current_icon(combo)
        ic = combo.iconSize()
        if ic.isValid() and ic.width() > 0 and ic.height() > 0:
            opt.iconSize = ic
        else:
            pm = combo.style().pixelMetric(QStyle.PixelMetric.PM_SmallIconSize, None, combo)
            opt.iconSize = QSize(max(1, int(pm)), max(1, int(pm)))
        # Only paint the arrow indicator via the native style — NOT the frame.
        # The native combo frame (SC_ComboBoxFrame) is where macOS/Fusion inject a
        # vertical gradient/bevel ("sheen"); we draw a flat fill ourselves instead
        # so dark mode stays uniformly matte regardless of the platform style.
        opt.subControls = QStyle.SubControl.SC_ComboBoxArrow
        opt.activeSubControls = QStyle.SubControl.SC_None
        h = max(1, combo.height())
        chamfer = max(5, min(12, h // 3))
        path = self._chamfer_path(combo.rect(), chamfer)

        p = QStylePainter(combo)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setClipPath(path)
        # Flat fill matching the QComboBox QSS background (panel_bg == bg_graphite).
        p.fillPath(path, cvops_qcolor("bg_graphite"))
        p.drawComplexControl(QStyle.ComplexControl.CC_ComboBox, opt)
        p.drawControl(QStyle.ControlElement.CE_ComboBoxLabel, opt)
        # stroke the chamfer outline with clip off so the edge line is fully visible
        p.setClipping(False)
        pen = QPen(cvops_qcolor("line_med", 220))
        pen.setWidthF(1.0)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
        p.end()


_CHAMFER_COMBO_FILTER: Optional[_ChamferComboPaintFilter] = None
_COMBO_POPUP_TOP_FILTER: Optional["_ComboPopupTopAlignFilter"] = None


class _ComboPopupTopAlignFilter(QObject):
    """Ensure QComboBox popups open scrolled from the first row (top-aligned).

    Qt's default behavior often centers the current item in the popup, which
    makes the visible list appear to "start in the middle". This filter hooks
    combo popup views globally and snaps their scroll position to top on show.
    """

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # type: ignore[override]
        et = event.type()
        if isinstance(obj, QComboBox) and et in (QEvent.Type.Polish, QEvent.Type.Show):
            self._attach_combo_popup_hooks(obj)
            return False
        if et == QEvent.Type.Show:
            if isinstance(obj, QAbstractItemView):
                self._schedule_top_align(obj)
            elif isinstance(obj, QWidget):
                if bool(obj.property("_cvopsComboPopupWindow")):
                    for view in obj.findChildren(QAbstractItemView):
                        self._schedule_top_align(view)
        return False

    def _attach_combo_popup_hooks(self, combo: QComboBox) -> None:
        view = combo.view()
        if view is None:
            return
        if not bool(view.property("_cvopsComboPopupHooked")):
            view.setProperty("_cvopsComboPopupHooked", True)
            view.installEventFilter(self)
        popup = view.window()
        if isinstance(popup, QWidget) and not bool(popup.property("_cvopsComboPopupHooked")):
            popup.setProperty("_cvopsComboPopupHooked", True)
            popup.setProperty("_cvopsComboPopupWindow", True)
            popup.installEventFilter(self)

    def _schedule_top_align(self, view: QAbstractItemView) -> None:
        def _align() -> None:
            try:
                view.scrollToTop()
            except Exception:
                pass
            try:
                sb = view.verticalScrollBar()
                if sb is not None:
                    sb.setValue(sb.minimum())
            except Exception:
                pass

        # Do one immediate-post-show and one deferred snap so we win against
        # Qt's late "center current item" adjustments.
        QTimer.singleShot(0, _align)
        QTimer.singleShot(12, _align)


def install_cvops_chamfer_combo_style(window: QWidget) -> None:  # noqa: ARG001
    """Install chamfered combo rendering on the application event loop (idempotent)."""
    global _CHAMFER_COMBO_FILTER
    if _CHAMFER_COMBO_FILTER is not None:
        return
    app = QApplication.instance()
    if app is None:
        return
    _CHAMFER_COMBO_FILTER = _ChamferComboPaintFilter(app)
    app.installEventFilter(_CHAMFER_COMBO_FILTER)


def install_combo_popup_top_align(app: QApplication) -> None:
    """Install global combo-popup top alignment (idempotent)."""
    global _COMBO_POPUP_TOP_FILTER
    if _COMBO_POPUP_TOP_FILTER is not None:
        return
    _COMBO_POPUP_TOP_FILTER = _ComboPopupTopAlignFilter(app)
    app.installEventFilter(_COMBO_POPUP_TOP_FILTER)


def install_selectable_labels(app: QApplication) -> None:
    """Install the QLabel selectability filter on *app* (idempotent)."""
    global _SELECTABLE_FILTER
    if _SELECTABLE_FILTER is not None:
        return
    _SELECTABLE_FILTER = _SelectableLabelsFilter(app)
    app.installEventFilter(_SELECTABLE_FILTER)


def install_title_fit_filter(app: QApplication) -> None:
    """Install content-width sizing for title widgets (idempotent)."""
    global _TITLE_FIT_FILTER
    if _TITLE_FIT_FILTER is not None:
        return
    _TITLE_FIT_FILTER = _TitleFitFilter(app)
    app.installEventFilter(_TITLE_FIT_FILTER)


def repolish(widget: Optional[QWidget]) -> None:
    """Force QSS to re-evaluate after dynamic property changes."""
    if widget is None:
        return
    try:
        style = widget.style()
        style.unpolish(widget)
        style.polish(widget)
        widget.update()
    except Exception:
        # QSS must never break UI.
        return


def _set_stylesheet_without_normalizing(widget: QWidget, style: str) -> None:
    if _ORIGINAL_WIDGET_SET_STYLESHEET is not None:
        _ORIGINAL_WIDGET_SET_STYLESHEET(widget, style)
    else:
        widget.setStyleSheet(style)


def install_cvops_local_stylesheet_normalizer() -> None:
    """Normalize future widget-local QSS assignments through the active Cv Ops palette."""
    global _ORIGINAL_WIDGET_SET_STYLESHEET, _STYLESHEET_NORMALIZER_ACTIVE
    if _STYLESHEET_NORMALIZER_ACTIVE:
        return
    _ORIGINAL_WIDGET_SET_STYLESHEET = QWidget.setStyleSheet

    def _normalized_set_stylesheet(widget: QWidget, style: str) -> None:
        raw = str(style or "")
        if bool(widget.property("_cvopsUsesStyleFactory")):
            _ORIGINAL_WIDGET_SET_STYLESHEET(widget, raw)  # type: ignore[misc]
            return
        if bool(widget.property("_cvopsSkipStyleNormalize")):
            _ORIGINAL_WIDGET_SET_STYLESHEET(widget, raw)  # type: ignore[misc]
            return
        if raw:
            widget.setProperty("_cvopsRawLocalStyleSheet", raw)
        _ORIGINAL_WIDGET_SET_STYLESHEET(widget, cvops_themed_css(raw))  # type: ignore[misc]

    QWidget.setStyleSheet = _normalized_set_stylesheet  # type: ignore[method-assign]
    _STYLESHEET_NORMALIZER_ACTIVE = True


def normalize_ui_scale_pct(value: object, *, default: int = 100) -> int:
    try:
        pct = int(round(float(value)))
    except Exception:
        pct = int(default)
    return max(_UI_SCALE_MIN_PCT, min(_UI_SCALE_MAX_PCT, pct))


def _scale_metric(value: int, scale: float, *, floor: int = 0, allow_zero: bool = True) -> int:
    if value <= 0:
        return 0
    scaled = int(round(value * scale))
    if not allow_zero:
        scaled = max(1, scaled)
    if floor > 0:
        scaled = max(floor, scaled)
    return scaled


def _scale_qsize(size: QSize, scale: float) -> QSize:
    if not size.isValid():
        return QSize()
    return QSize(
        _scale_metric(size.width(), scale, allow_zero=False),
        _scale_metric(size.height(), scale, allow_zero=False),
    )


def resolve_ui_scale_factor(root: Optional[QWidget | QSize | tuple[int, int]], ui_scale_pct: object) -> float:
    scale = normalize_ui_scale_pct(ui_scale_pct) / 100.0
    width = 0.0
    height = 0.0
    if isinstance(root, QWidget):
        width = float(max(root.width(), root.minimumWidth()))
        height = float(max(root.height(), root.minimumHeight()))
    elif isinstance(root, QSize):
        width = float(root.width())
        height = float(root.height())
    elif isinstance(root, tuple) and len(root) >= 2:
        try:
            width = float(root[0])
            height = float(root[1])
        except Exception:
            width = 0.0
            height = 0.0
    if width <= 0.0 or height <= 0.0:
        return round(scale, 3)
    viewport_fit = min(1.0, width / _UI_SCALE_BASE_WIDTH, height / _UI_SCALE_BASE_HEIGHT)
    viewport_fit = max(0.76, viewport_fit)
    return round(scale * viewport_fit, 3)


def scale_qss_pixel_metrics(css: str, scale: float) -> str:
    text = str(css or "")
    if not text or abs(scale - 1.0) < 0.01:
        return text

    def _replace(match: re.Match[str]) -> str:
        sign = -1 if match.group("sign") == "-" else 1
        value = float(match.group("value"))
        if value <= 0.0:
            return "0px"
        scaled = max(1, int(round(value * scale)))
        return f"{'-' if sign < 0 else ''}{scaled}px"

    return _QSS_PX_RE.sub(_replace, text)


def _snap_metric(value: int, scale: float, *, floor: int = 0) -> int:
    if value <= 0:
        return 0
    scaled = int(round((value * scale) / 2.0) * 2)
    return max(floor, scaled)


def apply_ui_scale_compaction(
    root: Optional[QWidget],
    *,
    ui_scale_pct: object = 100,
) -> None:
    if root is None:
        return
    ui_scale = resolve_ui_scale_factor(root, ui_scale_pct)
    layout_scale = 0.50 * ui_scale
    control_scale = 0.46 * ui_scale
    progress_scale = 0.44 * ui_scale
    font_scale = 0.82 * ui_scale
    width_scale = ui_scale
    control_floor = 14
    progress_floor = 8

    layouts: list[QLayout] = []
    root_layout = root.layout()
    if root_layout is not None:
        layouts.append(root_layout)
    layouts.extend(root.findChildren(QLayout))

    seen_layouts: set[int] = set()
    for layout in layouts:
        ident = id(layout)
        if ident in seen_layouts:
            continue
        seen_layouts.add(ident)
        base_spacing = layout.property("_cvopsBaseSpacing")
        if base_spacing is None:
            layout.setProperty("_cvopsBaseSpacing", layout.spacing())
            margins = layout.contentsMargins()
            layout.setProperty("_cvopsBaseMarginL", margins.left())
            layout.setProperty("_cvopsBaseMarginT", margins.top())
            layout.setProperty("_cvopsBaseMarginR", margins.right())
            layout.setProperty("_cvopsBaseMarginB", margins.bottom())
            base_spacing = layout.spacing()
        if int(base_spacing) >= 0:
            layout.setSpacing(_snap_metric(int(base_spacing), layout_scale, floor=2))
        left = int(layout.property("_cvopsBaseMarginL") or 0)
        top = int(layout.property("_cvopsBaseMarginT") or 0)
        right = int(layout.property("_cvopsBaseMarginR") or 0)
        bottom = int(layout.property("_cvopsBaseMarginB") or 0)
        layout.setContentsMargins(
            _snap_metric(left, layout_scale, floor=2 if left else 0),
            _snap_metric(top, layout_scale, floor=2 if top else 0),
            _snap_metric(right, layout_scale, floor=2 if right else 0),
            _snap_metric(bottom, layout_scale, floor=2 if bottom else 0),
        )

    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        if bool(widget.property("densityExempt")):
            continue
        is_control = isinstance(widget, (QAbstractButton, QLineEdit, QComboBox, QAbstractSpinBox))
        is_progress = isinstance(widget, QProgressBar)

        base_style = widget.property("_cvopsBaseStyleSheet")
        current_style = widget.styleSheet()
        last_scaled_style = widget.property("_cvopsLastScaledStyleSheet")
        if base_style is None or (current_style and current_style != last_scaled_style):
            widget.setProperty("_cvopsBaseStyleSheet", current_style)
            base_style = current_style
        if isinstance(base_style, str) and "px" in base_style:
            scaled_style = scale_qss_pixel_metrics(base_style, ui_scale)
            if scaled_style != current_style:
                widget.setStyleSheet(scaled_style)
                widget.setProperty("_cvopsLastScaledStyleSheet", scaled_style)

        base_font_pt = widget.property("_cvopsBasePointSizeF")
        if base_font_pt is None:
            widget.setProperty("_cvopsBasePointSizeF", widget.font().pointSizeF())
            base_font_pt = widget.font().pointSizeF()
        try:
            base_font = float(base_font_pt or 0.0)
        except Exception:
            base_font = 0.0
        if base_font >= 11.0:
            next_font = max(8.0, round(base_font * font_scale, 1))
            font = widget.font()
            if abs(font.pointSizeF() - next_font) >= 0.09:
                font.setPointSizeF(next_font)
                widget.setFont(font)

        if _is_title_widget(widget):
            fit_title_widget_to_text(widget)
        else:
            base_min_w = widget.property("_cvopsBaseMinWidth")
            if base_min_w is None:
                widget.setProperty("_cvopsBaseMinWidth", widget.minimumWidth())
                widget.setProperty("_cvopsBaseMaxWidth", widget.maximumWidth())
                base_min_w = widget.minimumWidth()
            base_min_width = int(base_min_w or 0)
            if base_min_width > 0:
                next_min_width = _scale_metric(base_min_width, width_scale, floor=12, allow_zero=False)
                widget.setMinimumWidth(next_min_width)
                base_max_width = int(widget.property("_cvopsBaseMaxWidth") or 0)
                if 0 < base_max_width < _WIDGET_SIZE_MAX:
                    widget.setMaximumWidth(
                        max(
                            next_min_width,
                            _scale_metric(base_max_width, width_scale, floor=next_min_width, allow_zero=False),
                        )
                    )

        if hasattr(widget, "iconSize") and hasattr(widget, "setIconSize"):
            try:
                icon_size = widget.iconSize()
            except Exception:
                icon_size = QSize()
            base_icon = widget.property("_cvopsBaseIconSize")
            if base_icon is None and icon_size.isValid():
                widget.setProperty("_cvopsBaseIconSize", icon_size)
                base_icon = icon_size
            if isinstance(base_icon, QSize) and base_icon.isValid():
                try:
                    widget.setIconSize(_scale_qsize(base_icon, width_scale))
                except Exception:
                    pass

        if hasattr(widget, "gridSize") and hasattr(widget, "setGridSize"):
            try:
                grid_size = widget.gridSize()
            except Exception:
                grid_size = QSize()
            base_grid = widget.property("_cvopsBaseGridSize")
            if base_grid is None and grid_size.isValid():
                widget.setProperty("_cvopsBaseGridSize", grid_size)
                base_grid = grid_size
            if isinstance(base_grid, QSize) and base_grid.isValid():
                try:
                    widget.setGridSize(_scale_qsize(base_grid, width_scale))
                except Exception:
                    pass

        if not is_control and not is_progress:
            continue
        base_min = widget.property("_cvopsBaseMinHeight")
        if base_min is None:
            widget.setProperty("_cvopsBaseMinHeight", widget.minimumHeight())
            widget.setProperty("_cvopsBaseMaxHeight", widget.maximumHeight())
            base_min = widget.minimumHeight()
        base_min_height = int(base_min or 0)
        if base_min_height <= 0:
            continue
        scale = progress_scale if is_progress else control_scale
        floor = progress_floor if is_progress else control_floor
        next_min = _snap_metric(base_min_height, scale, floor=floor)
        widget.setMinimumHeight(next_min)
        base_max_height = int(widget.property("_cvopsBaseMaxHeight") or 0)
        if 0 < base_max_height <= 72:
            widget.setMaximumHeight(
                max(next_min, _snap_metric(base_max_height, scale, floor=next_min))
            )


# ---------------------------------------------------------------------------
# Generated theme assets (noise tile + status dots)
# ---------------------------------------------------------------------------

# Neo-Swiss substrate (UI-001) — marathon CV Ops layout + shared graph tokens.
# Aligned to UI_Guide/DarkModeExample.jsx (BG / BORDER / TEXT / ACCENT / semantic ramps).
NS_PURE_VOID = "#0e0e0e"
NS_VOID = "#121212"
NS_GRAPHITE_1 = "#151515"
NS_GRAPHITE_2 = "#1a1a1a"
NS_GRAPHITE_3 = "#222222"
NS_GRAPHITE_4 = "#2a2a2a"
NS_GRAPHITE_5 = "#333333"
NS_MIST_1 = "#444444"
NS_MIST_2 = "#666666"
NS_MIST_3 = "#cccccc"
NS_BONE = "#aaaaaa"
NS_PAPER = "#cccccc"
NS_VERMILLION = "#aa5555"
NS_AMBER = "#c57a2e"
NS_CYAN = "#5aaaaa"
MARATHON_RADIATION = "#5a9a6a"

# Aurora Substrate (UI-002) — graphite/green monochrome colorway.
AURORA_PURE_VOID = "#050807"
AURORA_VOID = "#0A0E0C"
AURORA_GRAPHITE_1 = "#121815"
AURORA_GRAPHITE_2 = "#18201C"
AURORA_GRAPHITE_3 = "#222A26"
AURORA_GRAPHITE_4 = "#2D3833"
AURORA_GRAPHITE_5 = "#3F4C46"
AURORA_MIST_1 = "#5A6862"
AURORA_MIST_2 = "#7E8C84"
AURORA_MIST_3 = "#A8B4AC"
AURORA_BONE = "#D6DDD8"
AURORA_PAPER = "#E8EDE9"
# UI_Guide Aurora colorway — `--ns-radiation` (alive / success / completed).
AURORA_RADIATION = "#7AE860"

# Dark mode variant — Aurora semantics with a near-black canvas and Solarized
# deep-teal structure chrome (#002B36) for borders and edge rails.
DARK_MODE_PURE_VOID = "#040607"
DARK_MODE_VOID = "#080A0C"
DARK_MODE_GRAPHITE_1 = "#121418"
DARK_MODE_GRAPHITE_2 = "#181B1F"
DARK_MODE_GRAPHITE_3 = "#21252B"
DARK_MODE_GRAPHITE_4 = "#2F343C"
DARK_MODE_EDGE_SOFT = "#002B36"
DARK_MODE_EDGE_STRONG = "#002B36"

# ---------------------------------------------------------------------------
# Workbench semantic color tokens — used by custom-drawn Qt widgets and the
# Cytoscape.js ontology graph. These are FIXED values (not scheme-dependent).
#
# Rule: saturated accent colors have EXCLUSIVE semantic meaning.
# Nothing else in the UI may use these values.
# ---------------------------------------------------------------------------

# Base palette — surfaces and structural chrome (Neo-Swiss substrate / UI-001)
WB_BG_VOID      = NS_PURE_VOID
WB_BG_GRAPHITE  = NS_GRAPHITE_1
WB_BG_PANEL     = NS_GRAPHITE_2
WB_LINE_LIGHT   = NS_GRAPHITE_3
WB_LINE_MED     = NS_GRAPHITE_4

# Typography — bilingual split: mono for machine, sans for human
WB_TEXT_IRON    = NS_MIST_1
WB_TEXT_SIGNAL  = NS_MIST_3
WB_TEXT_BRIGHT  = NS_PAPER
WB_FONT_MONO    = "JetBrains Mono, IBM Plex Mono, Courier New, monospace"
WB_FONT_SANS    = "Inter, -apple-system, Segoe UI, Roboto, sans-serif"

# Exclusive accent channels — semantic (marathon primary / alive uses lime)
WB_ACCENT_ALERT  = NS_VERMILLION
WB_ACCENT_WARN   = NS_AMBER
WB_ACCENT_ACTIVE = NS_CYAN
WB_ACCENT_SELECT = MARATHON_RADIATION
CVOPS_SELECTION_ACTIVE = "#6c71c4"
CVOPS_SELECTION_EDGE = "#6c71c4"
CVOPS_SELECTION_TEXT = "#FFFFFF"
CVOPS_TITLE_CARD_BG = "#DC322F"
CVOPS_TITLE_CARD_RED = CVOPS_TITLE_CARD_BG
CVOPS_TITLE_CARD_TEXT = "#FFFFFF"

# Entity type palette for the Ontology Surface graph nodes
WB_NODE_SCENARIO   = "#c57a2e"   # accent (scenario)
WB_NODE_BACKBONE   = "#5588bb"   # blue
WB_NODE_CELL       = "#5aaaaa"   # cyan
WB_NODE_DATASET    = "#5a9a6a"   # green
WB_NODE_MODEL      = WB_ACCENT_ACTIVE   # cyan
WB_NODE_JOB        = "#5588bb"   # blue
WB_NODE_SNAPSHOT   = "#666666"   # dim grey
WB_NODE_LINEAGE    = "#8a5520"   # accent dim
WB_NODE_RANGE      = WB_ACCENT_WARN    # amber
WB_NODE_CATALOG    = "#aa5555"   # red
WB_NODE_DATABASE   = "#c57a2e"   # accent — DB origin / central figure

# Edge type palette for the Ontology Surface graph edges
WB_EDGE_BELONGS_TO   = "#2a2a2a"   # structural / ownership (DarkModeExample BORDER)
WB_EDGE_GOVERNED_BY  = WB_ACCENT_ACTIVE   # data provenance
WB_EDGE_PRODUCES     = WB_NODE_JOB        # job → model output
WB_EDGE_EVALUATES    = WB_ACCENT_WARN     # range evaluation
WB_EDGE_DERIVED_FROM = "#666666"           # snapshot parent chain (TEXT_DIM)
WB_EDGE_HAS_HEAD     = WB_NODE_LINEAGE    # lineage → snapshot

_CVOPS_DEFAULT_ROLE_COLORS: dict[str, str] = {
    "bg_void": WB_BG_VOID,
    "bg_graphite": WB_BG_GRAPHITE,
    "bg_panel": WB_BG_PANEL,
    "line_light": WB_LINE_LIGHT,
    "line_med": WB_LINE_MED,
    "text_iron": WB_TEXT_IRON,
    "text_signal": WB_TEXT_SIGNAL,
    "text_bright": WB_TEXT_BRIGHT,
    "accent_alert": WB_ACCENT_ALERT,
    "accent_warn": WB_ACCENT_WARN,
    "accent_active": WB_ACCENT_ACTIVE,
    "accent_select": WB_ACCENT_SELECT,
    "selection_active": CVOPS_SELECTION_ACTIVE,
    "selection_edge": CVOPS_SELECTION_EDGE,
    "selection_text": CVOPS_SELECTION_TEXT,
    "title_card_bg": CVOPS_TITLE_CARD_BG,
    "title_card_text": CVOPS_TITLE_CARD_TEXT,
}

_CVOPS_AURORA_ROLE_COLORS: dict[str, str] = {
    "bg_void": AURORA_PURE_VOID,
    "bg_graphite": AURORA_GRAPHITE_1,
    "bg_panel": AURORA_GRAPHITE_2,
    "line_light": AURORA_GRAPHITE_3,
    "line_med": AURORA_GRAPHITE_4,
    "text_iron": AURORA_MIST_1,
    "text_signal": AURORA_MIST_3,
    "text_bright": AURORA_PAPER,
    # Fail / cancel must read as true alert red (mono substrate keeps warn/drift for softer signals).
    "accent_alert": NS_VERMILLION,
    "accent_warn": AURORA_BONE,
    "accent_active": AURORA_MIST_3,
    "accent_select": AURORA_RADIATION,
    "selection_active": CVOPS_SELECTION_ACTIVE,
    "selection_edge": CVOPS_SELECTION_EDGE,
    "selection_text": CVOPS_SELECTION_TEXT,
    "title_card_bg": CVOPS_TITLE_CARD_BG,
    "title_card_text": CVOPS_TITLE_CARD_TEXT,
}

_CVOPS_DARK_MODE_ROLE_COLORS: dict[str, str] = {
    **_CVOPS_AURORA_ROLE_COLORS,
    "bg_void": DARK_MODE_PURE_VOID,
    "bg_graphite": DARK_MODE_GRAPHITE_1,
    "bg_panel": DARK_MODE_GRAPHITE_2,
    "line_light": DARK_MODE_EDGE_SOFT,
    "line_med": DARK_MODE_EDGE_STRONG,
}


def cvops_color(role: str) -> str:
    """Return dynamic Cv Ops chrome colors for custom-painted widgets."""
    raw = str(role or "").strip()
    if not raw:
        return theme_hex("accent_dark")
    if raw.startswith("#") or raw.lower().startswith("rgb"):
        return raw
    key = raw.lower()
    override = _CVOPS_RUNTIME_ROLE_OVERRIDES.get(key)
    if override:
        return override
    scheme = current_color_scheme()
    if scheme == "beacon":
        beacon_map = {
            "bg_void": "#A4ACB8",
            "bg_graphite": "#8E96A2",
            "bg_panel": "#7C8593",
            "line_light": "#6B7480",
            "line_med": "#5A6370",
            "text_iron": "#FFD235",
            "text_signal": "#222933",
            "text_bright": "#ECEEF2",
            "accent_alert": "#F70D1A",
            "accent_warn": "#FFD235",
            "accent_active": "#0A8FA8",
            "accent_select": "#2D9A40",
            "selection_active": CVOPS_SELECTION_ACTIVE,
            "selection_edge": CVOPS_SELECTION_EDGE,
            "selection_text": CVOPS_SELECTION_TEXT,
            "title_card_bg": CVOPS_TITLE_CARD_BG,
            "title_card_text": CVOPS_TITLE_CARD_TEXT,
        }
        return beacon_map.get(key, theme_hex("accent_dark"))
    if scheme == "dark mode":
        return _CVOPS_DARK_MODE_ROLE_COLORS.get(key, AURORA_MIST_3)
    if scheme == "aurora":
        return _CVOPS_AURORA_ROLE_COLORS.get(key, AURORA_MIST_3)
    if scheme == "fire":
        fire_map = {
            "bg_void": "#060709",
            "bg_graphite": "#14161a",
            "bg_panel": "#1c1f24",
            "line_light": "#2a2e34",
            "line_med": "#3f464f",
            "text_iron": "#a89f91",
            "text_signal": "#f2e7cf",
            "text_bright": "#fff4db",
            "accent_alert": "#ff8c5c",
            "accent_warn": "#ffc884",
            "accent_active": "#f8a440",
            "accent_select": "#ffc884",
            "selection_active": CVOPS_SELECTION_ACTIVE,
            "selection_edge": CVOPS_SELECTION_EDGE,
            "selection_text": CVOPS_SELECTION_TEXT,
            "title_card_bg": CVOPS_TITLE_CARD_BG,
            "title_card_text": CVOPS_TITLE_CARD_TEXT,
        }
        return fire_map.get(key, theme_hex("accent_dark"))
    return _CVOPS_DEFAULT_ROLE_COLORS.get(key, theme_hex("accent_dark"))


def cvops_qcolor(role: str, alpha: int = 255):
    from PyQt6.QtGui import QColor  # noqa: PLC0415

    color = QColor(cvops_color(role))
    color.setAlpha(max(0, min(255, int(alpha))))
    return color


def cvops_rgba(role: str, alpha: float) -> str:
    color = cvops_qcolor(role)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {_css_alpha(a)})"


def _normalize_rgb_components(parts: list[str]) -> str:
    if len(parts) < 3:
        return ""
    values: list[int] = []
    for part in parts[:3]:
        raw = str(part or "").strip()
        if not raw:
            return ""
        try:
            if raw.endswith("%"):
                parsed = float(raw[:-1])
                if parsed < 0 or parsed > 100:
                    return ""
                value = int(round(parsed * 2.55))
            else:
                parsed = float(raw)
                if parsed < 0 or parsed > 255:
                    return ""
                value = int(round(parsed))
        except Exception:
            return ""
        values.append(max(0, min(255, value)))
    return "#{:02X}{:02X}{:02X}".format(*values)


def normalize_color_override(value: object) -> str:
    """Normalize typed theme colors to #RRGGBB.

    Accepted forms: #RGB, #RRGGBB, RGB/RRGGBB without #, rgb(r,g,b),
    rgba(r,g,b,a), and plain comma/space-separated RGB triples.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""

    hex_raw = raw[1:] if raw.startswith("#") else raw
    if re.fullmatch(r"[0-9A-Fa-f]{6}", hex_raw):
        return f"#{hex_raw.upper()}"
    if re.fullmatch(r"[0-9A-Fa-f]{3}", hex_raw):
        r, g, b = hex_raw[0], hex_raw[1], hex_raw[2]
        return f"#{r}{r}{g}{g}{b}{b}".upper()

    rgb_match = re.fullmatch(r"rgba?\((.+)\)", raw, re.IGNORECASE)
    if rgb_match:
        body = rgb_match.group(1).split("/", 1)[0].strip()
        parts = [p.strip() for p in body.split(",")] if "," in body else body.split()
        return _normalize_rgb_components(parts)

    if "," in raw:
        return _normalize_rgb_components([p.strip() for p in raw.split(",")])
    parts = raw.split()
    if len(parts) == 3:
        return _normalize_rgb_components(parts)

    color = QColor(raw)
    if color.isValid():
        return color.name(QColor.NameFormat.HexRgb).upper()
    return ""


def _qss_color_override(value: object) -> str:
    return normalize_color_override(value)


def _set_cvops_runtime_role_override(role: str, color_value: object) -> None:
    key = str(role or "").strip().lower()
    if not key:
        return
    normalized = normalize_color_override(color_value)
    if normalized:
        _CVOPS_RUNTIME_ROLE_OVERRIDES[key] = normalized
    else:
        _CVOPS_RUNTIME_ROLE_OVERRIDES.pop(key, None)


def _rgba_from_override(color_value: str, alpha: float) -> str:
    color = QColor(normalize_color_override(color_value))
    if not color.isValid():
        return ""
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({color.red()}, {color.green()}, {color.blue()}, {_css_alpha(a)})"


def _flatten_surface(value: str, *, fallback: str) -> str:
    """Collapse a surface background to a flat, fully opaque color.

    Used when no wallpaper is painted behind the chrome: gradients and
    translucent rgba fills (which otherwise composite against the native window
    and read as a glossy macOS sheen) are replaced with a solid color. Gradients
    and image-backed fills fall back to ``fallback``; translucent ``rgba(...)``
    fills are promoted to opaque ``rgb(...)``; flat hex/rgb values pass through.
    """
    s = str(value).strip()
    if "gradient" in s or "url(" in s:
        return _flatten_surface(fallback, fallback=fallback) if fallback != value else fallback
    m = re.match(r"rgba\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*[\d.]+\s*\)", s)
    if m:
        r, g, b = (int(round(float(m.group(i)))) for i in (1, 2, 3))
        return f"rgb({r}, {g}, {b})"
    return s


def cvops_mapped_qcolor(value: object, alpha: Optional[int] = None, *, fallback_role: str = "text_signal"):
    """Map legacy hard-coded UI colors to the active palette for non-QSS painters."""
    from PyQt6.QtGui import QColor  # noqa: PLC0415

    fallback_alpha = 255 if alpha is None else max(0, min(255, int(alpha)))
    if isinstance(value, QColor):
        source = QColor(value)
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return cvops_qcolor(fallback_role, fallback_alpha)
        if not text.startswith("#"):
            return cvops_qcolor(text, fallback_alpha)
        source = QColor(text)
    elif isinstance(value, (tuple, list)) and len(value) >= 3:
        source = QColor(int(value[0]), int(value[1]), int(value[2]))
        if len(value) >= 4 and alpha is None:
            source.setAlpha(max(0, min(255, int(value[3]))))
    else:
        return cvops_qcolor(fallback_role, fallback_alpha)

    source_alpha = fallback_alpha if alpha is not None else max(0, min(255, source.alpha()))
    hex_key = source.name().lower()
    role = _CVOPS_HEX_THEME_MAP.get(hex_key)
    if role is None:
        role = _CVOPS_RGBA_THEME_MAP.get((source.red(), source.green(), source.blue()))

    if role is None:
        mapped = QColor(source)
        mapped.setAlpha(source_alpha)
        return mapped
    if role == "text":
        return text_qcolor(source_alpha / 255.0)
    if role == "text-muted":
        if current_color_scheme() == "beacon":
            mapped = QColor(theme_hex("privacy_warn"))
            mapped.setAlpha(source_alpha)
            return mapped
        return text_qcolor((source_alpha / 255.0) * 0.72)

    mapped = QColor(theme_hex(role))
    mapped.setAlpha(source_alpha)
    return mapped


def _css_alpha(value: float) -> str:
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def set_cvops_stylesheet(widget: Optional[QWidget], factory: Callable[[], str]) -> None:
    """Register and apply a palette-aware local stylesheet factory."""
    if widget is None:
        return
    _STYLE_FACTORIES[widget] = factory
    try:
        widget.setProperty("_cvopsUsesStyleFactory", True)
        _set_stylesheet_without_normalizing(widget, str(factory() or ""))
    except Exception:
        return


def apply_registered_cvops_stylesheet(widget: QWidget) -> bool:
    factory = _STYLE_FACTORIES.get(widget)
    if factory is None:
        return False
    try:
        _set_stylesheet_without_normalizing(widget, str(factory() or ""))
        return True
    except Exception:
        return False


_CVOPS_HEX_THEME_MAP: dict[str, str] = {
    NS_PURE_VOID: "input_fill",
    NS_VOID: "input_fill",
    NS_GRAPHITE_1: "panel",
    NS_GRAPHITE_2: "panel",
    NS_GRAPHITE_3: "hover",
    NS_GRAPHITE_4: "hover",
    NS_GRAPHITE_5: "pressed",
    NS_MIST_1: "text-muted",
    NS_MIST_2: "text-muted",
    NS_MIST_3: "text",
    NS_BONE: "text",
    NS_PAPER: "text",
    MARATHON_RADIATION: "strip_soft",
    NS_CYAN: "accent_dark",
    "#2fa4b5": "accent_dark",
    "#4dd0e1": "accent_dark",
    "#80deea": "strip_soft",
    "#ff5252": "privacy_warn",
    "#ff6b6b": "privacy_warn",
    "#ffb74d": "strip_soft",
    "#268bd2": "accent_dark",
    "#b58900": "strip_soft",
    "#cb4b16": "privacy_warn",
    "#d33682": "privacy_warn",
    "#6c71c4": "cvops:selection_active",
    "#dc322f": "cvops:title_card_bg",
    AURORA_PURE_VOID: "cvops:bg_void",
    AURORA_VOID: "cvops:bg_void",
    AURORA_GRAPHITE_1: "cvops:bg_graphite",
    AURORA_GRAPHITE_2: "cvops:bg_panel",
    AURORA_GRAPHITE_3: "cvops:line_light",
    AURORA_GRAPHITE_4: "cvops:line_med",
    AURORA_GRAPHITE_5: "cvops:line_med",
    AURORA_MIST_1: "text-muted",
    AURORA_MIST_2: "text-muted",
    AURORA_MIST_3: "text-muted",
    AURORA_BONE: "text",
    AURORA_PAPER: "text",
    "#586e75": "text-muted",
    "#fdf6e3": "text",
    "#b8b8b8": "text-muted",
    "#e8e8e8": "text",
}

_CVOPS_RGBA_THEME_MAP: dict[tuple[int, int, int], str] = {
    (27, 27, 27): "panel",
    (0, 43, 54): "input_fill",
    (197, 255, 70): "strip_soft",
    (43, 196, 217): "accent_dark",
    (47, 164, 181): "accent_dark",
    (77, 208, 225): "accent_dark",
    (128, 222, 234): "strip_soft",
    (133, 153, 0): "strip_soft",
    (38, 139, 210): "accent_dark",
    (181, 137, 0): "strip_soft",
    (203, 75, 22): "privacy_warn",
    (211, 54, 130): "privacy_warn",
    (42, 161, 152): "accent_dark",
    (108, 113, 196): "cvops:selection_active",
    (220, 50, 47): "cvops:title_card_bg",
    (5, 8, 7): "cvops:bg_void",
    (10, 14, 12): "cvops:bg_void",
    (18, 24, 21): "cvops:bg_graphite",
    (24, 32, 28): "cvops:bg_panel",
    (34, 42, 38): "cvops:line_light",
    (45, 56, 51): "cvops:line_med",
    (63, 76, 70): "cvops:line_med",
    (90, 104, 98): "text-muted",
    (126, 140, 132): "text-muted",
    (168, 180, 172): "text-muted",
    (214, 221, 216): "text",
    (232, 237, 233): "text",
    (147, 161, 161): "text-muted",
    (124, 133, 147): "text-muted",
    (68, 68, 68): "text-muted",
    (102, 102, 102): "text-muted",
    (204, 204, 204): "text",
    (160, 160, 160): "text-muted",
    (170, 170, 170): "text-muted",
    (180, 180, 180): "text-muted",
    (184, 184, 184): "text-muted",
    (200, 200, 200): "text",
    (255, 251, 242): "text",
    (255, 255, 255): "text",
    (60, 60, 60): "text-muted",
    (120, 120, 120): "text-muted",
    (239, 68, 68): "privacy_warn",
    (203, 130, 28): "strip_soft",
    (255, 82, 82): "privacy_warn",
    (7, 54, 66): "panel",
}

_RGBA_COLOR_RE = re.compile(
    r"rgba\(\s*(?P<r>\d{1,3})\s*,\s*(?P<g>\d{1,3})\s*,\s*(?P<b>\d{1,3})\s*,\s*(?P<a>[0-9.]+)\s*\)",
    re.IGNORECASE,
)
_RGB_COLOR_RE = re.compile(
    r"rgb\(\s*(?P<r>\d{1,3})\s*,\s*(?P<g>\d{1,3})\s*,\s*(?P<b>\d{1,3})\s*\)",
    re.IGNORECASE,
)

_CVOPS_FIXED_RGBA_ROLE_MAP: dict[tuple[int, int, int], str] = {
    (108, 113, 196): "cvops:selection_active",
    (220, 50, 47): "cvops:title_card_bg",
}
_CVOPS_FIXED_HEX_ROLE_MAP: dict[str, str] = {
    "#6c71c4": "cvops:selection_active",
    "#dc322f": "cvops:title_card_bg",
}


def _role_css(role: str, alpha: float = 1.0) -> str:
    if role.startswith("cvops:"):
        return cvops_rgba(role.split(":", 1)[1], alpha)
    if role == "text":
        return text_css(alpha)
    if role == "text-muted":
        if current_color_scheme() == "beacon":
            return theme_rgba("privacy_warn", alpha)
        return text_css(alpha * 0.72)
    return theme_rgba(role, alpha)


def _protect_cvops_fixed_colors(value: str) -> tuple[str, dict[str, str]]:
    """Shield fixed CV Ops semantic colors from generic theme rewrites.

    The app-level ``themed_css`` treats Solarized red as ``privacy_warn`` and
    violet as ``accent_dark``. In CV Ops those same literals are explicit
    semantic tokens: title-card red and selected-item violet. Protect them before
    applying generic rewrites, then restore them via cvops roles.
    """
    replacements: dict[str, str] = {}
    counter = 0

    def _next_token(css_value: str) -> str:
        nonlocal counter
        token = f"__CVOPS_FIXED_COLOR_{counter}__"
        counter += 1
        replacements[token] = css_value
        return token

    def rgba_replace(match: re.Match[str]) -> str:
        try:
            rgb = (
                int(match.group("r")),
                int(match.group("g")),
                int(match.group("b")),
            )
            alpha = float(match.group("a"))
        except Exception:
            return match.group(0)
        role = _CVOPS_FIXED_RGBA_ROLE_MAP.get(rgb)
        if role is None:
            return match.group(0)
        return _next_token(_role_css(role, alpha))

    def rgb_replace(match: re.Match[str]) -> str:
        try:
            rgb = (
                int(match.group("r")),
                int(match.group("g")),
                int(match.group("b")),
            )
        except Exception:
            return match.group(0)
        role = _CVOPS_FIXED_RGBA_ROLE_MAP.get(rgb)
        if role is None:
            return match.group(0)
        return _next_token(_role_css(role, 1.0))

    text = _RGBA_COLOR_RE.sub(rgba_replace, str(value or ""))
    text = _RGB_COLOR_RE.sub(rgb_replace, text)
    for raw_hex, role in _CVOPS_FIXED_HEX_ROLE_MAP.items():
        text = re.sub(
            re.escape(raw_hex),
            lambda _match, role=role: _next_token(_role_css(role, 1.0)),
            text,
            flags=re.IGNORECASE,
        )
    return text, replacements


def cvops_themed_css(value: str) -> str:
    """Rewrite legacy/local Cv Ops QSS colors through the active theme palette."""
    protected, fixed_replacements = _protect_cvops_fixed_colors(str(value or ""))
    text = themed_css(protected)
    if not text:
        return text

    def rgba_replace(match: re.Match[str]) -> str:
        try:
            rgb = (
                int(match.group("r")),
                int(match.group("g")),
                int(match.group("b")),
            )
            alpha = float(match.group("a"))
        except Exception:
            return match.group(0)
        role = _CVOPS_RGBA_THEME_MAP.get(rgb)
        if role is None:
            return match.group(0)
        return _role_css(role, alpha)

    text = _RGBA_COLOR_RE.sub(rgba_replace, text)
    for raw_hex, role in _CVOPS_HEX_THEME_MAP.items():
        text = re.sub(re.escape(raw_hex), _role_css(role, 1.0), text, flags=re.IGNORECASE)
    for token, replacement in fixed_replacements.items():
        text = text.replace(token, replacement)
    return text


def _apply_local_theme_stylesheet(widget: QWidget) -> None:
    current = widget.styleSheet()
    if not current:
        return
    raw_local = widget.property("_cvopsRawLocalStyleSheet")
    if isinstance(raw_local, str):
        themed = cvops_themed_css(raw_local)
        if themed != current:
            _set_stylesheet_without_normalizing(widget, themed)
            widget.setProperty("_cvopsLastThemedStyleSheet", themed)
        return
    base = widget.property("_cvopsThemeBaseStyleSheet")
    last_themed = widget.property("_cvopsLastThemedStyleSheet")
    last_scaled = widget.property("_cvopsLastScaledStyleSheet")
    if not isinstance(base, str) or (
        current != last_themed and current != last_scaled
    ):
        base = current
        widget.setProperty("_cvopsThemeBaseStyleSheet", base)
    themed = cvops_themed_css(base)
    if themed != current:
        _set_stylesheet_without_normalizing(widget, themed)
        widget.setProperty("_cvopsLastThemedStyleSheet", themed)


def refresh_cvops_theme_tree(root: Optional[QWidget]) -> None:
    """Refresh child-owned QSS and custom paint surfaces after a color scheme change."""
    if root is None:
        return
    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        for method_name in ("refresh_theme_styles", "refresh_theme"):
            method = getattr(widget, method_name, None)
            if not callable(method):
                continue
            try:
                method()
            except Exception:
                pass
    for widget in widgets:
        if apply_registered_cvops_stylesheet(widget):
            continue
        _apply_local_theme_stylesheet(widget)
    for widget in widgets:
        repolish(widget)
        try:
            widget.update()
        except Exception:
            pass

_ASSETS_CACHE: Optional[dict[str, str]] = None

# Bump the suffix to force regeneration when tuning.
_NOISE_KEY = "noise_rb_v1"
_DOT_KEY = "v2"

_DOT_SPECS: dict[str, tuple[int, int, int, float]] = {
    "accent":  (197, 122, 46, 0.95),
    "ok":      (90, 154, 106, 0.95),
    "warn":    (170, 85, 85, 0.95),
    "drift":   (197, 122, 46, 0.95),
    "standby": (102, 102, 102, 0.82),
    "muted":   (68, 68, 68, 0.38),
}


def _theme_cache_dir() -> Path:
    base = Path.home() / ".cache" / "insight" / "cvops_theme"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _write_dot_png(path: Path, r: int, g: int, b: int, a: float) -> None:
    from PyQt6.QtGui import QColor, QImage, QPainter

    img = QImage(8, 8, QImage.Format.Format_ARGB32)
    img.fill(0)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    color = QColor(r, g, b)
    color.setAlphaF(a)
    painter.setBrush(color)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(1, 1, 6, 6)
    painter.end()
    img.save(str(path), "PNG")


def _write_noise_png(path: Path) -> None:
    import numpy as np
    from PyQt6.QtGui import QImage

    rng = np.random.default_rng(42)
    grey = rng.integers(210, 256, (256, 256), dtype=np.uint8)
    alpha = rng.integers(3, 10, (256, 256), dtype=np.uint8)
    arr = np.stack([grey, grey, grey, alpha], axis=-1)
    buf = bytes(arr.tobytes())
    img = QImage(buf, 256, 256, 256 * 4, QImage.Format.Format_RGBA8888)
    img.save(str(path), "PNG")


def _ensure_theme_assets() -> dict[str, str]:
    global _ASSETS_CACHE
    if _ASSETS_CACHE is not None:
        return _ASSETS_CACHE
    cache_dir = _theme_cache_dir()
    paths: dict[str, str] = {}

    noise_path = cache_dir / f"{_NOISE_KEY}.png"
    if not noise_path.exists():
        try:
            _write_noise_png(noise_path)
        except Exception:
            # Non-fatal: QSS will fall back to solid fill.
            pass
    if noise_path.exists():
        paths["noise"] = str(noise_path).replace("\\", "/")

    for name, (r, g, b, a) in _DOT_SPECS.items():
        dot_path = cache_dir / f"dot_{name}_{_DOT_KEY}.png"
        if not dot_path.exists():
            try:
                _write_dot_png(dot_path, r, g, b, a)
            except Exception:
                continue
        if dot_path.exists():
            paths[f"dot_{name}"] = str(dot_path).replace("\\", "/")

    _ASSETS_CACHE = paths
    return paths


def _text_spec() -> tuple[str, str]:
    """Return (text_primary, text_muted) matching `theme.get_global_stylesheet()` defaults."""
    if str(os.environ.get("INSIGHT_CVOPS_JAPANDI", "0")).strip().lower() not in {"0", "false", "off"}:
        return "#26302b", "rgba(38, 48, 43, 0.68)"
    red_black_default = "0" if current_color_scheme() in ("marathon", "wear_marathon", "fire", "beacon") or is_aurora_family_scheme() else "1"
    if str(os.environ.get("INSIGHT_CVOPS_RED_BLACK", red_black_default)).strip().lower() not in {"0", "false", "off"}:
        return "#f6eee8", "rgba(246, 238, 232, 0.78)"
    scheme = current_color_scheme()
    if scheme == "beacon":
        return "#222933", "rgba(255, 210, 53, 0.92)"
    # Neo-Swiss body / meta ramp for marathon schemes (matches DarkModeExample text ramp).
    if scheme in ("marathon", "wear_marathon"):
        return NS_PAPER, "rgba(102, 102, 102, 0.88)"
    if scheme == "default":
        return "#f6e9e5", "rgba(246, 233, 229, 0.76)"
    if scheme == "fire":
        return "#f4ffff", "rgba(244, 255, 255, 0.74)"
    if is_aurora_family_scheme(scheme):
        return AURORA_PAPER, "rgba(168, 180, 172, 0.78)"
    if scheme == "solarized_dark":
        return "#a7b6b6", "rgba(167, 182, 182, 0.66)"
    if scheme == "tactical":
        return "#d4dbe0", "rgba(212, 219, 224, 0.64)"
    if scheme == "material_dark":
        return "#e8eaed", "rgba(232, 234, 237, 0.64)"
    if scheme == "material":
        return "#202124", "rgba(32, 33, 36, 0.64)"
    # For non-solarized schemes, defer to inherited palette.
    return "inherit", "inherit"


def get_cvops_stylesheet(
    workspace_wallpaper: Optional[Path | str] = None,
    *,
    backdrop_blend: Optional[WorkspaceBackdropBlend] = None,
    title_text_color: str = "",
    title_background_color: str = "",
    ui_text_color: str = "",
    ui_muted_text_color: str = "",
    ui_background_color: str = "",
    ui_panel_background_color: str = "",
    ui_control_background_color: str = "",
    ui_accent_color: str = "",
) -> str:
    """Additional CV Ops QSS that composes with `ui.theme.get_global_stylesheet()`."""
    text_primary, text_muted = _text_spec()
    custom_text = _qss_color_override(ui_text_color)
    custom_muted_text = _qss_color_override(ui_muted_text_color)
    custom_root_bg = _qss_color_override(ui_background_color)
    custom_panel_bg = _qss_color_override(ui_panel_background_color)
    custom_control_bg = _qss_color_override(ui_control_background_color)
    custom_accent = _qss_color_override(ui_accent_color)

    scheme = current_color_scheme()
    japandi = str(os.environ.get("INSIGHT_CVOPS_JAPANDI", "0")).strip().lower() not in {"0", "false", "off"}
    aurora_family = is_aurora_family_scheme(scheme)
    use_dark_mode = (not japandi) and scheme == "dark mode"
    red_black_default = "0" if scheme in ("marathon", "wear_marathon", "fire", "beacon") or aurora_family else "1"
    red_black = (not japandi) and str(os.environ.get("INSIGHT_CVOPS_RED_BLACK", red_black_default)).strip().lower() not in {"0", "false", "off"}
    is_dark = True if red_black else False if japandi else scheme in ("solarized_dark", "tactical", "material_dark", "wear_marathon") or aurora_family
    is_solarized_dark = (not japandi) and scheme == "solarized_dark"
    use_neo_swiss = (not japandi) and scheme in ("marathon", "wear_marathon")
    use_aurora = (not japandi) and aurora_family
    use_beacon = (not japandi) and scheme == "beacon"
    use_substrate = use_neo_swiss or use_aurora or use_beacon

    # Accent stops — quiet fill, idle-state highlight, glass tint, active state.
    if japandi:
        accent = "rgba(85, 115, 92, 0.92)"
        accent_quiet = "rgba(85, 115, 92, 0.07)"
        idle_highlight = "rgba(85, 115, 92, 0.13)"
        accent_glass = "rgba(85, 115, 92, 0.11)"
        accent_glass_hover = "rgba(85, 115, 92, 0.18)"
        accent_active = "rgba(85, 115, 92, 0.24)"
        accent_ring = "rgba(85, 115, 92, 0.36)"
        border_idle = "rgba(38, 48, 43, 0.10)"
        border_idle_strong = "rgba(38, 48, 43, 0.18)"
        panel_bg = "rgba(255, 251, 242, 0.94)"
        panel_bg_soft = "rgba(248, 241, 228, 0.86)"
        panel_glass = "rgba(255, 251, 242, 0.70)"
        panel_orbit = "rgba(246, 240, 226, 0.94)"
        ok = "rgba(85, 115, 92, 0.92)"
        ok_quiet = "rgba(85, 115, 92, 0.08)"
        standby = "rgba(91, 102, 96, 0.78)"
        standby_quiet = "rgba(91, 102, 96, 0.08)"
        drift = "rgba(166, 118, 65, 0.92)"
        drift_quiet = "rgba(166, 118, 65, 0.11)"
        warn = "rgba(164, 82, 62, 0.92)"
        warn_quiet = "rgba(164, 82, 62, 0.10)"
    elif use_beacon:
        accent = "rgba(247, 13, 26, 0.96)"
        accent_quiet = "rgba(247, 13, 26, 0.08)"
        idle_highlight = "rgba(247, 13, 26, 0.14)"
        accent_glass = "rgba(247, 13, 26, 0.14)"
        accent_glass_hover = "rgba(247, 13, 26, 0.22)"
        accent_active = "rgba(247, 13, 26, 0.28)"
        accent_ring = "rgba(247, 13, 26, 0.42)"
        border_idle = "rgba(90, 99, 112, 0.58)"
        border_idle_strong = "rgba(90, 99, 112, 0.82)"
        panel_bg = "rgba(142, 150, 162, 0.94)"
        panel_bg_soft = "rgba(164, 172, 184, 0.82)"
        panel_glass = "rgba(232, 237, 233, 0.32)"
        panel_orbit = "rgba(164, 172, 184, 0.88)"
        ok = "rgba(45, 154, 64, 0.92)"
        ok_quiet = "rgba(45, 154, 64, 0.08)"
        standby = "rgba(255, 210, 53, 0.88)"
        standby_quiet = "rgba(255, 210, 53, 0.12)"
        drift = "rgba(255, 210, 53, 0.96)"
        drift_quiet = "rgba(255, 210, 53, 0.14)"
        warn = "rgba(247, 13, 26, 0.96)"
        warn_quiet = "rgba(247, 13, 26, 0.12)"
    elif red_black:
        accent = "rgba(197, 122, 46, 0.95)"
        accent_quiet = "rgba(197, 122, 46, 0.07)"
        idle_highlight = "rgba(197, 122, 46, 0.14)"
        accent_glass = "rgba(197, 122, 46, 0.12)"
        accent_glass_hover = "rgba(197, 122, 46, 0.20)"
        accent_active = "rgba(197, 122, 46, 0.28)"
        accent_ring = "rgba(197, 122, 46, 0.44)"
        border_idle = "rgba(42, 42, 42, 0.14)"
        border_idle_strong = "rgba(51, 51, 51, 0.24)"
        panel_bg = "rgba(26, 26, 26, 0.96)"
        panel_bg_soft = "rgba(21, 21, 21, 0.94)"
        panel_glass = "rgba(34, 34, 34, 0.82)"
        panel_orbit = (
            "qradialgradient(cx:0.52, cy:0.42, radius:1.10, fx:0.52, fy:0.42, "
            "stop:0 rgba(34, 34, 34, 0.98), stop:0.70 rgba(21, 21, 21, 0.98), stop:1 rgba(14, 14, 14, 1.0))"
        )
        ok = "rgba(90, 154, 106, 0.92)"
        ok_quiet = "rgba(90, 154, 106, 0.08)"
        standby = "rgba(102, 102, 102, 0.80)"
        standby_quiet = "rgba(102, 102, 102, 0.07)"
        drift = "rgba(197, 122, 46, 0.92)"
        drift_quiet = "rgba(197, 122, 46, 0.10)"
        warn = "rgba(170, 85, 85, 0.95)"
        warn_quiet = "rgba(170, 85, 85, 0.10)"
    elif use_aurora:
        accent = "rgba(168, 180, 172, 0.95)"
        accent_quiet = "rgba(168, 180, 172, 0.07)"
        idle_highlight = "rgba(168, 180, 172, 0.12)"
        accent_glass = "rgba(168, 180, 172, 0.12)"
        accent_glass_hover = "rgba(168, 180, 172, 0.18)"
        accent_active = "rgba(168, 180, 172, 0.24)"
        accent_ring = "rgba(168, 180, 172, 0.36)"
        if use_dark_mode:
            border_idle = "rgba(0, 43, 54, 0.82)"
            border_idle_strong = "rgba(0, 43, 54, 0.95)"
            panel_bg = DARK_MODE_GRAPHITE_1
            panel_bg_soft = DARK_MODE_GRAPHITE_2
            panel_glass = DARK_MODE_VOID
            panel_orbit = DARK_MODE_GRAPHITE_1
        else:
            border_idle = "rgba(90, 104, 98, 0.62)"
            border_idle_strong = "rgba(126, 140, 132, 0.82)"
            panel_bg = AURORA_GRAPHITE_1
            panel_bg_soft = AURORA_GRAPHITE_2
            panel_glass = AURORA_VOID
            panel_orbit = AURORA_GRAPHITE_1
        # Success / alive — Aurora `--ns-radiation` (UI_Guide/Neo_swiss_ui_guide_html_template.html).
        ok = "rgba(122, 232, 96, 0.92)"
        ok_quiet = "rgba(122, 232, 96, 0.10)"
        standby = "rgba(126, 140, 132, 0.88)"
        standby_quiet = "rgba(126, 140, 132, 0.10)"
        drift = "rgba(168, 180, 172, 0.92)"
        drift_quiet = "rgba(168, 180, 172, 0.10)"
        warn = "rgba(232, 237, 233, 0.94)"
        warn_quiet = "rgba(232, 237, 233, 0.10)"
    elif scheme == "wear_marathon":
        accent = "rgba(197, 122, 46, 0.94)"
        accent_quiet = "rgba(197, 122, 46, 0.06)"
        idle_highlight = "rgba(197, 122, 46, 0.12)"
        accent_glass = "rgba(197, 122, 46, 0.10)"
        accent_glass_hover = "rgba(197, 122, 46, 0.18)"
        accent_active = "rgba(197, 122, 46, 0.26)"
        accent_ring = "rgba(197, 122, 46, 0.38)"
        border_idle = "rgba(42, 42, 42, 0.82)"
        border_idle_strong = "rgba(51, 51, 51, 0.95)"
        panel_bg = NS_GRAPHITE_1
        panel_bg_soft = NS_GRAPHITE_2
        panel_glass = NS_VOID
        panel_orbit = NS_GRAPHITE_1
        ok = "rgba(90, 154, 106, 0.90)"
        ok_quiet = "rgba(90, 154, 106, 0.08)"
        standby = "rgba(102, 102, 102, 0.88)"
        standby_quiet = "rgba(102, 102, 102, 0.10)"
        drift = "rgba(197, 122, 46, 0.94)"
        drift_quiet = "rgba(197, 122, 46, 0.12)"
        warn = "rgba(170, 85, 85, 0.94)"
        warn_quiet = "rgba(170, 85, 85, 0.12)"
    elif scheme == "marathon":
        accent = "rgba(197, 122, 46, 0.95)"
        accent_quiet = "rgba(197, 122, 46, 0.07)"
        idle_highlight = "rgba(197, 122, 46, 0.14)"
        accent_glass = "rgba(197, 122, 46, 0.12)"
        accent_glass_hover = "rgba(197, 122, 46, 0.20)"
        accent_active = "rgba(197, 122, 46, 0.28)"
        accent_ring = "rgba(197, 122, 46, 0.42)"
        border_idle = "rgba(42, 42, 42, 0.82)"
        border_idle_strong = "rgba(51, 51, 51, 0.95)"
        panel_bg = NS_GRAPHITE_1
        panel_bg_soft = NS_GRAPHITE_2
        panel_glass = NS_VOID
        panel_orbit = NS_GRAPHITE_1
        ok = "rgba(90, 154, 106, 0.92)"
        ok_quiet = "rgba(90, 154, 106, 0.08)"
        standby = "rgba(102, 102, 102, 0.88)"
        standby_quiet = "rgba(102, 102, 102, 0.10)"
        drift = "rgba(197, 122, 46, 0.94)"
        drift_quiet = "rgba(197, 122, 46, 0.12)"
        warn = "rgba(170, 85, 85, 0.94)"
        warn_quiet = "rgba(170, 85, 85, 0.12)"
    elif scheme == "fire":
        # [FIRE THEME] Matte — all solid flat colors, no gradients, no shine
        # Dark Orange (#FF6A19) borders, Orange (#FF7F00) fills, Warm White text on Darkness bg
        accent = "rgba(255, 106, 19, 1.0)"
        accent_quiet = "rgba(255, 127, 0, 0.08)"
        idle_highlight = "rgba(255, 127, 0, 0.16)"
        accent_glass = "rgba(255, 127, 0, 0.14)"
        accent_glass_hover = "rgba(255, 127, 0, 0.22)"
        accent_active = "rgba(255, 127, 0, 0.32)"
        accent_ring = "rgba(255, 106, 19, 0.50)"
        border_idle = "rgba(255, 106, 19, 0.45)"
        border_idle_strong = "rgba(255, 106, 19, 0.75)"
        panel_bg = "#202226"
        panel_bg_soft = "#202226"
        panel_glass = "#202226"
        panel_orbit = "#202226"
        ok = "rgba(255, 127, 0, 0.90)"
        ok_quiet = "rgba(255, 127, 0, 0.08)"
        standby = "rgba(244, 255, 255, 0.60)"
        standby_quiet = "rgba(244, 255, 255, 0.06)"
        drift = "rgba(255, 127, 0, 0.90)"
        drift_quiet = "rgba(255, 127, 0, 0.10)"
        warn = "rgba(255, 80, 40, 0.95)"
        warn_quiet = "rgba(255, 80, 40, 0.10)"
    else:
        accent = theme_rgba("accent_dark", 0.86)
        accent_quiet = theme_rgba("accent_dark", 0.03)
        idle_highlight = theme_rgba("accent_dark", 0.09)
        accent_glass = theme_rgba("accent_dark", 0.07)
        accent_glass_hover = theme_rgba("accent_dark", 0.12)
        accent_active = theme_rgba("accent_dark", 0.16)
        accent_ring = theme_rgba("accent_dark", 0.34)
        border_idle = "rgba(255, 255, 255, 0.08)" if is_dark else "rgba(0, 0, 0, 0.08)"
        border_idle_strong = "rgba(255, 255, 255, 0.14)" if is_dark else "rgba(0, 0, 0, 0.14)"
        panel_bg = theme_rgba("panel", 0.84 if is_solarized_dark else 0.76)
        panel_bg_soft = theme_rgba("panel", 0.62 if is_solarized_dark else 0.52)
        panel_glass = theme_rgba("panel", 0.40 if is_solarized_dark else 0.32)
        panel_orbit = (
            "qradialgradient(cx:0.5, cy:0.45, radius:0.98, fx:0.5, fy:0.45, "
            f"stop:0 {panel_bg}, stop:0.6 {panel_bg_soft}, stop:1 {theme_rgba('panel', 0.22)})"
        )
        ok = theme_rgba("strip_soft", 0.86)
        ok_quiet = theme_rgba("strip_soft", 0.05)
        standby = theme_rgba("hover", 0.84)
        standby_quiet = theme_rgba("hover", 0.08)
        drift = "rgba(181, 137, 0, 0.88)"
        drift_quiet = "rgba(181, 137, 0, 0.10)"
        warn = theme_rgba("privacy_warn", 0.88)
        warn_quiet = theme_rgba("privacy_warn", 0.06)

    # Failure / cancellation chrome — in Aurora, `warn` stays the soft bone ramp for at-risk / flagged
    # UI; hard failures and cancelled runs use `danger` (vermillion) instead.
    danger = warn
    danger_quiet = warn_quiet
    if use_aurora:
        danger = "rgba(230, 62, 43, 0.94)"
        danger_quiet = "rgba(230, 62, 43, 0.12)"

    # Completed / success-adjacent badge text: Aurora uses radiation green; other schemes keep process accent.
    success_mark = accent
    if use_aurora:
        success_mark = ok

    if use_neo_swiss:
        accent = "rgba(197, 122, 46, 0.95)"
        accent_quiet = "rgba(197, 122, 46, 0.07)"
        idle_highlight = "rgba(197, 122, 46, 0.12)"
        accent_glass = "rgba(197, 122, 46, 0.12)"
        accent_glass_hover = "rgba(197, 122, 46, 0.18)"
        accent_active = "rgba(197, 122, 46, 0.22)"
        accent_ring = "rgba(197, 122, 46, 0.34)"
        success_mark = ok

    # [SELECTION] App-wide selected/clicked color is its own semantic channel.
    # Do not reuse the scheme accent here: dark mode and aurora use different
    # surface/accent ramps but share the same selected-state violet.
    selection_bg = cvops_rgba("selection_active", 0.88)
    selection_bg_pressed = cvops_rgba("selection_active", 0.96)
    selection_edge = cvops_rgba("selection_edge", 0.92)
    selection_fg = cvops_color("selection_text")
    scenario_selection_bg = selection_bg
    scenario_selection_fg = selection_fg

    if custom_text:
        text_primary = custom_text
    if custom_muted_text:
        text_muted = custom_muted_text
    if custom_accent:
        accent = custom_accent
        accent_quiet = _rgba_from_override(custom_accent, 0.08)
        idle_highlight = _rgba_from_override(custom_accent, 0.14)
        accent_glass = _rgba_from_override(custom_accent, 0.12)
        accent_glass_hover = _rgba_from_override(custom_accent, 0.20)
        accent_active = _rgba_from_override(custom_accent, 0.28)
        accent_ring = _rgba_from_override(custom_accent, 0.42)

    custom_title_text = _qss_color_override(title_text_color)
    custom_title_bg = _qss_color_override(title_background_color)
    _set_cvops_runtime_role_override("title_card_text", custom_title_text)
    _set_cvops_runtime_role_override("title_card_bg", custom_title_bg)
    title_card_bg = cvops_color("title_card_bg")
    title_card_text = cvops_color("title_card_text")

    top_tab_text = title_card_text if use_beacon else text_muted
    top_tab_hover_text = title_card_text if use_beacon else text_primary

    # Shared layout tokens — button shape can soften controls without rounding every panel.
    is_fire = scheme == "fire"
    r_surface = "0px"
    r_tile = "0px"
    r_input = "0px"
    r_pill = "0px"
    r_button = "0px"
    r_tab = "0px"
    if cvops_button_shape() == "radial":
        r_button = "8px"
        r_tab = "8px"
    if use_substrate:
        fz_caps = "10px"
        fz_ui = "11px"
        fz_title = "14px"
        fz_mono = "10px"
        fz_emphasis = "12px"
    else:
        fz_caps = "11px"
        fz_ui = "13px"
        fz_title = "17px"
        fz_mono = "11px"
        fz_emphasis = fz_ui
    btn_pad = "3px 10px"
    input_pad = "1px 5px"
    tab_pad_orbital = "5px 11px"
    tab_pad_detail = "4px 10px"
    tab_pad_generic = "3px 8px"
    item_pad = "2px 5px"
    header_pad = "3px 6px"
    tab_margin = "2px"
    tab_min_width = "76px"
    pane_pad = "3px"

    root_font_stack = (
        '"Avenir Next", "Hiragino Sans", "Yu Gothic", "Noto Sans JP", "IBM Plex Sans", "Segoe UI", sans-serif'
        if japandi
        else '"Inter", "IBM Plex Sans", "Avenir Next", "Segoe UI", "Roboto", sans-serif'
        if red_black or scheme in ("marathon", "wear_marathon", "fire") or use_substrate
        else '"IBM Plex Sans", "Avenir Next", "Segoe UI", "Roboto", sans-serif'
        if is_solarized_dark
        else "inherit"
    )
    root_letter_spacing = "0px"
    assets = _ensure_theme_assets() if red_black else {}
    has_dots = red_black and all(
        f"dot_{k}" in assets for k in ("accent", "ok", "warn", "drift", "standby", "muted")
    )
    pill_pad_ws = "3px 9px 3px 16px" if has_dots else "3px 8px"
    pill_pad_signal = "3px 9px 3px 16px" if has_dots else "3px 9px"
    pill_pad_status = "3px 10px 3px 18px" if has_dots else "3px 10px"
    dot_pos = "background-repeat: no-repeat; background-position: 6px center;" if has_dots else ""

    if has_dots:
        def _dot(name: str) -> str:
            return f'background-image: url("{assets[f"dot_{name}"]}");'

        dot_rules = f"""
/* ---------- Status pill leading dots (red/black theme) ---------- */
QLabel#wsStatus                            {{ {_dot('standby')} }}
QLabel#wsStatus[state="connecting"]        {{ {_dot('standby')} }}
QLabel#wsStatus[state="live"]              {{ {_dot('ok')} }}
QLabel#wsStatus[state="disconnected"]      {{ {_dot('warn')} }}

QLabel#signalPill                          {{ {_dot('muted')} }}
QLabel#signalPill[signal="clear"]          {{ {_dot('ok')} }}
QLabel#signalPill[signal="flagged"]        {{ {_dot('warn')} }}
QLabel#signalPill[signal="warning"]        {{ {_dot('drift')} }}
QLabel#signalPill[signal="idle"]           {{ {_dot('standby')} }}

QLabel#statusPill                          {{ {_dot('muted')} }}
QLabel#statusPill[status="dataset"],
QLabel#statusPill[status="training"]       {{ {_dot('accent')} }}
QLabel#statusPill[status="trained"]        {{ {_dot('ok')} }}
QLabel#statusPill[status="ready"]          {{ {_dot('ok')} }}
QLabel#statusPill[status="error"]          {{ {_dot('warn')} }}
QLabel#statusPill[status="warning"],
QLabel#statusPill[status="drift"]          {{ {_dot('drift')} }}
QLabel#statusPill[status="idle"]           {{ {_dot('standby')} }}
"""
    else:
        dot_rules = ""
    if red_black and assets.get("noise"):
        surface_root_background = f'#0e0e0e url("{assets["noise"]}") repeat'
    elif use_beacon:
        surface_root_background = "#A4ACB8"
    elif red_black:
        surface_root_background = "#0e0e0e"
    elif japandi:
        surface_root_background = "#f3eee3"
    elif use_dark_mode:
        surface_root_background = DARK_MODE_PURE_VOID
    elif use_aurora:
        surface_root_background = AURORA_PURE_VOID
    elif scheme == "wear_marathon":
        surface_root_background = NS_PURE_VOID if use_substrate else (
            "qradialgradient(cx:0.5, cy:0.20, radius:1.24, fx:0.5, fy:0.18, "
            "stop:0 rgba(25, 34, 27, 0.98), stop:0.42 rgba(8, 12, 10, 0.99), stop:1 rgba(0, 0, 0, 1.0))"
        )
    elif scheme == "marathon":
        surface_root_background = NS_PURE_VOID if use_substrate else (
            "qradialgradient(cx:0.5, cy:0.22, radius:1.20, fx:0.5, fy:0.22, "
            "stop:0 rgba(54, 58, 60, 0.98), stop:0.48 rgba(43, 45, 47, 0.99), stop:1 rgba(31, 33, 35, 1.0))"
        )
    elif scheme == "fire":
        surface_root_background = "#202226"
    elif is_solarized_dark:
        surface_root_background = (
            "qradialgradient(cx:0.5, cy:0.45, radius:1.15, fx:0.5, fy:0.45, "
            "stop:0 rgba(0, 59, 71, 0.94), stop:0.45 rgba(0, 43, 54, 0.97), stop:1 rgba(1, 28, 37, 0.99))"
        )
    else:
        surface_root_background = "transparent"

    root_background = surface_root_background
    orbit_pane_background = panel_orbit
    workspace_wallpaper_qss = ""
    wallpaper_applied = False
    if workspace_wallpaper is not None:
        wp_path = Path(str(workspace_wallpaper)).expanduser()
        if wp_path.is_file():
            wallpaper_applied = True
            root_background = "transparent"
            # Tiered translucent shells so wallpaper reads through stacks, cards, and inputs.
            if japandi:
                wp_deep = "rgba(246, 238, 220, 0.44)"
                wp_frame = "rgba(255, 251, 242, 0.56)"
                wp_cell = "rgba(255, 251, 242, 0.64)"
                wp_control = "rgba(255, 251, 242, 0.78)"
            elif scheme == "fire":
                wp_deep = "rgba(26, 28, 32, 0.40)"
                wp_frame = "rgba(32, 34, 38, 0.52)"
                wp_cell = "rgba(38, 40, 44, 0.62)"
                wp_control = "rgba(44, 46, 50, 0.74)"
            elif use_aurora:
                if use_dark_mode:
                    wp_deep = "rgba(4, 6, 7, 0.40)"
                    wp_frame = "rgba(18, 20, 24, 0.50)"
                    wp_cell = "rgba(24, 27, 31, 0.60)"
                    wp_control = "rgba(33, 37, 43, 0.72)"
                else:
                    wp_deep = "rgba(5, 8, 7, 0.40)"
                    wp_frame = "rgba(18, 24, 21, 0.50)"
                    wp_cell = "rgba(24, 32, 28, 0.60)"
                    wp_control = "rgba(34, 42, 38, 0.72)"
            elif red_black or scheme in ("marathon", "wear_marathon"):
                wp_deep = "rgba(14, 14, 14, 0.40)"
                wp_frame = "rgba(21, 21, 21, 0.50)"
                wp_cell = "rgba(26, 26, 26, 0.60)"
                wp_control = "rgba(34, 34, 34, 0.72)"
            elif is_solarized_dark:
                wp_deep = "rgba(0, 36, 45, 0.44)"
                wp_frame = "rgba(0, 51, 63, 0.54)"
                wp_cell = "rgba(7, 65, 78, 0.62)"
                wp_control = "rgba(13, 80, 95, 0.74)"
            elif is_dark:
                wp_deep = "rgba(16, 18, 20, 0.42)"
                wp_frame = "rgba(26, 28, 32, 0.52)"
                wp_cell = "rgba(34, 36, 40, 0.62)"
                wp_control = "rgba(42, 44, 48, 0.74)"
            else:
                wp_deep = "rgba(252, 252, 254, 0.50)"
                wp_frame = "rgba(253, 253, 255, 0.62)"
                wp_cell = "rgba(253, 254, 255, 0.70)"
                wp_control = "rgba(255, 255, 255, 0.82)"
            if backdrop_blend is not None:
                wp_deep = scale_rgba_string(
                    wp_deep, scale_pct=backdrop_blend.tabs_scale_pct
                )
                wp_frame = scale_rgba_string(
                    wp_frame, scale_pct=backdrop_blend.frames_scale_pct
                )
                wp_cell = scale_rgba_string(
                    wp_cell, scale_pct=backdrop_blend.cells_scale_pct
                )
                wp_control = scale_rgba_string(
                    wp_control, scale_pct=backdrop_blend.controls_scale_pct
                )
            orbit_pane_background = wp_deep
            panel_glass = wp_frame
            panel_bg_soft = wp_cell
            panel_bg = wp_control

            workspace_wallpaper_qss = f"""
/* [WORKBENCH WALLPAPER] image backbone + lifted transparent stacks */
QMainWindow#cvOpsWindow QStackedWidget {{
    background: transparent;
}}

QMainWindow#cvOpsWindow QScrollArea,
QMainWindow#cvOpsWindow QScrollArea > QWidget,
QMainWindow#cvOpsWindow QScrollArea > QWidget > QWidget {{
    background: transparent;
}}

QMainWindow#cvOpsWindow QScrollArea QWidget#qt_scrollarea_viewport {{
    background: transparent;
}}

QMainWindow#cvOpsWindow QSplitter {{
    background: transparent;
}}
"""

    if not wallpaper_applied:
        # No image is painted behind the chrome, so every surface tier must be
        # flat and fully opaque. Some schemes otherwise keep translucent rgba
        # fills and a radial-gradient orbit pane that — with nothing behind them —
        # composite against the native window and read as a glossy macOS sheen.
        # Flattening here makes "background image off" mean uniformly matte,
        # independent of the platform style. (Dark mode / Aurora are already flat
        # opaque hex, so this is a no-op for them.)
        panel_bg = _flatten_surface(panel_bg, fallback=panel_bg)
        panel_bg_soft = _flatten_surface(panel_bg_soft, fallback=panel_bg)
        panel_glass = _flatten_surface(panel_glass, fallback=panel_bg)
        orbit_pane_background = _flatten_surface(orbit_pane_background, fallback=panel_bg)
        if root_background == "transparent":
            root_background = panel_glass

    if custom_root_bg:
        root_background = custom_root_bg
    if custom_panel_bg:
        panel_glass = custom_panel_bg
        panel_bg_soft = custom_panel_bg
        orbit_pane_background = custom_panel_bg
    if custom_control_bg:
        panel_bg = custom_control_bg

    cell_out_bg = (
        panel_bg_soft if use_beacon else
        DARK_MODE_PURE_VOID if use_dark_mode else
        AURORA_PURE_VOID if use_aurora else
        NS_PURE_VOID if use_substrate else
        panel_glass
    )
    log_view_bg = (
        panel_bg if use_beacon else
        DARK_MODE_PURE_VOID if use_dark_mode else
        AURORA_PURE_VOID if use_aurora else
        NS_PURE_VOID if use_substrate else
        panel_bg
    )
    if custom_panel_bg:
        cell_out_bg = custom_panel_bg
    if custom_control_bg:
        log_view_bg = custom_control_bg
    raw_json_bg = (
        panel_bg_soft if use_beacon else
        DARK_MODE_VOID if use_dark_mode else
        AURORA_VOID if use_aurora else
        NS_VOID if use_substrate else
        panel_bg
    )
    if custom_control_bg:
        raw_json_bg = custom_control_bg
    elif custom_panel_bg:
        raw_json_bg = custom_panel_bg

    neo_swiss_qss = ""
    if use_substrate:
        neo_swiss_qss = f"""
/* ---------- Neo-Swiss overlay (UI-001) — hairline box + instrument density ---------- */
/* Global QFrame top/bottom borders were removed: they boxed every nested
   frame, stacking border-on-border. Cards opt into their own edge via
   #opsCell; everything else stays free-floating (group by spacing + type). */
QMainWindow#cvOpsWindow QLineEdit,
QMainWindow#cvOpsWindow QTextEdit,
QMainWindow#cvOpsWindow QPlainTextEdit,
QMainWindow#cvOpsWindow QSpinBox,
QMainWindow#cvOpsWindow QDoubleSpinBox {{
    border-top: 1px solid {border_idle};
    border-bottom: 1px solid {border_idle};
}}
QMainWindow#cvOpsWindow QTabWidget::pane {{
    border-top: 1px solid {border_idle};
    border-bottom: 1px solid {border_idle};
}}
QMainWindow#cvOpsWindow QTableWidget {{
    border-top: 1px solid {border_idle};
    border-bottom: 1px solid {border_idle};
}}
QMainWindow#cvOpsWindow QListWidget {{
    border-top: 1px solid {border_idle};
    border-bottom: 1px solid {border_idle};
}}
QMainWindow#cvOpsWindow QTabWidget#orbitalTabs::pane,
QMainWindow#cvOpsWindow QTabWidget#catalogDetailTabs::pane {{
    border-top: 1px solid {border_idle_strong};
    border-bottom: 1px solid {border_idle_strong};
}}
/* Splitter dividers blend in: a faint neutral line (not the saturated navy
   border accent), so the pane "sliders" read as the same super-thin chrome as
   every other border instead of bold cyan bars. */
QMainWindow#cvOpsWindow QSplitter::handle {{
    background: {idle_highlight};
}}
QMainWindow#cvOpsWindow QSplitter::handle:horizontal {{
    width: 1px;
}}
QMainWindow#cvOpsWindow QSplitter::handle:vertical {{
    height: 1px;
}}
QMainWindow#cvOpsWindow QSplitter::handle:hover {{
    background: {accent_active};
}}
QMainWindow#cvOpsWindow QScrollBar::handle:vertical,
QMainWindow#cvOpsWindow QScrollBar::handle:horizontal {{
    border-radius: 1px;
}}
QTextEdit#logView, QTextEdit#cellOutput {{
    padding: {pane_pad};
}}
QLabel#wsStatus, QLabel#signalPill, QLabel#statusPill {{
    border-radius: 0px;
    padding: 2px 5px;
}}
QLabel#cellHeader {{
    font-weight: 500;
}}
QLabel#cellName {{
    font-weight: 500;
}}
QMainWindow#cvOpsWindow QTabBar::tab {{
    font-weight: 500;
}}
"""

    beacon_qss = ""
    if use_beacon:
        beacon_gold = "rgba(255, 210, 53, 0.94)"
        beacon_red = title_card_bg
        beacon_paper = title_card_text
        beacon_qss = f"""
/* ---------- Beacon colorway — vermillion title tags + golden auxiliary text ---------- */
QMainWindow#cvOpsWindow QLabel[isTitle="true"],
QMainWindow#cvOpsWindow QToolButton[isTitle="true"] {{
    background: {beacon_red};
    color: {beacon_paper};
    border: 1px solid {beacon_red};
    padding: 3px 8px;
    font-family: "JetBrains Mono", "IBM Plex Mono", "Menlo", monospace;
    font-size: {fz_caps};
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}
QMainWindow#cvOpsWindow QLabel[muted="true"],
QMainWindow#cvOpsWindow QLabel#serviceStatus,
QMainWindow#cvOpsWindow QLabel#datasetMeta,
QMainWindow#cvOpsWindow QLabel#artifactDetail,
QMainWindow#cvOpsWindow QLabel#scenarioCardMeta,
QMainWindow#cvOpsWindow QLabel#cellSubtitle,
QMainWindow#cvOpsWindow QLabel#cellPath {{
    color: {beacon_gold};
}}
"""

    custom_title_qss = ""
    custom_title_text = _qss_color_override(title_text_color)
    custom_title_bg = _qss_color_override(title_background_color)
    if custom_title_text or custom_title_bg:
        custom_title_props: list[str] = []
        if custom_title_text:
            custom_title_props.append(f"color: {custom_title_text};")
        if custom_title_bg:
            custom_title_props.extend(
                [
                    f"background: {custom_title_bg};",
                    f"border: 1px solid {custom_title_bg};",
                    "padding: 3px 8px;",
                ]
            )
        custom_title_qss = (
            "/* ---------- User title color overrides ---------- */\n"
            "QMainWindow#cvOpsWindow QLabel[isTitle=\"true\"],\n"
            "QMainWindow#cvOpsWindow QToolButton[isTitle=\"true\"] {\n"
            + "\n".join(f"    {line}" for line in custom_title_props)
            + "\n}\n"
        )

    notes_webcam_aurora_qss = ""
    if use_aurora:
        notes_webcam_aurora_qss = f"""
/* Notes — webcam stop control reads as alert red on Aurora */
QWidget#notesPanel QPushButton#notesWebcamToggle[webcamActive="true"] {{
    color: {danger};
    background-color: {danger_quiet};
    border: 1px solid {danger};
    font-weight: 700;
}}
QWidget#notesPanel QPushButton#notesWebcamToggle[webcamActive="true"]:hover {{
    background-color: rgba(255, 255, 255, 0.07);
    color: {danger};
    border-color: {danger};
}}
QWidget#notesPanel QPushButton#notesWebcamToggle[webcamActive="true"]:pressed {{
    background-color: rgba(255, 255, 255, 0.12);
}}
"""

    semantic_token_qss = f"""
/* ---------- CV Ops semantic token guardrails ----------
   These rules intentionally sit last. They keep scheme-neutral state tokens
   from being reinterpreted as theme accent/warning colors by the global app
   stylesheet or by widget-local stylesheet normalization. */
QMainWindow#cvOpsWindow QLabel[isTitle="true"],
QMainWindow#cvOpsWindow QToolButton[isTitle="true"],
QMainWindow#cvOpsWindow QLabel#cvOpsSplitPaneTitle {{
    color: {title_card_text};
    background: {title_card_bg};
    border: 1px solid {title_card_bg};
}}
QMainWindow#cvOpsWindow QTabBar::tab:selected,
QMainWindow#cvOpsWindow QPushButton#cvOpsMainTabNavButton:checked,
QMainWindow#cvOpsWindow QPushButton[navToggle="true"]:checked,
QMainWindow#cvOpsWindow QPushButton[navToggle="true"][paneVisible="true"] {{
    color: {selection_fg};
    background: {selection_bg};
}}
QMainWindow#cvOpsWindow QListWidget::item:selected,
QMainWindow#cvOpsWindow QListWidget::item:selected:active,
QMainWindow#cvOpsWindow QListWidget::item:selected:!active,
QMainWindow#cvOpsWindow QTableWidget::item:selected,
QMainWindow#cvOpsWindow QTableWidget::item:selected:active,
QMainWindow#cvOpsWindow QTableWidget::item:selected:!active {{
    background: {selection_bg};
    color: {selection_fg};
}}
QMainWindow#cvOpsWindow QComboBox QAbstractItemView {{
    selection-background-color: {selection_bg};
    selection-color: {selection_fg};
}}
"""

    return f"""
/* ---------- Root tone / typography ---------- */
QMainWindow, QWidget {{
    font-family: {root_font_stack};
    letter-spacing: {root_letter_spacing};
}}

QMainWindow#cvOpsWindow {{
    background: {root_background};
    color: {text_primary};
}}

QMainWindow#cvOpsWindow > QWidget {{
    background: {root_background};
}}

QWidget#cvOpsRoot {{
    background: {root_background};
    color: {text_primary};
}}

QMainWindow#cvOpsWindow QStatusBar {{
    border: none;
    background: {root_background};
}}

QMainWindow#cvOpsWindow QWidget#eventPulse {{
    border: none;
}}

QMainWindow#cvOpsWindow QWidget#cvOpsBottomPulseBar {{
    border: none;
    background: {root_background};
}}

QStackedWidget,
QScrollArea,
QScrollArea > QWidget,
QScrollArea > QWidget > QWidget {{
    background: {root_background};
}}

/* ---------- Radial dense ops scaffold — sides only ---------- */
QTabWidget#orbitalTabs::pane {{
    border: none;
    border-left: 1px solid {border_idle_strong};
    border-right: 1px solid {border_idle_strong};
    border-radius: {r_surface};
    background: {orbit_pane_background};
    top: -1px;
}}
QTabWidget#orbitalTabs QTabBar::tab {{
    border: none;
    border-radius: {r_tab};
    background: transparent;
    color: {top_tab_text};
    padding: {tab_pad_orbital};
    margin-right: {tab_margin};
    font-size: {fz_ui};
    font-weight: 600;
}}
QTabWidget#orbitalTabs QTabBar::tab:hover {{
    color: {top_tab_hover_text};
    background: {accent_quiet};
}}
QTabWidget#orbitalTabs QTabBar::tab:selected {{
    color: {selection_fg};
    background: {selection_bg};
    border-left: 1px solid {selection_edge};
    border-right: 1px solid {selection_edge};
}}

QPushButton#cvOpsMainTabNavButton {{
    border: none;
    border-radius: {r_button};
    background: transparent;
    color: {top_tab_text};
    padding: {tab_pad_orbital};
    margin-right: {tab_margin};
    font-size: {fz_ui};
    font-weight: 600;
}}
QPushButton#cvOpsMainTabNavButton:hover {{
    color: {top_tab_hover_text};
    background: {accent_quiet};
}}
QPushButton#cvOpsMainTabNavButton:checked {{
    color: {selection_fg};
    background: {selection_bg};
    border-left: 1px solid {selection_edge};
    border-right: 1px solid {selection_edge};
}}

QTabWidget#catalogDetailTabs::pane {{
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_surface};
    background: transparent;
    top: -1px;
    padding: {pane_pad};
}}
QTabWidget#catalogDetailTabs QTabBar::tab {{
    border: none;
    border-radius: {r_tab};
    background: transparent;
    color: {top_tab_text};
    padding: {tab_pad_detail};
    margin-right: {tab_margin};
    min-width: {tab_min_width};
    font-size: {fz_ui};
    font-weight: 600;
}}
QTabWidget#catalogDetailTabs QTabBar::tab:hover {{
    color: {top_tab_hover_text};
    background: {accent_quiet};
}}
QTabWidget#catalogDetailTabs QTabBar::tab:selected {{
    color: {selection_fg};
    background: {selection_bg};
    border-left: 1px solid {selection_edge};
    border-right: 1px solid {selection_edge};
}}

/* ---------- Card/cell chrome — sides only, squared ---------- */
QFrame {{
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    background: {panel_glass};
    border-radius: {r_tile};
}}
QFrame#opsCell {{
    border: 1px solid {border_idle};
    background: {panel_bg_soft};
    border-radius: {r_surface};
}}
QFrame#opsCell:hover {{
    border-color: {border_idle_strong};
    background: {panel_bg};
}}
QLabel[isTitle="true"] {{
    color: {title_card_text};
    background: {title_card_bg};
    border: 1px solid {title_card_bg};
    padding: 3px 8px;
    border-radius: {r_button};
    font-size: {fz_emphasis};
    font-weight: 500;
    letter-spacing: -0.01em;
}}
QToolButton[isTitle="true"] {{
    border: 1px solid {title_card_bg};
    background: {title_card_bg};
    color: {title_card_text};
    border-radius: {r_button};
    padding: 3px 8px;
    font-size: {fz_emphasis};
    font-weight: 500;
    letter-spacing: -0.01em;
}}
QToolButton[isTitle="true"]:hover {{
    background: {title_card_bg};
    color: {title_card_text};
    border-color: {title_card_bg};
}}

/* Primary action — outline from CvOpsParallelogram* paint; keep QSS for fill + text. */
QPushButton {{
    border: none;
    background: {panel_bg};
    color: {text_primary};
    padding: {btn_pad};
    min-height: 14px;
    font-size: {fz_ui};
    font-weight: 600;
    letter-spacing: 0px;
    border-radius: {r_button};
}}
QPushButton:hover {{
    background: {panel_bg};
    color: {accent};
}}
QPushButton:pressed {{
    background: {selection_bg_pressed};
    color: {selection_fg};
}}
QPushButton:checked {{
    background: {selection_bg};
    color: {selection_fg};
}}
QPushButton[navToggle="true"] {{
    background: transparent;
    color: {top_tab_text};
}}
QPushButton[navToggle="true"]:hover {{
    background: {accent_quiet};
    color: {top_tab_hover_text};
}}
QPushButton[navToggle="true"]:checked,
QPushButton[navToggle="true"][paneVisible="true"] {{
    background: {selection_bg};
    color: {selection_fg};
    border-left: 1px solid {selection_edge};
    border-right: 1px solid {selection_edge};
}}
QPushButton[navToggle="true"]:disabled {{
    color: {text_muted};
    border-left: 1px solid transparent;
    border-right: 1px solid transparent;
}}
QPushButton[idleHighlight="true"] {{
    background: {idle_highlight};
}}
QPushButton[idleHighlight="true"]:hover {{
    background: {accent_glass_hover};
}}
QPushButton[idleHighlight="true"]:checked {{
    background: {selection_bg};
    color: {selection_fg};
}}
QPushButton[slotFilter="true"] {{
    padding: 3px 6px;
    font-size: {fz_caps};
    font-weight: 600;
    background: {idle_highlight};
}}
QPushButton[slotFilter="true"]:hover {{
    background: {accent_glass_hover};
}}
QPushButton[slotFilter="true"]:checked {{
    background: {selection_bg};
    color: {selection_fg};
}}
QPushButton:disabled {{
    background: transparent;
    color: {text_muted};
}}

/* Inputs — sides only borders, squared. */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    padding: {input_pad};
    border-radius: {r_input};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    background: {panel_bg};
    color: {text_primary};
    min-height: 14px;
}}
QTextEdit, QPlainTextEdit {{
    min-height: 0;
}}
QComboBox {{
    padding: {input_pad};
    border-radius: 0px;
    border: none;
    background: {panel_bg};
    color: {text_primary};
    min-height: 14px;
}}
/* Flat dark menus + dropdown popups. These are top-level floating widgets, so
   the rules are intentionally unscoped (no cvOpsWindow ancestor) to reach them.
   Solid panel_bg fills override macOS's native glossy/vibrant menu rendering so
   dark mode stays universally flat — no sheen, no gradient. */
QMenu {{
    background: {panel_bg};
    color: {text_primary};
    border: 1px solid {border_idle};
    border-radius: 0px;
    padding: 2px;
}}
QMenu::item {{
    background: transparent;
    padding: 4px 18px 4px 14px;
}}
QMenu::item:selected {{
    background: {selection_bg};
    color: {selection_fg};
}}
QMenu::item:disabled {{
    color: {text_muted};
}}
QMenu::separator {{
    height: 1px;
    background: {border_idle};
    margin: 3px 6px;
}}
QComboBox::drop-down {{
    border: none;
    background: transparent;
    width: 16px;
    subcontrol-origin: padding;
    subcontrol-position: center right;
}}
QComboBox::down-arrow {{
    border: none;
    background: transparent;
}}
QComboBox QAbstractItemView {{
    background: {panel_bg};
    color: {text_primary};
    border: 1px solid {border_idle};
    border-radius: 0px;
    outline: none;
    selection-background-color: {selection_bg};
    selection-color: {selection_fg};
}}
QToolButton {{
    padding: {item_pad};
    border-radius: {r_button};
    border: none;
    background: transparent;
    color: {text_primary};
}}
QToolButton:hover {{
    background: {accent_quiet};
    color: {text_primary};
}}
QToolButton:pressed {{
    background: {selection_bg_pressed};
    color: {selection_fg};
}}
QToolButton:checked {{
    background: {selection_bg};
    color: {selection_fg};
}}
QToolButton:disabled {{
    color: {text_muted};
}}
QFrame#systemGuardCard QLabel {{
    font-size: {fz_ui};
    color: {text_primary};
}}
QFrame#systemGuardCard QComboBox {{
    min-height: 20px;
    padding: 4px 8px;
    border-left: 1px solid {border_idle_strong};
    border-right: 1px solid {border_idle_strong};
}}
QFrame#systemGuardCard QPushButton,
QFrame#systemGuardCard QToolButton {{
    min-height: 20px;
    padding: 3px 8px;
}}
QFrame#systemGuardCard QToolButton {{
    text-align: left;
    color: {text_muted};
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    background: {panel_bg_soft};
}}
QFrame#systemGuardCard QToolButton:checked {{
    color: {selection_fg};
    border-left: 1px solid {selection_edge};
    border-right: 1px solid {selection_edge};
    background: {selection_bg};
}}
QListWidget {{
    border-radius: {r_input};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    background: {panel_bg};
    color: {text_primary};
}}
QListWidget::item {{ padding: {item_pad}; }}
QListWidget::item:selected,
QListWidget::item:selected:active,
QListWidget::item:selected:!active {{
    background: {selection_bg};
    color: {selection_fg};
}}

QTableWidget {{
    background: {panel_bg};
    alternate-background-color: {panel_bg_soft};
    color: {text_primary};
    gridline-color: {border_idle};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
}}
QTableWidget::item {{
    padding: {item_pad};
}}
QTableWidget::item:selected,
QTableWidget::item:selected:active,
QTableWidget::item:selected:!active {{
    background: {selection_bg};
    color: {selection_fg};
}}
QTableWidget#scenarioCatalogTable::item:selected,
QTableWidget#scenarioCatalogTable::item:selected:active,
QTableWidget#scenarioCatalogTable::item:selected:!active {{
    background: {scenario_selection_bg};
    color: {scenario_selection_fg};
}}

QHeaderView::section {{
    letter-spacing: 0px;
    padding: {header_pad};
    font-size: {fz_caps};
    font-weight: 600;
    color: {text_muted};
    background: {panel_bg_soft};
    border: 0;
}}

QTabWidget::pane {{
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_tile};
    background: {panel_bg_soft};
    top: -1px;
}}
QTabBar::tab {{
    border: none;
    border-radius: {r_tab};
    background: transparent;
    color: {top_tab_text};
    padding: {tab_pad_generic};
    margin-right: {tab_margin};
    font-size: {fz_ui};
    font-weight: 600;
}}
QTabBar::tab:hover {{
    color: {top_tab_hover_text};
    background: {accent_quiet};
}}
QTabBar::tab:selected {{
    color: {selection_fg};
    background: {selection_bg};
    border-left: 1px solid {selection_edge};
    border-right: 1px solid {selection_edge};
}}

/* ---------- App chrome (CV Ops window stacks global + this sheet) ---------- */
QSplitter::handle {{
    background: {idle_highlight};
}}
QSplitter::handle:horizontal {{
    width: 1px;
    margin: 0;
}}
QSplitter::handle:vertical {{
    height: 1px;
    margin: 0;
}}
QSplitter::handle:hover {{
    background: {accent_active};
}}

QScrollBar:vertical {{
    width: 3px;
    background: transparent;
    margin: 1px 0;
}}
QScrollBar::handle:vertical {{
    background: {border_idle_strong};
    min-height: 16px;
    border-radius: 1px;
    margin: 0px;
}}
QScrollBar::handle:vertical:hover {{
    background: {accent_active};
}}
QScrollBar:horizontal {{
    height: 3px;
    background: transparent;
    margin: 0 1px;
}}
QScrollBar::handle:horizontal {{
    background: {border_idle_strong};
    min-width: 16px;
    border-radius: 1px;
    margin: 0px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {accent_active};
}}

/* Sliders — fully flat groove + handle so the native style cannot inject a
   gradient/bevel ("sheen") into the track or thumb. */
QSlider {{
    background: transparent;
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {border_idle};
    border: none;
    border-radius: 1px;
}}
QSlider::sub-page:horizontal {{
    background: {accent};
    border: none;
    border-radius: 1px;
}}
QSlider::add-page:horizontal {{
    background: {border_idle};
    border: none;
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    background: {accent};
    border: none;
    width: 12px;
    margin: -5px 0;
    border-radius: 6px;
}}
QSlider::handle:horizontal:hover {{
    background: {accent_active};
}}
QSlider::groove:vertical {{
    width: 3px;
    background: {border_idle};
    border: none;
    border-radius: 1px;
}}
QSlider::sub-page:vertical {{
    background: {border_idle};
    border: none;
    border-radius: 1px;
}}
QSlider::add-page:vertical {{
    background: {accent};
    border: none;
    border-radius: 1px;
}}
QSlider::handle:vertical {{
    background: {accent};
    border: none;
    height: 12px;
    margin: 0 -5px;
    border-radius: 6px;
}}
QSlider::handle:vertical:hover {{
    background: {accent_active};
}}

/* ---------- Status labels ---------- */
QLabel#serviceStatus {{
    font-size: {fz_ui};
    color: {text_muted};
    letter-spacing: 0px;
    font-weight: 600;
}}

/* Generic muted helper for labels created dynamically */
QLabel[muted="true"] {{
    color: {text_muted};
}}

QLabel#trainingHealthBadge {{
    font-family: "JetBrains Mono", "IBM Plex Mono", "SFMono-Regular", "Menlo", "Consolas", monospace;
    font-size: {fz_caps};
    font-weight: 600;
    letter-spacing: 0.05em;
    padding: 1px 5px;
    border-radius: {r_pill};
    background: transparent;
    border: 1px solid {border_idle_strong};
    color: {text_muted};
}}
QLabel#trainingHealthBadge[healthState="healthy"] {{
    color: {ok};
    border-color: {ok};
}}
QLabel#trainingHealthBadge[healthState="watch"] {{
    color: {drift};
    border-color: {drift};
}}
QLabel#trainingHealthBadge[healthState="at_risk"] {{
    color: {warn};
    border-color: {warn};
}}
QLabel#trainingHealthBadge[healthState="completed"] {{
    color: {success_mark};
    border-color: {success_mark};
}}
QLabel#trainingHealthBadge[healthState="starting"] {{
    color: {accent};
    border-color: {accent};
}}
QLabel#trainingHealthBadge[healthState="failed"] {{
    color: {danger};
    border-color: {danger};
}}
QLabel#trainingHealthBadge[healthState="cancelled"] {{
    color: {danger};
    border-color: {danger};
}}
QLabel#trainingHealthBadge[healthState="idle"] {{
    color: {standby};
    border-color: {standby};
}}

QLabel#wsStatus {{
    font-size: {fz_caps};
    padding: {pill_pad_ws};
    border-radius: {r_pill};
    background-color: {standby_quiet};
    color: {standby};
    border: 1px solid {standby};
    {dot_pos}
}}
QLabel#wsStatus[state="connecting"] {{
    background-color: {standby_quiet};
    color: {standby};
    border: 1px solid {standby};
}}
QLabel#wsStatus[state="live"] {{
    background-color: {ok_quiet};
    color: {ok};
    border: 1px solid {ok};
}}
QLabel#wsStatus[state="disconnected"] {{
    background-color: {warn_quiet};
    color: {warn};
    border: 1px solid {warn};
}}
QToolButton#wsRefreshButton {{
    min-width: 20px;
    max-width: 24px;
    min-height: 18px;
    max-height: 22px;
    margin-left: 3px;
    padding: 1px;
    border-radius: {r_button};
    background-color: {panel_bg};
    color: {text_muted};
    border: 1px solid {border_idle};
}}
QToolButton#wsRefreshButton:hover {{
    background-color: {accent_quiet};
    color: {accent};
    border-color: {accent};
}}
QToolButton#wsRefreshButton:pressed {{
    background-color: {selection_bg_pressed};
    color: {selection_fg};
}}
QToolButton#wsRefreshButton:disabled {{
    background-color: {panel_bg};
    color: {text_muted};
    border-color: {border_idle};
}}

/* ---------- Pills ---------- */
QLabel#signalPill {{
    padding: {pill_pad_signal};
    border-radius: {r_pill};
    background-color: transparent;
    border: 1px solid {border_idle_strong};
    font-weight: 600;
    font-size: {fz_caps};
    letter-spacing: 0px;
    color: {text_muted};
    {dot_pos}
}}
QLabel#signalPill[signal="clear"] {{
    color: {ok};
    border-color: {ok};
}}
QLabel#signalPill[signal="flagged"] {{
    color: {warn};
    border-color: {warn};
}}
QLabel#signalPill[signal="warning"] {{
    color: {drift};
    border-color: {drift};
}}
QLabel#signalPill[signal="idle"] {{
    color: {standby};
    border-color: {standby};
}}

QLabel#statusPill {{
    padding: {pill_pad_status};
    border-radius: {r_pill};
    font-weight: 600;
    font-size: {fz_caps};
    letter-spacing: 0px;
    background-color: {panel_bg};
    color: {text_primary};
    border: 1px solid {border_idle};
    {dot_pos}
}}
QLabel#statusPill[status="dataset"],
QLabel#statusPill[status="training"] {{
    color: {accent};
    border-color: {accent};
}}
QLabel#statusPill[status="trained"] {{
    color: {ok};
    border-color: {ok};
}}
QLabel#statusPill[status="ready"] {{
    color: {ok};
    border-color: {ok};
}}
QLabel#statusPill[status="error"] {{
    color: {danger};
    border-color: {danger};
}}
QLabel#statusPill[status="warning"],
QLabel#statusPill[status="drift"] {{
    color: {drift};
    border-color: {drift};
    background-color: {drift_quiet};
}}
QLabel#statusPill[status="idle"] {{
    color: {standby};
    border-color: {standby};
    background-color: {standby_quiet};
}}

/* Readiness row: a faint background band, not a box. The individual items
   are free-floating label + value pairs; their ok/warn/error state is carried
   by the VALUE text color instead of a per-item border box. */
QFrame#readinessStrip {{
    background: {panel_bg_soft};
    border: none;
    border-radius: {r_surface};
}}
QFrame#readinessItem {{
    background: transparent;
    border: none;
}}
QLabel#readinessLabel {{
    color: {text_muted};
    font-size: {fz_caps};
    font-weight: 600;
}}
QLabel#readinessValue {{
    color: {text_primary};
    font-size: {fz_ui};
    font-weight: 700;
}}
QFrame#readinessItem[state="ok"] QLabel#readinessValue {{
    color: {ok};
}}
QFrame#readinessItem[state="warning"] QLabel#readinessValue {{
    color: {drift};
}}
QFrame#readinessItem[state="error"] QLabel#readinessValue {{
    color: {danger};
}}

/* ---------- Drop zone — sides only dashed ---------- */
QFrame#dropZone {{
    border: none;
    border-left: 1px dashed {border_idle_strong};
    border-right: 1px dashed {border_idle_strong};
    border-radius: {r_surface};
    background: {panel_bg_soft};
}}
QFrame#dropZone[state="dragover"] {{
    border-left-color: {ok};
    border-right-color: {ok};
    background: {ok_quiet};
}}
QFrame#dropZone[state="ready"] {{
    border-left-color: {ok};
    border-right-color: {ok};
    background: {panel_bg};
}}
QFrame#dropZone[state="error"] {{
    border-left-color: {danger};
    border-right-color: {danger};
    background: {danger_quiet};
}}

QFrame#notesLibraryToolbar {{
    background: {panel_bg_soft};
    border: 1px solid {border_idle};
    border-radius: {r_surface};
}}
QLabel#notesLibraryProject {{
    color: {text_primary};
    background: transparent;
    font-size: {fz_title};
    font-weight: 700;
}}
QFrame#notesToolbarGroup {{
    background: {panel_bg};
    border: 1px solid {border_idle};
    border-radius: {r_input};
}}
QLabel#notesToolbarGroupLabel {{
    color: {text_muted};
    background: transparent;
    font-size: {fz_caps};
    font-weight: 700;
}}

QListWidget#notesAssetList {{
    background: transparent;
    border: none;
    outline: none;
    padding: 0px;
}}
QListWidget#notesAssetList::item {{
    margin: 0px;
    padding: 0px;
    border: none;
    background: transparent;
}}
QListWidget#notesAssetList::item:selected {{
    background: transparent;
}}

QFrame#notesAssetCard {{
    background: {panel_bg_soft};
    border: 1px solid {border_idle};
    border-radius: {r_surface};
}}
QFrame#notesAssetCard:hover {{
    background: {panel_bg};
    border-color: {accent_glass_hover};
}}
QFrame#notesAssetCard[role="audio"] {{
    background: {panel_bg};
}}
QLabel#notesAssetKindBadge {{
    color: {text_primary};
    background: {accent_glass};
    border: 1px solid {border_idle};
    border-radius: {r_input};
    padding: 2px 8px;
    font-size: {fz_caps};
    font-weight: 700;
}}
QLabel#notesAssetTitle {{
    color: {text_primary};
    background: transparent;
    border: none;
    font-size: {fz_ui};
    font-weight: 700;
}}
QLabel#notesAssetMeta {{
    color: {text_muted};
    background: transparent;
    border: none;
    font-size: {fz_caps};
}}
QLabel#notesAssetPreview {{
    color: {text_primary};
    background: transparent;
    border: none;
    font-size: {fz_ui};
}}

/* ---------- Dashboard — sides only ---------- */
QWidget#dashboardEmbedPanel,
QWidget#dashboardNativePanel {{
    background: {panel_bg};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_surface};
}}
QWidget#dashboardEmbedPanel:hover,
QWidget#dashboardNativePanel:hover {{
    border-left-color: {accent_glass_hover};
    border-right-color: {accent_glass_hover};
}}
QLabel#dashboardStatus {{
    background: {panel_bg_soft};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
    padding: 3px 6px;
    color: {text_muted};
    font-size: {fz_caps};
}}
QLabel#dashboardSummaryLine {{
    background: {panel_glass};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
    padding: 3px 6px;
    color: {text_primary};
    font-size: {fz_ui};
}}

/* ---------- Workbench bottom tray ---------- */
QWidget#cvOpsMainPaneHost {{
    background: transparent;
    border: none;
}}
QWidget#cvOpsSettingsPanel,
QScrollArea#cvOpsSettingsScroll,
QWidget#cvOpsSettingsViewport,
QWidget#cvOpsSettingsInner {{
    background: {panel_bg};
    border: none;
}}
QWidget#cvOpsBottomPaneTray {{
    background: {panel_bg_soft};
    border: none;
    border-top: 1px solid {border_idle_strong};
}}
QFrame#cvOpsBottomTrayCard {{
    background: {panel_bg};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
}}
QWidget#cvOpsBottomTrayCardHeader {{
    background: {panel_glass};
    border: none;
    border-bottom: 1px solid {border_idle};
}}
QLabel#cvOpsSplitPaneTitle {{
    background: {title_card_bg};
    color: {title_card_text};
    border: 1px solid {title_card_bg};
    border-radius: {r_button};
    padding: 3px 8px;
    font-size: {fz_emphasis};
    font-weight: 500;
    letter-spacing: -0.01em;
}}

/* ---------- Result panel — sides only ---------- */
QLabel#overlayPreview {{
    background: {panel_bg_soft};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
}}

QLabel#threeDSectionTitle {{
    color: {text_muted};
    font-size: {fz_caps};
    font-weight: 700;
    border: none;
    padding: 2px 0 6px 0;
}}
QLabel#threeDBreadcrumb {{
    color: {text_muted};
    font-size: {fz_mono};
    font-weight: 600;
    font-family: {WB_FONT_MONO};
    border: none;
    padding: 0 0 2px 0;
}}
QLabel#threeDNavContext {{
    color: {text_muted};
    font-size: {fz_mono};
    border: none;
    padding: 0 0 4px 0;
}}
QTreeWidget#threeDNavOutline {{
    border: none;
    border-radius: {r_input};
    background: {panel_bg};
    border-left: 1px solid {border_idle_strong};
    border-right: 1px solid {border_idle_strong};
    padding: 4px 0;
    outline: none;
}}
QTreeWidget#threeDNavOutline::item {{
    padding: 4px 8px;
    border: none;
}}
QTreeWidget#threeDNavOutline::item:selected {{
    background: {accent_quiet};
    color: {text_primary};
}}
QTreeWidget#threeDNavOutline::item:hover {{
    background: {panel_bg_soft};
}}
QTreeWidget#threeDNavOutline::branch {{
    background: transparent;
}}
QFrame#threeDNavAnchor {{
    border: none;
    border-top: 1px solid {border_idle};
    padding-top: 10px;
    margin-top: 2px;
}}
QFrame#threeDNavAnchor[navId="overview"] {{
    border-top: none;
    padding-top: 0;
    margin-top: 0;
}}
QWidget#threeDControlsColumn {{
    border: none;
    background: transparent;
}}
QFrame#threeDInset {{
    border-radius: {r_input};
    background: {panel_bg};
    border: none;
    border-left: 1px solid {border_idle_strong};
    border-right: 1px solid {border_idle_strong};
}}
QFrame#threeDInset:hover {{
    border-left-color: {accent_glass_hover};
    border-right-color: {accent_glass_hover};
    background: {panel_bg_soft};
}}
QWidget#threeDPanel QPushButton[isPrimary="true"] {{
    border: none;
    background: {accent_quiet};
    padding: 5px 12px;
    min-height: 16px;
}}
QWidget#threeDPanel QPushButton[isPrimary="true"]:hover {{
    background: {accent_glass};
}}
QWidget#threeDPanel QPushButton[buttonRole="secondary"] {{
    border: none;
    background: {panel_bg_soft};
    font-weight: 600;
}}
QWidget#threeDPanel QPushButton[buttonRole="secondary"]:hover {{
    background: {panel_bg};
    color: {accent};
}}
QWidget#threeDPanel QPushButton[buttonRole="secondary"]:disabled {{
    color: {text_muted};
    background: transparent;
}}
QWidget#threeDPanel QComboBox {{
    border: none;
}}
QWidget#threeDPanel QProgressBar {{
    border: none;
    border-left: 1px solid {border_idle_strong};
    border-right: 1px solid {border_idle_strong};
    border-radius: {r_input};
    background: {panel_bg};
    min-height: 10px;
}}
QWidget#threeDPanel QProgressBar::chunk {{
    background: {accent_glass_hover};
}}
QLabel#previewThumb {{
    background: {panel_bg};
    border: none;
    border-left: 1px solid {border_idle_strong};
    border-right: 1px solid {border_idle_strong};
    border-radius: {r_input};
}}
QWidget#threeDPanel QLabel[state="warning"] {{
    color: {warn};
    border-left: 1px solid {warn};
    border-right: 1px solid {warn};
    border-radius: {r_input};
    padding: 3px 6px;
    background: {warn_quiet};
}}
QWidget#threeDPanel QLabel[state="success"] {{
    color: {ok};
    border-left: 1px solid {ok};
    border-right: 1px solid {ok};
    border-radius: {r_input};
    padding: 3px 6px;
    background: {ok_quiet};
}}

QTextEdit#rawJson {{
    background: {raw_json_bg};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
    color: {text_primary};
    selection-background-color: {selection_bg};
    selection-color: {selection_fg};
}}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    selection-background-color: {selection_bg};
    selection-color: {selection_fg};
}}

QTextEdit#logView {{
    background: {log_view_bg};
    color: {text_primary};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
    padding: {pane_pad};
}}

QTextEdit#rawJson:focus,
QTextEdit#logView:focus,
QTextEdit#cellOutput:focus,
QPlainTextEdit#artifactPreview:focus,
QTextBrowser#cardsBrowser:focus {{
    border-color: {accent};
    outline: none;
}}

/* ---------- Section hooks (future CV/ML split) ---------- */
QFrame[section="cv"] {{
    border-color: {accent_active};
}}
QFrame[section="ml"] {{
    border-color: {ok};
}}

/* ---------- Colab-style cell progress ---------- */
QLabel#cellHeader {{
    font-size: {fz_ui};
    font-weight: 600;
    color: {text_muted};
    letter-spacing: 0px;
    padding: 3px 1px;
}}

QFrame#cellCard {{
    border-radius: {r_tile};
    background: {panel_bg};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    padding: {pane_pad};
}}
QFrame#cellCard:hover {{
    border-left-color: {accent_glass_hover};
    border-right-color: {accent_glass_hover};
    background: {panel_bg_soft};
}}
QFrame#cellCard[cellStatus="running"] {{ border-left-color: {accent}; border-right-color: {accent}; }}
QFrame#cellCard[cellStatus="done"] {{ border-left-color: {ok}; border-right-color: {ok}; }}
QFrame#cellCard[cellStatus="error"] {{ border-left-color: {danger}; border-right-color: {danger}; }}
QFrame#cellCard[cellStatus="warning"] {{ border-left-color: {drift}; border-right-color: {drift}; }}
QFrame#cellCard[cellStatus="skipped"] {{ border-left-color: {border_idle}; border-right-color: {border_idle}; }}
QFrame#cellCard[cellStatus="pending"] {{ border-left-color: {border_idle}; border-right-color: {border_idle}; }}

QLabel#cellIcon {{
    font-size: {fz_caps};
    font-weight: 700;
    border: none;
    color: {text_muted};
}}
QLabel#cellIcon[cellStatus="running"] {{ color: {accent}; }}
QLabel#cellIcon[cellStatus="done"] {{ color: {ok}; }}
QLabel#cellIcon[cellStatus="error"] {{ color: {danger}; }}

QLabel#cellName {{
    font-size: {fz_ui};
    font-weight: 600;
    border: none;
    color: {text_primary};
}}
QLabel#cellElapsed {{
    font-size: {fz_caps};
    border: none;
    color: {text_muted};
}}

QTextEdit#cellOutput {{
    font-size: {fz_mono};
    background: {cell_out_bg};
    color: {text_primary};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
    padding: {pane_pad};
}}

QTextEdit#rawJson,
QTextEdit#logView,
QTextEdit#cellOutput,
QPlainTextEdit#artifactPreview,
QLabel#artifactDetail {{
    font-family: "JetBrains Mono", "IBM Plex Mono", "SFMono-Regular", "Menlo", "Consolas", monospace;
}}

/* ---------- Run artifacts — sides only ---------- */
QLabel#artifactPanelTitle {{
    border: none;
    border-left: 1px solid {accent};
    background: {accent_quiet};
    color: {text_primary};
    border-radius: {r_input};
    padding: 1px 7px;
    font-size: {fz_caps};
    font-weight: 700;
}}
QLabel#artifactSectionTitle {{
    border: none;
    border-left: 1px solid {border_idle_strong};
    background: transparent;
    color: {text_muted};
    border-radius: {r_input};
    padding: 0px 6px;
    font-size: {fz_caps};
    font-weight: 700;
}}
QLabel#artifactThumb {{
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    background: {panel_bg};
    border-radius: {r_input};
}}
QLabel#artifactCaption {{
    font-size: {fz_caps};
    color: {text_muted};
}}
QLabel#artifactDetail {{
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
    background: {panel_bg};
    padding: {item_pad};
    font-size: {fz_mono};
    color: {text_muted};
}}
QPlainTextEdit#artifactPreview {{
    font-size: {fz_mono};
    background: {panel_bg};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
    color: {text_primary};
}}

/* ---------- Dataset panel tiles — sides only ---------- */
QLabel#datasetThumb {{
    background: {panel_bg};
    border-radius: {r_tile};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
}}
QLabel#datasetThumb[hasLabel="false"] {{
    border-left-color: {danger};
    border-right-color: {danger};
}}
QLabel#datasetTitle {{
    font-size: {fz_ui};
    color: {text_primary};
    font-weight: 600;
}}
QLabel#datasetTitle[hasLabel="false"] {{
    color: {danger};
}}

/* ---------- Catalog cards browser — sides only ---------- */
QTextBrowser#cardsBrowser {{
    background: {panel_glass};
    color: {text_primary};
    border: none;
    border-left: 1px solid {border_idle};
    border-right: 1px solid {border_idle};
    border-radius: {r_input};
    padding: {pane_pad};
    font-size: {fz_ui};
}}

QLabel#datasetMeta[state="ok"] {{
    color: {text_muted};
}}
QLabel#datasetMeta[state="error"] {{
    color: {danger};
    font-weight: 600;
}}

/* ---------- Orbital depth ---------- */
QMainWindow#cvOpsWindow QFrame#opsCell {{
    border-width: 1px;
    padding: 0px;
}}
QMainWindow#cvOpsWindow QLabel#serviceStatus {{
    font-size: {fz_ui};
}}
QMainWindow#cvOpsWindow QTabWidget#orbitalTabs QTabBar::tab {{
    padding: {tab_pad_orbital};
}}
QMainWindow#cvOpsWindow QPushButton#cvOpsMainTabNavButton {{
    padding: {tab_pad_orbital};
}}

{notes_webcam_aurora_qss}
{dot_rules}
""" + neo_swiss_qss + beacon_qss + custom_title_qss + workspace_wallpaper_qss + semantic_token_qss
