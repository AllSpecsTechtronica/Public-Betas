from __future__ import annotations

import os
import re
from typing import Any, Optional

from PyQt6.QtCore import QEvent, QObject, QTimer
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QLabel, QToolButton, QWidget

# -- Palette roles (RGB); updated by configure_color_scheme --

_SCHEMES: dict[str, dict[str, Any]] = {
    "default": {
        "bg_hex": "#0a0a0a",
        "accent_dark": (220, 38, 38),
        "panel": (22, 22, 22),
        "hover": (34, 34, 34),
        "pressed": (176, 24, 24),
        "input_fill": (16, 16, 16),
        "input_list": (22, 22, 22),
        "strip_soft": (235, 88, 88),
        "privacy_warn": (245, 130, 130),
        "roi": (235, 88, 88),
    },
    "marathon": {
        "bg_hex": "#2b2d2f",
        "accent_dark": (197, 255, 70),
        "panel": (42, 45, 47),
        "hover": (54, 58, 60),
        "pressed": (92, 132, 35),
        "input_fill": (35, 38, 40),
        "input_list": (43, 47, 49),
        "strip_soft": (228, 255, 145),
        "privacy_warn": (197, 255, 70),
        "roi": (208, 255, 115),
    },
    "wear_marathon": {
        "bg_hex": "#000000",
        "accent_dark": (197, 255, 70),
        "panel": (12, 16, 14),
        "hover": (25, 34, 27),
        "pressed": (92, 132, 35),
        "input_fill": (8, 12, 10),
        "input_list": (14, 19, 16),
        "strip_soft": (220, 255, 140),
        "privacy_warn": (255, 180, 171),
        "roi": (208, 255, 115),
    },
    "fire": {
        "bg_hex": "#060709",
        "accent_dark": (248, 164, 64),
        "panel": (28, 31, 36),
        "hover": (42, 46, 52),
        "pressed": (112, 68, 23),
        "input_fill": (20, 22, 26),
        "input_list": (24, 27, 31),
        "strip_soft": (255, 200, 132),
        "privacy_warn": (255, 140, 92),
        "roi": (248, 164, 64),
    },
    "aurora": {
        "bg_hex": "#050807",
        "accent_dark": (168, 180, 172),
        "panel": (24, 32, 28),
        "hover": (34, 42, 38),
        "pressed": (63, 76, 70),
        "input_fill": (18, 24, 21),
        "input_list": (24, 32, 28),
        "strip_soft": (214, 221, 216),
        "privacy_warn": (168, 180, 172),
        "roi": (214, 221, 216),
        "pure_void": (5, 8, 7),
        "void": (10, 14, 12),
        "graphite_1": (18, 24, 21),
        "graphite_2": (24, 32, 28),
        "graphite_3": (34, 42, 38),
        "graphite_4": (45, 56, 51),
        "graphite_5": (63, 76, 70),
        "mist_1": (90, 104, 98),
        "mist_2": (126, 140, 132),
        "mist_3": (168, 180, 172),
        "bone": (214, 221, 216),
        "paper": (232, 237, 233),
    },
    "dark mode": {
        "bg_hex": "#040607",
        "accent_dark": (168, 180, 172),
        "panel": (18, 20, 24),
        "hover": (28, 31, 36),
        "pressed": (48, 54, 62),
        "input_fill": (14, 16, 19),
        "input_list": (18, 20, 24),
        "strip_soft": (214, 221, 216),
        "privacy_warn": (168, 180, 172),
        "roi": (214, 221, 216),
        "pure_void": (4, 6, 7),
        "void": (8, 10, 12),
        "graphite_1": (18, 20, 24),
        "graphite_2": (24, 27, 31),
        "graphite_3": (33, 37, 43),
        "graphite_4": (47, 52, 60),
        "graphite_5": (63, 70, 80),
        "mist_1": (90, 104, 98),
        "mist_2": (126, 140, 132),
        "mist_3": (168, 180, 172),
        "bone": (214, 221, 216),
        "paper": (232, 237, 233),
    },
    # [BEACON] Neo-Swiss AR HUD register — dark navy substrate (pure-void #0A0E14),
    # hot vermillion (#F70D1A) data containers, bone text, golden amber (#FFD235)
    # for auxiliary/context text. Matches the rendered output of
    # UI_Guide/Neo_swiss_ui_guide_html_template.html beacon colorway.
    "beacon": {
        "bg_hex": "#0A0E14",
        "accent_dark": (247, 13, 26),
        "panel": (34, 41, 51),
        "hover": (46, 54, 64),
        "pressed": (63, 70, 81),
        "input_fill": (22, 27, 35),
        "input_list": (34, 41, 51),
        "strip_soft": (247, 13, 26),
        "privacy_warn": (255, 210, 53),
        "roi": (247, 13, 26),
        "pure_void": (10, 14, 20),
        "void": (22, 27, 35),
        "graphite_1": (34, 41, 51),
        "graphite_2": (46, 54, 64),
        "graphite_3": (63, 70, 81),
        "graphite_4": (90, 99, 112),
        "graphite_5": (107, 116, 128),
        "mist_1": (124, 133, 147),
        "mist_2": (142, 150, 162),
        "mist_3": (164, 172, 184),
        "bone": (216, 220, 226),
        "paper": (232, 237, 233),
        "vermillion": (247, 13, 26),
        "cyan": (10, 143, 168),
        "radiation": (45, 154, 64),
        "oak": (92, 72, 48),
        "clay": (160, 104, 64),
        "botanical": (63, 122, 82),
    },
}

_CURRENT_SCHEME_NAME = "default"
_FIRE_BLUE_ROLES = {"panel", "hover", "pressed", "input_fill", "input_list"}


def current_color_scheme() -> str:
    return _CURRENT_SCHEME_NAME


def is_aurora_family_scheme(name: object | None = None) -> bool:
    scheme = _CURRENT_SCHEME_NAME if name is None else str(name or "").strip().lower()
    return scheme in {"aurora", "dark mode"}


def theme_rgba(role: str, alpha: float) -> str:
    key = str(role or "").strip().lower()
    a = max(0.0, min(1.0, alpha))
    if _CURRENT_SCHEME_NAME == "fire" and key in _FIRE_BLUE_ROLES:
        # Fire blue surfaces should stay airy/frosted instead of opaque blocks.
        a *= 0.62
    r, g, b = _scheme_rgb(key)
    return f"rgba({r}, {g}, {b}, {_css_number(a)})"


def theme_hex(role: str) -> str:
    r, g, b = _scheme_rgb(role)
    return QColor(r, g, b).name()


def contrast_text_hex(role: str) -> str:
    r, g, b = _scheme_rgb(role)
    # Choose a readable foreground for the current accent or surface color.
    luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return "#101114" if luminance > 0.60 else "#fff8f1"


def _shade(rgb: tuple[int, int, int], delta: int) -> tuple[int, int, int]:
    r, g, b = rgb
    return (
        max(0, min(255, r + delta)),
        max(0, min(255, g + delta)),
        max(0, min(255, b + delta)),
    )


def theme_metallic(role: str, alpha: float = 1.0, direction: str = "vertical") -> str:
    """Return a Qt qlineargradient string for `role`, faking a brushed-metal sheen
    by ramping a darker shadow through the base midtone to a lighter specular
    highlight. Works in any QSS property that accepts a brush (background, etc).
    """
    a = _css_number(max(0.0, min(1.0, alpha)))
    base = _scheme_rgb(role)
    dark = _shade(base, -38)
    mid_dark = _shade(base, -12)
    highlight = _shade(base, 42)
    mid_light = _shade(base, 12)
    if direction == "horizontal":
        stops_xy = "x1:0, y1:0, x2:1, y2:0"
    elif direction == "diagonal":
        stops_xy = "x1:0, y1:0, x2:1, y2:1"
    else:
        stops_xy = "x1:0, y1:0, x2:0, y2:1"

    def rgba(rgb: tuple[int, int, int]) -> str:
        return f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {a})"

    return (
        f"qlineargradient({stops_xy},"
        f" stop:0 {rgba(dark)},"
        f" stop:0.35 {rgba(mid_dark)},"
        f" stop:0.5 {rgba(highlight)},"
        f" stop:0.65 {rgba(mid_light)},"
        f" stop:1 {rgba(dark)})"
    )


def theme_holographic_fire(alpha: float = 1.0, state: str = "idle") -> str:
    """Fire-only holographic orange gradient for premium CTA surfaces."""
    a = max(0.0, min(1.0, float(alpha)))
    mode = str(state or "idle").strip().lower()
    if mode == "hover":
        warm_boost = 1.14
        glow_mix = 0.30
    elif mode == "pressed":
        warm_boost = 0.88
        glow_mix = 0.20
    else:
        warm_boost = 1.00
        glow_mix = 0.24

    def stop(r: int, g: int, b: int, mix: float = 1.0) -> str:
        return f"rgba({r}, {g}, {b}, {_css_number(max(0.0, min(1.0, a * mix)))})"

    return (
        "qlineargradient("
        "x1:0, y1:0, x2:1, y2:1,"
        f" stop:0.00 {stop(255, 126, 24, 0.94 * warm_boost)},"
        f" stop:0.18 {stop(255, 168, 48, 0.94 * warm_boost)},"
        f" stop:0.38 {stop(255, 102, 14, 0.96 * warm_boost)},"
        f" stop:0.58 {stop(255, 186, 76, 0.92 * warm_boost)},"
        f" stop:0.72 {stop(255, 206, 110, glow_mix)},"
        f" stop:0.84 {stop(255, 146, 36, 0.94 * warm_boost)},"
        f" stop:1.00 {stop(255, 116, 18, 0.94 * warm_boost)})"
    )


def _scheme_rgb(role: str) -> tuple[int, int, int]:
    spec = _SCHEMES.get(_CURRENT_SCHEME_NAME) or _SCHEMES["solarized_dark"]
    raw = spec.get(role)
    if isinstance(raw, tuple) and len(raw) == 3:
        return int(raw[0]), int(raw[1]), int(raw[2])
    return 42, 161, 152


def _apply_scheme_to_globals() -> None:
    global HUD_BG, PANEL, RED_IRON, RED_SIGNAL, RED_CAUTION, HUD_STRIP_BG_CSS, HUD_STRIP_BORDER_CSS
    spec = _SCHEMES.get(_CURRENT_SCHEME_NAME) or _SCHEMES["solarized_dark"]
    bg_hex = str(spec.get("bg_hex") or "#002b36")
    HUD_BG.setNamedColor(bg_hex)
    pr, pg, pb = _scheme_rgb("panel")
    PANEL.setRgb(pr, pg, pb, 212)
    ar, ag, ab = _scheme_rgb("accent_dark")
    RED_IRON.setRgb(ar, ag, ab)
    RED_SIGNAL.setRgb(pr, pg, pb)
    RED_CAUTION.setRgb(pr, pg, pb)
    sr, sg, sb = _scheme_rgb("strip_soft")
    HUD_STRIP_BG_CSS = f"rgba({sr}, {sg}, {sb}, 0.30)"
    HUD_STRIP_BORDER_CSS = theme_rgba("accent_dark", 0.14)


HUD_BG = QColor(10, 10, 10)
PANEL = QColor(22, 22, 22, 220)
RED_IRON = QColor(220, 38, 38)
RED_SIGNAL = QColor(220, 38, 38)
RED_CAUTION = QColor(220, 38, 38)
HUD_YELLOW = QColor(255, 208, 96)
HUD_TEXT = QColor(232, 232, 232)
HUD_MUTED = QColor(232, 232, 232, 150)
HUD_STRIP_BG_CSS = "rgba(220,38,38,0.14)"
HUD_STRIP_BORDER_CSS = "rgba(220,38,38,0.28)"

_TEXT_COLOR_MODES: dict[str, tuple[int, int, int]] = {
    "black": (20, 8, 8),
    "bright-cyan": (0, 255, 255),
    "lime": (190, 242, 100),
}
_CURRENT_TEXT_MODE = "black"
_TEXT_RGBA_RE = re.compile(r"rgba\(\s*20\s*,\s*8\s*,\s*8\s*,\s*([0-9.]+)\s*\)", re.IGNORECASE)
_TEXT_HEX_RE = re.compile(r"#140808", re.IGNORECASE)
_ALT_TEXT_HEX_RE = re.compile(r"#ffd0d0|#ffe0e0", re.IGNORECASE)
_SOLAR_GREEN_RGBA_RE = re.compile(r"rgba\(\s*133\s*,\s*153\s*,\s*0\s*,\s*([0-9.]+)\s*\)", re.IGNORECASE)
_SOLAR_CYAN_RGBA_RE = re.compile(r"rgba\(\s*42\s*,\s*161\s*,\s*152\s*,\s*([0-9.]+)\s*\)", re.IGNORECASE)
_SOLAR_RED_RGBA_RE = re.compile(r"rgba\(\s*220\s*,\s*50\s*,\s*47\s*,\s*([0-9.]+)\s*\)", re.IGNORECASE)
_SOLAR_GREEN_HEX_RE = re.compile(r"#859900", re.IGNORECASE)
_SOLAR_CYAN_HEX_RE = re.compile(r"#2aa198", re.IGNORECASE)
_SOLAR_RED_HEX_RE = re.compile(r"#dc322f", re.IGNORECASE)
_TEXT_FILTER: QObject | None = None


def _css_number(value: float) -> str:
    text = f"{value:.2f}"
    return text.rstrip("0").rstrip(".")


def _current_text_rgb() -> tuple[int, int, int]:
    if _CURRENT_SCHEME_NAME == "default" and _CURRENT_TEXT_MODE == "black":
        return 232, 232, 232
    if _CURRENT_SCHEME_NAME in ("marathon", "wear_marathon") and _CURRENT_TEXT_MODE == "black":
        return 245, 247, 242
    if _CURRENT_SCHEME_NAME == "fire" and _CURRENT_TEXT_MODE == "black":
        return 242, 231, 207
    if is_aurora_family_scheme() and _CURRENT_TEXT_MODE == "black":
        return 232, 237, 233
    if _CURRENT_SCHEME_NAME == "beacon" and _CURRENT_TEXT_MODE == "black":
        # Dark navy substrate — primary text is bone (D8DCE2), matching the
        # rendered output of the beacon colorway in the Neo-Swiss guide.
        return 216, 220, 226
    return _TEXT_COLOR_MODES[_CURRENT_TEXT_MODE]


def text_qcolor(alpha: float = 1.0) -> QColor:
    red, green, blue = _current_text_rgb()
    color = QColor(red, green, blue)
    color.setAlphaF(max(0.0, min(1.0, alpha)))
    return color


def text_hex() -> str:
    return text_qcolor().name()


def text_css(alpha: float = 1.0) -> str:
    alpha = max(0.0, min(1.0, alpha))
    if alpha >= 1.0:
        return text_hex()
    red, green, blue = _current_text_rgb()
    return f"rgba({red}, {green}, {blue}, {_css_number(alpha)})"


def surface_muted_css(alpha: float = 0.65) -> str:
    if _CURRENT_SCHEME_NAME == "beacon":
        r, g, b = _scheme_rgb("privacy_warn")
        a = max(0.0, min(1.0, alpha))
        if a >= 1.0:
            return QColor(r, g, b).name()
        return f"rgba({r}, {g}, {b}, {_css_number(a)})"
    return text_css(alpha)


# Fixed vermillion / paper for the "beacon" title-tag look. Hardcoded so the
# red+white title chip stays consistent even when the active scheme's
# accent_dark is not red (e.g. aurora's grey-green accent).
TITLE_BEACON_RED = "#DC322F"
TITLE_BEACON_TEXT = "#FFFFFF"


def beacon_title_tag_css(*, font_size: int = 10, padding: str = "2px 6px") -> str:
    # Sans face (Inter / system) renders far cleaner than monospace at 10 px;
    # tighter padding and a smaller letter-spacing keep strokes on whole-pixel
    # boundaries so the chip stops looking aliased.
    return (
        f"background: {TITLE_BEACON_RED}; "
        f"color: {TITLE_BEACON_TEXT}; "
        f"border: 1px solid {TITLE_BEACON_RED}; "
        f"padding: {padding}; "
        'font-family: "Inter", "Söhne", -apple-system, "Segoe UI", system-ui, sans-serif; '
        f"font-size: {int(font_size)}px; "
        "font-weight: 600; "
        "letter-spacing: 0.02em; "
        "text-transform: uppercase; "
        "border-radius: 0px;"
    )


def configure_text_mode(mode: object) -> str:
    global _CURRENT_TEXT_MODE, GLOBAL_STYLESHEET
    requested = str(mode or "").strip().lower()
    if requested not in _TEXT_COLOR_MODES:
        requested = "black"
    _CURRENT_TEXT_MODE = requested
    red, green, blue = _current_text_rgb()
    HUD_TEXT.setRgb(red, green, blue)
    HUD_MUTED.setRgb(red, green, blue, 150)
    GLOBAL_STYLESHEET = _build_global_stylesheet()
    return _CURRENT_TEXT_MODE


def configure_color_scheme(name: object) -> str:
    global _CURRENT_SCHEME_NAME
    requested = str(name or "").strip().lower()
    if requested not in _SCHEMES:
        requested = "default"
    _CURRENT_SCHEME_NAME = requested
    _apply_scheme_to_globals()
    configure_text_mode(_CURRENT_TEXT_MODE)
    return _CURRENT_SCHEME_NAME


def _soft_alpha(value: float, *, scale: float = 0.72, floor: float = 0.14, ceil: float = 0.70) -> float:
    scaled = float(value) * scale
    return max(floor, min(ceil, scaled))


def themed_css(value: str) -> str:
    if not value:
        return value
    value = _TEXT_HEX_RE.sub(text_hex(), value)
    value = _TEXT_RGBA_RE.sub(lambda match: text_css(float(match.group(1))), value)
    value = _SOLAR_GREEN_RGBA_RE.sub(
        lambda match: theme_rgba("strip_soft", _soft_alpha(float(match.group(1)))),
        value,
    )
    value = _SOLAR_CYAN_RGBA_RE.sub(
        lambda match: theme_rgba("accent_dark", _soft_alpha(float(match.group(1)))),
        value,
    )
    value = _SOLAR_RED_RGBA_RE.sub(
        lambda match: theme_rgba("privacy_warn", _soft_alpha(float(match.group(1)), floor=0.18, ceil=0.78)),
        value,
    )
    value = _SOLAR_GREEN_HEX_RE.sub(theme_hex("strip_soft"), value)
    value = _SOLAR_CYAN_HEX_RE.sub(theme_hex("accent_dark"), value)
    value = _SOLAR_RED_HEX_RE.sub(theme_hex("privacy_warn"), value)
    return _ALT_TEXT_HEX_RE.sub(text_hex(), value)


def apply_text_palette(widget: QWidget | None) -> None:
    if not _widget_is_alive(widget):
        return
    _apply_widget_text_palette(widget)
    try:
        children = widget.findChildren(QWidget)
    except RuntimeError:
        return
    for child in children:
        _apply_widget_text_palette(child)


class _TextPaletteFilter(QObject):
    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.ChildAdded:
            child = event.child()
            if isinstance(child, QWidget):
                QTimer.singleShot(0, lambda target=child: apply_text_palette(target))
        return False


def install_text_palette_filter(app: QApplication) -> None:
    global _TEXT_FILTER
    if _TEXT_FILTER is None:
        _TEXT_FILTER = _TextPaletteFilter(app)
    app.installEventFilter(_TEXT_FILTER)


def detection_label_text(value: object) -> str:
    return str(value or "").upper()


def _apply_widget_text_palette(widget: QWidget) -> None:
    if not _widget_is_alive(widget):
        return
    # Beacon-style title chip (red bg + white text) is applied in every scheme.
    if bool(widget.property("isTitle")):
        if isinstance(widget, (QLabel, QToolButton)):
            try:
                widget.setStyleSheet(beacon_title_tag_css())
                # QLabel's default `indent` plus stylesheet padding cause Qt's
                # sizeHint to under-report by the indent amount, clipping the
                # last few pixels of long titles like "SCENARIOS". Killing the
                # indent and re-running adjustSize() forces the chip to grow to
                # exactly text-width + padding so nothing gets cut off.
                if isinstance(widget, QLabel):
                    try:
                        widget.setIndent(0)
                        widget.setMargin(0)
                    except Exception:
                        pass
                widget.adjustSize()
                widget.updateGeometry()
            except RuntimeError:
                return
            return
    try:
        style = widget.styleSheet()
    except RuntimeError:
        return
    if style:
        themed = themed_css(style)
        if themed != style:
            try:
                widget.setStyleSheet(themed)
            except RuntimeError:
                return
    if isinstance(widget, QLabel):
        try:
            text = widget.text()
        except RuntimeError:
            return
        if text and ("#140808" in text.lower() or "#ffd0d0" in text.lower() or "rgba(20" in text.lower()):
            themed_text = themed_css(text)
            if themed_text != text:
                try:
                    widget.setText(themed_text)
                except RuntimeError:
                    return


def _widget_is_alive(widget: QWidget | None) -> bool:
    if widget is None:
        return False
    try:
        widget.objectName()
    except RuntimeError:
        return False
    return True


def _material_font_stack() -> str:
    return (
        '"Roboto", "Google Sans", "Segoe UI", -apple-system, '
        '"Helvetica Neue", Arial, sans-serif'
    )


def _build_material_stylesheet() -> str:
    """Material Design 3–style sheet (flat surfaces, large radii, sans typography)."""
    dark = _CURRENT_SCHEME_NAME == "material_dark"
    if dark:
        bg = "#131314"
        surf = "rgba(41, 42, 45, 0.98)"
        surf_soft = "rgba(41, 42, 45, 0.88)"
        surf_muted = "rgba(60, 64, 67, 0.95)"
        text_primary = "#E3E3E3"
        text_soft = "rgba(227, 227, 227, 0.82)"
        text_muted = "rgba(227, 227, 227, 0.58)"
        outline = "rgba(154, 160, 166, 0.38)"
        outline_strong = "rgba(154, 160, 166, 0.55)"
    else:
        bg = "#F8F9FA"
        surf = "rgba(255, 255, 255, 0.98)"
        surf_soft = "rgba(255, 255, 255, 0.92)"
        surf_muted = "rgba(248, 249, 250, 0.98)"
        text_primary = "#202124"
        text_soft = "rgba(32, 33, 36, 0.78)"
        text_muted = "rgba(95, 99, 104, 0.95)"
        outline = "rgba(60, 64, 67, 0.16)"
        outline_strong = "rgba(60, 64, 67, 0.28)"

    primary = theme_hex("accent_dark")
    primary_container = theme_rgba("accent_dark", 0.22 if dark else 0.14)
    primary_hover = theme_rgba("accent_dark", 0.42 if dark else 0.88)
    on_primary = "#ffffff" if not dark else "#202124"
    err = theme_hex("privacy_warn")
    err_bg = theme_rgba("privacy_warn", 0.18)
    ff = _material_font_stack()
    # Rounded only on buttons, pickers/lists/tables/tabs/checks, and title chips; everything else square.
    r_round = "12px"
    r_pill = "9999px"

    return f"""
QMainWindow, QWidget {{
    background: {bg};
    color: {text_primary};
    font-family: {ff};
    font-size: 13px;
    font-weight: 400;
}}

QFrame {{
    border: 1px solid {outline};
    background: {surf_muted};
    border-radius: 0px;
}}

QLabel {{ background: transparent; border-radius: 0px; }}
QLabel[isTitle="true"] {{
    color: {TITLE_BEACON_TEXT};
    font-weight: 600;
    background: {TITLE_BEACON_RED};
    border: 1px solid {TITLE_BEACON_RED};
    padding: 4px 10px;
    border-radius: {r_round};
}}
QToolButton[isTitle="true"] {{
    color: {TITLE_BEACON_TEXT};
    font-weight: 600;
    background: {TITLE_BEACON_RED};
    border: 1px solid {TITLE_BEACON_RED};
    padding: 4px 10px;
    border-radius: {r_round};
}}

QPushButton {{
    border: 1px solid transparent;
    background: {primary};
    color: {on_primary};
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
    border-radius: {r_pill};
}}
QPushButton:hover {{
    background: {primary_hover};
    border-color: transparent;
}}
QPushButton:pressed {{
    background: {theme_hex("pressed")};
    color: {on_primary};
}}
QPushButton:checked {{
    background: {primary_container};
    color: {primary};
    border: 1px solid {outline_strong};
}}
QPushButton:disabled {{
    color: {text_muted};
    border-color: {outline};
    background: {surf_soft};
}}

QToolButton {{
    border: 1px solid {outline};
    background: {surf};
    color: {text_soft};
    padding: 6px 12px;
    font-size: 12px;
    text-align: left;
    border-radius: {r_round};
}}
QToolButton:hover {{ background: {primary_container}; color: {text_primary}; border-color: {outline_strong}; }}
QToolButton:checked {{ background: {primary_container}; color: {primary}; border-color: {primary}; }}

QTabWidget::pane {{
    border: none;
    border-top: 1px solid {outline};
    background: {surf};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {text_muted};
    padding: 10px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    border-top-left-radius: {r_round};
    border-top-right-radius: {r_round};
    min-width: 64px;
    font-size: 13px;
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {primary_container};
    color: {primary};
    border-bottom: 2px solid {primary};
}}
QTabBar::tab:hover:!selected {{
    color: {text_primary};
    background: {primary_container};
}}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ width: 8px; background: transparent; margin: 4px 0; }}
QScrollBar::handle:vertical {{
    background: {outline_strong};
    min-height: 36px;
    border-radius: 0px;
    margin: 2px;
}}
QScrollBar::handle:vertical:hover {{ background: {primary}; }}
QScrollBar:horizontal {{ height: 8px; background: transparent; margin: 0 4px; }}
QScrollBar::handle:horizontal {{
    background: {outline_strong};
    min-width: 36px;
    border-radius: 0px;
    margin: 2px;
}}
QScrollBar::handle:horizontal:hover {{ background: {primary}; }}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    border: 1px solid {outline_strong};
    background: {surf};
    color: {text_primary};
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {primary_container};
    selection-color: {text_primary};
    border-radius: 0px;
}}
QComboBox {{
    border: 1px solid {outline_strong};
    background: {surf};
    color: {text_primary};
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {primary_container};
    selection-color: {text_primary};
    border-radius: {r_round};
}}
QComboBox::drop-down {{
    border: none;
    width: 28px;
}}
QComboBox QAbstractItemView {{
    background: {surf};
    color: {text_primary};
    border: 1px solid {outline};
    selection-background-color: {primary_container};
    selection-color: {text_primary};
    border-radius: {r_round};
}}

QListWidget {{
    border: 1px solid {outline};
    background: {surf};
    color: {text_primary};
    border-radius: {r_round};
}}
QListWidget::item:selected {{
    background: {primary_container};
    color: {text_primary};
}}

QProgressBar {{
    border: 1px solid {outline};
    background: {surf_soft};
    color: {text_muted};
    text-align: center;
    border-radius: 0px;
    min-height: 8px;
}}
QProgressBar::chunk {{
    background: {primary};
    border-radius: 0px;
}}

QCheckBox {{
    color: {text_soft};
    font-size: 13px;
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border: 2px solid {outline_strong};
    background: {surf};
    border-radius: {r_round};
}}
QCheckBox::indicator:checked {{
    background: {primary};
    border-color: {primary};
}}

QTableWidget {{
    background: {surf};
    color: {text_soft};
    gridline-color: {outline};
    selection-background-color: {primary_container};
    selection-color: {text_primary};
    border: 1px solid {outline};
    border-radius: {r_round};
}}
QHeaderView::section {{
    background: {surf_muted};
    color: {text_muted};
    border: none;
    border-bottom: 1px solid {outline};
    padding: 8px 10px;
    font-size: 12px;
    font-weight: 600;
}}

QLabel#criticalAlert {{
    background: {err_bg};
    border: 1px solid {theme_rgba("privacy_warn", 0.45)};
    color: {text_primary};
    border-radius: 0px;
    padding: 8px;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {outline};
    border-radius: 0px;
}}
QSlider::handle:horizontal {{
    width: 18px;
    height: 18px;
    margin: -7px 0;
    background: {primary};
    border-radius: 0px;
    border: 2px solid {surf};
}}
"""


def _build_solarized_dense_stylesheet() -> str:
    """Solarized Dark main UI: rounded, radial, dense-ops styling."""
    mode = str(os.environ.get("INSIGHT_UI_DENSITY", "operator")).strip().lower()
    if mode not in {"operator", "scout"}:
        mode = "operator"

    bg = (
        "qradialgradient(cx:0.5, cy:0.42, radius:1.18, fx:0.5, fy:0.42, "
        "stop:0 rgba(0, 59, 71, 0.94), stop:0.45 rgba(0, 43, 54, 0.97), stop:1 rgba(1, 28, 37, 0.99))"
    )
    surface = "rgba(7, 54, 66, 0.72)"
    surface_soft = "rgba(7, 54, 66, 0.58)"
    panel_orbit = (
        "qradialgradient(cx:0.5, cy:0.45, radius:0.98, fx:0.5, fy:0.45, "
        "stop:0 rgba(7, 54, 66, 0.84), stop:0.6 rgba(7, 54, 66, 0.64), stop:1 rgba(7, 54, 66, 0.26))"
    )
    glass = "rgba(7, 54, 66, 0.38)"
    glass_hover = "rgba(42, 161, 152, 0.14)"
    glass_focus = "rgba(42, 161, 152, 0.24)"
    text_primary = "#a7b6b6"
    text_soft = "rgba(167, 182, 182, 0.80)"
    text_muted = "rgba(167, 182, 182, 0.58)"
    accent = "#2aa198"
    accent_hover = "rgba(42, 161, 152, 0.90)"
    accent_pressed = "#1f7d76"
    accent_quiet = "rgba(42, 161, 152, 0.08)"
    accent_active = "rgba(42, 161, 152, 0.16)"
    ok = "#859900"
    warn = "#dc322f"
    border_idle = "rgba(167, 182, 182, 0.12)"
    border_soft = "rgba(167, 182, 182, 0.08)"
    border_glass = "rgba(167, 182, 182, 0.18)"
    border_ring = "rgba(42, 161, 152, 0.34)"

    if mode == "scout":
        font_ui = "13px"
        font_caps = "11px"
        pad_btn = "7px 16px"
        pad_in = "6px 10px"
    else:
        font_ui = "11px"
        font_caps = "10px"
        pad_btn = "5px 12px"
        pad_in = "4px 8px"

    return f"""
QMainWindow, QWidget {{
    color: {text_primary};
    font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", "Roboto", sans-serif;
    font-size: {font_ui};
    letter-spacing: 0.2px;
}}

QMainWindow#insightMainWindow {{
    background: {bg};
}}
QWidget#centralHud {{
    background: transparent;
}}

QFrame {{
    border: 1px solid {border_soft};
    background: {glass};
    border-radius: 14px;
}}

QPushButton {{
    border: 1px solid {accent_hover};
    background: {accent_quiet};
    color: {text_primary};
    padding: {pad_btn};
    min-height: 18px;
    font-size: {font_ui};
    font-weight: 600;
    border-radius: 14px;
}}
QPushButton:hover {{
    background: {glass_hover};
    border-color: {accent};
}}
QPushButton:pressed {{
    background: {accent_active};
    border-color: {accent_pressed};
}}
QPushButton:checked {{
    background: {accent_active};
    border-color: {accent};
    color: {text_primary};
}}
QPushButton:disabled {{
    background: transparent;
    border-color: {border_soft};
    color: {text_muted};
}}

QToolButton {{
    border: 1px solid transparent;
    background: transparent;
    color: {text_soft};
    padding: 5px 9px;
    border-radius: 12px;
}}
QToolButton:hover {{
    background: {glass_hover};
    border-color: {border_glass};
    color: {text_primary};
}}
QToolButton:checked {{
    background: {accent_active};
    border-color: {accent};
    color: {accent};
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    border: 1px solid {border_glass};
    background: {glass};
    color: {text_primary};
    padding: {pad_in};
    selection-background-color: {accent_active};
    selection-color: {text_primary};
    border-radius: 12px;
}}
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    background: {glass_hover};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    background: {glass_focus};
    border: 1px solid {accent};
}}

QComboBox {{
    border: 1px solid {border_glass};
    background: {glass};
    color: {text_primary};
    padding: {pad_in};
    min-height: 18px;
    border-radius: 12px;
}}
QComboBox:hover {{ background: {glass_hover}; }}
QComboBox:focus {{ background: {glass_focus}; border: 1px solid {accent}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background: {surface};
    color: {text_primary};
    border: 1px solid {border_glass};
    border-radius: 12px;
    selection-background-color: {accent_active};
    selection-color: {text_primary};
}}

QTabWidget#orbitalTabs::pane, QTabWidget#catalogOrbitalTabs::pane {{
    border: 1px solid {border_ring};
    border-radius: 16px;
    background: {panel_orbit};
    top: -1px;
}}
QTabWidget#orbitalTabs QTabBar::tab, QTabWidget#catalogOrbitalTabs QTabBar::tab {{
    border: 1px solid transparent;
    background: transparent;
    color: {text_muted};
    padding: 6px 12px;
    margin-right: 4px;
    border-radius: 13px;
    font-size: {font_ui};
    font-weight: 600;
}}
QTabWidget#orbitalTabs QTabBar::tab:hover, QTabWidget#catalogOrbitalTabs QTabBar::tab:hover {{
    color: {text_primary};
    background: {accent_quiet};
    border-color: {border_glass};
}}
QTabWidget#orbitalTabs QTabBar::tab:selected, QTabWidget#catalogOrbitalTabs QTabBar::tab:selected {{
    color: {text_primary};
    background: {glass_hover};
    border-color: {border_ring};
}}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ width: 6px; background: transparent; margin: 2px 0; }}
QScrollBar::handle:vertical {{ background: {border_idle}; min-height: 24px; border-radius: 3px; }}
QScrollBar::handle:vertical:hover {{ background: {accent_active}; }}
QScrollBar:horizontal {{ height: 6px; background: transparent; margin: 0 2px; }}
QScrollBar::handle:horizontal {{ background: {border_idle}; min-width: 24px; border-radius: 3px; }}
QScrollBar::handle:horizontal:hover {{ background: {accent_active}; }}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}

QListWidget {{
    border: 1px solid {border_glass};
    background: {glass};
    color: {text_primary};
    border-radius: 12px;
}}
QListWidget::item {{ padding: 5px 8px; }}
QListWidget::item:hover {{ background: {glass_hover}; }}
QListWidget::item:selected {{ background: {accent_active}; color: {text_primary}; }}

QTableWidget {{
    background: transparent;
    color: {text_primary};
    gridline-color: {border_soft};
    selection-background-color: {accent_active};
    selection-color: {text_primary};
    border: 1px solid {border_idle};
    border-radius: 12px;
}}
QHeaderView::section {{
    background: transparent;
    color: {text_muted};
    border: none;
    border-bottom: 1px solid {border_idle};
    padding: 6px 8px;
    font-size: {font_caps};
    font-weight: 600;
    letter-spacing: 0.4px;
}}

QProgressBar {{
    border: none;
    background: {border_soft};
    color: {text_muted};
    border-radius: 4px;
    min-height: 7px;
    max-height: 7px;
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 4px;
}}

QLabel[isTitle="true"], QToolButton[isTitle="true"] {{
    color: {text_muted};
    background: transparent;
    padding: 4px 0;
    font-size: {font_caps};
    font-weight: 600;
    letter-spacing: 0.55px;
}}

QLabel#criticalAlert {{
    background: rgba(220, 50, 47, 0.12);
    border: 1px solid {warn};
    color: {text_primary};
    padding: 6px 10px;
    border-radius: 12px;
}}

QLabel[signal="clear"] {{ color: {ok}; }}
QLabel[signal="warning"] {{ color: rgba(181, 137, 0, 0.92); }}
QLabel[signal="error"] {{ color: {warn}; }}
"""


def _build_global_stylesheet() -> str:
    return _build_wear_stylesheet()


def _build_wear_stylesheet() -> str:
    bg_hex = str((_SCHEMES.get(_CURRENT_SCHEME_NAME) or _SCHEMES["default"]).get("bg_hex", "#070707"))
    bg = QColor(bg_hex)
    marathon_like = _CURRENT_SCHEME_NAME in ("marathon", "wear_marathon")
    # [BEACON] Flag raised before all other vars so every token can branch on it.
    beacon_like = _CURRENT_SCHEME_NAME == "beacon"
    accent = theme_hex("accent_dark")
    if beacon_like:
        # Beacon: solid graphite-1 panels on dark navy page.
        panel = theme_hex("panel")
        panel_soft = theme_rgba("panel", 0.96)
    else:
        panel = theme_rgba("panel", 0.88 if marathon_like else 0.94)
        panel_soft = theme_rgba("panel", 0.72 if marathon_like else 0.82)
    panel_raised = theme_rgba("hover", 0.90)
    glass = theme_rgba("input_fill", 0.86)
    if beacon_like:
        # Beacon hairlines use graphite-4/3 (#5A6370 / #3F4651), not vermillion.
        _g4 = _scheme_rgb("graphite_4")
        _g3 = _scheme_rgb("graphite_3")
        _g2 = _scheme_rgb("graphite_2")
        border = f"rgba({_g4[0]},{_g4[1]},{_g4[2]},0.65)"
        border_soft = f"rgba({_g3[0]},{_g3[1]},{_g3[2]},0.45)"
        # Beacon chips use graphite-2/4 fills, not faint vermillion.
        chip = f"rgba({_g2[0]},{_g2[1]},{_g2[2]},0.85)"
        chip_active = f"rgba({_g4[0]},{_g4[1]},{_g4[2]},0.85)"
    elif _CURRENT_SCHEME_NAME == "dark mode":
        border = "rgba(0, 43, 54, 0.28)"
        border_soft = "rgba(0, 43, 54, 0.16)"
        chip = theme_rgba("accent_dark", 0.16)
        chip_active = theme_rgba("accent_dark", 0.28)
    else:
        border = theme_rgba("accent_dark", 0.28)
        border_soft = theme_rgba("accent_dark", 0.16)
        chip = theme_rgba("accent_dark", 0.16)
        chip_active = theme_rgba("accent_dark", 0.28)
    text_primary = text_hex()
    if beacon_like:
        # Beacon secondary text: mist-2 grey for inactive elements (not amber).
        _m2 = _scheme_rgb("mist_2")
        text_soft = f"rgba({_m2[0]},{_m2[1]},{_m2[2]},0.92)"
        # Amber (#FFD235) reserved for aux/contextual labels — the Beacon identity.
        text_muted = surface_muted_css(0.88)
    else:
        text_soft = text_css(0.78)
        text_muted = text_css(0.56)
    on_accent = contrast_text_hex("accent_dark")
    warn = theme_hex("privacy_warn")
    # Titles are always the beacon-style chip (red bg + white text) regardless
    # of the active scheme — pinned to the vermillion constant so aurora's
    # grey-green accent doesn't leak in.
    title_bg = TITLE_BEACON_RED
    title_color = TITLE_BEACON_TEXT
    title_border = f"1px solid {TITLE_BEACON_RED}"
    title_padding = "2px 6px"
    beacon_title_mono = (
        'font-family: "Inter", "Söhne", -apple-system, "Segoe UI", system-ui, sans-serif;'
    )
    font_family = (
        '"Inter", "Söhne", -apple-system, "Segoe UI", system-ui, sans-serif'
        if beacon_like
        else '"Roboto", "Google Sans", "Segoe UI", "Helvetica Neue", Arial, sans-serif'
    )
    if beacon_like:
        # Beacon substrate is flat dark navy — no radial gradient.
        root_bg = bg_hex
    else:
        root_bg = (
            f"qradialgradient(cx:0.5, cy:0.28, radius:1.25, fx:0.5, fy:0.15, "
            f"stop:0 {theme_rgba('hover', 0.22)}, stop:0.45 rgba({bg.red()}, {bg.green()}, {bg.blue()}, 0.96), "
            f"stop:1 rgba({bg.red()}, {bg.green()}, {bg.blue()}, 1.0))"
        )
    return f"""
QMainWindow, QWidget {{
    background: {root_bg};
    color: {text_primary};
    font-family: {font_family};
    font-size: 13px;
    font-weight: 400;
}}

QWidget#centralHud {{
    background: transparent;
}}

QFrame, QWidget#wearCard, QWidget#previewsHost, QWidget#focusRoot, QWidget#focusBody,
QWidget#focusScanHost, QWidget#eventsHost, QWidget#swapBanner {{
    background: {panel};
    border: 1px solid {border_soft};
    border-radius: 0px;
}}

QWidget#focusBody, QWidget#focusScanHost {{
    background: {panel_soft};
}}

QLabel {{
    background: transparent;
    color: {text_primary};
    border: none;
}}

QLabel[isTitle="true"], QToolButton[isTitle="true"] {{
    color: {title_color};
    background: {title_bg};
    border: {title_border};
    padding: {title_padding};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
    {beacon_title_mono}
}}

QPushButton {{
    border: 1px solid transparent;
    background: {accent};
    color: {on_accent};
    padding: 8px 16px;
    min-height: 24px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.6px;
    border-radius: 0px;
}}

QPushButton:hover {{
    background: {theme_rgba('accent_dark', 0.88)};
}}

QPushButton:pressed {{
    background: {theme_rgba('pressed', 0.95)};
}}

QPushButton:checked {{
    background: {chip_active};
    color: {text_primary};
    border: 1px solid {border};
}}

QPushButton:disabled {{
    background: {panel_soft};
    color: {text_muted};
    border: 1px solid {border_soft};
}}

QPushButton[variant="ghost"], QToolButton {{
    background: {chip};
    color: {text_soft};
    border: 1px solid {border_soft};
    border-radius: 0px;
    padding: 6px 12px;
    letter-spacing: 0.4px;
}}

QPushButton[variant="ghost"]:hover, QToolButton:hover {{
    background: {chip_active};
    color: {text_primary};
    border-color: {border};
}}

QPushButton[variant="ghost"]:checked, QToolButton:checked {{
    background: {chip_active};
    color: {text_primary};
    border-color: {accent};
}}

QTabWidget::pane {{
    border: none;
    background: transparent;
    top: -1px;
}}

QTabBar::tab {{
    background: {chip};
    color: {accent};
    padding: 6px 14px;
    border: 1px dotted {border_soft};
    border-bottom: 2px solid transparent;
    border-radius: 0px;
    min-width: 84px;
    margin-right: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.4px;
}}

QTabBar::tab:selected {{
    background: {chip_active};
    color: {accent};
    border: 1px dotted {border};
    border-bottom: 2px solid {accent};
}}

QTabBar::tab:hover:!selected {{
    color: {accent};
    background: {chip_active};
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QScrollBar:vertical {{
    width: 8px;
    background: transparent;
    margin: 4px 0;
}}

QScrollBar::handle:vertical {{
    background: {border};
    min-height: 36px;
    border-radius: 0px;
}}

QScrollBar:horizontal {{
    height: 8px;
    background: transparent;
    margin: 0 4px;
}}

QScrollBar::handle:horizontal {{
    background: {border};
    min-width: 36px;
    border-radius: 0px;
}}

QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {glass};
    color: {text_primary};
    border: 1px solid {border_soft};
    border-radius: 0px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {chip_active};
    selection-color: {text_primary};
}}

QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    background: {panel_raised};
    border-color: {border};
}}

QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {accent};
}}

QComboBox::drop-down {{
    border: none;
    width: 28px;
}}

QComboBox QAbstractItemView, QListWidget, QTableWidget {{
    background: {panel_raised};
    color: {text_primary};
    border: 1px solid {border};
    border-radius: 0px;
    selection-background-color: {chip_active};
    selection-color: {text_primary};
}}

QHeaderView::section {{
    background: {chip};
    color: {accent};
    border: 1px dotted {border_soft};
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.4px;
}}

QProgressBar {{
    border: none;
    background: {chip};
    border-radius: 0px;
    min-height: 6px;
    max-height: 6px;
}}

QProgressBar::chunk {{
    background: {accent};
    border-radius: 0px;
}}

QCheckBox {{
    color: {text_soft};
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {border};
    background: {glass};
    border-radius: 0px;
}}

QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
}}

QLabel#criticalAlert {{
    background: {theme_rgba('privacy_warn', 0.14)};
    border: 1px solid {warn};
    border-radius: 0px;
    padding: 8px 12px;
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {chip};
    border-radius: 0px;
}}

QSlider::handle:horizontal {{
    width: 10px;
    height: 14px;
    margin: -5px 0;
    background: {accent};
    border: 1px solid {panel};
    border-radius: 0px;
}}

QSlider::sub-page:horizontal {{
    background: {chip_active};
    border-radius: 0px;
}}
"""

    if _CURRENT_SCHEME_NAME in ("material", "material_dark"):
        return _build_material_stylesheet()
    if _CURRENT_SCHEME_NAME == "solarized_dark":
        return _build_solarized_dense_stylesheet()
    if _CURRENT_SCHEME_NAME in ("tactical", "fire"):
        is_solarized = _CURRENT_SCHEME_NAME == "solarized_dark"
        is_fire = _CURRENT_SCHEME_NAME == "fire"
        # Material-polish on solarized: surfaces whisper, primary action speaks.
        if is_fire:
            bg = "#002b36"
        else:
            bg = "#002b36" if is_solarized else "#1b1b1b"
        # Frame surface: a hint of elevation, no metal sheen.
        if is_fire:
            # [STONE MIRROR] desaturated slate-blue glass with warm fire accents.
            surface = "rgba(34, 46, 56, 0.34)"
            surface_high = "rgba(44, 58, 70, 0.44)"
        else:
            surface = "rgba(7, 54, 66, 0.55)" if is_solarized else "rgba(27, 27, 27, 0.55)"
            surface_high = "rgba(7, 54, 66, 0.85)" if is_solarized else "rgba(27, 27, 27, 0.85)"
        # Glass surface for interactables: more transparent than frames, frosted feel.
        if is_fire:
            glass = "rgba(24, 34, 42, 0.15)"
            glass_hover = "rgba(248, 164, 64, 0.12)"
            glass_focus = "rgba(248, 164, 64, 0.20)"
        else:
            glass = "rgba(147, 161, 161, 0.06)" if is_solarized else "rgba(212, 219, 224, 0.05)"
            glass_hover = "rgba(147, 161, 161, 0.10)" if is_solarized else "rgba(212, 219, 224, 0.09)"
            glass_focus = "rgba(42, 161, 152, 0.08)"
        # Text — body uses base1, supporting text fades down.
        if is_fire:
            # [MIRROR BODY] warm ivory on cool graphite — gold is reserved for
            # accent/state; body text reads like engraved text on polished metal.
            text_primary = "#f2e7cf"
            text_soft = "rgba(242, 231, 207, 0.74)"
            text_muted = "rgba(242, 231, 207, 0.50)"
        else:
            text_primary = "#93a1a1" if is_solarized else "#d4dbe0"
            text_soft = "rgba(147, 161, 161, 0.78)" if is_solarized else "rgba(212, 219, 224, 0.78)"
            text_muted = "rgba(147, 161, 161, 0.50)" if is_solarized else "rgba(212, 219, 224, 0.50)"
        # Accent — the only loud color. Reserved for primary action + focus.
        if is_fire:
            accent = "#f8a43f"
            accent_text = "#fff3dd"
            accent_hover = "#ffb05a"
            accent_pressed = "#e88f2e"
            accent_quiet = "rgba(248, 164, 64, 0.12)"
            accent_active = "rgba(248, 164, 64, 0.22)"
            btn_bg = theme_holographic_fire(0.76, "idle")
            btn_bg_hover = theme_holographic_fire(0.86, "hover")
            btn_bg_pressed = theme_holographic_fire(0.70, "pressed")
        else:
            accent = "#2aa198"
            accent_text = "#002b36" if is_solarized else "#0a0a0a"
            accent_hover = "rgba(42, 161, 152, 0.85)"
            accent_pressed = "#1f7d76"
            accent_quiet = "rgba(42, 161, 152, 0.10)"
            accent_active = "rgba(42, 161, 152, 0.20)"
            btn_bg = accent
            btn_bg_hover = accent_hover
            btn_bg_pressed = accent_pressed
        # Semantic — only appear on actual state.
        if is_fire:
            ok = "#c8ac74"
            warn = "#d97b5c"
            warn_quiet = "rgba(217, 123, 92, 0.14)"
        else:
            ok = "#859900"
            warn = "#dc322f"
            warn_quiet = "rgba(220, 50, 47, 0.14)"
        # Borders: thin gold hairline so panel edges read as lit mirror rims.
        if is_fire:
            border_idle = "rgba(248, 164, 64, 0.20)"
            border_subtle = "rgba(242, 231, 207, 0.05)"
            border_glass = "rgba(248, 164, 64, 0.28)"
        else:
            border_idle = "rgba(147, 161, 161, 0.10)"
            border_subtle = "rgba(147, 161, 161, 0.06)"
            border_glass = "rgba(147, 161, 161, 0.16)"
        if is_fire:
            font_family = (
                '"IBM Plex Sans", "Segoe UI", "SF Pro Text", "Inter", '
                '"Roboto", "Helvetica Neue", Arial, sans-serif'
            )
        else:
            font_family = (
                '"JetBrains Mono", "JetBrainsMono Nerd Font", "SF Mono", '
                'Menlo, "IBM Plex Mono", monospace'
            )
        return f"""
QMainWindow, QWidget {{
    background: {bg};
    color: {text_primary};
    font-family: {font_family};
    font-size: 11px;
    font-weight: 400;
    border-radius: 0px;
}}

QFrame {{
    border: 1px solid {border_idle};
    background: {surface};
    border-radius: 4px;
}}

QLabel {{ background: transparent; border-radius: 0px; }}
QLabel[isTitle="true"] {{
    color: {TITLE_BEACON_TEXT};
    background: {TITLE_BEACON_RED};
    border: 1px solid {TITLE_BEACON_RED};
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}
QToolButton[isTitle="true"] {{
    color: {TITLE_BEACON_TEXT};
    background: {TITLE_BEACON_RED};
    border: 1px solid {TITLE_BEACON_RED};
    padding: 3px 8px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.6px;
    text-transform: uppercase;
}}

/* Primary action — filled accent. This is the loud element. */
QPushButton {{
    border: 1px solid {accent_hover};
    background: {btn_bg};
    color: {accent_text};
    padding: 6px 14px;
    min-height: 22px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.3px;
    border-radius: 6px;
}}
QPushButton:hover {{
    background: {btn_bg_hover};
    border-color: {accent};
}}
QPushButton:pressed {{
    background: {btn_bg_pressed};
    border-color: {accent_pressed};
}}
QPushButton:checked {{
    background: {btn_bg_pressed};
    color: {accent_text};
    border-color: {accent_pressed};
}}
QPushButton:disabled {{
    background: {glass};
    border-color: {border_idle};
    color: {text_muted};
}}

/* Ghost / secondary push buttons — opt-in via QPushButton[variant="ghost"]. */
QPushButton[variant="ghost"] {{
    background: {glass};
    border: 1px solid {border_glass};
    color: {text_soft};
}}
QPushButton[variant="ghost"]:hover {{
    background: {glass_hover};
    color: {text_primary};
    border-color: {accent};
}}
QPushButton[variant="ghost"]:pressed {{
    background: {accent_active};
    color: {text_primary};
}}
QPushButton[variant="ghost"]:checked {{
    background: {accent_quiet};
    border-color: {accent};
    color: {accent};
}}
QPushButton[variant="ghost"]:disabled {{
    background: {border_subtle};
    border-color: {border_subtle};
    color: {text_muted};
}}

/* Tool buttons — quiet utility chrome, glass on hover. */
QToolButton {{
    border: 1px solid transparent;
    background: transparent;
    color: {text_soft};
    padding: 5px 9px;
    font-size: 11px;
    text-align: left;
    border-radius: 4px;
}}
QToolButton:hover {{
    background: {glass_hover};
    border-color: {border_glass};
    color: {text_primary};
}}
QToolButton:checked {{
    background: {accent_active};
    border-color: {accent};
    color: {accent};
}}

QTabWidget::pane {{
    border: none;
    border-top: 1px solid {border_idle};
    background: transparent;
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {text_muted};
    padding: 8px 14px;
    border: none;
    border-bottom: 2px solid transparent;
    min-width: 64px;
    font-size: 11px;
    font-weight: 500;
    letter-spacing: 0.2px;
}}
QTabBar::tab:selected {{
    background: transparent;
    color: {accent};
    border-bottom: 2px solid {accent};
}}
QTabBar::tab:hover:!selected {{
    color: {text_primary};
    background: transparent;
}}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ width: 6px; background: transparent; margin: 2px 0; }}
QScrollBar::handle:vertical {{ background: {border_idle}; min-height: 24px; border-radius: 3px; }}
QScrollBar::handle:vertical:hover {{ background: {accent_active}; }}
QScrollBar:horizontal {{ height: 6px; background: transparent; margin: 0 2px; }}
QScrollBar::handle:horizontal {{ background: {border_idle}; min-width: 24px; border-radius: 3px; }}
QScrollBar::handle:horizontal:hover {{ background: {accent_active}; }}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}

QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    border: 1px solid {border_glass};
    background: {glass};
    color: {text_primary};
    padding: 5px 8px;
    font-size: 11px;
    selection-background-color: {accent_active};
    selection-color: {text_primary};
    border-radius: 4px;
}}
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover,
QSpinBox:hover, QDoubleSpinBox:hover {{
    background: {glass_hover};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{
    background: {glass_focus};
    border: 1px solid {accent};
}}
QComboBox {{
    border: 1px solid {border_glass};
    background: {glass};
    color: {text_primary};
    padding: 5px 8px;
    min-height: 22px;
    font-size: 11px;
    selection-background-color: {accent_active};
    selection-color: {text_primary};
    border-radius: 4px;
}}
QComboBox:hover {{ background: {glass_hover}; }}
QComboBox:focus {{ background: {glass_focus}; border: 1px solid {accent}; }}
QComboBox::drop-down {{
    border: none;
    width: 20px;
}}
QComboBox QAbstractItemView {{
    background: {surface_high};
    color: {text_primary};
    border: 1px solid {border_glass};
    border-radius: 4px;
    selection-background-color: {accent_active};
    selection-color: {text_primary};
    padding: 2px;
}}

QListWidget {{
    border: 1px solid {border_glass};
    background: {glass};
    color: {text_primary};
    border-radius: 4px;
}}
QListWidget::item {{ padding: 5px 8px; border-radius: 3px; }}
QListWidget::item:hover {{ background: {glass_hover}; }}
QListWidget::item:selected {{
    background: {accent_active};
    color: {text_primary};
}}

QProgressBar {{
    border: none;
    background: {border_subtle};
    color: {text_muted};
    text-align: center;
    border-radius: 3px;
    min-height: 6px;
    max-height: 6px;
}}
QProgressBar::chunk {{
    background: {accent};
    border-radius: 3px;
}}

QCheckBox {{
    color: {text_soft};
    font-size: 11px;
    spacing: 6px;
}}
QCheckBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {border_glass};
    background: {glass};
    border-radius: 3px;
}}
QCheckBox::indicator:hover {{
    background: {glass_hover};
    border-color: {accent};
}}
QCheckBox::indicator:checked {{
    background: {accent};
    border-color: {accent};
}}

QTableWidget {{
    background: transparent;
    color: {text_primary};
    gridline-color: {border_subtle};
    selection-background-color: {accent_active};
    selection-color: {text_primary};
    border: 1px solid {border_idle};
    border-radius: 4px;
}}
QHeaderView::section {{
    background: transparent;
    color: {text_muted};
    border: none;
    border-bottom: 1px solid {border_idle};
    padding: 6px 8px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
}}

/* Critical alerts — semantic warn, only when active. */
QLabel#criticalAlert {{
    background: {warn_quiet};
    border: 1px solid {warn};
    color: {text_primary};
    padding: 6px 10px;
    border-radius: 4px;
}}
"""

    ad = theme_rgba("accent_dark", 0.55)
    ad72 = theme_rgba("accent_dark", 0.72)
    ad44 = theme_rgba("accent_dark", 0.44)
    ad40 = theme_rgba("accent_dark", 0.40)
    ad38 = theme_rgba("accent_dark", 0.38)
    ad30 = theme_rgba("accent_dark", 0.30)
    ad28 = theme_rgba("accent_dark", 0.28)
    ad26 = theme_rgba("accent_dark", 0.26)
    ad25 = theme_rgba("accent_dark", 0.25)
    ad18 = theme_rgba("accent_dark", 0.18)
    p88 = theme_rgba("panel", 0.88)
    p78 = theme_rgba("panel", 0.78)
    p40 = theme_rgba("panel", 0.40)
    h96 = theme_rgba("hover", 0.96)
    pr96 = theme_rgba("pressed", 0.96)
    pr82 = theme_rgba("pressed", 0.82)
    pr78 = theme_rgba("pressed", 0.78)
    ifill = theme_rgba("input_fill", 0.86)
    ilist = theme_rgba("input_list", 0.98)
    accent_hex = theme_hex("accent_dark")
    bg_hex = (_SCHEMES.get(_CURRENT_SCHEME_NAME) or _SCHEMES["material"]).get("bg_hex", "#F8F9FA")
    btn_bg = "#859900"
    btn_bg_hover = "#9aae19"
    btn_bg_pressed = "#6f8500"
    btn_text = "#002b36"
    return f"""
QMainWindow, QWidget#centralHud {{
    background: {bg_hex};
    color: {text_hex()};
    font-family: "IBM Plex Mono", "SF Mono", Menlo, monospace;
    font-size: 12px;
}}
QPushButton {{
    border: 1px solid {ad};
    background: {btn_bg};
    color: {btn_text};
    padding: 4px 10px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
    border-radius: 10px;
}}
QPushButton:hover {{
    background: {btn_bg_hover};
    color: {btn_text};
    border-color: {ad72};
}}
QPushButton:pressed {{
    background: {btn_bg_pressed};
    border-color: {pr78};
}}
QPushButton:checked {{
    background: {btn_bg_pressed};
    border-color: {pr82};
    color: {btn_text};
}}
QPushButton:disabled {{
    color: rgba(27,27,27,0.45);
    border-color: {ad18};
    background: rgba(134,239,172,0.38);
}}
QToolButton {{
    border: 1px solid {ad30};
    background: {btn_bg};
    color: {btn_text};
    padding: 4px 8px;
    text-align: left;
    border-radius: 10px;
}}
QToolButton:hover {{
    background: {btn_bg_hover};
    border-color: {ad72};
}}
QToolButton:checked {{
    background: {btn_bg_pressed};
    color: {btn_text};
}}
QTabWidget::pane {{
    border: none;
    border-top: 1px solid {ad30};
    background: {p40};
    top: -1px;
}}
QTabBar::tab {{
    background: {p78};
    color: {text_css(0.68)};
    padding: 8px 0;
    border: none;
    border-bottom: 1px solid {ad25};
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    min-width: 72px;
    font-size: 10px;
    letter-spacing: 1px;
}}
QTabBar::tab:selected {{
    background: {pr96};
    color: {text_hex()};
    border-bottom: 1px solid {pr82};
}}
QTabBar::tab:hover:!selected {{
    color: {text_hex()};
    background: {h96};
}}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ width: 8px; background: transparent; }}
QScrollBar::handle:vertical {{ background: {theme_rgba("accent_dark", 0.42)}; min-height: 24px; border-radius: 0px; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; }}
QScrollBar::handle:horizontal {{ background: {theme_rgba("accent_dark", 0.42)}; min-width: 24px; border-radius: 0px; }}
QScrollBar::add-line, QScrollBar::sub-line,
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
    border: none;
}}
QLineEdit, QTextEdit, QPlainTextEdit {{
    border: 1px solid {ad44};
    background: {ifill};
    color: {text_hex()};
    padding: 4px;
    selection-background-color: {ad28};
    border-radius: 0px;
}}
QComboBox {{
    border: 1px solid {ad44};
    background: {ifill};
    color: {text_hex()};
    padding: 4px;
    selection-background-color: {ad28};
    border-radius: 10px;
}}
QComboBox::drop-down {{
    border: none;
    width: 18px;
}}
QComboBox QAbstractItemView {{
    background: {ilist};
    color: {text_hex()};
    border: 1px solid {ad40};
    border-radius: 10px;
    selection-background-color: {ad28};
}}
QSlider::groove:horizontal {{
    height: 3px;
    background: {ad26};
    border-radius: 0px;
}}
QSlider::handle:horizontal {{
    width: 10px;
    height: 10px;
    margin: -4px 0;
    background: {accent_hex};
    border-radius: 0px;
}}
QLabel {{
    background: transparent;
}}
QLabel[isTitle="true"] {{
    background: {TITLE_BEACON_RED};
    color: {TITLE_BEACON_TEXT};
    border: 1px solid {TITLE_BEACON_RED};
    padding: 4px 10px;
    border-radius: 10px;
}}
QToolButton[isTitle="true"] {{
    background: {TITLE_BEACON_RED};
    color: {TITLE_BEACON_TEXT};
    border: 1px solid {TITLE_BEACON_RED};
    padding: 4px 10px;
    border-radius: 10px;
}}
"""


def get_global_stylesheet(*, cv_ops_wallpaper_blend: Optional[Any] = None) -> str:
    """Return the Insight global Wear stylesheet, optionally peeled under Cv Ops chrome.

    When ``cv_ops_wallpaper_blend`` is a
    ``insight_local.cvops.ui.backdrop_blend.WorkspaceBackdropBlend``, a short QSS
    appendix scopes the Wear root fill to ``#cvOpsWindow`` so a workbench
    wallpaper can show through.
    """
    if cv_ops_wallpaper_blend is None:
        return GLOBAL_STYLESHEET
    try:
        from insight_local.cvops.ui.backdrop_blend import WorkspaceBackdropBlend, compose_wallpaper_global_qss_addon
    except Exception:
        return GLOBAL_STYLESHEET
    if not isinstance(cv_ops_wallpaper_blend, WorkspaceBackdropBlend):
        return GLOBAL_STYLESHEET
    return GLOBAL_STYLESHEET + compose_wallpaper_global_qss_addon(
        cv_ops_wallpaper_blend.wear_shell_alpha_pct,
    )


def get_hud_strip_bg_css() -> str:
    if _CURRENT_SCHEME_NAME in ("marathon", "wear_marathon"):
        return theme_rgba("panel", 0.14)
    if _CURRENT_SCHEME_NAME == "fire":
        return theme_rgba("panel", 0.10)
    return theme_rgba("panel", 0.16)


def get_hud_bottom_strip_bg_css() -> str:
    """Bottom HUD strip fill — fully transparent. Buttons carry their own tinted backing; the strip itself is invisible so passthrough reads through the bottom band."""
    return "transparent"


def get_hud_strip_border_css() -> str:
    return HUD_STRIP_BORDER_CSS


def get_hud_meter_css(radius: int = 3) -> str:
    track_alpha = 0.18 if _CURRENT_SCHEME_NAME in ("marathon", "wear_marathon") else 0.14
    return (
        f"QProgressBar {{ border: none; background: {theme_rgba('accent_dark', track_alpha)}; "
        f"border-radius: {radius}px; }}"
        f"QProgressBar::chunk {{ background: {theme_hex('accent_dark')}; border-radius: {radius}px; }}"
    )


def get_hud_slider_css(*, handle_size: int = 10, handle_radius: int = 5) -> str:
    margin = -max(1, (handle_size - 4) // 2)
    return f"""
QSlider::groove:horizontal {{
    background: {theme_rgba('accent_dark', 0.18)};
    height: 4px;
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {theme_hex('accent_dark')};
    width: {handle_size}px;
    height: {handle_size}px;
    margin: {margin}px 0;
    border-radius: {handle_radius}px;
}}
QSlider::sub-page:horizontal {{
    background: {theme_rgba('accent_dark', 0.34)};
    border-radius: 2px;
}}
"""


_apply_scheme_to_globals()
configure_text_mode("black")
