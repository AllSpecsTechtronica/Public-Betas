"""Workbench wallpaper layering — scales CV Ops tiers and peels global Wear QSS."""

from __future__ import annotations

import re
from dataclasses import dataclass

_RGBA_RE = re.compile(
    r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)",
    re.IGNORECASE,
)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _fmt_alpha(a: float) -> str:
    """Short stable alpha for QSS."""
    s = f"{a:.4f}".rstrip("0").rstrip(".")
    return s if s else "0"


def scale_rgba_string(css_rgba: str, *, scale_pct: int, pct_ref: float = 50.0, lo: float = 0.03, hi: float = 0.98) -> str:
    """Multiply alpha in an rgba(...) string by (scale_pct / pct_ref).

    ``scale_pct == 50`` keeps the authored alpha unchanged.
    """
    m = _RGBA_RE.match(str(css_rgba).strip())
    if not m:
        return css_rgba
    r, g, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
    a0 = float(m.group(4))
    factor = max(1e-3, float(scale_pct)) / max(1e-3, float(pct_ref))
    anew = clamp01(a0 * factor)
    anew = max(lo, min(hi, anew))
    return f"rgba({r},{g},{b},{_fmt_alpha(anew)})"


def compose_wallpaper_global_qss_addon(wear_shell_alpha_pct: int) -> str:
    """Neutralize Wear ``QMainWindow, QWidget`` fill under Cv Ops chrome.

    When ``wear_shell_alpha_pct`` is 0 the main window peel is transparent; higher
    values add a tinted veil so text stays readable on busy photos.
    """
    # Local import avoids import cycles at startup (theme pulls this lazily).
    from ...ui.theme import theme_rgba

    shell = clamp01(wear_shell_alpha_pct / 100.0)
    if shell <= 0.003:
        main_bg = "transparent"
    else:
        # Flat tint so stacking stays predictable vs gradient + wallpaper.
        main_bg = theme_rgba("panel", min(0.94, shell * 0.98))
    return f"""
/* [Cv Ops wallpaper — peel Insight global Wear root under #cvOpsWindow] */
QMainWindow#cvOpsWindow {{
    background: {main_bg} !important;
}}
QMainWindow#cvOpsWindow QWidget#cvOpsRoot {{
    background: transparent !important;
}}
"""


@dataclass(frozen=True)
class WorkspaceBackdropBlend:
    """Percents persisted in Cv Ops ``settings.json`` — see sliders in Appearance."""

    wear_shell_alpha_pct: int = 5
    """Wash over the Insight global Wear sheet behind Cv Ops (**0–100**, 0 = fully clear)."""

    tabs_scale_pct: int = 50
    frames_scale_pct: int = 50
    cells_scale_pct: int = 50
    controls_scale_pct: int = 50
    """Layer alpha multipliers (**50** = authored wallpaper preset; higher = meatier chrome)."""


def blend_from_cvops_settings(s: object) -> WorkspaceBackdropBlend:
    return WorkspaceBackdropBlend(
        wear_shell_alpha_pct=max(0, min(100, int(getattr(s, "workspace_backdrop_wear_alpha", 5)))),
        tabs_scale_pct=max(10, min(180, int(getattr(s, "workspace_backdrop_scale_tabs", 50)))),
        frames_scale_pct=max(10, min(180, int(getattr(s, "workspace_backdrop_scale_frames", 50)))),
        cells_scale_pct=max(10, min(180, int(getattr(s, "workspace_backdrop_scale_cells", 50)))),
        controls_scale_pct=max(10, min(180, int(getattr(s, "workspace_backdrop_scale_controls", 50)))),
    )
