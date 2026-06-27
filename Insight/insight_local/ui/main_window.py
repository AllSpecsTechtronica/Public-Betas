from __future__ import annotations
import hashlib
from html import escape
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QEasingCurve, QObject, QPoint, QPointF, QPropertyAnimation, QRect, QRectF, QSize, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QIcon,
    QImage,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PyQt6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFormLayout,
    QFrame,
    QGraphicsDropShadowEffect,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
    QHeaderView,
    QStackedWidget,
    QTabWidget,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    ASSETS_DIR,
    CVOPS_BASE_URL,
    GALLERY_DB_PATH,
    HISTORY_TTL_SECONDS,
    INSIGHT_ANTHROPIC_MODEL,
    INSIGHT_OLLAMA_MODEL,
    INSIGHT_OPENAI_MODEL,
    ROOT_DIR,
    RuntimeConfig,
    heatmap_category,
    model_choice_name,
    normalize_color_scheme,
)
from ..filtering import matches_detection_view
from ..engine.local_session import LocalInsightSession
from ..engine.effects_worker import EffectsWorker
from ..engine.video_effects import apply_hairline_edge_background
from .gallery_panel import GalleryPanel
from .loading_gate import BootSelectorOverlay, LoadingGateOverlay
from .media_utils import bgr_to_qimage, pixmap_from_b64_jpeg, pixmap_from_b64_png
from .settings_panel import SettingsPanel
from .radial_menu import RadialAction, RadialMenuOverlay
from .sidebar_panel import SidebarPanel
from .theme import (
    HUD_MUTED,
    _scheme_rgb,
    apply_text_palette,
    contrast_text_hex,
    configure_color_scheme,
    current_color_scheme,
    detection_label_text,
    get_global_stylesheet,
    get_hud_meter_css,
    get_hud_slider_css,
    get_hud_bottom_strip_bg_css,
    get_hud_strip_bg_css,
    get_hud_strip_border_css,
    theme_holographic_fire,
    surface_muted_css,
    text_css,
    text_hex,
    theme_hex,
    theme_metallic,
    theme_rgba,
)
from .timeline_card import TimelineCardWidget
from ..suite_manager import SuiteManager
from ..cvops.ui.range_panel import TestRangePanel


def format_confidence_percent(value: object, *, normalized: bool = True) -> str:
    """Format confidence with low/high markers: <50 => ~, >65 => +."""
    try:
        raw = float(value)
    except (TypeError, ValueError):
        raw = 0.0
    pct = int(round(raw * 100.0)) if normalized else int(round(raw))
    pct = max(0, min(100, pct))
    prefix = "~" if pct < 50 else ("+" if pct > 65 else "")
    return f"{prefix}{pct}%"


class _RoiControlRail(QFrame):
    """Thin horizontal strip of ROI controls anchored to the ROI rect.

    Circle ROI scale presets live next to the ROI itself.
    """

    preset_requested = pyqtSignal(float)

    _PRESETS = ((".5", 0.5), ("1", 1.0), ("2", 2.0), ("3", 3.0))

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("roiControlRail")
        self.setMaximumHeight(24)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 2, 3, 2)
        layout.setSpacing(2)

        self._preset_buttons: list[tuple[float, QPushButton]] = []
        for label, scale in self._PRESETS:
            btn = QPushButton(label, self)
            btn.setFlat(True)
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip(f"ROI scale {label}")
            btn.clicked.connect(lambda _c, s=scale: self.preset_requested.emit(s))
            layout.addWidget(btn)
            self._preset_buttons.append((scale, btn))

        self.setStyleSheet(self._build_css())

    def _build_css(self) -> str:
        accent = theme_hex("accent_dark")
        on_accent = contrast_text_hex("accent_dark")
        paper = theme_hex("paper")
        if current_color_scheme() == "beacon":
            # Neo-Swiss / Beacon: cool substrate, vermillion data chips, paper on filled state.
            bg = theme_rgba("mist_2", 0.94)
            border = theme_rgba("graphite_5", 0.55)
            chip = theme_rgba("accent_dark", 0.12)
            chip_hover = theme_rgba("accent_dark", 0.26)
            return f"""
QFrame#roiControlRail {{
    background: {bg};
    border: 1px solid {border};
    border-radius: 0px;
}}
QFrame#roiControlRail QPushButton {{
    background: {chip};
    color: {accent};
    border: 1px solid transparent;
    padding: 1px 4px;
    font-family: "JetBrains Mono", "IBM Plex Mono", "Menlo", monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.06em;
    min-width: 20px;
    max-width: 28px;
    min-height: 16px;
    max-height: 18px;
}}
QFrame#roiControlRail QPushButton:hover {{
    background: {chip_hover};
    border-color: {accent};
}}
QFrame#roiControlRail QPushButton:checked {{
    background: {accent};
    color: {paper};
    border-color: {accent};
}}
"""
        chip = theme_rgba("accent_dark", 0.18)
        chip_hover = theme_rgba("accent_dark", 0.34)
        bg = theme_rgba("panel", 0.86)
        border = theme_rgba("accent_dark", 0.42)
        return f"""
QFrame#roiControlRail {{
    background: {bg};
    border: 1px solid {border};
    border-radius: 0px;
}}
QFrame#roiControlRail QPushButton {{
    background: {chip};
    color: {accent};
    border: 1px solid transparent;
    padding: 1px 4px;
    font-family: "Roboto Mono", "JetBrains Mono", "Menlo", monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.5px;
    min-width: 20px;
    max-width: 28px;
    min-height: 16px;
    max-height: 18px;
}}
QFrame#roiControlRail QPushButton:hover {{
    background: {chip_hover};
    border-color: {accent};
}}
QFrame#roiControlRail QPushButton:checked {{
    background: {accent};
    color: {on_accent};
    border-color: {accent};
}}
"""

    def refresh_theme(self) -> None:
        self.setStyleSheet(self._build_css())

    def set_active_preset(self, scale: float) -> None:
        for s, btn in self._preset_buttons:
            btn.blockSignals(True)
            btn.setChecked(abs(s - float(scale)) < 0.02)
            btn.blockSignals(False)


class VideoPane(QWidget):
    """Full-bleed video with optional ROI overlay, corner handles, and move/resize."""

    roi_capture_requested = pyqtSignal()
    roi_norm_changed = pyqtSignal(dict)
    roi_preset_requested = pyqtSignal(float)
    track_clicked = pyqtSignal(int)

    ROI_BASE = 0.16
    _HANDLE_HIT = 12.0
    _BBOX_FADE_IN_SECONDS = 0.045
    _BBOX_FADE_OUT_SECONDS = 0.070
    _BBOX_FADE_TICK_MS = 16

    # Pre-rendered thermal blob (outer + core) at a fixed size, per category
    _BLOB_SIZE = 128
    _thermal_blobs: dict[str, QPixmap] = {}
    _thermal_cores: dict[str, QPixmap] = {}

    _BLOB_COLORS: dict[str, dict[str, list[tuple[float, tuple[int, int, int, int]]]]] = {
        "human": {
            "outer": [
                (0.00, (255, 100, 0, 204)), (0.12, (255, 160, 0, 166)),
                (0.28, (255, 220, 0, 122)), (0.42, (160, 230, 30, 89)),
                (0.58, (0, 200, 180, 64)),  (0.75, (20, 100, 220, 41)),
                (0.90, (30, 40, 180, 20)),  (1.00, (20, 20, 120, 0)),
            ],
            "core": [
                (0.0, (255, 80, 0, 230)), (0.6, (255, 140, 0, 102)),
                (1.0, (255, 180, 0, 0)),
            ],
        },
        "plant": {
            "outer": [
                (0.00, (40, 200, 50, 204)),  (0.12, (60, 210, 60, 166)),
                (0.28, (100, 230, 60, 122)), (0.42, (80, 210, 100, 89)),
                (0.58, (40, 180, 80, 64)),   (0.75, (20, 140, 60, 41)),
                (0.90, (10, 100, 50, 20)),   (1.00, (5, 60, 30, 0)),
            ],
            "core": [
                (0.0, (40, 220, 50, 230)), (0.6, (60, 190, 40, 102)),
                (1.0, (80, 210, 50, 0)),
            ],
        },
        "animal": {
            "outer": [
                (0.00, (0, 200, 190, 204)),  (0.12, (0, 210, 200, 166)),
                (0.28, (0, 220, 200, 122)),  (0.42, (20, 200, 180, 89)),
                (0.58, (30, 180, 170, 64)),  (0.75, (20, 140, 140, 41)),
                (0.90, (10, 100, 110, 20)),  (1.00, (5, 60, 70, 0)),
            ],
            "core": [
                (0.0, (0, 210, 200, 230)), (0.6, (0, 180, 170, 102)),
                (1.0, (0, 200, 190, 0)),
            ],
        },
        "inorganic": {
            "outer": [
                (0.00, (30, 100, 255, 204)),  (0.12, (40, 130, 255, 166)),
                (0.28, (50, 170, 255, 122)),  (0.42, (40, 190, 230, 89)),
                (0.58, (30, 160, 200, 64)),   (0.75, (20, 110, 180, 41)),
                (0.90, (10, 70, 150, 20)),    (1.00, (5, 40, 110, 0)),
            ],
            "core": [
                (0.0, (30, 100, 255, 230)), (0.6, (20, 120, 230, 102)),
                (1.0, (30, 150, 255, 0)),
            ],
        },
        "tech": {
            "outer": [
                (0.00, (170, 50, 255, 204)),  (0.12, (180, 70, 255, 166)),
                (0.28, (200, 100, 255, 122)), (0.42, (180, 110, 230, 89)),
                (0.58, (150, 80, 200, 64)),   (0.75, (110, 50, 180, 41)),
                (0.90, (70, 30, 150, 20)),    (1.00, (50, 15, 110, 0)),
            ],
            "core": [
                (0.0, (170, 50, 255, 230)), (0.6, (150, 60, 230, 102)),
                (1.0, (180, 80, 255, 0)),
            ],
        },
    }

    _LABEL_ACCENT: dict[str, tuple[int, int, int]] = {
        "human": (255, 140, 0),
        "plant": (80, 210, 60),
        "animal": (0, 200, 190),
        "inorganic": (60, 140, 255),
        "tech": (170, 80, 255),
    }

    _THERMAL_ACCENT: dict[str, tuple[int, int, int]] = {
        "human": (255, 80, 40),
        "plant": (60, 210, 80),
        "animal": (0, 200, 190),
        "inorganic": (255, 150, 60),
        "tech": (255, 80, 180),
    }

    _THERMAL_HEX: dict[str, str] = {
        "human": "#ff5028",
        "plant": "#3cd250",
        "animal": "#00c8be",
        "inorganic": "#ff963c",
        "tech": "#ff50b4",
    }
    _LABEL_ICON_FILES: dict[str, str] = {
        "people": "people.svg",
        "animals": "animals.svg",
        "tech": "tech.svg",
        "objects": "objects.svg",
    }
    _LABEL_ICON_CATEGORY: dict[str, str] = {
        "human": "people",
        "animal": "animals",
        "tech": "tech",
        "plant": "objects",
        "inorganic": "objects",
    }
    _label_icon_cache: dict[tuple[str, int], QPixmap] = {}

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self._bgr: Optional[np.ndarray] = None
        self._qimg = QImage()
        self.roi_active = False
        self.roi_shape = "circle"
        self.roi_norm = {"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}
        self.roi_scale_preset = 1.0
        self.bbox_style = "square"
        self.show_heat_labels = False
        self.labels_only = False
        self.filter_text = ""
        self.filter_categories: set[str] = set()
        self._drag: Optional[dict[str, Any]] = None
        self.active_mode = False
        self.active_heat = False
        # thermal_mode: "edge" | "clouds" | "edge+clouds"
        self.thermal_mode = "edge"
        self._overlays: list[dict[str, Any]] = []
        self._highlight_track_id: Optional[int] = None
        # Monocle capture animation state
        self._monocle_flash = 0.0
        self._monocle_timer = QTimer(self)
        self._monocle_timer.setInterval(16)
        self._monocle_timer.timeout.connect(self._tick_monocle)
        # -- Render caches --
        self._bg_cache: Optional[QPixmap] = None
        self._bg_cache_size: tuple[int, int] = (0, 0)
        self._veil_cache: Optional[QPixmap] = None
        self._overlay_cache: Optional[QPixmap] = None
        self._overlay_sig: tuple = ()
        self._overlay_payload_sig: tuple = ()
        self._overlay_cr: Optional[QRectF] = None
        self._overlay_fade_states: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._overlay_fade_timer = QTimer(self)
        self._overlay_fade_timer.setInterval(self._BBOX_FADE_TICK_MS)
        self._overlay_fade_timer.timeout.connect(self._tick_overlay_fade)
        self._overlay_fade_last_ts = time.perf_counter()
        self.show_grid = False
        self.subgrid_mode = "quads"
        self._ensure_thermal_blobs()
        # [ROI RAIL] Inline control strip pinned to the left of the ROI rect.
        # Replaces the popup/radial entries for shape cycling and scale presets.
        self._roi_rail = _RoiControlRail(self)
        self._roi_rail.hide()
        self._roi_rail.preset_requested.connect(self.roi_preset_requested.emit)
        self._roi_rail.set_active_preset(self.roi_scale_preset)
        self._roi_rail_last_geom: Optional[QRect] = None

    @classmethod
    def _ensure_thermal_blobs(cls) -> None:
        if cls._thermal_blobs:
            return
        s = cls._BLOB_SIZE
        c = s / 2.0
        for cat, colors in cls._BLOB_COLORS.items():
            # Outer blob
            img = QImage(s, s, QImage.Format.Format_ARGB32_Premultiplied)
            img.fill(QColor(0, 0, 0, 0))
            p = QPainter(img)
            p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p.setPen(Qt.PenStyle.NoPen)
            g = QRadialGradient(c, c, c)
            for stop, (r, gv, b, a) in colors["outer"]:
                g.setColorAt(stop, QColor(r, gv, b, a))
            p.setBrush(QBrush(g))
            p.drawEllipse(QPointF(c, c), c, c)
            p.end()
            cls._thermal_blobs[cat] = QPixmap.fromImage(img)
            # Core blob
            img2 = QImage(s, s, QImage.Format.Format_ARGB32_Premultiplied)
            img2.fill(QColor(0, 0, 0, 0))
            p2 = QPainter(img2)
            p2.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            p2.setPen(Qt.PenStyle.NoPen)
            gc = QRadialGradient(c, c, c)
            for stop, (r, gv, b, a) in colors["core"]:
                gc.setColorAt(stop, QColor(r, gv, b, a))
            p2.setBrush(QBrush(gc))
            p2.drawEllipse(QPointF(c, c), c, c)
            p2.end()
            cls._thermal_cores[cat] = QPixmap.fromImage(img2)

    def _build_overlay_sig(self) -> tuple:
        """Compact signature for overlay data — only regenerate cache when this changes."""
        parts: list[tuple] = []
        parts.append((
            self.active_mode,
            self.active_heat,
            self.thermal_mode,
            self.bbox_style,
            self.show_heat_labels,
            self.labels_only,
            self.filter_text,
            tuple(sorted(self.filter_categories)),
        ))
        for ov, fade_alpha in self._visible_overlays():
            bn = ov.get("bbox_norm")
            if not bn or len(bn) < 4:
                continue
            parts.append((
                int(self._highlight_track_id or 0),
                ov.get("track_id", 0),
                round(bn[0], 3), round(bn[1], 3), round(bn[2], 3), round(bn[3], 3),
                round(float(ov.get("confidence", 0)), 2),
                round(float(ov.get("alpha_scale", 1.0) or 1.0), 2),
                round(float(fade_alpha), 2),
                ov.get("identity", ""),
                ov.get("label", ""),
            ))
        return tuple(parts)

    def _invalidate_bg(self) -> None:
        self._bg_cache = None
        self._veil_cache = None

    def _get_bg(self) -> QPixmap:
        """Cached base gradient (behind video). Veil is drawn separately after video."""
        w, h = self.width(), self.height()
        if self._bg_cache is not None and self._bg_cache_size == (w, h):
            return self._bg_cache
        pm = QPixmap(w, h)
        p = QPainter(pm)
        base_grad = QLinearGradient(0, 0, w, h)
        base_grad.setColorAt(0.0, QColor(10, 4, 4))
        base_grad.setColorAt(0.62, QColor(6, 2, 2))
        base_grad.setColorAt(1.0, QColor(4, 1, 1))
        p.fillRect(0, 0, w, h, QBrush(base_grad))
        p.end()
        self._bg_cache = pm
        self._bg_cache_size = (w, h)
        return pm

    def _get_veil(self) -> QPixmap:
        """Cached veil overlay (on top of video). Separate from bg for correct layering."""
        w, h = self.width(), self.height()
        if self._veil_cache is not None and self._veil_cache.width() == w and self._veil_cache.height() == h:
            return self._veil_cache
        pm = QPixmap(w, h)
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        veil = QLinearGradient(0, 0, 0, h)
        veil.setColorAt(0.0, QColor(8, 2, 2, 5))
        veil.setColorAt(1.0, QColor(8, 2, 2, 30))
        p.fillRect(0, 0, w, h, QBrush(veil))
        p.end()
        self._veil_cache = pm
        return pm

    def set_frame(self, bgr: np.ndarray) -> None:
        self._bgr = bgr
        self._qimg = bgr_to_qimage(bgr)
        self.update()

    def invalidate_overlay_cache(self) -> None:
        self._overlay_cache = None
        self.update()

    def set_overlays(self, overlays: list[dict]) -> None:
        next_overlays = list(overlays or [])
        payload_sig = self._build_overlay_payload_sig(next_overlays)
        if payload_sig == self._overlay_payload_sig:
            return
        self._overlay_payload_sig = payload_sig
        self._overlays = next_overlays
        self._sync_overlay_fade_states()
        self._overlay_cache = None  # force rebuild only when overlay data changes

    @staticmethod
    def _build_overlay_payload_sig(overlays: list[dict]) -> tuple:
        parts: list[tuple[Any, ...]] = []
        for index, overlay in enumerate(overlays or []):
            bn = overlay.get("bbox_norm")
            if bn and len(bn) >= 4:
                bbox = (
                    round(float(bn[0]), 3),
                    round(float(bn[1]), 3),
                    round(float(bn[2]), 3),
                    round(float(bn[3]), 3),
                )
            else:
                bbox = ()
            parts.append((
                int(overlay.get("track_id", 0) or 0),
                index,
                bbox,
                round(float(overlay.get("confidence", 0) or 0), 2),
                round(float(overlay.get("alpha_scale", 1.0) or 1.0), 2),
                str(overlay.get("label", "") or ""),
                str(overlay.get("identity", "") or ""),
                str(overlay.get("event_tag", "") or ""),
            ))
        return tuple(parts)

    def _overlay_key(self, overlay: dict[str, Any], index: int) -> tuple[Any, ...]:
        track_id = int(overlay.get("track_id", 0) or 0)
        if track_id > 0:
            return ("track", track_id)
        bn = overlay.get("bbox_norm")
        if not bn or len(bn) < 4:
            return ("anon", index, str(overlay.get("label", "")), str(overlay.get("identity", "")))
        return (
            "anon",
            index,
            round(float(bn[0]), 2),
            round(float(bn[1]), 2),
            round(float(bn[2]), 2),
            round(float(bn[3]), 2),
            str(overlay.get("label", "")),
            str(overlay.get("identity", "")),
        )

    def _sync_overlay_fade_states(self) -> None:
        now = time.perf_counter()
        self._overlay_fade_last_ts = now
        seen_keys: set[tuple[Any, ...]] = set()
        for index, overlay in enumerate(self._overlays):
            key = self._overlay_key(overlay, index)
            seen_keys.add(key)
            state = self._overlay_fade_states.get(key)
            if state is None:
                self._overlay_fade_states[key] = {
                    "overlay": overlay,
                    "alpha": 0.0,
                    "target": 1.0,
                }
            else:
                state["overlay"] = overlay
                state["target"] = 1.0
        for key, state in self._overlay_fade_states.items():
            if key not in seen_keys:
                state["target"] = 0.0
        changed = self._step_overlay_fades(now, force=True)
        if any(abs(float(state["target"]) - float(state["alpha"])) > 0.001 for state in self._overlay_fade_states.values()):
            if not self._overlay_fade_timer.isActive():
                self._overlay_fade_timer.start()
        else:
            self._overlay_fade_timer.stop()
        if changed:
            self.update()

    def _step_overlay_fades(self, now: float, *, force: bool = False) -> bool:
        dt = max(0.0, now - self._overlay_fade_last_ts)
        if force and dt < (self._BBOX_FADE_TICK_MS / 1000.0):
            dt = self._BBOX_FADE_TICK_MS / 1000.0
        self._overlay_fade_last_ts = now
        changed = False
        remove_keys: list[tuple[Any, ...]] = []
        in_speed = 1.0 / max(0.001, self._BBOX_FADE_IN_SECONDS)
        out_speed = 1.0 / max(0.001, self._BBOX_FADE_OUT_SECONDS)
        for key, state in self._overlay_fade_states.items():
            alpha = float(state.get("alpha", 0.0))
            target = float(state.get("target", 1.0))
            next_alpha = alpha
            if target > alpha:
                next_alpha = min(target, alpha + dt * in_speed)
            elif target < alpha:
                next_alpha = max(target, alpha - dt * out_speed)
            if abs(next_alpha - alpha) > 0.0001:
                state["alpha"] = next_alpha
                changed = True
            if target <= 0.0 and float(state.get("alpha", 0.0)) <= 0.001:
                remove_keys.append(key)
        for key in remove_keys:
            self._overlay_fade_states.pop(key, None)
            changed = True
        return changed

    def _tick_overlay_fade(self) -> None:
        if not self._overlay_fade_states:
            self._overlay_fade_timer.stop()
            return
        changed = self._step_overlay_fades(time.perf_counter())
        if changed:
            self.update()
        if not any(abs(float(state["target"]) - float(state["alpha"])) > 0.001 for state in self._overlay_fade_states.values()):
            self._overlay_fade_timer.stop()

    def set_highlight_track(self, track_id: Optional[int]) -> None:
        track_id = int(track_id) if track_id not in (None, "") else None
        if self._highlight_track_id == track_id:
            return
        self._highlight_track_id = track_id
        self._overlay_cache = None
        self.update()

    def set_bbox_style(self, style: str) -> None:
        s = str(style).strip().lower()
        next_style = s if s in ("diamond", "circle") else "square"
        if self.bbox_style == next_style:
            return
        self.bbox_style = next_style
        self._overlay_cache = None
        self.update()

    def set_heat_labels_visible(self, visible: bool) -> None:
        next_visible = bool(visible)
        if self.show_heat_labels == next_visible:
            return
        self.show_heat_labels = next_visible
        self._overlay_cache = None
        self.update()

    def set_labels_only(self, enabled: bool) -> None:
        next_enabled = bool(enabled)
        if self.labels_only == next_enabled:
            return
        self.labels_only = next_enabled
        self._overlay_cache = None
        self.update()

    @classmethod
    def _label_icon_key(cls, label: object) -> str:
        category = heatmap_category(str(label or ""))
        return cls._LABEL_ICON_CATEGORY.get(category, "objects")

    @classmethod
    def _label_icon_pixmap(cls, label: object, size: int = 12) -> QPixmap:
        icon_key = cls._label_icon_key(label)
        cache_key = (icon_key, int(size))
        cached = cls._label_icon_cache.get(cache_key)
        if cached is not None:
            return cached
        file_name = cls._LABEL_ICON_FILES.get(icon_key, "objects.svg")
        icon_path = ASSETS_DIR / "icons" / file_name
        pixmap = QPixmap()
        if icon_path.exists():
            pixmap = QIcon(str(icon_path)).pixmap(QSize(size, size))
        cls._label_icon_cache[cache_key] = pixmap
        return pixmap

    def set_filter_text(self, text: str) -> None:
        next_text = str(text or "").strip().lower()
        if self.filter_text == next_text:
            return
        self.filter_text = next_text
        self._overlay_cache = None
        self.update()

    def set_filter_categories(self, categories: set[str]) -> None:
        next_categories = {str(item).strip().lower() for item in categories if str(item).strip()}
        if self.filter_categories == next_categories:
            return
        self.filter_categories = next_categories
        self._overlay_cache = None
        self.update()

    def _overlay_matches_filter(self, overlay: dict[str, Any]) -> bool:
        return matches_detection_view(
            self.filter_text,
            self.filter_categories,
            overlay.get("label", ""),
            overlay.get("event_tag", ""),
            overlay.get("identity", ""),
            heatmap_category(str(overlay.get("label", "") or "")),
        )

    def _visible_overlays(self) -> list[tuple[dict[str, Any], float]]:
        overlays = [
            (state["overlay"], float(state.get("alpha", 1.0)))
            for state in self._overlay_fade_states.values()
            if float(state.get("alpha", 0.0)) > 0.001 and self._overlay_matches_filter(state["overlay"])
        ]
        if self.active_heat:
            # Clouds mode needs overlays for blob rendering; pure edge mode hides them
            if "clouds" in self.thermal_mode:
                return overlays
            return []
        if self.active_mode:
            return overlays
        if self._highlight_track_id is None:
            return []
        return [
            (ov, alpha) for ov, alpha in overlays
            if int(ov.get("track_id", 0) or 0) == self._highlight_track_id
        ]

    def _hit_overlay_track(self, pos: QPoint, cr: QRectF) -> Optional[int]:
        for ov in reversed(self._overlays):
            bn = ov.get("bbox_norm")
            if not bn or len(bn) < 4:
                continue
            bx1 = cr.left() + float(bn[0]) * cr.width()
            by1 = cr.top() + float(bn[1]) * cr.height()
            bx2 = cr.left() + float(bn[2]) * cr.width()
            by2 = cr.top() + float(bn[3]) * cr.height()
            if QRectF(bx1, by1, bx2 - bx1, by2 - by1).contains(QPointF(pos)):
                return int(ov.get("track_id", 0) or 0)
        return None

    def _content_rect(self) -> QRectF:
        if self._qimg.isNull():
            return QRectF(self.rect())
        vw, vh = self.width(), self.height()
        fw, fh = self._qimg.width(), self._qimg.height()
        if fw <= 0 or fh <= 0:
            return QRectF(0, 0, vw, vh)
        scale = max(vw / fw, vh / fh)
        nw, nh = fw * scale, fh * scale
        ox = (vw - nw) / 2
        oy = (vh - nh) / 2
        return QRectF(ox, oy, nw, nh)

    def _norm_from_pos(self, pos: QPoint) -> tuple[float, float]:
        cr = self._content_rect()
        x = (pos.x() - cr.left()) / cr.width() if cr.width() > 0 else 0
        y = (pos.y() - cr.top()) / cr.height() if cr.height() > 0 else 0
        return max(0.0, min(1.0, x)), max(0.0, min(1.0, y))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._invalidate_bg()
        self._overlay_cache = None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        # 1. Cached background gradient (only rebuilt on resize)
        painter.drawPixmap(0, 0, self._get_bg())
        cr = self._content_rect()
        if not self._qimg.isNull():
            painter.drawImage(cr, self._qimg)
        # 1b. Cached veil on top of video
        painter.drawPixmap(0, 0, self._get_veil())
        # 2. Cached overlay layer (only rebuilt when detection data changes)
        visible_overlays = self._visible_overlays()
        if visible_overlays:
            sig = self._build_overlay_sig()
            if self._overlay_cache is None or sig != self._overlay_sig or self._overlay_cr != cr:
                self._overlay_sig = sig
                self._overlay_cr = QRectF(cr)
                self._overlay_cache = self._render_overlay_layer(cr)
            painter.drawPixmap(0, 0, self._overlay_cache)
        # 3. ROI drawn live (interactive, changes on drag)
        if self.roi_active:
            self._draw_roi(painter, cr)
        # 4. 3x3 grid overlay
        if self.show_grid:
            self._draw_grid(painter, cr)
        painter.end()
        # 5. ROI control rail — repositioned every paint so it tracks drags.
        self._layout_roi_rail(cr)

    def _layout_roi_rail(self, cr: QRectF) -> None:
        if not self.roi_active:
            self._roi_rail_last_geom = None
            if self._roi_rail.isVisible():
                self._roi_rail.hide()
            return
        rf = self._roi_screen_rectf(cr)
        size = self._roi_rail.sizeHint()
        rail_w = max(118, min(148, size.width()))
        rail_h = max(20, min(24, size.height()))
        gap = 6
        x = int(rf.center().x() - rail_w / 2)
        x = max(4, min(self.width() - rail_w - 4, x))
        y = int(rf.top()) - rail_h - gap
        if y < 4:
            y = int(rf.bottom()) + gap
        y = max(4, min(self.height() - rail_h - 4, y))
        geom = QRect(x, y, rail_w, rail_h)
        need_raise = False
        if self._roi_rail_last_geom != geom:
            self._roi_rail_last_geom = QRect(geom)
            self._roi_rail.setGeometry(geom)
            need_raise = True
        if not self._roi_rail.isVisible():
            self._roi_rail.show()
            need_raise = True
        if need_raise:
            self._roi_rail.raise_()

    def update_roi_rail_preset(self, scale: float) -> None:
        self._roi_rail.set_active_preset(scale)

    def refresh_roi_rail_theme(self) -> None:
        self._roi_rail.refresh_theme()

    def _draw_grid(self, painter: QPainter, cr: QRectF) -> None:
        """Draw a 3x3 main grid with optional per-cell subgrid (quads or halves)."""
        cw = cr.width() / 3
        ch = cr.height() / 3
        # Subgrid lines (midpoint of each cell) — less transparent than main
        if self.subgrid_mode in {"quads", "halves"}:
            painter.setPen(QPen(QColor(220, 220, 220, 95), 0.6))
            for i in range(3):
                x = cr.left() + cw * i + cw * 0.5
                painter.drawLine(QPointF(x, cr.top()), QPointF(x, cr.bottom()))
                if self.subgrid_mode == "quads":
                    y = cr.top() + ch * i + ch * 0.5
                    painter.drawLine(QPointF(cr.left(), y), QPointF(cr.right(), y))
        # Main 3x3 dividers — more subtle
        painter.setPen(QPen(QColor(220, 220, 220, 50), 0.8))
        for i in range(1, 3):
            x = cr.left() + cw * i
            painter.drawLine(QPointF(x, cr.top()), QPointF(x, cr.bottom()))
            y = cr.top() + ch * i
            painter.drawLine(QPointF(cr.left(), y), QPointF(cr.right(), y))

        # Number the 8 outer cells (center intentionally blank):
        # 1 4 6
        # 2   7
        # 3 5 8
        number_map = {
            (0, 0): "1",
            (1, 0): "2",
            (2, 0): "3",
            (0, 1): "4",
            (2, 1): "5",
            (0, 2): "6",
            (1, 2): "7",
            (2, 2): "8",
        }
        painter.save()
        try:
            label_font = painter.font()
            label_font.setPointSizeF(max(9.0, min(18.0, min(cw, ch) * 0.12)))
            painter.setFont(label_font)
        except Exception:
            pass
        painter.setPen(QPen(QColor(245, 245, 245, 172), 1.0))
        pad_x = cw * 0.08
        pad_y = ch * 0.18
        for (row, col), label in number_map.items():
            tx = cr.left() + col * cw + pad_x
            ty = cr.top() + row * ch + pad_y
            painter.drawText(QPointF(tx, ty), label)
        painter.restore()

    def _roi_color(self) -> tuple[int, int, int]:
        return _scheme_rgb("roi")

    def _draw_roi(self, painter: QPainter, cr: QRectF) -> None:
        rf = self._roi_screen_rectf(cr)
        rr, rg, rb = self._roi_color()
        glow_pen = QPen(QColor(rr, rg, rb, 52), 1.4)
        frame_pen = QPen(QColor(rr, rg, rb, 168), 0.9)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        side = min(rf.width(), rf.height())
        cx = rf.center().x()
        cy = rf.center().y()
        er = QRectF(cx - side / 2, cy - side / 2, side, side)
        self._draw_circle_roi_ticks(painter, er, glow_pen, frame_pen, (rr, rg, rb))
        self._draw_handles(painter, er, circle=True, rgb=(rr, rg, rb))
        if self._monocle_flash > 0.0:
            flash_alpha = int(255 * 0.18 * self._monocle_flash)
            painter.setPen(QPen(QColor(rr, rg, rb, min(255, flash_alpha + 56)), 1.2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            scale = 1.0 + 0.14 * self._monocle_flash
            flash_side = side * scale
            flash_rect = QRectF(cx - flash_side / 2, cy - flash_side / 2, flash_side, flash_side)
            painter.drawEllipse(flash_rect)

    def _draw_circle_roi_ticks(self, painter: QPainter, er: QRectF, glow_pen: QPen, frame_pen: QPen, rgb: tuple[int, int, int] = (220, 16, 16)) -> None:
        rr, rg, rb = rgb
        painter.setBrush(Qt.BrushStyle.NoBrush)
        arc_span_deg = 68
        arc_starts_deg = (11, 101, 191, 281)
        painter.setPen(glow_pen)
        for start_deg in arc_starts_deg:
            painter.drawArc(er, int(start_deg * 16), int(arc_span_deg * 16))
        painter.setPen(frame_pen)
        for start_deg in arc_starts_deg:
            painter.drawArc(er, int(start_deg * 16), int(arc_span_deg * 16))
        inner_scale = 0.72
        inner_w = er.width() * inner_scale
        inner_h = er.height() * inner_scale
        inner_rect = QRectF(
            er.center().x() - inner_w / 2,
            er.center().y() - inner_h / 2,
            inner_w,
            inner_h,
        )
        painter.setPen(QPen(QColor(rr, rg, rb, 74), 0.8))
        painter.drawEllipse(inner_rect)
        cx = inner_rect.center().x()
        cy = inner_rect.center().y()
        span_x = min(18.0, inner_rect.width() * 0.24)
        span_y = min(18.0, inner_rect.height() * 0.24)
        painter.setPen(QPen(QColor(rr, rg, rb, 128), 0.9))
        painter.drawLine(QPointF(cx - span_x, cy), QPointF(cx + span_x, cy))
        painter.drawLine(QPointF(cx, cy), QPointF(cx, cy + span_y))

    def _render_overlay_layer(self, cr: QRectF) -> QPixmap:
        """Pre-render all detection overlays into a single transparent pixmap."""
        pm = QPixmap(self.width(), self.height())
        pm.fill(QColor(0, 0, 0, 0))
        p = QPainter(pm)
        if self.active_heat:
            if "clouds" in self.thermal_mode:
                self._paint_thermal_cached(p, cr)
        elif self.active_mode or self._highlight_track_id is not None:
            self._paint_active_cached(p, cr)
        p.end()
        return pm

    def _roi_screen_rectf(self, cr: QRectF) -> QRectF:
        x1, y1 = self.roi_norm["x1"], self.roi_norm["y1"]
        x2, y2 = self.roi_norm["x2"], self.roi_norm["y2"]
        px1 = cr.left() + x1 * cr.width()
        py1 = cr.top() + y1 * cr.height()
        px2 = cr.left() + x2 * cr.width()
        py2 = cr.top() + y2 * cr.height()
        return QRectF(px1, py1, px2 - px1, py2 - py1)

    def _draw_handles(self, painter: QPainter, rf: QRectF, circle: bool, rgb: tuple[int, int, int] = (220, 16, 16)) -> None:
        rr, rg, rb = rgb
        painter.setBrush(QBrush(QColor(rr, rg, rb, 54)))
        painter.setPen(QPen(QColor(rr, rg, rb, 132), 0.9))
        hs = 3.0 if circle else 4.0
        corners = [(rf.left(), rf.top()), (rf.right(), rf.top()), (rf.left(), rf.bottom()), (rf.right(), rf.bottom())]
        if circle:
            corners = [corners[0], corners[3]]
        for cx, cy in corners:
            marker = QRectF(cx - hs, cy - hs, 2 * hs, 2 * hs)
            if circle:
                painter.drawEllipse(marker)
            else:
                painter.drawRect(marker)

    def _draw_roi_crosshair(self, painter: QPainter, rf: QRectF, rgb: tuple[int, int, int] = (220, 16, 16)) -> None:
        rr, rg, rb = rgb
        cx = rf.center().x()
        cy = rf.center().y()
        span_x = min(24.0, rf.width() * 0.18)
        span_y = min(24.0, rf.height() * 0.18)
        painter.setPen(QPen(QColor(rr, rg, rb, 140), 1))
        painter.drawLine(QPointF(cx - span_x, cy), QPointF(cx + span_x, cy))
        painter.drawLine(QPointF(cx, cy - span_y), QPointF(cx, cy + span_y))

    def _draw_roi_label(self, painter: QPainter, rf: QRectF, text: str) -> None:
        metrics = painter.fontMetrics()
        label_w = metrics.horizontalAdvance(text) + 14
        label_h = metrics.height() + 6
        rect = QRectF(rf.left(), max(0.0, rf.top() - label_h - 6), label_w, label_h)
        painter.fillRect(rect, QColor(10, 4, 4, 220))
        painter.setPen(QColor(220, 16, 16, 220))
        painter.drawText(rect.adjusted(7, 0, -4, 0), int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), text)

    @staticmethod
    def _draw_quiet_bbox(
        painter: QPainter,
        rect: QRectF,
        *,
        line_alpha: int = 176,
        corner_alpha: int = 128,
        line_width: float = 1.15,
        corner_size: float = 5.0,
    ) -> None:
        # Debug square interior: 60% transparent (40% opacity), scaled by fade.
        fade_scale = max(0.0, min(1.0, float(line_alpha) / 176.0))
        fill_alpha = int(round(255.0 * 0.40 * fade_scale))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(220, 16, 16, fill_alpha)))
        painter.drawRect(rect)

        painter.setPen(QPen(QColor(220, 16, 16, line_alpha), line_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(220, 16, 16, corner_alpha)))
        hs = corner_size / 2.0
        for cx, cy in (
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.left(), rect.bottom()),
            (rect.right(), rect.bottom()),
        ):
            painter.drawRect(QRectF(cx - hs, cy - hs, corner_size, corner_size))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    @staticmethod
    def _draw_diamond_bbox(
        painter: QPainter,
        rect: QRectF,
        *,
        line_alpha: int = 153,
        line_width: float = 1.0,
    ) -> None:
        cx = rect.center().x()
        cy = rect.top() + (rect.height() * 0.35)
        half = max(6.0, min(12.0, min(rect.width(), rect.height()) * 0.16))
        top = QPointF(cx, cy - half)
        right = QPointF(cx + half, cy)
        bottom = QPointF(cx, cy + half)
        left = QPointF(cx - half, cy)
        painter.setPen(QPen(QColor(220, 16, 16, line_alpha), line_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawLine(top, right)
        painter.drawLine(right, bottom)
        painter.drawLine(bottom, left)
        painter.drawLine(left, top)
        # [STYLE] "+" crosshair at diamond center, 50% opacity
        arm = half * 0.4
        painter.setPen(QPen(QColor(220, 16, 16, line_alpha // 2), line_width))
        painter.drawLine(QPointF(cx - arm, cy), QPointF(cx + arm, cy))
        painter.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy + arm))

    @staticmethod
    def _draw_circle_bbox(
        painter: QPainter,
        rect: QRectF,
        *,
        line_alpha: int = 176,
        line_width: float = 1.15,
    ) -> None:
        cx = rect.center().x()
        cy = rect.center().y()
        radius = 5.0
        painter.setPen(QPen(QColor(220, 16, 16, line_alpha), line_width))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))
        arm = 4.0
        painter.setPen(QPen(QColor(220, 16, 16, line_alpha // 2), line_width))
        painter.drawLine(QPointF(cx - arm, cy), QPointF(cx + arm, cy))
        painter.drawLine(QPointF(cx, cy - arm), QPointF(cx, cy + arm))

    def _draw_bbox(
        self,
        painter: QPainter,
        rect: QRectF,
        *,
        line_alpha: int = 176,
        corner_alpha: int = 128,
        line_width: float = 1.15,
        corner_size: float = 5.0,
    ) -> None:
        if self.bbox_style == "diamond":
            self._draw_diamond_bbox(
                painter,
                rect,
                line_alpha=line_alpha,
                line_width=1.0,
            )
            return
        if self.bbox_style == "circle":
            self._draw_circle_bbox(
                painter,
                rect,
                line_alpha=line_alpha,
                line_width=line_width,
            )
            return
        self._draw_quiet_bbox(
            painter,
            rect,
            line_alpha=line_alpha,
            corner_alpha=corner_alpha,
            line_width=line_width,
            corner_size=corner_size,
        )

    def _paint_active_cached(self, painter: QPainter, cr: QRectF) -> None:
        """Render bbox overlays into the given painter (used for cache layer)."""
        font = painter.font()
        font.setBold(True)
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for ov, fade_alpha in self._visible_overlays():
            bn = ov.get("bbox_norm")
            if not bn or len(bn) < 4:
                continue
            bx1 = cr.left() + bn[0] * cr.width()
            by1 = cr.top() + bn[1] * cr.height()
            bx2 = cr.left() + bn[2] * cr.width()
            by2 = cr.top() + bn[3] * cr.height()
            conf = float(ov.get("confidence", 0.5))
            # [ANTI-FLICKER] tier_scale dims grace-period and lower-confidence tracks
            tier_scale = float(ov.get("alpha_scale", 1.0))
            alpha = int(176 * fade_alpha * tier_scale)
            identity = ov.get("identity", "")
            self._draw_bbox(
                painter,
                QRectF(bx1, by1, bx2 - bx1, by2 - by1),
                line_alpha=alpha,
                corner_alpha=int(128 * fade_alpha * tier_scale),
                line_width=1.15,
                corner_size=5.0,
            )
            label = ov.get("label", "")
            icon_side = 12
            icon_pad = 3
            if self.labels_only:
                text = ""
                lw = icon_side + (icon_pad * 2)
                lh = icon_side + (icon_pad * 2)
            else:
                pct = format_confidence_percent(conf)
                if identity and identity != "unknown":
                    text = f"{identity} {pct}"
                else:
                    text = f"{detection_label_text(label)} {pct}"
                lw = metrics.horizontalAdvance(text) + 8
                lh = metrics.height() + 4
            if self.bbox_style == "diamond":
                icon_half = max(6.0, min(12.0, min(bx2 - bx1, by2 - by1) * 0.16))
                center_x = (bx1 + bx2) / 2
                center_y = by1 + ((by2 - by1) * 0.35)
                lx = max(cr.left(), min(cr.right() - lw, center_x - (lw / 2)))
                ly = max(cr.top(), center_y - icon_half - lh - 4)
                painter.fillRect(QRectF(lx, ly, lw, lh), QColor(220, 16, 16, int(210 * fade_alpha * tier_scale)))
                painter.setPen(QColor(255, 255, 255, int(235 * fade_alpha * tier_scale)))
            elif self.bbox_style == "circle":
                circ_cx = (bx1 + bx2) / 2
                circ_cy = (by1 + by2) / 2
                lx = max(cr.left(), min(cr.right() - lw, circ_cx - lw / 2))
                ly = max(cr.top(), circ_cy - 5.0 - lh - 4)
                painter.fillRect(QRectF(lx, ly, lw, lh), QColor(220, 16, 16, int(210 * fade_alpha * tier_scale)))
                painter.setPen(QColor(255, 255, 255, int(235 * fade_alpha * tier_scale)))
            else:
                lx = max(cr.left(), min(cr.right() - lw, bx1))
                ly = max(cr.top(), by1 - lh - 2)
                painter.fillRect(QRectF(lx, ly, lw, lh), QColor(248, 98, 98, int(206 * fade_alpha * tier_scale)))
                painter.setPen(QColor(255, 255, 255, int(235 * fade_alpha * tier_scale)))
            if self.labels_only:
                icon_pm = self._label_icon_pixmap(label, icon_side)
                if not icon_pm.isNull():
                    painter.drawPixmap(int(lx + icon_pad), int(ly + icon_pad), icon_pm)
                else:
                    fallback = detection_label_text(label)[:1].upper()
                    painter.drawText(
                        QRectF(lx, ly, lw, lh),
                        int(Qt.AlignmentFlag.AlignCenter),
                        fallback,
                    )
            else:
                painter.drawText(
                    QRectF(lx + 4, ly, lw - 4, lh),
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                    text,
                )
            if self.bbox_style != "square":
                chev_cx = lx + lw / 2
                chev_top = ly + lh
                target_y = by1
                if self.bbox_style == "diamond":
                    target_y = (by1 + ((by2 - by1) * 0.35)) - max(6.0, min(12.0, min(bx2 - bx1, by2 - by1) * 0.16))
                elif self.bbox_style == "circle":
                    target_y = (by1 + by2) / 2 - 5.0
                chev_bot = min(chev_top + 5.0, target_y)
                if chev_bot > chev_top + 1.0:
                    painter.setPen(QPen(QColor(220, 16, 16, int(150 * fade_alpha * tier_scale)), 1.1))
                    painter.drawLine(QPointF(chev_cx - 3.5, chev_top), QPointF(chev_cx, chev_bot))
                    painter.drawLine(QPointF(chev_cx, chev_bot), QPointF(chev_cx + 3.5, chev_top))
            tid_text = f"T{ov.get('track_id', 0)}"
            tw = metrics.horizontalAdvance(tid_text) + 4
            painter.setPen(QColor(160, 140, 140, int(alpha * 0.5)))
            painter.drawText(
                QRectF(bx2 - tw, by2 + 1, tw, lh),
                int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight),
                tid_text,
            )

    def _paint_thermal_cached(self, painter: QPainter, cr: QRectF) -> None:
        """Render thermal heatmap using pre-rendered blob templates (used for cache layer)."""
        font = painter.font()
        font.setBold(True)
        font.setPointSize(8)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for ov, fade_alpha in self._visible_overlays():
            bn = ov.get("bbox_norm")
            if not bn or len(bn) < 4:
                continue
            label = ov.get("label", "")
            cat = heatmap_category(label)
            blob = self._thermal_blobs.get(cat)
            core = self._thermal_cores.get(cat)
            bx1 = cr.left() + bn[0] * cr.width()
            by1 = cr.top() + bn[1] * cr.height()
            bx2 = cr.left() + bn[2] * cr.width()
            by2 = cr.top() + bn[3] * cr.height()
            bw = bx2 - bx1
            bh = by2 - by1
            cx = (bx1 + bx2) / 2
            cy = (by1 + by2) / 2
            rad = max(bw, bh) * 0.30
            conf = float(ov.get("confidence", 0.5))
            # [ANTI-FLICKER] tier_scale dims grace-period tracks in heatmap view too
            tier_scale = float(ov.get("alpha_scale", 1.0))
            op = 0.35 + conf * 0.55
            # Blit pre-rendered blob scaled to detection size
            d = int(rad * 2)
            if d > 0 and blob is not None:
                painter.setOpacity(op * fade_alpha * tier_scale)
                target = QRectF(cx - rad, cy - rad, d, d)
                painter.drawPixmap(target.toRect(), blob)
                # Core bloom
                core_d = int(rad * 0.60)
                if core_d > 0 and core is not None:
                    ct = QRectF(cx - core_d / 2, cy - core_d / 2, core_d, core_d)
                    painter.drawPixmap(ct.toRect(), core)
                painter.setOpacity(1.0)
            if not self.show_heat_labels:
                continue
            # Label plate
            icon_side = 12
            icon_pad = 3
            if self.labels_only:
                text = ""
                lw = icon_side + (icon_pad * 2)
                lh = icon_side + (icon_pad * 2)
            else:
                identity = ov.get("identity", "")
                pct = format_confidence_percent(conf)
                if identity and identity != "unknown":
                    text = f"{identity} {pct}"
                else:
                    text = f"{detection_label_text(label)} {pct}"
                lw = metrics.horizontalAdvance(text) + 8
                lh = metrics.height() + 4
            lx = max(cr.left(), min(cr.right() - lw, bx1))
            ly = max(cr.top(), by1 - lh - 2)
            painter.fillRect(QRectF(lx, ly, lw, lh), QColor(248, 98, 98, int(206 * fade_alpha * tier_scale)))
            alpha = int(255 * op * fade_alpha * tier_scale)
            painter.setPen(QColor(255, 255, 255, min(255, alpha + 30)))
            if self.labels_only:
                icon_pm = self._label_icon_pixmap(label, icon_side)
                if not icon_pm.isNull():
                    painter.drawPixmap(int(lx + icon_pad), int(ly + icon_pad), icon_pm)
                else:
                    fallback = detection_label_text(label)[:1].upper()
                    painter.drawText(
                        QRectF(lx, ly, lw, lh),
                        int(Qt.AlignmentFlag.AlignCenter),
                        fallback,
                    )
            else:
                painter.drawText(
                    QRectF(lx + 4, ly, lw - 4, lh),
                    int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                    text,
                )

    def _hit_handle(self, pos: QPoint, cr: QRectF) -> Optional[str]:
        if not self.roi_active:
            return None
        rf = self._roi_screen_rectf(cr)
        side = min(rf.width(), rf.height())
        cx = rf.center().x()
        cy = rf.center().y()
        rf = QRectF(cx - side / 2, cy - side / 2, side, side)
        pts = [("tl", QPoint(int(rf.left()), int(rf.top()))), ("br", QPoint(int(rf.right()), int(rf.bottom())))]
        for name, pt in pts:
            d = ((pos.x() - pt.x()) ** 2 + (pos.y() - pt.y()) ** 2) ** 0.5
            if d <= self._HANDLE_HIT:
                return name
        if rf.contains(QPointF(pos)):
            return "move"
        return None

    def _start_monocle_flash(self) -> None:
        self._monocle_flash = 1.0
        self._monocle_timer.start()

    def _tick_monocle(self) -> None:
        self._monocle_flash -= 0.07
        if self._monocle_flash <= 0.0:
            self._monocle_flash = 0.0
            self._monocle_timer.stop()
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        if self.roi_active and event.button() == Qt.MouseButton.LeftButton:
            cr = self._content_rect()
            if self._roi_screen_rectf(cr).contains(event.position()):
                self._drag = None
                self._start_monocle_flash()
                self.roi_capture_requested.emit()
        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return
        cr = self._content_rect()
        if self.roi_active:
            mode = self._hit_handle(event.pos(), cr)
            if mode is None:
                super().mousePressEvent(event)
                return
            self._drag = {"mode": mode, "start": QPoint(event.pos()), "orig": dict(self.roi_norm)}
            if mode != "move":
                self.grabMouse()
            super().mousePressEvent(event)
            return
        track_id = self._hit_overlay_track(event.pos(), cr)
        if track_id:
            self.track_clicked.emit(track_id)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if not self.roi_active or self._drag is None:
            return
        cr = self._content_rect()
        dx = (event.pos().x() - self._drag["start"].x()) / cr.width() if cr.width() > 0 else 0
        dy = (event.pos().y() - self._drag["start"].y()) / cr.height() if cr.height() > 0 else 0
        o = self._drag["orig"]
        mode = self._drag["mode"]
        mn = 0.05

        def clamp_box(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            if x2 - x1 < mn:
                x2 = min(1.0, x1 + mn)
            if y2 - y1 < mn:
                y2 = min(1.0, y1 + mn)
            x1 = max(0.0, min(1.0 - mn, x1))
            y1 = max(0.0, min(1.0 - mn, y1))
            x2 = max(x1 + mn, min(1.0, x2))
            y2 = max(y1 + mn, min(1.0, y2))
            return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

        if mode == "move":
            w, h = o["x2"] - o["x1"], o["y2"] - o["y1"]
            nx1 = max(0.0, min(1.0 - w, o["x1"] + dx))
            ny1 = max(0.0, min(1.0 - h, o["y1"] + dy))
            self.roi_norm = {"x1": nx1, "y1": ny1, "x2": nx1 + w, "y2": ny1 + h}
        else:
            x1, y1, x2, y2 = o["x1"], o["y1"], o["x2"], o["y2"]
            if "l" in mode:
                x1 = o["x1"] + dx
            if "r" in mode:
                x2 = o["x2"] + dx
            if "t" in mode:
                y1 = o["y1"] + dy
            if "b" in mode:
                y2 = o["y2"] + dy
            nn = clamp_box(x1, y1, x2, y2)
            side = min(nn["x2"] - nn["x1"], nn["y2"] - nn["y1"])
            cx = (nn["x1"] + nn["x2"]) / 2
            cy = (nn["y1"] + nn["y2"]) / 2
            half = max(mn / 2, side / 2)
            nn = clamp_box(cx - half, cy - half, cx + half, cy + half)
            self.roi_norm = nn
        self.update()
        self.roi_norm_changed.emit(dict(self.roi_norm))

    def mouseReleaseEvent(self, event) -> None:
        if self._drag is not None and self._drag.get("mode") != "move":
            self.releaseMouse()
        self._drag = None
        super().mouseReleaseEvent(event)

    def apply_roi_scale(self, scale: float) -> None:
        cx = (self.roi_norm["x1"] + self.roi_norm["x2"]) / 2
        cy = (self.roi_norm["y1"] + self.roi_norm["y2"]) / 2
        half = self.ROI_BASE * scale
        x1 = max(0.0, cx - half)
        y1 = max(0.0, cy - half)
        x2 = min(1.0, cx + half)
        y2 = min(1.0, cy + half)
        w, h = x2 - x1, y2 - y1
        if x1 == 0:
            x2 = w
        if y1 == 0:
            y2 = h
        if x2 == 1:
            x1 = 1 - w
        if y2 == 1:
            y1 = 1 - h
        self.roi_norm = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        self.roi_scale_preset = float(scale)
        self.update()
        self.roi_norm_changed.emit(dict(self.roi_norm))

    def center_roi(self) -> None:
        w = self.roi_norm["x2"] - self.roi_norm["x1"]
        h = self.roi_norm["y2"] - self.roi_norm["y1"]
        cx, cy = 0.5, 0.5
        x1 = max(0.0, min(1.0 - w, cx - w / 2))
        y1 = max(0.0, min(1.0 - h, cy - h / 2))
        self.roi_norm = {"x1": x1, "y1": y1, "x2": x1 + w, "y2": y1 + h}
        self.update()
        self.roi_norm_changed.emit(dict(self.roi_norm))


class FocusOverlay(QWidget):
    quick_box_changed = pyqtSignal(object)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.scan_results: list[dict[str, Any]] = []
        self.show_boxes = False
        self.show_heatmap = False
        self.show_heat_labels = False
        self.labels_only = False
        self.heatmap_opacity = 0.55
        self.bbox_style = "square"
        self._cache: Optional[QPixmap] = None
        self._cache_size: tuple[int, int] = (0, 0)
        self._quick_box_pct: Optional[tuple[float, float, float, float]] = None
        self._quick_draw_mode = False
        self._quick_drawing = False
        self._quick_start = QPointF()
        self._quick_current = QPointF()

    def set_scan(self, scan: Optional[list], boxes: bool, heat: bool, op: float, heat_labels: bool = False) -> None:
        self.scan_results = list(scan or [])
        self.show_boxes = boxes
        self.show_heatmap = heat
        self.show_heat_labels = bool(heat_labels)
        self.heatmap_opacity = op
        self._cache = None
        self.update()

    def set_quick_box(self, bbox_pct: Optional[tuple[float, float, float, float]]) -> None:
        self._quick_box_pct = bbox_pct
        self._cache = None
        self.update()

    def quick_box(self) -> Optional[tuple[float, float, float, float]]:
        return self._quick_box_pct

    def set_quick_draw_mode(self, enabled: bool) -> None:
        self._quick_draw_mode = bool(enabled)
        self._quick_drawing = False
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, not self._quick_draw_mode)
        self.setCursor(Qt.CursorShape.CrossCursor if self._quick_draw_mode else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_bbox_style(self, style: str) -> None:
        s = str(style).strip().lower()
        next_style = s if s in ("diamond", "circle") else "square"
        if self.bbox_style == next_style:
            return
        self.bbox_style = next_style
        self._cache = None
        self.update()

    def set_labels_only(self, enabled: bool) -> None:
        next_enabled = bool(enabled)
        if self.labels_only == next_enabled:
            return
        self.labels_only = next_enabled
        self._cache = None
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._cache = None

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if not self._quick_draw_mode or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        self._quick_drawing = True
        self._quick_start = QPointF(event.position())
        self._quick_current = QPointF(event.position())
        self._cache = None
        self.update()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not self._quick_draw_mode or not self._quick_drawing:
            return super().mouseMoveEvent(event)
        self._quick_current = QPointF(event.position())
        self._cache = None
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if not self._quick_draw_mode or event.button() != Qt.MouseButton.LeftButton:
            return super().mouseReleaseEvent(event)
        if not self._quick_drawing:
            event.accept()
            return
        self._quick_drawing = False
        self._quick_current = QPointF(event.position())
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            event.accept()
            return
        x1 = max(0.0, min(float(w), min(self._quick_start.x(), self._quick_current.x())))
        y1 = max(0.0, min(float(h), min(self._quick_start.y(), self._quick_current.y())))
        x2 = max(0.0, min(float(w), max(self._quick_start.x(), self._quick_current.x())))
        y2 = max(0.0, min(float(h), max(self._quick_start.y(), self._quick_current.y())))
        if (x2 - x1) < 4.0 or (y2 - y1) < 4.0:
            event.accept()
            self._cache = None
            self.update()
            return
        bbox = (x1 / float(w), y1 / float(h), x2 / float(w), y2 / float(h))
        self._quick_box_pct = bbox
        self.quick_box_changed.emit(bbox)
        self._cache = None
        self.update()
        event.accept()

    def paintEvent(self, event) -> None:
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return
        if self._cache is not None and self._cache_size == (w, h):
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self._cache)
            painter.end()
            return
        pm = QPixmap(w, h)
        pm.fill(QColor(0, 0, 0, 0))
        painter = QPainter(pm)
        if self.show_heatmap:
            painter.setClipRect(0, 0, w, h)
            painter.setPen(Qt.PenStyle.NoPen)
            for item in self.scan_results:
                bp = item.get("bbox_pct")
                if not bp or len(bp) < 4:
                    continue
                cat = heatmap_category(item.get("label", ""))
                blob = VideoPane._thermal_blobs.get(cat)
                core = VideoPane._thermal_cores.get(cat)
                bx1, by1, bx2, by2 = bp[0] / 100 * w, bp[1] / 100 * h, bp[2] / 100 * w, bp[3] / 100 * h
                cx = (bx1 + bx2) / 2
                cy = (by1 + by2) / 2
                conf = float(item.get("confidence", 0.5))
                bw = max(2.0, bx2 - bx1)
                bh = max(2.0, by2 - by1)
                rad = max(bw, bh) * (0.25 + conf * 0.20) * 0.4
                op = self.heatmap_opacity
                d = int(rad * 2)
                if d > 0 and blob is not None:
                    painter.setOpacity(op)
                    painter.drawPixmap(QRect(int(cx - rad), int(cy - rad), d, d), blob)
                    core_d = int(rad * 0.60)
                    if core_d > 0 and core is not None:
                        painter.drawPixmap(QRect(int(cx - core_d / 2), int(cy - core_d / 2), core_d, core_d), core)
                    painter.setOpacity(1.0)
        if self.show_boxes:
            font = painter.font()
            font.setBold(True)
            font.setPointSize(9)
            painter.setFont(font)
            for item in self.scan_results:
                bp = item.get("bbox_pct")
                if not bp or len(bp) < 4:
                    continue
                bx1, by1, bx2, by2 = bp[0] / 100 * w, bp[1] / 100 * h, bp[2] / 100 * w, bp[3] / 100 * h
                conf = float(item.get("confidence", 0.5))
                alpha = 176
                rect = QRectF(bx1, by1, bx2 - bx1, by2 - by1)
                if self.bbox_style == "diamond":
                    VideoPane._draw_diamond_bbox(
                        painter,
                        rect,
                        line_alpha=153,
                        line_width=1.0,
                    )
                elif self.bbox_style == "circle":
                    VideoPane._draw_circle_bbox(
                        painter,
                        rect,
                        line_alpha=alpha,
                        line_width=1.15,
                    )
                else:
                    VideoPane._draw_quiet_bbox(
                        painter,
                        rect,
                        line_alpha=alpha,
                        corner_alpha=128,
                        line_width=1.15,
                        corner_size=5.0,
                    )
                if not self.show_heatmap or self.show_heat_labels:
                    raw_label = item.get("label", "")
                    metrics = painter.fontMetrics()
                    icon_side = 12
                    icon_pad = 3
                    if self.labels_only:
                        label = ""
                        label_w = icon_side + (icon_pad * 2)
                        label_h = icon_side + (icon_pad * 2)
                    else:
                        label = f"{detection_label_text(raw_label)} {format_confidence_percent(conf)}".strip()
                        label_w = metrics.horizontalAdvance(label) + 6
                        label_h = metrics.height() + 2
                    if self.bbox_style == "diamond":
                        icon_half = max(6.0, min(12.0, min(bx2 - bx1, by2 - by1) * 0.16))
                        center_x = (bx1 + bx2) / 2
                        center_y = by1 + ((by2 - by1) * 0.35)
                        label_x = max(0.0, min(float(w - label_w), center_x - (label_w / 2)))
                        label_y = max(0.0, center_y - icon_half - label_h - 4)
                        painter.fillRect(QRectF(label_x, label_y, label_w, label_h), QColor(220, 16, 16, 210))
                        painter.setPen(QColor(255, 255, 255, 235))
                    elif self.bbox_style == "circle":
                        circ_cx = (bx1 + bx2) / 2
                        circ_cy = (by1 + by2) / 2
                        label_x = max(0.0, min(float(w - label_w), circ_cx - label_w / 2))
                        label_y = max(0.0, circ_cy - 5.0 - label_h - 4)
                        painter.fillRect(QRectF(label_x, label_y, label_w, label_h), QColor(220, 16, 16, 210))
                        painter.setPen(QColor(255, 255, 255, 235))
                    else:
                        label_x = max(0.0, min(float(w - label_w), bx1))
                        label_y = max(0.0, by1 - label_h)
                        painter.fillRect(QRectF(label_x, label_y, label_w, label_h), QColor(248, 98, 98, 206))
                        painter.setPen(QColor(255, 255, 255, 235))
                    if self.labels_only:
                        icon_pm = VideoPane._label_icon_pixmap(raw_label, icon_side)
                        if not icon_pm.isNull():
                            painter.drawPixmap(int(label_x + icon_pad), int(label_y + icon_pad), icon_pm)
                        else:
                            fallback = detection_label_text(raw_label)[:1].upper()
                            painter.drawText(
                                QRectF(label_x, label_y, label_w, label_h),
                                int(Qt.AlignmentFlag.AlignCenter),
                                fallback,
                            )
                    else:
                        painter.drawText(
                            QRectF(label_x + 3, label_y, label_w - 3, label_h),
                            int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft),
                            label,
                        )
                    if self.bbox_style != "square":
                        chev_cx = label_x + label_w / 2
                        chev_top = label_y + label_h
                        target_y = by1
                        if self.bbox_style == "diamond":
                            target_y = (by1 + ((by2 - by1) * 0.35)) - max(6.0, min(12.0, min(bx2 - bx1, by2 - by1) * 0.16))
                        elif self.bbox_style == "circle":
                            target_y = (by1 + by2) / 2 - 5.0
                        chev_bot = min(chev_top + 6.0, target_y)
                        chev_hw = 4.0
                        if chev_bot > chev_top + 1.0:
                            painter.setPen(QPen(QColor(220, 16, 16, 150), 1.1))
                            painter.drawLine(QPointF(chev_cx - chev_hw, chev_top), QPointF(chev_cx, chev_bot))
                            painter.drawLine(QPointF(chev_cx, chev_bot), QPointF(chev_cx + chev_hw, chev_top))
        # Quick-label annotation box (single-face fast path).
        quick = self._quick_box_pct
        if quick is not None:
            qx1 = quick[0] * float(w)
            qy1 = quick[1] * float(h)
            qx2 = quick[2] * float(w)
            qy2 = quick[3] * float(h)
            rect = QRectF(qx1, qy1, max(1.0, qx2 - qx1), max(1.0, qy2 - qy1))
            painter.setPen(QPen(QColor(255, 208, 96, 230), 2.0))
            painter.setBrush(QColor(255, 208, 96, 24))
            painter.drawRect(rect)
            painter.setPen(QColor(255, 233, 190, 240))
            painter.setBrush(QColor(255, 208, 96, 220))
            tag_w = 74.0
            tag_h = 16.0
            tag_rect = QRectF(
                max(0.0, min(float(w - tag_w), rect.left())),
                max(0.0, rect.top() - tag_h),
                tag_w,
                tag_h,
            )
            painter.drawRect(tag_rect)
            painter.setPen(QColor(255, 255, 255, 235))
            painter.drawText(tag_rect, int(Qt.AlignmentFlag.AlignCenter), "FACE BOX")
        if self._quick_draw_mode and self._quick_drawing:
            x1 = max(0.0, min(float(w), min(self._quick_start.x(), self._quick_current.x())))
            y1 = max(0.0, min(float(h), min(self._quick_start.y(), self._quick_current.y())))
            x2 = max(0.0, min(float(w), max(self._quick_start.x(), self._quick_current.x())))
            y2 = max(0.0, min(float(h), max(self._quick_start.y(), self._quick_current.y())))
            preview_rect = QRectF(x1, y1, max(1.0, x2 - x1), max(1.0, y2 - y1))
            dash_pen = QPen(QColor(255, 208, 96, 220), 1.6)
            dash_pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(dash_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(preview_rect)
        painter.end()
        self._cache = pm
        self._cache_size = (w, h)
        screen_painter = QPainter(self)
        screen_painter.drawPixmap(0, 0, pm)
        screen_painter.end()


class FocusImageComposite(QWidget):
    """Base crop with silhouette blended (screen, web parity)."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._base = QPixmap()
        self._sil = QPixmap()
        self._zoom = 1.0
        # Scaled pixmap cache
        self._scaled_cache: Optional[QPixmap] = None
        self._scaled_key: tuple = ()
        self._bg_cache: Optional[QPixmap] = None
        self._bg_cache_size: tuple[int, int] = (0, 0)

    def set_pixmaps(self, base: QPixmap, sil: QPixmap) -> None:
        self._base = QPixmap(base)
        self._sil = QPixmap(sil)
        self._scaled_cache = None
        self.update()

    def set_zoom(self, z: float) -> None:
        self._zoom = max(1.0, min(3.5, z))
        self._scaled_cache = None
        self.update()

    def clear_images(self) -> None:
        self._base = QPixmap()
        self._sil = QPixmap()
        self._scaled_cache = None
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._scaled_cache = None
        self._bg_cache = None

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        br = self.rect()
        tw, th = br.width(), br.height()
        # Cached background gradient
        if self._bg_cache is None or self._bg_cache_size != (tw, th):
            bg = QPixmap(tw, th)
            bgp = QPainter(bg)
            panel_grad = QLinearGradient(0, 0, 0, th)
            panel_grad.setColorAt(0.0, QColor(220, 16, 16, 45))
            panel_grad.setColorAt(1.0, QColor(140, 8, 8, 20))
            bgp.fillRect(0, 0, tw, th, panel_grad)
            bgp.setPen(QPen(QColor(160, 20, 20, 110), 1))
            bgp.drawRect(0, 0, tw - 1, th - 1)
            bgp.end()
            self._bg_cache = bg
            self._bg_cache_size = (tw, th)
        painter.drawPixmap(0, 0, self._bg_cache)
        if tw <= 0 or th <= 0 or self._base.isNull():
            painter.end()
            return
        bw, bh = self._base.width(), self._base.height()
        if bw <= 0 or bh <= 0:
            painter.end()
            return
        # Cache the composited (base + silhouette) scaled pixmap
        key = (tw, th, bw, bh, self._zoom, self._sil.cacheKey() if not self._sil.isNull() else 0)
        if self._scaled_cache is None or self._scaled_key != key:
            cover = max(tw / bw, th / bh)
            scale = min(cover * self._zoom, cover * 3.5)
            nw = max(1, int(bw * scale))
            nh = max(1, int(bh * scale))
            scaled_b = self._base.scaled(nw, nh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            if not self._sil.isNull():
                comp = QPixmap(scaled_b.size())
                comp.fill(QColor(0, 0, 0, 0))
                cp = QPainter(comp)
                cp.drawPixmap(0, 0, scaled_b)
                scaled_s = self._sil.scaled(
                    scaled_b.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                sx = (scaled_b.width() - scaled_s.width()) // 2
                sy = (scaled_b.height() - scaled_s.height()) // 2
                cp.setCompositionMode(QPainter.CompositionMode.CompositionMode_Screen)
                cp.setOpacity(0.92)
                cp.drawPixmap(sx, sy, scaled_s)
                cp.end()
                self._scaled_cache = comp
            else:
                self._scaled_cache = scaled_b
            self._scaled_key = key
        ox = br.left() + (tw - self._scaled_cache.width()) // 2
        oy = br.top() + (th - self._scaled_cache.height()) // 2
        painter.setClipRect(br)
        painter.drawPixmap(ox, oy, self._scaled_cache)
        painter.setPen(QPen(QColor(220, 16, 16, 52), 1))
        painter.drawRect(br.adjusted(0, 0, -1, -1))
        painter.end()


class ClickFrame(QFrame):
    clicked = pyqtSignal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class PreviewTileWidget(ClickFrame):
    _FADE_MS = 95
    _ANIMATE = os.environ.get("INSIGHT_PREVIEW_TILE_ANIM", "0") == "1"
    _LIVE_IMAGE_MIN_INTERVAL_SEC = 0.35

    def __init__(self) -> None:
        super().__init__()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame {{ background: transparent; border: none; "
            f"border-bottom: 1px solid {theme_rgba('accent_dark', 0.20)}; }}"
        )
        self._signature: tuple[Any, ...] = ()
        self._image_b64 = ""
        self._last_image_update_ts = 0.0
        self._preview_state = "live"
        self._on_open: Optional[Callable[[], None]] = None
        self._on_add: Optional[Callable[[], None]] = None
        self._fade_anim: Optional[QPropertyAnimation] = None
        self._is_retiring = False
        self.clicked.connect(self._handle_open)

        hl = QHBoxLayout(self)
        hl.setContentsMargins(6, 6, 6, 6)
        hl.setSpacing(8)

        self._thumb = QLabel()
        self._thumb.setMinimumSize(48, 48)
        self._thumb.setMaximumSize(96, 96)
        self._thumb.setScaledContents(True)
        self._thumb.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        self._thumb.setStyleSheet(
            f"border: 1px solid {theme_rgba('accent_dark', 0.45)}; background: transparent;"
        )
        hl.addWidget(self._thumb)

        meta = QVBoxLayout()
        meta.setSpacing(6)
        hl.addLayout(meta, stretch=1)

        title_r = QHBoxLayout()
        self._title = QLabel()
        self._title.setTextFormat(Qt.TextFormat.RichText)
        self._title.setStyleSheet(
            f"QLabel {{ background: {theme_rgba('panel', 0.25)}; border: 1px solid {theme_rgba('accent_dark', 0.45)}; padding: 2px 8px; }}"
        )
        title_r.addWidget(self._title)
        title_r.addStretch(1)
        self._track = QLabel()
        self._track.setStyleSheet(
            f"font-size: 10px; color: {text_css(0.70)}; background: {theme_rgba('panel', 0.25)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.45)}; padding: 1px 5px;"
        )
        title_r.addWidget(self._track)
        self._add_btn = QPushButton("Add")
        self._add_btn.setToolTip("Add this crop to a CV Ops dataset")
        self._add_btn.setFixedHeight(22)
        self._add_btn.setStyleSheet(
            f"QPushButton {{ background: {theme_rgba('panel', 0.32)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.45)}; "
            f"color: {text_css(0.92)}; font-size: 10px; padding: 1px 7px; }}"
            f"QPushButton:hover {{ background: {theme_rgba('accent_dark', 0.18)}; }}"
        )
        self._add_btn.clicked.connect(lambda _checked=False: self._handle_add())
        title_r.addWidget(self._add_btn)
        meta.addLayout(title_r)

        self._pill = QLabel()
        self._pill.setStyleSheet(
            f"QLabel {{ background: {theme_rgba('panel', 0.25)}; border: 1px solid {theme_rgba('accent_dark', 0.45)}; "
            "color: #140808; font-size: 10px; letter-spacing: 1.2px; padding: 2px 8px; }"
        )
        meta.addWidget(self._pill, alignment=Qt.AlignmentFlag.AlignLeft)

        stats_w = QWidget()
        stats_w.setStyleSheet(
            f"QWidget {{ border-top: 1px solid {theme_rgba('accent_dark', 0.20)}; padding-top: 4px; }}"
        )
        self._stats_grid = QGridLayout(stats_w)
        self._stats_grid.setContentsMargins(0, 4, 0, 0)
        self._stats_grid.setHorizontalSpacing(4)
        self._stats_grid.setVerticalSpacing(4)
        self._stat_values: dict[str, QLabel] = {}
        for idx, label in enumerate(("Confidence", "Motion", "Age", "Score")):
            cell = QWidget()
            cell.setStyleSheet(
                f"background: {theme_rgba('panel', 0.25)}; border: 1px solid {theme_rgba('accent_dark', 0.45)}; padding: 2px 4px;"
            )
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(2, 2, 2, 2)
            cl.setSpacing(1)
            k = QLabel(label.upper())
            k.setStyleSheet(f"font-size: 9px; letter-spacing: 1.2px; color: {text_css(0.52)};")
            v = QLabel("")
            v.setStyleSheet(f"font-size: 11px; color: {text_css(1.0)};")
            cl.addWidget(k)
            cl.addWidget(v)
            self._stats_grid.addWidget(cell, idx // 2, idx % 2)
            self._stat_values[label] = v
        meta.addWidget(stats_w)

        self._recog_pill = QLabel()
        self._recog_pill.setStyleSheet(
            f"QLabel {{ background: rgba(255,208,96,0.66); border: 1px solid {theme_rgba('accent_dark', 0.35)}; "
            "color: #140808; font-size: 10px; letter-spacing: 1px; padding: 2px 8px; }"
        )
        self._recog_pill.hide()
        meta.addWidget(self._recog_pill, alignment=Qt.AlignmentFlag.AlignLeft)

        for lab in self.findChildren(QLabel):
            lab.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

    def _handle_open(self) -> None:
        if self._on_open is not None:
            self._on_open()

    def _handle_add(self) -> None:
        if self._on_add is not None:
            self._on_add()

    def _run_opacity(self, start: float, end: float, duration_ms: int, on_done: Optional[Callable[[], None]] = None) -> None:
        if not self._ANIMATE:
            self.setGraphicsEffect(None)
            if end <= 0.0:
                self.hide()
            else:
                self.show()
            if on_done is not None:
                on_done()
            return
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim.deleteLater()
            self._fade_anim = None
        # Create a fresh effect for this animation — applying setGraphicsEffect
        # replaces (and deletes) any previous one owned by Qt.
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(start)
        self.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(duration_ms)
        anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        anim.setStartValue(start)
        anim.setEndValue(end)

        def _finished(e=effect) -> None:
            self._fade_anim = None
            e.setOpacity(end)
            if end >= 1.0:
                # Remove the effect once fully opaque so the tile renders
                # directly without an offscreen pass — fixes Retina alignment drift.
                self.setGraphicsEffect(None)
            if on_done is not None:
                on_done()

        anim.finished.connect(_finished)
        self._fade_anim = anim
        anim.start()

    def soften_show(self) -> None:
        self._is_retiring = False
        self.show()
        self._run_opacity(0.34, 1.0, self._FADE_MS)

    def soften_refresh(self) -> None:
        if self._is_retiring or not self.isVisible():
            return
        eff = self.graphicsEffect()
        current = float(eff.opacity()) if isinstance(eff, QGraphicsOpacityEffect) else 1.0
        self._run_opacity(min(current, 0.72), 1.0, self._FADE_MS - 10)

    def soften_hide(self) -> None:
        if self._is_retiring or not self.isVisible():
            self.hide()
            self.setGraphicsEffect(None)
            return
        self._is_retiring = True

        def _done() -> None:
            self.hide()
            self.setGraphicsEffect(None)
            self._is_retiring = False

        eff = self.graphicsEffect()
        start = float(eff.opacity()) if isinstance(eff, QGraphicsOpacityEffect) else 1.0
        self._run_opacity(start, 0.0, self._FADE_MS, _done)

    def soften_remove(self, on_done: Callable[[], None]) -> None:
        if self._is_retiring:
            return
        self._is_retiring = True

        def _done() -> None:
            self.setGraphicsEffect(None)
            self._is_retiring = False
            on_done()

        eff = self.graphicsEffect()
        start = float(eff.opacity()) if isinstance(eff, QGraphicsOpacityEffect) else 1.0
        self._run_opacity(start, 0.0, self._FADE_MS, _done)

    def dispose(self) -> None:
        if self._fade_anim is not None:
            self._fade_anim.stop()
            self._fade_anim.deleteLater()
            self._fade_anim = None
        self.setGraphicsEffect(None)

    @staticmethod
    def _signature_for(tile: dict[str, Any]) -> tuple[Any, ...]:
        image = str(tile.get("image", ""))
        return (
            str(tile.get("preview_state", "live")),
            int(tile.get("track_id", 0)),
            str(tile.get("label", "")),
            round(float(tile.get("confidence", 0.0)), 3),
            round(float(tile.get("motion_score", 0.0)), 3),
            round(float(tile.get("age_seconds", 0.0)), 1),
            str(tile.get("event_tag", "")),
            round(float(tile.get("score", 0.0)), 3),
            len(image),
            image[:24],
            image[-24:] if image else "",
            str(tile.get("recognized_identity", "") or ""),
            round(float(tile.get("recognition_confidence", 0.0)), 3),
        )

    _CAT_HEX: dict[str, str] = {
        "human": "#ffa03c", "plant": "#3cd23c", "animal": "#00c8be",
        "inorganic": "#3c8cff", "tech": "#aa50ff",
    }
    _CAT_ICON: dict[str, str] = {
        "human": "people.svg", "animal": "animals.svg",
        "tech": "tech.svg", "inorganic": "objects.svg", "plant": "objects.svg",
    }
    _CAT_HEX_THERMAL: dict[str, str] = {
        "human": "#ff5028", "plant": "#3cd250", "animal": "#00c8be",
        "inorganic": "#ff963c", "tech": "#ff50b4",
    }

    def update_tile(
        self,
        tile: dict[str, Any],
        on_open: Callable[[], None],
        *,
        on_add: Optional[Callable[[], None]] = None,
        thermal: bool = False,
    ) -> None:
        self._on_open = on_open
        self._on_add = on_add
        signature = (*self._signature_for(tile), thermal)
        if signature == self._signature:
            return
        had_signature = bool(self._signature)
        previous_state = self._preview_state
        self._signature = signature

        image_b64 = str(tile.get("image", ""))
        image_changed = image_b64 != self._image_b64
        preview_state = str(tile.get("preview_state", "live") or "live")
        if not image_b64:
            self._image_b64 = ""
            label = detection_label_text(tile.get("label", ""))
            self._thumb.clear()
            self._thumb.setText(f"ROI\n{label.upper()}" if preview_state == "live" else label.upper())
            self._thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._thumb.setStyleSheet(
                f"border: 1px solid {theme_rgba('accent_dark', 0.38)}; "
                f"background: {theme_rgba('panel', 0.20)}; "
                f"color: {text_css(0.82)}; font-size: 10px; font-weight: 700;"
            )
        elif image_b64 != self._image_b64:
            now = time.monotonic()
            should_update_image = (
                not self._image_b64
                or preview_state != "live"
                or now - self._last_image_update_ts >= self._LIVE_IMAGE_MIN_INTERVAL_SEC
            )
            if should_update_image:
                try:
                    pixmap = pixmap_from_b64_jpeg(image_b64)
                except Exception:
                    pixmap = QPixmap()
                if not pixmap.isNull():
                    self._image_b64 = image_b64
                    self._last_image_update_ts = now
                    self._thumb.setText("")
                    self._thumb.setPixmap(
                        pixmap.scaled(
                            96,
                            96,
                            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    )
                elif not self._image_b64:
                    self._thumb.clear()
                    self._thumb.setText("NO\nPREVIEW")

        hex_map = self._CAT_HEX_THERMAL if thermal else self._CAT_HEX
        _chx = hex_map.get(heatmap_category(tile.get("label", "")), "#3c8cff")
        self._preview_state = preview_state
        is_past = preview_state == "past"
        self.setStyleSheet(
            f"QFrame {{ background: transparent; border: none; "
            f"border-bottom: 1px solid {theme_rgba('accent_dark', 0.12) if is_past else theme_rgba('accent_dark', 0.20)}; }}"
        )
        if image_b64:
            self._thumb.setStyleSheet(
                f"border: 1px solid {theme_rgba('accent_dark', 0.26) if is_past else theme_rgba('accent_dark', 0.45)}; "
                "background: transparent;"
            )
        cat = heatmap_category(tile.get("label", ""))
        icon_file = self._CAT_ICON.get(cat, "objects.svg")
        icon_path = ASSETS_DIR / "icons" / icon_file
        tile_label = detection_label_text(tile.get("label", ""))
        icon_html = f"<img src='{icon_path}' width='12' height='12' style='vertical-align:middle;margin-right:4px;'>" if icon_path.exists() else ""
        self._title.setStyleSheet(
            f"QLabel {{ background: {theme_rgba('panel', 0.25)}; border: 1px solid {theme_rgba('accent_dark', 0.45)}; padding: 2px 8px; }}"
        )
        self._title.setText(
            f"{icon_html}<b style='color:{text_hex()}'>{tile_label}</b>"
        )
        track_text = f"T{tile.get('track_id')}"
        if is_past:
            track_text = f"{track_text}  {float(tile.get('age_seconds', 0.0)):.0f}s ago"
        self._track.setText(track_text)
        pill_text = str(tile.get("event_tag", "")).upper()
        if is_past:
            pill_text = f"PAST  {pill_text}".strip()
            self._pill.setStyleSheet(
                f"QLabel {{ background: {theme_rgba('panel', 0.25)}; border: 1px solid {theme_rgba('accent_dark', 0.25)}; "
                "color: #140808; font-size: 10px; letter-spacing: 1.2px; padding: 2px 8px; }"
            )
        else:
            self._pill.setStyleSheet(
                f"QLabel {{ background: {theme_rgba('panel', 0.25)}; border: 1px solid {theme_rgba('accent_dark', 0.45)}; "
                "color: #140808; font-size: 10px; letter-spacing: 1.2px; padding: 2px 8px; }"
            )
        self._pill.setText(pill_text)
        self._stat_values["Confidence"].setText(format_confidence_percent(tile.get("confidence", 0)))
        self._stat_values["Motion"].setText(f"{round(float(tile.get('motion_score', 0)) * 100)}%")
        self._stat_values["Age"].setText(f"{float(tile.get('age_seconds', 0)):.1f}s")
        self._stat_values["Score"].setText(f"{float(tile.get('score', 0)):.2f}")

        recog_id = str(tile.get("recognized_identity", "") or "")
        recog_conf = float(tile.get("recognition_confidence", 0.0))
        if recog_id and recog_id != "unknown":
            self._recog_pill.setText(f"[ID] {recog_id}  {format_confidence_percent(recog_conf)}")
            self._recog_pill.show()
        else:
            self._recog_pill.hide()
        if had_signature and (image_changed or previous_state != preview_state):
            self.soften_refresh()


class GridAddOverlay(QWidget):
    """Transparent overlay that places [add] buttons next to each numbered grid cell.

    Buttons are only visible while the grid is active. Each click emits
    cell_add_requested with the cell number (1–8).
    """

    cell_add_requested = pyqtSignal(int)
    cell_settings_requested = pyqtSignal(int)

    # (row, col) for each cell number — mirrors VideoPane._draw_grid number_map
    _CELL_MAP: dict[int, tuple[int, int]] = {
        1: (0, 0), 2: (1, 0), 3: (2, 0),
        4: (0, 1), 5: (2, 1),
        6: (0, 2), 7: (1, 2), 8: (2, 2),
    }
    _BTN_STYLE = (
        "QPushButton {"
        "  font-size: 9px;"
        "  padding: 1px 3px;"
        "  background: rgba(20,8,8,0.65);"
        "  color: rgba(245,245,245,0.88);"
        "  border: 1px solid rgba(245,245,245,0.22);"
        "}"
        "QPushButton:hover {"
        "  background: rgba(40,16,16,0.85);"
        "  border-color: rgba(245,245,245,0.45);"
        "}"
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self._buttons: dict[int, QPushButton] = {}
        self._occupied: dict[int, bool] = {n: False for n in self._CELL_MAP}
        self._is_overlay_visible = False
        # Initial stub mask so the overlay never blocks clicks before buttons show.
        from PyQt6.QtGui import QRegion
        self.setMask(QRegion(-1, -1, 1, 1))
        btn_style = self._BTN_STYLE
        for cell_num in self._CELL_MAP:
            btn = QPushButton("Add", self)
            btn.setIcon(MainWindow._resolve_button_icon("[+] Add", icon_name="add"))
            btn.setIconSize(QSize(16, 16))
            btn.setFixedSize(42, 18)
            btn.setStyleSheet(btn_style)
            btn.setToolTip(f"Add source to cell {cell_num}")
            btn.clicked.connect(lambda _checked=False, n=cell_num: self._on_button_clicked(n))
            self._buttons[cell_num] = btn

    def _on_button_clicked(self, cell_num: int) -> None:
        self.cell_add_requested.emit(cell_num)

    def set_cell_occupied(self, cell_num: int, occupied: bool) -> None:
        if cell_num not in self._buttons:
            return
        self._occupied[cell_num] = occupied
        btn = self._buttons[cell_num]
        btn.setVisible(self._is_overlay_visible and not occupied)
        self._update_mask()

    def _update_mask(self) -> None:
        """Mask overlay to only the visible button rects so clicks elsewhere
        pass through to the cell widgets / video pane underneath."""
        from PyQt6.QtGui import QRegion
        region = QRegion()
        for btn in self._buttons.values():
            if btn.isVisible():
                region += QRegion(btn.geometry())
        if region.isEmpty():
            # An empty mask disables the widget entirely, so give it a 1px stub
            # off-screen to effectively make it click-through everywhere.
            region = QRegion(-1, -1, 1, 1)
        self.setMask(region)

    def set_visible(self, visible: bool) -> None:
        self._is_overlay_visible = visible
        for cell_num, btn in self._buttons.items():
            btn.setVisible(visible and not self._occupied.get(cell_num, False))
        self._update_mask()

    def _reposition_buttons(self) -> None:
        if self.width() == 0 or self.height() == 0:
            return
        cw = self.width() / 3
        ch = self.height() / 3
        for cell_num, (row, col) in self._CELL_MAP.items():
            # Place button to the right of the painted number (which is at 8% from cell left)
            x = int(col * cw + cw * 0.08 + 26)
            y = int(row * ch + ch * 0.04)
            self._buttons[cell_num].move(x, y)

    def resizeEvent(self, event) -> None:
        self._reposition_buttons()
        self._update_mask()
        super().resizeEvent(event)


class _ScaledPixmapLabel(QLabel):
    """QLabel that scales its pixmap to fill available space while preserving aspect ratio."""

    def __init__(self, pixmap: QPixmap, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._src = pixmap
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("background: #0a0404;")
        self.setMinimumSize(1, 1)

    def resizeEvent(self, event) -> None:
        if not self._src.isNull():
            scaled = self._src.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.setPixmap(scaled)
        super().resizeEvent(event)


class VideoHost(QWidget):
    """Video, optional bootstrap loading gate, then dimmed swap-confirmation layer."""

    def __init__(
        self,
        video: VideoPane,
        swap_banner: QFrame,
        loading: Optional[LoadingGateOverlay] = None,
    ) -> None:
        super().__init__()
        self._video = video
        video.setParent(self)
        self._loading = loading
        if loading is not None:
            loading.setParent(self)
        self._dim = QFrame(self)
        self._dim.setStyleSheet("background: rgba(6, 2, 2, 0.55);")
        self._dim.hide()
        inner = QVBoxLayout(self._dim)
        inner.setContentsMargins(0, 0, 0, 0)
        inner.addStretch(2)
        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(swap_banner, stretch=0)
        row.addStretch(1)
        inner.addLayout(row)
        inner.addStretch(3)
        self._grid_overlay = GridAddOverlay(self)
        self._grid_overlay.set_visible(False)
        self._grid_visible = False
        self._center_overlay: Optional[QWidget] = None
        # cell_num -> content widget placed at that cell's geometry
        self._cell_widgets: dict[int, QWidget] = {}

    # (row, col) mirrors GridAddOverlay._CELL_MAP
    _CELL_POSITIONS: dict[int, tuple[int, int]] = {
        1: (0, 0), 2: (1, 0), 3: (2, 0),
        4: (0, 1), 5: (2, 1),
        6: (0, 2), 7: (1, 2), 8: (2, 2),
    }

    def set_cell_widget(self, cell_num: int, widget: QWidget) -> None:
        self.clear_cell_widget(cell_num)
        widget.setParent(self)
        self._cell_widgets[cell_num] = widget
        self._reposition_cell_widget(cell_num)
        widget.setVisible(self._grid_visible)
        self._grid_overlay.raise_()

    def set_cell_opacity(self, cell_num: int, opacity: float) -> None:
        from PyQt6.QtWidgets import QGraphicsOpacityEffect
        w = self._cell_widgets.get(cell_num)
        if w is None:
            return
        eff = QGraphicsOpacityEffect(w)
        eff.setOpacity(max(0.1, min(1.0, opacity)))
        w.setGraphicsEffect(eff)

    def clear_cell_widget(self, cell_num: int) -> None:
        w = self._cell_widgets.pop(cell_num, None)
        if w is not None:
            if w.__class__.__name__ == "QWebEngineView":
                try:
                    w.setUrl(QUrl("about:blank"))  # type: ignore[attr-defined]
                except Exception:
                    pass
            w.hide()
            w.setParent(None)  # type: ignore[arg-type]
            w.deleteLater()

    def show_center_overlay(self, overlay: QWidget) -> None:
        self.clear_center_overlay()
        overlay.setParent(self)
        self._center_overlay = overlay
        overlay.setGeometry(self.rect())
        overlay.show()
        overlay.raise_()

    def clear_center_overlay(self) -> None:
        overlay = self._center_overlay
        self._center_overlay = None
        if overlay is not None:
            overlay.hide()
            overlay.setParent(None)  # type: ignore[arg-type]
            overlay.deleteLater()

    def _reposition_cell_widget(self, cell_num: int) -> None:
        w = self._cell_widgets.get(cell_num)
        if w is None or cell_num not in self._CELL_POSITIONS:
            return
        row, col = self._CELL_POSITIONS[cell_num]
        cr = self._video._content_rect()
        cw = cr.width() / 3
        ch = cr.height() / 3
        x = cr.left() + col * cw
        y = cr.top() + row * ch
        # Clip to VideoHost bounds so widgets never paint past the visible area.
        host_w, host_h = self.width(), self.height()
        x0 = max(0.0, x)
        y0 = max(0.0, y)
        x1 = min(float(host_w), x + cw)
        y1 = min(float(host_h), y + ch)
        w.setGeometry(int(x0), int(y0), max(0, int(x1 - x0)), max(0, int(y1 - y0)))

    @property
    def grid_overlay(self) -> GridAddOverlay:
        return self._grid_overlay

    def set_grid_overlay_visible(self, visible: bool) -> None:
        self._grid_visible = visible
        self._grid_overlay.set_visible(visible)
        for w in self._cell_widgets.values():
            w.setVisible(visible)
        if visible:
            for cell_num in list(self._cell_widgets):
                self._reposition_cell_widget(cell_num)
            self._grid_overlay.raise_()

    def resizeEvent(self, event) -> None:
        r = self.rect()
        self._video.setGeometry(r)
        if self._loading is not None:
            self._loading.setGeometry(r)
        self._dim.setGeometry(r)
        self._grid_overlay.setGeometry(r)
        for cell_num in list(self._cell_widgets):
            self._reposition_cell_widget(cell_num)
        self._video.lower()
        if self._loading is not None:
            self._loading.stackUnder(self._dim)
        if self._dim.isVisible():
            self._dim.raise_()
        elif self._loading is not None and self._loading.isVisible():
            self._loading.raise_()
        else:
            self._grid_overlay.raise_()
        if self._center_overlay is not None and self._center_overlay.isVisible():
            self._center_overlay.setGeometry(r)
            self._center_overlay.raise_()
        super().resizeEvent(event)

    def set_swap_visible(self, visible: bool) -> None:
        self._dim.setVisible(visible)
        if visible and self._loading is not None:
            self._dim.raise_()


class _CvopsAddUploadBridge(QObject):
    finished = pyqtSignal(object, object)


class _WfFolderTree(QWidget):
    """Compact folder hierarchy browser used inside _WorkflowTab.

    Builds a QTreeWidget from a list of slash-separated folder paths, adds a
    root node, lets the user create new pending folders inline, and emits
    folder_selected(path) whenever the selection changes.
    """

    folder_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._last_folders: list[str] = []
        self._pending_folders: list[str] = []
        self._selected_path = ""

        vlay = QVBoxLayout(self)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(3)

        # [HEADER ROW] label + new-folder toggle
        hdr = QHBoxLayout()
        hdr.setSpacing(4)
        lbl = QLabel("FOLDER")
        lbl.setObjectName("wfFieldLabel")
        hdr.addWidget(lbl)
        hdr.addStretch(1)
        self._new_btn = QPushButton("[+] New")
        self._new_btn.setObjectName("wfButton")
        self._new_btn.setToolTip("Create a new folder")
        self._new_btn.clicked.connect(self._toggle_new_input)
        hdr.addWidget(self._new_btn)
        vlay.addLayout(hdr)

        # [FOLDER TREE]
        self._tree = QTreeWidget()
        self._tree.setColumnCount(2)
        self._tree.setHeaderLabels(["Folder", "Items"])
        self._tree.setRootIsDecorated(True)
        self._tree.setUniformRowHeights(True)
        self._tree.setAlternatingRowColors(False)
        self._tree.setMinimumHeight(50)
        self._tree.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._tree.setObjectName("wfFolderTree")
        hdr_view = self._tree.header()
        hdr_view.setStretchLastSection(False)
        hdr_view.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr_view.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._tree.itemSelectionChanged.connect(self._on_selection)
        vlay.addWidget(self._tree)

        # [NEW FOLDER INPUT ROW] hidden until [+] is pressed
        self._new_row = QWidget()
        self._new_row.setObjectName("wfNewFolderRow")
        nr_lay = QHBoxLayout(self._new_row)
        nr_lay.setContentsMargins(0, 2, 0, 0)
        nr_lay.setSpacing(4)
        self._new_edit = QLineEdit()
        self._new_edit.setObjectName("wfNewFolderEdit")
        self._new_edit.setPlaceholderText("new/folder/path")
        nr_lay.addWidget(self._new_edit, 1)
        add_btn = QPushButton("Add")
        add_btn.setObjectName("wfButton")
        add_btn.clicked.connect(self._commit_new_folder)
        self._new_edit.returnPressed.connect(self._commit_new_folder)
        nr_lay.addWidget(add_btn)
        self._new_row.hide()
        vlay.addWidget(self._new_row)

    # ---- Public API ----

    def populate(self, folders: list[str], counts: Optional[dict[str, int]] = None) -> None:
        """Rebuild the tree from a list of slash-delimited folder paths.

        Pending (locally created, not-yet-sent) folders are merged in and shown
        with a [new] tag.  counts maps path -> image count for the Items column.
        """
        self._last_folders = list(folders)
        merged = sorted(
            {f.strip("/") for f in list(folders) + self._pending_folders if f.strip("/")},
            key=lambda x: (x.count("/"), x.lower()),
        )
        counts = counts or {}
        prev = self._selected_path

        self._tree.blockSignals(True)
        self._tree.clear()

        # Root node — always first, selectable, path = ""
        root_total = sum(counts.values()) if counts else 0
        root_item = QTreeWidgetItem(["/ root", str(root_total) if root_total else ""])
        root_item.setData(0, Qt.ItemDataRole.UserRole, "")
        self._tree.addTopLevelItem(root_item)

        nodes: dict[str, QTreeWidgetItem] = {}
        for path in merged:
            name = path.rsplit("/", 1)[-1]
            is_pending = path in self._pending_folders
            label = f"{name}/" + (" [new]" if is_pending else "")
            count_str = str(counts[path]) if path in counts else ""
            item = QTreeWidgetItem([label, count_str])
            item.setData(0, Qt.ItemDataRole.UserRole, path)
            if is_pending:
                item.setForeground(0, QColor(255, 220, 100))
            nodes[path] = item
            parent_key = path.rsplit("/", 1)[0] if "/" in path else ""
            if parent_key and parent_key in nodes:
                nodes[parent_key].addChild(item)
            else:
                self._tree.addTopLevelItem(item)

        self._tree.expandAll()
        self._tree.blockSignals(False)

        # Restore previous selection; fall back to root
        if not self._restore_selection(prev):
            self._select_root()

    def selected_folder(self) -> str:
        return self._selected_path

    def set_enabled(self, enabled: bool) -> None:
        self._tree.setEnabled(enabled)
        self._new_btn.setEnabled(enabled)

    # ---- Internal ----

    def _restore_selection(self, path: str) -> bool:
        root = self._tree.invisibleRootItem()
        return self._walk_select(root, path)

    def _walk_select(self, parent: QTreeWidgetItem, path: str) -> bool:
        for i in range(parent.childCount()):
            child = parent.child(i)
            if str(child.data(0, Qt.ItemDataRole.UserRole) or "") == path:
                self._tree.setCurrentItem(child)
                self._selected_path = path
                return True
            if self._walk_select(child, path):
                return True
        return False

    def _select_root(self) -> None:
        if self._tree.topLevelItemCount():
            item = self._tree.topLevelItem(0)
            self._tree.setCurrentItem(item)
            self._selected_path = ""

    def _on_selection(self) -> None:
        items = self._tree.selectedItems()
        path = str(items[0].data(0, Qt.ItemDataRole.UserRole) or "") if items else ""
        if path != self._selected_path:
            self._selected_path = path
            self.folder_selected.emit(path)

    def _toggle_new_input(self) -> None:
        visible = not self._new_row.isVisible()
        self._new_row.setVisible(visible)
        if visible:
            self._new_edit.setFocus()
            self._new_edit.clear()

    def _commit_new_folder(self) -> None:
        raw = self._new_edit.text().strip().strip("/")
        if not raw:
            return
        clean = re.sub(r"[^a-zA-Z0-9_/.\- ]", "_", raw).strip("/")
        if not clean:
            return
        if clean not in self._pending_folders:
            self._pending_folders.append(clean)
        self._new_row.hide()
        self._new_edit.clear()
        self.populate(self._last_folders)
        if not self._restore_selection(clean):
            self._select_root()
        self.folder_selected.emit(self._selected_path)


class _WorkflowTab(QWidget):
    """Workflow tab in the data column — compact DB send form + result bridge."""

    cancel_requested = pyqtSignal()
    send_requested = pyqtSignal()

    _STATE_IDLE = 0
    _STATE_FORM = 1
    _STATE_BRIDGE = 2

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._tile: dict[str, Any] = {}
        self._image_raw = b""
        self._default_label = "object"
        self._load_classes: Optional[Callable[[str], list[str]]] = None
        self._load_folders: Optional[Callable[[str, str], list[str]]] = None
        self._load_previews: Optional[Callable[..., list[dict[str, Any]]]] = None
        self._send_state = "Ready"
        self._send_detail = ""
        self._classes_by_dataset: dict[str, list[str]] = {}
        self._folders_by_dataset_split: dict[tuple[str, str], list[str]] = {}
        self._source_pixmap: Optional[QPixmap] = None
        self._bridge_pixmap: Optional[QPixmap] = None

        self.setObjectName("workflowTab")
        self.setStyleSheet(self._style())

        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)

        # [HEADER] always visible
        hdr = QHBoxLayout()
        hdr.setSpacing(6)
        ttl = QLabel("WORKFLOW")
        ttl.setObjectName("wfTitle")
        hdr.addWidget(ttl)
        hdr.addStretch(1)
        self._status_lbl = QLabel("")
        self._status_lbl.setObjectName("wfStatus")
        hdr.addWidget(self._status_lbl)
        root.addLayout(hdr)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setObjectName("wfSep")
        root.addWidget(sep)

        # [STACKED PAGES]
        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_idle_page())
        self._stack.addWidget(self._build_form_page())
        self._stack.addWidget(self._build_bridge_page())
        root.addWidget(self._stack, 1)

        self._stack.setCurrentIndex(self._STATE_IDLE)
        self._apply_sizing()

    # ---- Page builders ----

    def _build_idle_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("wfIdlePage")
        lay = QVBoxLayout(page)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg = QLabel("No active workflow.\nSelect an image from the live feed\nand use Add to Database.")
        msg.setObjectName("wfMeta")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setWordWrap(True)
        lay.addWidget(msg)
        return page

    def _build_form_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("wfFormPage")
        vlay = QVBoxLayout(page)
        vlay.setContentsMargins(0, 2, 0, 0)
        vlay.setSpacing(3)

        # [DATABASE ROW] label + combo on one line
        db_row = QHBoxLayout()
        db_row.setSpacing(4)
        db_lbl = QLabel("DATABASE")
        db_lbl.setObjectName("wfFieldLabel")
        self._dataset_combo = QComboBox()
        self._dataset_combo.setEditable(True)
        self._split_combo = QComboBox()
        self._split_combo.addItem("Train", "train")
        self._split_combo.addItem("Val", "val")
        self._split_combo.setEnabled(False)
        db_row.addWidget(db_lbl, 0)
        db_row.addWidget(self._dataset_combo, 1)
        vlay.addLayout(db_row)

        # [FOLDER TREE] — expands to fill remaining vertical space
        self._folder_tree = _WfFolderTree()
        self._folder_tree.folder_selected.connect(self._on_folder_selected)
        vlay.addWidget(self._folder_tree, 1)

        # [FOLDER PREVIEW STRIP] — proportionally sized in resizeEvent
        self._preview_strip_lbl = QLabel("")
        self._preview_strip_lbl.setObjectName("wfFieldLabel")
        vlay.addWidget(self._preview_strip_lbl)

        self._preview_strip = QWidget()
        self._preview_strip.setObjectName("wfPreviewStrip")
        strip_lay = QHBoxLayout(self._preview_strip)
        strip_lay.setContentsMargins(2, 2, 2, 2)
        strip_lay.setSpacing(2)
        strip_lay.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._preview_thumbs: list[QLabel] = []
        for _ in range(5):
            th = QLabel()
            th.setObjectName("wfImage")
            th.setScaledContents(True)
            th.setAlignment(Qt.AlignmentFlag.AlignCenter)
            th.hide()
            strip_lay.addWidget(th)
            self._preview_thumbs.append(th)
        self._preview_empty_lbl = QLabel("Empty")
        self._preview_empty_lbl.setObjectName("wfMeta")
        strip_lay.addWidget(self._preview_empty_lbl)
        strip_lay.addStretch(1)
        vlay.addWidget(self._preview_strip)

        # [LABEL ROW] label + combo on one line
        label_row = QHBoxLayout()
        label_row.setSpacing(4)
        label_lbl = QLabel("LABEL")
        label_lbl.setObjectName("wfFieldLabel")
        self._label_combo = QComboBox()
        self._label_combo.setEditable(True)
        label_row.addWidget(label_lbl, 0)
        label_row.addWidget(self._label_combo, 1)
        vlay.addLayout(label_row)

        # [SOURCE ROW] thumb + meta text
        src_row = QHBoxLayout()
        src_row.setSpacing(6)
        src_row.setContentsMargins(0, 0, 0, 0)
        self._source_thumb = QLabel()
        self._source_thumb.setObjectName("wfImage")
        self._source_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._source_thumb.setMinimumSize(36, 36)
        src_row.addWidget(self._source_thumb, 0, Qt.AlignmentFlag.AlignTop)
        self._source_meta = QLabel("")
        self._source_meta.setObjectName("wfMeta")
        self._source_meta.setWordWrap(True)
        src_row.addWidget(self._source_meta, 1)
        vlay.addLayout(src_row)

        # [ACTIONS]
        actions = QHBoxLayout()
        actions.setSpacing(4)
        actions.addStretch(1)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setObjectName("wfButton")
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("wfButtonPrimary")
        self._send_btn.clicked.connect(self.send_requested.emit)
        actions.addWidget(self._cancel_btn)
        actions.addWidget(self._send_btn)
        vlay.addLayout(actions)

        self._dataset_combo.currentTextChanged.connect(self._on_dataset_changed)
        self._label_combo.currentTextChanged.connect(lambda _t="": self._refresh_status())

        return page

    def _build_bridge_page(self) -> QWidget:
        page = QWidget()
        page.setObjectName("wfBridgePage")
        vlay = QVBoxLayout(page)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(8)

        self._bridge_status_lbl = QLabel("")
        self._bridge_status_lbl.setObjectName("wfBridgeGood")
        self._bridge_status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vlay.addWidget(self._bridge_status_lbl)

        row = QHBoxLayout()
        row.setSpacing(10)
        self._bridge_thumb = QLabel()
        self._bridge_thumb.setObjectName("wfImage")
        self._bridge_thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._bridge_thumb.setMinimumSize(36, 36)
        row.addWidget(self._bridge_thumb, 0, Qt.AlignmentFlag.AlignTop)

        details_col = QVBoxLayout()
        details_col.setSpacing(3)
        self._bridge_dest_lbl = QLabel("")
        self._bridge_dest_lbl.setObjectName("wfBridgeBody")
        self._bridge_dest_lbl.setWordWrap(True)
        self._bridge_label_lbl = QLabel("")
        self._bridge_label_lbl.setObjectName("wfMeta")
        self._bridge_label_lbl.setWordWrap(True)
        self._bridge_detail_lbl = QLabel("")
        self._bridge_detail_lbl.setObjectName("wfMeta")
        self._bridge_detail_lbl.setWordWrap(True)
        details_col.addWidget(self._bridge_dest_lbl)
        details_col.addWidget(self._bridge_label_lbl)
        details_col.addWidget(self._bridge_detail_lbl)
        details_col.addStretch(1)
        row.addLayout(details_col, 1)
        vlay.addLayout(row)

        self._bridge_preview_lbl = QLabel("")
        self._bridge_preview_lbl.setObjectName("wfFieldLabel")
        vlay.addWidget(self._bridge_preview_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.viewport().setStyleSheet("background: transparent;")
        self._bridge_preview_inner = QWidget()
        self._bridge_preview_inner.setStyleSheet("background: transparent;")
        self._bridge_preview_layout = QVBoxLayout(self._bridge_preview_inner)
        self._bridge_preview_layout.setContentsMargins(0, 0, 0, 0)
        self._bridge_preview_layout.setSpacing(4)
        self._bridge_preview_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        scroll.setWidget(self._bridge_preview_inner)
        vlay.addWidget(scroll, 1)

        reset_row = QHBoxLayout()
        reset_row.addStretch(1)
        self._bridge_new_btn = QPushButton("New Send")
        self._bridge_new_btn.setObjectName("wfButton")
        self._bridge_new_btn.clicked.connect(self._back_to_form)
        reset_row.addWidget(self._bridge_new_btn)
        vlay.addLayout(reset_row)

        return page

    def _style(self) -> str:
        return (
            # [CONTAINER] — solid red surface
            "QWidget#workflowTab { background: #bf1f26; }"
            "QWidget#wfIdlePage, QWidget#wfFormPage, QWidget#wfBridgePage { background: transparent; }"
            # [SEPARATOR]
            "QFrame#wfSep { background: rgba(255,255,255,0.28); border: none; max-height: 1px; }"
            # [TITLE CHIP] — saturated red chip on container red, white text, dotted border
            "QLabel#wfTitle {"
            " color: #ffffff; font-size: 9px; font-weight: 800; letter-spacing: 2px;"
            " padding: 3px 8px;"
            " background: #9a1820;"
            " border: 1px dotted rgba(255,255,255,0.42); border-radius: 0px; }"
            # [STATUS CHIP]
            "QLabel#wfStatus {"
            " color: #ffffff; font-size: 10px; font-weight: 700;"
            " background: #9a1820;"
            " border: 1px dotted rgba(255,255,255,0.38);"
            " padding: 2px 6px; border-radius: 0px; }"
            # [FIELD LABELS] — same chip pattern as settings data cards
            "QLabel#wfFieldLabel {"
            " color: #ffffff; font-size: 9px; font-weight: 800; letter-spacing: 0.8px;"
            " background: #9a1820;"
            " border: 1px dotted rgba(255,255,255,0.42);"
            " border-radius: 0px; padding: 2px 6px; }"
            # [META TEXT] — same red family, slightly lighter chip
            "QLabel#wfMeta {"
            " color: #ffffff; font-size: 10px;"
            " background: #a81c23;"
            " border: 1px dotted rgba(255,255,255,0.32);"
            " padding: 2px 5px; border-radius: 0px; }"
            # [IMAGE CELLS]
            "QLabel#wfImage {"
            " background: rgba(0,0,0,0.28);"
            " border: 1px solid rgba(255,255,255,0.35); }"
            # [BRIDGE STATUS BANNER] — chip, colour indicates success/fail
            "QLabel#wfBridgeGood, QLabel#wfBridgeFailed {"
            " font-size: 11px; font-weight: 800; padding: 4px 10px;"
            " background: #9a1820;"
            " border: 1px dotted rgba(255,255,255,0.42); border-radius: 0px; }"
            "QLabel#wfBridgeGood { color: #a8ffb0; }"
            "QLabel#wfBridgeFailed { color: #ffb3b3; }"
            # [BRIDGE DESTINATION BODY]
            "QLabel#wfBridgeBody {"
            " color: #ffffff; font-size: 11px; font-weight: 700;"
            " background: #a81c23;"
            " border: 1px dotted rgba(255,255,255,0.32);"
            " padding: 2px 6px; border-radius: 0px; }"
            # [BRIDGE PREVIEW ROWS]
            "QFrame#wfPreviewRow {"
            " background: rgba(0,0,0,0.15);"
            " border: 1px solid rgba(255,255,255,0.20); border-radius: 0px; }"
            # [FOLDER TREE] — dark surface for contrast, white text
            "QTreeWidget#wfFolderTree {"
            " background: rgba(0,0,0,0.25);"
            " border: 1px solid rgba(255,255,255,0.30);"
            " color: #ffffff; font-size: 10px; border-radius: 0px; }"
            "QTreeWidget#wfFolderTree::item {"
            " padding: 2px 4px; border: none; }"
            "QTreeWidget#wfFolderTree::item:selected {"
            " background: rgba(255,255,255,0.22); color: #ffffff; }"
            "QTreeWidget#wfFolderTree::item:hover:!selected {"
            " background: rgba(255,255,255,0.10); }"
            "QHeaderView::section {"
            " background: rgba(0,0,0,0.35); color: #ffffff;"
            " border: none; border-bottom: 1px solid rgba(255,255,255,0.22);"
            " padding: 1px 4px; font-size: 9px; font-weight: 700; letter-spacing: 0.8px; }"
            "QTreeWidget#wfFolderTree::branch:has-children:!has-siblings:closed,"
            "QTreeWidget#wfFolderTree::branch:closed:has-children:has-siblings {"
            " border-image: none; image: none;"
            " background: rgba(255,255,255,0.08); }"
            "QTreeWidget#wfFolderTree::branch:open:has-children:!has-siblings,"
            "QTreeWidget#wfFolderTree::branch:open:has-children:has-siblings {"
            " background: rgba(255,255,255,0.04); }"
            # [NEW FOLDER INPUT]
            "QWidget#wfNewFolderRow { background: transparent; }"
            "QLineEdit#wfNewFolderEdit {"
            " background: rgba(0,0,0,0.28);"
            " border: 1px solid rgba(255,255,255,0.40);"
            " color: #ffffff; padding: 2px 5px; font-size: 10px; border-radius: 0px; }"
            # [FOLDER PREVIEW STRIP]
            "QWidget#wfPreviewStrip {"
            " background: rgba(0,0,0,0.20);"
            " border: 1px solid rgba(255,255,255,0.22); }"
            # [COMBO BOXES]
            "QComboBox {"
            " background: rgba(0,0,0,0.25);"
            " border: 1px solid rgba(255,255,255,0.38);"
            " color: #ffffff; padding: 3px 6px; font-size: 10px; }"
            "QComboBox:hover {"
            " background: rgba(0,0,0,0.35);"
            " border-color: rgba(255,255,255,0.60); }"
            "QComboBox:focus {"
            " background: rgba(0,0,0,0.30);"
            " border: 1px solid rgba(255,255,255,0.72); }"
            "QComboBox::drop-down { border: none; width: 18px; background: rgba(0,0,0,0.18); }"
            "QComboBox QAbstractItemView {"
            " background: #7b0e12; color: #ffffff;"
            " border: 1px solid rgba(255,255,255,0.28); }"
            # [CANCEL / SECONDARY BUTTON]
            "QPushButton#wfButton {"
            " border: 1px solid rgba(255,255,255,0.40);"
            " background: transparent;"
            " color: #ffffff;"
            " padding: 3px 10px; font-size: 9px; letter-spacing: 1px; border-radius: 0px; }"
            "QPushButton#wfButton:hover {"
            " background: rgba(255,255,255,0.12);"
            " border-color: rgba(255,255,255,0.65); }"
            # [SEND / PRIMARY BUTTON] — inverted: white fill, red text
            "QPushButton#wfButtonPrimary {"
            " border: 1px solid #ffffff;"
            " background: #ffffff; color: #bf1f26;"
            " padding: 3px 10px; font-size: 9px; font-weight: 800; letter-spacing: 1px; border-radius: 0px; }"
            "QPushButton#wfButtonPrimary:hover {"
            " background: rgba(255,255,255,0.88); }"
        )

    # ---- Proportional sizing ----

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_sizing()

    def _apply_sizing(self) -> None:
        w = max(100, self.width() - 12)
        # [PREVIEW STRIP] thumbnails — each gets 1/5 of available width
        th_side = max(28, (w - 16) // 5)
        for th in self._preview_thumbs:
            th.setFixedSize(th_side, th_side)
        self._preview_strip.setFixedHeight(th_side + 6)
        # [SOURCE THUMB] — ~1/4 width, slightly taller than wide
        sw = max(36, min(72, w // 4))
        sh = max(36, int(sw * 1.2))
        self._source_thumb.setFixedSize(sw, sh)
        if self._source_pixmap and not self._source_pixmap.isNull():
            self._source_thumb.setPixmap(
                self._source_pixmap.scaled(sw, sh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        # [BRIDGE THUMB] — same proportions as source
        bw = max(36, min(80, w // 4))
        bh = max(36, int(bw * 1.2))
        self._bridge_thumb.setFixedSize(bw, bh)
        if self._bridge_pixmap and not self._bridge_pixmap.isNull():
            self._bridge_thumb.setPixmap(
                self._bridge_pixmap.scaled(bw, bh, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )

    # ---- Public API ----

    def load(
        self,
        *,
        tile: dict[str, Any],
        image_raw: bytes,
        default_label: str,
        dataset_names: list[str],
        load_classes: Callable[[str], list[str]],
        load_folders: Callable[[str, str], list[str]],
        load_previews: Callable[..., list[dict[str, Any]]],
    ) -> None:
        self._tile = dict(tile)
        self._image_raw = bytes(image_raw or b"")
        self._default_label = str(default_label or "object").strip() or "object"
        self._load_classes = load_classes
        self._load_folders = load_folders
        self._load_previews = load_previews
        self._send_state = "Ready"
        self._send_detail = ""
        self._classes_by_dataset.clear()
        self._folders_by_dataset_split.clear()

        raw_px = QPixmap()
        raw_px.loadFromData(self._image_raw)
        self._source_pixmap = raw_px if not raw_px.isNull() else None
        if self._source_pixmap is None:
            self._source_thumb.clear()
            self._source_thumb.setText("NO IMG")
        self._apply_sizing()

        self._source_meta.setText(
            f"{detection_label_text(tile.get('label', '')).upper()} | "
            f"{format_confidence_percent(tile.get('confidence', 0))}"
        )

        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        self._dataset_combo.addItems(dataset_names)
        self._dataset_combo.blockSignals(False)

        self._on_dataset_changed(self._dataset_combo.currentText())
        self._stack.setCurrentIndex(self._STATE_FORM)

    def show_bridge(
        self,
        *,
        status: str,
        dataset_slug: str,
        target_folder: str,
        label_name: str,
        detail: str = "",
        source_pixmap: Optional[QPixmap] = None,
    ) -> None:
        if status == "good":
            self._bridge_status_lbl.setText("Send: Good")
            self._bridge_status_lbl.setObjectName("wfBridgeGood")
        else:
            self._bridge_status_lbl.setText("Send: Failed")
            self._bridge_status_lbl.setObjectName("wfBridgeFailed")
        self._bridge_status_lbl.style().unpolish(self._bridge_status_lbl)
        self._bridge_status_lbl.style().polish(self._bridge_status_lbl)

        self._bridge_pixmap = source_pixmap if (source_pixmap and not source_pixmap.isNull()) else None
        if self._bridge_pixmap is None:
            self._bridge_thumb.clear()
            self._bridge_thumb.setText("IMG")
        self._apply_sizing()

        folder_display = target_folder or "root"
        self._bridge_dest_lbl.setText(f"{dataset_slug} / {folder_display}")
        self._bridge_label_lbl.setText(f"Label: {label_name}")
        self._bridge_detail_lbl.setText(detail or ("Saved" if status == "good" else "Failed"))
        self._status_lbl.setText(f"Send: {'Good' if status == 'good' else 'Failed'}")

        self._clear_bridge_preview()
        if self._load_previews:
            try:
                items = self._load_previews(dataset_slug, split="", target_folder=target_folder, limit=4)
                if items:
                    self._bridge_preview_lbl.setText(f"Recent in {dataset_slug} / {folder_display}")
                    for item in items:
                        self._add_bridge_preview_row(item)
                else:
                    self._bridge_preview_lbl.setText(f"No images yet in {dataset_slug} / {folder_display}")
            except Exception:
                self._bridge_preview_lbl.setText("Preview unavailable")

        self._stack.setCurrentIndex(self._STATE_BRIDGE)

    def reset_to_idle(self) -> None:
        self._send_state = "Ready"
        self._send_detail = ""
        self._status_lbl.setText("")
        self._stack.setCurrentIndex(self._STATE_IDLE)

    def _back_to_form(self) -> None:
        self._send_state = "Ready"
        self._send_detail = ""
        self._refresh_status()
        self._stack.setCurrentIndex(self._STATE_FORM)

    # ---- Internal helpers ----

    def _clear_bridge_preview(self) -> None:
        while self._bridge_preview_layout.count():
            item = self._bridge_preview_layout.takeAt(0)
            w = item.widget() if item else None
            if w:
                w.deleteLater()

    def _add_bridge_preview_row(self, item: dict[str, Any]) -> None:
        row = QFrame()
        row.setObjectName("wfPreviewRow")
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(4, 4, 4, 4)
        row_layout.setSpacing(6)
        thumb = QLabel()
        thumb.setObjectName("wfImage")
        thumb.setFixedSize(36, 36)
        thumb.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = item.get("pixmap")
        if isinstance(pixmap, QPixmap) and not pixmap.isNull():
            thumb.setPixmap(
                pixmap.scaled(
                    36, 36,
                    Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        else:
            thumb.setText("IMG")
        row_layout.addWidget(thumb)
        name_lbl = QLabel(str(item.get("name") or item.get("path") or "image"))
        name_lbl.setObjectName("wfMeta")
        name_lbl.setWordWrap(True)
        row_layout.addWidget(name_lbl, 1)
        self._bridge_preview_layout.addWidget(row)

    def _on_dataset_changed(self, _text: str = "") -> None:
        self._load_label_choices()
        self._load_target_folder_choices()
        self._refresh_status()

    def _on_folder_selected(self, path: str) -> None:
        self._refresh_status()
        self._refresh_folder_preview(path)

    def _load_label_choices(self) -> None:
        if not self._load_classes:
            return
        slug = self.dataset_slug()
        classes = self._classes_by_dataset.get(slug)
        if classes is None:
            try:
                classes = self._load_classes(slug)
            except Exception:
                classes = []
            self._classes_by_dataset[slug] = list(classes)
        self._label_combo.blockSignals(True)
        self._label_combo.clear()
        for cls in classes:
            self._label_combo.addItem(cls)
        if self._default_label:
            match = next(
                (idx for idx, cls in enumerate(classes) if cls.lower() == self._default_label.lower()), -1
            )
            if match >= 0:
                self._label_combo.setCurrentIndex(match)
            else:
                self._label_combo.setEditText(self._default_label)
        self._label_combo.blockSignals(False)

    def _load_target_folder_choices(self) -> None:
        if not self._load_folders:
            return
        slug = self.dataset_slug()
        split = self.split_value()
        cache_key = (slug, split)
        folders = self._folders_by_dataset_split.get(cache_key)
        if folders is None:
            try:
                folders = self._load_folders(slug, split)
            except Exception:
                folders = []
            self._folders_by_dataset_split[cache_key] = list(folders)
        self._folder_tree.populate(folders)
        self._refresh_folder_preview(self._folder_tree.selected_folder())

    def _refresh_folder_preview(self, folder_path: str) -> None:
        slug = self.dataset_slug()
        folder_display = folder_path or "root"
        for th in self._preview_thumbs:
            th.hide()
            th.clear()
        self._preview_empty_lbl.show()
        if not slug or not self._load_previews:
            self._preview_strip_lbl.setText("")
            return
        self._preview_strip_lbl.setText(f"PREVIEW — {slug} / {folder_display}")
        try:
            items = self._load_previews(slug, split="", target_folder=folder_path, limit=5)
        except Exception:
            items = []
        if not items:
            self._preview_empty_lbl.show()
            return
        self._preview_empty_lbl.hide()
        for idx, item in enumerate(items[:len(self._preview_thumbs)]):
            th = self._preview_thumbs[idx]
            pixmap = item.get("pixmap")
            if isinstance(pixmap, QPixmap) and not pixmap.isNull():
                th.setPixmap(
                    pixmap.scaled(
                        48, 48,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
            else:
                th.setText("IMG")
            th.setToolTip(str(item.get("name") or item.get("path") or ""))
            th.show()

    def _refresh_status(self) -> None:
        state_txt = self._send_state
        if self._send_detail:
            state_txt += f" | {self._send_detail}"
        self._status_lbl.setText(f"Send: {state_txt}")

    def set_send_state(self, state: str, detail: str = "") -> None:
        normalized = str(state or "").strip().lower()
        if normalized == "good":
            self._send_state = "Good"
        elif normalized == "failed":
            self._send_state = "Failed"
        elif normalized == "loading":
            self._send_state = "Loading"
        else:
            self._send_state = "Ready"
        self._send_detail = str(detail or "").strip()
        self._refresh_status()

    def set_controls_enabled(self, enabled: bool) -> None:
        self._dataset_combo.setEnabled(enabled)
        self._folder_tree.set_enabled(enabled)
        self._label_combo.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        self._cancel_btn.setEnabled(enabled)
        self._split_combo.setEnabled(False)

    def dataset_slug(self) -> str:
        return str(self._dataset_combo.currentText() or "").strip()

    def split_value(self) -> str:
        return str(self._split_combo.currentData() or "train").strip() or "train"

    def target_folder(self) -> str:
        return self._folder_tree.selected_folder()

    def label_name(self) -> str:
        return str(self._label_combo.currentText() or "").strip()

    def destination_label(self) -> str:
        db = self.dataset_slug() or "database"
        folder = self.target_folder() or "root"
        return f"{db}/{folder}"

    def selection(self) -> dict[str, str]:
        return {
            "dataset_slug": self.dataset_slug(),
            "split": self.split_value(),
            "target_folder": self.target_folder(),
            "label_name": self.label_name(),
        }

    def refresh_after_send(self) -> None:
        self._folders_by_dataset_split.pop((self.dataset_slug(), self.split_value()), None)
        self._load_target_folder_choices()


class FocusVisualHost(QWidget):
    """Composite image + silhouette + scan overlay."""

    def __init__(self) -> None:
        super().__init__()
        self.composite = FocusImageComposite(self)
        self.overlay = FocusOverlay(self)

    def resizeEvent(self, event) -> None:
        r = self.rect()
        self.composite.setGeometry(r)
        self.overlay.setGeometry(r)
        super().resizeEvent(event)


class _OpaqueHudStrip(QWidget):
    """Bottom HUD strip that paints its own opaque fill (independent of QSS alpha)."""

    def __init__(self, *, top_edge: bool = False, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._top_edge = bool(top_edge)
        self._bg = QColor("#000000")
        self._border = QColor("#000000")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAutoFillBackground(False)

    def apply_strip_colors(self, bg_css: str, border_css: str) -> None:
        next_bg = QColor(str(bg_css))
        next_border = QColor(str(border_css))
        if not next_bg.isValid():
            next_bg = QColor("#000000")
        if not next_border.isValid():
            next_border = QColor("#000000")
        if next_bg == self._bg and next_border == self._border:
            return
        self._bg = next_bg
        self._border = next_border
        # Allow true transparency for bottom strip bars when background alpha is zero.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, self._bg.alpha() == 0)
        self.update()

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        if self._bg.alpha() > 0:
            p.fillRect(self.rect(), self._bg)
        p.setPen(QPen(self._border, 1))
        y = 0 if self._top_edge else max(0, self.height() - 1)
        p.drawLine(0, y, max(0, self.width() - 1), y)
        p.end()


class _StatusDotButton(QPushButton):
    """HUD button with an optional top-right status indicator."""

    _STATUS_COLORS: dict[str, QColor] = {
        "running": QColor(72, 214, 114, 245),
        "loading": QColor(251, 205, 70, 245),
        "error": QColor(232, 86, 86, 245),
    }

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(text, parent)
        self._status_state = "none"
        self.setProperty("status_state", self._status_state)

    def set_status_state(self, state: str) -> None:
        normalized = str(state or "none").strip().lower()
        if normalized not in {"none", "running", "loading", "error"}:
            normalized = "none"
        if normalized == self._status_state:
            return
        self._status_state = normalized
        self.setProperty("status_state", self._status_state)
        self.update()

    def status_state(self) -> str:
        return self._status_state

    def paintEvent(self, event) -> None:
        icon_centered = False
        if bool(self.property("hudCircle")) and not self.text() and not self.icon().isNull():
            opt = QStyleOptionButton()
            self.initStyleOption(opt)
            opt.icon = QIcon()
            opt.text = ""
            painter = QStylePainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.drawControl(QStyle.ControlElement.CE_PushButton, opt)
            mode = QIcon.Mode.Normal
            if not self.isEnabled():
                mode = QIcon.Mode.Disabled
            elif self.isDown():
                mode = QIcon.Mode.Selected
            elif self.underMouse():
                mode = QIcon.Mode.Active
            state = QIcon.State.On if self.isChecked() else QIcon.State.Off
            pm = self.icon().pixmap(self.iconSize(), mode, state)
            x = int(round((self.width() - pm.width()) / 2.0))
            y = int(round((self.height() - pm.height()) / 2.0))
            painter.drawPixmap(max(0, x), max(0, y), pm)
            painter.end()
            icon_centered = True
        if not icon_centered:
            super().paintEvent(event)
        color = self._STATUS_COLORS.get(self._status_state)
        if color is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if bool(self.property("hudButton")):
            ring_rect = self.rect().adjusted(1, 1, -1, -1)
            p.setPen(QPen(color, 2.2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawEllipse(ring_rect)
            p.end()
            return
        radius = 4.1
        inset = 3.2
        center = QPointF(float(self.width()) - inset - radius, inset + radius)
        p.setPen(QPen(QColor(20, 20, 20, 215), 1.15))
        p.setBrush(color)
        p.drawEllipse(center, radius, radius)
        p.end()


class MainWindow(QMainWindow):
    _TAB_PREVIEWS = 0
    _TAB_FOCUS = 1
    _TAB_EVENTS = 2
    _CAT_TAB_GALLERY = 0
    _CAT_TAB_SETTINGS = 1
    _CAT_TAB_WORKFLOW = 2
    _CAT_TAB_SUBROUTINE = 3
    _BUTTON_TOKEN_RE = re.compile(r"^\[([^\]]+)\]\s*(.*)$")
    _AUTO_BUTTON_ICON_DIR = ROOT_DIR / "state" / "insight_local" / "button_icons"
    _auto_button_icon_cache: dict[tuple[str, str], QIcon] = {}
    _BUTTON_CAPTION_MODES = {"both", "title", "icon"}

    @classmethod
    def _split_button_caption(cls, caption: str) -> tuple[str, str]:
        text = str(caption or "").strip()
        if not text:
            return "", ""
        match = cls._BUTTON_TOKEN_RE.match(text)
        if not match:
            return "", text
        token = (match.group(1) or "").strip()
        title = (match.group(2) or "").strip()
        return token, title

    @classmethod
    def _token_from_title(cls, title: str) -> str:
        words = re.findall(r"[A-Za-z0-9]+", title)
        if not words:
            return ""
        if len(words) == 1:
            return words[0][:2].upper()
        return "".join(word[0] for word in words[:2]).upper()

    @classmethod
    def _resolve_button_icon(
        cls,
        caption: str,
        *,
        icon_name: Optional[str] = None,
    ) -> QIcon:
        token, title = cls._split_button_caption(caption)
        cache_key = (str(icon_name or ""), f"{token}|{title}")
        cached = cls._auto_button_icon_cache.get(cache_key)
        if cached is not None:
            return cached

        icon_path = None
        if icon_name:
            file_name = icon_name if icon_name.endswith(".svg") else f"{icon_name}.svg"
            candidate = ASSETS_DIR / "icons" / file_name
            if candidate.exists():
                icon_path = candidate
        if icon_path is None:
            normalized_title = title.strip().lower()
            if normalized_title:
                candidate = ASSETS_DIR / "icons" / f"{normalized_title}.svg"
                if candidate.exists():
                    icon_path = candidate

        if icon_path is not None:
            icon = QIcon(str(icon_path))
            cls._auto_button_icon_cache[cache_key] = icon
            return icon

        glyph = token or cls._token_from_title(title)
        glyph = (glyph or "").strip()[:3]
        if not glyph:
            icon = QIcon()
            cls._auto_button_icon_cache[cache_key] = icon
            return icon

        safe = re.sub(r"[^a-z0-9]+", "_", glyph.lower()).strip("_") or "btn"
        digest = hashlib.sha1(glyph.encode("utf-8")).hexdigest()[:10]
        icon_file = cls._AUTO_BUTTON_ICON_DIR / f"{safe}_{digest}.svg"
        if not icon_file.exists():
            try:
                cls._AUTO_BUTTON_ICON_DIR.mkdir(parents=True, exist_ok=True)
                font_size = "10.5" if len(glyph) == 1 else ("9.0" if len(glyph) == 2 else "7.5")
                svg = (
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none">'
                    '<rect x="1.5" y="1.5" width="21" height="21" rx="6" '
                    'stroke="#1f1a17" stroke-width="1.8"/>'
                    f'<text x="12" y="15" text-anchor="middle" font-family="Arial, Helvetica, sans-serif" '
                    f'font-size="{font_size}" font-weight="700" fill="#1f1a17">{escape(glyph)}</text>'
                    "</svg>"
                )
                icon_file.write_text(svg, encoding="utf-8")
            except OSError:
                icon = QIcon()
                cls._auto_button_icon_cache[cache_key] = icon
                return icon
        icon = QIcon(str(icon_file))
        cls._auto_button_icon_cache[cache_key] = icon
        return icon

    def _apply_button_caption(
        self,
        button: QPushButton,
        caption: str,
        *,
        mode: str = "both",
        icon_name: Optional[str] = None,
        icon_size: int = 14,
        tooltip: Optional[str] = None,
    ) -> None:
        token, title = self._split_button_caption(caption)
        clean_title = title or token
        base_mode = mode if mode in self._BUTTON_CAPTION_MODES else "both"
        global_mode = str(getattr(self, "_button_caption_mode", "both") or "both")
        resolved_mode = global_mode if global_mode in self._BUTTON_CAPTION_MODES else base_mode
        button.setProperty("caption_text", caption)
        button.setProperty("caption_base_mode", base_mode)
        button.setProperty("caption_mode", resolved_mode)
        button.setProperty("caption_icon_name", icon_name or "")
        button.setProperty("caption_icon_size", int(icon_size))
        button.setProperty("caption_fallback", clean_title)
        button.setProperty("caption_tooltip", tooltip if tooltip is not None else "")
        if resolved_mode in {"icon", "both"}:
            icon = self._resolve_button_icon(caption, icon_name=icon_name)
            button.setIcon(icon)
            button.setIconSize(QSize(icon_size, icon_size))
        else:
            button.setIcon(QIcon())
        if resolved_mode == "icon":
            button.setText("")
            button.setToolTip(tooltip or clean_title)
        elif resolved_mode == "title":
            button.setText(clean_title)
            if tooltip:
                button.setToolTip(tooltip)
        else:
            button.setText(clean_title)
            if tooltip:
                button.setToolTip(tooltip)

    def _refresh_button_caption(self, button: QPushButton, caption: str) -> None:
        raw_mode = str(button.property("caption_base_mode") or button.property("caption_mode") or "both")
        raw_icon_name = str(button.property("caption_icon_name") or "")
        raw_icon_size = button.property("caption_icon_size")
        raw_tooltip = button.property("caption_tooltip")
        tooltip = str(raw_tooltip) if raw_tooltip not in (None, "") else None
        icon_size = int(raw_icon_size) if raw_icon_size is not None else 14
        self._apply_button_caption(
            button,
            caption,
            mode=raw_mode,
            icon_name=raw_icon_name or None,
            icon_size=icon_size,
            tooltip=tooltip,
        )

    def _apply_button_caption_mode(self, mode: str) -> None:
        selected = str(mode or "both").strip().lower()
        if selected not in self._BUTTON_CAPTION_MODES:
            selected = "both"
        self._button_caption_mode = selected
        for button in list(getattr(self, "_caption_buttons", [])):
            caption = str(button.property("caption_text") or button.property("caption_fallback") or "")
            if not caption:
                continue
            self._refresh_button_caption(button, caption)

    def _make_button(
        self,
        caption: str,
        *,
        mode: str = "both",
        icon_name: Optional[str] = None,
        icon_size: int = 14,
        checkable: bool = False,
        style: Optional[str] = None,
        tooltip: Optional[str] = None,
        hud: bool = False,
        hud_circle: Optional[bool] = None,
    ) -> QPushButton:
        button = _StatusDotButton("")
        button.setCheckable(checkable)
        if hud:
            button.setProperty("hudButton", True)
            circle = (mode == "icon") if hud_circle is None else bool(hud_circle)
            button.setProperty("hudCircle", circle)
        if style:
            button.setStyleSheet(style)
        self._apply_button_caption(
            button,
            caption,
            mode=mode,
            icon_name=icon_name,
            icon_size=icon_size,
            tooltip=tooltip,
        )
        if isinstance(button, _StatusDotButton):
            button.set_status_state("none")
            if checkable:
                def _on_toggled(on: bool, b: _StatusDotButton = button) -> None:
                    b.set_status_state("running" if bool(on) else "none")
                    self._sync_radial_menu_actions()

                button.toggled.connect(_on_toggled)
            else:
                def _on_clicked(_checked: bool = False, b: _StatusDotButton = button) -> None:
                    self._pulse_action_status(b)
                    self._sync_radial_menu_actions()

                button.clicked.connect(_on_clicked)
        if hasattr(self, "_caption_buttons"):
            self._caption_buttons.append(button)
        return button

    def _pulse_action_status(self, button: QPushButton) -> None:
        if not isinstance(button, _StatusDotButton):
            return
        seq = int(button.property("_statusPulseSeq") or 0) + 1
        button.setProperty("_statusPulseSeq", seq)
        button.set_status_state("loading")
        self._sync_radial_menu_actions()

        def _to_running() -> None:
            if int(button.property("_statusPulseSeq") or 0) != seq:
                return
            button.set_status_state("running")
            self._sync_radial_menu_actions()

            def _to_idle() -> None:
                if int(button.property("_statusPulseSeq") or 0) != seq:
                    return
                if not button.isCheckable():
                    button.set_status_state("none")
                    self._sync_radial_menu_actions()

            QTimer.singleShot(850, _to_idle)

        QTimer.singleShot(120, _to_running)

    def _hud_strip_frame_style(self, *, top_edge: bool) -> str:
        edge = "border-bottom" if top_edge else "border-top"
        bg = get_hud_strip_bg_css() if top_edge else get_hud_bottom_strip_bg_css()
        return (
            f"background: {bg}; "
            f"border: none; {edge}: 1px solid {get_hud_strip_border_css()};"
        )

    def _hud_restore_button_css(self, *, top_edge: bool) -> str:
        if current_color_scheme() == "fire":
            border_side = "border-top: none;" if top_edge else "border-bottom: none;"
            bg_idle = theme_holographic_fire(0.34, "idle")
            bg_hover = theme_holographic_fire(0.48, "hover")
            return (
                f"QPushButton[hudButton='true'] {{ background: {bg_idle}; border: 1px solid {theme_rgba('accent_dark', 0.44)}; "
                f"{border_side} color: {text_css(0.90)}; font-size: 8px; "
                "letter-spacing: 1.0px; padding: 0px; border-radius: 15px; }"
                f"QPushButton[hudButton='true']:hover {{ color: {text_css(0.98)}; border-color: {theme_rgba('accent_dark', 0.68)}; "
                f"background: {bg_hover}; }}"
            )
        border_side = "border-top: none;" if top_edge else "border-bottom: none;"
        alpha_idle = 0.72 if top_edge else 0.66
        alpha_hover = 0.95 if top_edge else 0.90
        return (
            f"QPushButton[hudButton='true'] {{ background: {theme_rgba('panel', 0.40)}; border: 1px solid {theme_rgba('accent_dark', 0.30)}; "
            f"{border_side} color: {surface_muted_css(alpha_idle)}; font-size: 9px; font-weight: 600; "
            "letter-spacing: 0.8px; padding: 0px; border-radius: 15px; }"
            f"QPushButton[hudButton='true']:hover {{ color: {surface_muted_css(alpha_hover)}; border-color: {theme_rgba('accent_dark', 0.52)}; "
            f"background: {theme_rgba('accent_dark', 0.16)}; }}"
        )

    def _hud_dismiss_button_css(self, *, strong: bool = False) -> str:
        if current_color_scheme() == "fire":
            bg_idle = theme_holographic_fire(0.26, "idle")
            bg_hover = theme_holographic_fire(0.40, "hover")
            return (
                f"QPushButton {{ border: 1px solid {theme_rgba('accent_dark', 0.34)}; "
                f"background: {bg_idle}; color: {text_css(0.90)}; padding: 1px 8px; border-radius: 7px; font-size: 10px; }}"
                f"QPushButton:hover {{ background: {bg_hover}; border-color: {theme_rgba('accent_dark', 0.60)}; color: {text_css(0.98)}; }}"
            )
        idle = 0.56 if strong else 0.44
        hover = 0.90 if strong else 0.78
        return (
            f"QPushButton {{ border: 1px solid {theme_rgba('accent_dark', 0.24)}; "
            f"background: {theme_rgba('panel', 0.30)}; color: {surface_muted_css(idle)}; "
            "padding: 2px 8px; border-radius: 9px; font-size: 10px; font-weight: 600; }}"
            f"QPushButton:hover {{ color: {surface_muted_css(hover)}; border-color: {theme_rgba('accent_dark', 0.44)}; "
            f"background: {theme_rgba('accent_dark', 0.14)}; }}"
        )

    def _hud_ghost_button_css(self) -> str:
        if current_color_scheme() == "fire":
            bg_idle = theme_holographic_fire(0.42, "idle")
            bg_hover = theme_holographic_fire(0.58, "hover")
            bg_checked = theme_holographic_fire(0.72, "hover")
            return (
                f"QPushButton[hudButton='true'] {{ border: 1px solid {theme_rgba('accent_dark', 0.44)}; background: {bg_idle}; "
                f"color: {text_css(0.90)}; padding: 5px 10px; font-size: 11px; font-weight: 600; "
                f"letter-spacing: 0.3px; border-radius: 9999px; min-height: 30px; }}"
                f"QPushButton[hudButton='true']:hover {{ color: {text_css(0.98)}; border-color: {theme_rgba('accent_dark', 0.66)}; "
                f"background: {bg_hover}; }}"
                f"QPushButton[hudButton='true']:checked {{ color: {text_css(0.98)}; border-color: {theme_rgba('accent_dark', 0.82)}; "
                f"background: {bg_checked}; }}"
                "QPushButton[hudCircle='true'] { min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; "
                "padding: 0px; border-radius: 15px; }"
            )
        return (
            f"QPushButton[hudButton='true'] {{ border: 1px solid {theme_rgba('accent_dark', 0.28)}; background: {theme_rgba('panel', 0.34)}; "
            f"color: {surface_muted_css(0.82)}; padding: 5px 11px; font-size: 11px; font-weight: 600; border-radius: 11px; }}"
            f"QPushButton[hudButton='true']:hover {{ color: {surface_muted_css(0.98)}; border-color: {theme_rgba('accent_dark', 0.52)}; "
            f"background: {theme_rgba('accent_dark', 0.16)}; }}"
            f"QPushButton[hudButton='true']:pressed {{ border-color: {theme_rgba('pressed', 0.56)}; background: {theme_rgba('accent_dark', 0.24)}; }}"
            f"QPushButton[hudButton='true']:checked {{ color: {text_css(0.98)}; border-color: {theme_rgba('accent_dark', 0.64)}; "
            f"background: {theme_rgba('accent_dark', 0.32)}; }}"
            "QPushButton[hudCircle='true'] { min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; "
            "padding: 0px; border-radius: 15px; }"
        )

    def _hud_toast_css(self) -> str:
        """Toast plaque — same mirror-backed look as text chips but larger for readability."""
        if current_color_scheme() == "fire":
            bg = theme_holographic_fire(0.30, "idle")
            return (
                f"QLabel {{ background: {bg}; border: 1px solid {theme_rgba('accent_dark', 0.36)}; "
                f"padding: 5px 14px; color: {text_css(0.92)}; font-size: 10px; font-weight: 600; "
                f"letter-spacing: 0.4px; border-radius: 5px; }}"
            )
        if current_color_scheme() == "beacon":
            return (
                f"QLabel {{ background: {theme_hex('accent_dark')}; "
                f"border: 1px solid {theme_hex('accent_dark')}; "
                f"padding: 4px 12px; color: {theme_hex('paper')}; font-size: 10px; font-weight: 700; "
                f"letter-spacing: 0.45px; border-radius: 0px; }}"
            )
        return (
            f"QLabel {{ background: {theme_rgba('panel', 0.82)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.32)}; "
            f"padding: 5px 14px; color: {text_css(0.95)}; font-size: 10px; border-radius: 5px; }}"
        )

    def _hud_text_chip_css(self) -> str:
        """Pill backing for read-only text labels in the bottom HUD so they pop against passthrough."""
        if current_color_scheme() == "fire":
            bg = theme_holographic_fire(0.28, "idle")
            return (
                f"color: {text_css(0.88)}; font-size: 9px; font-weight: 600; "
                f"letter-spacing: 0.4px; "
                f"background: {bg}; "
                f"border: 1px solid {theme_rgba('accent_dark', 0.38)}; "
                f"border-radius: 4px; padding: 2px 6px;"
            )
        return (
            f"color: {text_css(0.92)}; font-size: 9px; font-weight: 600; letter-spacing: 0.35px; "
            f"background: {theme_rgba('panel', 0.34)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.28)}; "
            f"border-radius: 8px; padding: 2px 7px;"
        )

    def _hud_scrub_button_css(self) -> str:
        if current_color_scheme() == "fire":
            bg_idle = theme_holographic_fire(0.44, "idle")
            bg_hover = theme_holographic_fire(0.60, "hover")
            bg_checked = theme_holographic_fire(0.76, "hover")
            return (
                f"QPushButton[hudButton='true'] {{ border: 1px solid {theme_rgba('accent_dark', 0.44)}; background: {bg_idle}; "
                f"color: {text_css(0.88)}; padding: 1px 4px; font-size: 9px; font-weight: bold; border-radius: 5px; }}"
                f"QPushButton[hudButton='true']:hover {{ color: {text_css(0.98)}; border-color: {theme_rgba('accent_dark', 0.66)}; "
                f"background: {bg_hover}; }}"
                f"QPushButton[hudButton='true']:checked {{ color: {text_css(0.98)}; border-color: {theme_rgba('accent_dark', 0.82)}; "
                f"background: {bg_checked}; }}"
                "QPushButton[hudCircle='true'] { min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; "
                "padding: 0px; border-radius: 15px; }"
            )
        return (
            f"QPushButton[hudButton='true'] {{ border: 1px solid {theme_rgba('accent_dark', 0.30)}; background: {theme_rgba('panel', 0.28)}; "
            f"color: {text_css(0.84)}; padding: 2px 6px; font-size: 9px; font-weight: 700; border-radius: 8px; }}"
            f"QPushButton[hudButton='true']:hover {{ color: {text_css(0.96)}; border-color: {theme_rgba('accent_dark', 0.52)}; "
            f"background: {theme_rgba('accent_dark', 0.16)}; }}"
            f"QPushButton[hudButton='true']:checked {{ color: {text_css(0.98)}; border-color: {theme_rgba('accent_dark', 0.64)}; "
            f"background: {theme_rgba('accent_dark', 0.32)}; }}"
            "QPushButton[hudCircle='true'] { min-width: 30px; max-width: 30px; min-height: 30px; max-height: 30px; "
            "padding: 0px; border-radius: 15px; }"
        )

    def _apply_hud_theme(self) -> None:
        if hasattr(self, "_bottom_hud"):
            if isinstance(self._bottom_hud, _OpaqueHudStrip):
                self._bottom_hud.apply_strip_colors(
                    get_hud_bottom_strip_bg_css(),
                    theme_rgba("accent_dark", 0.22),
                )
            self._bottom_hud.setStyleSheet(
                "QWidget#bottomHud { background: transparent; border: none; }"
                "QWidget#bottomHud > QWidget { background: transparent; border: none; }"
            )
        if hasattr(self, "_top_panel_toggle"):
            self._top_panel_toggle.setStyleSheet("QWidget#topPanelToggle { background: transparent; border: none; }")
        if hasattr(self, "_scrub_bar"):
            self._scrub_bar.setStyleSheet(
                f"border-top: 1px solid {get_hud_strip_border_css()}; background: transparent;"
            )
        if hasattr(self, "_timeline"):
            self._timeline.setStyleSheet(
                f"QFrame {{ background: {get_hud_bottom_strip_bg_css()}; border-top: 1px solid {get_hud_strip_border_css()}; }}"
            )
        if hasattr(self, "_tl_scroll"):
            self._tl_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
            self._tl_scroll.viewport().setStyleSheet("background: transparent;")
        if hasattr(self, "_tl_inner"):
            self._tl_inner.setStyleSheet("background: transparent;")
        if hasattr(self, "_bottom_hud_restore"):
            self._bottom_hud_restore.setStyleSheet(self._hud_restore_button_css(top_edge=False))
        if hasattr(self, "_bottom_hud_dismiss"):
            self._bottom_hud_dismiss.setStyleSheet(self._hud_dismiss_button_css(strong=True))
        if hasattr(self, "_hud_ghost_buttons"):
            ghost_css = self._hud_ghost_button_css()
            for button in self._hud_ghost_buttons:
                button.setStyleSheet(ghost_css)
        metric_chip_css = self._hud_text_chip_css()
        if hasattr(self, "_side_metrics_left"):
            self._side_metrics_left.setStyleSheet(metric_chip_css + " letter-spacing: 0.9px;")
        if hasattr(self, "_side_metrics_right"):
            self._side_metrics_right.setStyleSheet(metric_chip_css + " letter-spacing: 0.9px;")
        if hasattr(self, "_scrub_play_btn"):
            self._scrub_play_btn.setStyleSheet(self._hud_scrub_button_css())
        if hasattr(self, "_scrub_speed_btn"):
            self._scrub_speed_btn.setStyleSheet(self._hud_scrub_button_css())
        if hasattr(self, "_scrub_speed_value"):
            self._scrub_speed_value.setStyleSheet(metric_chip_css)
        if hasattr(self, "_scrub_time_left"):
            self._scrub_time_left.setStyleSheet(metric_chip_css)
        if hasattr(self, "_scrub_time_right"):
            self._scrub_time_right.setStyleSheet(metric_chip_css)
        if hasattr(self, "_scrub_speed_slider"):
            self._scrub_speed_slider.setStyleSheet(get_hud_slider_css(handle_size=8, handle_radius=4))
        if hasattr(self, "_scrub_slider"):
            self._scrub_slider.setStyleSheet(get_hud_slider_css(handle_size=10, handle_radius=5))
        if hasattr(self, "_tl_count"):
            self._tl_count.setStyleSheet(f"color: {text_hex()};")
        if hasattr(self, "_toast"):
            self._toast.setStyleSheet(self._hud_toast_css())
        if hasattr(self, "_focus_burst_identity"):
            self._focus_burst_identity.setStyleSheet(
                f"QLineEdit {{ background: {theme_rgba('input_fill', 0.72)}; "
                f"border: 1px solid {theme_rgba('accent_dark', 0.22)}; "
                f"color: {surface_muted_css(0.92)}; padding: 4px 6px; font-size: 10px; }}"
                f"QLineEdit:focus {{ border-color: {theme_rgba('accent_dark', 0.56)}; }}"
            )
        if hasattr(self, "_video"):
            self._video.refresh_roi_rail_theme()

    def _apply_wear_layout(self) -> None:
        if not getattr(self, "_wear_ui_enabled", False):
            return
        if hasattr(self, "_top_panel_toggle"):
            self._top_panel_toggle.hide()
        self._refresh_button_caption(self._btn_scan, "[S] Scan")
        self._refresh_button_caption(self._btn_radial, "[+] More")
        self._btn_radial.setToolTip("More actions")
        self._btn_radial.setProperty("hudCircle", False)
        self._btn_roi.setProperty("hudCircle", False)
        self._btn_radial.setMinimumWidth(96)
        self._btn_radial.setMaximumWidth(140)
        self._btn_radial.setMinimumHeight(42)
        self._btn_radial.setMaximumHeight(42)
        self._btn_scan.setMinimumWidth(120)
        self._btn_scan.setMinimumHeight(42)
        self._btn_roi.setMinimumWidth(84)
        self._btn_roi.setMaximumWidth(84)
        self._btn_roi.setMinimumHeight(42)
        for widget in (
            self._btn_active,
            self._btn_active_heat,
            self._btn_source_swap,
            self._btn_tab_quick,
            self._btn_filter,
            self._roi_preset_wrap,
            self._bottom_hud_dismiss,
            self._btn_suite,
            self._btn_subgrid,
            self._btn_grid,
            self._btn_nreal,
            self._btn_timeline,
            self._btn_data,
            self._btn_catalog,
            *self._quick_filter_buttons.values(),
        ):
            widget.hide()

    def _handle_primary_scan_action(self) -> None:
        self._reveal_focus_sidebar()
        if self._roi_capture_payload:
            self._refresh_roi_capture()

    def __init__(self, config: RuntimeConfig) -> None:
        super().__init__()
        density_mode = str(os.environ.get("INSIGHT_UI_DENSITY", "operator")).strip().lower()
        if density_mode not in {"operator", "scout"}:
            density_mode = "operator"
        self.setObjectName("insightMainWindow")
        self.setProperty("densityMode", density_mode)
        self.setWindowTitle("Insight — Local")
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.resize(1400, 900)
        self._config = config
        self.session = LocalInsightSession(config, defer_boot=True)
        self.session.hud_payload.connect(self._on_hud_payload)

        self._last_hud: dict[str, Any] = {}
        self._system_health: list[dict[str, Any]] = []
        self._capability_report: dict[str, Any] = {}
        self._operating_mode = "boot"
        self._recovery_events: list[dict[str, Any]] = []
        self._last_preview_tiles: list[dict[str, Any]] = []
        self._last_event_items: list[dict[str, Any]] = []
        self._last_timeline_entries: list[dict[str, Any]] = []
        self._preview_mode = "preview"
        self._tile_cache: dict[int, tuple[dict, float]] = {}
        self._preview_widgets: dict[str, PreviewTileWidget] = {}
        self._preview_empty: Optional[QLabel] = None
        self._preview_live_header: Optional[QLabel] = None
        self._preview_past_header: Optional[QLabel] = None
        self._preview_filter_text = ""
        self._quick_filters: set[str] = set()
        self._quick_filter_toggle_seq: dict[str, int] = {}
        self._active_mode = False
        self._active_heat = False
        self._history_focus_lock = False
        self._tab_peek_active = False
        self._tab_peek_activated: set[str] = set()
        self._tab_takeover_active = False
        self._tab_takeover_snapshot: Optional[dict[str, Any]] = None
        self._hotkey_peek_options: set[str] = {"roi", "sidebar"}
        self._roi_sending = False
        self._source_swap_pending: Optional[dict[str, Any]] = None
        self._focus_zoom = 1.0
        self._focus_payload: Optional[dict[str, Any]] = None
        self._roi_capture_payload: Optional[dict[str, Any]] = None
        self._show_bbox = False
        self._show_heat = False
        self._show_heat_tags = False
        self._labels_only = False
        self._show_scan_previews = False
        self._heat_opacity = 0.55
        self._ai_open = False
        self._ai_busy = False
        self._ai_text = ""
        self._ai_err = ""
        self._ai_provider = "auto"
        self._ai_prompt_draft = ""
        self._ai_models: dict[str, str] = {
            "ollama": INSIGHT_OLLAMA_MODEL,
            "openai": INSIGHT_OPENAI_MODEL,
            "anthropic": INSIGHT_ANTHROPIC_MODEL,
        }
        self._cvops_base_url = str(CVOPS_BASE_URL).rstrip("/")
        self._focus_quick_box_pct: Optional[tuple[float, float, float, float]] = None
        self._focus_quick_scenarios: list[dict[str, Any]] = []
        self._focus_quick_dataset_by_scenario: dict[str, str] = {}
        self._focus_quick_classes_by_dataset: dict[str, list[str]] = {}
        self._focus_quick_class_id_by_dataset: dict[str, int] = {}
        self._focus_quick_loading_scenarios = False
        self._cvops_add_upload_bridge: Optional[_CvopsAddUploadBridge] = None
        self._saved_settings = self.session.get_saved_settings()
        self._saved_settings["tab_takeover_active"] = False
        self._saved_settings["tab_takeover_snapshot"] = {}
        self._send(
            {
                "type": "update_settings",
                "settings": {
                    "tab_takeover_active": False,
                    "tab_takeover_snapshot": {},
                },
            }
        )
        self._caption_buttons: list[QPushButton] = []
        _saved_caption_mode = str(self._saved_settings.get("button_caption_mode", "both") or "both").strip().lower()
        self._button_caption_mode = _saved_caption_mode if _saved_caption_mode in self._BUTTON_CAPTION_MODES else "both"
        self._wear_ui_enabled = True
        configure_color_scheme(
            normalize_color_scheme(self._saved_settings.get("color_scheme", self._config.color_scheme))
        )
        self._focus_base_pixmap = QPixmap()
        self._boot_gate_done = False
        self._video_frame_ticks = 0
        self._edge_background_enabled = True
        self._panel_background_style = "hexagons"
        self._thermal_mode = "edge"  # "edge" | "clouds" | "edge+clouds"
        self._edge_brightness: float = 0.15   # blend alpha 0.05–0.40
        self._edge_fade_alpha: float = 0.0    # animated 0→1
        self._edge_fade_target: float = 0.0
        self._edge_fade_timer = QTimer(self)
        self._edge_fade_timer.setInterval(16)
        self._edge_fade_timer.timeout.connect(self._tick_edge_fade)
        self._last_raw_frame: np.ndarray | None = None
        self._scrub_poll_interval_sec = 0.20
        self._last_scrub_poll_ts = 0.0

        # -- EffectsWorker: dedicated thread for all heavy visual ops --
        self._effects_worker = EffectsWorker()
        # Wired to session.pipeline in _deferred_session_boot after heavy init
        self._effects_worker_pending = True
        self._roi_preset_buttons: list[tuple[float, QPushButton]] = []
        self._swap_frame_pending = False
        self._timeline_signature: tuple[Any, ...] = ()
        self._recog_result_widget: Optional[QWidget] = None

        self.setStyleSheet(get_global_stylesheet())

        central = QWidget()
        central.setObjectName("centralHud")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self._top_panel_toggle = QWidget(central)
        self._top_panel_toggle.setObjectName("topPanelToggle")
        top_toggle = QHBoxLayout(self._top_panel_toggle)
        top_toggle.setContentsMargins(0, 0, 0, 0)
        top_toggle.setSpacing(8)

        # [SUITES] manager only — button lives in the bottom HUD
        self._suite_manager = SuiteManager()

        self._side_metrics_left = QLabel(central)
        self._side_metrics_right = QLabel(central)
        for label in (self._side_metrics_left, self._side_metrics_right):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            label.setAlignment(Qt.AlignmentFlag.AlignTop)
            label.hide()

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._video = VideoPane()
        self._video.roi_capture_requested.connect(self._on_roi_capture_dblclick)
        self._video.roi_norm_changed.connect(self._push_roi_if_active)
        self._video.roi_preset_requested.connect(self._apply_roi_preset)
        self._video.track_clicked.connect(self._on_video_track_clicked)

        # Register all heavy visual effects on the dedicated worker thread.
        self._effects_worker.register_effect(
            "edge_background", self._edge_bg_for_worker,
        )
        self._effects_worker.start()

        self._swap_banner = QFrame()
        self._swap_banner.setObjectName("swapBanner")
        self._swap_banner.setStyleSheet(
            f"QFrame {{ background: {theme_metallic('panel', 0.94)}; border: 1px solid {theme_rgba('accent_dark', 0.40)}; }}"
        )
        self._swap_banner.setMinimumWidth(420)
        sl = QVBoxLayout(self._swap_banner)
        self._swap_title = QLabel("Ready To Swap")
        self._swap_title.setStyleSheet("color: #140808; font-weight: 600;")
        self._swap_copy = QLabel("")
        self._swap_copy.setWordWrap(True)
        self._swap_copy.setStyleSheet("color: rgba(20,8,8,0.82); font-size: 11px;")
        bh = QHBoxLayout()
        self._swap_stay = self._make_button("Stay Here", mode="title")
        self._swap_go = self._make_button("Swap View", mode="title")
        self._swap_go.setStyleSheet(self._hud_ghost_button_css())
        self._swap_stay.clicked.connect(self._cancel_source_swap_ui)
        self._swap_go.clicked.connect(self._confirm_source_swap_ui)
        bh.addWidget(self._swap_stay)
        bh.addWidget(self._swap_go)
        sl.addWidget(self._swap_title)
        sl.addWidget(self._swap_copy)
        sl.addLayout(bh)

        self._loading_gate = LoadingGateOverlay()
        self._video_host = VideoHost(self._video, self._swap_banner, self._loading_gate)
        self._grid_cell_sources: dict[int, dict] = {}
        # [SUITES] restore grid cells from the active suite (if any)
        active = self._suite_manager.active_suite
        if active:
            self._grid_cell_sources = dict(active.grid_cells)
        self._pending_grid_restore = bool(self._grid_cell_sources)
        self._video_host.grid_overlay.cell_add_requested.connect(self._on_grid_cell_add)
        self._video_host.grid_overlay.cell_settings_requested.connect(self._on_grid_cell_settings)
        # Cmd+click on a cell widget opens its settings menu.
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        self._global_event_filter_installed = app is not None
        if app is not None:
            app.installEventFilter(self)
        body.addWidget(self._video_host, stretch=1)

        self._tabs = QTabWidget()
        self._tabs.setObjectName("orbitalTabs")
        self._tabs.currentChanged.connect(lambda _index: self._refresh_sidebar_views())
        self._previews_host = QWidget()
        self._previews_host.setObjectName("previewsHost")
        self._previews_host_layout = QVBoxLayout(self._previews_host)
        self._previews_host_layout.setContentsMargins(0, 0, 0, 0)
        self._previews_host_layout.setSpacing(6)
        self._preview_filter = QLineEdit()
        self._preview_filter.setPlaceholderText("Filter previews and overlays: person, cat, human...")
        self._preview_filter.setStyleSheet(
            f"QLineEdit {{ background: {theme_rgba('panel', 0.12)}; border: 1px solid {theme_rgba('accent_dark', 0.32)}; "
            f"color: {text_css(1.0)}; padding: 5px 8px; font-size: 10px; }}"
        )
        self._preview_filter.textChanged.connect(self._on_preview_filter_changed)
        self._previews_host_layout.addWidget(self._preview_filter)
        self._preview_mode_btn = QPushButton("Mode: Preview")
        self._preview_mode_btn.setCheckable(True)
        self._preview_mode_btn.setToolTip("Switch live cards between full preview and ROI selective mode")
        self._preview_mode_btn.setStyleSheet(self._hud_ghost_button_css())
        self._preview_mode_btn.clicked.connect(self._toggle_preview_mode)
        self._previews_host_layout.addWidget(self._preview_mode_btn)
        self._previews_scroll = QScrollArea()
        self._previews_scroll.setWidgetResizable(True)
        self._previews_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._previews_scroll.viewport().setStyleSheet("background: transparent;")
        self._previews_inner = QWidget()
        self._previews_inner.setStyleSheet("background: transparent;")
        self._previews_layout = QVBoxLayout(self._previews_inner)
        self._previews_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._previews_layout.setSpacing(10)
        self._previews_scroll.setWidget(self._previews_inner)
        self._previews_host_layout.addWidget(self._previews_scroll, stretch=1)
        self._tabs.addTab(self._previews_host, "Live")

        focus_scroll = QScrollArea()
        focus_scroll.setWidgetResizable(True)
        focus_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        focus_scroll.viewport().setStyleSheet("background: transparent;")
        focus_w = QWidget()
        focus_w.setObjectName("focusRoot")
        focus_w.setStyleSheet("background: transparent;")
        flayout = QVBoxLayout(focus_w)
        flayout.setContentsMargins(0, 0, 0, 0)
        flayout.setSpacing(0)
        self._focus_empty = QLabel("Select a live or past preview card to inspect it.")
        self._focus_empty.setWordWrap(True)
        self._focus_empty.setStyleSheet("color: rgba(20,8,8,0.68); padding: 24px;")
        flayout.addWidget(self._focus_empty)
        self._focus_body = QWidget()
        self._focus_body.setObjectName("focusBody")
        fbl = QVBoxLayout(self._focus_body)
        fbl.setContentsMargins(8, 4, 8, 4)
        fbl.setSpacing(4)
        head = QHBoxLayout()
        head.setContentsMargins(0, 0, 0, 0)
        self._focus_title = QLabel("")
        self._focus_title.setStyleSheet("font-size: 15px; font-weight: 700; color: #140808;")
        self._focus_sub = QLabel("")
        self._focus_sub.setStyleSheet("font-size: 10px; color: rgba(20,8,8,0.72);")
        clear_btn = self._make_button("Clear", mode="title")
        clear_btn.setStyleSheet(self._hud_ghost_button_css())
        clear_btn.clicked.connect(self._clear_focus_ui)
        self._focus_cvops_submit = self._make_button("[CV] Queue Job", mode="both")
        self._focus_cvops_submit.setStyleSheet(self._hud_ghost_button_css())
        self._focus_cvops_submit.clicked.connect(self._on_submit_cvops_job)
        head.addWidget(self._focus_title, stretch=1)
        head.addWidget(self._focus_cvops_submit)
        head.addWidget(clear_btn)
        fbl.addLayout(head)
        fbl.addWidget(self._focus_sub)
        ctrl_top = QHBoxLayout()
        ctrl_top.setContentsMargins(0, 0, 0, 0)
        ctrl_top.setSpacing(4)
        self._focus_zoom_out = self._make_button(
            "[-] Zoom Out", mode="icon", icon_name="remove", icon_size=18, tooltip="Zoom out"
        )
        self._focus_zoom_in = self._make_button(
            "[+] Zoom In", mode="icon", icon_name="add", icon_size=18, tooltip="Zoom in"
        )
        self._focus_zoom_lbl = QLabel("100%")
        self._focus_zoom_lbl.setStyleSheet(f"color: {HUD_MUTED.name()}; font-size: 9px;")
        self._focus_zoom_out.clicked.connect(lambda: self._bump_focus_zoom(-0.2))
        self._focus_zoom_in.clicked.connect(lambda: self._bump_focus_zoom(0.2))
        ctrl_top.addWidget(self._focus_zoom_out)
        ctrl_top.addWidget(self._focus_zoom_lbl)
        ctrl_top.addWidget(self._focus_zoom_in)
        ctrl_top.addStretch(1)
        ctrl_btm = QHBoxLayout()
        ctrl_btm.setContentsMargins(0, 0, 0, 0)
        ctrl_btm.setSpacing(4)
        self._focus_refresh = self._make_button(
            "[R] Refresh", mode="both", icon_name="refresh", icon_size=18, tooltip="Refresh focus view"
        )
        self._focus_recognize = self._make_button("[ID] Recognize", mode="both")
        self._focus_burst_identity = QLineEdit()
        self._focus_burst_identity.setPlaceholderText("identity")
        self._focus_burst_identity.setText("new_hire")
        self._focus_burst_identity.setMaximumWidth(92)
        self._focus_burst_identity.setToolTip("Identity for ROI burst enrollment")
        self._focus_burst_identity.setStyleSheet(
            f"QLineEdit {{ background: {theme_rgba('input_fill', 0.72)}; "
            f"border: 1px solid {theme_rgba('accent_dark', 0.22)}; "
            f"color: {surface_muted_css(0.92)}; padding: 4px 6px; font-size: 10px; }}"
            f"QLineEdit:focus {{ border-color: {theme_rgba('accent_dark', 0.56)}; }}"
        )
        self._focus_burst_enroll = self._make_button("[+] Burst Enroll", mode="both")
        self._focus_recognize.setToolTip("Run attendance recognition on this image")
        self._focus_burst_enroll.setToolTip("Capture a short ROI burst and enroll multiple face samples")
        self._focus_refresh.clicked.connect(self._refresh_roi_capture)
        self._focus_recognize.clicked.connect(self._on_manual_recognize)
        self._focus_burst_enroll.clicked.connect(self._on_roi_burst_enroll)
        ctrl_btm.addWidget(self._focus_refresh)
        ctrl_btm.addWidget(self._focus_recognize)
        ctrl_btm.addWidget(self._focus_burst_identity)
        ctrl_btm.addWidget(self._focus_burst_enroll)
        ctrl_btm.addStretch(1)
        quick_row = QHBoxLayout()
        quick_row.setContentsMargins(0, 0, 0, 0)
        quick_row.setSpacing(4)
        self._focus_quick_detect_btn = self._make_button("Detect Face", mode="both")
        self._focus_quick_draw_btn = self._make_button("Draw Box", mode="both", checkable=True)
        self._focus_quick_draw_btn.toggled.connect(self._on_focus_quick_draw_toggled)
        self._focus_quick_clear_btn = self._make_button("Clear Box", mode="both")
        self._focus_quick_clear_btn.clicked.connect(self._on_focus_quick_clear_box)
        self._focus_quick_scenario_combo = QComboBox()
        self._focus_quick_scenario_combo.setMinimumWidth(200)
        self._focus_quick_scenario_combo.setToolTip("Target scenario for quick label upload/update")
        self._focus_quick_scenario_combo.currentIndexChanged.connect(self._on_focus_quick_scenario_changed)
        self._focus_quick_class_combo = QComboBox()
        self._focus_quick_class_combo.setMinimumWidth(140)
        self._focus_quick_class_combo.setToolTip("Label class for quick dataset add")
        self._focus_quick_mode_combo = QComboBox()
        self._focus_quick_mode_combo.addItem("Box Label", "box")
        self._focus_quick_mode_combo.addItem("Full Image", "full")
        self._focus_quick_mode_combo.setToolTip("Use the drawn/detected box or label full image")
        self._focus_quick_split_combo = QComboBox()
        self._focus_quick_split_combo.addItem("Train", "train")
        self._focus_quick_split_combo.addItem("Val", "val")
        self._focus_quick_split_combo.setToolTip("Dataset split for this labeled sample")
        self._focus_quick_add_btn = self._make_button("Quick Add", mode="both")
        self._focus_quick_sendoff_btn = self._make_button("Send Off", mode="both")
        self._focus_quick_detect_btn.clicked.connect(self._on_focus_quick_detect_face)
        self._focus_quick_add_btn.clicked.connect(lambda: self._on_focus_quick_submit(auto_update=False))
        self._focus_quick_sendoff_btn.clicked.connect(lambda: self._on_focus_quick_submit(auto_update=True))
        quick_row.addWidget(self._focus_quick_detect_btn)
        quick_row.addWidget(self._focus_quick_draw_btn)
        quick_row.addWidget(self._focus_quick_clear_btn)
        quick_row.addWidget(self._focus_quick_scenario_combo, stretch=1)
        quick_row.addWidget(self._focus_quick_class_combo)
        quick_row.addWidget(self._focus_quick_mode_combo)
        quick_row.addWidget(self._focus_quick_split_combo)
        quick_row.addWidget(self._focus_quick_add_btn)
        quick_row.addWidget(self._focus_quick_sendoff_btn)
        ctrl_layout = QVBoxLayout()
        ctrl_layout.setContentsMargins(0, 0, 0, 0)
        ctrl_layout.setSpacing(2)
        ctrl_layout.addLayout(ctrl_top)
        ctrl_layout.addLayout(ctrl_btm)
        ctrl_layout.addLayout(quick_row)
        self._focus_controls = QWidget()
        self._focus_controls.setLayout(ctrl_layout)
        self._focus_controls.setStyleSheet(self._hud_ghost_button_css())
        fbl.addWidget(self._focus_controls)
        self._focus_visual = FocusVisualHost()
        self._focus_visual.setMinimumHeight(120)
        self._focus_visual.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._focus_visual.setStyleSheet(f"background: {theme_rgba('panel', 0.26)}; border: 1px solid {theme_rgba('accent_dark', 0.45)};")
        self._focus_composite = self._focus_visual.composite
        self._focus_overlay = self._focus_visual.overlay
        self._focus_overlay.quick_box_changed.connect(self._on_focus_quick_box_changed)
        fbl.addWidget(self._focus_visual)
        self._focus_stats = QWidget()
        self._focus_stats.setStyleSheet(
            f"QWidget#focusStats {{ background: {theme_rgba('accent_dark', 0.22)}; "
            f"border-top: 1px solid {theme_rgba('accent_dark', 0.25)}; }}"
        )
        self._focus_stats.setObjectName("focusStats")
        self._focus_stats_layout = QGridLayout(self._focus_stats)
        self._focus_stats_layout.setContentsMargins(8, 6, 8, 6)
        self._focus_stats_layout.setHorizontalSpacing(8)
        self._focus_stats_layout.setVerticalSpacing(4)
        fbl.addWidget(self._focus_stats)
        self._focus_scan_host = QWidget()
        self._focus_scan_host.setObjectName("focusScanHost")
        self._focus_scan_host.setStyleSheet(self._hud_ghost_button_css())
        self._focus_scan_layout = QVBoxLayout(self._focus_scan_host)
        self._focus_scan_layout.setContentsMargins(0, 0, 0, 0)
        self._focus_scan_layout.setSpacing(6)
        fbl.addWidget(self._focus_scan_host)
        self._focus_body.hide()
        flayout.addWidget(self._focus_body)
        focus_scroll.setWidget(focus_w)
        self._tabs.addTab(focus_scroll, "Focus")

        ev_scroll = QScrollArea()
        ev_scroll.setWidgetResizable(True)
        self._events_inner = QWidget()
        self._events_inner.setObjectName("eventsHost")
        self._events_layout = QVBoxLayout(self._events_inner)
        self._events_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._events_layout.setSpacing(6)
        ev_scroll.setWidget(self._events_inner)
        self._tabs.addTab(ev_scroll, "History")

        self._catalog_tabs = QTabWidget()
        self._catalog_tabs.setObjectName("catalogOrbitalTabs")
        self._catalog_tabs.currentChanged.connect(lambda _index: self._refresh_catalog_views())
        self._gallery_panel = GalleryPanel(
            send_message=self._send,
            get_gallery_images=lambda name: self.session.gallery.get_identity_images(name),
        )
        self._catalog_tabs.addTab(self._gallery_panel, "Gallery")

        self._settings_panel = SettingsPanel(
            send_message=self._send,
            storage_paths={
                "App Folder": ROOT_DIR,
                "Gallery DB": GALLERY_DB_PATH,
                "State Folder": self.session.config.state_dir,
            },
        )
        self._settings_panel.ai_provider_changed.connect(self._on_settings_ai_provider)
        self._settings_panel.ai_model_changed.connect(self._on_settings_ai_model)
        self._settings_panel.bbox_style_changed.connect(self._on_settings_bbox_style)
        self._settings_panel.panel_background_style_changed.connect(self._on_settings_panel_background_style)
        self._settings_panel.edge_background_changed.connect(self._on_settings_edge_background)
        self._settings_panel.edge_brightness_changed.connect(self._on_settings_edge_brightness)
        self._settings_panel.heat_tags_changed.connect(self._on_settings_heat_tags)
        self._settings_panel.labels_only_changed.connect(self._on_settings_labels_only)
        self._settings_panel.thermal_mode_changed.connect(self._on_settings_thermal_mode)
        self._settings_panel.button_caption_mode_changed.connect(self._on_settings_button_caption_mode)
        self._settings_panel.fps_changed.connect(self._on_settings_fps_changed)
        self._settings_panel.color_scheme_changed.connect(self._on_settings_color_scheme)
        self._settings_panel.hotkey_peek_changed.connect(self._on_hotkey_peek_changed)
        self._settings_panel.set_ai_selection(self._ai_provider, self._ai_models)
        self._restore_settings_panel_state()
        self._catalog_tabs.addTab(self._settings_panel, "Controls")

        self._workflow_tab = _WorkflowTab()
        self._workflow_tab.cancel_requested.connect(self._close_inline_cvops_add_flow)
        self._workflow_tab.send_requested.connect(self._submit_inline_cvops_add_flow)
        self._catalog_tabs.addTab(self._workflow_tab, "Workflow")

        self._subroutine_panel = TestRangePanel(
            http_get=lambda path: self._cvops_http_json("GET", path),
            http_post=lambda path, payload=None: self._cvops_http_json("POST", path, payload),
            http_delete=lambda path: self._cvops_http_json("DELETE", path),
            parent=self,
        )
        self._subroutine_panel.errorRaised.connect(
            lambda msg: print(f"[Subroutine] {msg}", flush=True)
        )
        self._catalog_tabs.addTab(self._subroutine_panel, "Subroutine")

        self._sidebar_x: int = -1  # -1 = default right-edge anchor
        self._sidebar_panel = SidebarPanel(self._tabs, self.centralWidget())
        self._sidebar_panel.visibility_changed.connect(self._on_sidebar_visibility)
        self._sidebar_panel.moved.connect(self._on_sidebar_moved)
        self._sidebar_panel.width_changed.connect(self._on_sidebar_width_changed)
        self._sidebar_panel.hide()
        # [SESSION] Restore saved sidebar state
        _saved_sw = int(self._saved_settings.get("sidebar_width", 320))
        self._sidebar_panel._saved_width = max(240, min(560, _saved_sw))
        self._pending_sidebar_restore = False
        if not self._pending_sidebar_restore:
            self._on_sidebar_visibility(False)

        # [CATALOG] Second data column
        self._catalog_x: int = -1
        self._catalog_panel = SidebarPanel(self._catalog_tabs, self.centralWidget())
        self._catalog_panel.set_handle_side("right")
        self._catalog_panel.visibility_changed.connect(self._on_catalog_visibility)
        self._catalog_panel.moved.connect(self._on_catalog_moved)
        self._catalog_panel.width_changed.connect(self._on_catalog_width_changed)
        self._catalog_panel.hide()
        _saved_cw = int(self._saved_settings.get("catalog_width", 300))
        self._catalog_panel._saved_width = max(240, min(560, _saved_cw))
        self._pending_catalog_restore = False
        if not self._pending_catalog_restore:
            self._on_catalog_visibility(False)
        self._apply_panel_background_style(self._panel_background_style)

        root.addLayout(body, stretch=1)


        self._bottom_hud = _OpaqueHudStrip(top_edge=False, parent=central)
        self._bottom_hud.setObjectName("bottomHud")
        self._bottom_hud.apply_strip_colors(
            get_hud_bottom_strip_bg_css(),
            theme_rgba("accent_dark", 0.22),
        )
        self._bottom_hud.setStyleSheet(
            "QWidget#bottomHud { background: transparent; border: none; }"
            "QWidget#bottomHud > QWidget { background: transparent; border: none; }"
        )
        _bottom_root = QVBoxLayout(self._bottom_hud)
        _bottom_root.setContentsMargins(0, 0, 0, 0)
        _bottom_root.setSpacing(0)
        bottom = QHBoxLayout()
        bottom.setContentsMargins(16, 10, 16, 12)
        bottom.setSpacing(10)
        _ghost_btn = self._hud_ghost_button_css()
        self._btn_scan = self._make_button("[S] Scan", mode="icon", style=_ghost_btn, hud=True, tooltip="Open scan focus")
        self._btn_scan.clicked.connect(self._handle_primary_scan_action)
        self._btn_roi = self._make_button("[R] ROI", mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip="ROI")
        self._btn_roi.clicked.connect(self._toggle_roi)
        self._roi_preset_wrap = QWidget()
        rpw = QHBoxLayout(self._roi_preset_wrap)
        rpw.setContentsMargins(0, 0, 0, 0)
        self._roi_preset_group = QButtonGroup(self)
        self._roi_preset_group.setExclusive(True)
        for label, scale in [(".5x", 0.5), ("1x", 1.0), ("2x", 2.0), ("3x", 3.0)]:
            b = self._make_button(label, mode="icon", checkable=True, hud=True, hud_circle=True, tooltip=f"ROI preset {label}")
            self._roi_preset_group.addButton(b)
            b.clicked.connect(lambda _c, s=scale: self._apply_roi_preset(s))
            self._roi_preset_buttons.append((scale, b))
            rpw.addWidget(b)
        self._roi_preset_wrap.hide()
        self._btn_timeline = self._make_button("[T] Timeline", mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip="Timeline")
        self._btn_timeline.clicked.connect(self._toggle_timeline)
        self._btn_timeline.hide()
        self._btn_grid = self._make_button("[G] Grid", mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip="Grid")
        self._btn_grid.clicked.connect(self._toggle_grid)
        self._btn_subgrid = self._make_button("[Q] Quads", mode="icon", style=_ghost_btn, hud=True, hud_circle=True, tooltip="Subgrid")
        self._btn_subgrid.clicked.connect(self._toggle_subgrid)
        self._btn_subgrid.hide()
        self._sync_subgrid_button()
        self._btn_data = self._make_button("[D] Data", mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip="Data panel")
        self._btn_data.clicked.connect(self._toggle_sidebar)
        self._btn_catalog = self._make_button("[C] Catalog", mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip="Catalog panel")
        self._btn_catalog.clicked.connect(self._toggle_catalog)
        top_toggle.addWidget(self._btn_catalog)
        top_toggle.addWidget(self._btn_data)
        self._btn_nreal = self._make_button("[N] NREAL", mode="icon", style=_ghost_btn, hud=True, hud_circle=True, tooltip="NREAL fit")
        self._btn_nreal.clicked.connect(self._fit_nreal)
        self._btn_active = self._make_button("[A] Active", mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip="Active mode")
        self._btn_active.clicked.connect(self._toggle_active_mode)
        self._btn_active_heat = self._make_button("[H] Thermal", mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip="Thermal mode")
        self._btn_active_heat.clicked.connect(self._toggle_active_heat)
        self._btn_source_swap = self._make_button("[V] Source", mode="icon", style=_ghost_btn, hud=True, hud_circle=True, tooltip="Swap source")
        self._btn_source_swap.clicked.connect(self._request_source_switch)
        self._btn_radial = self._make_button("AB", mode="icon", style=_ghost_btn, tooltip="Open radial menu", hud=True, hud_circle=True)
        self._btn_radial.clicked.connect(self._open_radial_menu_from_button)
        self._btn_tab_quick = self._make_button("Tab", mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip="Quick scan takeover")
        self._btn_tab_quick.toggled.connect(self._on_tab_takeover_toggled)
        self._quick_filter_buttons: dict[str, QPushButton] = {}
        self._quick_filter_keys: list[tuple[str, str]] = [
            ("people", "People"),
            ("animals", "Animals"),
            ("tech", "Tech"),
            ("objects", "Objects"),
        ]
        for key, label in self._quick_filter_keys:
            btn = self._make_button(label, mode="icon", checkable=True, style=_ghost_btn, hud=True, hud_circle=True, tooltip=label)
            btn.toggled.connect(lambda checked, k=key: self._toggle_quick_filter(k, checked))
            self._quick_filter_buttons[key] = btn
        for key, _label in self._quick_filter_keys:
            self._set_quick_filter_button_status(key, "none")
        self._btn_filter = self._make_button("[F] Filter", mode="icon", style=_ghost_btn, hud=True, hud_circle=True, tooltip="Filter menu")
        self._btn_filter.clicked.connect(self._show_filter_menu)
        self._bottom_hud_dismiss = self._make_button(
            "[X] Dismiss", mode="icon", icon_name="close", icon_size=18, tooltip="Dismiss bottom HUD", hud=True
        )
        self._bottom_hud_dismiss.setFixedSize(30, 30)
        self._bottom_hud_dismiss.setStyleSheet(self._hud_dismiss_button_css(strong=True))
        self._bottom_hud_dismiss.clicked.connect(self._toggle_bottom_hud)
        # [SUITES] suite picker button — lives at the center seam of the bottom HUD
        _active_suite = self._suite_manager.active_suite
        self._btn_suite = self._make_button(
            f"[S] {_active_suite.name}" if _active_suite else "[S] Suite",
            mode="icon",
            style=_ghost_btn,
            hud=True,
            hud_circle=True,
            tooltip="Suite",
        )
        self._btn_suite.clicked.connect(self._show_suite_menu)
        bottom.addStretch(1)
        bottom.addWidget(self._btn_scan)
        bottom.addWidget(self._btn_roi)
        bottom.addWidget(self._btn_radial)
        bottom.addStretch(1)
        _bottom_root.addLayout(bottom)

        # --- Video scrub row (embedded in bottom HUD, visible only for pre-recorded video) ---
        self._scrub_bar = QWidget()
        self._scrub_bar.setFixedHeight(28)
        self._scrub_bar.setStyleSheet(
            f"border-top: 1px solid {get_hud_strip_border_css()}; background: transparent;"
        )
        sb_layout = QHBoxLayout(self._scrub_bar)
        sb_layout.setContentsMargins(10, 2, 10, 2)
        sb_layout.setSpacing(6)
        self._scrub_play_btn = self._make_button(
            "[||] Play", mode="icon", icon_name="play_arrow", icon_size=20, tooltip="Play or pause", hud=True
        )
        self._scrub_play_btn.setFixedSize(30, 30)
        self._scrub_play_btn.setStyleSheet(self._hud_scrub_button_css())
        self._scrub_play_btn.clicked.connect(self._toggle_video_pause)
        sb_layout.addWidget(self._scrub_play_btn)
        self._scrub_speed_btn = self._make_button("[SPD] Speed", mode="icon", checkable=True, tooltip="Toggle speed control", hud=True)
        self._scrub_speed_btn.setFixedSize(30, 30)
        self._scrub_speed_btn.setStyleSheet(self._hud_scrub_button_css())
        self._scrub_speed_btn.toggled.connect(self._toggle_scrub_speed)
        sb_layout.addWidget(self._scrub_speed_btn)
        self._scrub_speed_value = QLabel("1.0x")
        self._scrub_speed_value.setFixedWidth(44)
        self._scrub_speed_value.setStyleSheet(self._hud_text_chip_css())
        self._scrub_speed_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb_layout.addWidget(self._scrub_speed_value)
        self._scrub_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self._scrub_speed_slider.setRange(-50, 50)
        self._scrub_speed_slider.setSingleStep(1)
        self._scrub_speed_slider.setPageStep(5)
        self._scrub_speed_slider.setValue(10)
        self._scrub_speed_slider.setFixedWidth(120)
        self._scrub_speed_slider.setStyleSheet(get_hud_slider_css(handle_size=8, handle_radius=4))
        self._scrub_speed_slider.valueChanged.connect(self._on_scrub_speed_changed)
        self._scrub_speed_slider.hide()
        sb_layout.addWidget(self._scrub_speed_slider)
        self._scrub_time_left = QLabel("0:00")
        self._scrub_time_left.setFixedWidth(52)
        self._scrub_time_left.setStyleSheet(self._hud_text_chip_css())
        self._scrub_time_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb_layout.addWidget(self._scrub_time_left)
        self._scrub_slider = QSlider(Qt.Orientation.Horizontal)
        self._scrub_slider.setRange(0, 1000)
        self._scrub_slider.setStyleSheet(get_hud_slider_css(handle_size=10, handle_radius=5))
        self._scrub_slider.sliderPressed.connect(self._scrub_pressed)
        self._scrub_slider.sliderReleased.connect(self._scrub_released)
        self._scrub_slider.sliderMoved.connect(self._scrub_moved)
        self._scrub_dragging = False
        sb_layout.addWidget(self._scrub_slider)
        self._scrub_time_right = QLabel("0:00")
        self._scrub_time_right.setFixedWidth(58)
        self._scrub_time_right.setStyleSheet(self._hud_text_chip_css())
        self._scrub_time_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sb_layout.addWidget(self._scrub_time_right)
        _bottom_root.addWidget(self._scrub_bar)
        self._scrub_bar.hide()

        self._bottom_hud_restore = self._make_button("CTRL", mode="icon", hud=True, hud_circle=True, tooltip="Restore controls")
        self._bottom_hud_restore.setFixedSize(30, 30)
        self._bottom_hud_restore.hide()
        self._bottom_hud_restore.setStyleSheet(self._hud_restore_button_css(top_edge=False))
        self._bottom_hud_restore.clicked.connect(self._toggle_bottom_hud)
        self._bottom_hud_restore.setParent(central)

        self._timeline = QFrame(central)
        self._timeline.setFixedHeight(180)
        self._timeline.setStyleSheet(
            f"QFrame {{ background: {get_hud_bottom_strip_bg_css()}; border-top: 1px solid {get_hud_strip_border_css()}; }}"
        )
        self._timeline.hide()
        tl = QVBoxLayout(self._timeline)
        hdr = QHBoxLayout()
        hdr.addWidget(QLabel("Detection History"))
        self._tl_count = QLabel("0")
        self._tl_count.setStyleSheet(f"color: {text_hex()};")
        hdr.addWidget(self._tl_count)
        hdr.addStretch(1)
        clr = self._make_button("Clear All", mode="title")
        clr.clicked.connect(lambda: self._send({"type": "clear_history"}))
        clo = self._make_button("Close", mode="title")
        clo.clicked.connect(self._close_timeline)
        hdr.addWidget(clo)
        hdr.addWidget(clr)
        tl.addLayout(hdr)
        self._tl_scroll = QScrollArea()
        self._tl_scroll.setWidgetResizable(True)
        self._tl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self._tl_scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        self._tl_scroll.viewport().setStyleSheet("background: transparent;")
        self._tl_inner = QWidget()
        self._tl_inner.setStyleSheet("background: transparent;")
        self._tl_layout = QHBoxLayout(self._tl_inner)
        self._tl_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._tl_layout.setSpacing(10)
        self._tl_scroll.setWidget(self._tl_inner)
        tl.addWidget(self._tl_scroll)
        self._hud_ghost_buttons = [
            self._btn_scan,
            self._btn_roi,
            self._btn_timeline,
            self._btn_grid,
            self._btn_subgrid,
            self._btn_data,
            self._btn_catalog,
            self._btn_nreal,
            self._btn_active,
            self._btn_active_heat,
            self._btn_source_swap,
            self._btn_radial,
            self._btn_tab_quick,
            self._btn_suite,
            self._btn_filter,
            *self._quick_filter_buttons.values(),
        ]
        self._hud_ghost_buttons.extend(button for _scale, button in self._roi_preset_buttons)
        self._apply_wear_layout()
        self._apply_hud_theme()
        self._position_bottom_overlays()
        self._update_side_metric_visibility()
        self._position_top_panel_toggle()
        self._position_side_metrics()

        self._radial_menu = RadialMenuOverlay(central)
        self._radial_menu.setGeometry(central.rect())
        self._radial_menu.action_triggered.connect(
            lambda key: self._show_toast(f"Menu: {key}", "info")
        )

        self._toast = QLabel(central)
        self._toast.setStyleSheet(self._hud_toast_css())
        self._toast.hide()
        self._toast_timer = QTimer()
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._toast.hide)

        # Wire up loading gate signals — shown only after user picks a source.
        self._loading_gate.cancel_clicked.connect(self._cancel_source_swap_ui)
        self._loading_gate.confirm_clicked.connect(self._confirm_source_swap_ui)
        self._loading_gate.finished.connect(self._on_loading_gate_finished)
        self._apply_panel_glow(self._focus_visual, 24, 44)

        # [BOOT SELECTOR] Show source picker before starting anything.
        self._boot_selector = BootSelectorOverlay(self._config.video_path, parent=central)
        self._boot_selector.source_selected.connect(self._on_boot_source_selected)
        self._boot_selector.raise_()

    def _on_boot_source_selected(self, source: str, value: object) -> None:
        """Called when the user picks an input source from the boot selector."""
        if source == "camera":
            self._config.source = "camera"
            self._config.camera_index = int(value)
        else:
            self._config.source = "video"
            self._config.video_path = value
        self._config.source_locked = True
        self._boot_selector.hide()
        self._loading_gate.set_copy("Starting", "Preparing signaling, transport, and live video feed.")
        self._loading_gate.reset_progress()
        self._loading_gate.set_progress("signal", 1.0)
        self._loading_gate.set_progress("data", 0.2)
        self._loading_gate.show()
        self._loading_gate.raise_()
        self._loading_gate.start_fallback(9000)
        QTimer.singleShot(0, self._deferred_session_boot)

    def _deferred_session_boot(self) -> None:
        """Run heavy session init after the event loop has rendered the loading gate."""
        self.session.boot()
        if self.session.pipeline is not None:
            self.session.pipeline.effects_worker = self._effects_worker
        self.session.handle_client_message({"type": "client_ready"})
        self.session.handle_client_message({"type": "get_gallery_state"})

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._position_bottom_overlays()
        # Restore sidebar/catalog panels after the window is fully laid out so
        # positioning uses correct centralWidget dimensions.
        if getattr(self, "_pending_sidebar_restore", False):
            self._pending_sidebar_restore = False
            QTimer.singleShot(0, self._restore_sidebar)
        if getattr(self, "_pending_catalog_restore", False):
            self._pending_catalog_restore = False
            QTimer.singleShot(0, self._restore_catalog)
        # [SUITES] redeploy grid cell widgets once the window has real geometry
        if getattr(self, "_pending_grid_restore", False):
            self._pending_grid_restore = False
            QTimer.singleShot(0, self._redeploy_all_cell_widgets)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_top_panel_toggle()
        self._position_bottom_overlays()
        self._position_side_metrics()
        self._position_bottom_hud_restore()
        self._position_toast()
        self._position_sidebar()
        self._position_catalog()
        if hasattr(self, "_boot_selector") and self._boot_selector.isVisible():
            cw = self.centralWidget()
            if cw:
                self._boot_selector.setGeometry(cw.rect())
        if hasattr(self, "_radial_menu") and self.centralWidget():
            self._radial_menu.setGeometry(self.centralWidget().rect())

    def _position_top_panel_toggle(self) -> None:
        if hasattr(self, "_top_panel_toggle") and self.centralWidget():
            cw = self.centralWidget()
            self._top_panel_toggle.adjustSize()
            x = (cw.width() - self._top_panel_toggle.width()) // 2
            self._top_panel_toggle.move(max(0, x), 10)
            self._top_panel_toggle.raise_()

    @staticmethod
    def _radial_label_from_caption(caption: str) -> str:
        text = re.sub(r"^\[[^\]]+\]\s*", "", str(caption or "")).strip()
        return text or "Action"

    def _radial_action_for_button(self, key: str, button: QPushButton, fallback: str = "") -> RadialAction:
        raw = str(button.property("caption_text") or button.text() or fallback or key)
        label = self._radial_label_from_caption(raw)
        status = str(button.property("status_state") or "none").strip().lower()
        return RadialAction(
            key=key,
            label=label,
            trigger=button.click,
            icon=button.icon(),
            checked=bool(button.isCheckable() and button.isChecked()),
            enabled=bool(button.isEnabled()),
            status=status,
        )

    def _build_radial_actions(self) -> list[RadialAction]:
        actions: list[RadialAction] = []
        actions.append(self._radial_action_for_button("active", self._btn_active))
        actions.append(self._radial_action_for_button("thermal", self._btn_active_heat))
        actions.append(self._radial_action_for_button("source", self._btn_source_swap))
        actions.append(self._radial_action_for_button("tab_takeover", self._btn_tab_quick, "Tab"))
        actions.append(self._radial_action_for_button("filter", self._btn_filter))
        actions.append(self._radial_action_for_button("roi", self._btn_roi))
        # ROI scale presets live on the inline rail anchored to the ROI.
        actions.append(self._radial_action_for_button("suite", self._btn_suite))
        actions.append(self._radial_action_for_button("grid", self._btn_grid))
        actions.append(self._radial_action_for_button("subgrid", self._btn_subgrid))
        actions.append(self._radial_action_for_button("timeline", self._btn_timeline))
        for key, _label in self._quick_filter_keys:
            qbtn = self._quick_filter_buttons.get(key)
            if qbtn is not None:
                actions.append(self._radial_action_for_button(f"quick_{key}", qbtn, key.title()))
        # Keep these adjacent so Catalog sits between Data and the dev/NREAL action.
        actions.append(self._radial_action_for_button("data", self._btn_data))
        actions.append(self._radial_action_for_button("catalog", self._btn_catalog))
        actions.append(self._radial_action_for_button("nreal", self._btn_nreal))
        actions.append(self._radial_action_for_button("dismiss_hud", self._bottom_hud_dismiss, "Dismiss"))
        return actions

    def _sync_radial_menu_actions(self) -> None:
        if not hasattr(self, "_radial_menu"):
            return
        if not self._radial_menu.isVisible():
            return
        self._radial_menu.sync_actions(self._build_radial_actions())

    def _open_radial_menu(self, center_local: QPoint | None = None) -> None:
        if not self.centralWidget():
            return
        actions = self._build_radial_actions()
        if not actions:
            return
        host = self.centralWidget()
        local_pos = center_local if center_local is not None else host.rect().center()
        self._radial_menu.setGeometry(host.rect())
        self._radial_menu.open_at(local_pos, actions)

    def _open_radial_menu_from_button(self) -> None:
        if not self.centralWidget():
            return
        self._open_radial_menu(self.centralWidget().rect().center())

    def _position_side_metrics(self) -> None:
        """Corner flight labels removed; Live Session block is in Controls."""
        return

    def _position_bottom_hud_restore(self) -> None:
        if self._bottom_hud_restore.isVisible() and self.centralWidget():
            cw = self.centralWidget()
            self._bottom_hud_restore.adjustSize()
            x = (cw.width() - self._bottom_hud_restore.width()) // 2
            y = max(0, cw.height() - self._bottom_hud_restore.height())
            self._bottom_hud_restore.move(max(0, x), y)

    def _position_bottom_overlays(self) -> None:
        if not self.centralWidget():
            return
        cw = self.centralWidget()
        width = max(0, cw.width())
        y = max(0, cw.height())
        if hasattr(self, "_timeline") and self._timeline.isVisible():
            h = max(1, self._timeline.height())
            self._timeline.setGeometry(0, max(0, y - h), width, h)
            self._timeline.raise_()
            y -= h
        if hasattr(self, "_bottom_hud") and self._bottom_hud.isVisible():
            h = max(1, self._bottom_hud.sizeHint().height(), self._bottom_hud.minimumSizeHint().height())
            self._bottom_hud.setGeometry(0, max(0, y - h), width, h)
            self._bottom_hud.raise_()

    def _position_toast(self) -> None:
        self._toast.adjustSize()
        g = self.centralWidget().geometry()
        x = g.center().x() - self._toast.width() // 2
        y = g.bottom() - 80
        self._toast.move(max(8, x), max(8, y))

    def _set_frame_timer_fps(self, fps: int) -> None:
        return

    def _on_settings_fps_changed(self, fps: int) -> None:
        self._set_frame_timer_fps(fps)

    def _persist_setting(self, key: str, value: Any) -> None:
        self._saved_settings[key] = value
        self._send({"type": "update_settings", "settings": {key: value}})

    def _apply_bbox_style(self, style: str) -> None:
        self._video.set_bbox_style(style)
        self._focus_overlay.set_bbox_style(style)

    def _apply_heat_tags_visible(self, visible: bool) -> None:
        self._show_heat_tags = bool(visible)
        self._video.set_heat_labels_visible(self._show_heat_tags)
        scan = self._filtered_scan_results(self._roi_capture_payload.get("scan_results") if self._roi_capture_payload else None)
        self._focus_overlay.set_scan(
            scan or [],
            self._show_bbox,
            self._show_heat,
            self._heat_opacity,
            self._show_heat_tags,
        )

    def _apply_labels_only(self, enabled: bool) -> None:
        self._labels_only = bool(enabled)
        self._video.set_labels_only(self._labels_only)
        self._focus_overlay.set_labels_only(self._labels_only)

    def _apply_edge_background_enabled(self, enabled: bool) -> None:
        self._edge_background_enabled = bool(enabled)

    def _apply_panel_background_style(self, style: str) -> None:
        normalized = SidebarPanel.normalize_background_style(style)
        self._panel_background_style = normalized
        if hasattr(self, "_sidebar_panel"):
            self._sidebar_panel.set_background_style(normalized)
        if hasattr(self, "_catalog_panel"):
            self._catalog_panel.set_background_style(normalized)

    def _apply_thermal_mode(self, mode: str) -> None:
        raw = str(mode or "").strip().lower()
        if raw not in ("edge", "clouds", "edge+clouds"):
            raw = "edge"
        if getattr(self, "_thermal_mode", None) == raw:
            return
        self._thermal_mode = raw
        if hasattr(self, "_video"):
            self._video.thermal_mode = raw
            self._video.invalidate_overlay_cache()

    def _restore_settings_panel_state(self) -> None:
        settings = dict(self._saved_settings or {})
        if not settings:
            settings = {}

        provider = str(settings.get("ai_provider", self._ai_provider) or self._ai_provider)
        if provider in ("auto", "ollama", "openai", "anthropic"):
            self._ai_provider = provider
        for key in ("ollama", "openai", "anthropic"):
            model_key = f"ai_model_{key}"
            value = str(settings.get(model_key, self._ai_models.get(key, "")) or "").strip()
            if value:
                self._ai_models[key] = value
        self._settings_panel.set_ai_selection(self._ai_provider, self._ai_models)

        self._settings_panel.set_combo_value(
            "color_scheme",
            normalize_color_scheme(settings.get("color_scheme", self._config.color_scheme)),
        )
        _caption_mode = str(settings.get("button_caption_mode", self._button_caption_mode) or self._button_caption_mode)
        self._settings_panel.set_combo_value("button_caption_mode", _caption_mode)
        self._apply_button_caption_mode(_caption_mode)

        self._apply_bbox_style(str(settings.get("bbox_style", "square") or "square"))
        self._settings_panel.set_combo_value("bbox_style", self._video.bbox_style)

        panel_style_value = settings.get("panel_background_style", self._panel_background_style)
        self._apply_panel_background_style(str(panel_style_value or self._panel_background_style))
        self._settings_panel.set_combo_value("panel_background_style", self._panel_background_style)

        self._apply_heat_tags_visible(bool(settings.get("heat_tags", self._show_heat_tags)))
        self._settings_panel.set_toggle_value("heat_tags", self._show_heat_tags)

        self._apply_labels_only(bool(settings.get("labels_only", self._labels_only)))
        self._settings_panel.set_toggle_value("labels_only", self._labels_only)

        self._apply_edge_background_enabled(bool(settings.get("edge_background", self._edge_background_enabled)))
        self._settings_panel.set_toggle_value("edge_background", self._edge_background_enabled)

        _eb = float(settings.get("edge_brightness", self._edge_brightness * 100) or self._edge_brightness * 100)
        self._edge_brightness = max(0.05, min(0.40, _eb / 100.0))
        self._settings_panel.set_slider_value("edge_brightness", _eb)

        self._apply_thermal_mode(str(settings.get("thermal_mode", self._thermal_mode) or self._thermal_mode))
        self._settings_panel.set_combo_value("thermal_mode", self._thermal_mode)
        self._settings_panel.set_toggle_value("recog_auto", bool(settings.get("recog_auto", self.session.recognition_auto)))

        for key in (
            "confidence", "iou", "image_size", "max_det",
            "stale_seconds", "stale_frames", "new_track_sec", "persistent_sec",
            "max_cards", "preview_quality", "fps", "recog_threshold", "recog_top_k",
        ):
            if key in settings:
                self._settings_panel.set_slider_value(key, settings[key])

        self._settings_panel.set_combo_value("detector_model", model_choice_name(self.session.config.model_path))

        _raw_peek = settings.get("hotkey_peek", "roi,sidebar")
        _peek_set = {s.strip() for s in str(_raw_peek).split(",") if s.strip()}
        if _peek_set:
            self._hotkey_peek_options = _peek_set
            self._settings_panel.set_hotkey_peek(_peek_set)

    def _on_settings_bbox_style(self, style: str) -> None:
        self._apply_bbox_style(style)
        self._persist_setting("bbox_style", style)

    def _on_settings_panel_background_style(self, style: str) -> None:
        self._apply_panel_background_style(style)
        self._persist_setting("panel_background_style", self._panel_background_style)

    def _on_settings_heat_tags(self, visible: bool) -> None:
        self._apply_heat_tags_visible(visible)
        self._persist_setting("heat_tags", bool(visible))

    def _on_settings_labels_only(self, enabled: bool) -> None:
        self._apply_labels_only(enabled)
        self._persist_setting("labels_only", bool(enabled))

    def _on_settings_edge_background(self, enabled: bool) -> None:
        self._apply_edge_background_enabled(enabled)
        self._persist_setting("edge_background", bool(enabled))

    def _on_settings_edge_brightness(self, value: float) -> None:
        self._edge_brightness = max(0.05, min(0.40, float(value)))
        self._persist_setting("edge_brightness", round(self._edge_brightness * 100))

    def _on_settings_thermal_mode(self, mode: str) -> None:
        self._apply_thermal_mode(mode)
        self._persist_setting("thermal_mode", mode)

    def _on_settings_button_caption_mode(self, mode: str) -> None:
        self._apply_button_caption_mode(mode)
        self._persist_setting("button_caption_mode", self._button_caption_mode)

    def _on_hotkey_peek_changed(self, options: object) -> None:
        self._hotkey_peek_options = set(options)  # type: ignore[arg-type]
        self._persist_setting("hotkey_peek", ",".join(sorted(self._hotkey_peek_options)))

    def _on_settings_color_scheme(self, scheme: str) -> None:
        resolved = configure_color_scheme(scheme)
        self._persist_setting("color_scheme", resolved)
        self.setStyleSheet(get_global_stylesheet())
        self._apply_wear_layout()
        TimelineCardWidget.refresh_theme()
        self._timeline_signature = ()
        self._render_timeline(self._last_timeline_entries)
        self._apply_hud_theme()
        self._settings_panel.refresh_theme_styles()
        self._loading_gate.refresh_theme()
        if hasattr(self, "_sidebar_panel"):
            self._sidebar_panel.refresh_theme()
        if hasattr(self, "_catalog_panel"):
            self._catalog_panel.refresh_theme()
        if hasattr(self, "_boot_selector"):
            self._boot_selector.refresh_theme()
        apply_text_palette(self)

    # -- EffectsWorker callback wrappers --------------------------------
    # These thin wrappers let us gate each effect with a simple bool
    # flag that the UI thread can flip without touching the worker.

    def _edge_bg_for_worker(
        self, frame: np.ndarray, overlays: list,
    ) -> np.ndarray:
        thermal = self._active_heat and "edge" in self._thermal_mode
        if self._edge_background_enabled or thermal:
            return apply_hairline_edge_background(
                frame, overlays, thermal=thermal, blend_alpha=self._edge_brightness,
            )
        return frame

    def _tick_edge_fade(self) -> None:
        dt = 0.016
        alpha = self._edge_fade_alpha
        target = self._edge_fade_target
        if target > alpha:
            alpha = min(target, alpha + dt * (1.0 / 0.25))   # 250ms fade in
        elif target < alpha:
            alpha = max(target, alpha - dt * (1.0 / 0.15))   # 150ms fade out
        self._edge_fade_alpha = alpha
        if abs(alpha - target) < 0.001:
            self._edge_fade_alpha = target
            self._edge_fade_timer.stop()

    def _on_sidebar_moved(self, x: int) -> None:
        self._sidebar_x = x

    def _position_sidebar(self) -> None:
        cw = self.centralWidget()
        if cw is None:
            return
        sw = self._sidebar_panel.width()
        ch = cw.height()
        if ch <= 0:
            return
        if getattr(self, "_wear_ui_enabled", False):
            bottom_visible = bool(hasattr(self, "_bottom_hud") and self._bottom_hud.isVisible())
            margin = 24
            sw = min(max(340, sw), max(340, cw.width() - margin * 2))
            # Anchor to right edge; respect user drag if they moved it.
            if self._sidebar_x < 0:
                x = max(margin, cw.width() - sw - margin)
            else:
                x = max(margin, min(cw.width() - sw - margin, self._sidebar_x))
            y = margin
            sh = max(100, ch - margin * 2 - (84 if bottom_visible else 24))
            self._sidebar_panel.setGeometry(x, y, sw, sh)
            self._sidebar_panel.raise_()
            return
        margin = int(ch * 0.10)
        y = margin
        sh = ch - margin * 2
        if self._sidebar_x < 0:
            x = cw.width() - sw
        else:
            x = max(0, min(cw.width() - sw, self._sidebar_x))
        self._sidebar_panel.setGeometry(x, y, sw, max(100, sh))
        self._sidebar_panel.raise_()

    def _on_sidebar_visibility(self, visible: bool) -> None:
        if hasattr(self, "_btn_data"):
            self._btn_data.setChecked(visible)
        self._persist_setting("sidebar_open", visible)
        # Reposition catalog so it doesn't overlap
        if hasattr(self, "_catalog_panel"):
            self._position_catalog()

    def _on_sidebar_width_changed(self, width: int) -> None:
        self._position_sidebar()
        self._persist_setting("sidebar_width", width)

    def _restore_sidebar(self) -> None:
        self._position_sidebar()
        self._sidebar_panel.restore()
        self._refresh_sidebar_views()

    def _toggle_sidebar(self) -> None:
        if self._sidebar_panel.isVisible():
            self._sidebar_panel.dismiss_animated()
            return
        self._restore_sidebar()

    def _refresh_sidebar_views(self) -> None:
        if not hasattr(self, "_sidebar_panel") or not self._sidebar_panel.isVisible():
            return
        current = self._tabs.currentIndex()
        if current == self._TAB_PREVIEWS:
            self._render_tiles(self._last_preview_tiles)
        elif current == self._TAB_EVENTS:
            self._render_events(self._last_event_items)

    def _refresh_catalog_views(self) -> None:
        if not hasattr(self, "_catalog_panel") or not self._catalog_panel.isVisible():
            return
        current = self._catalog_tabs.currentIndex()
        if current == self._CAT_TAB_SETTINGS:
            self._render_recovery_log()
        elif current == self._CAT_TAB_SUBROUTINE:
            try:
                self._subroutine_panel.reload()
            except Exception:
                pass

    # -- Catalog panel ---------------------------------------------------

    def _on_catalog_moved(self, x: int) -> None:
        self._catalog_x = x

    def _position_catalog(self) -> None:
        cw = self.centralWidget()
        if cw is None:
            return
        cw_w = cw.width()
        cat_w = self._catalog_panel.width()
        ch = cw.height()
        if ch <= 0:
            return
        if getattr(self, "_wear_ui_enabled", False):
            bottom_visible = bool(hasattr(self, "_bottom_hud") and self._bottom_hud.isVisible())
            margin = 24
            cat_w = min(max(320, cat_w), max(320, cw_w - margin * 2))
            # Anchor to left edge; respect user drag if they moved it.
            if self._catalog_x < 0:
                x = margin
            else:
                x = max(margin, min(cw_w - cat_w - margin, self._catalog_x))
            y = margin
            sh = max(100, ch - margin * 2 - (84 if bottom_visible else 24))
            self._catalog_panel.setGeometry(x, y, cat_w, sh)
            self._catalog_panel.raise_()
            if hasattr(self, "_sidebar_panel") and self._sidebar_panel.isVisible():
                self._position_sidebar()
            return
        margin = int(ch * 0.10)
        y = margin
        sh = ch - margin * 2
        if self._catalog_x < 0:
            # Default catalog anchor: left edge.
            x = 0
        else:
            x = max(0, min(cw_w - cat_w, self._catalog_x))
        self._catalog_panel.setGeometry(x, y, cat_w, max(100, sh))
        self._catalog_panel.raise_()
        # Nudge data panel so it doesn't sit under catalog
        if hasattr(self, "_sidebar_panel") and self._sidebar_panel.isVisible():
            self._position_sidebar()

    def _on_catalog_visibility(self, visible: bool) -> None:
        if hasattr(self, "_btn_catalog"):
            self._btn_catalog.setChecked(visible)
        self._persist_setting("catalog_open", visible)
        if visible:
            self._refresh_catalog_views()
        if hasattr(self, "_sidebar_panel"):
            self._position_sidebar()

    def _on_catalog_width_changed(self, width: int) -> None:
        self._position_catalog()
        self._persist_setting("catalog_width", width)

    def _restore_catalog(self) -> None:
        self._position_catalog()
        self._catalog_panel.restore()
        self._refresh_catalog_views()

    def _toggle_catalog(self) -> None:
        if self._catalog_panel.isVisible():
            self._catalog_panel.dismiss_animated()
            return
        self._restore_catalog()


    def _apply_roi_preset(self, scale: float) -> None:
        self._video.apply_roi_scale(scale)
        self._sync_roi_preset_buttons(scale)
        if self._video.roi_active:
            self._push_roi_if_active(self._video.roi_norm)

    def _sync_roi_preset_buttons(self, scale: float) -> None:
        for s, b in self._roi_preset_buttons:
            b.blockSignals(True)
            b.setChecked(abs(s - scale) < 0.02)
            b.blockSignals(False)
        self._video.update_roi_rail_preset(scale)

    def _reveal_focus_sidebar(self) -> None:
        self._tabs.setCurrentIndex(self._TAB_FOCUS)
        if self._sidebar_panel.isVisible():
            self._refresh_sidebar_views()
            return
        self._restore_sidebar()

    def _on_loading_gate_finished(self) -> None:
        self._boot_gate_done = True

    def _try_finish_boot_gate(self) -> None:
        if self._boot_gate_done:
            return
        lg = self._loading_gate
        if not lg.isVisible():
            return
        if lg.boot_ready():
            self._boot_gate_done = True
            lg.stop_fallback()
            lg.set_copy("Live", "Telemetry online. Scene controls and analysis are active.")
            lg.hide_with_success(2)

    def _toggle_flight_strip(self) -> None:
        return

    def _toggle_grid(self) -> None:
        self._video.show_grid = not self._video.show_grid
        self._btn_grid.setChecked(self._video.show_grid)
        if self._video.show_grid:
            self._btn_subgrid.show()
        else:
            self._btn_subgrid.hide()
        self._sync_subgrid_button()
        self._video_host.set_grid_overlay_visible(self._video.show_grid)
        self._video.update()

    def _sync_subgrid_button(self) -> None:
        mode = str(getattr(self._video, "subgrid_mode", "quads"))
        if mode == "halves":
            self._refresh_button_caption(self._btn_subgrid, "[Q] Halves")
        elif mode == "off":
            self._refresh_button_caption(self._btn_subgrid, "[Q] Off")
        else:
            self._refresh_button_caption(self._btn_subgrid, "[Q] Quads")

    def _toggle_subgrid(self) -> None:
        order = ("quads", "halves", "off")
        current = str(getattr(self._video, "subgrid_mode", "quads"))
        try:
            idx = order.index(current)
        except ValueError:
            idx = 0
        self._video.subgrid_mode = order[(idx + 1) % len(order)]
        self._sync_subgrid_button()
        self._video.update()

    def eventFilter(self, obj, event) -> bool:
        from PyQt6.QtCore import QEvent
        if isinstance(obj, QWidget) and obj is not self and not self.isAncestorOf(obj):
            return False
        if event.type() == QEvent.Type.MouseButtonPress:
            mods = event.modifiers()
            if (
                mods & Qt.KeyboardModifier.ControlModifier
                and event.button() == Qt.MouseButton.LeftButton
                and getattr(self, "_video_host", None) is not None
                and self._video_host._grid_visible
            ):
                global_pos = event.globalPosition().toPoint()
                host_pos = self._video_host.mapFromGlobal(global_pos)
                for cell_num, w in self._video_host._cell_widgets.items():
                    if w.geometry().contains(host_pos):
                        self._on_grid_cell_settings(cell_num)
                        return True
        return super().eventFilter(obj, event)

    def _on_grid_cell_add(self, cell_num: int) -> None:
        from .program_view_dialog import SourcePickerDialog
        dlg = SourcePickerDialog(cell_num, parent=self)
        dlg.source_confirmed.connect(self._apply_grid_cell_source)
        dlg.exec()

    def _on_grid_cell_settings(self, cell_num: int) -> None:
        from PyQt6.QtWidgets import QMenu, QSlider, QLabel, QWidgetAction, QWidget, QVBoxLayout
        btn = self._video_host.grid_overlay._buttons.get(cell_num)
        menu = QMenu(self)

        config = self._grid_cell_sources.get(cell_num, {})
        current_opacity = float(config.get("opacity", 1.0))

        slider_host = QWidget(menu)
        sl_layout = QVBoxLayout(slider_host)
        sl_layout.setContentsMargins(10, 4, 10, 4)
        sl_layout.setSpacing(2)
        label = QLabel(f"Opacity: {int(current_opacity * 100)}%")
        label.setStyleSheet("color: rgba(245,245,245,0.85); font-size: 10px;")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(10)
        slider.setMaximum(100)
        slider.setValue(int(current_opacity * 100))
        slider.setFixedWidth(160)
        slider.valueChanged.connect(
            lambda v, n=cell_num, lb=label: (
                lb.setText(f"Opacity: {v}%"),
                self._set_cell_opacity(n, v / 100.0),
            )
        )
        sl_layout.addWidget(label)
        sl_layout.addWidget(slider)
        wa = QWidgetAction(menu)
        wa.setDefaultWidget(slider_host)
        menu.addAction(wa)
        menu.addSeparator()

        change_act = menu.addAction("[change source]")
        remove_act = menu.addAction("[remove]")
        anchor = btn.mapToGlobal(btn.rect().bottomLeft()) if btn else self.cursor().pos()
        chosen = menu.exec(anchor)
        if chosen is change_act:
            self._on_grid_cell_add(cell_num)
        elif chosen is remove_act:
            self._grid_cell_sources.pop(cell_num, None)
            self._suite_manager.update_grid_cells(
                self._suite_manager.active_idx, self._grid_cell_sources
            )
            self._video_host.clear_cell_widget(cell_num)
            self._video_host.grid_overlay.set_cell_occupied(cell_num, False)

    def _set_cell_opacity(self, cell_num: int, opacity: float) -> None:
        opacity = max(0.1, min(1.0, opacity))
        if cell_num in self._grid_cell_sources:
            self._grid_cell_sources[cell_num]["opacity"] = opacity
            self._suite_manager.update_grid_cells(
                self._suite_manager.active_idx, self._grid_cell_sources
            )
        self._video_host.set_cell_opacity(cell_num, opacity)

    def _apply_grid_cell_source(self, cell_num: int, config: dict) -> None:
        self._grid_cell_sources[cell_num] = config
        # [SUITES] keep the active suite in sync with every cell assignment
        self._suite_manager.update_grid_cells(
            self._suite_manager.active_idx, self._grid_cell_sources
        )
        self._deploy_cell_widget(cell_num, config)

    def _deploy_cell_widget(self, cell_num: int, config: dict) -> None:
        """Create and place the appropriate widget for a grid cell source."""
        src_type = config.get("type", "")
        widget: Optional[QWidget] = None

        if src_type == "web":
            url = config.get("url", "").strip()
            if url:
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                try:
                    from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
                    from PyQt6.QtWebEngineWidgets import QWebEngineView

                    class _GridWebPage(QWebEnginePage):
                        def javaScriptConsoleMessage(
                            self, level, message, line_number, source_id
                        ) -> None:  # type: ignore[override]
                            text = str(message or "")
                            if "generate_204" in text and "preloaded" in text:
                                return
                            print(f"[PORTAL GRID JS] {source_id}:{line_number} {text}", flush=True)

                    view = QWebEngineView()
                    view.setPage(_GridWebPage(view))
                    s = view.settings()
                    s.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
                    s.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
                    s.setAttribute(QWebEngineSettings.WebAttribute.PlaybackRequiresUserGesture, False)
                    s.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)
                    view.load(QUrl(url))
                    widget = view
                except Exception as exc:
                    widget = self._make_cell_label(f"[web unavailable]\n{url}\n{exc}")

        elif src_type == "media":
            path = config.get("path", "").strip()
            if path:
                px = QPixmap(path)
                if not px.isNull():
                    lbl = _ScaledPixmapLabel(px)
                else:
                    lbl = self._make_cell_label(f"[media]\n{path}")
                widget = lbl

        elif src_type == "terminal":
            widget = self._make_cell_label(f"[terminal]\n{config.get('cwd', '') or '~'}")

        elif src_type == "widget":
            widget = self._make_cell_label(f"[widget]\n{config.get('widget_name', '')}")

        if widget is not None:
            self._video_host.set_cell_widget(cell_num, widget)
            opacity = float(config.get("opacity", 1.0))
            if opacity < 1.0:
                self._video_host.set_cell_opacity(cell_num, opacity)
            self._video_host.grid_overlay.set_cell_occupied(cell_num, True)
        else:
            self._video_host.clear_cell_widget(cell_num)
            self._video_host.grid_overlay.set_cell_occupied(cell_num, False)

    def _make_cell_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setWordWrap(True)
        lbl.setStyleSheet(
            "background: rgba(10,4,4,0.82);"
            "color: rgba(245,245,245,0.78);"
            "font-size: 11px;"
            "border: 1px solid rgba(245,245,245,0.12);"
        )
        return lbl

    # ------------------------------------------------------------------
    # Suite profile handlers
    # ------------------------------------------------------------------

    def _on_suite_selected(self, idx: int) -> None:
        """Save current grid state to the departing suite, then switch."""
        self._suite_manager.update_grid_cells(
            self._suite_manager.active_idx, self._grid_cell_sources
        )
        self._suite_manager.set_active(idx)
        active = self._suite_manager.active_suite
        self._grid_cell_sources = dict(active.grid_cells) if active else {}
        self._refresh_suite_btn()
        self._redeploy_all_cell_widgets()

    def _redeploy_all_cell_widgets(self) -> None:
        """Clear all cell widgets and re-render from current _grid_cell_sources."""
        for cell_num in list(VideoHost._CELL_POSITIONS):
            self._video_host.clear_cell_widget(cell_num)
            self._video_host.grid_overlay.set_cell_occupied(cell_num, False)
        for cell_num, config in self._grid_cell_sources.items():
            self._deploy_cell_widget(cell_num, config)

    def _on_suite_new(self) -> None:
        name, ok = QInputDialog.getText(self, "[new suite]", "Suite name:")
        if ok and name.strip():
            idx = self._suite_manager.add_suite(name.strip())
            self._on_suite_selected(idx)

    def _on_suite_rename(self, idx: int) -> None:
        suites = self._suite_manager.suites
        if not suites or idx >= len(suites):
            return
        current_name = suites[idx].name
        name, ok = QInputDialog.getText(
            self, "[rename suite]", "New name:", text=current_name
        )
        if ok and name.strip():
            self._suite_manager.rename_suite(idx, name.strip())
            self._refresh_suite_btn()

    def _on_suite_delete(self, idx: int) -> None:
        if not self._suite_manager.suites or idx >= len(self._suite_manager.suites):
            return
        self._suite_manager.delete_suite(idx)
        active = self._suite_manager.active_suite
        self._grid_cell_sources = dict(active.grid_cells) if active else {}
        self._refresh_suite_btn()

    def _refresh_suite_btn(self) -> None:
        active = self._suite_manager.active_suite
        self._refresh_button_caption(self._btn_suite, f"[S] {active.name}" if active else "[S] Suite")

    def _show_suite_menu(self) -> None:
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        for idx, suite in enumerate(self._suite_manager.suites):
            act = menu.addAction(suite.name)
            act.setCheckable(True)
            act.setChecked(idx == self._suite_manager.active_idx)
            act.triggered.connect(lambda _checked=False, i=idx: self._on_suite_selected(i))
        menu.addSeparator()
        rename_act = menu.addAction("[rename active]")
        rename_act.triggered.connect(lambda: self._on_suite_rename(self._suite_manager.active_idx))
        delete_act = menu.addAction("[delete active]")
        delete_act.triggered.connect(lambda: self._on_suite_delete(self._suite_manager.active_idx))
        menu.addSeparator()
        new_act = menu.addAction("[+ new suite]")
        new_act.triggered.connect(self._on_suite_new)
        menu.exec(self._btn_suite.mapToGlobal(self._btn_suite.rect().topLeft()))

    def _toggle_bottom_hud(self) -> None:
        if self._bottom_hud.isVisible():
            self._bottom_hud.hide()
            self._bottom_hud_restore.show()
            self._bottom_hud_restore.raise_()
            self._position_bottom_hud_restore()
        else:
            self._bottom_hud.show()
            self._bottom_hud_restore.hide()
        self._position_bottom_overlays()
        self._update_side_metric_visibility()
        self._position_side_metrics()

    def _on_settings_ai_provider(self, provider: str) -> None:
        if provider in ("auto", "ollama", "openai", "anthropic"):
            self._ai_provider = provider
            self._persist_setting("ai_provider", provider)

    def _on_settings_ai_model(self, provider: str, model: str) -> None:
        if provider in ("ollama", "openai", "anthropic") and model.strip():
            self._ai_models[provider] = model.strip()
            self._persist_setting(f"ai_model_{provider}", model.strip())

    def _send(self, payload: dict[str, Any]) -> None:
        self.session.handle_client_message(payload)

    def _reset_roi_ai_state(self) -> None:
        self._ai_open = False
        self._ai_busy = False
        self._ai_text = ""
        self._ai_err = ""
        self._ai_provider = "auto"
        self._ai_prompt_draft = ""

    def _tick_frame(self) -> None:
        return

    def _on_video_frame(self, payload: dict[str, Any]) -> None:
        pipeline = getattr(self.session, "pipeline", None)
        try:
            self._render_video_frame(payload)
        finally:
            # Release the backpressure slot AFTER we finish rendering so the
            # capture thread paces to the UI's real processing rate rather
            # than queueing ahead and creating visible skitter.
            if pipeline is not None:
                pipeline.notify_frame_consumed()

    def _render_video_frame(self, payload: dict[str, Any]) -> None:
        frame_ok = bool(payload.get("frame_ok", True))
        is_new = bool(payload.get("is_new", True))
        raw_frame = payload.get("frame")
        if raw_frame is None:
            return
        self._video_frame_ticks += 1
        if not self._boot_gate_done:
            self._loading_gate.set_progress("video", min(1.0, 0.28 + self._video_frame_ticks * 0.12))
            self._try_finish_boot_gate()
        if is_new:
            # All heavy visual effects run on the EffectsWorker thread.
            # Fade between raw and composed frame so edges appear gradually.
            composed = self._effects_worker.get_composed_frame()
            edges_wanted = self._edge_background_enabled or (
                self._active_heat and "edge" in self._thermal_mode
            )
            self._edge_fade_target = 1.0 if (edges_wanted and composed is not None) else 0.0
            if abs(self._edge_fade_alpha - self._edge_fade_target) > 0.001:
                if not self._edge_fade_timer.isActive():
                    self._edge_fade_timer.start()

            if composed is not None and self._edge_fade_alpha > 0.001:
                if self._edge_fade_alpha >= 0.999:
                    frame = composed
                else:
                    # Blend raw → composed to animate edge opacity
                    frame = cv2.addWeighted(
                        raw_frame, 1.0 - self._edge_fade_alpha,
                        composed, self._edge_fade_alpha,
                        0,
                    )
            else:
                frame = raw_frame

            overlays = list(payload.get("overlays") or [])
            self._video.set_frame(frame)
            self._video.set_overlays(overlays)
        self._video.set_highlight_track(payload.get("active_focus", self._last_hud.get("active_focus")))
        if frame_ok and self._swap_frame_pending and self._source_swap_pending:
            label = str(self._source_swap_pending.get("label", "source feed"))
            self._swap_frame_pending = False
            self._source_swap_pending = None
            self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 1.0, "live": 1.0})
            self._loading_gate.set_copy("Source Ready", f"{label} is active and rendering live frames.")
            self._loading_gate.hide_with_success(2, on_done=lambda: self._show_toast(f"Source ready: {label}", "info"))
        self._update_scrub_bar(force=is_new)

    def _on_manual_recognize(self) -> None:
        focus = self._last_hud
        track_id = focus.get("active_focus") or 0
        image_b64 = ""
        if self._roi_capture_payload:
            image_b64 = str(self._roi_capture_payload.get("image", ""))
        elif not self._focus_base_pixmap.isNull():
            from PyQt6.QtCore import QByteArray, QBuffer
            import base64 as _b64
            buf = QBuffer()
            buf.open(QBuffer.OpenModeFlag.WriteOnly)
            self._focus_base_pixmap.save(buf, "JPEG", 85)
            image_b64 = _b64.b64encode(buf.data().data()).decode("ascii")
        if not image_b64:
            self._show_toast("No image in focus to recognize", "warn")
            return
        self._send({
            "type": "recognize_entry",
            "entry_id": 0,
            "track_id": int(track_id) if track_id else 0,
            "image_b64": image_b64,
        })
        self._show_toast("Running recognition...", "info")

    def _cvops_http_json(self, method: str, path: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        url = f"{self._cvops_base_url}{path}"
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        with urllib.request.urlopen(req, timeout=1.8) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _cvops_http_multipart(
        self,
        path: str,
        *,
        fields: Optional[dict[str, str]] = None,
        files: Optional[dict[str, tuple[str, str, bytes]]] = None,
        timeout: float = 5.0,
    ) -> dict[str, Any]:
        url = f"{self._cvops_base_url}{path}"
        boundary = f"----insight-{uuid.uuid4().hex}"
        body = bytearray()
        for key, value in (fields or {}).items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8"))
            body.extend(str(value).encode("utf-8"))
            body.extend(b"\r\n")
        for key, (name, content_type, payload) in (files or {}).items():
            body.extend(f"--{boundary}\r\n".encode("utf-8"))
            body.extend(
                f'Content-Disposition: form-data; name="{key}"; filename="{name}"\r\n'.encode("utf-8")
            )
            body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
            body.extend(payload)
            body.extend(b"\r\n")
        body.extend(f"--{boundary}--\r\n".encode("utf-8"))
        req = urllib.request.Request(
            url,
            data=bytes(body),
            method="POST",
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}

    def _focus_image_b64(self) -> str:
        focus = self._focus_payload or {}
        image_b64 = str(focus.get("image", "") or "")
        if image_b64:
            return image_b64
        if self._focus_base_pixmap.isNull():
            return ""
        from PyQt6.QtCore import QBuffer
        import base64 as _b64

        buf = QBuffer()
        buf.open(QBuffer.OpenModeFlag.WriteOnly)
        self._focus_base_pixmap.save(buf, "JPEG", 85)
        return _b64.b64encode(buf.data().data()).decode("ascii")

    def _focus_image_bytes(self) -> bytes:
        import base64 as _b64

        image_b64 = self._focus_image_b64()
        if not image_b64:
            return b""
        return _b64.b64decode(image_b64)

    def _preview_tile_image_bytes(self, tile: dict[str, Any]) -> bytes:
        import base64 as _b64

        image_b64 = str(tile.get("source_image") or tile.get("image") or "").strip()
        if not image_b64:
            return b""
        return _b64.b64decode(image_b64)

    def _cvops_image_dataset_names(self) -> list[str]:
        payload = self._cvops_http_json("GET", "/database")
        names = [str(name).strip() for name in payload.get("datasets") or [] if str(name).strip()]
        categories = payload.get("categories") if isinstance(payload.get("categories"), dict) else {}
        if categories:
            names = [
                name for name in names
                if str(categories.get(name, "image") or "image").strip().lower() == "image"
            ]
        return names

    @staticmethod
    def _cvops_dataset_image_rel_path(item: object) -> str:
        if isinstance(item, dict):
            return str(
                item.get("relative_path")
                or item.get("rel_path")
                or item.get("path")
                or item.get("name")
                or item.get("file")
                or ""
            ).strip()
        return str(item or "").strip()

    @staticmethod
    def _cvops_dataset_image_target_folder(item: object) -> str:
        rel_path = MainWindow._cvops_dataset_image_rel_path(item)
        if not rel_path:
            return ""
        parts = Path(rel_path).parts
        split = ""
        if isinstance(item, dict):
            split = str(item.get("split") or "").strip()
        if "images" in parts:
            idx = parts.index("images")
            if split and split != "root" and len(parts) > idx + 3:
                return "/".join(parts[idx + 2 : -1])
            if not split and len(parts) > idx + 2:
                return "/".join(parts[idx + 1 : -1])
        parent = Path(rel_path).parent.as_posix()
        return "" if parent in {"", "."} else parent

    def _cvops_dataset_target_folders(self, dataset_slug: str, split: str = "") -> list[str]:
        slug = str(dataset_slug or "").strip()
        if not slug:
            return []
        quoted_slug = urllib.parse.quote(slug, safe="-_.~")
        payload = self._cvops_http_json("GET", f"/database/{quoted_slug}")
        raw_items = payload.get("images") if isinstance(payload, dict) else []
        raw_folders = payload.get("folders") if isinstance(payload, dict) else []
        split_l = str(split or "").strip().lower()
        folders: set[str] = set()
        if isinstance(raw_folders, list):
            for item in raw_folders:
                if isinstance(item, dict):
                    folder = str(item.get("path") or item.get("relative_path") or item.get("name") or "").strip()
                else:
                    folder = str(item or "").strip()
                folder = folder.strip("/")
                if folder:
                    folders.add(folder)
        if isinstance(raw_items, list):
            for item in raw_items:
                if split_l and isinstance(item, dict) and str(item.get("split") or "").strip().lower() != split_l:
                    continue
                folder = self._cvops_dataset_image_target_folder(item).strip("/")
                if folder:
                    folders.add(folder)
        return sorted(folders, key=lambda value: value.lower())

    def _cvops_dataset_preview_items(
        self,
        dataset_slug: str,
        *,
        split: str = "",
        target_folder: str = "",
        limit: int = 8,
    ) -> list[dict[str, Any]]:
        slug = str(dataset_slug or "").strip()
        if not slug:
            return []
        quoted_slug = urllib.parse.quote(slug, safe="-_.~")
        payload = self._cvops_http_json("GET", f"/database/{quoted_slug}")
        raw_items = payload.get("images") if isinstance(payload, dict) else []
        if not isinstance(raw_items, list):
            return []
        split_l = str(split or "").strip().lower()
        folder_l = str(target_folder or "").strip().strip("/").lower()
        filtered_items: list[object] = []
        for item in raw_items:
            if split_l and isinstance(item, dict) and str(item.get("split") or "").strip().lower() != split_l:
                continue
            item_folder = self._cvops_dataset_image_target_folder(item).strip("/").lower()
            if folder_l:
                if item_folder != folder_l and not item_folder.startswith(f"{folder_l}/"):
                    continue
            elif item_folder:
                continue
            filtered_items.append(item)
        items = list(reversed(filtered_items))[: max(1, int(limit))]
        previews: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                rel_path = self._cvops_dataset_image_rel_path(item)
                split = str(item.get("split") or item.get("folder") or "").strip()
                label = str(item.get("label") or item.get("classification_label") or "").strip()
            else:
                rel_path = self._cvops_dataset_image_rel_path(item)
                split = ""
                label = ""
            if not rel_path:
                continue
            pixmap = QPixmap()
            try:
                encoded_path = urllib.parse.quote(rel_path, safe="")
                thumb = self._cvops_http_json("GET", f"/database/{quoted_slug}/thumb/{encoded_path}")
                thumb_b64 = str(thumb.get("thumb_b64") or "")
                if thumb_b64:
                    pixmap = pixmap_from_b64_jpeg(thumb_b64)
            except Exception:
                pixmap = QPixmap()
            previews.append(
                {
                    "path": rel_path,
                    "name": Path(rel_path).name,
                    "split": split,
                    "label": label,
                    "pixmap": pixmap,
                }
            )
        return previews

    def _on_add_preview_tile_to_cvops(self, tile: dict[str, Any]) -> None:
        try:
            image_raw = self._preview_tile_image_bytes(tile)
        except Exception:
            image_raw = b""
        if not image_raw:
            self._show_toast("No card image available to add", "warn")
            return
        default_label = detection_label_text(tile.get("label", "")).strip().lower() or "object"
        try:
            dataset_names = self._cvops_image_dataset_names()
        except Exception as exc:
            self._show_toast(f"CV Ops database unavailable: {exc}", "warn")
            return
        if not dataset_names:
            dataset_names = [default_label]

        self._show_inline_cvops_add_flow(
            tile=tile,
            image_raw=image_raw,
            default_label=default_label,
            dataset_names=dataset_names,
        )

    def _show_inline_cvops_add_flow(
        self,
        *,
        tile: dict[str, Any],
        image_raw: bytes,
        default_label: str,
        dataset_names: list[str],
    ) -> None:
        if self._workflow_tab._send_state == "Loading":
            self._show_toast("Database send is already running", "warn")
            return
        self._workflow_tab.load(
            tile=tile,
            image_raw=image_raw,
            default_label=default_label,
            dataset_names=dataset_names,
            load_classes=self._focus_quick_load_classes,
            load_folders=self._cvops_dataset_target_folders,
            load_previews=self._cvops_dataset_preview_items,
        )
        self._catalog_tabs.setCurrentIndex(self._CAT_TAB_WORKFLOW)
        if not self._catalog_panel.isVisible():
            self._restore_catalog()

    def _close_inline_cvops_add_flow(self) -> None:
        if self._workflow_tab._send_state == "Loading":
            return
        self._workflow_tab.reset_to_idle()

    def _submit_inline_cvops_add_flow(self) -> None:
        panel = self._workflow_tab
        image_raw = panel._image_raw
        selection = panel.selection()
        dataset_slug = str(selection.get("dataset_slug") or "").strip()
        split_value = str(selection.get("split") or "train").strip() or "train"
        target_folder_value = str(selection.get("target_folder") or "").strip().strip("/")
        label_name = str(selection.get("label_name") or "").strip()
        if not dataset_slug:
            panel.set_send_state("failed", "select database")
            self._show_toast("Select a destination database", "warn")
            return
        if not label_name:
            panel.set_send_state("failed", "enter label")
            self._show_toast("Enter a label", "warn")
            return

        label_text = f"{label_name}\n"
        ts = int(time.time() * 1000.0)
        safe_label = re.sub(r"[^a-zA-Z0-9_-]+", "-", label_name).strip("-") or "object"
        image_name = f"preview-{safe_label}-{ts}.jpg"
        label_file = f"preview-{safe_label}-{ts}.txt"
        quoted_slug = urllib.parse.quote(dataset_slug, safe="-_.~")
        panel.set_controls_enabled(False)
        panel.set_send_state("loading", "uploading")
        bridge = _CvopsAddUploadBridge(self)
        self._cvops_add_upload_bridge = bridge

        def _run_upload() -> None:
            try:
                result = self._cvops_http_multipart(
                    f"/database/{quoted_slug}/add",
                    fields={
                        "split": split_value,
                        "target_folder": target_folder_value,
                        "storage_mode": "loose",
                        "label_name": label_name,
                        "create_empty_label": "0",
                    },
                    files={
                        "image": (image_name, "image/jpeg", image_raw),
                        "label": (label_file, "text/plain", label_text.encode("utf-8")),
                    },
                    timeout=6.0,
                )
                bridge.finished.emit(result, None)
            except urllib.error.HTTPError as exc:
                try:
                    detail = exc.read().decode("utf-8", errors="replace")
                except Exception:
                    detail = str(exc)
                bridge.finished.emit(None, f"Add rejected: {detail}")
            except Exception as exc:
                bridge.finished.emit(None, f"Add failed: {exc}")

        bridge.finished.connect(
            lambda result, error, db=dataset_slug, label=label_name: self._finish_inline_cvops_add_flow(
                result, error, db, label,
            )
        )
        threading.Thread(target=_run_upload, daemon=True).start()

    def _finish_inline_cvops_add_flow(
        self,
        result: object,
        error: object,
        dataset_slug: str,
        label_name: str,
    ) -> None:
        panel = self._workflow_tab
        if error:
            message = str(error)
            panel.set_controls_enabled(True)
            panel.set_send_state("failed", message[:120])
            self._show_toast(message, "warn")
            return
        result_payload = result if isinstance(result, dict) else {}
        saved_slug = str(result_payload.get("slug") or dataset_slug)
        source_pixmap = QPixmap()
        source_pixmap.loadFromData(panel._image_raw)
        panel.show_bridge(
            status="good",
            dataset_slug=saved_slug,
            target_folder=panel.target_folder(),
            label_name=label_name,
            detail="Saved",
            source_pixmap=source_pixmap if not source_pixmap.isNull() else None,
        )
        gallery_panel = getattr(self, "_gallery_panel", None)
        if gallery_panel is not None and hasattr(gallery_panel, "refresh"):
            try:
                gallery_panel.refresh()
            except Exception:
                pass
        self._show_toast(f"Added {label_name} to {saved_slug}", "info")

    def _load_focus_quick_scenarios(self) -> None:
        if self._focus_quick_loading_scenarios:
            return
        self._focus_quick_loading_scenarios = True
        try:
            payload = self._cvops_http_json("GET", "/scenarios")
            scenarios = list(payload.get("scenarios") or [])
        except Exception as exc:
            self._focus_quick_scenario_combo.clear()
            self._focus_quick_scenario_combo.addItem("CV Ops unavailable")
            self._focus_quick_scenario_combo.setEnabled(False)
            self._focus_quick_dataset_by_scenario = {}
            self._focus_quick_class_combo.clear()
            self._focus_quick_class_combo.addItem("0: default", 0)
            self._focus_quick_class_combo.setEnabled(False)
            self._show_toast(f"Quick Label scenario load failed: {exc}", "warn")
            return
        finally:
            self._focus_quick_loading_scenarios = False
        self._focus_quick_scenarios = scenarios
        self._focus_quick_dataset_by_scenario = {}
        self._focus_quick_scenario_combo.blockSignals(True)
        self._focus_quick_scenario_combo.clear()
        for entry in scenarios:
            name = str(entry.get("name", "") or "").strip()
            if not name:
                continue
            dataset_slug = str(entry.get("dataset", "") or "").strip()
            display_name = str(entry.get("display_name", "") or "").strip() or name
            label = f"{display_name} [{name}]"
            if dataset_slug:
                label += f" -> {dataset_slug}"
            self._focus_quick_dataset_by_scenario[name] = dataset_slug
            self._focus_quick_scenario_combo.addItem(label, name)
        self._focus_quick_scenario_combo.blockSignals(False)
        has_options = self._focus_quick_scenario_combo.count() > 0
        if not has_options:
            self._focus_quick_scenario_combo.addItem("No scenarios")
        self._focus_quick_scenario_combo.setEnabled(has_options)
        if not has_options:
            self._focus_quick_class_combo.clear()
            self._focus_quick_class_combo.addItem("0: default", 0)
            self._focus_quick_class_combo.setEnabled(False)
            return
        self._refresh_focus_quick_label_controls()

    def _on_focus_quick_scenario_changed(self, _index: int) -> None:
        self._refresh_focus_quick_label_controls()

    def _focus_quick_load_classes(self, dataset_slug: str) -> list[str]:
        slug = str(dataset_slug or "").strip()
        if not slug:
            return []
        cached = self._focus_quick_classes_by_dataset.get(slug)
        if cached is not None:
            return list(cached)
        quoted = urllib.parse.quote(slug, safe="-_.~")
        classes: list[str] = []
        try:
            payload = self._cvops_http_json("GET", f"/database/{quoted}/classes")
            raw = payload.get("classes") if isinstance(payload, dict) else []
            if isinstance(raw, list):
                classes = [str(x).strip() for x in raw if str(x).strip()]
        except Exception:
            classes = []
        self._focus_quick_classes_by_dataset[slug] = classes
        return list(classes)

    def _refresh_focus_quick_label_controls(self) -> None:
        scenario, dataset_slug = self._focus_quick_selected_target()
        _ = scenario
        classes = self._focus_quick_load_classes(dataset_slug) if dataset_slug else []
        self._focus_quick_class_combo.blockSignals(True)
        self._focus_quick_class_combo.clear()
        if classes:
            for idx, name in enumerate(classes):
                self._focus_quick_class_combo.addItem(f"{idx}: {name}", idx)
            preferred_idx = int(self._focus_quick_class_id_by_dataset.get(dataset_slug, 0))
            if preferred_idx < 0 or preferred_idx >= self._focus_quick_class_combo.count():
                preferred_idx = 0
            self._focus_quick_class_combo.setCurrentIndex(preferred_idx)
            self._focus_quick_class_combo.setEnabled(True)
        else:
            self._focus_quick_class_combo.addItem("0: default", 0)
            self._focus_quick_class_combo.setEnabled(bool(dataset_slug))
        self._focus_quick_class_combo.blockSignals(False)

    def _on_focus_quick_draw_toggled(self, enabled: bool) -> None:
        self._focus_overlay.set_quick_draw_mode(bool(enabled))

    def _on_focus_quick_box_changed(self, bbox: object) -> None:
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            return
        self._focus_quick_box_pct = (
            float(bbox[0]),
            float(bbox[1]),
            float(bbox[2]),
            float(bbox[3]),
        )

    def _on_focus_quick_clear_box(self) -> None:
        self._focus_quick_box_pct = None
        self._focus_overlay.set_quick_box(None)

    def _on_focus_quick_detect_face(self) -> None:
        raw = self._focus_image_bytes()
        if not raw:
            self._show_toast("No focus image available", "warn")
            return
        arr = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            self._show_toast("Focus image decode failed", "warn")
            return

        face_box: Optional[tuple[float, float, float, float]] = None
        try:
            from ..cvops.detection_backends import YuNetFaceDetectorBackend
            from ..engine.face_biometrics import FACE_DETECTOR_MODEL

            if FACE_DETECTOR_MODEL.exists():
                backend = YuNetFaceDetectorBackend(FACE_DETECTOR_MODEL, score_threshold=0.25)
                detections = backend.predict(frame)
                if detections:
                    best = max(
                        detections,
                        key=lambda det: (
                            float(det.get("conf", 0.0))
                            * max(1.0, float(det.get("x2", 0.0)) - float(det.get("x1", 0.0)))
                            * max(1.0, float(det.get("y2", 0.0)) - float(det.get("y1", 0.0)))
                        ),
                    )
                    face_box = (
                        float(best.get("x1", 0.0)),
                        float(best.get("y1", 0.0)),
                        float(best.get("x2", 0.0)),
                        float(best.get("y2", 0.0)),
                    )
        except Exception:
            face_box = None

        if face_box is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            detector = cv2.CascadeClassifier(cascade_path)
            if detector.empty():
                self._show_toast("Face detector unavailable", "warn")
                return
            faces = detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(24, 24),
            )
            if faces is not None and len(faces) > 0:
                x, y, w, h = max(faces, key=lambda f: int(f[2]) * int(f[3]))
                face_box = (float(x), float(y), float(x + w), float(y + h))

        if face_box is None:
            self._show_toast("No face detected. Draw a box manually.", "warn")
            return

        ih, iw = frame.shape[:2]
        x1_px, y1_px, x2_px, y2_px = face_box
        box_w = max(1.0, x2_px - x1_px)
        box_h = max(1.0, y2_px - y1_px)
        pad_x = box_w * 0.08
        pad_y = box_h * 0.10
        x1 = max(0.0, min(1.0, (x1_px - pad_x) / float(iw)))
        y1 = max(0.0, min(1.0, (y1_px - pad_y) / float(ih)))
        x2 = max(0.0, min(1.0, (x2_px + pad_x) / float(iw)))
        y2 = max(0.0, min(1.0, (y2_px + pad_y) / float(ih)))
        self._focus_quick_box_pct = (x1, y1, x2, y2)
        self._focus_overlay.set_quick_box(self._focus_quick_box_pct)
        self._focus_quick_draw_btn.setChecked(False)
        self._show_toast("Face box detected", "info")

    def _focus_quick_selected_target(self) -> tuple[str, str]:
        scenario = str(self._focus_quick_scenario_combo.currentData() or "").strip()
        dataset_slug = self._focus_quick_dataset_by_scenario.get(scenario, "")
        return scenario, str(dataset_slug or "").strip()

    def _focus_quick_yolo_label_text(self, *, class_id: int, mode: str) -> str:
        mode_key = str(mode or "box").strip().lower()
        if mode_key == "full":
            return f"{int(class_id)} 0.5 0.5 1.0 1.0\n"
        box = self._focus_quick_box_pct
        if box is None:
            return ""
        x1, y1, x2, y2 = box
        bw = max(0.0, min(1.0, x2 - x1))
        bh = max(0.0, min(1.0, y2 - y1))
        if bw <= 0.0 or bh <= 0.0:
            return ""
        xc = max(0.0, min(1.0, (x1 + x2) / 2.0))
        yc = max(0.0, min(1.0, (y1 + y2) / 2.0))
        return f"{int(class_id)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n"

    def _on_focus_quick_submit(self, *, auto_update: bool) -> None:
        scenario, dataset_slug = self._focus_quick_selected_target()
        if not scenario:
            self._show_toast("Select a scenario first", "warn")
            return
        if not dataset_slug:
            self._show_toast("Selected scenario has no dataset target", "warn")
            return
        class_id = int(self._focus_quick_class_combo.currentData() or 0)
        self._focus_quick_class_id_by_dataset[dataset_slug] = class_id
        label_mode = str(self._focus_quick_mode_combo.currentData() or "box")
        label_text = self._focus_quick_yolo_label_text(class_id=class_id, mode=label_mode)
        if not label_text:
            self._show_toast("Draw/detect a valid box, or switch Label Mode to Full Image", "warn")
            return
        image_raw = self._focus_image_bytes()
        if not image_raw:
            self._show_toast("No focus image available", "warn")
            return
        ts = int(time.time() * 1000.0)
        image_name = f"focus-label-{ts}.jpg"
        label_name = f"focus-label-{ts}.txt"
        split_value = str(self._focus_quick_split_combo.currentData() or "train")
        quoted_slug = urllib.parse.quote(dataset_slug, safe="-_.~")
        path = f"/database/{quoted_slug}/add"
        try:
            upload_result = self._cvops_http_multipart(
                path,
                fields={"split": split_value, "create_empty_label": "0"},
                files={
                    "image": (image_name, "image/jpeg", image_raw),
                    "label": (label_name, "text/plain", label_text.encode("utf-8")),
                },
                timeout=6.0,
            )
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            self._show_toast(f"Quick Add rejected: {detail}", "warn")
            return
        except Exception as exc:
            self._show_toast(f"Quick Add failed: {exc}", "warn")
            return

        if not auto_update:
            self._show_toast(
                f"Quick Add saved to {upload_result.get('slug', dataset_slug)}",
                "info",
            )
            return

        try:
            update_result = self._cvops_http_json("POST", f"/scenarios/{scenario}/update")
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            self._show_toast(
                f"Quick Add saved; update failed: {detail}",
                "warn",
            )
            return
        except Exception as exc:
            self._show_toast(
                f"Quick Add saved; update failed: {exc}",
                "warn",
            )
            return
        self._show_toast(
            f"Send Off complete: update queued {update_result.get('job_id', '?')}",
            "info",
        )

    def _collect_focus_image_for_cvops(self) -> tuple[str, str, dict[str, Any]]:
        focus = self._focus_payload or {}
        image_b64 = str(focus.get("image", "") or "")
        if not image_b64 and not self._focus_base_pixmap.isNull():
            from PyQt6.QtCore import QBuffer
            import base64 as _b64

            buf = QBuffer()
            buf.open(QBuffer.OpenModeFlag.WriteOnly)
            self._focus_base_pixmap.save(buf, "JPEG", 85)
            image_b64 = _b64.b64encode(buf.data().data()).decode("ascii")
        if not image_b64:
            return "", "", {}

        event_tag = str(focus.get("event_tag", "") or "")
        if event_tag.startswith("roi-"):
            source = "roi"
            meta = {
                "captured_at": focus.get("captured_at"),
                "track_id": int(focus.get("track_id", 0) or 0),
            }
            return image_b64, source, meta

        entry_id = int(focus.get("entry_id", 0) or 0)
        if entry_id > 0:
            source = "history"
            meta = {
                "entry_id": entry_id,
                "captured_at": focus.get("captured_at"),
                "track_id": int(focus.get("track_id", 0) or 0),
            }
            return image_b64, source, meta

        source = "focus"
        track_id = focus.get("track_id", self._last_hud.get("active_focus", 0))
        meta = {"track_id": int(track_id) if track_id else 0}
        return image_b64, source, meta

    def _on_submit_cvops_job(self) -> None:
        if not self._focus_body.isVisible():
            self._show_toast("No focus image available", "warn")
            return
        try:
            catalog = self._cvops_http_json("GET", "/scenarios")
        except Exception as exc:
            self._show_toast(f"CV Ops unavailable: {exc}", "warn")
            return

        scenarios = list(catalog.get("scenarios") or [])
        if not scenarios:
            err = str(catalog.get("error", "") or "")
            self._show_toast(err or "No CV scenarios available", "warn")
            return
        options = [f"{s.get('display_name') or s.get('name')} [{s.get('name')}]" for s in scenarios]
        selected, ok = QInputDialog.getItem(
            self,
            "Queue CV Ops Job",
            "Select scenario:",
            options,
            0,
            False,
        )
        if not ok or not selected:
            return
        picked_index = max(0, options.index(selected))
        scenario = str(scenarios[picked_index].get("name", "") or "").strip()
        if not scenario:
            self._show_toast("Invalid scenario selection", "warn")
            return

        image_b64, source, meta = self._collect_focus_image_for_cvops()
        if not image_b64:
            self._show_toast("No focus image available", "warn")
            return

        payload = {
            "scenario": scenario,
            "image_b64": image_b64,
            "source": source,
            **meta,
        }
        try:
            job = self._cvops_http_json("POST", "/jobs", payload=payload)
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")
            except Exception:
                detail = str(exc)
            self._show_toast(f"CV Ops rejected job: {detail}", "warn")
            return
        except Exception as exc:
            self._show_toast(f"CV Ops submit failed: {exc}", "warn")
            return
        self._show_toast(
            f"CV Ops queued: {job.get('job_id', '?')} ({job.get('job_type', 'job')})",
            "info",
        )

    def _on_video_track_clicked(self, track_id: int) -> None:
        if track_id <= 0:
            return
        self._history_focus_lock = False
        self._send({"type": "select_track", "track_id": int(track_id)})

    def _on_roi_burst_enroll(self) -> None:
        if not self._video.roi_active:
            self._show_toast("Enable ROI before burst enrollment", "warn")
            return
        identity = self._focus_burst_identity.text().strip() if hasattr(self, "_focus_burst_identity") else ""
        if not identity:
            self._show_toast("Enter an identity before ROI burst enrollment", "warn")
            if hasattr(self, "_focus_burst_identity"):
                self._focus_burst_identity.setFocus()
            return
        self._send(
            {
                "type": "enroll_roi_burst",
                "identity": identity,
                "group": "attendance",
                "samples": 10,
                "duration_sec": 3.5,
            }
        )
        self._show_toast(f"Capturing ROI burst for '{identity}'...", "info")

    def _on_roi_burst_enroll_result(self, payload: dict[str, Any]) -> None:
        identity = str(payload.get("identity", "") or "identity")
        added = int(payload.get("added", 0) or 0)
        captured = int(payload.get("captured", 0) or 0)
        self._gallery_panel.apply_burst_enroll_result(payload)
        self._catalog_tabs.setCurrentIndex(self._CAT_TAB_GALLERY)
        if self._catalog_panel.isVisible():
            self._refresh_catalog_views()
        else:
            self._restore_catalog()
        if added > 0:
            self._show_toast(f"Enrolled {added}/{captured} ROI burst samples for {identity}", "info")
            return
        errors = payload.get("errors") or []
        detail = str(errors[0]) if errors else "No usable face samples captured"
        self._show_toast(f"ROI burst enroll failed: {detail}", "warn")

    def _on_roi_capture_dblclick(self) -> None:
        self._reveal_focus_sidebar()
        self._send({"type": "capture_roi"})

    def _push_roi_if_active(self, norm: dict[str, Any]) -> None:
        if not self._video.roi_active:
            return
        if self._roi_sending:
            return
        self._roi_sending = True
        try:
            self._send(
                {
                    "type": "set_roi",
                    "shape": self._video.roi_shape,
                    "x1": norm["x1"],
                    "y1": norm["y1"],
                    "x2": norm["x2"],
                    "y2": norm["y2"],
                }
            )
        finally:
            QTimer.singleShot(80, lambda: setattr(self, "_roi_sending", False))

    def _toggle_roi(self) -> None:
        self._video.roi_active = self._btn_roi.isChecked()
        self._video.roi_shape = "circle"
        self._video.update()
        if self._video.roi_active:
            self._sync_roi_preset_buttons(self._video.roi_scale_preset)
            self._push_roi_if_active(self._video.roi_norm)
        else:
            self._send({"type": "clear_roi"})
        if self._preview_mode == "selective":
            self._tile_cache.clear()
            self._render_tiles(self._last_preview_tiles)

    def _toggle_active_mode(self) -> None:
        self._active_mode = self._btn_active.isChecked()
        self._video.active_mode = self._active_mode
        if self._active_mode and self._active_heat:
            self._active_heat = False
            self._btn_active_heat.setChecked(False)
            self._video.active_heat = False
        self._video.invalidate_overlay_cache()
        label = "ACTIVE" if self._active_mode else "PASSIVE"
        self._show_toast(f"Mode: {label}", "info")

    def _toggle_active_heat(self) -> None:
        self._active_heat = self._btn_active_heat.isChecked()
        self._video.active_heat = self._active_heat
        if self._active_heat and self._active_mode:
            self._active_mode = False
            self._btn_active.setChecked(False)
            self._video.active_mode = False
        self._video.invalidate_overlay_cache()
        label = "THERMAL" if self._active_heat else "PASSIVE"
        self._show_toast(f"Mode: {label}", "info")

    def _request_source_switch(self) -> None:
        self._send({"type": "switch_source"})

    def _toggle_video_pause(self) -> None:
        fs = self.session.frame_source
        if not fs.is_video:
            return
        if fs.video_paused or abs(fs.video_speed) < 0.01:
            if abs(fs.video_speed) < 0.01:
                fs.set_video_speed(1.0)
            fs.video_paused = False
        else:
            fs.video_paused = True
        self._sync_video_speed_controls()
        self._update_scrub_bar(force=True)

    def _toggle_scrub_speed(self, on: bool) -> None:
        self._scrub_speed_slider.setVisible(on)

    def _on_scrub_speed_changed(self, value: int) -> None:
        fs = self.session.frame_source
        speed = float(value) / 10.0
        self._scrub_speed_value.setText(f"{speed:.1f}x")
        if not fs.is_video:
            return
        fs.set_video_speed(speed)
        if abs(speed) > 0.01 and fs.video_paused:
            fs.video_paused = False
        self._sync_video_speed_controls()
        self._update_scrub_bar(force=True)

    def _sync_video_speed_controls(self) -> None:
        fs = self.session.frame_source
        speed = fs.video_speed if fs.is_video else 1.0
        speed_value = 0.0 if abs(speed) < 0.01 else speed
        self._scrub_speed_value.setText(f"{speed_value:.1f}x")
        slider_target = int(round(speed_value * 10.0))
        slider_target = max(-50, min(50, slider_target))
        if self._scrub_speed_slider.value() != slider_target:
            self._scrub_speed_slider.blockSignals(True)
            self._scrub_speed_slider.setValue(slider_target)
            self._scrub_speed_slider.blockSignals(False)
        stopped = fs.video_paused or abs(speed_value) < 0.01
        self._refresh_button_caption(self._scrub_play_btn, "[>] Play" if stopped else "[||] Pause")

    def _scrub_pressed(self) -> None:
        self._scrub_dragging = True

    def _scrub_released(self) -> None:
        self._scrub_dragging = False
        self._scrub_seek(self._scrub_slider.value())

    def _scrub_moved(self, value: int) -> None:
        if self._scrub_dragging:
            self._scrub_seek(value)

    def _scrub_seek(self, slider_value: int) -> None:
        fs = self.session.frame_source
        if not fs.is_video:
            return
        fraction = slider_value / 1000.0
        fs.seek_fraction(fraction)
        self._update_scrub_bar(force=True)

    def _update_scrub_bar(self, force: bool = False) -> None:
        fs = self.session.frame_source
        is_vid = fs.is_video
        if is_vid and not self._scrub_bar.isVisible():
            self._scrub_bar.show()
            fs.video_paused = False
            fs.set_video_speed(1.0)
            self._scrub_speed_btn.setChecked(False)
            self._scrub_speed_slider.hide()
            force = True
            self._position_bottom_overlays()
        elif not is_vid and self._scrub_bar.isVisible():
            self._scrub_bar.hide()
            self._position_bottom_overlays()
        if not is_vid:
            return
        now = time.monotonic()
        if not force and (now - self._last_scrub_poll_ts) < self._scrub_poll_interval_sec:
            return
        self._last_scrub_poll_ts = now
        self._sync_video_speed_controls()
        if not self._scrub_dragging:
            total = fs.video_frame_count
            pos = fs.video_position_frame
            if total > 0:
                self._scrub_slider.setValue(int(pos * 1000 / total))
            else:
                self._scrub_slider.setValue(0)
        pos_sec = fs.video_position_sec
        dur_sec = fs.video_duration_sec
        self._scrub_time_left.setText(self._fmt_time(pos_sec))
        self._scrub_time_right.setText(self._fmt_time(dur_sec))

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        s = max(0, int(seconds))
        m, sec = divmod(s, 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"{h}:{m:02d}:{sec:02d}"
        return f"{m}:{sec:02d}"

    def _toggle_timeline(self) -> None:
        self._btn_timeline.setChecked(False)
        self._timeline.hide()
        self._position_bottom_overlays()

    def _fit_nreal(self) -> None:
        """Snap window to 1920x1080 for nreal Air glass display."""
        self.resize(1920, 1080)
        self._sidebar_panel.setFixedWidth(260)
        self._sidebar_panel._saved_width = 260
        self._show_toast("Fitted to 1920x1080 nreal Air", "info")

    def _close_timeline(self) -> None:
        self._btn_timeline.setChecked(False)
        self._timeline.hide()
        self._position_bottom_overlays()

    def _cancel_source_swap_ui(self) -> None:
        if not self._source_swap_pending:
            return
        self._send({"type": "cancel_source_switch"})
        self._source_swap_pending = None
        self._swap_frame_pending = False
        self._loading_gate.set_swap_actions_visible(False)
        self._loading_gate.hide_with_success(0)
        self._show_toast("Source swap cancelled", "info")

    def _confirm_source_swap_ui(self) -> None:
        if not self._source_swap_pending:
            return
        tgt = self._source_swap_pending.get("target")
        label = str(self._source_swap_pending.get("label", "source feed"))
        self._send({"type": "confirm_source_switch", "source": tgt})
        self._loading_gate.set_swap_actions_visible(False)
        self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 0.98, "live": 0.82})
        self._loading_gate.set_copy(
            "Swapping",
            f"Switching to {label}. Keeping the current view visible until the new feed lands.",
        )

    def _bump_focus_zoom(self, delta: float) -> None:
        self._focus_zoom = max(1.0, min(3.5, self._focus_zoom + delta))
        self._focus_zoom_lbl.setText(f"{round(self._focus_zoom * 100)}%")
        self._focus_composite.set_zoom(self._focus_zoom)

    def _refresh_roi_capture(self) -> None:
        if self._roi_capture_payload:
            self._send({"type": "capture_roi"})

    def _clear_focus_ui(self) -> None:
        self._history_focus_lock = False
        self._send({"type": "clear_focus"})
        self._tabs.setCurrentIndex(0)
        self._apply_focus_panel(None)

    def _on_hud_payload(self, payload: dict[str, Any]) -> None:
        t = payload.get("type")
        if t == "operating_mode":
            self._operating_mode = str(payload.get("mode", "boot"))
            self._render_flight_metrics()
            if self._catalog_panel.isVisible() and self._catalog_tabs.currentIndex() == self._CAT_TAB_SETTINGS:
                self._render_recovery_log()
            return
        if t == "capability_report":
            self._capability_report = dict(payload.get("capabilities") or {})
            if self._catalog_panel.isVisible() and self._catalog_tabs.currentIndex() == self._CAT_TAB_SETTINGS:
                self._render_recovery_log()
            return
        if t == "system_health":
            self._system_health = list(payload.get("health") or [])
            if self._catalog_panel.isVisible() and self._catalog_tabs.currentIndex() == self._CAT_TAB_SETTINGS:
                self._render_recovery_log()
            return
        if t == "recovery_event":
            self._recovery_events.append(dict(payload))
            self._recovery_events = self._recovery_events[-40:]
            if self._catalog_panel.isVisible() and self._catalog_tabs.currentIndex() == self._CAT_TAB_SETTINGS:
                self._render_recovery_log()
            return
        if t == "hud_state":
            self._last_hud = payload.get("state") or {}
            model_name = str(self._last_hud.get("model", "") or "").strip()
            if model_name:
                self._settings_panel.set_combo_value("detector_model", model_name)
            self._render_flight_metrics()
            if not self._boot_gate_done:
                self._loading_gate.set_progress("data", 1.0)
                self._loading_gate.set_progress("live", 1.0)
                self._loading_gate.set_copy("Syncing", "Control channel and scene state online.")
                self._try_finish_boot_gate()
            return
        if t == "preview_tiles":
            self._last_preview_tiles = list(payload.get("tiles") or [])
            if self._sidebar_panel.isVisible() and self._tabs.currentIndex() == self._TAB_PREVIEWS:
                self._render_tiles(self._last_preview_tiles)
            return
        if t == "event_feed":
            self._last_event_items = list(payload.get("items") or [])
            if self._sidebar_panel.isVisible() and self._tabs.currentIndex() == self._TAB_EVENTS:
                self._render_events(self._last_event_items)
            return
        if t == "focus_state":
            if not self._history_focus_lock:
                self._apply_focus_panel(payload.get("focus"))
            return
        if t == "roi_capture":
            self._history_focus_lock = True
            self._apply_focus_panel(payload.get("focus"), switch_tab=True)
            return
        if t == "detection_history":
            self._last_timeline_entries = list(payload.get("entries") or [])
            if self._sidebar_panel.isVisible() and self._tabs.currentIndex() == self._TAB_PREVIEWS:
                self._render_tiles(self._last_preview_tiles)
            return
        if t == "status":
            self._show_toast(str(payload.get("message", "")), str(payload.get("level", "info")))
            return
        if t == "video_frame":
            self._on_video_frame(payload)
            return
        if t == "source_switch":
            self._handle_source_switch(payload)
            return
        if t == "roi_ai_status":
            self._ai_busy = payload.get("stage") == "started"
            if self._ai_busy:
                self._ai_err = ""
                self._ai_text = ""
                self._ai_provider = str(payload.get("provider", self._ai_provider))
            if self._roi_capture_payload:
                self._rebuild_focus_scan_ui()
            return
        if t == "roi_ai_result":
            cap = self._roi_capture_payload or {}
            if payload.get("captured_at") is not None and payload.get("captured_at") != cap.get("captured_at"):
                return
            self._ai_busy = False
            self._ai_err = str(payload.get("error", "") or "")
            self._ai_text = str(payload.get("text", "") or "")
            self._ai_provider = str(payload.get("provider", self._ai_provider))
            self._rebuild_focus_scan_ui()
            return
        if t == "recognition_result":
            self._on_recognition_result(payload)
            return
        if t == "roi_burst_enroll_result":
            self._on_roi_burst_enroll_result(payload)
            return
        if t == "attendance_event":
            self._on_attendance_event(payload)
            return
        if t == "gallery_state":
            self._gallery_panel.apply_gallery_state(payload)
            return
        if t == "gallery_ingest_progress":
            self._gallery_panel.apply_ingest_progress(payload)
            return
        if t == "gallery_ingest_result":
            self._gallery_panel.apply_ingest_result(payload)
            return
        if t == "similarity_search_result":
            self._gallery_panel.apply_similarity_search_result(payload)
            return

    def _handle_source_switch(self, payload: dict[str, Any]) -> None:
        stage = payload.get("stage")
        label = str(payload.get("label") or "source feed")
        if self._boot_gate_done:
            if stage == "starting":
                self._source_swap_pending = {"target": payload.get("target"), "label": label}
                self._swap_frame_pending = False
                self._loading_gate.set_swap_actions_visible(False)
                self._loading_gate.hide_with_success(0)
                self._show_toast(f"Preparing {label} in background", "info")
            elif stage == "prepared":
                self._source_swap_pending = {
                    "target": payload.get("target"),
                    "label": label,
                }
                self._swap_frame_pending = False
                self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 0.9, "live": 0.64})
                self._loading_gate.set_copy(
                    "Ready To Swap",
                    f"{label} is warmed and ready. Confirm when you want to leave the current view.",
                )
                self._loading_gate.set_swap_actions_visible(True)
                self._loading_gate.show()
                self._loading_gate.raise_()
            elif stage == "committing":
                self._source_swap_pending = {"target": payload.get("target"), "label": label}
                self._swap_frame_pending = False
                self._loading_gate.set_swap_actions_visible(False)
                self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 0.98, "live": 0.82})
                self._loading_gate.set_copy(
                    "Swapping",
                    f"Switching to {label}. Keeping the current view visible until the new feed lands.",
                )
                self._loading_gate.show()
                self._loading_gate.raise_()
            elif stage == "ready":
                self._source_swap_pending = None
                self._swap_frame_pending = False
                self._loading_gate.set_swap_actions_visible(False)
                self._loading_gate.hide_with_success(0)
                self._show_toast(f"Source ready: {label}", "info")
            elif stage == "cancelled":
                self._source_swap_pending = None
                self._swap_frame_pending = False
                self._loading_gate.set_swap_actions_visible(False)
                self._loading_gate.hide_with_success(0)
                self._show_toast("Source swap cancelled", "info")
            elif stage == "failed":
                self._source_swap_pending = None
                self._swap_frame_pending = False
                self._loading_gate.set_swap_actions_visible(False)
                self._loading_gate.hide_with_success(0)
                self._show_toast(str(payload.get("message") or "The source could not be prepared."), "warn")
            return
        if stage == "starting":
            self._source_swap_pending = {"target": payload.get("target"), "label": label}
            self._swap_frame_pending = False
            self._loading_gate.stop_fallback()
            self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 0.24, "live": 0.08})
            self._loading_gate.set_copy(
                "Switching Source",
                f"Preparing {label}. Holding the current view until the new feed is ready.",
            )
            self._loading_gate.set_swap_actions_visible(False)
            self._loading_gate.show()
            self._loading_gate.raise_()
        elif stage == "prepared":
            self._source_swap_pending = {
                "target": payload.get("target"),
                "label": label,
            }
            self._swap_frame_pending = False
            self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 0.9, "live": 0.64})
            self._loading_gate.set_copy(
                "Ready To Swap",
                f"{label} is warmed and ready. Confirm when you want to leave the current view.",
            )
            self._loading_gate.set_swap_actions_visible(True)
            self._loading_gate.show()
            self._loading_gate.raise_()
        elif stage == "committing":
            if self._source_swap_pending is None:
                self._source_swap_pending = {"target": payload.get("target"), "label": label}
            self._swap_frame_pending = False
            self._loading_gate.set_swap_actions_visible(False)
            self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 0.98, "live": 0.82})
            self._loading_gate.set_copy(
                "Swapping",
                f"Moving to {label}. Waiting for the new live view to render.",
            )
            self._loading_gate.show()
            self._loading_gate.raise_()
        elif stage == "ready":
            if self._source_swap_pending is None:
                self._source_swap_pending = {"target": payload.get("target"), "label": label}
            self._swap_frame_pending = True
            self._loading_gate.set_swap_actions_visible(False)
            self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 1.0, "live": 0.92})
            self._loading_gate.set_copy("Source Ready", f"{label} is active and rendering live frames.")
        elif stage == "cancelled":
            self._source_swap_pending = None
            self._swap_frame_pending = False
            self._loading_gate.set_swap_actions_visible(False)
            self._loading_gate.hide_with_success(0)
        elif stage == "failed":
            self._source_swap_pending = None
            self._swap_frame_pending = False
            self._loading_gate.set_swap_actions_visible(False)
            self._loading_gate.set_progress_exact({"signal": 1.0, "data": 1.0, "video": 0.12, "live": 0.12})
            self._loading_gate.set_copy("Swap Failed", str(payload.get("message") or "The source could not be prepared."))
            self._loading_gate.show()
            self._loading_gate.raise_()
            self._loading_gate.hide_with_failure(3)

    def _on_recognition_result(self, payload: dict[str, Any]) -> None:
        identity = str(payload.get("identity", "unknown"))
        track_id = int(payload.get("track_id", 0))
        threshold_met = bool(payload.get("threshold_met", False))
        source = str(payload.get("source", "auto"))
        if source != "manual":
            return
        if not threshold_met or identity == "unknown":
            self._show_toast("No confident attendance match", "warn")
            self._apply_recognition_to_focus(payload)
            return
        confidence = format_confidence_percent(payload.get("confidence", 0))
        self._show_toast(f"[ID] {identity}  {confidence}  (T{track_id})", "info")
        if source == "manual":
            self._apply_recognition_to_focus(payload)

    def _on_attendance_event(self, payload: dict[str, Any]) -> None:
        event = str(payload.get("event", "") or "")
        identity = str(payload.get("identity", "unknown") or "unknown")
        if event == "check_in":
            self._show_toast(f"Check-in: {identity}", "info")
            return
        if event == "check_out":
            self._show_toast(f"Check-out: {identity}", "info")

    def _apply_recognition_to_focus(self, payload: dict[str, Any]) -> None:
        if not self._focus_body.isVisible():
            return
        self._rebuild_recognition_ui(payload)

    def _rebuild_recognition_ui(self, payload: dict[str, Any]) -> None:
        identity = str(payload.get("identity", "unknown"))
        confidence = format_confidence_percent(payload.get("confidence", 0))
        top_matches = payload.get("top_matches", [])
        threshold_met = bool(payload.get("threshold_met", False))

        if hasattr(self, "_recog_result_widget") and self._recog_result_widget is not None:
            self._recog_result_widget.deleteLater()
            self._recog_result_widget = None

        result_w = QWidget()
        result_w.setStyleSheet(
            f"QWidget {{ background: {theme_rgba('accent_dark', 0.30)}; border-top: 1px solid {theme_rgba('accent_dark', 0.35)}; }}"
        )
        rl = QVBoxLayout(result_w)
        rl.setContentsMargins(8, 8, 8, 8)
        rl.setSpacing(6)

        head_row = QHBoxLayout()
        head_lbl = QLabel("[ID] Recognition Result")
        head_lbl.setStyleSheet("font-size: 9px; color: rgba(20,8,8,0.60); letter-spacing: 1px;")
        head_row.addWidget(head_lbl)
        head_row.addStretch(1)
        rl.addLayout(head_row)

        if threshold_met and identity != "unknown":
            name_lbl = QLabel(f"<b style='color:#140808; font-size:13px'>{identity}</b>")
            name_lbl.setTextFormat(Qt.TextFormat.RichText)
            conf_lbl = QLabel(f"{confidence} confidence")
            conf_lbl.setStyleSheet("font-size: 10px; color: #140808;")
            rl.addWidget(name_lbl)
            rl.addWidget(conf_lbl)
        else:
            unk = QLabel("Unknown - no gallery match above threshold")
            unk.setStyleSheet(f"font-size: 10px; color: {theme_rgba('accent_dark', 0.86)}; font-style: italic;")
            rl.addWidget(unk)

        if top_matches:
            matches_lbl = QLabel("Top matches:")
            matches_lbl.setStyleSheet("font-size: 9px; color: rgba(20,8,8,0.52); margin-top: 4px;")
            rl.addWidget(matches_lbl)
            for m in top_matches[:3]:
                sim = int(round(float(m.get("similarity", 0)) * 100))
                m_name = str(m.get("identity", ""))
                m_row = QHBoxLayout()
                m_lbl = QLabel(m_name)
                m_lbl.setStyleSheet("font-size: 10px; color: #140808;")
                m_bar = QProgressBar()
                m_bar.setRange(0, 100)
                m_bar.setValue(sim)
                m_bar.setTextVisible(False)
                m_bar.setFixedHeight(6)
                m_bar.setStyleSheet(get_hud_meter_css(2))
                m_pct = QLabel(f"{sim}%")
                m_pct.setStyleSheet(f"font-size: 9px; color: {text_css(0.72)}; min-width: 28px;")
                m_row.addWidget(m_lbl, stretch=1)
                m_row.addWidget(m_bar, stretch=2)
                m_row.addWidget(m_pct)
                m_rw = QWidget()
                m_rw.setLayout(m_row)
                rl.addWidget(m_rw)

        self._recog_result_widget = result_w
        self._focus_scan_layout.addWidget(result_w)

    def _show_toast(self, message: str, level: str) -> None:
        if not message or not hasattr(self, "_toast"):
            return
        border = {
            "error": theme_rgba("accent_dark", 0.45),
            "warn": theme_rgba("accent_dark", 0.32),
        }.get(level, theme_rgba("accent_dark", 0.20))
        self._toast.setText(message)
        if current_color_scheme() == "beacon":
            paper = theme_hex("paper")
            accent = theme_hex("accent_dark")
            if level == "error":
                bg = theme_rgba("panel", 0.92)
                fg = text_hex()
                self._toast.setStyleSheet(
                    f"QLabel {{ background: {bg}; border: 1px solid {border}; "
                    f"padding: 4px 12px; color: {fg}; font-size: 10px; border-radius: 0px; }}"
                )
            elif level == "warn":
                bg = theme_rgba("panel", 0.88)
                fg = text_hex()
                self._toast.setStyleSheet(
                    f"QLabel {{ background: {bg}; border: 1px solid {border}; "
                    f"padding: 4px 12px; color: {fg}; font-size: 10px; border-radius: 0px; }}"
                )
            else:
                self._toast.setStyleSheet(
                    f"QLabel {{ background: {accent}; border: 1px solid {accent}; "
                    f"padding: 4px 12px; color: {paper}; font-size: 10px; font-weight: 700; border-radius: 0px; }}"
                )
        else:
            self._toast.setStyleSheet(
                f"QLabel {{ background: {theme_rgba('panel', 0.76)}; border: 1px solid {border}; "
                f"padding: 4px 12px; color: {text_hex()}; font-size: 10px; }}"
            )
        self._toast.show()
        self._toast.raise_()
        self._position_toast()
        self._toast_timer.start(2200)

    @staticmethod
    def _apply_panel_glow(widget: QWidget, blur: int, alpha: int) -> None:
        glow = QGraphicsDropShadowEffect(widget)
        glow.setBlurRadius(blur)
        glow.setOffset(0, 5)
        glow.setColor(QColor(220, 10, 10, alpha))
        widget.setGraphicsEffect(glow)

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            child_layout = item.layout()
            if widget is not None:
                widget.deleteLater()
            elif child_layout is not None:
                MainWindow._clear_layout(child_layout)

    def _build_focus_stat_chip(self, label: str, value: str) -> QWidget:
        chip = QWidget()
        layout = QVBoxLayout(chip)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        key = QLabel(label.upper())
        key.setStyleSheet("font-size: 9px; color: rgba(20,8,8,0.52); letter-spacing: 1.2px;")
        val = QLabel(value)
        val.setStyleSheet("font-size: 11px; color: #140808;")
        layout.addWidget(key)
        layout.addWidget(val)
        return chip

    def _update_side_metric_visibility(self) -> None:
        # Flight metrics moved to Catalog -> Controls -> Live Session (no corner overlays on video).
        if hasattr(self, "_side_metrics_left"):
            self._side_metrics_left.hide()
        if hasattr(self, "_side_metrics_right"):
            self._side_metrics_right.hide()

    def _render_flight_metrics(self) -> None:
        s = self._last_hud
        entries = [
            ("MODE", self._operating_mode.upper()),
            ("VIEW", s.get("mode", "PASSIVE")),
            ("DETECT", str(s.get("detection_mode", "boxes")).upper()),
            ("SOURCE", s.get("source", "--")),
            ("MODEL", s.get("model", "--")),
            ("TRACKS", str(s.get("track_count", 0))),
            ("LAT", f"{s.get('latency_ms', 0)} ms"),
            ("FOCUS", "none" if s.get("active_focus") is None else f"#{s['active_focus']}"),
        ]
        if hasattr(self, "_settings_panel"):
            self._settings_panel.set_live_session_metrics(entries)

    def _render_recovery_log(self) -> None:
        caps = ", ".join(
            f"{key}={'on' if value else 'off'}"
            for key, value in sorted(self._capability_report.items())
        ) or "no capability report yet"
        degraded = [item for item in self._system_health if str(item.get("state")) not in {"healthy", "disabled"}]
        summary = f"Operating mode: {self._operating_mode}. Capabilities: {caps}."
        if degraded:
            summary += f" Active faults: {len(degraded)}."
        else:
            summary += " All supervised subsystems nominal."
        lines: list[str] = []
        for item in self._system_health:
            detail = str(item.get("last_error") or "ok")
            lines.append(f"[{str(item.get('state', 'starting')).upper()}] {item.get('name', 'subsystem')}: {detail}")
        if self._recovery_events:
            lines.append("")
            lines.append("Recovery events:")
            for event in self._recovery_events[-12:]:
                action = str(event.get("action", "event"))
                detail = str(event.get("detail", ""))
                ts = event.get("ts", "")
                lines.append(f"{ts}  {action}  {detail}")
        self._settings_panel.apply_supervisor_state(summary, lines)

    def _ensure_preview_section_header(self, attr_name: str, text: str) -> QLabel:
        label = getattr(self, attr_name)
        if label is None:
            label = QLabel(text)
            label.setStyleSheet(
                f"padding: 10px 6px 2px 6px; color: {text_css(0.62)}; "
                "font-size: 9px; font-weight: 700; letter-spacing: 1.6px;"
            )
            setattr(self, attr_name, label)
            self._previews_layout.addWidget(label)
        label.setText(text)
        return label

    def _tile_matches_preview_filter(self, tile: dict[str, Any]) -> bool:
        return matches_detection_view(
            self._preview_filter_text,
            self._quick_filters,
            tile.get("label", ""),
            tile.get("event_tag", ""),
            tile.get("recognized_identity", ""),
            heatmap_category(str(tile.get("label", "") or "")),
        )

    def _tile_intersects_active_roi(self, tile: dict[str, Any]) -> bool:
        if not self._video.roi_active:
            return True
        bbox = tile.get("bbox_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            return True
        try:
            bx1, by1, bx2, by2 = [float(v) for v in bbox[:4]]
            cx = (bx1 + bx2) / 2.0
            cy = (by1 + by2) / 2.0
            roi = self._video.roi_norm
            rx1 = float(roi.get("x1", 0.0))
            ry1 = float(roi.get("y1", 0.0))
            rx2 = float(roi.get("x2", 1.0))
            ry2 = float(roi.get("y2", 1.0))
        except Exception:
            return True
        if self._video.roi_shape == "circle":
            rcx = (rx1 + rx2) / 2.0
            rcy = (ry1 + ry2) / 2.0
            radius = min(abs(rx2 - rx1), abs(ry2 - ry1)) / 2.0
            return ((cx - rcx) ** 2 + (cy - rcy) ** 2) <= radius ** 2
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def _filtered_scan_results(self, scan: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
        if not scan:
            return []
        if not self._preview_filter_text:
            if not self._quick_filters:
                return list(scan)
        if not self._preview_filter_text and not self._quick_filters:
            return list(scan)
        return [
            item for item in scan
            if matches_detection_view(
                self._preview_filter_text,
                self._quick_filters,
                item.get("label", ""),
                heatmap_category(str(item.get("label", "") or "")),
            )
        ]

    def _apply_detection_filters(self) -> None:
        self._video.set_filter_text(self._preview_filter_text)
        self._video.set_filter_categories(self._quick_filters)
        self._render_tiles(self._last_preview_tiles)
        if self._focus_body.isVisible():
            self._rebuild_focus_scan_ui()
        else:
            self._focus_overlay.set_scan([], self._show_bbox, self._show_heat, self._heat_opacity, self._show_heat_tags)

    def _on_preview_filter_changed(self, text: str) -> None:
        self._preview_filter_text = str(text or "").strip().lower()
        self._apply_detection_filters()

    def _toggle_preview_mode(self) -> None:
        self._set_preview_mode("selective" if self._preview_mode == "preview" else "preview")

    def _set_preview_mode(self, mode: str) -> None:
        next_mode = "selective" if str(mode or "").strip().lower() == "selective" else "preview"
        if next_mode == self._preview_mode:
            return
        self._preview_mode = next_mode
        self._tile_cache.clear()
        if hasattr(self, "_preview_mode_btn"):
            selective = next_mode == "selective"
            self._preview_mode_btn.blockSignals(True)
            self._preview_mode_btn.setChecked(selective)
            self._preview_mode_btn.setText("Mode: Selective" if selective else "Mode: Preview")
            self._preview_mode_btn.blockSignals(False)
        self._render_tiles(self._last_preview_tiles)

    def _show_filter_menu(self) -> None:
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        for key, label in self._quick_filter_keys:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(key in self._quick_filters)
            act.toggled.connect(lambda checked, k=key: self._toggle_quick_filter(k, checked))
        self._sync_filter_button_caption()
        menu.exec(self._btn_filter.mapToGlobal(self._btn_filter.rect().topLeft()))

    def _sync_filter_button_caption(self) -> None:
        active_count = sum(1 for k, _ in self._quick_filter_keys if k in self._quick_filters)
        self._refresh_button_caption(
            self._btn_filter,
            f"[F] Filter ({active_count})" if active_count else "[F] Filter",
        )

    def _set_quick_filter_button_status(self, key: str, status: str) -> None:
        button = self._quick_filter_buttons.get(key)
        if button is None:
            return
        if isinstance(button, _StatusDotButton):
            button.set_status_state(status)
        else:
            button.setProperty("status_state", str(status or "none"))
            button.update()
        self._sync_radial_menu_actions()

    def _sync_quick_filter_buttons(self) -> None:
        for key, _label in self._quick_filter_keys:
            button = self._quick_filter_buttons.get(key)
            if button is None:
                continue
            target_checked = key in self._quick_filters
            if button.isChecked() == target_checked:
                # Keep status dot synchronized even when checked state is already right.
                self._set_quick_filter_button_status(key, "running" if target_checked else "none")
                continue
            button.blockSignals(True)
            button.setChecked(target_checked)
            button.blockSignals(False)
            self._set_quick_filter_button_status(key, "running" if target_checked else "none")

    def _toggle_quick_filter(self, key: str, checked: bool) -> None:
        if checked:
            self._quick_filters.add(key)
        else:
            self._quick_filters.discard(key)
        self._sync_quick_filter_buttons()
        self._set_quick_filter_button_status(key, "loading")
        self._sync_filter_button_caption()
        seq = int(self._quick_filter_toggle_seq.get(key, 0)) + 1
        self._quick_filter_toggle_seq[key] = seq

        def _apply_for_seq() -> None:
            if self._quick_filter_toggle_seq.get(key) != seq:
                return
            try:
                self._apply_detection_filters()
            except Exception as exc:
                self._set_quick_filter_button_status(key, "error")
                self._show_toast(f"Filter '{key}' failed: {exc}", "error")
                return
            self._set_quick_filter_button_status(
                key,
                "running" if key in self._quick_filters else "none",
            )
            self._sync_filter_button_caption()

        QTimer.singleShot(120, _apply_for_seq)

    def _history_entry_to_preview_tile(self, entry: dict[str, Any], now_ts: float) -> dict[str, Any]:
        captured_at = float(entry.get("captured_at", 0.0) or 0.0)
        return {
            "preview_state": "past",
            "track_id": int(entry.get("track_id", 0) or 0),
            "label": str(entry.get("label", "")),
            "confidence": float(entry.get("confidence", 0.0) or 0.0),
            "motion_score": 0.0,
            "age_seconds": max(0.0, now_ts - captured_at),
            "event_tag": str(entry.get("event_tag", "") or ""),
            "score": float(entry.get("score", 0.0) or 0.0),
            "image": str(entry.get("image", "") or ""),
            "recognized_identity": str(entry.get("recognized_identity", "") or ""),
            "recognition_confidence": float(entry.get("recognition_confidence", 0.0) or 0.0),
            "entry_id": int(entry.get("entry_id", 0) or 0),
            "captured_at": captured_at,
        }

    def _render_tiles(self, tiles: list) -> None:
        now = time.monotonic()
        now_ts = time.time()
        selective = self._preview_mode == "selective"
        LINGER_SEC = 1.5
        live_ids: set[int] = set()
        for t in tiles:
            tid = int(t["track_id"])
            live_ids.add(tid)
            if not selective:
                self._tile_cache[tid] = (dict(t), now)
        live_tiles: list[dict[str, Any]] = [dict(tile, preview_state="live") for tile in tiles]
        if selective:
            live_tiles = [
                dict(
                    tile,
                    source_image=str(tile.get("image", "") or ""),
                    image=str(tile.get("image", "") or ""),
                )
                for tile in live_tiles
                if self._tile_intersects_active_roi(tile)
            ]
            past_tiles: list[dict[str, Any]] = []
        else:
            for tid, (cached_tile, ts) in list(self._tile_cache.items()):
                if tid in live_ids:
                    continue
                if now - ts <= LINGER_SEC:
                    live_tiles.append(dict(cached_tile, preview_state="live"))
                else:
                    del self._tile_cache[tid]
            past_tiles = [
                self._history_entry_to_preview_tile(entry, now_ts)
                for entry in reversed(self._last_timeline_entries)
            ]
        live_tiles = [tile for tile in live_tiles if self._tile_matches_preview_filter(tile)]
        past_tiles = [tile for tile in past_tiles if self._tile_matches_preview_filter(tile)]
        if not live_tiles and not past_tiles:
            for widget in list(self._preview_widgets.values()):
                self._previews_layout.removeWidget(widget)
                widget.dispose()
                widget.deleteLater()
            self._preview_widgets.clear()
            if self._preview_live_header is not None:
                self._preview_live_header.hide()
            if self._preview_past_header is not None:
                self._preview_past_header.hide()
            if self._preview_empty is None:
                self._preview_empty = QLabel(
                    "Insight is scanning. Matching live detections and recent past cards will appear here."
                )
                self._preview_empty.setWordWrap(True)
                self._preview_empty.setStyleSheet(
                    f"padding: 16px; border: 1px dashed {theme_rgba('accent_dark', 0.45)}; color: rgba(20,8,8,0.74);"
                )
                self._previews_layout.addWidget(self._preview_empty)
            if selective:
                self._preview_empty.setText(
                    "Selective mode is waiting for current detections. Enable ROI to limit this list to that area."
                )
            else:
                self._preview_empty.setText(
                    "Insight is scanning. Matching live detections and recent past cards will appear here."
                )
            self._preview_empty.show()
            return
        if self._preview_empty is not None:
            self._preview_empty.hide()

        desired_keys = {
            f"live:{int(tile['track_id'])}"
            for tile in live_tiles
        } | {
            f"past:{int(tile['entry_id'])}"
            for tile in past_tiles
        }
        for key in list(self._preview_widgets.keys()):
            if key in desired_keys:
                continue
            widget = self._preview_widgets.pop(key)
            self._previews_layout.removeWidget(widget)
            widget.dispose()
            widget.deleteLater()

        live_header = self._ensure_preview_section_header("_preview_live_header", "LIVE")
        past_header = self._ensure_preview_section_header("_preview_past_header", f"PAST {HISTORY_TTL_SECONDS}S")
        index = 0
        if live_tiles:
            live_header.show()
            self._previews_layout.insertWidget(index, live_header)
            index += 1
        else:
            live_header.hide()
        for tile in live_tiles:
            key = f"live:{int(tile['track_id'])}"
            widget = self._preview_widgets.get(key)
            is_new_widget = widget is None
            if widget is None:
                widget = PreviewTileWidget()
                self._preview_widgets[key] = widget
            self._previews_layout.insertWidget(index, widget)
            if is_new_widget or not widget.isVisible():
                widget.soften_show()
            else:
                widget.show()
            index += 1

            def _on_tile(self=self, t=dict(tile)) -> None:
                self._history_focus_lock = False
                self._send({"type": "select_track", "track_id": int(t.get("track_id", 0) or 0)})
                self._apply_focus_panel(
                    {
                        "active": True,
                        "track_id": t.get("track_id"),
                        "label": t.get("label"),
                        "confidence": t.get("confidence"),
                        "motion_score": t.get("motion_score", 0),
                        "age_seconds": t.get("age_seconds", 0),
                        "event_tag": t.get("event_tag", ""),
                        "image": t.get("image"),
                        "silhouette": None,
                        "captured_at": t.get("captured_at"),
                        "entry_id": 0,
                    },
                    switch_tab=True,
                )

            widget.update_tile(
                tile,
                _on_tile,
                on_add=lambda t=dict(tile), self=self: self._on_add_preview_tile_to_cvops(t),
                thermal=self._active_heat,
            )
        if past_tiles:
            past_header.show()
            self._previews_layout.insertWidget(index, past_header)
            index += 1
        else:
            past_header.hide()
        for tile in past_tiles:
            key = f"past:{int(tile['entry_id'])}"
            widget = self._preview_widgets.get(key)
            is_new_widget = widget is None
            if widget is None:
                widget = PreviewTileWidget()
                self._preview_widgets[key] = widget
            self._previews_layout.insertWidget(index, widget)
            if is_new_widget or not widget.isVisible():
                widget.soften_show()
            else:
                widget.show()
            index += 1

            def _on_past_tile(self=self, t=dict(tile)) -> None:
                self._history_focus_lock = True
                self._apply_focus_panel(
                    {
                        "active": True,
                        "track_id": t.get("track_id"),
                        "label": t.get("label"),
                        "confidence": t.get("confidence"),
                        "motion_score": t.get("motion_score", 0),
                        "age_seconds": t.get("age_seconds", 0),
                        "event_tag": t.get("event_tag", ""),
                        "image": t.get("image"),
                        "silhouette": None,
                        "captured_at": t.get("captured_at"),
                        "entry_id": int(t.get("entry_id", 0) or 0),
                    },
                    switch_tab=True,
                )

            widget.update_tile(
                tile,
                _on_past_tile,
                on_add=lambda t=dict(tile), self=self: self._on_add_preview_tile_to_cvops(t),
                thermal=self._active_heat,
            )

    def _render_events(self, items: list) -> None:
        self._clear_layout(self._events_layout)
        if not items:
            empty = QLabel("No event transitions yet.")
            empty.setStyleSheet("padding: 16px; color: rgba(20,8,8,0.74);")
            self._events_layout.addWidget(empty)
            return
        for idx, it in enumerate(items):
            row = QFrame()
            row.setStyleSheet(
                "QFrame { background: transparent; border: none; "
                f"border-bottom: 1px solid {theme_rgba('accent_dark', 0.20)}; }}"
            )
            hl = QHBoxLayout(row)
            hl.setContentsMargins(10, 6, 10, 6)
            hl.setSpacing(8)
            tag = QLabel(str(it.get("tag", "")).upper())
            tag.setStyleSheet(
                "color: #140808; font-weight: 600; font-size: 10px; letter-spacing: 1px;"
            )
            txt = QLabel(str(it.get("text", "")))
            txt.setWordWrap(True)
            txt.setStyleSheet("color: #140808; font-size: 11px;")
            hl.addWidget(tag)
            hl.addWidget(txt, stretch=1)
            self._events_layout.addWidget(row)

    def _render_timeline(self, entries: list) -> None:
        signature = tuple(
            (
                int(entry.get("entry_id", 0)),
                int(entry.get("track_id", 0)),
                str(entry.get("label", "")),
                float(entry.get("confidence", 0.0)),
                str(entry.get("event_tag", "")),
                float(entry.get("age_seconds", 0.0)),
                float(entry.get("captured_at", 0.0)),
            )
            for entry in entries
        )
        self._tl_count.setText(str(len(entries)))
        if signature == self._timeline_signature:
            return

        bar = self._tl_scroll.horizontalScrollBar()
        prev_value = bar.value()
        prev_max = bar.maximum()
        while self._tl_layout.count():
            w = self._tl_layout.takeAt(0).widget()
            if w:
                w.deleteLater()
        self._timeline_signature = signature
        if not entries:
            self._tl_layout.addWidget(QLabel("No detections captured yet."))
            return
        for entry in reversed(entries):

            def _open_hist(e=entry, self=self):
                self._history_focus_lock = True
                self._apply_focus_panel(
                    {
                        "active": True,
                        "track_id": e.get("track_id"),
                        "label": e.get("label"),
                        "confidence": e.get("confidence"),
                        "motion_score": 0,
                        "age_seconds": e.get("age_seconds"),
                        "event_tag": e.get("event_tag"),
                        "image": e.get("image"),
                        "silhouette": None,
                        "captured_at": e.get("captured_at"),
                        "entry_id": int(e.get("entry_id", 0) or 0),
                    },
                    switch_tab=True,
                )

            def _del(e=entry, self=self):
                self._send({"type": "delete_history_entry", "entry_id": int(e["entry_id"])})

            card = TimelineCardWidget(
                entry,
                on_open=_open_hist,
                on_delete=_del,
                ttl_seconds=float(HISTORY_TTL_SECONDS),
            )
            self._tl_layout.addWidget(card)
        QTimer.singleShot(
            0,
            lambda bar=bar, prev_value=prev_value, prev_max=prev_max: bar.setValue(
                max(0, prev_value + max(0, bar.maximum() - prev_max)) if prev_value > 0 else 0
            ),
        )

    def _apply_focus_panel(self, focus: Optional[dict], switch_tab: bool = False) -> None:
        if not focus or not focus.get("active"):
            self._focus_payload = None
            self._focus_empty.show()
            self._focus_body.hide()
            self._focus_quick_box_pct = None
            self._focus_overlay.set_quick_box(None)
            self._focus_overlay.set_quick_draw_mode(False)
            self._focus_quick_draw_btn.setChecked(False)
            self._reset_roi_ai_state()
            self._roi_capture_payload = None
            self._focus_composite.clear_images()
            self._clear_layout(self._focus_stats_layout)
            self._focus_overlay.set_scan(None, False, False, self._heat_opacity, self._show_heat_tags)
            if self._recog_result_widget is not None:
                self._recog_result_widget.deleteLater()
                self._recog_result_widget = None
            return
        self._focus_payload = dict(focus)
        self._focus_empty.hide()
        self._focus_body.show()
        et = str(focus.get("event_tag", "") or "")
        is_roi = et.startswith("roi-")
        self._focus_controls.setVisible(True)
        self._focus_refresh.setVisible(is_roi)
        self._focus_quick_box_pct = None
        self._focus_overlay.set_quick_box(None)
        self._focus_overlay.set_quick_draw_mode(False)
        self._focus_quick_draw_btn.setChecked(False)
        self._load_focus_quick_scenarios()
        self._focus_title.setText(f"{detection_label_text(focus.get('label'))} #{focus.get('track_id')}")
        self._focus_sub.setText(
            f"Event: {et}  |  Confidence {format_confidence_percent(focus.get('confidence', 0))}"
        )
        pix = pixmap_from_b64_jpeg(str(focus.get("image", "")))
        sil = pixmap_from_b64_png(str(focus.get("silhouette", "") or ""))
        self._focus_base_pixmap = QPixmap(pix)
        self._focus_zoom = 1.0
        self._focus_zoom_lbl.setText("100%")
        self._focus_composite.set_pixmaps(pix, sil)
        self._focus_composite.set_zoom(1.0)
        self._clear_layout(self._focus_stats_layout)
        for idx, (label, value) in enumerate((
            ("Track", f"#{focus.get('track_id')}"),
            ("Motion", f"{round(float(focus.get('motion_score', 0)) * 100)}%"),
            ("Age", f"{float(focus.get('age_seconds', 0)):.1f}s"),
            ("State", et),
        )):
            self._focus_stats_layout.addWidget(
                self._build_focus_stat_chip(label, value), idx // 2, idx % 2
            )
        scan = focus.get("scan_results")
        prev_capture_at = self._roi_capture_payload.get("captured_at") if self._roi_capture_payload else None
        next_capture_at = focus.get("captured_at")
        if is_roi:
            if next_capture_at != prev_capture_at:
                self._reset_roi_ai_state()
            self._roi_capture_payload = dict(focus)
        else:
            self._reset_roi_ai_state()
            self._roi_capture_payload = None
        self._rebuild_focus_scan_ui(scan if isinstance(scan, list) else None)
        self._focus_visual.update()
        if switch_tab:
            self._tabs.setCurrentIndex(1)

    def _rebuild_focus_scan_ui(self, scan: Optional[list] = None) -> None:
        if scan is None and self._roi_capture_payload:
            scan = self._roi_capture_payload.get("scan_results")
        self._recog_result_widget = None
        self._clear_layout(self._focus_scan_layout)
        if scan is None:
            self._focus_overlay.set_scan(None, self._show_bbox, self._show_heat, self._heat_opacity, self._show_heat_tags)
            return
        raw_scan = list(scan)
        scan = self._filtered_scan_results(raw_scan)
        head = QLabel(f"[SCAN] {len(scan)} object(s) detected")
        head.setStyleSheet("font-size: 9px; color: rgba(20,8,8,0.72);")
        self._focus_scan_layout.addWidget(head)
        if len(scan) == 0:
            empty = QLabel("No objects match the current preview filter" if self._preview_filter_text else "No objects detected in ROI")
            empty.setStyleSheet("font-style: italic; color: rgba(20,8,8,0.68); font-size: 10px;")
            self._focus_scan_layout.addWidget(empty)
        toggles = QHBoxLayout()
        toggles.setContentsMargins(0, 0, 0, 0)
        toggles.setSpacing(4)
        cap = self._roi_capture_payload
        if cap:
            ai_btn = self._make_button("[AI] Ask AI", mode="both", checkable=True)
            ai_btn.blockSignals(True)
            ai_btn.setChecked(self._ai_open)
            ai_btn.blockSignals(False)
            ai_btn.toggled.connect(self._on_ai_toggled)
            toggles.addWidget(ai_btn)
        bb = self._make_button("[B] Boxes", mode="both", checkable=True)
        bb.blockSignals(True)
        bb.setChecked(self._show_bbox)
        bb.blockSignals(False)
        bb.toggled.connect(self._on_toggle_boxes)
        hh = self._make_button("[H] Heat M", mode="both", checkable=True)
        hh.blockSignals(True)
        hh.setChecked(self._show_heat)
        hh.blockSignals(False)
        hh.toggled.connect(self._on_toggle_heat)
        toggles.addWidget(bb)
        toggles.addWidget(hh)
        tw = QWidget()
        tw.setLayout(toggles)
        self._focus_scan_layout.addWidget(tw)
        if cap and self._ai_open:
            _OPENAI_MODELS = [
                "gpt-4.1-mini",
                "gpt-4.1",
                "gpt-4o",
                "gpt-4o-mini",
                "o4-mini",
            ]
            _ANTHROPIC_MODELS = [
                "claude-3-5-sonnet-latest",
                "claude-3-5-haiku-latest",
                "claude-3-7-sonnet-latest",
                "claude-opus-4-5",
            ]

            # --- provider row ---
            prov_row = QHBoxLayout()
            prov = QComboBox()
            for p in ("auto", "ollama", "openai", "anthropic"):
                prov.addItem(p)
            idx = max(0, prov.findText(self._ai_provider))
            prov.setCurrentIndex(idx)
            send = self._make_button("Sending..." if self._ai_busy else "Send", mode="title")
            send.setEnabled(not self._ai_busy)
            prov_row.addWidget(prov, stretch=1)
            prov_row.addWidget(send)
            prov_w = QWidget()
            prov_w.setLayout(prov_row)
            self._focus_scan_layout.addWidget(prov_w)

            # --- model row (hidden for "auto") ---
            model_row = QHBoxLayout()
            model_lbl = QLabel("Model")
            model_lbl.setStyleSheet("color: rgba(20,8,8,0.68); font-size: 10px;")
            model_row.addWidget(model_lbl)

            model_combo = QComboBox()
            model_combo.setEditable(False)
            model_edit = QLineEdit()
            model_edit.setPlaceholderText("e.g. llava:latest")

            # placeholder that we swap out by provider
            model_stack = QWidget()
            model_stack_layout = QHBoxLayout(model_stack)
            model_stack_layout.setContentsMargins(0, 0, 0, 0)

            def _populate_model_widgets(provider: str) -> None:
                for i in reversed(range(model_stack_layout.count())):
                    w = model_stack_layout.itemAt(i).widget()
                    if w:
                        w.setParent(None)
                if provider == "openai":
                    model_combo.clear()
                    for m in _OPENAI_MODELS:
                        model_combo.addItem(m)
                    saved = self._ai_models.get("openai", INSIGHT_OPENAI_MODEL)
                    saved_idx = model_combo.findText(saved)
                    model_combo.setCurrentIndex(max(0, saved_idx))
                    model_stack_layout.addWidget(model_combo)
                elif provider == "anthropic":
                    model_combo.clear()
                    for m in _ANTHROPIC_MODELS:
                        model_combo.addItem(m)
                    saved = self._ai_models.get("anthropic", INSIGHT_ANTHROPIC_MODEL)
                    saved_idx = model_combo.findText(saved)
                    model_combo.setCurrentIndex(max(0, saved_idx))
                    model_stack_layout.addWidget(model_combo)
                elif provider == "ollama":
                    model_edit.setText(self._ai_models.get("ollama", INSIGHT_OLLAMA_MODEL))
                    model_stack_layout.addWidget(model_edit)

            model_row.addWidget(model_stack, stretch=1)
            model_w = QWidget()
            model_w.setLayout(model_row)
            model_w.setVisible(self._ai_provider != "auto")
            self._focus_scan_layout.addWidget(model_w)

            _populate_model_widgets(self._ai_provider)

            def _prov_changed(i: int, self=self, c=prov, mw=model_w) -> None:
                self._ai_provider = c.currentText()
                mw.setVisible(self._ai_provider != "auto")
                _populate_model_widgets(self._ai_provider)

            prov.currentIndexChanged.connect(_prov_changed)

            def _model_combo_changed(i: int, self=self, cb=model_combo) -> None:
                if self._ai_provider in ("openai", "anthropic"):
                    self._ai_models[self._ai_provider] = cb.currentText()

            model_combo.currentIndexChanged.connect(_model_combo_changed)

            def _model_edit_changed(_text: str, _self=self, le=model_edit) -> None:
                _self._ai_models["ollama"] = le.text().strip() or INSIGHT_OLLAMA_MODEL

            model_edit.textChanged.connect(_model_edit_changed)

            def _send_ai(_checked=False, _self=self) -> None:
                _self._pin_sidebar_from_tab_peek()
                _self._ai_busy = True
                _self._ai_err = ""
                _self._ai_text = ""
                _self._send(
                    {
                        "type": "ask_ai_roi",
                        "provider": _self._ai_provider,
                        "model": _self._ai_models.get(_self._ai_provider, ""),
                        "prompt": _self._ai_prompt_draft,
                    }
                )
                _self._rebuild_focus_scan_ui(raw_scan)

            send.clicked.connect(_send_ai)

            prompt = QTextEdit()
            prompt.setPlaceholderText("Optional prompt. Leave empty for automatic analysis.")
            prompt.setMaximumHeight(72)
            prompt.setPlainText(self._ai_prompt_draft)
            prompt.setStyleSheet(
                f"QTextEdit {{ background: {theme_rgba('input_fill', 0.92)}; "
                f"border: 1px solid {theme_rgba('accent_dark', 0.42)}; "
                "color: #140808; font-size: 11px; padding: 6px 8px; }"
                f"QTextEdit:focus {{ border: 1px solid {theme_rgba('accent_dark', 0.72)}; }}"
            )
            prompt.textChanged.connect(lambda: setattr(self, "_ai_prompt_draft", prompt.toPlainText()))
            self._focus_scan_layout.addWidget(prompt)
        op_row = QHBoxLayout()
        op_row.addWidget(QLabel("Opacity"))
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(10, 100)
        sl.setValue(int(self._heat_opacity * 100))

        def _op(v: int, self=self):
            self._heat_opacity = v / 100.0
            self._focus_overlay.set_scan(scan, self._show_bbox, self._show_heat, self._heat_opacity, self._show_heat_tags)

        sl.valueChanged.connect(_op)
        op_l = QLabel(f"{round(self._heat_opacity * 100)}%")

        def _op2(v: int, self=self, lbl=op_l):
            lbl.setText(f"{v}%")

        sl.valueChanged.connect(_op2)
        op_row.addWidget(sl)
        op_row.addWidget(op_l)
        opw = QWidget()
        opw.setLayout(op_row)
        opw.setVisible(self._show_heat)
        self._heat_opacity_row = opw
        self._focus_scan_layout.addWidget(opw)
        # Detection list — compact summary rows
        det_container = QWidget()
        det_layout = QVBoxLayout(det_container)
        det_layout.setContentsMargins(0, 0, 0, 0)
        det_layout.setSpacing(0)
        # Preview panel (hidden until Show More)
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 4, 0, 0)
        preview_layout.setSpacing(4)
        preview_container.setVisible(self._show_scan_previews)
        for item in scan:
            pct = round(float(item.get("confidence", 0)) * 100)
            conf_text = format_confidence_percent(item.get("confidence", 0))
            # Compact summary line
            row = QFrame()
            row.setStyleSheet(
                "QFrame { background: transparent; border: none; "
                f"border-bottom: 1px solid {theme_rgba('accent_dark', 0.16)}; }}"
            )
            line = QHBoxLayout(row)
            line.setContentsMargins(4, 3, 4, 3)
            line.setSpacing(6)
            _hex_map2 = VideoPane._THERMAL_HEX if self._active_heat else PreviewTileWidget._CAT_HEX
            _chx2 = _hex_map2.get(heatmap_category(item.get("label", "")), "#3c8cff")
            item_label = detection_label_text(item.get("label", ""))
            label = QLabel(
                f"<span style='display:inline-block;width:7px;height:7px;border-radius:50%;background:{_chx2};margin-right:3px;'></span>"
                f"<b style='color:#140808'>{item_label}</b>"
            )
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setStyleSheet("font-size: 10px;")
            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(pct)
            bar.setTextVisible(False)
            bar.setFixedHeight(6)
            bar.setStyleSheet(get_hud_meter_css(2))
            conf_lbl = QLabel(conf_text)
            conf_lbl.setStyleSheet(f"color: {text_hex()}; font-size: 9px;")
            line.addWidget(label)
            line.addWidget(bar, stretch=1)
            line.addWidget(conf_lbl)
            det_layout.addWidget(row)
            # Preview row (inside expandable section)
            crop_b64 = item.get("crop_b64", "")
            if crop_b64:
                prev_row = QFrame()
                prev_row.setStyleSheet(
                    f"QFrame {{ background: {theme_rgba('accent_dark', 0.30)}; "
                    f"border-bottom: 1px solid {theme_rgba('accent_dark', 0.12)}; }}"
                )
                pr_layout = QHBoxLayout(prev_row)
                pr_layout.setContentsMargins(4, 4, 4, 4)
                pr_layout.setSpacing(6)
                thumb = QLabel()
                thumb.setFixedSize(56, 56)
                thumb.setScaledContents(True)
                thumb.setPixmap(
                    pixmap_from_b64_jpeg(crop_b64).scaled(
                        56, 56,
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
                thumb.setStyleSheet(f"border: 1px solid {theme_rgba('accent_dark', 0.35)};")
                pr_info = QVBoxLayout()
                pr_info.setSpacing(1)
                pr_name = QLabel(f"<b style='color:#140808'>{item_label}</b>")
                pr_name.setTextFormat(Qt.TextFormat.RichText)
                pr_name.setStyleSheet("font-size: 10px;")
                pr_conf = QLabel(f"{conf_text} conf  |  {item.get('area_pct', 0)}% area")
                pr_conf.setStyleSheet("font-size: 9px; color: rgba(20,8,8,0.62);")
                pr_info.addWidget(pr_name)
                pr_info.addWidget(pr_conf)
                pr_layout.addWidget(thumb)
                pr_layout.addLayout(pr_info, stretch=1)
                enroll_btn = self._make_button("[+] Enroll", mode="icon", tooltip="Enroll this detection")
                enroll_btn.setFixedWidth(28)
                enroll_btn.setToolTip("Enroll this detection into the recognition gallery")

                def _on_enroll(_=False, self=self, c=crop_b64, lbl=item.get("label", "unknown")):
                    name = self._focus_burst_identity.text().strip() if hasattr(self, "_focus_burst_identity") else ""
                    if not name:
                        name = str(lbl or "unknown").strip()
                    if not name or name == "unknown":
                        self._show_toast("Enter an identity before enrolling this ROI crop", "warn")
                        if hasattr(self, "_focus_burst_identity"):
                            self._focus_burst_identity.setFocus()
                        return
                    self._send({
                        "type": "enroll_scan_crop",
                        "crop_b64": c,
                        "identity": name,
                        "group": "scan",
                    })
                    self._show_toast(f"Enrolling '{name}'...", "info")

                enroll_btn.clicked.connect(_on_enroll)
                pr_layout.addWidget(enroll_btn)
                preview_layout.addWidget(prev_row)
        self._focus_scan_layout.addWidget(det_container)
        # Show More / Show Less toggle
        if scan:
            show_more_btn = self._make_button(
                "Show Less" if self._show_scan_previews else "Show More",
                mode="title",
            )
            show_more_btn.setStyleSheet(
                "QPushButton { font-size: 9px; color: rgba(20,8,8,0.76); "
                f"border: 1px solid {theme_rgba('accent_dark', 0.25)}; background: {theme_rgba('panel', 0.34)}; padding: 2px 8px; }}"
                f"QPushButton:hover {{ color: rgba(20,8,8,0.94); border-color: {theme_rgba('accent_dark', 0.40)}; }}"
            )

            def _toggle_previews(_=False, self=self, pc=preview_container, btn=show_more_btn):
                self._show_scan_previews = not self._show_scan_previews
                pc.setVisible(self._show_scan_previews)
                self._refresh_button_caption(btn, "Show Less" if self._show_scan_previews else "Show More")

            show_more_btn.clicked.connect(_toggle_previews)
            self._focus_scan_layout.addWidget(show_more_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        self._focus_scan_layout.addWidget(preview_container)
        if cap and (self._ai_busy or self._ai_text or self._ai_err):
            ai_head = QLabel(f"[AI] {self._ai_provider}")
            self._focus_scan_layout.addWidget(ai_head)
            body = QLabel(self._ai_err or self._ai_text or "Waiting...")
            body.setWordWrap(True)
            if self._ai_err:
                body.setStyleSheet(f"color: {theme_rgba('accent_dark', 0.86)};")
            self._focus_scan_layout.addWidget(body)
        self._focus_overlay.set_scan(scan, self._show_bbox, self._show_heat, self._heat_opacity, self._show_heat_tags)

    def _on_ai_toggled(self, on: bool) -> None:
        self._pin_sidebar_from_tab_peek()
        self._ai_open = on
        self._rebuild_focus_scan_ui()

    def _pin_sidebar_from_tab_peek(self) -> None:
        """If Tab-peek opened sidebar, keep it open after Tab release once AI is interacted with."""
        if self._tab_peek_active and "sidebar" in self._tab_peek_activated:
            self._tab_peek_activated.discard("sidebar")

    def _on_toggle_boxes(self, on: bool) -> None:
        self._show_bbox = on
        scan = self._filtered_scan_results(self._roi_capture_payload.get("scan_results") if self._roi_capture_payload else None)
        self._focus_overlay.set_scan(scan or [], self._show_bbox, self._show_heat, self._heat_opacity, self._show_heat_tags)

    def _on_toggle_heat(self, on: bool) -> None:
        self._show_heat = on
        if hasattr(self, "_heat_opacity_row"):
            self._heat_opacity_row.setVisible(on)
        scan = self._filtered_scan_results(self._roi_capture_payload.get("scan_results") if self._roi_capture_payload else None)
        self._focus_overlay.set_scan(scan or [], self._show_bbox, self._show_heat, self._heat_opacity, self._show_heat_tags)

    def _capture_tab_takeover_snapshot(self) -> dict[str, Any]:
        return {
            "bottom_hud_visible": bool(self._bottom_hud.isVisible()),
            "sidebar_visible": bool(self._sidebar_panel.isVisible()),
            "catalog_visible": bool(self._catalog_panel.isVisible()),
            "timeline_visible": bool(self._timeline.isVisible()),
            "tab_index": int(self._tabs.currentIndex()),
            "catalog_tab_index": int(self._catalog_tabs.currentIndex()),
        }

    @staticmethod
    def _tab_takeover_snapshot_meta(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "bottom_hud_visible": bool(snapshot.get("bottom_hud_visible", True)),
            "sidebar_visible": bool(snapshot.get("sidebar_visible", False)),
            "catalog_visible": bool(snapshot.get("catalog_visible", False)),
            "timeline_visible": bool(snapshot.get("timeline_visible", False)),
            "tab_index": int(snapshot.get("tab_index", 0) or 0),
            "catalog_tab_index": int(snapshot.get("catalog_tab_index", 0) or 0),
        }

    def _dismiss_for_tab_takeover(self) -> None:
        if hasattr(self, "_radial_menu") and self._radial_menu.isVisible():
            self._radial_menu.close_menu()
        if self._sidebar_panel.isVisible():
            self._sidebar_panel.dismiss_animated()
        if self._catalog_panel.isVisible():
            self._catalog_panel.dismiss_animated()
        if self._timeline.isVisible():
            self._close_timeline()
        if not self._bottom_hud.isVisible():
            self._toggle_bottom_hud()
        self._tabs.setCurrentIndex(self._TAB_PREVIEWS)

    def _apply_tab_takeover_snapshot(self, snapshot: dict[str, Any]) -> None:
        target_bottom_visible = bool(snapshot.get("bottom_hud_visible", True))
        if self._bottom_hud.isVisible() != target_bottom_visible:
            self._toggle_bottom_hud()
        if bool(snapshot.get("sidebar_visible", False)):
            self._restore_sidebar()
        elif self._sidebar_panel.isVisible():
            self._sidebar_panel.dismiss_animated()
        if bool(snapshot.get("catalog_visible", False)):
            self._restore_catalog()
        elif self._catalog_panel.isVisible():
            self._catalog_panel.dismiss_animated()
        if bool(snapshot.get("timeline_visible", False)):
            self._btn_timeline.blockSignals(True)
            self._btn_timeline.setChecked(True)
            self._btn_timeline.blockSignals(False)
            self._timeline.show()
            self._position_bottom_overlays()
        elif self._timeline.isVisible():
            self._close_timeline()
        self._tabs.setCurrentIndex(int(snapshot.get("tab_index", self._TAB_PREVIEWS)))
        self._catalog_tabs.setCurrentIndex(int(snapshot.get("catalog_tab_index", self._CAT_TAB_GALLERY)))

    def _on_tab_takeover_toggled(self, enabled: bool) -> None:
        if enabled:
            snapshot = self._capture_tab_takeover_snapshot()
            self._tab_takeover_snapshot = dict(snapshot)
            self._dismiss_for_tab_takeover()
            self._tab_peek_engage()
            self._tab_takeover_active = True
            self._persist_setting("tab_takeover_active", True)
            self._persist_setting("tab_takeover_snapshot", self._tab_takeover_snapshot_meta(snapshot))
            return
        if self._tab_peek_active:
            self._tab_peek_dismiss()
        if self._tab_takeover_snapshot:
            self._apply_tab_takeover_snapshot(self._tab_takeover_snapshot)
        self._tab_takeover_snapshot = None
        self._tab_takeover_active = False
        self._persist_setting("tab_takeover_active", False)
        self._persist_setting("tab_takeover_snapshot", {})

    def _tab_peek_engage(self) -> None:
        """Activate each configured hotkey peek option that isn't already on."""
        opts = self._hotkey_peek_options
        activated: set[str] = set()
        if "roi" in opts and not self._video.roi_active:
            self._btn_roi.setChecked(True)
            self._toggle_roi()
            activated.add("roi")
        if "sidebar" in opts and not self._sidebar_panel.isVisible():
            self._restore_sidebar()
            activated.add("sidebar")
        if "active" in opts and not self._active_mode:
            self._btn_active.setChecked(True)
            self._toggle_active_mode()
            activated.add("active")
        if "thermal" in opts and not self._active_heat:
            self._btn_active_heat.setChecked(True)
            self._toggle_active_heat()
            activated.add("thermal")
        if activated:
            self._tab_peek_active = True
            self._tab_peek_activated = activated

    def _tab_peek_dismiss(self) -> None:
        """Tear down only what the peek activated."""
        activated = self._tab_peek_activated
        if "roi" in activated:
            self._btn_roi.setChecked(False)
            self._toggle_roi()
        if "sidebar" in activated:
            self._sidebar_panel.dismiss_animated()
        if "active" in activated:
            self._btn_active.setChecked(False)
            self._toggle_active_mode()
        if "thermal" in activated:
            self._btn_active_heat.setChecked(False)
            self._toggle_active_heat()
        self._tab_peek_active = False
        self._tab_peek_activated = set()

    def keyPressEvent(self, event) -> None:
        k = event.key()
        if k == Qt.Key.Key_Tab and not event.isAutoRepeat():
            if self._tab_takeover_active:
                event.accept()
                return
            if not self._tab_peek_active:
                self._tab_peek_engage()
            event.accept()
            return
        if k == Qt.Key.Key_Escape:
            if self._history_focus_lock or self._focus_body.isVisible() and not self._focus_empty.isVisible():
                self._clear_focus_ui()
                return
            if self._video.roi_active:
                self._btn_roi.setChecked(False)
                self._toggle_roi()
                return
        if k == Qt.Key.Key_V:
            self._send({"type": "switch_source"})
        if k == Qt.Key.Key_R:
            self._btn_roi.setChecked(not self._btn_roi.isChecked())
            self._toggle_roi()
        if k == Qt.Key.Key_G:
            self._toggle_grid()
        if k == Qt.Key.Key_Q:
            self._toggle_subgrid()
        if k == Qt.Key.Key_N:
            self._fit_nreal()
        if k == Qt.Key.Key_D:
            self._toggle_sidebar()
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if (
            event.key() == Qt.Key.Key_Tab
            and not event.isAutoRepeat()
            and self._tab_peek_active
            and not self._tab_takeover_active
        ):
            self._tab_peek_dismiss()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:
        if getattr(self, "_global_event_filter_installed", False):
            try:
                from PyQt6.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None:
                    app.removeEventFilter(self)
            except Exception:
                pass
            self._global_event_filter_installed = False
        self._effects_worker.stop()
        self.session.close()
        super().closeEvent(event)
