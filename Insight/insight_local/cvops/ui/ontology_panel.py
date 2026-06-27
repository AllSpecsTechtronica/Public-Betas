"""Ontology Surface panel — Ecosystem star-chart view.

Renders the full entity graph (scenarios, models, datasets, jobs, snapshots,
lineages, ranges, catalog assets) as a Cytoscape.js force-directed graph
inside a QWebEngineView. Node clicks emit ``entitySelected(type, id)`` so the
main window can route focus to the relevant native panel.

Falls back gracefully if PyQt6-WebEngine is not installed.
"""
from __future__ import annotations

import base64
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import QPropertyAnimation, QThread, QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ...ui.theme import theme_rgba
from ...ui.media_utils import pixmap_from_b64_jpeg
from .cvops_theme import (
    WB_ACCENT_ACTIVE,
    WB_ACCENT_ALERT,
    WB_ACCENT_SELECT,
    WB_ACCENT_WARN,
    WB_BG_GRAPHITE,
    WB_BG_PANEL,
    WB_BG_VOID,
    WB_EDGE_BELONGS_TO,
    WB_EDGE_DERIVED_FROM,
    WB_EDGE_EVALUATES,
    WB_EDGE_GOVERNED_BY,
    WB_EDGE_HAS_HEAD,
    WB_EDGE_PRODUCES,
    WB_LINE_LIGHT,
    WB_LINE_MED,
    WB_NODE_BACKBONE,
    WB_NODE_CATALOG,
    WB_NODE_CELL,
    WB_NODE_DATABASE,
    WB_NODE_DATASET,
    WB_NODE_JOB,
    WB_NODE_LINEAGE,
    WB_NODE_MODEL,
    WB_NODE_RANGE,
    WB_NODE_SCENARIO,
    WB_NODE_SNAPSHOT,
    WB_TEXT_BRIGHT,
    WB_TEXT_IRON,
    WB_TEXT_SIGNAL,
    cvops_color,
    cvops_qcolor,
    set_cvops_stylesheet,
)

# Primary source is the local service — always reachable, no TLS / proxy issues.
# Fallbacks are public CDNs in case the local cache endpoint fails for any reason.
_CYTOSCAPE_LOCAL_PATH = "/ontology/cytoscape.js"
_CYTOSCAPE_FALLBACK = "https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js"


class _GraphFetcher(QThread):
    """Fetches /ontology/graph in a background thread to avoid blocking the UI."""

    fetched: pyqtSignal = pyqtSignal(str, dict)
    failed: pyqtSignal = pyqtSignal(str, str)

    def __init__(self, key: str, url: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key
        self._url = url

    def run(self) -> None:
        try:
            with urllib.request.urlopen(self._url, timeout=10) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            self.fetched.emit(self._key, data if isinstance(data, dict) else {})
        except Exception as exc:
            self.failed.emit(self._key, str(exc))


class _JsonFetcher(QThread):
    """Fetches a JSON endpoint in the background."""

    fetched: pyqtSignal = pyqtSignal(str, dict)
    failed: pyqtSignal = pyqtSignal(str, str)

    def __init__(self, key: str, url: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._key = key
        self._url = url

    def run(self) -> None:
        try:
            with urllib.request.urlopen(self._url, timeout=8) as resp:  # noqa: S310
                data = json.loads(resp.read().decode("utf-8"))
            self.fetched.emit(self._key, data if isinstance(data, dict) else {})
        except Exception as exc:
            self.failed.emit(self._key, str(exc))


class _ActionRunner(QThread):
    """Runs a small command-deck HTTP action without blocking the Qt UI."""

    finishedOk: pyqtSignal = pyqtSignal(str, dict)
    failed: pyqtSignal = pyqtSignal(str, str)

    def __init__(
        self,
        *,
        label: str,
        method: str,
        url: str,
        body: Optional[dict[str, Any]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._method = method.upper()
        self._url = url
        self._body = body

    def run(self) -> None:
        try:
            data = None
            headers: dict[str, str] = {}
            if self._body is not None:
                data = json.dumps(self._body).encode("utf-8")
                headers["Content-Type"] = "application/json"
            req = urllib.request.Request(
                self._url,
                data=data,
                headers=headers,
                method=self._method,
            )
            with urllib.request.urlopen(req, timeout=15) as resp:  # noqa: S310
                raw = resp.read().decode("utf-8")
            try:
                payload = json.loads(raw) if raw else {}
            except Exception:
                payload = {"raw": raw}
            self.finishedOk.emit(self._label, payload if isinstance(payload, dict) else {})
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = str(exc)
            self.failed.emit(self._label, f"[{exc.code}] {detail[:240]}")
        except Exception as exc:
            self.failed.emit(self._label, str(exc))


class _ViewportLockedScrollArea(QScrollArea):
    """Scroll area that keeps its child width locked to the current viewport width."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._locked_widget: Optional[QWidget] = None
        self.horizontalScrollBar().rangeChanged.connect(self._on_horizontal_range_changed)

    def setWidget(self, widget: QWidget) -> None:  # type: ignore[override]
        self._locked_widget = widget
        super().setWidget(widget)
        self._sync_locked_width()
        QTimer.singleShot(0, self._sync_locked_width)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._sync_locked_width()

    def _on_horizontal_range_changed(self, _minimum: int, _maximum: int) -> None:
        self._sync_locked_width()

    def _sync_locked_width(self) -> None:
        if self._locked_widget is None:
            return
        width = max(0, self.viewport().width())
        if width <= 0:
            return
        self._locked_widget.setMinimumWidth(width)
        self._locked_widget.setMaximumWidth(width)
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().minimum())


class _MatrixStatusWidget(QWidget):
    """Stable-size status strip with old-computer LED matrix activity."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._status = "IDLE"
        self._rng = random.Random()
        self._rows = 8
        self._cols = 44
        self._cells: list[float] = [0.0] * (self._rows * self._cols)
        self._timer = QTimer(self)
        self._timer.setInterval(70)
        self._timer.timeout.connect(self._advance)
        self.setMinimumHeight(58)
        self.setMaximumHeight(58)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

    def set_status(self, status: str) -> None:
        normalized = str(status or "").strip().upper()
        if normalized not in {"TRAINING", "RUNNING", "FAILED", "SUCCESS", "IDLE"}:
            normalized = "IDLE"
        if normalized == "RUNNING":
            normalized = "TRAINING"
        if self._status == normalized:
            return
        self._status = normalized
        if self._status == "TRAINING":
            self._timer.start()
        else:
            self._timer.stop()
        self.update()

    def _advance(self) -> None:
        self._cells = [max(0.0, value - 0.22) for value in self._cells]
        for _ in range(76):
            self._cells[self._rng.randrange(len(self._cells))] = self._rng.uniform(0.55, 1.0)
        for _ in range(14):
            col = self._rng.randrange(self._cols)
            row = self._rng.randrange(self._rows)
            idx = row * self._cols + col
            self._cells[idx] = 1.0
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -1, -1)
        if self._status == "SUCCESS":
            painter.fillRect(rect, cvops_qcolor("accent_select"))
            painter.setPen(cvops_qcolor("bg_void"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "SUCCESS")
            return

        if self._status == "FAILED":
            painter.setPen(QPen(cvops_qcolor("accent_alert"), 1.4))
            painter.drawRect(rect)
            painter.drawText(rect.adjusted(8, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, "FAILED")
            self._draw_matrix(painter, rect.adjusted(72, 6, -6, -6), failed=True)
            return

        if self._status != "TRAINING":
            painter.setPen(QPen(cvops_qcolor("line_light"), 1.0))
            painter.drawRect(rect)
            painter.setPen(cvops_qcolor("text_iron"))
            painter.drawText(rect.adjusted(8, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, "IDLE")
            self._draw_matrix(painter, rect.adjusted(72, 6, -6, -6), idle=True)
            return

        painter.setPen(QPen(cvops_qcolor("line_light"), 1.0))
        painter.drawRect(rect)
        painter.setPen(cvops_qcolor("accent_active"))
        painter.drawText(rect.adjusted(8, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, "TRAINING")
        self._draw_matrix(painter, rect.adjusted(72, 6, -6, -6))

    def _draw_matrix(
        self,
        painter: QPainter,
        rect,
        *,
        failed: bool = False,
        idle: bool = False,
    ) -> None:
        rows = self._rows
        cols = self._cols
        gap = 1
        cell_w = max(2, (rect.width() - gap * (cols - 1)) / max(1, cols))
        cell_h = max(2, (rect.height() - gap * (rows - 1)) / max(1, rows))
        base = cvops_qcolor("line_light")
        active = cvops_qcolor("accent_alert" if failed else "accent_active")
        idle_color = cvops_qcolor("text_iron")
        for row in range(rows):
            for col in range(cols):
                idx = row * cols + col
                if failed:
                    lit = row in {3, 4, 5} and col % 3 != 1
                    color = active if lit else base
                    alpha = 210 if lit else 45
                elif idle:
                    lit = (row + col) % 7 == 0
                    color = idle_color if lit else base
                    alpha = 90 if lit else 35
                else:
                    value = self._cells[idx]
                    color = active if value > 0 else base
                    alpha = int(35 + value * 220)
                color.setAlpha(max(0, min(255, alpha)))
                painter.fillRect(
                    int(rect.left() + col * (cell_w + gap)),
                    int(rect.top() + row * (cell_h + gap)),
                    max(1, int(cell_w)),
                    max(1, int(cell_h)),
                    color,
                )


_NODE_COLORS: dict[str, str] = {
    "scenario":         WB_NODE_SCENARIO,
    "backbone":         WB_NODE_BACKBONE,
    "cell":             WB_NODE_CELL,
    "dataset":          WB_NODE_DATASET,
    "dataset_snapshot": WB_NODE_SNAPSHOT,
    "model_version":    WB_NODE_MODEL,
    "model_snapshot":   WB_NODE_SNAPSHOT,
    "job":              WB_NODE_JOB,
    "lineage":          WB_NODE_LINEAGE,
    "range":            WB_NODE_RANGE,
    "catalog_asset":    WB_NODE_CATALOG,
    "database":         WB_NODE_DATABASE,
    "prov_activity":    WB_ACCENT_WARN,
    "prov_agent":       WB_TEXT_IRON,
    "prov_entity":      WB_ACCENT_SELECT,
    "identity":         WB_ACCENT_SELECT,
    "correction_event": WB_ACCENT_WARN,
    "sector":           WB_TEXT_IRON,
    "collection":       WB_LINE_MED,
}

_EDGE_COLORS: dict[str, str] = {
    "belongs_to":    WB_EDGE_BELONGS_TO,
    "governed_by":   WB_EDGE_GOVERNED_BY,
    "produces":      WB_EDGE_PRODUCES,
    "evaluates":     WB_EDGE_EVALUATES,
    "derived_from":  WB_EDGE_DERIVED_FROM,
    "has_head":      WB_EDGE_HAS_HEAD,
    "uses_backbone": WB_NODE_BACKBONE,
    "contains_cell": WB_NODE_CELL,
    "trains_on":     WB_NODE_DATASET,
    "branched_from": WB_EDGE_DERIVED_FROM,
    "stores_in":     WB_NODE_DATABASE,
    "prov_generates": WB_EDGE_PRODUCES,
    "prov_used":      WB_ACCENT_WARN,
    "prov_informed_by": WB_LINE_MED,
    "prov_associated": WB_TEXT_IRON,
    "had_member":    WB_EDGE_HAS_HEAD,
    "specialization_of": WB_EDGE_DERIVED_FROM,
    "prov_invalidated": WB_ACCENT_ALERT,
    "prov_attributed": WB_EDGE_GOVERNED_BY,
    "flagged_in":          WB_ACCENT_WARN,
    "flagged_by_model":    WB_ACCENT_WARN,
    "contains_sector":     WB_TEXT_IRON,
    "organized_in":        WB_LINE_MED,
    "catalogued_in":       WB_NODE_CATALOG,
    "shares_backbone_with": WB_NODE_BACKBONE,
    "shares_dataset_with":  WB_NODE_DATASET,
}


def _node_colors() -> dict[str, str]:
    # Keep entity distinctions but derive the active/process channels from the current Cv Ops palette.
    return {
        **_NODE_COLORS,
        "model_version": cvops_color("accent_active"),
        "range": cvops_color("accent_warn"),
        "database": cvops_color("accent_select"),
        "prov_activity": cvops_color("accent_warn"),
        "prov_entity": cvops_color("accent_select"),
    }


def _edge_colors() -> dict[str, str]:
    return {
        **_EDGE_COLORS,
        "governed_by": cvops_color("accent_active"),
        "evaluates": cvops_color("accent_warn"),
        "stores_in": cvops_color("accent_select"),
        "prov_used": cvops_color("accent_warn"),
        "prov_informed_by": cvops_color("line_med"),
    }

_ENTITY_TYPE_LABELS: list[tuple[str, str]] = [
    ("scenario",         "Scenario"),
    ("backbone",         "Backbone"),
    ("dataset",          "Dataset"),
    ("model_version",    "Model"),
    ("dataset_snapshot", "Data-Snap"),
    ("job",              "Job"),
    ("model_snapshot",   "Snap"),
    ("lineage",          "Lineage"),
    ("range",            "Range"),
    ("catalog_asset",    "Catalog"),
    ("prov_activity",    "PROV Act"),
    ("prov_agent",       "PROV Agt"),
    ("prov_entity",      "PROV Ent"),
    ("identity",         "Identity"),
    ("correction_event", "Correction"),
    ("sector",           "Sector"),
    ("collection",       "Collection"),
]

def _filter_btn_style() -> str:
    return f"""
QPushButton {{
    font-size: 10px;
    font-weight: 600;
    padding: 6px 14px;
    border: 1px solid {cvops_color('line_light')};
    background: transparent;
    color: {cvops_color('text_iron')};
    border-radius: 0px;
}}
QPushButton:checked {{
    background: {cvops_color('selection_active')};
    color: {cvops_color('selection_text')};
    border-color: {cvops_color('selection_edge')};
}}
QPushButton:hover:!checked {{
    color: {cvops_color('text_signal')};
    border-color: {cvops_color('text_iron')};
}}
"""


def _card_qss() -> str:
    return f"""
QFrame[ecoCard="true"] {{
    background: {theme_rgba("panel", 0.58)};
    border: 1px solid {cvops_color('line_light')};
}}
QFrame[ecoCard="true"][selected="true"] {{
    border: 1px solid {cvops_color('accent_active')};
    background: {theme_rgba("panel", 0.74)};
}}
QLabel[ecoTitle="true"] {{
    color: {cvops_color('text_bright')};
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
}}
QLabel[ecoMeta="true"] {{
    color: {cvops_color('text_iron')};
    font-size: 9px;
}}
QLabel[ecoValue="true"] {{
    color: {cvops_color('text_signal')};
    font-size: 10px;
}}
QLabel[ecoPulse="true"] {{
    color: {cvops_color('accent_active')};
    font-size: 11px;
    font-weight: 700;
}}
"""


def _card_button_qss() -> str:
    return f"""
QPushButton {{
    font-size: 9px;
    font-weight: 700;
    padding: 8px 14px;
    border: 1px solid {cvops_color('line_med')};
    background: transparent;
    color: {cvops_color('text_signal')};
    border-radius: 0px;
    font-family: "JetBrains Mono";
}}
QPushButton:hover {{
    border-color: {cvops_color('accent_active')};
    color: {cvops_color('accent_active')};
}}
QPushButton[buttonRole="primary"] {{
    border-color: {cvops_color('accent_active')};
    color: {cvops_color('accent_active')};
}}
QPushButton[buttonRole="danger"] {{
    border-color: {cvops_color('accent_alert')};
    color: {cvops_color('accent_alert')};
}}
QPushButton:checked {{
    background: {cvops_color('selection_active')};
    color: {cvops_color('selection_text')};
    border-color: {cvops_color('selection_edge')};
}}
QPushButton:disabled {{
    opacity: 0.42;
}}
"""


def _short(value: Any, limit: int = 80) -> str:
    text = str(value if value is not None else "").strip()
    if not text:
        return "-"
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "..."


def _fmt_bytes(value: Any) -> str:
    try:
        n = float(value)
    except Exception:
        return "-"
    units = ("B", "KB", "MB", "GB", "TB")
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    if idx == 0:
        return f"{int(n)} {units[idx]}"
    return f"{n:.1f} {units[idx]}"


def _fmt_metric(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.3f}"
    return _short(value, 24)


def _fmt_pct(value: Any) -> str:
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "-"


def _status_label(state: str) -> str:
    normalized = str(state or "").strip().lower()
    if normalized in {"running", "queued", "accepted", "starting"}:
        return "RUNNING"
    if normalized in {"done", "completed", "success", "succeeded"}:
        return "SUCCESS"
    if normalized in {"error", "failed", "cancelled", "interrupted"}:
        return "FAILED"
    return normalized.upper() if normalized else "-"


def _ontology_toolbar_bg() -> str:
    return theme_rgba("panel", 0.52)


def _ontology_field_bg() -> str:
    return theme_rgba("input_fill", 0.62)


def _ontology_float_bg() -> str:
    return theme_rgba("panel", 0.78)


def _ontology_float_bg_soft() -> str:
    return theme_rgba("panel", 0.66)


def _ontology_scrim_bg() -> str:
    return "rgba(0,0,0,0.42)"


def _ontology_accent_wash() -> str:
    return theme_rgba("accent_dark", 0.12)


def _ontology_accent_wash_strong() -> str:
    return theme_rgba("accent_dark", 0.18)


def _build_html(
    graph: dict[str, Any],
    base_url: str,
    cytoscape_cdn: str,
    cytoscape_fallback: str = "",
) -> str:
    ontology_float_bg = _ontology_float_bg()
    ontology_float_bg_soft = _ontology_float_bg_soft()
    ontology_scrim_bg = _ontology_scrim_bg()
    ontology_accent_wash = _ontology_accent_wash()
    ontology_accent_wash_strong = _ontology_accent_wash_strong()
    nodes_json = json.dumps(graph.get("nodes") or [])
    edges_json = json.dumps(graph.get("edges") or [])
    node_colors_json = json.dumps(_node_colors())
    edge_colors_json = json.dumps(_edge_colors())
    fallback_src = cytoscape_fallback or ""
    WB_ACCENT_ACTIVE = cvops_color("accent_active")
    WB_ACCENT_ALERT = cvops_color("accent_alert")
    WB_ACCENT_SELECT = cvops_color("accent_select")
    WB_BG_GRAPHITE = cvops_color("bg_graphite")
    WB_BG_VOID = cvops_color("bg_void")
    WB_LINE_LIGHT = cvops_color("line_light")
    WB_LINE_MED = cvops_color("line_med")
    WB_TEXT_BRIGHT = cvops_color("text_bright")
    WB_TEXT_IRON = cvops_color("text_iron")
    WB_TEXT_SIGNAL = cvops_color("text_signal")
    WB_NODE_DATABASE = _node_colors().get("database", cvops_color("accent_select"))

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  html, body {{ width: 100%; height: 100%; background: transparent; overflow: hidden; }}
  body {{
    color: {WB_TEXT_SIGNAL};
    background:
      radial-gradient(circle at 20% 16%, {ontology_accent_wash_strong} 0%, transparent 34%),
      linear-gradient(180deg, {ontology_accent_wash} 0%, transparent 26%);
  }}
  #cy {{ width: 100%; height: 100%; background: transparent; }}
  #boot-msg {{
    position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%);
    color: {WB_TEXT_IRON}; font-family: "JetBrains Mono";
    font-size: 11px; text-align: center; display: none;
    padding: 10px 12px;
    background: {ontology_float_bg_soft};
    border: 1px solid {WB_LINE_LIGHT};
  }}
  #tooltip {{
    position: fixed;
    background: {ontology_float_bg_soft};
    color: {WB_TEXT_SIGNAL};
    font-family: "JetBrains Mono";
    font-size: 10px;
    padding: 6px 10px;
    border: 1px solid {WB_LINE_LIGHT};
    border-left: 2px solid {WB_ACCENT_ACTIVE};
    pointer-events: none;
    display: none;
    max-width: 280px;
    z-index: 100;
    white-space: pre;
    line-height: 1.5;
  }}

  /* ---- detail sidebar ---- */
  #detail-panel {{
    position: fixed; top: 0; right: 0; bottom: 0;
    width: 284px;
    background: {ontology_float_bg};
    border-left: 1px solid {WB_LINE_MED};
    display: none;
    flex-direction: column;
    z-index: 200;
    font-family: "JetBrains Mono";
  }}
  #detail-hdr {{
    padding: 10px 12px 8px;
    border-bottom: 1px solid {WB_LINE_LIGHT};
    display: flex; align-items: baseline; gap: 8px;
    flex-wrap: nowrap;
  }}
  #detail-type-badge {{
    font-size: 9px; font-weight: 700;
    color: {WB_ACCENT_ACTIVE}; letter-spacing: 0.05em; flex-shrink: 0;
  }}
  #detail-title {{
    font-size: 11px; color: {WB_TEXT_BRIGHT};
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1;
    min-width: 0;
  }}
  #detail-close-btn {{
    background: none; border: none; color: {WB_TEXT_IRON};
    cursor: pointer; font-size: 16px; line-height: 1; flex-shrink: 0;
    padding: 0 2px; font-family: sans-serif;
  }}
  #detail-close-btn:hover {{ color: {WB_TEXT_SIGNAL}; }}
  #detail-meta {{
    flex: 1; overflow-y: auto; padding: 8px 12px;
    scrollbar-width: thin; scrollbar-color: {WB_LINE_MED} transparent;
  }}
  .meta-section {{
    font-size: 9px; font-weight: 700; color: {WB_TEXT_IRON};
    letter-spacing: 0.06em; margin: 10px 0 5px;
  }}
  .meta-section:first-child {{ margin-top: 2px; }}
  .meta-row {{
    display: flex; gap: 6px; margin-bottom: 4px; font-size: 10px; line-height: 1.4;
  }}
  .meta-key {{
    color: {WB_TEXT_IRON}; flex-shrink: 0; width: 100px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .meta-val {{
    color: {WB_TEXT_SIGNAL}; word-break: break-all; min-width: 0;
  }}
  #detail-footer {{
    padding: 8px 12px;
    border-top: 1px solid {WB_LINE_LIGHT};
    display: flex; gap: 6px;
  }}
  .detail-btn {{
    flex: 1; font-family: "JetBrains Mono";
    font-size: 9px; font-weight: 600; padding: 5px 6px;
    border: 1px solid {WB_LINE_MED}; background: transparent;
    color: {WB_TEXT_IRON}; cursor: pointer; letter-spacing: 0.03em;
  }}
  .detail-btn:hover {{ color: {WB_TEXT_SIGNAL}; border-color: {WB_ACCENT_ACTIVE}; }}
  .detail-btn.primary {{
    border-color: {WB_ACCENT_ACTIVE}; color: {WB_ACCENT_ACTIVE};
  }}
  .detail-btn.primary:hover {{
    background: {WB_ACCENT_ACTIVE}; color: {WB_BG_VOID};
  }}

  /* ---- right-click context menu ---- */
  #ctx-menu {{
    position: fixed; display: none;
    background: {ontology_float_bg_soft}; border: 1px solid {WB_LINE_MED};
    min-width: 160px; z-index: 300;
    font-family: "JetBrains Mono"; font-size: 10px;
  }}
  .ctx-item {{
    padding: 7px 12px; cursor: pointer; color: {WB_TEXT_SIGNAL};
    border-bottom: 1px solid {WB_LINE_LIGHT};
  }}
  .ctx-item:last-child {{ border-bottom: none; }}
  .ctx-item:hover {{ background: {WB_LINE_MED}; color: {WB_TEXT_BRIGHT}; }}

  /* ---- edge legend ---- */
  #edge-legend {{
    position: fixed; bottom: 12px; left: 12px;
    background: {ontology_float_bg_soft}; border: 1px solid {WB_LINE_LIGHT};
    padding: 8px 12px; display: none; z-index: 150;
    font-family: "JetBrains Mono"; font-size: 9px;
    min-width: 168px;
  }}
  .legend-title {{
    color: {WB_TEXT_IRON}; font-size: 9px; letter-spacing: .06em;
    margin-bottom: 7px; font-weight: 700;
  }}
  .legend-row {{
    display: flex; align-items: center; gap: 8px; margin-bottom: 5px;
  }}
  .legend-row:last-child {{ margin-bottom: 0; }}
  .legend-line {{
    width: 24px; height: 2px; flex-shrink: 0; display: inline-block;
  }}
  .legend-lbl {{ color: {WB_TEXT_IRON}; letter-spacing: 0.04em; }}

  /* ---- detail panel: actions / quicknav sections ---- */
  #detail-actions-section,
  #detail-quicknav-section {{
    padding: 8px 12px 10px;
    border-top: 1px solid {WB_LINE_LIGHT};
  }}
  .actions-title {{
    font-size: 9px; font-weight: 700; color: {WB_TEXT_IRON};
    letter-spacing: 0.06em; margin-bottom: 6px;
  }}
  .action-row {{
    display: flex; gap: 6px; margin-bottom: 5px;
  }}
  .action-row:last-child {{ margin-bottom: 0; }}
  .action-btn {{
    flex: 1; font-family: "JetBrains Mono";
    font-size: 9px; font-weight: 600; padding: 7px 4px;
    border: 1px solid {WB_LINE_MED}; background: transparent;
    color: {WB_TEXT_SIGNAL}; cursor: pointer;
    letter-spacing: 0.04em; text-align: center; white-space: nowrap;
    transition: background 0.12s, color 0.12s, border-color 0.12s;
  }}
  .action-btn:hover {{
    border-color: {WB_ACCENT_ACTIVE}; color: {WB_ACCENT_ACTIVE};
  }}
  .action-btn.primary {{
    border-color: {WB_ACCENT_ACTIVE}; color: {WB_ACCENT_ACTIVE};
  }}
  .action-btn.primary:hover {{
    background: {WB_ACCENT_ACTIVE}; color: {WB_BG_VOID};
  }}
  .action-btn.danger {{
    border-color: {WB_ACCENT_ALERT}; color: {WB_ACCENT_ALERT};
  }}
  .action-btn.danger:hover {{
    background: {WB_ACCENT_ALERT}; color: {WB_BG_VOID}; border-color: {WB_ACCENT_ALERT};
  }}
  .action-btn:disabled {{
    opacity: 0.35; cursor: not-allowed; background: transparent !important;
    color: {WB_TEXT_IRON} !important; border-color: {WB_LINE_LIGHT} !important;
  }}

  /* ---- toast notification ---- */
  #toast {{
    position: fixed; bottom: 14px; right: 14px;
    background: {ontology_float_bg_soft}; color: {WB_TEXT_BRIGHT};
    padding: 9px 14px; font-family: "JetBrains Mono";
    font-size: 10px; line-height: 1.4;
    border-left: 3px solid {WB_ACCENT_ACTIVE};
    z-index: 1000; max-width: 340px; min-width: 200px;
    display: none; box-shadow: 0 6px 24px rgba(0,0,0,0.4);
  }}
  #toast.error {{ border-left-color: {WB_ACCENT_ALERT}; }}
  #toast.success {{ border-left-color: {WB_ACCENT_SELECT}; }}
  @keyframes toastSlide {{
    from {{ transform: translateY(16px); opacity: 0; }}
    to   {{ transform: translateY(0);     opacity: 1; }}
  }}

  /* ---- confirmation overlay ---- */
  #confirm-overlay {{
    position: fixed; inset: 0; background: {ontology_scrim_bg};
    display: none; z-index: 500;
    align-items: center; justify-content: center;
  }}
  #confirm-box {{
    background: {ontology_float_bg}; border: 1px solid {WB_LINE_MED};
    padding: 18px 22px; max-width: 380px; min-width: 280px;
    font-family: "JetBrains Mono";
  }}
  #confirm-title {{
    font-size: 9px; font-weight: 700; letter-spacing: 0.07em;
    color: {WB_ACCENT_ACTIVE}; margin-bottom: 9px;
  }}
  #confirm-msg {{
    color: {WB_TEXT_BRIGHT}; font-size: 11px; line-height: 1.55;
    margin-bottom: 16px;
  }}
  #confirm-buttons {{ display: flex; gap: 8px; justify-content: flex-end; }}
  .confirm-btn {{
    font-family: "JetBrains Mono"; font-size: 10px;
    padding: 6px 14px; cursor: pointer; border: 1px solid {WB_LINE_MED};
    background: transparent; color: {WB_TEXT_SIGNAL}; font-weight: 600;
    letter-spacing: 0.03em;
  }}
  .confirm-btn:hover {{ color: {WB_TEXT_BRIGHT}; }}
  .confirm-btn.primary {{ border-color: {WB_ACCENT_ACTIVE}; color: {WB_ACCENT_ACTIVE}; }}
  .confirm-btn.primary:hover {{ background: {WB_ACCENT_ACTIVE}; color: {WB_BG_VOID}; }}
  .confirm-btn.danger {{ border-color: {WB_ACCENT_ALERT}; color: {WB_ACCENT_ALERT}; }}
  .confirm-btn.danger:hover {{ background: {WB_ACCENT_ALERT}; color: white; }}
</style>
</head>
<body>
<div id="cy"></div>
<div id="boot-msg"></div>
<div id="tooltip"></div>

<div id="detail-panel">
  <div id="detail-hdr">
    <div id="detail-type-badge">[TYPE]</div>
    <div id="detail-title">—</div>
    <button id="detail-close-btn" onclick="hideDetailPanel()">&#x2715;</button>
  </div>
  <div id="detail-meta"></div>
  <div id="detail-quicknav-section" style="display:none">
    <div class="actions-title">QUICK NAV</div>
    <div id="detail-quicknav-container"></div>
  </div>
  <div id="detail-actions-section" style="display:none">
    <div class="actions-title">ACTIONS</div>
    <div id="detail-actions-container"></div>
  </div>
  <div id="detail-footer">
    <button class="detail-btn primary" onclick="navigateFromPanel()">[NAVIGATE &#x2192;]</button>
    <button class="detail-btn" onclick="copyPanelId()">[COPY ID]</button>
  </div>
</div>

<div id="toast"></div>

<div id="confirm-overlay">
  <div id="confirm-box">
    <div id="confirm-title">CONFIRM</div>
    <div id="confirm-msg">—</div>
    <div id="confirm-buttons">
      <button class="confirm-btn" onclick="_confirmCancel()">[CANCEL]</button>
      <button id="confirm-ok-btn" class="confirm-btn primary" onclick="_confirmOk()">[CONFIRM]</button>
    </div>
  </div>
</div>

<div id="ctx-menu">
  <div class="ctx-item" onclick="ctxInspect()">[INSPECT]</div>
  <div class="ctx-item" onclick="ctxNavigate()">[NAVIGATE]</div>
  <div class="ctx-item" onclick="ctxCopyId()">[COPY ID]</div>
</div>

<div id="edge-legend"></div>

<script>
function _bootShowErr(msg) {{
  var m = document.getElementById('boot-msg');
  if (m) {{ m.style.display = 'block'; m.innerHTML = msg; }}
  console.error('[ONTOLOGY_BOOT] ' + msg);
}}
(function() {{
  var boot = document.getElementById('boot-msg');
  if (boot) {{ boot.style.display = 'block'; boot.textContent = '[ONTOLOGY] loading cytoscape.js...'; }}
  function loadScript(src, onOk, onErr) {{
    console.log('[ONTOLOGY_BOOT] fetching ' + src);
    var s = document.createElement('script');
    s.src = src;
    s.onload = onOk;
    s.onerror = onErr;
    document.head.appendChild(s);
  }}
  var primary = {json.dumps(cytoscape_cdn)};
  var fallback = {json.dumps(fallback_src)};
  loadScript(primary, function() {{
    console.log('[ONTOLOGY_BOOT] primary loaded');
    window._cytoscapeReady = true;
    try {{ initGraph(); }} catch (e) {{ _bootShowErr('initGraph error: ' + e.message); }}
  }}, function() {{
    console.warn('[ONTOLOGY_BOOT] primary failed, trying fallback');
    if (fallback) {{
      loadScript(fallback, function() {{
        console.log('[ONTOLOGY_BOOT] fallback loaded');
        window._cytoscapeReady = true;
        try {{ initGraph(); }} catch (e) {{ _bootShowErr('initGraph error: ' + e.message); }}
      }}, function() {{
        _bootShowErr('[ONTOLOGY] cytoscape.js failed to load.<br>Check network access.');
      }});
    }} else {{
      _bootShowErr('[ONTOLOGY] cytoscape.js failed (no fallback configured).');
    }}
  }});
}})();


function initGraph() {{
  if (typeof cytoscape === 'undefined') {{
    document.getElementById('boot-msg').style.display = 'block';
    document.getElementById('boot-msg').textContent = '[ONTOLOGY] cytoscape not defined after load.';
    return;
  }}

  const rawNodes = {nodes_json};
  const rawEdges = {edges_json};
  const NODE_COLORS = {node_colors_json};
  const EDGE_COLORS = {edge_colors_json};
  const ACCENT      = "{WB_ACCENT_ACTIVE}";
  const SELECT_COLOR = "{WB_ACCENT_SELECT}";
  const LINE        = "{WB_LINE_LIGHT}";
  const TEXT        = "{WB_TEXT_SIGNAL}";
  const IRON        = "{WB_TEXT_IRON}";
  const BASE_URL    = "{base_url}";
  const DB_COLOR    = "{WB_NODE_DATABASE}";

  let activeTypes = null;
  let searchTerm  = "";

  const DB_NODE_ID = 'database:__origin__';

  const elements = [];
  const nodeIds  = new Set();

  // Central database origin node — always at position (0, 0)
  nodeIds.add(DB_NODE_ID);
  elements.push({{
    data: {{ id: DB_NODE_ID, label: 'DATABASE', type: 'database', color: DB_COLOR }},
    position: {{ x: 0, y: 0 }},
    classes: 'db-origin',
  }});

  const CLUSTER_TITLES = {{
    'scenario':         'SCENARIOS',
    'backbone':         'BACKBONES',
    'cell':             'CELLS',
    'dataset':          'DATASETS',
    'dataset_snapshot': 'DATA SNAPS',
    'model_version':    'MODELS',
    'model_snapshot':   'MODEL SNAPS',
    'job':              'JOBS',
    'lineage':          'LINEAGES',
    'range':            'RANGES',
    'catalog_asset':    'CATALOG',
    'prov_activity':    'PROV ACT',
    'prov_agent':       'PROV AGENTS',
    'prov_entity':      'PROV ENT',
  }};

  function truncate(s, n) {{
    if (!s) return '';
    s = String(s);
    return s.length <= n ? s : s.slice(0, n - 1) + '…';
  }}

  function edgeColor(etype) {{
    return EDGE_COLORS[etype] || LINE;
  }}

  // --- Degree centrality: scale node size by connection count ---------------
  let nodeDegree = {{}};
  let maxDegree = 1;
  function _recomputeDegree(edges) {{
    nodeDegree = {{}};
    (edges || []).forEach(e => {{
      nodeDegree[e.source] = (nodeDegree[e.source] || 0) + 1;
      nodeDegree[e.target] = (nodeDegree[e.target] || 0) + 1;
    }});
    maxDegree = Object.values(nodeDegree).reduce((m, v) => Math.max(m, v), 1);
  }}
  _recomputeDegree(rawEdges);
  const MIN_NODE_PX = 210;
  const MAX_NODE_PX = 520;
  function nodeSize(nid) {{
    const d = nodeDegree[nid] || 0;
    return MIN_NODE_PX + Math.round((d / maxDegree) * (MAX_NODE_PX - MIN_NODE_PX));
  }}

  // --- Phyllotaxis constants -------------------------------------------------
  const GOLDEN_ANGLE  = Math.PI * (3 - Math.sqrt(5));
  const SEED_SPACING  = 170;
  const NODE_VISUAL_R = 110;
  const SUB_PADDING   = 220;
  const TOP_PADDING   = 560;

  function phyllotaxisRadius(n) {{
    if (n <= 0) return 40;
    return SEED_SPACING * Math.sqrt(n) + NODE_VISUAL_R * 2;
  }}

  function superClusterRadius(subRadii) {{
    if (subRadii.length === 0) return 200;
    if (subRadii.length === 1) return subRadii[0] + SUB_PADDING * 2;
    const maxSub = Math.max.apply(null, subRadii);
    const subRingR = (maxSub + SUB_PADDING) / Math.sin(Math.PI / Math.max(3, subRadii.length));
    return subRingR + maxSub + SUB_PADDING;
  }}

  // --- Group nodes by scenario -----------------------------------------------
  const scenarioEntities = {{}};
  const scenarioSubNodes = {{}};
  const globalByType     = {{}};

  rawNodes.forEach(n => {{
    if (n.type === 'scenario') {{
      const sn = n.id.substring('scenario:'.length);
      scenarioEntities[sn] = n;
      scenarioSubNodes[sn] = scenarioSubNodes[sn] || {{}};
      return;
    }}
    const meta = n.meta || {{}};
    const scen = meta.scenario ? String(meta.scenario).trim() : '';
    if (scen && scenarioSubNodes[scen] !== undefined) {{
      scenarioSubNodes[scen][n.type] = scenarioSubNodes[scen][n.type] || [];
      scenarioSubNodes[scen][n.type].push(n);
    }} else if (scen) {{
      scenarioSubNodes[scen] = scenarioSubNodes[scen] || {{}};
      scenarioSubNodes[scen][n.type] = scenarioSubNodes[scen][n.type] || [];
      scenarioSubNodes[scen][n.type].push(n);
    }} else {{
      globalByType[n.type] = globalByType[n.type] || [];
      globalByType[n.type].push(n);
    }}
  }});

  const SUB_TYPE_ORDER = ['backbone','cell','dataset_snapshot','model_version','job'];
  function orderSubTypes(types) {{
    return types.slice().sort((a, b) => {{
      const ai = SUB_TYPE_ORDER.indexOf(a), bi = SUB_TYPE_ORDER.indexOf(b);
      return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi);
    }});
  }}

  const scenarioNames = Object.keys(scenarioSubNodes).sort();
  const scenarioPlan  = {{}};
  scenarioNames.forEach(s => {{
    const present  = orderSubTypes(Object.keys(scenarioSubNodes[s]));
    const subRadii = present.map(t => phyllotaxisRadius(scenarioSubNodes[s][t].length));
    scenarioPlan[s] = {{ subOrder: present, subRadii, outerRadius: superClusterRadius(subRadii) }};
  }});

  const globalTypeOrder = [
    'lineage','range','catalog_asset','model_snapshot','dataset',
    'prov_activity','prov_agent','prov_entity',
  ].filter(t => globalByType[t] && globalByType[t].length > 0);
  Object.keys(globalByType).forEach(t => {{
    if (globalTypeOrder.indexOf(t) < 0) globalTypeOrder.push(t);
  }});

  // --- Top-level ring --------------------------------------------------------
  const topItems = [];
  scenarioNames.forEach(s =>
    topItems.push({{ kind: 'scenario', key: s, radius: scenarioPlan[s].outerRadius }}));
  globalTypeOrder.forEach(t =>
    topItems.push({{ kind: 'global', key: t, radius: phyllotaxisRadius(globalByType[t].length) }}));

  const topCount = topItems.length || 1;
  const topMaxR  = topItems.reduce((m, i) => Math.max(m, i.radius), 0);
  let topRingR = topCount === 1 ? 0
    : (topMaxR + TOP_PADDING) / Math.sin(Math.PI / Math.max(3, topCount));
  topRingR = Math.max(topRingR, 560);

  const viewportEl = document.getElementById('cy');
  const viewportW = Math.max(1, viewportEl ? viewportEl.clientWidth : window.innerWidth || 1);
  const viewportH = Math.max(1, viewportEl ? viewportEl.clientHeight : window.innerHeight || 1);
  const viewportMin = Math.max(1, Math.min(viewportW, viewportH));
  const graphRadius = Math.max(560, topRingR + topMaxR + TOP_PADDING);
  const viewScale = Math.max(1, (graphRadius * 2) / viewportMin);
  function scaledMetric(base, maxMult) {{
    return Math.round(base * Math.min(maxMult, viewScale));
  }}

  const BASE_EDGE_WIDTH = scaledMetric(9, 3.2);
  const EMPHASIS_EDGE_WIDTH = scaledMetric(16, 3.0);
  const HIGHLIGHT_EDGE_WIDTH = scaledMetric(18, 3.0);
  const DB_EDGE_WIDTH = scaledMetric(6, 3.4);
  const DB_EDGE_DASH = scaledMetric(32, 3.0);
  const DB_EDGE_GAP = scaledMetric(22, 3.0);
  const STRUCT_EDGE_DASH = scaledMetric(20, 2.8);
  const STRUCT_EDGE_GAP = scaledMetric(14, 2.8);
  const EDGE_LABEL_FONT = scaledMetric(70, 2.4);
  const DB_NODE_SIZE = scaledMetric(480, 3.2);
  const DB_NODE_BORDER = scaledMetric(28, 3.0);
  const DB_NODE_SELECTED_BORDER = scaledMetric(36, 3.0);
  const DB_NODE_FONT = scaledMetric(180, 2.8);
  const DB_NODE_TEXT_PAD = scaledMetric(52, 2.6);

  const topCenters = {{}};
  topItems.forEach((item, i) => {{
    const angle = -Math.PI / 2 + (i / topCount) * Math.PI * 2;
    topCenters[item.kind + ':' + item.key] = {{
      x: topCount === 1 ? 0 : topRingR * Math.cos(angle),
      y: topCount === 1 ? 0 : topRingR * Math.sin(angle),
    }};
  }});

  // --- Emit scenario super-clusters -----------------------------------------
  scenarioNames.forEach(s => {{
    const plan    = scenarioPlan[s];
    const center  = topCenters['scenario:' + s];
    const scId    = 'cluster:scenario:' + s;

    nodeIds.add(scId);
    elements.push({{
      data: {{ id: scId, label: s, type: 'scenario_group', color: NODE_COLORS['scenario'] || IRON }},
      classes: 'cluster scenario_group',
    }});

    const subCount = plan.subOrder.length;
    const maxSubR  = Math.max.apply(null, plan.subRadii.concat([0]));
    let subRingR = subCount > 1
      ? (maxSubR + SUB_PADDING) / Math.sin(Math.PI / Math.max(3, subCount)) : 0;

    plan.subOrder.forEach((t, i) => {{
      const angle  = -Math.PI / 2 + (i / Math.max(1, subCount)) * Math.PI * 2;
      const subCx  = center.x + (subCount > 1 ? subRingR * Math.cos(angle) : 0);
      const subCy  = center.y + (subCount > 1 ? subRingR * Math.sin(angle) : 0);
      const subId  = scId + ':' + t;

      nodeIds.add(subId);
      elements.push({{
        data: {{
          id: subId, label: CLUSTER_TITLES[t] || t.toUpperCase(),
          type: 'cluster', color: NODE_COLORS[t] || IRON, parent: scId,
        }},
        classes: 'cluster subcluster',
      }});

      scenarioSubNodes[s][t].forEach((n, j) => {{
        nodeIds.add(n.id);
        const r = SEED_SPACING * Math.sqrt(j + 0.5);
        const theta = j * GOLDEN_ANGLE;
        elements.push({{
          data: {{
            id: n.id, label: truncate(n.label, 20), fullLabel: n.label,
            type: n.type, meta: n.meta || {{}},
            color: NODE_COLORS[n.type] || IRON, parent: subId,
            size: nodeSize(n.id),
          }},
          position: {{ x: subCx + r * Math.cos(theta), y: subCy + r * Math.sin(theta) }},
        }});
      }});
    }});
  }});

  // --- Emit global cubbies --------------------------------------------------
  globalTypeOrder.forEach(t => {{
    const center    = topCenters['global:' + t];
    const clusterId = 'cluster:' + t;

    nodeIds.add(clusterId);
    elements.push({{
      data: {{ id: clusterId, label: CLUSTER_TITLES[t] || t.toUpperCase(), type: 'cluster', color: NODE_COLORS[t] || IRON }},
      classes: 'cluster',
    }});

    globalByType[t].forEach((n, i) => {{
      nodeIds.add(n.id);
      const r = SEED_SPACING * Math.sqrt(i + 0.5);
      const theta = i * GOLDEN_ANGLE;
      elements.push({{
        data: {{
          id: n.id, label: truncate(n.label, 20), fullLabel: n.label,
          type: n.type, meta: n.meta || {{}},
          color: NODE_COLORS[n.type] || IRON, parent: clusterId,
          size: nodeSize(n.id),
        }},
        position: {{ x: center.x + r * Math.cos(theta), y: center.y + r * Math.sin(theta) }},
      }});
    }});
  }});

  // --- Edges ----------------------------------------------------------------
  let droppedEdges = 0;
  function resolveEdgeEndpoint(id) {{
    if (nodeIds.has(id)) return id;
    if (String(id || '').startsWith('scenario:')) {{
      const scen = String(id).substring('scenario:'.length);
      const clusterId = 'cluster:scenario:' + scen;
      if (nodeIds.has(clusterId)) return clusterId;
    }}
    return '';
  }}
  rawEdges.forEach(e => {{
    const edgeSource = resolveEdgeEndpoint(e.source);
    const edgeTarget = resolveEdgeEndpoint(e.target);
    if (!edgeSource || !edgeTarget) {{ droppedEdges++; return; }}
    elements.push({{
      data: {{
        id: e.source + '__' + e.type + '__' + e.target,
        source: edgeSource, target: edgeTarget,
        originalSource: e.source, originalTarget: e.target,
        edgeType: e.type, lineColor: edgeColor(e.type),
      }}
    }});
  }});
  if (droppedEdges > 0)
    console.warn('[ONTOLOGY] dropped ' + droppedEdges + ' orphan edge(s)');

  // --- DB origin spokes — radiate from the central database to every top-level cluster
  const _topClusterIds = [];
  scenarioNames.forEach(s => _topClusterIds.push('cluster:scenario:' + s));
  globalTypeOrder.forEach(t => _topClusterIds.push('cluster:' + t));
  _topClusterIds.forEach(cid => {{
    if (nodeIds.has(cid)) {{
      elements.push({{
        data: {{
          id: DB_NODE_ID + '__stores_in__' + cid,
          source: DB_NODE_ID, target: cid,
          edgeType: 'stores_in', lineColor: DB_COLOR,
        }}
      }});
    }}
  }});

  const cy = cytoscape({{
    container: document.getElementById('cy'),
    elements,
    style: [
      {{
        selector: 'node',
        style: {{
          'background-color':       'data(color)',
          'label':                  'data(label)',
          'color':                  TEXT,
          'font-family':            'JetBrains Mono, monospace',
          'font-size':              '110px',
          'text-valign':            'bottom',
          'text-halign':            'center',
          'text-margin-y':          '70px',
          'width':                  '210px',
          'height':                 '210px',
          'border-width':           '10px',
          'border-color':           'data(color)',
          'border-opacity':         0.65,
          'text-background-color':  '{WB_BG_VOID}',
          'text-background-opacity': 0.48,
          'text-background-padding': '22px',
          'min-zoomed-font-size':   7,
          'text-wrap':              'none',
        }}
      }},
      {{
        selector: 'node[size]',
        style: {{
          'width':                  'data(size)',
          'height':                 'data(size)',
        }}
      }},
      {{
        selector: 'node.cluster',
        style: {{
          'shape':                   'ellipse',
          'background-color':        'data(color)',
          'background-opacity':      0.12,
          'border-width':            '18px',
          'border-color':            'data(color)',
          'border-opacity':          0.80,
          'border-style':            'solid',
          'padding':                 '480px',
          'text-valign':             'top',
          'text-halign':             'center',
          'text-margin-y':           '-80px',
          'font-size':               '130px',
          'font-weight':             'bold',
          'color':                   'data(color)',
          'text-background-color':   '{WB_BG_VOID}',
          'text-background-opacity': 0.52,
          'text-background-padding': '38px',
          'z-index':                 2,
        }}
      }},
      {{
        selector: 'node.scenario_group',
        style: {{
          'shape':                    'ellipse',
          'background-color':         'data(color)',
          'background-opacity':       0.07,
          'border-width':             '26px',
          'border-color':             'data(color)',
          'border-opacity':           0.95,
          'border-style':             'solid',
          'padding':                  '1000px',
          'text-valign':              'top',
          'text-halign':              'center',
          'text-margin-y':            '-200px',
          'font-size':                '200px',
          'font-weight':              'bold',
          'color':                    'data(color)',
          'text-background-color':    '{WB_BG_VOID}',
          'text-background-opacity':  0.62,
          'text-background-padding':  '80px',
          'text-background-shape':    'roundrectangle',
          'z-index':                  1,
        }}
      }},
      {{
        selector: 'node:selected',
        style: {{ 'border-width': '8px', 'border-color': SELECT_COLOR, 'background-color': SELECT_COLOR }}
      }},
      {{
        selector: 'node.cluster:selected',
        style: {{ 'border-color': SELECT_COLOR, 'background-color': '{WB_BG_GRAPHITE}' }}
      }},
      {{
        selector: 'node.faded',
        style: {{ opacity: 0.15 }}
      }},
      // --- edges: base style uses per-edge lineColor data ---
      {{
        selector: 'edge',
        style: {{
          'width':               BASE_EDGE_WIDTH,
          'line-color':          'data(lineColor)',
          'target-arrow-color':  'data(lineColor)',
          'target-arrow-shape':  'triangle',
          'arrow-scale':         4.0,
          'curve-style':         'bezier',
          'opacity':             0.5,
        }}
      }},
      // Dashed = structural / ownership (lower semantic weight)
      {{
        selector: 'edge[edgeType = "belongs_to"], edge[edgeType = "branched_from"]',
        style: {{ 'line-style': 'dashed', 'line-dash-pattern': [STRUCT_EDGE_DASH, STRUCT_EDGE_GAP], 'opacity': 0.28 }}
      }},
      {{
        selector: 'edge[edgeType = "prov_generates"], edge[edgeType = "prov_used"], edge[edgeType = "prov_informed_by"], edge[edgeType = "prov_associated"], edge[edgeType = "had_member"], edge[edgeType = "specialization_of"], edge[edgeType = "prov_invalidated"], edge[edgeType = "prov_attributed"]',
        style: {{ 'line-style': 'dotted', 'line-dash-pattern': [STRUCT_EDGE_DASH, STRUCT_EDGE_GAP], 'opacity': 0.62, 'width': BASE_EDGE_WIDTH * 0.85 }}
      }},
      // Thicker = high-importance provenance edges
      {{
        selector: 'edge[edgeType = "governed_by"]',
        style: {{ 'width': EMPHASIS_EDGE_WIDTH }}
      }},
      // Training input: scenario -> dataset used for fitting.
      {{
        selector: 'edge[edgeType = "trains_on"]',
        style: {{
          'width': EMPHASIS_EDGE_WIDTH * 0.92,
          'opacity': 0.80,
          'line-style': 'solid',
          'target-arrow-shape': 'triangle-tee',
          'arrow-scale': 5.2,
        }}
      }},
      // Training lineage: completed train/update job -> spawned registered model.
      {{
        selector: 'edge[edgeType = "produces"]',
        style: {{
          'width': EMPHASIS_EDGE_WIDTH * 1.15,
          'opacity': 0.92,
          'line-style': 'solid',
          'label': 'PRODUCES',
          'font-family': 'JetBrains Mono, monospace',
          'font-size': EDGE_LABEL_FONT,
          'font-weight': 'bold',
          'color': 'data(lineColor)',
          'text-rotation': 'autorotate',
          'text-background-color': '{WB_BG_VOID}',
          'text-background-opacity': 0.74,
          'text-background-padding': scaledMetric(24, 2.2),
        }}
      }},
      {{
        selector: 'edge:selected, edge.highlighted',
        style: {{ 'line-color': ACCENT, 'target-arrow-color': ACCENT, 'opacity': 1.0, 'width': HIGHLIGHT_EDGE_WIDTH }}
      }},
      {{
        selector: 'edge.faded',
        style: {{ opacity: 0.04 }}
      }},
      // --- Path and impact highlight classes
      {{
        selector: 'node.path-source-node',
        style: {{ 'border-width': '24px', 'border-color': SELECT_COLOR, 'border-opacity': 1.0 }}
      }},
      {{
        selector: 'node.path-highlight, edge.path-highlight',
        style: {{ opacity: 1.0 }}
      }},
      {{
        selector: 'edge.path-highlight',
        style: {{ 'line-color': SELECT_COLOR, 'target-arrow-color': SELECT_COLOR, 'width': HIGHLIGHT_EDGE_WIDTH }}
      }},
      {{
        selector: 'node.impact-pivot',
        style: {{ 'border-width': '24px', 'border-color': SELECT_COLOR, 'border-opacity': 1.0, opacity: 1.0 }}
      }},
      {{
        selector: 'node.impact-upstream',
        style: {{ 'border-width': '18px', 'border-color': '{WB_ACCENT_ALERT}', 'border-opacity': 0.9, opacity: 1.0 }}
      }},
      {{
        selector: 'node.impact-downstream',
        style: {{ 'border-width': '18px', 'border-color': '{WB_ACCENT_WARN}', 'border-opacity': 0.9, opacity: 1.0 }}
      }},
      // --- correction_event nodes: diamond shape
      {{
        selector: 'node[type = "correction_event"]',
        style: {{ 'shape': 'diamond', 'width': '180px', 'height': '180px' }}
      }},
      // --- sector/collection nodes: smaller round-rectangle
      {{
        selector: 'node[type = "sector"], node[type = "collection"]',
        style: {{ 'shape': 'round-rectangle', 'width': '160px', 'height': '160px' }}
      }},
      // --- shares_backbone_with / shares_dataset_with: dashed, lower opacity
      {{
        selector: 'edge[edgeType = "shares_backbone_with"], edge[edgeType = "shares_dataset_with"]',
        style: {{ 'line-style': 'dashed', 'line-dash-pattern': [STRUCT_EDGE_DASH, STRUCT_EDGE_GAP], 'opacity': 0.35 }}
      }},
      // --- DB origin node — diamond, saffron gold, always at center
      {{
        selector: 'node.db-origin',
        style: {{
          'shape':                    'diamond',
          'background-color':         DB_COLOR,
          'background-opacity':       0.14,
          'border-width':             DB_NODE_BORDER,
          'border-color':             DB_COLOR,
          'border-opacity':           1.0,
          'width':                    DB_NODE_SIZE,
          'height':                   DB_NODE_SIZE,
          'label':                    'data(label)',
          'font-size':                DB_NODE_FONT,
          'font-weight':              'bold',
          'color':                    DB_COLOR,
          'text-valign':              'center',
          'text-halign':              'center',
          'text-background-color':    '{WB_BG_VOID}',
          'text-background-opacity':  0.82,
          'text-background-padding':  DB_NODE_TEXT_PAD,
          'z-index':                  20,
        }}
      }},
      {{
        selector: 'node.db-origin:selected',
        style: {{ 'border-color': SELECT_COLOR, 'border-width': DB_NODE_SELECTED_BORDER }}
      }},
      // DB spoke edges — dashed gold, no arrowhead
      {{
        selector: 'edge[edgeType = "stores_in"]',
        style: {{
          'line-style':          'dashed',
          'line-dash-pattern':   [DB_EDGE_DASH, DB_EDGE_GAP],
          'target-arrow-shape':  'none',
          'opacity':             0.30,
          'width':               DB_EDGE_WIDTH,
        }}
      }},
    ],
    layout: {{
      name: 'preset',
      fit: true,
      padding: 80,
      stop: function() {{
        console.log('[ONTOLOGY] preset layout done');
        const boot = document.getElementById('boot-msg');
        if (boot) boot.style.display = 'none';
        _buildLegend();
      }},
    }},
    minZoom: 0.02,
    maxZoom: 6,
  }});
  window._cvopsCy = cy;

  // =====================================================================
  // Actions registry — entity-type → list of one-stop workflows
  // =====================================================================
  // Each entry is a workflow that maps to a verified service endpoint.
  // The runner reads .method / .path(eid, meta) / .body(eid, meta).
  // Optional .condition(meta) hides the button when not applicable.
  const ACTIONS = {{
    'scenario': [
      {{
        id: 'train', label: '[TRAIN NOW]', style: 'primary',
        method: 'POST',
        path: (eid) => '/scenarios/' + encodeURIComponent(eid) + '/train',
        body: () => ({{}}),
        confirm: (eid) => 'Start a training run for scenario "' + eid + '"?',
        toast: 'Training run submitted',
        reload: true,
      }},
      {{
        id: 'update', label: '[UPDATE-MODE]',
        method: 'POST',
        path: (eid) => '/scenarios/' + encodeURIComponent(eid) + '/update',
        body: () => ({{}}),
        confirm: (eid) => 'Run an update-mode training job for "' + eid + '"?',
        toast: 'Update-mode job submitted',
        reload: true,
      }},
      {{
        id: 'verify', label: '[VERIFY]',
        method: 'POST',
        path: (eid) => '/scenarios/' + encodeURIComponent(eid) + '/verify',
        body: () => ({{}}),
        toast: 'Verification triggered',
      }},
    ],
    'scenario_group': null,  // resolves to 'scenario' at lookup time
    'job': [
      {{
        id: 'cancel', label: '[CANCEL JOB]', style: 'danger',
        method: 'POST',
        path: (eid) => '/jobs/' + encodeURIComponent(eid) + '/cancel',
        body: () => ({{}}),
        condition: (m) => ['queued','running','accepted','starting'].includes(String(m.state || '').toLowerCase()),
        confirm: () => 'Cancel this job?',
        toast: 'Cancellation requested',
        reload: true,
      }},
      {{
        id: 'retry', label: '[RETRY]', style: 'primary',
        method: 'POST',
        path: (eid) => '/jobs/' + encodeURIComponent(eid) + '/retry',
        body: () => ({{}}),
        condition: (m) => ['error','cancelled','failed','interrupted'].includes(String(m.state || '').toLowerCase()),
        toast: 'Job retry submitted',
        reload: true,
      }},
    ],
    'lineage': [
      {{
        id: 'fork', label: '[FORK]',
        method: 'POST',
        path: (eid) => '/lineages/' + encodeURIComponent(eid) + '/fork',
        body: (eid, m) => ({{ new_name: (m.name || eid) + '-fork' }}),
        confirm: () => 'Fork this lineage into a new branch?',
        toast: 'Lineage forked',
        reload: true,
      }},
      {{
        id: 'archive', label: '[ARCHIVE]',
        method: 'POST',
        path: (eid) => '/lineages/' + encodeURIComponent(eid) + '/state',
        body: () => ({{ state: 'archived' }}),
        condition: (m) => String(m.state || '').toLowerCase() !== 'archived',
        toast: 'Lineage archived',
        reload: true,
      }},
      {{
        id: 'activate', label: '[ACTIVATE]',
        method: 'POST',
        path: (eid) => '/lineages/' + encodeURIComponent(eid) + '/state',
        body: () => ({{ state: 'active' }}),
        condition: (m) => String(m.state || '').toLowerCase() !== 'active',
        toast: 'Lineage activated',
        reload: true,
      }},
      {{
        id: 'delete', label: '[DELETE]', style: 'danger',
        method: 'DELETE',
        path: (eid) => '/lineages/' + encodeURIComponent(eid),
        body: () => null,
        confirm: () => 'Delete this lineage? This cannot be undone.',
        toast: 'Lineage deleted',
        reload: true, closeOnSuccess: true,
      }},
    ],
    'range': [
      {{
        id: 'delete', label: '[DELETE]', style: 'danger',
        method: 'DELETE',
        path: (eid) => '/ranges/' + encodeURIComponent(eid),
        body: () => null,
        confirm: () => 'Delete this range and its evaluations?',
        toast: 'Range deleted',
        reload: true, closeOnSuccess: true,
      }},
    ],
    'model_snapshot': [
      {{
        id: 'delete', label: '[DELETE]', style: 'danger',
        method: 'DELETE',
        path: (eid) => '/snapshots/' + encodeURIComponent(eid),
        body: () => null,
        confirm: () => 'Delete this snapshot? This cannot be undone.',
        toast: 'Snapshot deleted',
        reload: true, closeOnSuccess: true,
      }},
    ],
  }};

  // =====================================================================
  // Quick-nav registry — one-click jumps to other Qt panels, pre-focused.
  // Each entry returns the focus-id segment (or '' to suppress the button).
  // =====================================================================
  const QUICK_NAV = {{
    'scenario': [
      {{ id: 'data',    label: '[DATA SELECT]',  target: 'data_selection',  focus: (eid) => eid }},
      {{ id: 'train',   label: '[TRAIN]',        target: 'training',         focus: (eid) => eid }},
      {{ id: 'jobs',    label: '[JOBS]',         target: 'jobs',             focus: (eid) => eid }},
      {{ id: 'range',   label: '[Range]',   target: 'test_range',       focus: (eid) => eid }},
      {{ id: 'charts',  label: '[CHARTS]',       target: 'charts',           focus: (eid) => eid }},
      {{ id: 'config',  label: '[CONFIG]',       target: 'scenario_config',  focus: (eid) => eid }},
    ],
    'scenario_group': null,  // resolves to 'scenario' at lookup time
    'job': [
      {{ id: 'queue',    label: '[JOBS QUEUE]', target: 'jobs',            focus: (eid) => eid }},
      {{ id: 'results',  label: '[RESULTS]',    target: 'results',          focus: (eid) => eid }},
      {{ id: 'scenario', label: '[SCENARIO]',   target: 'scenario_config',
        focus: (eid, m) => String(m.scenario || '') }},
    ],
    'dataset': [
      {{ id: 'data',  label: '[DATA SELECT]', target: 'data_selection',
        focus: (eid, m) => String(m.scenario || eid) }},
    ],
    'dataset_snapshot': [
      {{ id: 'data',  label: '[DATA SELECT]', target: 'data_selection',
        focus: (eid, m) => String(m.scenario || '') }},
      {{ id: 'train', label: '[TRAIN]',       target: 'training',
        focus: (eid, m) => String(m.scenario || '') }},
    ],
    'model_version': [
      {{ id: 'train', label: '[TRAIN]',      target: 'training',
        focus: (eid, m) => String(m.scenario || '') }},
      {{ id: 'range', label: '[Range]', target: 'test_range',
        focus: (eid, m) => String(m.scenario || '') }},
    ],
    'lineage': [
      {{ id: 'charts', label: '[CHARTS]', target: 'charts', focus: (eid) => eid }},
    ],
    'range': [
      {{ id: 'range', label: '[Range]', target: 'test_range', focus: (eid) => eid }},
    ],
    'database': [
      {{ id: 'db', label: '[DATABASE VIEW]', target: 'database', focus: (eid) => eid }},
    ],
  }};

  function _renderQuickNav(etype, eid, meta) {{
    const section   = document.getElementById('detail-quicknav-section');
    const container = document.getElementById('detail-quicknav-container');
    container.innerHTML = '';

    const lookupType = (etype === 'scenario_group') ? 'scenario' : etype;
    const list = QUICK_NAV[lookupType] || [];
    const m = meta || {{}};

    const visible = list.map(t => ({{
      label: t.label, target: t.target, fid: t.focus(eid, m),
    }})).filter(t => t.fid && t.fid.length > 0);

    if (visible.length === 0) {{
      section.style.display = 'none';
      return;
    }}
    section.style.display = 'block';

    const scenarioHint = String(m.scenario || '');
    let row = null;
    visible.forEach((t, i) => {{
      if (i % 2 === 0) {{
        row = document.createElement('div');
        row.className = 'action-row';
        container.appendChild(row);
      }}
      const btn = document.createElement('button');
      btn.className = 'action-btn';
      btn.textContent = t.label;
      btn.onclick = () => {{
        const url = 'appbridge://goto/' + encodeURIComponent(t.target)
                  + '/' + encodeURIComponent(t.fid)
                  + (scenarioHint ? ('?scenario=' + encodeURIComponent(scenarioHint)) : '');
        window.location.href = url;
      }};
      row.appendChild(btn);
    }});
  }}

  // =====================================================================
  // Toast + confirm helpers
  // =====================================================================
  let _toastTimer = null;
  function showToast(msg, kind) {{
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = kind || 'success';
    t.style.display = 'block';
    t.style.animation = 'toastSlide 0.2s ease-out';
    if (_toastTimer) clearTimeout(_toastTimer);
    _toastTimer = setTimeout(() => {{ t.style.display = 'none'; }}, 3500);
  }}

  let _confirmResolver = null;
  function showConfirm(msg, danger) {{
    document.getElementById('confirm-msg').textContent = msg;
    const okBtn = document.getElementById('confirm-ok-btn');
    okBtn.className = 'confirm-btn ' + (danger ? 'danger' : 'primary');
    document.getElementById('confirm-overlay').style.display = 'flex';
    return new Promise(resolve => {{ _confirmResolver = resolve; }});
  }}

  window._confirmOk = function() {{
    document.getElementById('confirm-overlay').style.display = 'none';
    if (_confirmResolver) {{ _confirmResolver(true); _confirmResolver = null; }}
  }};

  window._confirmCancel = function() {{
    document.getElementById('confirm-overlay').style.display = 'none';
    if (_confirmResolver) {{ _confirmResolver(false); _confirmResolver = null; }}
  }};

  // =====================================================================
  // Action runner
  // =====================================================================
  async function runAction(action, etype, eid, meta) {{
    if (action.confirm) {{
      const msg = (typeof action.confirm === 'function')
        ? action.confirm(eid, meta || {{}})
        : action.confirm;
      const danger = action.style === 'danger';
      const ok = await showConfirm(msg, danger);
      if (!ok) return;
    }}

    const url = BASE_URL.replace(/\\/$/, '') + action.path(eid, meta || {{}});
    const opts = {{
      method: action.method,
      headers: {{ 'Content-Type': 'application/json' }},
    }};
    const bodyData = action.body ? action.body(eid, meta || {{}}) : null;
    if (bodyData !== null && bodyData !== undefined) {{
      opts.body = JSON.stringify(bodyData);
    }}

    try {{
      const resp = await fetch(url, opts);
      if (!resp.ok) {{
        const txt = await resp.text();
        let detail = txt;
        try {{ detail = (JSON.parse(txt) || {{}}).detail || txt; }} catch (e) {{}}
        showToast('[' + resp.status + '] ' + String(detail).slice(0, 160), 'error');
        return;
      }}
      showToast(action.toast || 'Done', 'success');
      if (action.closeOnSuccess) hideDetailPanel();
      if (action.reload) {{
        // Trigger a Qt-side reload via appbridge.
        setTimeout(() => {{ window.location.href = 'appbridge://reload'; }}, 600);
      }}
    }} catch (e) {{
      showToast('[NET] ' + String(e.message || e), 'error');
    }}
  }}

  function _renderActions(etype, eid, meta) {{
    const section   = document.getElementById('detail-actions-section');
    const container = document.getElementById('detail-actions-container');
    container.innerHTML = '';

    // 'scenario_group' clusters use the same actions as scenario entities
    const lookupType = (etype === 'scenario_group') ? 'scenario' : etype;
    const list = ACTIONS[lookupType] || [];

    const visible = list.filter(a => !a.condition || a.condition(meta || {{}}));
    if (visible.length === 0) {{
      section.style.display = 'none';
      return;
    }}
    section.style.display = 'block';

    // Pack buttons two per row.
    let row = null;
    visible.forEach((action, i) => {{
      if (i % 2 === 0) {{
        row = document.createElement('div');
        row.className = 'action-row';
        container.appendChild(row);
      }}
      const btn = document.createElement('button');
      btn.className = 'action-btn ' + (action.style || '');
      btn.textContent = action.label;
      btn.onclick = () => {{
        btn.disabled = true;
        runAction(action, etype, eid, meta).finally(() => {{ btn.disabled = false; }});
      }};
      row.appendChild(btn);
    }});
  }}

  // =====================================================================
  // Detail sidebar
  // =====================================================================
  let _currentNavTarget = null;

  function showDetailPanel(node) {{
    const nid   = node.data('id');
    const ntype = node.data('type');
    const label = node.data('fullLabel') || node.data('label');
    const meta  = node.data('meta') || {{}};

    let actionType, actionId;
    if (ntype === 'scenario_group') {{
      const sname = nid.substring('cluster:scenario:'.length);
      _currentNavTarget = 'appbridge://entity/scenario/' + encodeURIComponent(sname);
      document.getElementById('detail-type-badge').textContent = '[SCENARIO]';
      document.getElementById('detail-title').textContent = sname;
      actionType = 'scenario_group';
      actionId   = sname;
    }} else {{
      const parts = nid.split(':');
      const etype = parts[0];
      const eid   = parts.slice(1).join(':');
      _currentNavTarget = 'appbridge://entity/' + encodeURIComponent(etype) + '/' + encodeURIComponent(eid);
      document.getElementById('detail-type-badge').textContent =
        '[' + etype.toUpperCase().replace(/_/g, ' ') + ']';
      document.getElementById('detail-title').textContent = label;
      actionType = etype;
      actionId   = eid;
    }}

    const metaDiv = document.getElementById('detail-meta');
    metaDiv.innerHTML = '';

    const deg = nodeDegree[nid] || 0;
    const section = document.createElement('div');
    section.className = 'meta-section';
    section.textContent = 'GRAPH';
    metaDiv.appendChild(section);

    function addRow(k, v) {{
      const row = document.createElement('div');
      row.className = 'meta-row';
      row.innerHTML = '<span class="meta-key">' + k + '</span>'
                    + '<span class="meta-val">' + String(v).slice(0, 140) + '</span>';
      metaDiv.appendChild(row);
    }}

    addRow('connections', deg);
    addRow('type', ntype);

    // Impact tool button — universal across all entity types
    if (actionType !== 'scenario_group') {{
      const toolSection = document.createElement('div');
      toolSection.className = 'meta-section';
      toolSection.textContent = 'TOOLS';
      metaDiv.appendChild(toolSection);
      const impactBtn = document.createElement('button');
      impactBtn.className = 'action-btn';
      impactBtn.textContent = '[SHOW IMPACT]';
      impactBtn.style.cssText = 'width:100%;margin:2px 0;';
      impactBtn.onclick = function() {{
        const nodeId = actionType + ':' + actionId;
        window.location.href = 'appbridge://impact/' + encodeURIComponent(nodeId);
      }};
      metaDiv.appendChild(impactBtn);
    }}

    const metaKeys = Object.keys(meta).filter(k => meta[k] !== null && meta[k] !== undefined && meta[k] !== '' && k !== 'thumbnail_b64');
    if (metaKeys.length > 0) {{
      const s2 = document.createElement('div');
      s2.className = 'meta-section';
      s2.textContent = 'FIELDS';
      metaDiv.appendChild(s2);
      metaKeys.forEach(k => {{
        let v = meta[k];
        if (Array.isArray(v))       v = v.join(', ');
        else if (typeof v === 'object') v = JSON.stringify(v).slice(0, 120);
        addRow(k, v);
      }});
    }}

    // Identity thumbnail display
    if (ntype === 'identity' && meta.thumbnail_b64) {{
      const s3 = document.createElement('div');
      s3.className = 'meta-section';
      s3.textContent = 'FACE';
      metaDiv.appendChild(s3);
      const img = document.createElement('img');
      img.src = 'data:image/png;base64,' + meta.thumbnail_b64;
      img.style.cssText = 'width:80px;height:80px;object-fit:cover;border-radius:4px;margin:4px 0;display:block;';
      metaDiv.appendChild(img);
    }}

    _renderQuickNav(actionType, actionId, meta);
    _renderActions(actionType, actionId, meta);

    document.getElementById('detail-panel').style.display = 'flex';
    window.location.href = 'appbridge://inspect/' + encodeURIComponent(actionType)
                       + '/' + encodeURIComponent(actionId);
  }}

  function hideDetailPanel() {{
    document.getElementById('detail-panel').style.display = 'none';
    _currentNavTarget = null;
  }}

  function navigateFromPanel() {{
    if (_currentNavTarget) window.location.href = _currentNavTarget;
  }}

  function copyPanelId() {{
    const title = document.getElementById('detail-title').textContent;
    if (navigator.clipboard) navigator.clipboard.writeText(title);
  }}

  window.hideDetailPanel    = hideDetailPanel;
  window.navigateFromPanel  = navigateFromPanel;
  window.copyPanelId        = copyPanelId;

  // =====================================================================
  // Right-click context menu
  // =====================================================================
  let _ctxNode = null;

  cy.on('cxttap', 'node', function(evt) {{
    const node = evt.target;
    if (node.hasClass('cluster')) return;
    _ctxNode = node;
    const menu = document.getElementById('ctx-menu');
    menu.style.left = evt.originalEvent.clientX + 'px';
    menu.style.top  = evt.originalEvent.clientY + 'px';
    menu.style.display = 'block';
  }});

  document.addEventListener('click', function() {{
    document.getElementById('ctx-menu').style.display = 'none';
  }});

  window.ctxInspect = function() {{
    if (_ctxNode) showDetailPanel(_ctxNode);
  }};

  window.ctxNavigate = function() {{
    if (!_ctxNode) return;
    const nid   = _ctxNode.data('id');
    const parts = nid.split(':');
    const etype = parts[0];
    const eid   = parts.slice(1).join(':');
    window.location.href = 'appbridge://entity/' + encodeURIComponent(etype) + '/' + encodeURIComponent(eid);
  }};

  window.ctxCopyId = function() {{
    if (_ctxNode && navigator.clipboard)
      navigator.clipboard.writeText(_ctxNode.data('id'));
  }};

  // =====================================================================
  // Node tap — single click opens detail panel
  // =====================================================================
  cy.on('tap', 'node', function(evt) {{
    const node = evt.target;
    if (node.hasClass('cluster')) {{
      cy.animate({{ fit: {{ eles: node.descendants(), padding: 30 }}, duration: 250 }});
      return;
    }}
    if (_pathMode) {{
      const nid = node.data('id');
      if (!_pathSource) {{
        _pathSource = nid;
        cy.nodes().removeClass('path-source-node');
        node.addClass('path-source-node');
        showToast('[FIND PATH] Click destination node (or [CLEAR] to cancel)', 'info');
      }} else if (nid !== _pathSource) {{
        window.location.href = 'appbridge://path/' + encodeURIComponent(_pathSource) + '/' + encodeURIComponent(nid);
        cy.nodes().removeClass('path-source-node');
        _pathMode = false;
        _pathSource = null;
      }}
      return;
    }}
    if (node.hasClass('scenario_group')) {{
      showDetailPanel(node);
      return;
    }}
    showDetailPanel(node);
  }});

  cy.on('tap', function(evt) {{
    if (evt.target === cy) hideDetailPanel();
  }});

  // =====================================================================
  // Hover tooltip
  // =====================================================================
  const tooltip = document.getElementById('tooltip');

  cy.on('mouseover', 'node', function(evt) {{
    const n = evt.target;
    if (n.hasClass('db-origin')) {{
      const entityCount = cy.nodes().not('.cluster').not('.db-origin').length;
      const edgeCount   = cy.edges().filter(e => e.data('edgeType') !== 'stores_in').length;
      tooltip.textContent = '[DATABASE] origin — ' + entityCount + ' entities  ' + edgeCount + ' relations';
      tooltip.style.display = 'block';
      return;
    }}
    if (n.hasClass('scenario_group')) {{
      const kids = n.descendants().filter(d => !d.hasClass('cluster')).length;
      tooltip.textContent = '[SCENARIO] ' + n.data('label') + '  —  ' + kids + ' entity(ies)';
      tooltip.style.display = 'block';
      return;
    }}
    if (n.hasClass('cluster')) {{
      const kids = n.descendants().length;
      tooltip.textContent = '[' + n.data('label') + '] ' + kids + ' node(s)';
      tooltip.style.display = 'block';
      return;
    }}
    const meta  = n.data('meta') || {{}};
    const title = n.data('fullLabel') || n.data('label');
    const deg   = nodeDegree[n.data('id')] || 0;
    const lines = ['[' + n.data('type').toUpperCase() + '] ' + title];
    if (deg) lines.push('connections: ' + deg);
    Object.entries(meta).forEach(([k, v]) => {{
      if (v !== null && v !== undefined && v !== '')
        lines.push(k + ': ' + String(v).slice(0, 60));
    }});
    tooltip.textContent = lines.join('\\n');
    tooltip.style.display = 'block';
  }});

  cy.on('mouseout', 'node', function() {{ tooltip.style.display = 'none'; }});
  cy.on('mousemove', function(evt) {{
    tooltip.style.left = (evt.originalEvent.clientX + 14) + 'px';
    tooltip.style.top  = (evt.originalEvent.clientY + 8)  + 'px';
  }});

  // =====================================================================
  // Edge legend
  // =====================================================================
  function _buildLegend() {{
    const legend = document.getElementById('edge-legend');
    legend.innerHTML = '';
    const title = document.createElement('div');
    title.className = 'legend-title';
    title.textContent = 'EDGE TYPES';
    legend.appendChild(title);

    const entries = [
      ['stores_in',     DB_COLOR,                     'STORES IN',     true ],
      ['belongs_to',    EDGE_COLORS['belongs_to'],    'BELONGS TO',    true ],
      ['governed_by',   EDGE_COLORS['governed_by'],   'GOVERNED BY',   false],
      ['produces',      EDGE_COLORS['produces'],       'PRODUCES',      false],
      ['trains_on',     EDGE_COLORS['trains_on'],      'TRAINS ON',     false],
      ['evaluates',     EDGE_COLORS['evaluates'],      'EVALUATES',     false],
      ['derived_from',  EDGE_COLORS['derived_from'],   'DERIVED FROM',  false],
      ['has_head',      EDGE_COLORS['has_head'],       'HAS HEAD',      false],
      ['uses_backbone', EDGE_COLORS['uses_backbone'],  'USES BACKBONE', false],
      ['contains_cell', EDGE_COLORS['contains_cell'],  'CONTAINS CELL', false],
      ['prov_generates', EDGE_COLORS['prov_generates'], 'PROV generates', false],
      ['prov_used',      EDGE_COLORS['prov_used'],      'PROV used',      true],
      ['had_member',     EDGE_COLORS['had_member'],     'PROV hadMember', true],
      ['flagged_in',          EDGE_COLORS['flagged_in'],          'FLAGGED IN',          true],
      ['flagged_by_model',    EDGE_COLORS['flagged_by_model'],    'FLAGGED BY MODEL',    true],
      ['contains_sector',     EDGE_COLORS['contains_sector'],     'CONTAINS SECTOR',     true],
      ['organized_in',        EDGE_COLORS['organized_in'],        'ORGANIZED IN',        true],
      ['catalogued_in',       EDGE_COLORS['catalogued_in'],       'CATALOGUED IN',       false],
      ['shares_backbone_with', EDGE_COLORS['shares_backbone_with'], 'SHARES BACKBONE',   true],
      ['shares_dataset_with',  EDGE_COLORS['shares_dataset_with'], 'SHARES DATASET',     true],
    ];

    entries.forEach(([, color, label, dashed]) => {{
      const row  = document.createElement('div');
      row.className = 'legend-row';
      const line = document.createElement('span');
      line.className = 'legend-line';
      if (dashed) {{
        line.style.cssText = 'border-top:2px dashed ' + color + ';height:0;background:none;';
      }} else {{
        line.style.background = color;
      }}
      const lbl = document.createElement('span');
      lbl.className = 'legend-lbl';
      lbl.textContent = label;
      row.appendChild(line);
      row.appendChild(lbl);
      legend.appendChild(row);
    }});
  }}

  // =====================================================================
  // External API (callable from Qt via runJavaScript)
  // =====================================================================
  window._cvopsReplaceGraph = function(graph, fit) {{
    graph = graph || {{}};
    const newNodes = Array.isArray(graph.nodes) ? graph.nodes : [];
    const newEdges = Array.isArray(graph.edges) ? graph.edges : [];
    const wantedNodeIds = new Set([DB_NODE_ID]);
    const wantedEdgeIds = new Set();
    const wantedTopClusterIds = new Set();
    let seedIndex = 0;

    _recomputeDegree(newEdges);
    nodeIds.clear();
    nodeIds.add(DB_NODE_ID);

    function _clusterTitle(type) {{
      return CLUSTER_TITLES[type] || String(type || 'UNKNOWN').toUpperCase().replace(/_/g, ' ');
    }}

    function _ensureCluster(id, label, color, parent, classes) {{
      wantedNodeIds.add(id);
      nodeIds.add(id);
      const data = {{
        id: id,
        label: label,
        type: 'cluster',
        color: color || IRON,
      }};
      if (parent) data.parent = parent;
      const existing = cy.getElementById(id);
      if (existing.length) {{
        existing.data(data);
        existing.classes(classes || 'cluster');
        if (parent && existing.parent().id() !== parent) existing.move({{ parent: parent }});
        return existing;
      }}
      const parentNode = parent ? cy.getElementById(parent) : null;
      const base = parentNode && parentNode.length ? parentNode.position() : {{ x: 0, y: 0 }};
      seedIndex += 1;
      return cy.add({{
        group: 'nodes',
        data: data,
        classes: classes || 'cluster',
        position: {{
          x: base.x + 320 * Math.cos(seedIndex * GOLDEN_ANGLE),
          y: base.y + 320 * Math.sin(seedIndex * GOLDEN_ANGLE),
        }},
      }});
    }}

    function _ensureDbSpoke(cid) {{
      if (!cid) return;
      wantedTopClusterIds.add(cid);
      const eid = DB_NODE_ID + '__stores_in__' + cid;
      wantedEdgeIds.add(eid);
      const existing = cy.getElementById(eid);
      const data = {{
        id: eid,
        source: DB_NODE_ID,
        target: cid,
        edgeType: 'stores_in',
        lineColor: DB_COLOR,
      }};
      if (existing.length) existing.data(data);
      else cy.add({{ group: 'edges', data: data }});
    }}

    function _parentForNode(n) {{
      const type = String(n.type || 'unknown');
      if (type === 'scenario') {{
        const scen = String(n.id || '').startsWith('scenario:')
          ? String(n.id || '').substring('scenario:'.length)
          : String(n.label || n.id || 'scenario');
        const scId = 'cluster:scenario:' + scen;
        _ensureCluster(scId, scen, NODE_COLORS.scenario || IRON, '', 'cluster scenario_group');
        _ensureDbSpoke(scId);
        return '';
      }}
      const meta = n.meta || {{}};
      const scen = meta.scenario ? String(meta.scenario).trim() : '';
      if (scen) {{
        const scId = 'cluster:scenario:' + scen;
        _ensureCluster(scId, scen, NODE_COLORS.scenario || IRON, '', 'cluster scenario_group');
        const subId = scId + ':' + type;
        _ensureCluster(subId, _clusterTitle(type), NODE_COLORS[type] || IRON, scId, 'cluster subcluster');
        _ensureDbSpoke(scId);
        return subId;
      }}
      const clusterId = 'cluster:' + type;
      _ensureCluster(clusterId, _clusterTitle(type), NODE_COLORS[type] || IRON, '', 'cluster');
      _ensureDbSpoke(clusterId);
      return clusterId;
    }}

    function _fallbackPosition(parentId) {{
      const parent = parentId ? cy.getElementById(parentId) : null;
      const base = parent && parent.length ? parent.position() : {{ x: 0, y: 0 }};
      seedIndex += 1;
      return {{
        x: base.x + 220 * Math.cos(seedIndex * GOLDEN_ANGLE),
        y: base.y + 220 * Math.sin(seedIndex * GOLDEN_ANGLE),
      }};
    }}

    cy.startBatch();
    try {{
      newNodes.forEach(n => {{
        if (!n || !n.id) return;
        const type = String(n.type || 'unknown');
        const parentId = _parentForNode(n);
        if (type === 'scenario') return;
        const data = {{
          id: n.id,
          label: truncate(n.label || n.id, 20),
          fullLabel: n.label || n.id,
          type: type,
          meta: n.meta || {{}},
          color: NODE_COLORS[type] || IRON,
          size: nodeSize(n.id),
        }};
        if (parentId) data.parent = parentId;
        wantedNodeIds.add(n.id);
        nodeIds.add(n.id);
        const existing = cy.getElementById(n.id);
        if (existing.length) {{
          existing.data(data);
          existing.removeClass('faded path-highlight impact-upstream impact-downstream impact-pivot path-source-node');
          if (parentId && existing.parent().id() !== parentId) existing.move({{ parent: parentId }});
        }} else {{
          cy.add({{
            group: 'nodes',
            data: data,
            position: _fallbackPosition(parentId),
          }});
        }}
      }});

      wantedTopClusterIds.forEach(cid => _ensureDbSpoke(cid));

      newEdges.forEach(e => {{
        if (!e || !e.source || !e.target) return;
        const edgeSource = resolveEdgeEndpoint(e.source);
        const edgeTarget = resolveEdgeEndpoint(e.target);
        if (!edgeSource || !edgeTarget) return;
        const eid = e.source + '__' + e.type + '__' + e.target;
        wantedEdgeIds.add(eid);
        const data = {{
          id: eid,
          source: edgeSource,
          target: edgeTarget,
          originalSource: e.source,
          originalTarget: e.target,
          edgeType: e.type,
          lineColor: edgeColor(e.type),
        }};
        const existing = cy.getElementById(eid);
        if (existing.length) existing.data(data);
        else cy.add({{ group: 'edges', data: data }});
      }});

      cy.edges().forEach(e => {{
        if (!wantedEdgeIds.has(e.id())) e.remove();
      }});
      cy.nodes().not('.db-origin').forEach(n => {{
        if (!wantedNodeIds.has(n.id())) n.remove();
      }});
    }} finally {{
      cy.endBatch();
    }}

    const priorZoom = cy.zoom();
    const priorPan = cy.pan();
    const layout = cy.layout({{
      name: 'cose',
      animate: false,
      fit: !!fit,
      padding: 80,
      randomize: false,
      idealEdgeLength: 420,
      nodeRepulsion: 180000,
      nestingFactor: 1.2,
      gravity: 0.08,
    }});
    layout.on('layoutstop', function() {{
      if (!fit) {{
        cy.zoom(priorZoom);
        cy.pan(priorPan);
      }}
      _applyVisibility();
      _buildLegend();
      showToast('[GRAPH] ' + newNodes.length + ' nodes / ' + newEdges.length + ' edges', 'success');
    }});
    layout.run();
    return true;
  }};

  window._cySetFilter = function(types) {{
    activeTypes = types && types.length ? types : null;
    _applyVisibility();
  }};

  window._cySetSearch = function(term) {{
    searchTerm = (term || '').toLowerCase().trim();
    _applyVisibility();
  }};

  window._cyFitAll = function() {{ cy.fit(undefined, 80); }};

  window._cyShowLegend = function() {{
    document.getElementById('edge-legend').style.display = 'block';
  }};

  window._cyHideLegend = function() {{
    document.getElementById('edge-legend').style.display = 'none';
  }};

  window._cyNodeCount = function() {{ return cy.nodes().not('.cluster').length; }};
  window._cyEdgeCount = function() {{ return cy.edges().length; }};

  // ---- Path mode and impact highlighting --------------------------------
  let _pathMode = false;
  let _pathSource = null;

  window._cyActivatePathMode = function(active) {{
    _pathMode = !!active;
    _pathSource = null;
    cy.nodes().removeClass('path-source-node');
    if (_pathMode) {{
      showToast('[FIND PATH] Click source node', 'info');
    }} else {{
      window._cyClearHighlight();
    }}
  }};

  window._cyHighlightPath = function(pathNodeIds, edgePairs) {{
    const pathSet = new Set(pathNodeIds);
    const edgeSet = new Set(edgePairs.map(p => p[0] + '||' + p[1]));
    cy.nodes().forEach(n => {{
      if (n.hasClass('cluster')) return;
      if (pathSet.has(n.data('id'))) n.removeClass('faded').addClass('path-highlight');
      else n.addClass('faded').removeClass('path-highlight');
    }});
    cy.edges().forEach(e => {{
      const key = e.data('source') + '||' + e.data('target');
      const keyR = e.data('target') + '||' + e.data('source');
      if (edgeSet.has(key) || edgeSet.has(keyR)) e.removeClass('faded').addClass('path-highlight');
      else e.addClass('faded').removeClass('path-highlight');
    }});
    if (pathNodeIds && pathNodeIds.length > 0) {{
      showToast('[PATH] ' + pathNodeIds.length + ' nodes', 'success');
    }} else {{
      showToast('[PATH] No path found between those nodes', 'error');
    }}
  }};

  window._cyShowImpact = function(upstreamIds, downstreamIds, pivotId) {{
    const upSet   = new Set(upstreamIds);
    const downSet = new Set(downstreamIds);
    cy.nodes().forEach(n => {{
      if (n.hasClass('cluster')) return;
      const nid = n.data('id');
      n.removeClass('impact-upstream impact-downstream impact-pivot faded');
      if (nid === pivotId)        n.addClass('impact-pivot');
      else if (upSet.has(nid))   n.addClass('impact-upstream');
      else if (downSet.has(nid)) n.addClass('impact-downstream');
      else                       n.addClass('faded');
    }});
    cy.edges().forEach(e => {{
      const src = e.data('source'), tgt = e.data('target');
      if (upSet.has(src) || upSet.has(tgt) || downSet.has(src) || downSet.has(tgt) ||
          src === pivotId || tgt === pivotId) {{
        e.removeClass('faded');
      }} else {{
        e.addClass('faded');
      }}
    }});
    showToast('[IMPACT] ' + upstreamIds.length + ' upstream, ' + downstreamIds.length + ' downstream', 'success');
  }};

  window._cyClearHighlight = function() {{
    _pathSource = null;
    cy.nodes().removeClass('path-source-node path-highlight impact-upstream impact-downstream impact-pivot');
    cy.edges().removeClass('path-highlight');
    _applyVisibility();
  }};

  window._showToast = function(msg, type) {{ showToast(msg, type); }};

  function _applyVisibility() {{
    cy.nodes().forEach(n => {{
      if (n.hasClass('cluster')) return;
      const typeMatch  = !activeTypes || activeTypes.includes(n.data('type'));
      const labelMatch = !searchTerm  ||
        (n.data('fullLabel') || n.data('label') || '').toLowerCase().includes(searchTerm);
      if (typeMatch && labelMatch) n.removeClass('faded');
      else n.addClass('faded');
    }});
    cy.nodes('.cluster').forEach(cluster => {{
      const anyVisible = cluster.descendants().some(k => !k.hasClass('faded'));
      if (anyVisible) cluster.removeClass('faded');
      else cluster.addClass('faded');
    }});
    cy.edges().forEach(e => {{
      if (e.source().hasClass('faded') || e.target().hasClass('faded'))
        e.addClass('faded');
      else
        e.removeClass('faded');
    }});
  }}

}}  // end initGraph
</script>
</body>
</html>"""


class OntologyPanel(QWidget):
    """Entity graph panel — the Ecosystem / star-chart view.

    Attributes
    ----------
    entitySelected : pyqtSignal(str, str)
        Emitted when the user navigates to a graph node. Arguments are
        (entity_type, entity_id).
    """

    entitySelected = pyqtSignal(str, str)
    # (target_slug, focus_id, scenario_hint) — fired by the in-card QUICK NAV
    # buttons. The main window routes these to the corresponding Qt panel.
    navigateRequested = pyqtSignal(str, str, str)
    jobSubmitted = pyqtSignal(dict)

    def __init__(self, *, base_url: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._core_graph_url = f"{base_url}/ontology/graph?layer=core"
        self._full_graph_url = f"{base_url}/ontology/graph?layer=full"
        self._core_graph_view_url = f"{base_url}/ecosystem/graph_view?layer=core"
        self._graph_url = self._core_graph_url
        self._summary_url = f"{base_url}/ecosystem/summary"
        self._active_filters: list[str] = []
        self._legend_visible = False
        self._web_view: Optional[QWidget] = None
        self._web_available = False
        self._graph_page_loaded = False
        self._graph_fetchers: dict[str, _GraphFetcher] = {}
        self._full_graph_retry_count = 0
        self._pending_graph_update: dict[str, Any] = {}
        self._summary: dict[str, Any] = {}
        self._last_graph: dict[str, Any] = {}
        self._selected_entity: tuple[str, str] = ("", "")
        self._active_scenario = ""
        self._show_failed_only = False
        self._safe_actions = True
        self._summary_fetcher: Optional[_JsonFetcher] = None
        self._entity_fetcher: Optional[_JsonFetcher] = None
        self._action_runner: Optional[_ActionRunner] = None
        self._scenario_cards: dict[str, QFrame] = {}
        self._scenario_card_labels: dict[str, dict[str, QWidget]] = {}
        self._quick_run_jobs: dict[str, str] = {}
        self._result_poll_timer = QTimer(self)
        self._result_poll_timer.setInterval(900)
        self._result_poll_timer.timeout.connect(self._poll_quick_run_results)
        self._scenario_combo: Optional[QComboBox] = None
        self._training_progress: dict[str, dict[str, Any]] = {}
        self._auto_refresh_timer = QTimer(self)
        self._auto_refresh_timer.setInterval(5000)
        self._auto_refresh_timer.timeout.connect(self.refresh_deck)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setStyleSheet("background: transparent;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ---- toolbar ----
        toolbar = QFrame()
        toolbar.setObjectName("ontologyToolbar")
        toolbar.setFixedHeight(40)
        set_cvops_stylesheet(
            toolbar,
            lambda: (
                f"QFrame#ontologyToolbar {{ background: {_ontology_toolbar_bg()};"
                f"border-bottom: 1px solid {cvops_color('line_light')}; }}"
            ),
        )
        tbar_layout = QHBoxLayout(toolbar)
        tbar_layout.setContentsMargins(8, 0, 8, 0)
        tbar_layout.setSpacing(4)

        self._search = QLineEdit()
        self._search.setPlaceholderText("filter nodes...")
        self._search.setFixedWidth(160)
        set_cvops_stylesheet(
            self._search,
            lambda: (
                f"QLineEdit {{ background: {_ontology_field_bg()}; color: {cvops_color('text_signal')};"
                f"border: 1px solid {cvops_color('line_light')}; padding: 3px 6px;"
                "font-size: 10px; font-family: 'JetBrains Mono'; }"
            ),
        )
        self._search.textChanged.connect(self._on_search)
        tbar_layout.addWidget(self._search)

        tbar_layout.addSpacing(8)

        self._filter_btns: dict[str, QPushButton] = {}
        for etype, label in _ENTITY_TYPE_LABELS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(False)
            set_cvops_stylesheet(btn, _filter_btn_style)
            btn.toggled.connect(lambda checked, t=etype: self._on_filter_toggled(t, checked))
            self._filter_btns[etype] = btn
            tbar_layout.addWidget(btn)

        tbar_layout.addStretch()

        # stats label — populated after each graph load
        self._stats_label = QLabel("")
        set_cvops_stylesheet(
            self._stats_label,
            lambda: (
                f"QLabel {{ color: {cvops_color('text_iron')}; font-size: 10px;"
                "font-family: 'JetBrains Mono'; padding: 0 6px; }"
            ),
        )
        tbar_layout.addWidget(self._stats_label)

        self._legend_btn = QPushButton("[LEGEND]")
        self._legend_btn.setCheckable(True)
        set_cvops_stylesheet(self._legend_btn, _filter_btn_style)
        self._legend_btn.toggled.connect(self._toggle_legend)
        tbar_layout.addWidget(self._legend_btn)

        self._fit_btn = QPushButton("[FIT]")
        set_cvops_stylesheet(self._fit_btn, _filter_btn_style)
        self._fit_btn.clicked.connect(self._fit_graph)
        tbar_layout.addWidget(self._fit_btn)

        self._refresh_btn = QPushButton("[RELOAD]")
        set_cvops_stylesheet(self._refresh_btn, _filter_btn_style)
        self._refresh_btn.clicked.connect(self.reload)
        tbar_layout.addWidget(self._refresh_btn)

        self._path_btn = QPushButton("[FIND PATH]")
        self._path_btn.setCheckable(True)
        set_cvops_stylesheet(self._path_btn, _filter_btn_style)
        self._path_btn.toggled.connect(self._on_path_mode_toggled)
        tbar_layout.addWidget(self._path_btn)

        self._clear_impact_btn = QPushButton("[CLEAR]")
        set_cvops_stylesheet(self._clear_impact_btn, _filter_btn_style)
        self._clear_impact_btn.clicked.connect(self._on_clear_impact)
        self._clear_impact_btn.setToolTip("Clear path/impact highlight")
        tbar_layout.addWidget(self._clear_impact_btn)

        layout.addWidget(toolbar)

        # ---- graph + native command deck ----
        self._content_area = QWidget()
        self._content_area.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._content_area.setStyleSheet("background: transparent;")
        content_layout = QHBoxLayout(self._content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)
        layout.addWidget(self._content_area, stretch=1)

        graph_host = QWidget()
        graph_host.setObjectName("ontologyGraphHost")
        graph_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        graph_host.setStyleSheet("background: transparent;")
        graph_layout = QVBoxLayout(graph_host)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(graph_host, stretch=3)

        self._deck_scroll = _ViewportLockedScrollArea()
        self._deck_scroll.setObjectName("ecoCommandDeck")
        self._deck_scroll.setWidgetResizable(True)
        self._deck_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._deck_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._deck_scroll.setMinimumWidth(320)
        self._deck_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._deck_scroll.setStyleSheet("background: transparent;")
        self._deck_host = QWidget()
        self._deck_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        set_cvops_stylesheet(self._deck_host, _card_qss)
        self._deck_layout = QVBoxLayout(self._deck_host)
        self._deck_layout.setContentsMargins(0, 0, 0, 0)
        self._deck_layout.setSpacing(7)
        self._build_command_deck()
        self._deck_scroll.setWidget(self._deck_host)
        content_layout.addWidget(self._deck_scroll, stretch=1)

        self._init_web_view(graph_layout)
        QTimer.singleShot(350, self.refresh_deck)

    def _build_command_deck(self) -> None:
        header = QLabel("ECOSYSTEM COMMAND DECK")
        header.setProperty("ecoTitle", True)
        self._deck_layout.addWidget(header)

        self._ops_card = self._make_card("OPERATIONS")
        ops = self._ops_card.layout()
        assert isinstance(ops, QVBoxLayout)
        self._scenario_combo = QComboBox()
        set_cvops_stylesheet(
            self._scenario_combo,
            lambda: (
                f"QComboBox {{ background: {_ontology_field_bg()}; color: {cvops_color('text_signal')};"
                f"border: 1px solid {cvops_color('line_light')}; padding: 4px 6px;"
                f"font: 10px 'JetBrains Mono'; }}"
            ),
        )
        self._scenario_combo.currentTextChanged.connect(self._on_active_scenario_changed)
        ops.addWidget(self._scenario_combo)

        toggle_row = QGridLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        toggle_row.setSpacing(4)
        self._auto_refresh_btn = self._make_button("[AUTO REFRESH]", checkable=True)
        self._auto_refresh_btn.toggled.connect(self._toggle_auto_refresh)
        self._failed_only_btn = self._make_button("[FAILED JOBS]", checkable=True)
        self._failed_only_btn.toggled.connect(self._toggle_failed_only)
        self._safe_actions_btn = self._make_button("[SAFE ACTIONS]", checkable=True)
        self._safe_actions_btn.setChecked(True)
        self._safe_actions_btn.toggled.connect(self._toggle_safe_actions)
        self._active_only_btn = self._make_button("[ACTIVE SCENARIO]", checkable=True)
        self._active_only_btn.toggled.connect(self._toggle_active_scenario_filter)
        toggle_row.addWidget(self._auto_refresh_btn, 0, 0)
        toggle_row.addWidget(self._failed_only_btn, 0, 1)
        toggle_row.addWidget(self._safe_actions_btn, 1, 0)
        toggle_row.addWidget(self._active_only_btn, 1, 1)
        ops.addLayout(toggle_row)

        btn_row = QHBoxLayout()
        btn_row.setContentsMargins(0, 0, 0, 0)
        refresh = self._make_button("[REFRESH]")
        refresh.clicked.connect(self.refresh_deck)
        fit = self._make_button("[FIT GRAPH]")
        fit.clicked.connect(self._fit_graph)
        btn_row.addWidget(refresh)
        btn_row.addWidget(fit)
        ops.addLayout(btn_row)
        self._deck_layout.addWidget(self._ops_card)

        self._global_card = self._make_card("SYSTEM STATUS")
        self._global_grid = QGridLayout()
        self._global_grid.setContentsMargins(0, 0, 0, 0)
        self._global_grid.setSpacing(4)
        assert isinstance(self._global_card.layout(), QVBoxLayout)
        self._global_card.layout().addItem(self._global_grid)
        self._global_labels: dict[str, QLabel] = {}
        for i, key in enumerate(("service", "websocket", "queue", "training", "storage", "last_error")):
            self._global_labels[key] = self._add_kv(self._global_grid, i, key.replace("_", " ").upper(), "-")
        self._deck_layout.addWidget(self._global_card)

        self._pipeline_card = self._make_card("PIPELINE STAGES")
        stage_grid = QGridLayout()
        stage_grid.setContentsMargins(0, 0, 0, 0)
        stage_grid.setSpacing(4)
        stages = [
            ("INGEST", "scrape"),
            ("DATA", "data_selection"),
            ("CONFIG", "scenario_config"),
            ("TRAIN", "training"),
            ("EVAL", "test_range"),
            ("LINEAGE", "charts"),
            ("JOBS", "jobs"),
            ("DB", "database"),
        ]
        for idx, (label, target) in enumerate(stages):
            btn = self._make_button(f"[{label}]")
            btn.clicked.connect(lambda _c=False, t=target: self._nav(t, self._active_scenario))
            stage_grid.addWidget(btn, idx // 2, idx % 2)
        assert isinstance(self._pipeline_card.layout(), QVBoxLayout)
        self._pipeline_card.layout().addItem(stage_grid)
        self._deck_layout.addWidget(self._pipeline_card)

        self._scenario_section_title = QLabel("SCENARIO CONTROL CARDS")
        self._scenario_section_title.setProperty("ecoTitle", True)
        self._deck_layout.addWidget(self._scenario_section_title)

        self._scenario_host = QWidget()
        self._scenario_layout = QVBoxLayout(self._scenario_host)
        self._scenario_layout.setContentsMargins(0, 0, 0, 0)
        self._scenario_layout.setSpacing(6)
        self._deck_layout.addWidget(self._scenario_host)

        self._config_card = self._make_card("CONFIG DATA CARD")
        self._config_body = QLabel("Select a scenario to inspect runtime config.")
        self._config_body.setProperty("ecoValue", True)
        self._config_body.setWordWrap(True)
        assert isinstance(self._config_card.layout(), QVBoxLayout)
        self._config_card.layout().addWidget(self._config_body)
        self._deck_layout.addWidget(self._config_card)

        self._inspector_card = self._make_card("SELECTION INSPECTOR")
        self._inspector_body = QLabel("Click a graph node or scenario card.")
        self._inspector_body.setProperty("ecoValue", True)
        self._inspector_body.setWordWrap(True)
        assert isinstance(self._inspector_card.layout(), QVBoxLayout)
        self._inspector_card.layout().addWidget(self._inspector_body)
        self._deck_layout.addWidget(self._inspector_card)

        self._activity_card = self._make_card("ACTIVITY TIMELINE")
        self._activity_body = QLabel("-")
        self._activity_body.setProperty("ecoValue", True)
        self._activity_body.setWordWrap(True)
        assert isinstance(self._activity_card.layout(), QVBoxLayout)
        self._activity_card.layout().addWidget(self._activity_body)
        self._deck_layout.addWidget(self._activity_card)
        self._deck_layout.addStretch(1)

    def _make_card(self, title: str) -> QFrame:
        card = QFrame()
        card.setProperty("ecoCard", True)
        card.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 9, 10, 10)
        layout.setSpacing(7)
        label = QLabel(title)
        label.setProperty("ecoTitle", True)
        layout.addWidget(label)
        return card

    def _make_button(self, text: str, *, checkable: bool = False, role: str = "") -> QPushButton:
        btn = QPushButton(text)
        btn.setCheckable(checkable)
        if role:
            btn.setProperty("buttonRole", role)
        set_cvops_stylesheet(btn, _card_button_qss)
        return btn

    def _add_kv(self, grid: QGridLayout, row: int, key: str, value: str) -> QLabel:
        k = QLabel(key)
        k.setProperty("ecoMeta", True)
        v = QLabel(value)
        v.setProperty("ecoValue", True)
        v.setWordWrap(True)
        grid.addWidget(k, row, 0)
        grid.addWidget(v, row, 1)
        return v

    def refresh_deck(self) -> None:
        fetcher = self._summary_fetcher
        if fetcher is not None and fetcher.isRunning():
            return
        self._summary_fetcher = _JsonFetcher("summary", self._summary_url, parent=self)
        self._summary_fetcher.fetched.connect(self._on_json_fetched)
        self._summary_fetcher.failed.connect(self._on_json_failed)
        self._summary_fetcher.start()

    def _on_json_fetched(self, key: str, payload: dict[str, Any]) -> None:
        if key == "summary":
            self._summary = payload
            self._render_summary()
            return
        if key.startswith("entity:"):
            self._render_entity(payload)

    def _on_json_failed(self, key: str, message: str) -> None:
        if key == "summary":
            self._global_labels.get("last_error", QLabel()).setText(_short(message, 160))
        elif key.startswith("entity:"):
            self._inspector_body.setText(f"Entity lookup failed: {_short(message, 180)}")

    def _render_summary(self) -> None:
        payload = self._summary
        health = payload.get("health") if isinstance(payload.get("health"), dict) else {}
        storage = payload.get("storage") if isinstance(payload.get("storage"), dict) else {}
        jobs = payload.get("jobs") if isinstance(payload.get("jobs"), list) else []
        scenarios = payload.get("scenarios") if isinstance(payload.get("scenarios"), list) else []
        counts = payload.get("job_counts") if isinstance(payload.get("job_counts"), dict) else {}

        self._global_labels["service"].setText(
            "ok" if health.get("status") == "ok" and health.get("worker_alive") else "degraded"
        )
        self._global_labels["websocket"].setText("see top status rail")
        self._global_labels["queue"].setText(
            f"queued {counts.get('queued', 0)} / running {counts.get('running', 0)} / errors {counts.get('error', 0)}"
        )
        active_training = payload.get("active_training_scenarios") or []
        self._global_labels["training"].setText(", ".join(active_training) if active_training else "idle")
        self._global_labels["storage"].setText(
            f"{_fmt_bytes(storage.get('disk_free'))} free / cytoscape {'cached' if storage.get('cytoscape_cached') else 'missing'}"
        )
        self._global_labels["last_error"].setText(_short(payload.get("last_error") or payload.get("scenario_error"), 120))

        self._sync_scenario_combo(scenarios)
        self._render_scenario_cards(scenarios)
        self._render_config_card()
        self._render_activity(jobs)

    def _sync_scenario_combo(self, scenarios: list[Any]) -> None:
        if self._scenario_combo is None:
            return
        names = [str(s.get("name") or "") for s in scenarios if isinstance(s, dict) and s.get("name")]
        current = self._active_scenario or (names[0] if names else "")
        self._scenario_combo.blockSignals(True)
        self._scenario_combo.clear()
        self._scenario_combo.addItems(names)
        if current in names:
            self._scenario_combo.setCurrentText(current)
        self._scenario_combo.blockSignals(False)
        if not self._active_scenario and current:
            self._active_scenario = current

    def _render_scenario_cards(self, scenarios: list[Any]) -> None:
        while self._scenario_layout.count():
            item = self._scenario_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._scenario_cards = {}
        self._scenario_card_labels = {}
        for scenario in scenarios:
            if not isinstance(scenario, dict):
                continue
            name = str(scenario.get("name") or "").strip()
            if not name:
                continue
            if self._active_only_btn.isChecked() and self._active_scenario and name != self._active_scenario:
                continue
            card = self._build_scenario_card(scenario)
            self._scenario_cards[name] = card
            self._scenario_layout.addWidget(card)
        self._scenario_layout.addStretch(1)

    def _build_scenario_card(self, scenario: dict[str, Any]) -> QFrame:
        name = str(scenario.get("name") or "")
        card = self._make_card(_short(scenario.get("display_name") or name, 38))
        card.setProperty("selected", name == self._active_scenario)
        card.setMinimumHeight(470)
        context = scenario.get("training_context") if isinstance(scenario.get("training_context"), dict) else {}
        latest_job = scenario.get("latest_job") if isinstance(scenario.get("latest_job"), dict) else {}
        latest_run = scenario.get("latest_run") if isinstance(scenario.get("latest_run"), dict) else {}
        status = _status_label(str(context.get("state") or latest_job.get("state") or scenario.get("status") or ""))
        body_text, _pulse, log_text, err_text = self._scenario_card_text(scenario)

        body = QLabel(body_text)
        body.setProperty("ecoValue", True)
        body.setWordWrap(True)
        body.setMinimumHeight(150)
        assert isinstance(card.layout(), QVBoxLayout)
        card.layout().addWidget(body)

        pulse_widget = _MatrixStatusWidget()
        pulse_widget.set_status(status)
        card.layout().addWidget(pulse_widget)

        log_label = QLabel(log_text)
        log_label.setProperty("ecoMeta", True)
        log_label.setWordWrap(True)
        log_label.setMinimumHeight(104)
        log_label.setMaximumHeight(132)
        card.layout().addWidget(log_label)

        err_label = QLabel(err_text)
        err_label.setProperty("ecoMeta", True)
        err_label.setWordWrap(True)
        err_label.setMinimumHeight(0)
        err_label.setMaximumHeight(42)
        err_label.setVisible(bool(err_text))
        card.layout().addWidget(err_label)
        self._scenario_card_labels[name] = {
            "body": body,
            "pulse": pulse_widget,
            "log": log_label,
            "error": err_label,
        }

        nav_grid = QGridLayout()
        nav_grid.setContentsMargins(0, 0, 0, 0)
        nav_grid.setSpacing(4)
        navs = [
            ("[CONFIG]", "scenario_config"),
            ("[DATA]", "data_selection"),
            ("[TRAIN]", "training"),
            ("[TEST]", "test_range"),
            ("[RESULTS]", "results"),
            ("[JOBS]", "jobs"),
        ]
        for idx, (label, target) in enumerate(navs):
            btn = self._make_button(label)
            btn.clicked.connect(lambda _c=False, t=target, n=name: self._nav(t, n))
            nav_grid.addWidget(btn, idx // 3, idx % 3)
        card.layout().addItem(nav_grid)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        train = self._make_button("[TRAIN NOW]", role="primary")
        update = self._make_button("[UPDATE]")
        verify = self._make_button("[VERIFY]")
        quick = self._make_button("[QUICK RUN]", role="primary")
        train.clicked.connect(lambda _c=False, n=name: self._scenario_action(n, "train"))
        update.clicked.connect(lambda _c=False, n=name: self._scenario_action(n, "update"))
        verify.clicked.connect(lambda _c=False, n=name: self._scenario_action(n, "verify"))
        quick.clicked.connect(lambda _c=False, n=name: self._quick_run_scenario(n))
        action_row.addWidget(train)
        action_row.addWidget(update)
        action_row.addWidget(verify)
        action_row.addWidget(quick)
        card.layout().addLayout(action_row)
        result_face = self._build_quick_result_face(name)
        result_face.setVisible(False)
        card.layout().addWidget(result_face, stretch=1)
        card.mousePressEvent = lambda _event, n=name: self._select_scenario(n)  # type: ignore[method-assign]
        return card

    def _build_quick_result_face(self, scenario: str) -> QWidget:
        face = QFrame()
        face.setObjectName("ecoQuickRunResult")
        face.setMinimumHeight(230)
        set_cvops_stylesheet(
            face,
            lambda: (
                f"QFrame#ecoQuickRunResult {{ border: 1px solid {cvops_color('line_light')}; "
                f"background: {theme_rgba('panel', 0.72)}; }}"
            ),
        )
        layout = QVBoxLayout(face)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        title = QLabel("QUICK RUN RESULT")
        title.setProperty("ecoTitle", True)
        layout.addWidget(title)
        image = QLabel("[WAITING FOR DETECTION IMAGE]")
        image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image.setMinimumHeight(150)
        image.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        set_cvops_stylesheet(
            image,
            lambda: f"border: 1px solid {cvops_color('line_med')}; color: {cvops_color('text_iron')};",
        )
        layout.addWidget(image, stretch=1)
        summary = QLabel("")
        summary.setProperty("ecoValue", True)
        summary.setWordWrap(True)
        layout.addWidget(summary)
        row = QHBoxLayout()
        back = self._make_button("[GO BACK]")
        back.clicked.connect(lambda _c=False, n=scenario: self._show_quick_front(n))
        row.addStretch(1)
        row.addWidget(back)
        layout.addLayout(row)
        self._scenario_card_labels.setdefault(scenario, {})
        self._scenario_card_labels[scenario].update(
            {
                "result_face": face,
                "result_image": image,
                "result_summary": summary,
            }
        )
        return face

    def _scenario_card_text(self, scenario: dict[str, Any]) -> tuple[str, str, str, str]:
        context = scenario.get("training_context") if isinstance(scenario.get("training_context"), dict) else {}
        latest_job = scenario.get("latest_job") if isinstance(scenario.get("latest_job"), dict) else {}
        latest_run = scenario.get("latest_run") if isinstance(scenario.get("latest_run"), dict) else {}
        status = _status_label(str(context.get("state") or latest_job.get("state") or scenario.get("status") or ""))
        event = context.get("latest_event") if isinstance(context.get("latest_event"), dict) else {}
        batch = context.get("latest_batch") if isinstance(context.get("latest_batch"), dict) else {}
        logs = context.get("recent_logs") if isinstance(context.get("recent_logs"), list) else []
        split = scenario.get("split_counts") if isinstance(scenario.get("split_counts"), dict) else {}
        split_txt = ", ".join(f"{k}:{v}" for k, v in sorted(split.items())) or "-"
        mode = "UPDATE" if context.get("update_mode") else "TRAIN"
        if not context:
            mode = "TRAIN"
        body_text = "\n".join(
            [
                f"id: {scenario.get('name') or '-'}",
                f"state: {status}",
                f"mode: {mode}  trigger: {context.get('trigger') or '-'}",
                f"dataset: {scenario.get('dataset') or '-'}",
                f"data: count {scenario.get('dataset_count', '-')} / splits {split_txt}",
                f"snapshot: {_short(context.get('dataset_snapshot_id'), 34)}",
                f"backbone: {scenario.get('backbone_type') or '-'}",
                f"classes: {scenario.get('class_count', len(scenario.get('classes') or []))}",
                f"epoch: {self._epoch_text(event)}  progress: {_fmt_pct(event.get('progress'))}",
                f"loss: train {_fmt_metric(event.get('train_loss'))} / val {_fmt_metric(event.get('val_loss'))}",
                f"metrics: P {_fmt_metric(event.get('precision'))}  R {_fmt_metric(event.get('recall'))}  mAP50 {_fmt_metric(event.get('map50') or latest_run.get('map50'))}",
                f"batch: stall {_fmt_metric(batch.get('stall_pct'))}% / {batch.get('samples_per_sec') or '-'} samples/s",
            ]
        )
        log_text = ""
        if logs:
            log_text = "latest output:\n" + "\n".join(_short(line, 110) for line in logs[-5:])
        err = str(context.get("error") or latest_job.get("error") or scenario.get("error") or "").strip()
        err_text = f"failure: {_short(err, 140)}" if err else ""
        return body_text, status, log_text, err_text

    def _update_scenario_card(self, scenario_name: str) -> None:
        scenario = self._scenario_by_name(scenario_name)
        labels = self._scenario_card_labels.get(scenario_name)
        if not scenario or not labels:
            self._render_summary()
            return
        context = scenario.get("training_context") if isinstance(scenario.get("training_context"), dict) else {}
        latest_job = scenario.get("latest_job") if isinstance(scenario.get("latest_job"), dict) else {}
        status = _status_label(str(context.get("state") or latest_job.get("state") or scenario.get("status") or ""))
        body_text, pulse_text, log_text, err_text = self._scenario_card_text(scenario)
        labels["body"].setText(body_text)
        pulse = labels.get("pulse")
        if isinstance(pulse, _MatrixStatusWidget):
            pulse.set_status(status)
        labels["log"].setText(log_text)
        labels["error"].setText(err_text)
        labels["error"].setVisible(bool(err_text))

    def _show_quick_result(self, scenario: str, result: dict[str, Any]) -> None:
        labels = self._scenario_card_labels.get(str(scenario or ""))
        if not labels:
            return
        face = labels.get("result_face")
        image_label = labels.get("result_image")
        summary_label = labels.get("result_summary")
        if not isinstance(face, QWidget):
            return
        if isinstance(image_label, QLabel):
            overlay_b64 = str(result.get("overlay_image") or "")
            if overlay_b64:
                pix = pixmap_from_b64_jpeg(overlay_b64)
                if not pix.isNull():
                    target = image_label.size()
                    if target.width() > 20 and target.height() > 20:
                        pix = pix.scaled(
                            target,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation,
                        )
                    image_label.setPixmap(pix)
                    image_label.setText("")
                else:
                    image_label.clear()
                    image_label.setText("[INVALID DETECTION IMAGE]")
            else:
                image_label.clear()
                image_label.setText("[NO DETECTION IMAGE]")
        if isinstance(summary_label, QLabel):
            detections = result.get("detections") if isinstance(result.get("detections"), list) else []
            err = str(result.get("error") or "").strip()
            parts = [
                f"job: {result.get('job_id') or '-'}",
                f"detections: {len(detections)}",
                f"elapsed: {result.get('elapsed_ms', '-')} ms",
            ]
            if err:
                parts.append(f"error: {_short(err, 120)}")
            else:
                summary = str(result.get("summary") or "").strip()
                if summary:
                    parts.append(_short(summary, 120))
            summary_label.setText("\n".join(parts))
        self._flip_to_quick_result(str(scenario or ""))

    def _flip_to_quick_result(self, scenario: str) -> None:
        labels = self._scenario_card_labels.get(scenario)
        if not labels:
            return
        face = labels.get("result_face")
        if not isinstance(face, QWidget):
            return
        for key in ("body", "pulse", "log", "error"):
            widget = labels.get(key)
            if isinstance(widget, QWidget):
                widget.setVisible(False)
        face.setMaximumHeight(0)
        face.setVisible(True)
        anim = QPropertyAnimation(face, b"maximumHeight", self)
        anim.setDuration(260)
        anim.setStartValue(0)
        anim.setEndValue(260)
        anim.start()
        self._result_flip_anim = anim

    def _show_quick_front(self, scenario: str) -> None:
        labels = self._scenario_card_labels.get(str(scenario or ""))
        if not labels:
            return
        face = labels.get("result_face")
        if isinstance(face, QWidget):
            face.setVisible(False)
            face.setMaximumHeight(16777215)
        for key in ("body", "pulse", "log", "error"):
            widget = labels.get(key)
            if isinstance(widget, QWidget):
                widget.setVisible(True)

    def _epoch_text(self, event: dict[str, Any]) -> str:
        try:
            epoch = int(event.get("epoch"))
            epochs = int(event.get("epochs"))
            if epoch < 0:
                return f"0/{max(1, epochs)}"
            return f"{epoch + 1}/{max(1, epochs)}"
        except Exception:
            return "-"

    def _render_config_card(self) -> None:
        scenario = self._scenario_by_name(self._active_scenario)
        if not scenario:
            self._config_body.setText("No active scenario selected.")
            return
        hp = scenario.get("hyperparams") if isinstance(scenario.get("hyperparams"), dict) else {}
        split = scenario.get("split_counts") if isinstance(scenario.get("split_counts"), dict) else {}
        hp_txt = ", ".join(f"{k}={v}" for k, v in list(hp.items())[:8]) or "-"
        split_txt = ", ".join(f"{k}:{v}" for k, v in sorted(split.items())) or "-"
        self._config_body.setText(
            "\n".join(
                [
                    f"scenario: {scenario.get('name')}",
                    f"dataset binding: {scenario.get('dataset') or '-'}",
                    f"base model: {_short(scenario.get('base_model'), 44)}",
                    f"weights: {_short(scenario.get('weights'), 44)}",
                    f"hyperparams: {hp_txt}",
                    f"splits: {split_txt}",
                    "edit path: [CONFIG] for scenario YAML, [DATA] for dataset operations",
                ]
            )
        )

    def _render_activity(self, jobs: list[Any]) -> None:
        rows: list[str] = []
        for job in jobs[:20]:
            if not isinstance(job, dict):
                continue
            state = str(job.get("state") or "")
            if self._show_failed_only and state.lower() not in {"error", "failed", "cancelled"}:
                continue
            scen = str(job.get("scenario") or "-")
            jtype = str(job.get("job_type") or "-")
            err = str(job.get("error") or "")
            rows.append(f"{state.upper():<9} {scen} / {jtype}" + (f" / {_short(err, 64)}" if err else ""))
            if len(rows) >= 8:
                break
        self._activity_body.setText("\n".join(rows) if rows else "No matching activity.")

    def apply_job_status(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        jobs = self._summary.get("jobs") if isinstance(self._summary, dict) else []
        if not isinstance(jobs, list):
            jobs = []
            self._summary["jobs"] = jobs
        job_id = str(payload.get("job_id") or "")
        replaced = False
        for idx, job in enumerate(jobs):
            if isinstance(job, dict) and str(job.get("job_id") or "") == job_id:
                merged = dict(job)
                merged.update(payload)
                jobs[idx] = merged
                replaced = True
                break
        if not replaced and job_id:
            jobs.insert(0, dict(payload))
        scenario = str(payload.get("scenario") or "")
        if scenario:
            scen = self._scenario_by_name(scenario)
            if scen:
                latest_job = scen.get("latest_job") if isinstance(scen.get("latest_job"), dict) else {}
                merged_job = dict(latest_job)
                merged_job.update(payload)
                scen["latest_job"] = merged_job
                if str(payload.get("job_type") or "").lower() == "train":
                    ctx = scen.get("training_context") if isinstance(scen.get("training_context"), dict) else {}
                    ctx = dict(ctx)
                    ctx["job_id"] = job_id or ctx.get("job_id", "")
                    ctx["state"] = str(payload.get("state") or ctx.get("state") or "")
                    ctx["error"] = str(payload.get("error") or ctx.get("error") or "")
                    scen["training_context"] = ctx
        self._update_scenario_card(scenario) if scenario else self._render_activity(jobs)

    def apply_training_progress(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        scenario = str(payload.get("scenario") or "").strip()
        if not scenario:
            return
        self._training_progress[scenario] = dict(payload)
        scen = self._scenario_by_name(scenario)
        if not scen:
            return
        ctx = scen.get("training_context") if isinstance(scen.get("training_context"), dict) else {}
        ctx = dict(ctx)
        job_id = str(payload.get("job_id") or ctx.get("job_id") or "")
        if job_id:
            ctx["job_id"] = job_id
        event_type = str(payload.get("event") or "").strip().lower()
        if event_type in {"log", "log_batch"}:
            logs = ctx.get("recent_logs") if isinstance(ctx.get("recent_logs"), list) else []
            if event_type == "log_batch" and isinstance(payload.get("lines"), list):
                lines = [
                    str(item.get("line") or "").strip()
                    for item in payload.get("lines") or []
                    if isinstance(item, dict) and str(item.get("line") or "").strip()
                ]
            else:
                line = str(payload.get("line") or "").strip()
                lines = [line] if line else []
            if lines:
                logs = (logs + lines)[-4:]
            ctx["recent_logs"] = logs
        elif event_type == "batch_metrics":
            ctx["latest_batch"] = dict(payload)
            ctx["state"] = "running"
        else:
            ctx["latest_event"] = dict(payload)
            if event_type == "completed":
                ctx["state"] = "done"
            elif event_type == "failed":
                ctx["state"] = "error"
                ctx["error"] = str(payload.get("error") or "")
            elif event_type:
                ctx["state"] = "running"
        scen["training_context"] = ctx
        if self._active_scenario in {"", scenario}:
            self._active_scenario = scenario
        self._update_scenario_card(scenario)

    def _scenario_by_name(self, name: str) -> dict[str, Any]:
        scenarios = self._summary.get("scenarios") if isinstance(self._summary, dict) else []
        if not isinstance(scenarios, list):
            return {}
        for scenario in scenarios:
            if isinstance(scenario, dict) and str(scenario.get("name") or "") == name:
                return scenario
        return {}

    def _select_scenario(self, name: str) -> None:
        self._active_scenario = str(name or "")
        if self._scenario_combo is not None and self._active_scenario:
            self._scenario_combo.blockSignals(True)
            self._scenario_combo.setCurrentText(self._active_scenario)
            self._scenario_combo.blockSignals(False)
        self._render_summary()
        self._inspect_entity("scenario", self._active_scenario)

    def _on_active_scenario_changed(self, name: str) -> None:
        self._active_scenario = str(name or "")
        if self._active_only_btn.isChecked() and self._active_scenario:
            self._search.setText(self._active_scenario)
        self._render_summary()

    def _toggle_auto_refresh(self, enabled: bool) -> None:
        if enabled:
            self._auto_refresh_timer.start()
        else:
            self._auto_refresh_timer.stop()

    def _toggle_failed_only(self, enabled: bool) -> None:
        self._show_failed_only = bool(enabled)
        jobs = self._summary.get("jobs") if isinstance(self._summary, dict) else []
        self._render_activity(jobs if isinstance(jobs, list) else [])

    def _toggle_safe_actions(self, enabled: bool) -> None:
        self._safe_actions = bool(enabled)

    def _toggle_active_scenario_filter(self, _enabled: bool) -> None:
        if self._active_only_btn.isChecked() and self._active_scenario:
            self._search.setText(self._active_scenario)
        elif self._search.text() == self._active_scenario:
            self._search.clear()
        self._render_summary()

    def _nav(self, target: str, focus_id: str = "") -> None:
        self.navigateRequested.emit(str(target or ""), str(focus_id or ""), self._active_scenario)

    def _scenario_action(self, scenario: str, action: str) -> None:
        action = str(action or "").strip().lower()
        scenario = str(scenario or "").strip()
        if not scenario:
            return
        labels = {"train": "TRAIN", "update": "UPDATE", "verify": "VERIFY"}
        scen = self._scenario_by_name(scenario)
        split = scen.get("split_counts") if isinstance(scen.get("split_counts"), dict) else {}
        split_txt = ", ".join(f"{k}:{v}" for k, v in sorted(split.items())) or "-"
        latest_run = scen.get("latest_run") if isinstance(scen.get("latest_run"), dict) else {}
        action_detail = "\n".join(
            [
                f"Scenario: {scenario}",
                f"Operation: {labels.get(action, action.upper())}",
                f"Dataset: {scen.get('dataset') or '-'}",
                f"Data count: {scen.get('dataset_count', '-')} / splits {split_txt}",
                f"Backbone: {scen.get('backbone_type') or '-'}",
                f"Base model: {_short(scen.get('base_model'), 80)}",
                f"Latest run: {latest_run.get('version') or '-'}",
                f"Update mode: {'yes' if action == 'update' else 'no'}",
            ]
        )
        if self._safe_actions:
            ok = QMessageBox.question(
                self,
                f"{labels.get(action, action.upper())} Scenario",
                f"{action_detail}\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if ok != QMessageBox.StandardButton.Yes:
                return
        self._inspector_body.setText(f"Submitting operation...\n{action_detail}")
        body: Optional[dict[str, Any]] = {} if action in {"train", "update"} else {"note": "verified from Ecosystem command deck"}
        scenario_path = urllib.parse.quote(scenario, safe="")
        self._run_action(
            label=f"{action}:{scenario}",
            method="POST",
            path=f"/scenarios/{scenario_path}/{action}",
            body=body,
        )

    def _quick_run_scenario(self, scenario: str) -> None:
        scenario = str(scenario or "").strip()
        if not scenario:
            return
        scen = self._scenario_by_name(scenario)
        if not scen:
            return
        if not self._scenario_ready_for_quick_run(scen):
            QMessageBox.information(
                self,
                "Quick Run",
                f"Scenario '{scenario}' does not have a ready inference target yet. Train or verify it first.",
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Quick Run Image - {scenario}",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp)",
        )
        if not path:
            return
        try:
            raw = Path(path).read_bytes()
        except Exception as exc:
            self._inspector_body.setText(f"Quick run failed to read image:\n{exc}")
            return
        version = self._quick_run_version(scen)
        payload: dict[str, Any] = {
            "scenario": scenario,
            "version": version,
            "model_artifact": "",
            "image_b64": base64.b64encode(raw).decode("ascii"),
            "source": "cvops_ecosystem_quick_run",
        }
        btype = str(scen.get("backbone_type") or "").strip().lower()
        if btype == "face_recognition":
            payload["backbone_config_override"] = {
                "threshold": 0.72,
                "margin_threshold": 0.045,
                "top_k": 5,
            }
        else:
            payload["infer_overrides"] = {
                "conf": 0.25,
                "iou": 0.70,
                "max_det": 300,
            }
        self._inspector_body.setText(
            "\n".join(
                [
                    "Submitting quick run...",
                    f"scenario: {scenario}",
                    f"version: {version or 'configured weights'}",
                    f"image: {Path(path).name}",
                    f"backbone: {scen.get('backbone_type') or '-'}",
                ]
            )
        )
        self._run_action(
            label=f"quickrun:{scenario}",
            method="POST",
            path="/jobs",
            body=payload,
        )

    def _scenario_ready_for_quick_run(self, scenario: dict[str, Any]) -> bool:
        if bool(scenario.get("weights_ready")):
            return True
        latest = scenario.get("latest_run") if isinstance(scenario.get("latest_run"), dict) else {}
        return bool(latest.get("weights") or latest.get("final_model_path") or latest.get("version"))

    def _quick_run_version(self, scenario: dict[str, Any]) -> str:
        latest = scenario.get("latest_run") if isinstance(scenario.get("latest_run"), dict) else {}
        return str(latest.get("version") or "").strip()

    def _run_action(
        self,
        *,
        label: str,
        method: str,
        path: str,
        body: Optional[dict[str, Any]] = None,
    ) -> None:
        if self._action_runner is not None and self._action_runner.isRunning():
            QMessageBox.information(self, "Ecosystem Action", "Another Ecosystem action is still running.")
            return
        url = self._base_url.rstrip("/") + path
        self._action_runner = _ActionRunner(label=label, method=method, url=url, body=body, parent=self)
        self._action_runner.finishedOk.connect(self._on_action_ok)
        self._action_runner.failed.connect(self._on_action_failed)
        self._action_runner.start()

    def _get_json(self, path: str, *, timeout: float = 3.0) -> dict[str, Any]:
        url = self._base_url.rstrip("/") + path
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw) if raw else {}
        return payload if isinstance(payload, dict) else {}

    def _on_action_ok(self, label: str, _payload: dict[str, Any]) -> None:
        self._inspector_body.setText(f"Action completed: {label}")
        if str(label).startswith("quickrun:"):
            self.jobSubmitted.emit(dict(_payload))
            job_id = str(_payload.get("job_id") or "")
            scen = str(_payload.get("scenario") or label.split(":", 1)[-1])
            if job_id and scen:
                self._quick_run_jobs[job_id] = scen
                self._show_quick_waiting(scen, job_id)
                if not self._result_poll_timer.isActive():
                    self._result_poll_timer.start()
            self._inspector_body.setText(
                f"Quick run submitted: {job_id or '-'}\nscenario: {scen}\njob_type: {_payload.get('job_type') or '-'}"
            )
        QTimer.singleShot(250, self.refresh_deck)
        QTimer.singleShot(350, self.reload)

    def _on_action_failed(self, label: str, message: str) -> None:
        self._inspector_body.setText(f"Action failed: {label}\n{_short(message, 240)}")

    def _show_quick_waiting(self, scenario: str, job_id: str) -> None:
        labels = self._scenario_card_labels.get(str(scenario or ""))
        if not labels:
            return
        image = labels.get("result_image")
        summary = labels.get("result_summary")
        if isinstance(image, QLabel):
            image.clear()
            image.setText("[RUNNING DETECTION...]")
        if isinstance(summary, QLabel):
            summary.setText(f"job: {job_id}\nwaiting for result image")
        self._flip_to_quick_result(str(scenario or ""))

    def _poll_quick_run_results(self) -> None:
        if not self._quick_run_jobs:
            self._result_poll_timer.stop()
            return
        for job_id, scenario in list(self._quick_run_jobs.items()):
            try:
                result = self._get_json(f"/jobs/{urllib.parse.quote(job_id, safe='')}/result", timeout=1.2)
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    continue
                self._quick_run_jobs.pop(job_id, None)
                self._show_quick_result(scenario, {"job_id": job_id, "error": f"HTTP {exc.code}"})
                continue
            except Exception:
                continue
            if isinstance(result, dict):
                result.setdefault("job_id", job_id)
                result.setdefault("scenario", scenario)
                self._quick_run_jobs.pop(job_id, None)
                self._show_quick_result(scenario, result)
        if not self._quick_run_jobs:
            self._result_poll_timer.stop()

    def apply_job_result(self, job_id: str, result: dict[str, Any]) -> None:
        jid = str(job_id or result.get("job_id") or "")
        scenario = self._quick_run_jobs.pop(jid, "") if jid else ""
        scenario = scenario or str(result.get("scenario") or "")
        if not scenario:
            return
        payload = dict(result)
        payload.setdefault("job_id", jid)
        payload.setdefault("scenario", scenario)
        self._show_quick_result(scenario, payload)
        if not self._quick_run_jobs:
            self._result_poll_timer.stop()

    def _inspect_entity(self, entity_type: str, entity_id: str) -> None:
        entity_type = str(entity_type or "").strip()
        entity_id = str(entity_id or "").strip()
        if not entity_type or not entity_id:
            return
        self._selected_entity = (entity_type, entity_id)
        url = f"{self._base_url.rstrip('/')}/ontology/entity/{urllib.parse.quote(entity_type)}/{urllib.parse.quote(entity_id, safe='')}"
        fetcher = self._entity_fetcher
        if fetcher is not None and fetcher.isRunning():
            fetcher.terminate()
            fetcher.wait(200)
        self._entity_fetcher = _JsonFetcher(f"entity:{entity_type}:{entity_id}", url, parent=self)
        self._entity_fetcher.fetched.connect(self._on_json_fetched)
        self._entity_fetcher.failed.connect(self._on_json_failed)
        self._entity_fetcher.start()

    def _render_entity(self, entity: dict[str, Any]) -> None:
        etype = str(entity.get("type") or self._selected_entity[0] or "-")
        eid = str(entity.get("entity_id") or self._selected_entity[1] or "-")
        if etype == "scenario_group":
            etype = "scenario"
        interesting = []
        for key in (
            "name",
            "display_name",
            "status",
            "state",
            "dataset",
            "backbone_type",
            "job_type",
            "scenario",
            "mode",
            "created_at",
            "error",
        ):
            if entity.get(key) not in (None, "", []):
                interesting.append(f"{key}: {_short(entity.get(key), 92)}")
        edges = entity.get("edges") if isinstance(entity.get("edges"), list) else []
        if edges:
            interesting.append(f"relations: {len(edges)} direct")
        self._inspector_body.setText(
            f"[{etype.upper()}] {eid}\n" + ("\n".join(interesting) if interesting else "No extended metadata.")
        )

    def _init_web_view(self, parent_layout: QVBoxLayout) -> None:
        try:
            from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            class _ConsolePage(QWebEnginePage):
                def javaScriptConsoleMessage(
                    self, level, message, line_number, source_id
                ) -> None:  # type: ignore[override]
                    print(
                        f"[ONTOLOGY JS] {source_id}:{line_number} {message}",
                        flush=True,
                    )

            self._web_view = QWebEngineView()
            self._web_view.setObjectName("ontologyWebView")
            # Keep QtWebEngine's native NSView scoped to the web view itself.
            # Without this, Qt can promote ancestor widgets to native windows
            # during later layout changes, which lets the ecosystem web layer
            # intercept clicks intended for unrelated HUD/chrome controls.
            self._web_view.setAttribute(
                Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True
            )
            page = _ConsolePage(self._web_view)
            # Page-level transparency keeps the Cytoscape canvas painting over
            # the aurora backdrop without using widget-level
            # WA_TranslucentBackground, which on macOS is the documented cause
            # of QWebEngineView's NSView returning YES from hitTest: for points
            # outside its bounds. That hit-test bug is what was eating clicks on
            # the top nav while the Ecosystem plane was visible.
            page.setBackgroundColor(QColor(0, 0, 0, 0))
            self._web_view.setPage(page)
            self._web_view.setStyleSheet("background: transparent; border: none;")
            settings = self._web_view.settings()
            settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptEnabled, True)
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
            )
            settings.setAttribute(
                QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True
            )
            self._web_view.page().navigationRequested.connect(self._on_navigation)
            self._web_available = True
            parent_layout.addWidget(self._web_view)
        except Exception as exc:
            fallback = QLabel(
                f"[ONTOLOGY SURFACE] QWebEngineView unavailable.\n"
                f"Install PyQt6-WebEngine to enable the Ecosystem graph panel.\n\n"
                f"Reason: {exc}"
            )
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback.setWordWrap(True)
            fallback.setStyleSheet(
                f"color: {cvops_color('text_iron')}; font-size: 12px; padding: 40px;"
            )
            parent_layout.addWidget(fallback)

        QTimer.singleShot(300, self.reload)

    def reload(self) -> None:
        if not self._web_available or self._web_view is None:
            return
        self._full_graph_retry_count = 0
        self._start_graph_fetch("core")

    def _start_graph_fetch(self, layer: str) -> None:
        if not self._web_available or self._web_view is None:
            return
        layer = "core" if str(layer or "").lower() == "core" else "full"
        fetcher = self._graph_fetchers.get(layer)
        if fetcher is not None and fetcher.isRunning():
            return
        url = self._core_graph_url if layer == "core" else self._full_graph_url
        fetcher = _GraphFetcher(layer, url, parent=self)
        fetcher.fetched.connect(self._on_graph_fetched)
        fetcher.failed.connect(self._show_error)
        self._graph_fetchers[layer] = fetcher
        fetcher.start()

    def _on_graph_fetched(self, layer: str, graph: dict[str, Any]) -> None:
        graph = dict(graph) if isinstance(graph, dict) else {}
        cache_meta = graph.get("cache") if isinstance(graph.get("cache"), dict) else {}
        effective_layer = str(cache_meta.get("layer") or layer or "full").lower()
        requested_layer = str(cache_meta.get("requested_layer") or layer or effective_layer).lower()
        is_pending = bool(cache_meta.get("pending"))
        is_stale = bool(cache_meta.get("stale"))
        if requested_layer == "full" and is_pending:
            retry_ms = min(1800, 500 + self._full_graph_retry_count * 250)
            self._full_graph_retry_count = min(6, self._full_graph_retry_count + 1)
            QTimer.singleShot(retry_ms, lambda: self._start_graph_fetch("full"))
        elif requested_layer == "full":
            self._full_graph_retry_count = 0

        if requested_layer == "full" or not self._last_graph:
            self._last_graph = graph
        node_count = len(graph.get("nodes") or [])
        edge_count = len(graph.get("edges") or [])
        ts = time.strftime("%H:%M")
        layer_label = effective_layer.upper()
        if requested_layer != effective_layer:
            layer_label = f"{requested_layer.upper()}->{effective_layer.upper()}"
        flags = "".join(
            label for label, enabled in ((" STALE", is_stale), (" PENDING", is_pending)) if enabled
        )
        self._stats_label.setText(
            f"[{layer_label}{flags}  N:{node_count}  E:{edge_count}  @{ts}]"
        )

        if not self._graph_page_loaded:
            self._render_graph_html(graph)
        elif not self.isVisible():
            self._pending_graph_update = graph
        else:
            self._replace_graph_js(graph, fit=False)

        if requested_layer == "core":
            QTimer.singleShot(120, lambda: self._start_graph_fetch("full"))

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self._pending_graph_update and self._graph_page_loaded:
            graph = self._pending_graph_update
            self._pending_graph_update = {}
            QTimer.singleShot(0, lambda: self._replace_graph_js(graph, fit=False))

    def _render_graph_html(self, graph: dict[str, Any]) -> None:
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            assert isinstance(self._web_view, QWebEngineView)
            if hasattr(self._web_view, "load"):
                self._web_view.load(QUrl(self._core_graph_view_url))
            else:
                cytoscape_primary = self._base_url.rstrip("/") + _CYTOSCAPE_LOCAL_PATH
                html = _build_html(graph, self._base_url, cytoscape_primary, _CYTOSCAPE_FALLBACK)
                self._web_view.setHtml(html, QUrl(self._base_url + "/"))
            self._graph_page_loaded = True
        except Exception:
            pass

    def _replace_graph_js(self, graph: dict[str, Any], *, fit: bool = False, attempt: int = 0) -> None:
        if not self._web_available or self._web_view is None:
            return
        graph_json = json.dumps(graph if isinstance(graph, dict) else {})
        script = (
            "(function(){"
            "if (window._cvopsReplaceGraph) {"
            f"return window._cvopsReplaceGraph({graph_json}, {str(bool(fit)).lower()});"
            "}"
            "return false;"
            "})();"
        )
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            assert isinstance(self._web_view, QWebEngineView)

            def _done(result: Any) -> None:
                if result is False:
                    if attempt < 8:
                        QTimer.singleShot(
                            250,
                            lambda: self._replace_graph_js(graph, fit=fit, attempt=attempt + 1),
                        )
                    else:
                        self._render_graph_html(graph)

            self._web_view.page().runJavaScript(script, _done)
        except TypeError:
            try:
                self._web_view.page().runJavaScript(script)
            except Exception:
                self._render_graph_html(graph)
        except Exception:
            self._render_graph_html(graph)

    def refresh_theme_styles(self) -> None:
        if self._last_graph:
            self._graph_page_loaded = False
            self._render_graph_html(self._last_graph)
            QTimer.singleShot(120, lambda: self._start_graph_fetch("full"))
        else:
            self.reload()

    def _show_error(self, layer: str, message: str) -> None:
        if not self._web_available or self._web_view is None:
            return
        if self._graph_page_loaded and str(layer or "").lower() == "full":
            self._stats_label.setText(f"[FULL ERROR  {_short(message, 80)}]")
            return
        html = (
            f"<html><body style='background:transparent;color:{cvops_color('text_iron')};"
            f"background-image:linear-gradient(180deg,{_ontology_accent_wash()} 0%,transparent 28%);"
            f"font-family:JetBrains Mono,monospace;font-size:11px;padding:40px;'>"
            f"[ONTOLOGY] Failed to load graph data<br><br>{message}</body></html>"
        )
        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView

            assert isinstance(self._web_view, QWebEngineView)
            self._web_view.setHtml(html)
            self._graph_page_loaded = False
        except Exception:
            pass

    def _on_navigation(self, request: Any = None) -> None:
        if request is None:
            return
        try:
            url = request.url().toString()
        except AttributeError:
            url = request.requestUrl().toString()

        if url.startswith("appbridge://reload"):
            try:
                request.reject()
            except Exception:
                pass
            QTimer.singleShot(50, self.reload)
            return

        if url.startswith("appbridge://goto/"):
            raw = url[len("appbridge://goto/"):]
            # Split off optional ?scenario=... query.
            path_part, _, query_part = raw.partition("?")
            parts = path_part.split("/", 1)
            target = urllib.parse.unquote(parts[0]) if parts else ""
            focus_id = urllib.parse.unquote(parts[1]) if len(parts) == 2 else ""
            scenario_hint = ""
            if query_part:
                qs = urllib.parse.parse_qs(query_part)
                scenario_hint = (qs.get("scenario") or [""])[0]
            if target:
                self.navigateRequested.emit(target, focus_id, scenario_hint)
            try:
                request.reject()
            except Exception:
                pass
            return

        if url.startswith("appbridge://inspect/"):
            path = url[len("appbridge://inspect/"):]
            parts = path.split("/", 1)
            if len(parts) == 2:
                etype = urllib.parse.unquote(parts[0])
                eid = urllib.parse.unquote(parts[1])
                self._inspect_entity(etype, eid)
            try:
                request.reject()
            except Exception:
                pass
            return

        if url.startswith("appbridge://entity/"):
            path = url[len("appbridge://entity/"):]
            parts = path.split("/", 1)
            if len(parts) == 2:
                etype = urllib.parse.unquote(parts[0])
                eid = urllib.parse.unquote(parts[1])
                self._inspect_entity(etype, eid)
                self.entitySelected.emit(etype, eid)
            try:
                request.reject()
            except Exception:
                pass
            return

        if url.startswith("appbridge://path/"):
            raw = url[len("appbridge://path/"):]
            parts = raw.split("/", 1)
            if len(parts) == 2:
                from_id = urllib.parse.unquote(parts[0])
                to_id = urllib.parse.unquote(parts[1])
                QTimer.singleShot(0, lambda: self._run_ecosystem_path(from_id, to_id))
            try:
                request.reject()
            except Exception:
                pass
            return

        if url.startswith("appbridge://impact/"):
            raw = url[len("appbridge://impact/"):]
            entity_id = urllib.parse.unquote(raw)
            QTimer.singleShot(0, lambda: self._run_ecosystem_impact(entity_id))
            try:
                request.reject()
            except Exception:
                pass
            return

    def _run_ecosystem_path(self, from_id: str, to_id: str) -> None:
        try:
            fi = urllib.parse.quote(from_id, safe="")
            ti = urllib.parse.quote(to_id, safe="")
            data = self._get_json(f"/ecosystem/path?from_id={fi}&to_id={ti}", timeout=5.0)
            path_nodes = data.get("path") or []
            path_edges = data.get("edges") or []
            edge_pairs = [[e.get("source", ""), e.get("target", "")] for e in path_edges]
            self._run_js(
                f"if(window._cyHighlightPath) window._cyHighlightPath("
                f"{json.dumps(path_nodes)}, {json.dumps(edge_pairs)});"
            )
        except Exception as exc:
            self._run_js(
                f"if(window._showToast) window._showToast('Path error: {str(exc)[:80]}', 'error');"
            )
        finally:
            if self._path_btn.isChecked():
                self._path_btn.setChecked(False)

    def _run_ecosystem_impact(self, entity_id: str) -> None:
        try:
            eid_q = urllib.parse.quote(entity_id, safe="")
            data = self._get_json(f"/ecosystem/impact/{eid_q}", timeout=5.0)
            upstream = data.get("upstream") or []
            downstream = data.get("downstream") or []
            self._run_js(
                f"if(window._cyShowImpact) window._cyShowImpact("
                f"{json.dumps(upstream)}, {json.dumps(downstream)}, {json.dumps(entity_id)});"
            )
        except Exception as exc:
            self._run_js(
                f"if(window._showToast) window._showToast('Impact error: {str(exc)[:80]}', 'error');"
            )

    def _run_js(self, script: str) -> None:
        if not self._web_available or self._web_view is None:
            return
        try:
            self._web_view.page().runJavaScript(script)
        except Exception:
            pass

    def _on_search(self, text: str) -> None:
        safe = json.dumps(text.strip())
        self._run_js(f"if(window._cySetSearch) window._cySetSearch({safe});")

    def _on_filter_toggled(self, entity_type: str, checked: bool) -> None:
        if checked:
            if entity_type not in self._active_filters:
                self._active_filters.append(entity_type)
        else:
            self._active_filters = [t for t in self._active_filters if t != entity_type]

        active = self._active_filters if self._active_filters else []
        self._run_js(
            f"if(window._cySetFilter) window._cySetFilter({json.dumps(active)});"
        )

    def _on_path_mode_toggled(self, checked: bool) -> None:
        self._run_js(
            f"if(window._cyActivatePathMode) window._cyActivatePathMode({json.dumps(checked)});"
        )

    def _on_clear_impact(self) -> None:
        self._run_js("if(window._cyClearHighlight) window._cyClearHighlight();")
        if self._path_btn.isChecked():
            self._path_btn.setChecked(False)

    def _fit_graph(self) -> None:
        self._run_js("if(window._cyFitAll) window._cyFitAll();")

    def _toggle_legend(self, checked: bool) -> None:
        if checked:
            self._run_js("if(window._cyShowLegend) window._cyShowLegend();")
        else:
            self._run_js("if(window._cyHideLegend) window._cyHideLegend();")
