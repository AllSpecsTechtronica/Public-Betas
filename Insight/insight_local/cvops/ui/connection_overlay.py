"""ConnectionOverlay — transparent full-window overlay for cross-panel bezier lines.

Installed as a transparent child of QMainWindow, covering the full window.
When a SelectablePanel emits ``entitySelected``, the overlay fetches the entity's
direct edges from ``GET /ontology/entity/{type}/{id}`` and draws animated bezier
lines between the source panel and every registered panel whose entity type
appears in the edge list.

Usage (in window.py)::

    overlay = ConnectionOverlay(base_url=self.base_url, parent=self)
    overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
    overlay.raise_()

    # Register each panel by its primary entity type.
    overlay.register_panel("scenario", catalog_list_widget)
    overlay.register_panel("dataset", database_panel)
    overlay.register_panel("lineage", lineage_panel)

    # Wire every selectable panel.
    catalog_panel.entitySelected.connect(overlay.draw_connections)
    database_panel.entitySelected.connect(overlay.draw_connections)
    lineage_panel.entitySelected.connect(overlay.draw_connections)

    # Keep overlay covering the full window.
    # In CvOpsWindow.resizeEvent:
    #   overlay.resize(self.size())
"""
from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

from PyQt6.QtCore import QPointF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

from .cvops_theme import cvops_color

# Map ontology edge types to line colors.  Uses the same palette as the
# Cytoscape.js ontology graph so scientists get consistent visual encoding.
_EDGE_COLOR_ROLE: dict[str, str] = {
    "belongs_to":    "line_light",
    "uses_backbone": "line_light",
    "contains_cell": "line_light",
    "governed_by":   "accent_active",
    "trains_on":     "accent_active",
    "produces":      "accent_select",
    "evaluated_by":  "accent_warn",
    "evaluates":     "accent_warn",
    "branched_from": "line_med",
    "derived_from":  "line_med",
    "has_head":      "accent_select",
}
_COLOR_FALLBACK_ROLE = "accent_select"

# Fade animation: target opacity 0.40, step size yields ~200 ms at 16 ms ticks.
_TARGET_OPACITY = 0.40
_FADE_STEP = 0.025
_TICK_MS = 16


class _FetchWorker(QThread):
    """Background thread — fetches entity edges without blocking the UI thread."""

    fetched: pyqtSignal = pyqtSignal(dict)

    def __init__(self, url: str, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._url = url

    def run(self) -> None:
        try:
            with urllib.request.urlopen(self._url, timeout=5) as resp:  # noqa: S310
                data = json.loads(resp.read())
            self.fetched.emit(data if isinstance(data, dict) else {})
        except Exception:
            self.fetched.emit({})


@dataclass
class _Line:
    start: QPointF
    end: QPointF
    color_role: str


class ConnectionOverlay(QWidget):
    """Transparent full-window overlay that draws animated bezier entity-connection lines.

    Install on QMainWindow.  The overlay is completely click-transparent
    (``WA_TransparentForMouseEvents``) so it never interferes with panel
    interaction.  Lines fade in over ~200 ms and are cleared automatically
    when ``clear()`` is called or when a new ``draw_connections()`` request
    supersedes the previous one.
    """

    def __init__(self, *, base_url: str, parent: QWidget) -> None:
        super().__init__(parent)
        self._base_url = str(base_url).rstrip("/")

        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("background: transparent;")
        # Keep the overlay out of the hit-test stack unless it is actively
        # drawing. On macOS, transparent full-window widgets can still become
        # the top native view after layout/native-view churn, which makes
        # underlying page controls look dead even though they are enabled.
        self.hide()

        # entity_type → QWidget that currently displays entities of that type.
        self._panel_registry: dict[str, QWidget] = {}

        self._lines: list[_Line] = []
        self._opacity: float = 0.0
        self._fetch_worker: Optional[_FetchWorker] = None

        self._fade_timer = QTimer(self)
        self._fade_timer.setInterval(_TICK_MS)
        self._fade_timer.timeout.connect(self._tick_fade)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_panel(self, entity_type: str, widget: QWidget) -> None:
        """Register *widget* as the primary display surface for *entity_type*."""
        self._panel_registry[entity_type] = widget

    def draw_connections(self, entity_type: str, entity_id: str) -> None:
        """Fetch direct edges for the selected entity and draw animated bezier lines.

        Cancels any in-flight request before starting a new one.  Lines appear
        only when related panels are currently visible in the registry.
        """
        self.clear()
        if not entity_type or not entity_id:
            return
        url = f"{self._base_url}/ontology/entity/{entity_type}/{entity_id}"
        self._cancel_fetch()
        worker = _FetchWorker(url, parent=self)
        worker.fetched.connect(
            lambda data: self._on_fetched(entity_type, data)
        )
        self._fetch_worker = worker
        worker.start()

    def clear(self) -> None:
        """Remove all connection lines immediately."""
        self._fade_timer.stop()
        self._lines.clear()
        self._opacity = 0.0
        self.hide()
        self.update()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _cancel_fetch(self) -> None:
        w = self._fetch_worker
        if w is not None and w.isRunning():
            w.terminate()
            w.wait(200)
        self._fetch_worker = None

    def _on_fetched(self, source_type: str, data: dict[str, Any]) -> None:
        source_widget = self._panel_registry.get(source_type)
        if source_widget is None or not source_widget.isVisible():
            return

        source_center = self._widget_center(source_widget)
        edges: list[dict[str, Any]] = data.get("edges", [])
        new_lines: list[_Line] = []
        seen_targets: set[str] = set()

        for edge in edges:
            # Edges are "{type}:{id}" strings — extract the type prefix.
            src_full = str(edge.get("source") or "")
            tgt_full = str(edge.get("target") or "")
            edge_kind = str(edge.get("type") or "")

            src_type = src_full.split(":")[0] if ":" in src_full else src_full
            tgt_type = tgt_full.split(":")[0] if ":" in tgt_full else tgt_full

            # Identify which end is "the other" entity type.
            other_type = tgt_type if src_type == source_type else src_type
            if not other_type or other_type == source_type:
                continue
            if other_type in seen_targets:
                continue

            target_widget = self._panel_registry.get(other_type)
            if target_widget is None or not target_widget.isVisible():
                continue

            color_role = _EDGE_COLOR_ROLE.get(edge_kind, _COLOR_FALLBACK_ROLE)
            new_lines.append(
                _Line(
                    start=source_center,
                    end=self._widget_center(target_widget),
                    color_role=color_role,
                )
            )
            seen_targets.add(other_type)

        if not new_lines:
            return
        self._lines = new_lines
        self._opacity = 0.0
        self.show()
        self.raise_()
        self._fade_timer.start()

    def _widget_center(self, widget: QWidget) -> QPointF:
        """Return *widget*'s center in this overlay's local coordinate space."""
        global_pt = widget.mapToGlobal(widget.rect().center())
        local_pt = self.mapFromGlobal(global_pt)
        return QPointF(local_pt)

    def _tick_fade(self) -> None:
        self._opacity = min(_TARGET_OPACITY, self._opacity + _FADE_STEP)
        if self._opacity >= _TARGET_OPACITY:
            self._fade_timer.stop()
        self.update()

    # ------------------------------------------------------------------
    # Paint
    # ------------------------------------------------------------------

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if not self._lines or self._opacity <= 0.0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setOpacity(self._opacity)
        for line in self._lines:
            pen = QPen(QColor(cvops_color(line.color_role)), 1.5)
            painter.setPen(pen)
            path = QPainterPath()
            path.moveTo(line.start)
            # S-curve: control points share the horizontal midpoint so the
            # bezier exits horizontally from both source and target panels.
            mid_x = (line.start.x() + line.end.x()) * 0.5
            path.cubicTo(
                QPointF(mid_x, line.start.y()),
                QPointF(mid_x, line.end.y()),
                line.end,
            )
            painter.drawPath(path)
        painter.end()
