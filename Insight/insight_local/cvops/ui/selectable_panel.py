"""SelectablePanel — mixin for CV Ops panels that expose entity selection.

Any panel that lets the user focus on a specific entity (model version, job,
lineage, snapshot, catalog asset, etc.) should inherit this mixin alongside
QWidget and emit ``entitySelected`` when the focused entity changes.

The ConnectionOverlay subscribes to this signal from every registered panel
and draws bezier lines between related entities.

Usage::

    class CatalogPanel(SelectablePanel, QWidget):
        def _on_row_selected(self, asset_id: str) -> None:
            self.entitySelected.emit("catalog_asset", asset_id)

The ``panel_entity_type`` class attribute declares which entity type this
panel primarily shows (optional — used by the overlay for edge filtering).
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import pyqtSignal


class SelectablePanel:
    """Plain Python mixin for CV Ops panels that expose entity selection.

    Do NOT inherit from QObject here — all panels already inherit from QWidget
    which provides the Qt metaclass. The pyqtSignal declared on this class will
    be picked up automatically when the concrete panel is instantiated.

    Usage::

        class CatalogPanel(SelectablePanel, QWidget):
            panel_entity_type = "scenario"

            def _on_row_selected(self, scenario_name: str) -> None:
                self.entitySelected.emit("scenario", scenario_name)

    The ``entitySelected`` signal is then available on every panel instance
    for the ConnectionOverlay and any other subscriber.
    """

    # (entity_type, entity_id) — emitted whenever a row / item is selected
    entitySelected = pyqtSignal(str, str)

    # Declare the primary entity type this panel primarily shows.
    panel_entity_type: str = ""

    def emit_entity_selected(self, entity_type: str, entity_id: str) -> None:
        """Helper for subclasses — avoids calling .emit() on an unbound signal."""
        self.entitySelected.emit(str(entity_type), str(entity_id))
