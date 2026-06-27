from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .archive_viz_panel import ArchiveVizPanel
from .data_viz_panel import DataVizPanel as TabularDataVizPanel
from .scenario_flow_view import ScenarioFlowView


class DataVizHub(QWidget):
    """Host both tabular and archival visualization surfaces."""

    def __init__(
        self,
        *,
        http_get: Optional[Callable[[str], dict[str, Any]]] = None,
        http_post: Optional[Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]] = None,
        get_import_progress: Optional[Callable[[str], Optional[dict[str, Any]]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, stretch=1)

        self._tabular = TabularDataVizPanel(self)
        self._archival = ArchiveVizPanel(http_get=http_get, http_post=http_post, get_import_progress=get_import_progress, parent=self)
        self._flow = ScenarioFlowView(self)
        self._tabs.addTab(self._tabular, "Tabular")
        self._tabs.addTab(self._archival, "Archival")
        self._tabs.addTab(self._flow, "Flow")

    def show_tabular(self) -> None:
        self._tabs.setCurrentWidget(self._tabular)

    def show_archival(self) -> None:
        self._tabs.setCurrentWidget(self._archival)

    def show_flow(self) -> None:
        self._tabs.setCurrentWidget(self._flow)

    def set_flow(
        self,
        scenario: str,
        dataset: str,
        steps: Sequence[tuple[str, str, Sequence[str], str]],
    ) -> None:
        """Update the native Flow ecosystem diagram for the current scenario.

        Does not steal tab focus -- the user opts into Flow via the tab."""
        self._flow.set_flow(scenario, dataset, steps)

    def set_scenario_csv(self, csv_rel: str, root_dir: str) -> None:
        self.show_tabular()
        self._tabular.set_scenario_csv(csv_rel, root_dir)

    def set_csv_path(self, path: Path, *, max_rows: int = 100000) -> None:
        self.show_tabular()
        self._tabular.set_csv_path(path, max_rows=max_rows)

    def set_data_source_path(self, path: Path, *, max_rows: int = 20000) -> None:
        self.show_tabular()
        self._tabular.set_data_source_path(path, max_rows=max_rows)

    def set_archive_context(
        self,
        corpus_id: str = "",
        dataset_version_id: str = "",
        snapshot_id: str = "",
        scenario: str = "",
    ) -> None:
        self.show_archival()
        self._archival.set_archive_context(
            corpus_id=corpus_id,
            dataset_version_id=dataset_version_id,
            snapshot_id=snapshot_id,
            scenario=scenario,
        )

    def clear(self) -> None:
        self._tabular.clear()
        self._archival.clear()
        self._flow.set_flow("", "", [])
        self.show_tabular()
