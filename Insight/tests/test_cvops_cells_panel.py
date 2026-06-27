from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

try:
    from PyQt6.QtWidgets import QApplication, QLabel, QPushButton
except Exception:
    QApplication = None  # type: ignore[assignment]
    QLabel = None  # type: ignore[assignment]
    QPushButton = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class CellsPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_cells_panel_groups_scenarios_and_loads_custom_drafts(self) -> None:
        from insight_local.cvops.ui.cells_panel import CellsPanel

        def fake_get(path: str) -> dict:
            if path == "/scenarios/cc_demo/custom_cells":
                return {
                    "scenario": "cc_demo",
                    "cells": [
                        {
                            "name": "Draft Prep",
                            "path": "mlops/custom_cells/cc_demo/draft/cell_draft_prep.py",
                            "entry": "run",
                            "datasets": [{"name": "notes", "kind": "inline_text", "path": "mlops/custom_cells/cc_demo/draft/data/draft_prep/notes.txt"}],
                            "code": "def run(ctx, prev):\n    return {'data': {'ready': True}}\n",
                        }
                    ],
                    "scenario_datasets": [
                        {"name": "primary", "kind": "folder", "path": "database/custom", "format": "yolo", "mode": "reference"}
                    ],
                }
            return {}

        panel = CellsPanel(http_get=fake_get)
        try:
            panel.set_scenarios(
                [
                    {
                        "name": "vision_demo",
                        "display_name": "Vision Demo",
                        "dataset": "database/vision",
                        "status": "trained",
                        "dataset_count": 10,
                        "history_count": 2,
                        "backbone_type": "yolo_detection",
                        "backbone_config": {},
                    },
                    {
                        "name": "tab_demo",
                        "display_name": "Tab Demo",
                        "dataset": "mlops/datasets/signals.csv",
                        "status": "dataset",
                        "dataset_count": 3,
                        "history_count": 0,
                        "backbone_type": "torch_tabular",
                        "backbone_config": {
                            "cells": [{"path": "mlops/algos/tabular_cell_template.py"}],
                        },
                    },
                    {
                        "name": "cc_demo",
                        "display_name": "Custom Demo",
                        "dataset": "database/custom",
                        "status": "empty",
                        "dataset_count": 0,
                        "history_count": 0,
                        "backbone_type": "custom_code",
                        "backbone_config": {
                            "cells": [{"path": "mlops/algos/mytpl__c1.py"}],
                        },
                    },
                ]
            )

            self.assertEqual(panel._tabs.tabText(0), "Vision (1)")
            self.assertEqual(panel._tabs.tabText(1), "Tabular (1)")
            self.assertEqual(panel._tabs.tabText(4), "Custom (1)")

            custom_labels = [
                label.text()
                for label in panel._pages["custom"].findChildren(QLabel)
                if isinstance(label.text(), str)
            ]
            custom_blob = "\n".join(custom_labels)
            self.assertIn("Draft Cells", custom_blob)
            self.assertIn("Draft Prep", custom_blob)
            self.assertIn("Draft Scenario Datasets", custom_blob)
            self.assertIn("database/custom", custom_blob)

            if QPushButton is not None:
                btns = panel._pages["custom"].findChildren(QPushButton)
                texts = {b.text() for b in btns}
                self.assertIn("Open in Workflow (Train)", texts)
        finally:
            panel.deleteLater()

    def test_cells_panel_can_save_and_run_custom_draft(self) -> None:
        from insight_local.cvops.ui.cells_panel import CellsPanel

        saved: list[dict] = []
        posts: list[tuple[str, dict | None]] = []

        def fake_get(path: str) -> dict:
            if path == "/scenarios/cc_demo/custom_cells":
                return {"scenario": "cc_demo", "cells": [], "scenario_datasets": []}
            return {}

        def fake_put(path: str, body: dict | None) -> dict:
            self.assertEqual(path, "/scenarios/cc_demo/custom_cells")
            payload = body or {}
            saved.append(payload)
            cells = []
            for cell in payload.get("cells") or []:
                cells.append(
                    {
                        **cell,
                        "path": f"mlops/custom_cells/cc_demo/draft/cell_{cell.get('id')}.py",
                    }
                )
            return {
                "scenario": "cc_demo",
                "cells": cells,
                "scenario_datasets": payload.get("scenario_datasets") or [],
            }

        def fake_post(path: str, body: dict | None) -> dict:
            posts.append((path, body))
            return {"job_id": "job-cells-1", "state": "queued"}

        panel = CellsPanel(http_get=fake_get, http_put=fake_put, http_post=fake_post)
        try:
            panel.set_scenarios(
                [
                    {
                        "name": "cc_demo",
                        "display_name": "Custom Demo",
                        "dataset": "database/custom",
                        "status": "empty",
                        "dataset_count": 0,
                        "history_count": 0,
                        "backbone_type": "custom_code",
                        "backbone_config": {},
                    }
                ]
            )

            editor = panel._editor
            self.assertFalse(editor.isHidden())
            editor.add_cell()
            editor._cell_name.setText("prep")
            editor._code.setPlainText("def run(ctx, prev):\n    return {'data': {'ok': True}}\n")
            editor.save_draft()

            self.assertEqual(len(saved), 1)
            self.assertEqual(saved[0]["cells"][0]["name"], "prep")

            editor.run_draft()
            self.assertEqual(posts[0][0], "/scenarios/cc_demo/train")
            override = posts[0][1]["backbone_config_override"]  # type: ignore[index]
            self.assertEqual(override["cells"][0]["name"], "prep")
            self.assertIn("mlops/custom_cells/cc_demo/draft/cell_", override["cells"][0]["path"])
        finally:
            panel.deleteLater()


class CvOpsWindowCellsTabTests(unittest.TestCase):
    def test_cells_tab_is_before_notifications_and_three_d(self) -> None:
        window_path = ROOT / "Insight" / "insight_local" / "cvops" / "window.py"
        text = window_path.read_text(encoding="utf-8")
        pos_notifications = text.index("self._tab_notifications")
        pos_diagnostics = text.index("self._tab_diagnostics")
        pos_three_d = text.index("self._tab_three_d")
        pos_cells = text.index("self._tab_cells")
        pos_scrape = text.index("self._tab_scrape")

        self.assertLess(pos_cells, pos_notifications)
        self.assertLess(pos_notifications, pos_diagnostics)
        self.assertLess(pos_diagnostics, pos_three_d)
        self.assertLess(pos_cells, pos_scrape)

    def test_database_tab_has_workspace_sections_for_scenario_and_database(self) -> None:
        window_path = ROOT / "Insight" / "insight_local" / "cvops" / "window.py"
        text = window_path.read_text(encoding="utf-8")

        self.assertIn("self._database_workspace_tabs = QTabWidget()", text)
        self.assertIn("self._db_workspace_tab_scenario = self._database_workspace_tabs.addTab(", text)
        self.assertIn('"Scenario"', text)
        self.assertIn("self._db_workspace_tab_database = self._database_workspace_tabs.addTab(", text)
        self.assertIn('"DataBase"', text)

    def test_pannel_is_sidebar_split_and_tabs_follow_requested_order(self) -> None:
        window_path = ROOT / "Insight" / "insight_local" / "cvops" / "window.py"
        text = window_path.read_text(encoding="utf-8")

        self.assertIn('self._pannel_toggle_btn = QPushButton("Pannel")', text)
        self.assertIn('self._workspace_splitter = QSplitter(Qt.Orientation.Horizontal)', text)
        self.assertNotIn("side_rail = QWidget()", text)
        self.assertIn("status_row.addWidget(self._pannel_toggle_btn)", text)
        self.assertNotIn('self._tab_pannel = self._tabs.addTab(self._build_pannel_tab(), "Pannel")', text)
        self.assertIn('self._pannel_nav = QListWidget()', text)
        self.assertIn('self._pannel_nav.setObjectName("pannelSideNav")', text)
        self.assertIn('("Scenario", "Open Workbench > Scenario Creation.", "workflow_scenario")', text)
        self.assertIn('("DataBase", "Open Database > DataBase god-view.", "database_view")', text)
        self.assertIn('if route == "database_view":', text)
        self.assertIn("page = self._build_pannel_database_page()", text)
        self.assertIn("self._pannel_database_godview_panel = DatabaseGodViewPanel(", text)
        self.assertIn('self._ensure_pannel_sidebar_width(680)', text)
        self.assertIn('("Range", "Open Range workspace.", "test_range")', text)
        self.assertIn('self._tab_portal = self._tabs.addTab(self._build_portal_tab(), "Scope")', text)

        order = [
            "self._tab_ecosystem = self._tabs.addTab(",
            'self._tab_workflow = self._tabs.addTab(self._build_workflow_tab(), "Workbench")',
            'self._tab_test_range = self._tabs.addTab(self._build_test_range_tab(), "Range")',
            'self._tab_portal = self._tabs.addTab(self._build_portal_tab(), "Scope")',
            "self._tab_cells = self._tabs.addTab(",
            "self._tab_notifications = self._tabs.addTab(",
            'self._tab_diagnostics = self._tabs.addTab(self._build_diagnostics_tab(), "Diagnostics")',
            "self._tab_three_d = self._tabs.addTab(",
            "self._tab_scrape = self._tabs.addTab(",
            'self._tab_database = self._tabs.addTab(self._build_database_tab(), "Database")',
        ]
        positions = [text.index(marker) for marker in order]
        self.assertEqual(positions, sorted(positions))

    def test_event_pulse_has_setting_and_opens_notifications(self) -> None:
        window_path = ROOT / "Insight" / "insight_local" / "cvops" / "window.py"
        window_text = window_path.read_text(encoding="utf-8")
        settings_path = ROOT / "Insight" / "insight_local" / "cvops" / "ui" / "settings_panel.py"
        settings_text = settings_path.read_text(encoding="utf-8")
        pulse_path = ROOT / "Insight" / "insight_local" / "cvops" / "ui" / "event_pulse_widget.py"
        pulse_text = pulse_path.read_text(encoding="utf-8")

        self.assertIn("show_event_pulse: bool = True", settings_text)
        self.assertIn('QCheckBox("Show scrolling notification bar")', settings_text)
        self.assertIn("eventPulseVisibilityChanged = pyqtSignal(bool)", settings_text)
        self.assertIn("self._event_pulse.setVisible(bool(self._cvops_settings.show_event_pulse))", window_text)
        self.assertIn("self._event_pulse.openNotificationsRequested.connect(self._open_notifications_center)", window_text)
        self.assertIn("def _open_notifications_center(self) -> None:", window_text)
        self.assertIn("self._tabs.setCurrentIndex(self._tab_notifications)", window_text)
        self.assertIn("openNotificationsRequested = pyqtSignal()", pulse_text)
        self.assertIn("def mouseDoubleClickEvent(self, event) -> None:", pulse_text)


if __name__ == "__main__":
    unittest.main()
