from __future__ import annotations

import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Insight"))

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QLabel, QScrollArea, QSizePolicy, QSplitter
except Exception:  # pragma: no cover - allows non-Qt test environments to skip cleanly.
    QApplication = None  # type: ignore[assignment]
    QLabel = None  # type: ignore[assignment]
    QScrollArea = None  # type: ignore[assignment]
    QSplitter = None  # type: ignore[assignment]
    QSizePolicy = None  # type: ignore[assignment]
    Qt = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class CvOpsTrainingFlowUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _fake_get(self, path: str) -> dict:
        if path == "/models":
            return {
                "models": [
                    {"name": "YOLO Nano", "value": "assets/models/yolo26n.pt", "size_bytes": 1024},
                    {"name": "YOLO Small", "value": "assets/models/yolo26s.pt", "size_bytes": 2048},
                ]
            }
        if path == "/database":
            return {
                "datasets": ["Donut Defects", "People", "AudioRecognition"],
                "categories": {"Donut Defects": "image", "People": "image", "AudioRecognition": "audio"},
                "tabular_datasets": [{"filename": "signals.csv", "path": "mlops/datasets/signals.csv"}],
                "text_datasets": [{"filename": "instructions.jsonl", "path": "mlops/datasets/instructions.jsonl"}],
                "audio_root": "/tmp/assets/ml_audio",
            }
        if path == "/audio/assets":
            return {
                "root": "/tmp/assets/ml_audio",
                "count": 2,
                "items": [
                    {
                        "name": "frenchpeoplewalkinglong.mp4",
                        "relative_path": "frenchpeoplewalkinglong.mp4",
                        "path": "/tmp/assets/ml_audio/frenchpeoplewalkinglong.mp4",
                        "size": 4096,
                        "split": "source",
                        "classification_label": "",
                        "training_ready": False,
                    },
                    {
                        "name": "a.wav",
                        "relative_path": "AudioRecognition/train/alarm/a.wav",
                        "path": "/tmp/assets/ml_audio/AudioRecognition/train/alarm/a.wav",
                        "size": 1024,
                        "split": "train",
                        "classification_label": "alarm",
                        "training_ready": True,
                    },
                ],
            }
        if path == "/database/Donut%20Defects":
            return {
                "slug": "Donut Defects",
                "path": "/tmp/database/Donut Defects",
                "format": "yolo_detection",
                "category": "image",
                "count": 42,
                "classes": ["good", "scratch"],
                "split_counts": {"train": 40, "val": 2},
                "content_sha256": "a" * 64,
            }
        if path == "/database/People":
            return {
                "slug": "People",
                "path": "/tmp/database/People",
                "format": "imagefolder_classification",
                "category": "image",
                "count": 12,
                "classes": ["person"],
                "content_sha256": "b" * 64,
            }
        if path == "/database/AudioRecognition":
            return {
                "slug": "AudioRecognition",
                "path": "/tmp/assets/ml_audio/AudioRecognition",
                "format": "audiofolder_classification",
                "category": "audio",
                "count": 2,
                "classes": ["alarm", "speech"],
                "split_counts": {"train": 1, "val": 1},
                "audio_files": [
                    {
                        "name": "a.wav",
                        "relative_path": "train/alarm/a.wav",
                        "display_name": "train/alarm/a.wav",
                        "split": "train",
                        "size": 1024,
                        "has_label": True,
                        "classification_label": "alarm",
                    },
                    {
                        "name": "b.wav",
                        "relative_path": "val/speech/b.wav",
                        "display_name": "val/speech/b.wav",
                        "split": "val",
                        "size": 2048,
                        "has_label": True,
                        "classification_label": "speech",
                    },
                ],
            }
        if path == "/database/InstructionData":
            return {
                "slug": "InstructionData",
                "path": "/tmp/database/InstructionData",
                "format": "llm_instruction_jsonl",
                "category": "text",
                "count": 3,
                "classes": [],
                "text_files": [{"name": "instructions.jsonl", "row_count": 3}],
            }
        if path.endswith("/cards"):
            return {"model_card": "", "dataset_card": ""}
        if path.endswith("/history"):
            return {"runs": [], "count": 0}
        return {}

    def test_hyperparam_panel_absent_fields_start_clean_until_edited(self) -> None:
        from insight_local.cvops.ui.hyperparam_suite_panel import HyperparamSuitePanel

        schema = {
            "epochs": {"kind": "int", "min": 1, "max": 300},
            "lr0": {"kind": "float", "min": 0.0, "max": 1.0},
        }
        panel = HyperparamSuitePanel()
        try:
            panel.load({}, schema)

            self.assertEqual(panel.current_values(dirty_only=True), {})
            self.assertFalse(panel._save_btn.isEnabled())
            self.assertFalse(panel._reset_btn.isEnabled())

            epochs = panel._fields["epochs"]
            epochs.widget.setValue(2)

            self.assertEqual(panel.current_values(dirty_only=True), {"epochs": 2})
            self.assertTrue(panel._save_btn.isEnabled())
            self.assertTrue(panel._reset_btn.isEnabled())
        finally:
            panel.deleteLater()

    def test_hyperparam_panel_saved_fields_can_return_to_clean(self) -> None:
        from insight_local.cvops.ui.hyperparam_suite_panel import HyperparamSuitePanel

        schema = {"epochs": {"kind": "int", "min": 1, "max": 300}}
        panel = HyperparamSuitePanel()
        try:
            panel.load({"epochs": 20}, schema)

            self.assertEqual(panel.current_values(dirty_only=True), {})
            panel._fields["epochs"].widget.setValue(21)
            self.assertEqual(panel.current_values(dirty_only=True), {"epochs": 21})
            panel._fields["epochs"].widget.setValue(20)
            self.assertEqual(panel.current_values(dirty_only=True), {})
            self.assertFalse(panel._save_btn.isEnabled())
        finally:
            panel.deleteLater()

    def test_hyperparam_panel_renders_quality_stop_fields_and_saves(self) -> None:
        from insight_local.cvops.ui.hyperparam_suite_panel import HyperparamSuitePanel

        schema = {
            "quality_stop_enabled": {"kind": "bool"},
            "quality_stop_metric": {
                "kind": "str_choices",
                "choices": ["map50_95", "map50", "precision", "recall"],
            },
            "quality_stop_threshold": {"kind": "float", "min": 0.0, "max": 1.0},
            "quality_stop_min_epochs": {"kind": "int", "min": 1, "max": 100000},
            "quality_stop_consecutive_epochs": {"kind": "int", "min": 1, "max": 100000},
        }
        panel = HyperparamSuitePanel()
        try:
            panel.load(
                {
                    "quality_stop_enabled": True,
                    "quality_stop_metric": "map50_95",
                    "quality_stop_threshold": 0.90,
                    "quality_stop_min_epochs": 5,
                    "quality_stop_consecutive_epochs": 2,
                },
                schema,
            )

            self.assertEqual(set(panel._fields), set(schema))
            self.assertEqual(panel.current_values(dirty_only=True), {})

            panel._fields["quality_stop_enabled"].widget.setChecked(False)
            panel._fields["quality_stop_threshold"].widget.setValue(0.92)
            metric_widget = panel._fields["quality_stop_metric"].widget
            metric_widget.setCurrentIndex(metric_widget.findData("map50"))

            self.assertEqual(
                panel.current_values(dirty_only=True),
                {
                    "quality_stop_enabled": False,
                    "quality_stop_metric": "map50",
                    "quality_stop_threshold": 0.92,
                },
            )
        finally:
            panel.deleteLater()

    def test_scroll_wrapped_pages_do_not_attach_to_window_until_inserted(self) -> None:
        from insight_local.cvops.window import CvOpsWindow

        page = QScrollArea()
        wrapper = CvOpsWindow._wrap_scroll_page(None, page)  # type: ignore[arg-type]
        try:
            self.assertIsNone(wrapper.parent())
            self.assertIsNotNone(page.parent())
        finally:
            wrapper.deleteLater()

    def test_ws_refresh_button_is_separate_and_always_enabled(self) -> None:
        window_text = (ROOT / "Insight" / "insight_local" / "cvops" / "window.py").read_text(
            encoding="utf-8"
        )
        theme_text = (
            ROOT / "Insight" / "insight_local" / "cvops" / "ui" / "cvops_theme.py"
        ).read_text(encoding="utf-8")

        self.assertIn('self._ws_refresh_btn = QToolButton()', window_text)
        self.assertIn('self._ws_refresh_btn.setObjectName("wsRefreshButton")', window_text)
        self.assertIn('self._ws_refresh_btn.setText("↻")', window_text)
        self.assertIn("self._ws_refresh_btn.setEnabled(True)", window_text)
        self.assertIn("self._ws_refresh_btn.clicked.connect(self._refresh_ws_output)", window_text)
        self.assertIn("self._top_nav_row.addWidget(\n            self._ws_refresh_btn", window_text)
        self.assertIn("def _refresh_ws_output(self) -> None:", window_text)
        self.assertIn("self._ws.reconnect_now()", window_text)
        self.assertIn("self._start_ws_resync(force=True)", window_text)
        self.assertIn("QToolButton#wsRefreshButton", theme_text)

    def test_bottom_pane_toggle_button_is_wired_in_top_nav(self) -> None:
        window_text = (ROOT / "Insight" / "insight_local" / "cvops" / "window.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('self._bottom_pane_toggle_btn = QPushButton("Bottom pane")', window_text)
        self.assertIn("self._bottom_pane_toggle_btn.setCheckable(True)", window_text)
        self.assertIn("self._bottom_pane_toggle_btn.clicked.connect(self._on_bottom_pane_toggle_clicked)", window_text)
        self.assertIn("self._top_nav_row.addWidget(self._bottom_pane_toggle_btn)", window_text)
        self.assertIn("def _on_bottom_pane_toggle_clicked(self, _checked: bool = False) -> None:", window_text)
        self.assertIn("host.toggle_bottom_pane()", window_text)

    def test_assistant_button_is_wired_next_to_bottom_pane_toggle(self) -> None:
        window_text = (ROOT / "Insight" / "insight_local" / "cvops" / "window.py").read_text(
            encoding="utf-8"
        )
        overlay_text = (
            ROOT / "Insight" / "insight_local" / "cvops" / "ui" / "assistant_overlay.py"
        ).read_text(encoding="utf-8")

        self.assertIn('self._ai_assistant_btn = QPushButton("Assistant")', window_text)
        self.assertIn("self._ai_assistant_btn.setCheckable(True)", window_text)
        self.assertIn("self._ai_assistant_btn.clicked.connect(self._on_ai_assistant_clicked)", window_text)
        self.assertIn(
            "self._top_nav_row.addWidget(self._bottom_pane_toggle_btn)\n"
            "        self._ai_assistant_btn = QPushButton(\"Assistant\")",
            window_text,
        )
        self.assertIn("def _open_ai_assistant(self) -> None:", window_text)
        self.assertIn('self._preload_symbol(".ui.assistant_overlay", "AssistantOverlayWindow")', window_text)
        self.assertIn("closed.connect(self._on_ai_assistant_closed)", window_text)
        self.assertIn("def _update_ai_assistant_geometry(self) -> None:", window_text)
        self.assertIn("place_in_parent(anchor_rect, margin=12)", window_text)
        self.assertIn("anchor_rect.bottom() - height - margin", window_text)
        self.assertIn("class _AssistantResizeHandle(QFrame):", overlay_text)
        self.assertIn("closed = pyqtSignal()", overlay_text)
        self.assertIn("def place_in_parent(self, anchor_rect: QRect", overlay_text)
        self.assertIn("max_y = anchor_rect.bottom() - height - margin", overlay_text)
        self.assertIn("WA_TranslucentBackground", overlay_text)
        self.assertNotIn("setWindowFlags", overlay_text)
        self.assertNotIn("WA_DeleteOnClose", overlay_text)

    def test_run_artifacts_titles_are_compact_section_labels(self) -> None:
        from insight_local.cvops.ui.run_artifacts_panel import RunArtifactsPanel

        panel = RunArtifactsPanel(
            base_url="http://127.0.0.1:8787",
            http_get=lambda _path: {},
            http_get_text=lambda _path: "",
        )
        try:
            labels = panel.findChildren(QLabel)
            artifact_titles = [
                label for label in labels
                if label.objectName() in {"artifactPanelTitle", "artifactSectionTitle"}
            ]

            self.assertGreaterEqual(len(artifact_titles), 5)
            self.assertTrue(all(label.property("isTitle") is not True for label in artifact_titles))
            self.assertTrue(all(label.maximumHeight() <= 18 for label in artifact_titles))
            self.assertTrue(all(label.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Fixed for label in artifact_titles))
        finally:
            panel.deleteLater()

    def test_results_pane_hides_until_result_context_exists(self) -> None:
        from insight_local.cvops.ui.result_panel import ResultPanel
        from insight_local.cvops.ui.workbench_split_host import WorkbenchSplitHost, WorkbenchSplitRefs

        result_panel = ResultPanel()
        host = WorkbenchSplitHost(
            WorkbenchSplitRefs(
                catalog_list=QScrollArea(),
                catalog_detail=QScrollArea(),
                result_panel=result_panel,
                lineage_panel=QScrollArea(),
                test_range_page=QScrollArea(),
                data_page=QScrollArea(),
                viz_page=QScrollArea(),
                collect_page=QScrollArea(),
                notes_page=QScrollArea(),
                settings_page=QScrollArea(),
                diagnostics_page=QScrollArea(),
                cells_page=QScrollArea(),
                three_d_page=QScrollArea(),
                notifications_page=QScrollArea(),
                portal_page=QScrollArea(),
                queue_panel=QScrollArea(),
                collect_database_panel=QScrollArea(),
                collect_dataset_editor=QScrollArea(),
                data_viz_selector=QScrollArea(),
            )
        )
        try:
            host.apply_preset(WorkbenchSplitHost.PRESET_EVAL)

            self.assertNotIn("results", host.tray_pane_ids())

            result_panel.show_message("Job job-1: no result yet.")
            self._app.processEvents()

            self.assertIs(host.current_center_widget(), host._result_widget)
            self.assertNotIn("results", host.tray_pane_ids())

            result_panel.clear()
            self._app.processEvents()

            self.assertNotIn("results", host.tray_pane_ids())
        finally:
            host.deleteLater()

    def test_workbench_shell_uses_center_and_bottom_tray(self) -> None:
        from insight_local.cvops.ui.result_panel import ResultPanel
        from insight_local.cvops.ui.workbench_split_host import WorkbenchSplitHost, WorkbenchSplitRefs

        detail = QScrollArea()
        collect = QScrollArea()
        database = QScrollArea()
        editor = QScrollArea()
        notifications = QScrollArea()
        host = WorkbenchSplitHost(
            WorkbenchSplitRefs(
                catalog_list=QScrollArea(),
                catalog_detail=detail,
                result_panel=ResultPanel(),
                lineage_panel=QScrollArea(),
                test_range_page=QScrollArea(),
                data_page=QScrollArea(),
                viz_page=QScrollArea(),
                collect_page=collect,
                notes_page=QScrollArea(),
                settings_page=QScrollArea(),
                diagnostics_page=QScrollArea(),
                cells_page=QScrollArea(),
                three_d_page=QScrollArea(),
                notifications_page=notifications,
                portal_page=QScrollArea(),
                queue_panel=QScrollArea(),
                collect_database_panel=database,
                collect_dataset_editor=editor,
                data_viz_selector=QScrollArea(),
            )
        )
        try:
            self.assertIs(host.current_center_widget(), host._detail_widget)
            self.assertIn("lineage", host.tray_pane_ids())
            self.assertIn("queue", host.tray_pane_ids())

            host.set_mode(WorkbenchSplitHost.MODE_COLLECT)
            self.assertIs(host.current_center_widget(), collect)
            self.assertEqual(host.tray_pane_ids(), ["collect_helpers"])
            tray_cards = host._bottom_tray._cards
            self.assertEqual(len(tray_cards), 1)
            self.assertTrue(
                all(card.sizePolicy().horizontalPolicy() == QSizePolicy.Policy.Expanding for card in tray_cards)
            )
            self.assertTrue(
                all(card.sizePolicy().verticalPolicy() == QSizePolicy.Policy.Expanding for card in tray_cards)
            )
            self.assertGreaterEqual(host._bottom_tray._row_widget.minimumWidth(), 420)

            host.close_tray_pane("collect_helpers")
            self.assertEqual(host.tray_pane_ids(), [])
            host.reopen_all_panes()
            self.assertEqual(host.tray_pane_ids(), ["collect_helpers"])

            host.set_mode(WorkbenchSplitHost.MODE_NOTIFICATIONS)
            self.assertIs(host.current_center_widget(), notifications)
            self.assertEqual(host.tray_pane_ids(), [])
        finally:
            host.deleteLater()

    def test_workbench_shell_can_toggle_bottom_tray_visibility(self) -> None:
        from insight_local.cvops.ui.result_panel import ResultPanel
        from insight_local.cvops.ui.workbench_split_host import WorkbenchSplitHost, WorkbenchSplitRefs

        host = WorkbenchSplitHost(
            WorkbenchSplitRefs(
                catalog_list=QScrollArea(),
                catalog_detail=QScrollArea(),
                result_panel=ResultPanel(),
                lineage_panel=QScrollArea(),
                test_range_page=QScrollArea(),
                data_page=QScrollArea(),
                viz_page=QScrollArea(),
                collect_page=QScrollArea(),
                notes_page=QScrollArea(),
                settings_page=QScrollArea(),
                diagnostics_page=QScrollArea(),
                cells_page=QScrollArea(),
                three_d_page=QScrollArea(),
                notifications_page=QScrollArea(),
                portal_page=QScrollArea(),
                queue_panel=QScrollArea(),
                collect_database_panel=QScrollArea(),
                collect_dataset_editor=QScrollArea(),
                data_viz_selector=QScrollArea(),
            )
        )
        try:
            host.set_mode(WorkbenchSplitHost.MODE_COLLECT)
            self.assertTrue(host.has_bottom_panes())
            self.assertTrue(host.is_bottom_pane_visible())

            visible = host.toggle_bottom_pane()
            self.assertFalse(visible)
            self.assertFalse(host.is_bottom_pane_visible())
            self.assertEqual(host.tray_pane_ids(), ["collect_helpers"])

            visible = host.toggle_bottom_pane()
            self.assertTrue(visible)
            self.assertTrue(host.is_bottom_pane_visible())
            self.assertEqual(host.tray_pane_ids(), ["collect_helpers"])
        finally:
            host.deleteLater()

    def test_workbench_collect_bottom_tray_expands_from_tiny_saved_size(self) -> None:
        from insight_local.cvops.ui.result_panel import ResultPanel
        from insight_local.cvops.ui.workbench_split_host import WorkbenchSplitHost, WorkbenchSplitRefs

        host = WorkbenchSplitHost(
            WorkbenchSplitRefs(
                catalog_list=QScrollArea(),
                catalog_detail=QScrollArea(),
                result_panel=ResultPanel(),
                lineage_panel=QScrollArea(),
                test_range_page=QScrollArea(),
                data_page=QScrollArea(),
                viz_page=QScrollArea(),
                collect_page=QScrollArea(),
                notes_page=QScrollArea(),
                settings_page=QScrollArea(),
                diagnostics_page=QScrollArea(),
                cells_page=QScrollArea(),
                three_d_page=QScrollArea(),
                notifications_page=QScrollArea(),
                portal_page=QScrollArea(),
                queue_panel=QScrollArea(),
                collect_database_panel=QScrollArea(),
                collect_dataset_editor=QScrollArea(),
                data_viz_selector=QScrollArea(),
            )
        )
        try:
            host.resize(1200, 900)
            host.show()
            self._app.processEvents()
            host.set_mode(WorkbenchSplitHost.MODE_COLLECT)
            self._app.processEvents()

            host._body_split.setSizes([850, 50])
            self._app.processEvents()
            host.set_mode(WorkbenchSplitHost.MODE_COLLECT)
            self._app.processEvents()

            self.assertGreaterEqual(host._body_split.sizes()[1], 180)
        finally:
            host.hide()
            host.deleteLater()

    def test_workbench_shell_persists_catalog_tray_and_split_state(self) -> None:
        from insight_local.cvops.ui.result_panel import ResultPanel
        from insight_local.cvops.ui.workbench_split_host import WorkbenchSplitHost, WorkbenchSplitRefs

        host = WorkbenchSplitHost(
            WorkbenchSplitRefs(
                catalog_list=QScrollArea(),
                catalog_detail=QScrollArea(),
                result_panel=ResultPanel(),
                lineage_panel=QScrollArea(),
                test_range_page=QScrollArea(),
                data_page=QScrollArea(),
                viz_page=QScrollArea(),
                collect_page=QScrollArea(),
                notes_page=QScrollArea(),
                settings_page=QScrollArea(),
                diagnostics_page=QScrollArea(),
                cells_page=QScrollArea(),
                three_d_page=QScrollArea(),
                notifications_page=QScrollArea(),
                portal_page=QScrollArea(),
                queue_panel=QScrollArea(),
                collect_database_panel=QScrollArea(),
                collect_dataset_editor=QScrollArea(),
                data_viz_selector=QScrollArea(),
            )
        )
        try:
            host.set_mode(WorkbenchSplitHost.MODE_COLLECT)
            host.toggle_catalog()
            host.close_tray_pane("collect_helpers")
            state = host.save_split_state()

            restored = WorkbenchSplitHost(
                WorkbenchSplitRefs(
                    catalog_list=QScrollArea(),
                    catalog_detail=QScrollArea(),
                    result_panel=ResultPanel(),
                    lineage_panel=QScrollArea(),
                    test_range_page=QScrollArea(),
                    data_page=QScrollArea(),
                    viz_page=QScrollArea(),
                    collect_page=QScrollArea(),
                    notes_page=QScrollArea(),
                    settings_page=QScrollArea(),
                    diagnostics_page=QScrollArea(),
                    cells_page=QScrollArea(),
                    three_d_page=QScrollArea(),
                    notifications_page=QScrollArea(),
                    portal_page=QScrollArea(),
                    queue_panel=QScrollArea(),
                    collect_database_panel=QScrollArea(),
                    collect_dataset_editor=QScrollArea(),
                    data_viz_selector=QScrollArea(),
                )
            )
            try:
                restored.set_mode(WorkbenchSplitHost.MODE_COLLECT)
                restored.restore_split_state(state)
                self.assertFalse(restored.is_catalog_visible())
                self.assertEqual(restored.tray_pane_ids(), [])
            finally:
                restored.deleteLater()
        finally:
            host.deleteLater()

    def test_workbench_settings_mode_uses_settings_diagnostics_split(self) -> None:
        from insight_local.cvops.ui.result_panel import ResultPanel
        from insight_local.cvops.ui.workbench_split_host import WorkbenchSplitHost, WorkbenchSplitRefs

        settings_page = QScrollArea()
        diagnostics_page = QScrollArea()
        host = WorkbenchSplitHost(
            WorkbenchSplitRefs(
                catalog_list=QScrollArea(),
                catalog_detail=QScrollArea(),
                result_panel=ResultPanel(),
                lineage_panel=QScrollArea(),
                test_range_page=QScrollArea(),
                data_page=QScrollArea(),
                viz_page=QScrollArea(),
                collect_page=QScrollArea(),
                notes_page=QScrollArea(),
                settings_page=settings_page,
                diagnostics_page=diagnostics_page,
                cells_page=QScrollArea(),
                three_d_page=QScrollArea(),
                notifications_page=QScrollArea(),
                portal_page=QScrollArea(),
                queue_panel=QScrollArea(),
                collect_database_panel=QScrollArea(),
                collect_dataset_editor=QScrollArea(),
                data_viz_selector=QScrollArea(),
            )
        )
        try:
            host.set_mode(WorkbenchSplitHost.MODE_SETTINGS)

            center = host.current_center_widget()
            self.assertIsInstance(center, QSplitter)
            self.assertIs(center.widget(0), settings_page)
            self.assertIs(center.widget(1), diagnostics_page)
            self.assertEqual(host.tray_pane_ids(), [])
            state = host.save_split_state()
            self.assertIn("settings_diag", state)
            self.assertGreater(len(state["settings_diag"]), 0)
        finally:
            host.deleteLater()

    def test_workbench_train_preset_keeps_results_out_of_bottom_tray(self) -> None:
        from insight_local.cvops.ui.result_panel import ResultPanel
        from insight_local.cvops.ui.workbench_split_host import WorkbenchSplitHost, WorkbenchSplitRefs

        result_panel = ResultPanel()
        host = WorkbenchSplitHost(
            WorkbenchSplitRefs(
                catalog_list=QScrollArea(),
                catalog_detail=QScrollArea(),
                result_panel=result_panel,
                lineage_panel=QScrollArea(),
                test_range_page=QScrollArea(),
                data_page=QScrollArea(),
                viz_page=QScrollArea(),
                collect_page=QScrollArea(),
                notes_page=QScrollArea(),
                settings_page=QScrollArea(),
                diagnostics_page=QScrollArea(),
                cells_page=QScrollArea(),
                three_d_page=QScrollArea(),
                notifications_page=QScrollArea(),
                portal_page=QScrollArea(),
                queue_panel=QScrollArea(),
                collect_database_panel=QScrollArea(),
                collect_dataset_editor=QScrollArea(),
                data_viz_selector=QScrollArea(),
            )
        )
        try:
            result_panel.show_message("Job job-1: no result yet.")
            self._app.processEvents()

            host.apply_preset(WorkbenchSplitHost.PRESET_TRAIN)

            self.assertEqual(host.tray_pane_ids(), ["lineage", "queue"])
            self.assertNotIn("results", host.tray_pane_ids())
        finally:
            host.deleteLater()

    def test_hyperparam_panel_reload_does_not_orphan_old_widgets(self) -> None:
        from insight_local.cvops.ui.hyperparam_suite_panel import HyperparamSuitePanel

        schema = {
            "epochs": {"kind": "int", "min": 1, "max": 300},
            "lr0": {"kind": "float", "min": 0.0, "max": 1.0},
            "mystery": {"kind": "object"},
        }
        panel = HyperparamSuitePanel()
        try:
            panel.load({"epochs": 20, "lr0": 0.01}, schema)
            panel.show()
            self._app.processEvents()
            old_widgets = [field.widget for field in panel._fields.values()]

            panel.load({"epochs": 21}, schema)
            self._app.processEvents()

            top_levels = set(QApplication.topLevelWidgets())
            self.assertTrue(all(widget not in top_levels for widget in old_widgets))
            self.assertEqual(set(panel._fields), {"epochs", "lr0"})
            self.assertTrue(any(section.isVisible() for section in panel._sections))
        finally:
            panel.deleteLater()

    def test_new_scenario_dialog_autofills_from_dataset_and_classes(self) -> None:
        from insight_local.cvops.ui.new_scenario_dialog import NewScenarioDialog

        dlg = NewScenarioDialog(
            http_get=self._fake_get,
            http_post=lambda _path, _body: {"name": "Donut_Defects"},
            models=[{"name": "YOLO Nano", "value": "assets/models/yolo26n.pt"}],
        )
        try:
            idx = dlg._dataset.findData("Donut Defects")
            self.assertGreaterEqual(idx, 0)
            dlg._dataset.setCurrentIndex(idx)

            self.assertEqual(dlg._name.text(), "Donut_Defects")
            self.assertEqual(dlg._display.text(), "Donut Defects")
            self.assertIn("good", dlg._classes.toPlainText())
            self.assertTrue(dlg._create_btn.isEnabled())
        finally:
            dlg.deleteLater()

    def test_new_scenario_dialog_backbone_visibility(self) -> None:
        from insight_local.cvops.ui.new_scenario_dialog import NewScenarioDialog

        dlg = NewScenarioDialog(
            http_get=self._fake_get,
            http_post=lambda _path, _body: {"name": "signals"},
            models=[{"name": "YOLO Nano", "value": "assets/models/yolo26n.pt"}],
        )
        try:
            dlg._set_backbone_type("torch_tabular")
            self.assertFalse(dlg._tabular_rows[0][1].isHidden())
            self.assertTrue(dlg._cv_rows[2][1].isHidden())

            dlg._set_backbone_type("yolo_detection")
            self.assertFalse(dlg._cv_rows[2][1].isHidden())
            self.assertTrue(dlg._tabular_rows[0][1].isHidden())
        finally:
            dlg.deleteLater()

    def test_new_scenario_dialog_initial_dataset_uses_preloaded_cache(self) -> None:
        from insight_local.cvops.ui.new_scenario_dialog import NewScenarioDialog

        calls: list[str] = []

        def _get(path: str) -> dict:
            calls.append(path)
            if path == "/database/Donut%20Defects":
                raise AssertionError("dialog probed the first image dataset before the initial dataset")
            return self._fake_get(path)

        dlg = NewScenarioDialog(
            http_get=_get,
            http_post=lambda _path, _body: {"name": "audio"},
            models=[{"name": "YOLO Nano", "value": "assets/models/yolo26n.pt"}],
            datasets_payload=self._fake_get("/database"),
            dataset_info_cache={"AudioRecognition": self._fake_get("/database/AudioRecognition")},
            initial_dataset="AudioRecognition",
        )
        try:
            self.assertEqual(dlg._current_backbone(), "audio_recognition")
            self.assertEqual(dlg._dataset.currentData(), "AudioRecognition")
            self.assertTrue(dlg._create_btn.isEnabled())
            self.assertNotIn("/database/Donut%20Defects", calls)
        finally:
            dlg.deleteLater()

    def test_new_scenario_dialog_llm_mode_does_not_silent_scan_models(self) -> None:
        from insight_local.cvops.ui import new_scenario_dialog as nsd
        from insight_local.cvops.ui.new_scenario_dialog import NewScenarioDialog

        with patch.object(nsd, "list_finetune_base_candidates", return_value=(["llama3.2"], [])) as discover:
            dlg = NewScenarioDialog(
                http_get=self._fake_get,
                http_post=lambda _path, _body: {"name": "instructions"},
                models=[{"name": "YOLO Nano", "value": "assets/models/yolo26n.pt"}],
            )
            try:
                dlg._set_backbone_type("llm_fine_tuning")
                discover.assert_not_called()
                dlg._refresh_ollama_base_models(silent=False)
                discover.assert_called_once()
            finally:
                dlg.deleteLater()

    def test_range_catalog_stores_browsed_ocr_model(self) -> None:
        from insight_local.cvops.ui import test_range_subroutine as subroutine
        from insight_local.cvops.ui.test_range_subroutine import ModelCatalogDialog

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            external = root / "external"
            external.mkdir()
            src = external / "ocr-text-detector.onnx"
            src.write_bytes(b"fake onnx")
            ocr_dir = root / "assets" / "models" / "ocr"

            with (
                patch.object(subroutine, "_OCR_MODELS_DIR", ocr_dir),
                patch.object(subroutine.QFileDialog, "getOpenFileName", return_value=(str(src), "")),
                patch.object(subroutine.QFileDialog, "getExistingDirectory", return_value=""),
            ):
                dlg = ModelCatalogDialog([], mode=ModelCatalogDialog.MODE_SINGLE)
                try:
                    dlg._on_browse_weights()

                    stored = ocr_dir / src.name
                    self.assertTrue(stored.exists())
                    self.assertEqual(dlg.selected_paths(), [str(stored.resolve())])
                    self.assertEqual(dlg._list.item(0).text(), f"OCR / {src.name}")
                finally:
                    dlg.deleteLater()

    def test_range_catalog_discovers_stored_ocr_models(self) -> None:
        from insight_local.cvops.ui import test_range_subroutine as subroutine

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            models_dir = root / "assets" / "models"
            ocr_dir = models_dir / "ocr"
            ocr_dir.mkdir(parents=True)
            ocr_model = ocr_dir / "ocr-fusion.onnx"
            ocr_model.write_bytes(b"fake onnx")

            with (
                patch.object(subroutine, "_MODELS_DIR", models_dir),
                patch.object(subroutine, "_OCR_MODELS_DIR", ocr_dir),
            ):
                rows = subroutine.collect_video_test_models()

            self.assertIn((f"OCR / {ocr_model.name}", str(ocr_model.resolve())), rows)

    def test_new_scenario_dialog_llm_mode_payload(self) -> None:
        from insight_local.cvops.ui import new_scenario_dialog as nsd
        from insight_local.cvops.ui.new_scenario_dialog import NewScenarioDialog

        posted: list[dict] = []
        with patch.object(nsd, "list_finetune_base_candidates", return_value=([], [])):
            dlg = NewScenarioDialog(
                http_get=self._fake_get,
                http_post=lambda _path, body: posted.append(dict(body or {})) or {"name": "instructions"},
                models=[{"name": "YOLO Nano", "value": "assets/models/yolo26n.pt"}],
            )
            try:
                dlg._set_backbone_type("llm_fine_tuning")
                self.assertFalse(dlg._llm_rows[0][1].isHidden())
                self.assertTrue(dlg._cv_rows[0][1].isHidden())
                idx = dlg._dataset.findData("mlops/datasets/instructions.jsonl")
                self.assertGreaterEqual(idx, 0)
                dlg._dataset.setCurrentIndex(idx)
                dlg._name.setText("instructions")
                dlg._llm_base_model.setText("local/tiny")
                dlg._ollama_base_model.setCurrentText("llama3.2")
                dlg._create()
                self.assertEqual(posted[0]["backbone_type"], "llm_fine_tuning")
                self.assertEqual(posted[0]["dataset"], "mlops/datasets/instructions.jsonl")
                self.assertEqual(posted[0]["backbone_config"]["base_model"], "local/tiny")
                self.assertEqual(posted[0]["backbone_config"]["ollama_base_model"], "llama3.2")
                self.assertIn("jsonl", posted[0]["backbone_config"]["sources"])
            finally:
                dlg.deleteLater()

    def test_catalog_training_ctas_follow_scenario_state(self) -> None:
        from insight_local.cvops.ui.catalog_panel import CatalogPanel

        posts: list[tuple[str, dict | None]] = []
        panel = CatalogPanel(
            base_url="http://127.0.0.1:8787",
            http_get=self._fake_get,
            http_post=lambda path, body: posts.append((path, body)) or {"job_id": "job-test", "training_guard": {}},
            http_delete=lambda _path: {},
            http_get_text=lambda _path: "",
        )
        try:
            base = {
                "name": "donut",
                "display_name": "Donut",
                "description": "",
                "dataset": "Donut Defects",
                "backbone_type": "yolo_detection",
                "base_model": "assets/models/yolo26n.pt",
                "base_model_exists": True,
                "dataset_count": 42,
                "verified": False,
                "weights_ready": False,
                "latest_run": None,
                "history_count": 0,
                "training_guard": {},
            }

            panel._render_detail({**base, "status": "dataset"})
            self.assertTrue(panel._kick_btn.isEnabled())
            self.assertTrue(panel._update_btn.isHidden())
            self.assertFalse(panel._stop_btn.isEnabled())

            panel._render_detail({**base, "status": "training"})
            self.assertFalse(panel._kick_btn.isEnabled())
            self.assertTrue(panel._update_btn.isHidden())

            run = {"version": "v1", "map50": 0.71, "trained_at": "now"}
            panel._render_detail({**base, "status": "trained", "latest_run": run, "weights_ready": True})
            self.assertTrue(panel._kick_btn.isEnabled())
            self.assertFalse(panel._update_btn.isHidden())
            self.assertTrue(panel._update_btn.isEnabled())
            self.assertTrue(panel._verify_btn.isEnabled())

            panel._render_detail({**base, "status": "ready", "latest_run": run, "weights_ready": True, "verified": True})
            self.assertFalse(panel._update_btn.isHidden())
            self.assertFalse(panel._verify_btn.isEnabled())
            self.assertTrue(panel._unverify_btn.isEnabled())

            panel._final_model_name.setText("Donut Final V1")
            panel._model_combo.setCurrentIndex(1)
            payload = panel._training_payload("yolo_detection")
            self.assertEqual(payload["final_model_name"], "Donut Final V1")
            self.assertEqual(payload["base_model_override"], "assets/models/yolo26s.pt")
        finally:
            panel.deleteLater()

    def test_catalog_train_layout_places_results_below_data_visualization(self) -> None:
        from insight_local.cvops.ui.catalog_panel import CatalogPanel

        panel = CatalogPanel(
            base_url="http://127.0.0.1:8787",
            http_get=self._fake_get,
            http_post=lambda _path, _body: {},
            http_delete=lambda _path: {},
            http_get_text=lambda _path: "",
        )
        try:
            self.assertEqual(panel._detail_main_split.count(), 2)
            self.assertEqual(panel._advanced_splitter.count(), 5)
            self.assertEqual(panel._advanced_splitter.widget(3)._toggle.text(), "Data Visualization")
            self.assertEqual(panel._advanced_splitter.widget(4)._toggle.text(), "Results")
        finally:
            panel.deleteLater()

    def test_scenario_flow_view_renders_progressive_native_nodes(self) -> None:
        from insight_local.cvops.ui.scenario_flow_view import ScenarioFlowView, _FlowNode

        view = ScenarioFlowView()
        try:
            steps = [
                ("Scenario", "trained", ["Name: donut", "Type: Vision (YOLO)"], "ok"),
                ("Dataset", "ready", ["Source: Donut Defects", "Count: 42 items"], "ok"),
                ("Model", "ready", ["Base: yolo26n.pt"], "ok"),
                ("System & Guard", "ok", ["Profile: stable"], "ok"),
                ("Run Config", "prepared", ["Final model: donut-v2"], "active"),
                ("Training Action", "waiting", ["Job: none"], "idle"),
                ("Review Outputs", "available", ["Latest: donut-v2"], "ok"),
            ]
            view.set_flow("donut", "Donut Defects", steps)

            nodes = [it for it in view._scene.items() if isinstance(it, _FlowNode)]
            self.assertEqual(len(nodes), len(steps))
            titles = {n._title for n in nodes}
            self.assertIn("Scenario", titles)
            self.assertIn("System & Guard", titles)
            self.assertIn("Review Outputs", titles)

            # Empty selection collapses to a placeholder (no nodes).
            view.set_flow("", "", [])
            self.assertEqual(
                [it for it in view._scene.items() if isinstance(it, _FlowNode)],
                [],
            )
        finally:
            view.deleteLater()

    def test_data_viz_hub_exposes_native_flow_tab(self) -> None:
        from insight_local.cvops.ui.data_viz_hub import DataVizHub

        hub = DataVizHub()
        try:
            tabs = [hub._tabs.tabText(i) for i in range(hub._tabs.count())]
            self.assertIn("Flow", tabs)
            hub.set_flow(
                "donut",
                "Donut Defects",
                [("Scenario", "trained", ["Name: donut"], "ok")],
            )
            from insight_local.cvops.ui.scenario_flow_view import _FlowNode

            nodes = [it for it in hub._flow._scene.items() if isinstance(it, _FlowNode)]
            self.assertEqual(len(nodes), 1)
        finally:
            hub.deleteLater()

    def test_model_gallery_uses_na_for_missing_history_fields(self) -> None:
        from insight_local.cvops.ui.scenario_history_panel import ScenarioHistoryPanel

        runs = [
            {
                "version": "v2",
                "version_number": 2,
                "status": "partial",
                "artifact_count": 3,
            },
            {
                "version": "v1",
                "version_number": 1,
                "status": "trained",
                "map50": 0.4105,
                "trained_at": "2026-05-12T15:03:03Z",
                "verified": False,
                "base_model": "assets/models/yolo26s.pt",
                "artifact_count": 47,
                "run_dir": "/tmp/models/v1",
                "training_duration_seconds": 3661,
            },
        ]
        panel = ScenarioHistoryPanel(http_get=lambda _path: {"runs": runs})
        try:
            panel.load_scenario("TIGERTEST")
            self._app.processEvents()

            self.assertEqual(panel._table.item(0, 0).text(), "v2")
            self.assertEqual(panel._table.item(0, 2).text(), "N/A")
            self.assertEqual(panel._table.item(0, 3).text(), "N/A")
            self.assertEqual(panel._table.item(0, 4).text(), "N/A")
            self.assertEqual(panel._table.item(0, 5).text(), "N/A")
            self.assertEqual(panel._table.item(0, 6).text(), "N/A")
            self.assertEqual(panel._table.item(0, 7).text(), "3")

            panel._table.selectRow(0)
            self._app.processEvents()
            self.assertIn("metric: N/A", panel._detail.text())
            self.assertIn("duration: N/A", panel._detail.text())
            self.assertIn("base_model: N/A", panel._detail.text())
            self.assertIn("run_dir: N/A", panel._detail.text())

            self.assertEqual(panel._table.item(1, 4).text(), "1h 1m")
            self.assertEqual(panel._table.item(1, 5).text(), "No")
            self.assertEqual(panel._table.item(1, 6).text(), "yolo26s.pt")
        finally:
            panel.deleteLater()

    def test_cvops_duration_formatter_spans_seconds_to_years(self) -> None:
        from insight_local.cvops.ui.queue_panel import _fmt_elapsed
        from insight_local.cvops.ui.time_format import format_duration_seconds

        self.assertEqual(format_duration_seconds(9), "9s")
        self.assertEqual(format_duration_seconds(125), "2m 5s")
        self.assertEqual(format_duration_seconds(3_661), "1h 1m")
        self.assertEqual(format_duration_seconds(93_600), "1d 2h")
        self.assertEqual(format_duration_seconds(400 * 24 * 60 * 60), "1y 35d")
        self.assertEqual(_fmt_elapsed(100.0, 160.0, 3_760.0, "done"), "wait 1m | run 1h")

    def test_catalog_detail_widget_is_scrollable_for_train_cube(self) -> None:
        from insight_local.cvops.ui.catalog_panel import CatalogPanel

        panel = CatalogPanel(
            base_url="http://127.0.0.1:8787",
            http_get=self._fake_get,
            http_post=lambda _path, _body: {},
            http_delete=lambda _path: {},
            http_get_text=lambda _path: "",
        )
        try:
            detail = panel.detail_widget()
            self.assertIsInstance(detail, QScrollArea)
            self.assertTrue(detail.widgetResizable())
            self.assertEqual(
                detail.verticalScrollBarPolicy(),
                Qt.ScrollBarPolicy.ScrollBarAsNeeded,
            )
        finally:
            panel.deleteLater()

    def test_catalog_header_refresh_button_requests_scenario_reload(self) -> None:
        from insight_local.cvops.ui.catalog_panel import CatalogPanel

        emitted: list[str] = []
        panel = CatalogPanel(
            base_url="http://127.0.0.1:8787",
            http_get=self._fake_get,
            http_post=lambda _path, _body: {},
            http_delete=lambda _path: {},
            http_get_text=lambda _path: "",
        )
        try:
            panel.scenarioMutated.connect(emitted.append)
            self.assertEqual(panel._refresh_scenarios_btn.text(), "Refresh")

            panel._refresh_scenarios_btn.click()

            self.assertEqual(emitted, [""])
        finally:
            panel.deleteLater()

    def test_catalog_latest_run_time_uses_configured_12_hour_format(self) -> None:
        from insight_local.cvops.ui.catalog_panel import CatalogPanel
        from insight_local.cvops.ui.time_format import set_time_format

        set_time_format("12h")
        panel = CatalogPanel(
            base_url="http://127.0.0.1:8787",
            http_get=self._fake_get,
            http_post=lambda _path, _body: {},
            http_delete=lambda _path: {},
            http_get_text=lambda _path: "",
        )
        try:
            panel._render_detail(
                {
                    "name": "donut",
                    "display_name": "Donut",
                    "description": "",
                    "dataset": "Donut Defects",
                    "backbone_type": "yolo_detection",
                    "base_model": "assets/models/yolo26n.pt",
                    "base_model_exists": True,
                    "dataset_count": 42,
                    "verified": False,
                    "weights_ready": True,
                    "latest_run": {
                        "version": "v1",
                        "map50": 0.71,
                        "trained_at": "2026-01-01T14:05:06Z",
                    },
                    "history_count": 1,
                    "training_guard": {},
                    "status": "trained",
                }
            )

            self.assertIn("AM", panel._train_meta.text())
        finally:
            set_time_format("24h")
            panel.deleteLater()

    def test_settings_loads_time_format(self) -> None:
        from insight_local.cvops.ui.settings_panel import load_cvops_settings

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text('{"time_format": "12h"}', encoding="utf-8")

            self.assertEqual(load_cvops_settings(path).time_format, "12h")

    def test_settings_load_ui_scale_pct(self) -> None:
        from insight_local.cvops.ui.settings_panel import load_cvops_settings

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            path.write_text('{"ui_scale_pct": 114}', encoding="utf-8")

            self.assertEqual(load_cvops_settings(path).ui_scale_pct, 114)

    def test_button_shape_normalizer_supports_radial_aliases(self) -> None:
        from insight_local.cvops.ui.patch_parallelogram_buttons import normalize_cvops_button_shape

        self.assertEqual(normalize_cvops_button_shape("radial"), "radial")
        self.assertEqual(normalize_cvops_button_shape("rounded"), "radial")
        self.assertEqual(normalize_cvops_button_shape("square"), "none")

    def test_settings_button_shape_options_include_square_and_radial(self) -> None:
        from insight_local.cvops.ui.settings_panel import CvOpsSettings, CvOpsSettingsPanel

        with tempfile.TemporaryDirectory() as td:
            panel = CvOpsSettingsPanel(
                settings_path=Path(td) / "settings.json",
                settings=CvOpsSettings(),
                host="127.0.0.1",
                port=8787,
                dashboard_url="http://127.0.0.1:8501",
                state_dir=Path(td) / "state",
                jobs_db_path=Path(td) / "jobs.db",
            )
            try:
                items = [
                    (
                        panel._button_shape_combo.itemText(i),
                        panel._button_shape_combo.itemData(i),
                    )
                    for i in range(panel._button_shape_combo.count())
                ]
                self.assertIn(("Square", "none"), items)
                self.assertIn(("Radial", "radial"), items)
            finally:
                panel.deleteLater()

    def test_getting_started_guide_renders_quick_start_and_tabs(self) -> None:
        from insight_local.cvops.ui.getting_started_guide import (
            GettingStartedGuide,
            getting_started_html,
        )

        html = getting_started_html()
        self.assertIn("Quick start", html)
        self.assertIn("Create a scenario", html)
        self.assertIn("Tab reference", html)

        guide = GettingStartedGuide()
        try:
            text = guide.toPlainText()
            self.assertIn("Quick start", text)
            self.assertIn("Start training", text)
        finally:
            guide.deleteLater()

    def test_scale_qss_pixel_metrics_scales_px_values(self) -> None:
        from insight_local.cvops.ui.cvops_theme import scale_qss_pixel_metrics

        css = "QLabel { font-size: 12px; padding: 4px 8px; margin: -2px 0px; border-width: 1px; }"
        scaled = scale_qss_pixel_metrics(css, 0.75)

        self.assertIn("font-size: 9px", scaled)
        self.assertIn("padding: 3px 6px", scaled)
        self.assertIn("margin: -2px 0px", scaled)
        self.assertIn("border-width: 1px", scaled)

    def test_cvops_stylesheet_renders_ws_refresh_button(self) -> None:
        from insight_local.cvops.ui.cvops_theme import get_cvops_stylesheet
        from insight_local.cvops.ui.patch_parallelogram_buttons import (
            cvops_button_shape,
            set_cvops_button_shape,
        )
        from insight_local.ui.theme import configure_color_scheme, current_color_scheme

        previous = current_color_scheme()
        previous_shape = cvops_button_shape()
        try:
            configure_color_scheme("aurora")
            set_cvops_button_shape("radial")
            css = get_cvops_stylesheet()
        finally:
            set_cvops_button_shape(previous_shape)
            configure_color_scheme(previous)

        self.assertIn("QToolButton#wsRefreshButton", css)
        self.assertRegex(
            css,
            re.compile(r"QToolButton#wsRefreshButton\s*\{[^}]*border-radius:\s*8px;", re.S),
        )
        self.assertRegex(
            css,
            re.compile(r"QPushButton#cvOpsMainTabNavButton\s*\{[^}]*border-radius:\s*8px;", re.S),
        )

    def test_top_hud_buttons_do_not_force_square_opt_out(self) -> None:
        window_text = (ROOT / "Insight" / "insight_local" / "cvops" / "window.py").read_text(
            encoding="utf-8"
        )
        rail_text = (
            ROOT / "Insight" / "insight_local" / "cvops" / "ui" / "activity_rail.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn('self._catalog_toggle_btn.setProperty("cvopsNoSkew", True)', window_text)
        self.assertNotIn('self._restore_panes_btn.setProperty("cvopsNoSkew", True)', window_text)
        self.assertNotIn('self._bottom_pane_toggle_btn.setProperty("cvopsNoSkew", True)', window_text)
        self.assertNotIn('btn.setProperty("cvopsNoSkew", True)', rail_text)

    def test_ontology_html_uses_default_wheel_sensitivity(self) -> None:
        ontology_text = (
            ROOT / "Insight" / "insight_local" / "cvops" / "ui" / "ontology_panel.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("wheelSensitivity:", ontology_text)

    def test_ecosystem_trains_on_edges_are_styled_and_in_legend(self) -> None:
        from insight_local.cvops.ui.ontology_panel import _build_html

        graph = {
            "nodes": [
                {
                    "id": "scenario:demo",
                    "type": "scenario",
                    "label": "demo",
                    "meta": {"scenario": "demo"},
                },
                {
                    "id": "dataset:demo_data",
                    "type": "dataset",
                    "label": "demo_data",
                    "meta": {"scenario": "demo"},
                },
            ],
            "edges": [
                {
                    "source": "scenario:demo",
                    "target": "dataset:demo_data",
                    "type": "trains_on",
                }
            ],
        }

        html = _build_html(graph, "http://127.0.0.1:8787", "/ontology/cytoscape.js")

        self.assertIn("function resolveEdgeEndpoint(id)", html)
        self.assertIn("const edgeSource = resolveEdgeEndpoint(e.source);", html)
        self.assertIn("source: edgeSource, target: edgeTarget", html)
        self.assertIn('selector: \'edge[edgeType = "trains_on"]\'', html)
        self.assertIn("'target-arrow-shape': 'triangle-tee'", html)
        self.assertIn("['trains_on',", html)
        self.assertIn("'TRAINS ON'", html)

    def test_video_test_scope_categories_filter_detections(self) -> None:
        from insight_local.cvops.ui.video_test_panel import _filter_scoped_detections

        detections = [
            {"label": "person", "conf": 0.91},
            {"label": "dog", "conf": 0.88},
            {"label": "laptop", "conf": 0.77},
            {"label": "car", "conf": 0.66},
        ]

        filtered = _filter_scoped_detections(detections, categories={"people", "tech"})

        self.assertEqual([item["label"] for item in filtered], ["person", "laptop"])

    def test_range_panel_can_seal_selected_dataset_as_golden_set(self) -> None:
        from insight_local.cvops.ui.range_panel import TestRangePanel

        posts: list[tuple[str, dict]] = []

        panel = TestRangePanel(
            http_get=self._fake_get,
            http_post=lambda path, body: posts.append((path, body)) or {},
            http_delete=lambda _path: {},
        )
        try:
            panel._selected_id = "range-1"
            panel._reload_datasets()
            self.assertGreaterEqual(panel._dataset_table.rowCount(), 1)
            panel._dataset_table.selectRow(0)

            panel._on_seal_selected_dataset()

            self.assertEqual(posts[0][0], "/ranges/range-1/golden_sets")
            payload = posts[0][1]
            self.assertEqual(payload["name"], "Donut Defects")
            self.assertEqual(payload["row_count"], 42)
            self.assertEqual(payload["content_sha256"], "a" * 64)
            self.assertEqual(payload["split_spec"]["source"], "cvops_dataset_library")
        finally:
            panel.deleteLater()

    def test_video_test_playback_controls_share_seek_timeline_row(self) -> None:
        from insight_local.cvops.ui import video_test_panel as video_module

        old_assets = video_module._ASSETS_VIDEOS
        panel = None
        try:
            with tempfile.TemporaryDirectory() as td:
                video_module._ASSETS_VIDEOS = Path(td)
                panel = video_module.VideoTestPanel(
                    http_get=self._fake_get,
                    http_post=lambda _path, _body: {},
                    http_delete=lambda _path: {},
                )
            widgets = [
                panel._playback_row.itemAt(i).widget()
                for i in range(panel._playback_row.count())
                if panel._playback_row.itemAt(i).widget() is not None
            ]

            expected = [
                panel._back_btn,
                panel._play_btn,
                panel._stop_btn,
                panel._forward_btn,
                panel._current_label,
                panel._position_slider,
                panel._duration_label,
                panel._speed_label,
                panel._speed_combo,
            ]
            for widget in expected:
                self.assertIn(widget, widgets)
            self.assertLess(widgets.index(panel._forward_btn), widgets.index(panel._position_slider))
            self.assertLess(widgets.index(panel._position_slider), widgets.index(panel._speed_combo))
        finally:
            video_module._ASSETS_VIDEOS = old_assets
            if panel is not None:
                panel.deleteLater()

    def test_video_test_boxes_overlay_is_scene_item_above_video_surface(self) -> None:
        from insight_local.cvops.ui import video_test_panel as video_module

        old_assets = video_module._ASSETS_VIDEOS
        panel = None
        try:
            with tempfile.TemporaryDirectory() as td:
                video_module._ASSETS_VIDEOS = Path(td)
                panel = video_module.VideoTestPanel(
                    http_get=self._fake_get,
                    http_post=lambda _path, _body: {},
                    http_delete=lambda _path: {},
                )
            self.assertIn(panel._video_item, panel._video_scene.items())
            self.assertIn(panel._box_overlay, panel._video_scene.items())
            self.assertGreater(panel._box_overlay.zValue(), panel._video_item.zValue())

            panel._box_overlay.set_boxes([{
                "label": "person",
                "conf": 0.93,
                "x1": 10.0,
                "y1": 20.0,
                "x2": 80.0,
                "y2": 120.0,
                "frame_w": 160,
                "frame_h": 120,
            }])
            panel._on_boxes_toggled(False)
            self.assertFalse(panel._box_overlay.isVisible())
            panel._on_boxes_toggled(True)
            self.assertTrue(panel._box_overlay.isVisible())
        finally:
            video_module._ASSETS_VIDEOS = old_assets
            if panel is not None:
                panel.deleteLater()

    def test_lineage_panel_populates_dataset_library_for_drop_sources(self) -> None:
        from insight_local.cvops.ui.lineage_panel import LineageCatalogPanel

        panel = LineageCatalogPanel(
            http_get=self._fake_get,
            http_post=lambda _path, _body: {},
            http_delete=lambda _path: {},
        )
        try:
            panel._reload_datasets()
            self.assertGreaterEqual(panel._dataset_table.rowCount(), 1)
            self.assertIn("available for drops", panel._dataset_status.text())
            panel._dataset_table.selectRow(0)
            selected = panel._selected_dataset()
            self.assertIsNotNone(selected)
            self.assertEqual(selected.get("name"), "Donut Defects")
        finally:
            panel.deleteLater()

    def test_catalog_op_slots_filter_by_backbone(self) -> None:
        from insight_local.cvops.ui.catalog_panel import CatalogPanel

        panel = CatalogPanel(
            base_url="http://127.0.0.1:8787",
            http_get=self._fake_get,
            http_post=lambda _path, _body: {},
            http_delete=lambda _path: {},
            http_get_text=lambda _path: "",
        )
        try:
            scenarios = [
                {"name": "detector", "status": "dataset", "backbone_type": "yolo_detection", "dataset_count": 1},
                {"name": "signals", "status": "dataset", "backbone_type": "torch_tabular", "dataset_count": 1},
                {"name": "custom_scen", "status": "dataset", "backbone_type": "custom_code", "dataset_count": 1},
                {"name": "sound", "status": "dataset", "backbone_type": "audio_recognition", "dataset_count": 1},
                {"name": "faces", "status": "dataset", "backbone_type": "face_recognition", "dataset_count": 1},
            ]
            panel.apply_scenarios(scenarios)

            panel._set_op_slot("audio")
            visible = [
                str(panel._list.item(i).data(Qt.ItemDataRole.UserRole) or "")
                for i in range(panel._list.count())
                if not panel._list.item(i).isHidden()
            ]
            self.assertEqual(visible, ["sound"])
            self.assertIn("Audio 1", panel._op_buttons["audio"].text())

            panel._set_op_slot("tabular")
            visible = sorted(
                str(panel._list.item(i).data(Qt.ItemDataRole.UserRole) or "")
                for i in range(panel._list.count())
                if not panel._list.item(i).isHidden()
            )
            self.assertEqual(visible, ["custom_scen", "signals"])
        finally:
            panel.deleteLater()

    def test_audio_dataset_panel_shows_audio_assets_and_clips(self) -> None:
        from insight_local.cvops.ui.dataset_panel import DatasetPanel

        panel = DatasetPanel(
            base_url="http://127.0.0.1:8787",
            http_get=self._fake_get,
            http_post=lambda _path, _body: {},
            http_delete=lambda _path: {},
        )
        try:
            panel.set_scenario(
                "audio_recognition",
                "AudioRecognition",
                "audio_recognition",
                {},
            )
            self.assertEqual(panel._library_list.count(), 1)
            self.assertEqual(panel._library_list.item(0).text(), "AudioRecognition")
            self.assertFalse(panel._audio_assets_list.isHidden())
            self.assertEqual(panel._audio_assets_list.count(), 2)
            self.assertIn("frenchpeoplewalkinglong.mp4", panel._audio_assets_list.item(0).text())
            self.assertTrue(panel._audio_collect_btn.isEnabled())
            self.assertIn("Audio Assets", panel._header.text())
            self.assertEqual(panel._list.rowCount(), 2)
            self.assertEqual(panel._list.item(0, panel._list.COL_PREVIEW).text(), "WAV")
            self.assertEqual(panel._list.item(0, panel._list.COL_CLASS).text(), "alarm")
            self.assertIn("2 audio clip", panel._count.text())
            self.assertFalse(panel._open_editor_btn.isVisible())
            self.assertFalse(panel._upload_btn.isVisible())
        finally:
            panel.deleteLater()

    def test_dataset_panel_persists_selected_yolo_dataset_to_scenario(self) -> None:
        from insight_local.cvops.ui.dataset_panel import DatasetPanel

        posts: list[tuple[str, dict | None]] = []

        def fake_get(path: str) -> dict[str, Any]:
            if path == "/database":
                return {
                    "datasets": ["tiger111", "tiger111_subset"],
                    "categories": {"tiger111": "image", "tiger111_subset": "image"},
                    "tabular_datasets": [],
                    "text_datasets": [],
                }
            if path == "/database/tiger111":
                return {
                    "slug": "tiger111",
                    "path": "/tmp/database/tiger111",
                    "format": "yolo_detection",
                    "category": "image",
                    "count": 10,
                    "classes": ["tiger"],
                    "split_counts": {"train": 8, "val": 2},
                    "content_sha256": "a" * 64,
                }
            if path == "/database/tiger111_subset":
                return {
                    "slug": "tiger111_subset",
                    "path": "/tmp/database/tiger111_subset",
                    "format": "yolo_detection",
                    "category": "image",
                    "count": 4,
                    "classes": ["tiger"],
                    "split_counts": {"train": 3, "val": 1},
                    "content_sha256": "b" * 64,
                }
            return {}

        panel = DatasetPanel(
            base_url="http://127.0.0.1:8787",
            http_get=fake_get,
            http_post=lambda path, body: posts.append((path, body)) or {"dataset": "tiger111_subset"},
            http_delete=lambda _path: {},
        )
        try:
            panel.set_scenario("tiger_demo", "tiger111", "yolo_detection", {})
            idx = panel._library_combo.findData("tiger111_subset")
            self.assertGreaterEqual(idx, 0)

            panel._library_combo.setCurrentIndex(idx)

            self.assertEqual(
                posts,
                [("/scenarios/tiger_demo/dataset", {"dataset": "tiger111_subset"})],
            )
        finally:
            panel.deleteLater()

    def test_dataset_upload_expands_folders_recursively(self) -> None:
        from insight_local.cvops.ui.dataset_panel import _expand_upload_paths

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "images" / "train" / "class_a"
            nested.mkdir(parents=True)
            image = nested / "sample.jpg"
            image.write_bytes(b"fake")
            (nested / "notes.txt").write_text("ignore", encoding="utf-8")

            self.assertEqual(_expand_upload_paths([str(root)]), [str(image)])

    def test_tabular_upload_worker_reports_progress_and_payload_path(self) -> None:
        from insight_local.cvops.ui.collect_tabular_panel import _TabularUploadWorker

        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "signals.csv"
            csv_path.write_text("x,y\n1,2\n", encoding="utf-8")
            progress: list[str] = []
            finished: dict[str, object] = {}

            def fake_upload(url: str, *, files: dict, timeout: float) -> dict:
                self.assertEqual(url, "http://local/database/upload_csv")
                self.assertEqual(files["file"], csv_path)
                self.assertEqual(timeout, 120.0)
                return {
                    "slug": "signals",
                    "path": "mlops/datasets/signals.csv",
                    "filename": "signals.csv",
                }

            worker = _TabularUploadWorker(base_url="http://local", files=[str(csv_path)])
            worker.progress.connect(progress.append)
            worker.finished.connect(
                lambda uploaded, errors: finished.update(uploaded=uploaded, errors=errors)
            )

            with patch("insight_local.cvops.ui.dataset_panel._multipart_upload", fake_upload):
                worker.run()

            self.assertTrue(any("Validated signals.csv" in msg for msg in progress))
            self.assertTrue(any("Packaging and posting" in msg for msg in progress))
            self.assertTrue(any("Stored as dataset 'signals'" in msg for msg in progress))
            self.assertEqual(finished["errors"], [])
            self.assertEqual(finished["uploaded"][0][0], "signals")

    def test_cvops_theme_keeps_fixed_title_and_selection_tokens_in_dark_mode(self) -> None:
        from insight_local.ui.theme import configure_color_scheme, current_color_scheme
        from insight_local.cvops.ui.cvops_theme import cvops_themed_css, get_cvops_stylesheet

        previous = current_color_scheme()
        try:
            configure_color_scheme("dark mode")
            themed = cvops_themed_css(
                "background: #DC322F; selection-background-color: rgba(108, 113, 196, 0.88);"
            )
            self.assertIn("rgba(220, 50, 47, 1)", themed)
            self.assertIn("rgba(108, 113, 196, 0.88)", themed)
            self.assertIn("background: #DC322F", get_cvops_stylesheet())
            dark_surface = cvops_themed_css("background: #050807;")

            configure_color_scheme("aurora")
            aurora_surface = cvops_themed_css("background: #050807;")

            self.assertIn("rgba(4, 6, 7, 1)", dark_surface)
            self.assertIn("rgba(5, 8, 7, 1)", aurora_surface)
        finally:
            configure_color_scheme(previous)

    def test_cvops_title_background_override_updates_title_semantic_token(self) -> None:
        from insight_local.ui.theme import configure_color_scheme, current_color_scheme
        from insight_local.cvops.ui.cvops_theme import cvops_themed_css, get_cvops_stylesheet

        previous = current_color_scheme()
        try:
            configure_color_scheme("dark mode")
            css = get_cvops_stylesheet(title_background_color="#586E75")
            self.assertIn("background: #586E75", css)
            self.assertIn("border: 1px solid #586E75", css)

            themed = cvops_themed_css(
                "border: 1px solid #DC322F; background: rgba(220, 50, 47, 0.5);"
            )
            self.assertIn("border: 1px solid rgba(88, 110, 117, 1)", themed)
            self.assertIn("background: rgba(88, 110, 117, 0.5)", themed)

            get_cvops_stylesheet(title_background_color="")
            reset = cvops_themed_css("border: 1px solid #DC322F;")
            self.assertIn("border: 1px solid rgba(220, 50, 47, 1)", reset)
        finally:
            get_cvops_stylesheet(title_background_color="")
            configure_color_scheme(previous)

    def test_split_magnifier_capture_only_uses_screen_crop_without_running(self) -> None:
        from PyQt6.QtCore import QRect
        from PyQt6.QtGui import QColor, QImage
        from insight_local.cvops.ui.split_magnifier_panel import SplitMagnifierWindow

        win = SplitMagnifierWindow(http_get=lambda _path: {"models": []})
        try:
            self.assertTrue(win.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground))
            self.assertTrue(bool(win.windowFlags() & Qt.WindowType.WindowStaysOnTopHint))

            img = QImage(80, 40, QImage.Format.Format_RGB32)
            img.fill(QColor("#123456"))
            run_calls: list[bool] = []
            win._auto_run.setChecked(True)
            win._capture_auto_run = False
            win._run_active_selection = lambda: run_calls.append(True)  # type: ignore[method-assign]

            with patch.object(SplitMagnifierWindow, "_grab_screen_rect", return_value=img):
                win._grab_hidden_lens(QRect(0, 0, 80, 40))

            crop = win._subroutine_panel.raw_crop()
            self.assertIsNotNone(crop)
            self.assertEqual((crop.width(), crop.height()), (80, 40))
            self.assertEqual(run_calls, [])
        finally:
            win.close()
            win.deleteLater()

    def test_split_magnifier_lens_interior_renders_transparent(self) -> None:
        from PyQt6.QtGui import QColor, QImage, QPainter
        from insight_local.cvops.ui.split_magnifier_panel import SplitMagnifierWindow

        win = SplitMagnifierWindow(http_get=lambda _path: {"models": []})
        try:
            lens = win._lens
            lens.resize(140, 90)
            image = QImage(lens.size(), QImage.Format.Format_ARGB32)
            image.fill(QColor(0, 0, 0, 0))

            painter = QPainter(image)
            lens.render(painter)
            painter.end()

            self.assertEqual(image.pixelColor(22, 22).alpha(), 0)
            self.assertGreater(
                sum(
                    1
                    for x in range(image.width())
                    for y in range(image.height())
                    if image.pixelColor(x, y).alpha() > 0
                ),
                0,
            )
        finally:
            win.close()
            win.deleteLater()

    def test_notes_ai_workspace_can_keep_or_discard_scratch_chat(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace
        from insight_local.cvops.ui.notes_spaces import ensure_notes_spaces_layout

        with tempfile.TemporaryDirectory() as tmp:
            spaces_root = ensure_notes_spaces_layout(Path(tmp) / "notes")
            workspace = NotesAiWorkspace()
            try:
                workspace.set_space_root(spaces_root / "main")
                workspace.set_compact_overlay_mode(True)
                self.assertFalse(workspace._tabs.tabBar().isVisible())

                first = workspace.start_scratch_chat("Scratch assistant question")
                self.assertTrue(first)
                mgr = workspace._chat_mgr
                self.assertIsNotNone(mgr)
                assert mgr is not None
                self.assertIn(first, mgr.chats)
                mgr.add_message(first, "user", "How do I debug queue stalls?")

                self.assertTrue(
                    workspace.keep_chat_without_prompt(
                        first,
                        title=workspace.suggested_chat_title(first),
                    )
                )

                self.assertIn(first, mgr.chats)
                self.assertEqual(mgr.chats[first]["title"], "How do I debug queue stalls?")
                self.assertFalse(mgr.chats[first]["metadata"].get("cvops_scratch"))

                second = workspace.start_scratch_chat("Scratch assistant question")
                self.assertTrue(second)
                mgr.add_message(second, "user", "temporary question")

                replacement = workspace.discard_chat_without_prompt(
                    second,
                    replacement_title="Scratch assistant question",
                )

                self.assertNotIn(second, mgr.chats)
                self.assertTrue(replacement)
                self.assertIn(replacement, mgr.chats)
            finally:
                workspace.close()
                workspace.deleteLater()

    def test_notes_ai_workspace_composer_expands_for_multiline_text(self) -> None:
        from PyQt6.QtTest import QTest
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace

        workspace = NotesAiWorkspace()
        try:
            workspace.resize(900, 620)
            workspace.show()
            self._app.processEvents()

            editor = workspace.chat_input
            editor.setFocus()
            editor.setPlainText("first line")
            cursor = editor.textCursor()
            cursor.movePosition(cursor.MoveOperation.End)
            editor.setTextCursor(cursor)
            QTest.keyClick(editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
            editor.insertPlainText("second line")
            QTest.keyClick(editor, Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier)
            editor.insertPlainText("third line")
            self._app.processEvents()
            workspace._on_composer_text_height_changed()
            self._app.processEvents()

            self.assertEqual(editor.toPlainText(), "first line\nsecond line\nthird line")
            block = editor.document().firstBlock()
            text_height = 0.0
            while block.isValid():
                text_height += editor.blockBoundingRect(block).height()
                block = block.next()

            self.assertGreater(editor.height(), 44)
            self.assertGreaterEqual(editor.viewport().height(), int(text_height))
        finally:
            workspace.close()
            workspace.deleteLater()

    def test_notes_overlay_reuses_provided_notes_workspace(self) -> None:
        import inspect
        from insight_local.cvops.ui.assistant_overlay import AssistantOverlayWindow

        params = inspect.signature(AssistantOverlayWindow.__init__).parameters
        self.assertIn("workspace_provider", params)
        self.assertIn("workspace_restorer", params)
        self.assertNotIn("notes_vault", params)

    def test_dataset_upload_finds_yolo_mirrored_labels(self) -> None:
        from insight_local.cvops.ui.dataset_panel import _matching_upload_label_path

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "images" / "train" / "part" / "sample.jpg"
            label = root / "labels" / "train" / "part" / "sample.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"fake")
            label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")

            self.assertEqual(_matching_upload_label_path(image), label)

    def test_dataset_upload_infers_supported_splits(self) -> None:
        from insight_local.cvops.ui.dataset_panel import _infer_upload_split

        self.assertEqual(_infer_upload_split(Path("/tmp/ds/images/train/a.jpg"), "val"), "train")
        self.assertEqual(_infer_upload_split(Path("/tmp/ds/images/valid/a.jpg"), "train"), "val")
        self.assertEqual(_infer_upload_split(Path("/tmp/ds/images/test/a.jpg"), "train"), "val")
        self.assertEqual(_infer_upload_split(Path("/tmp/ds/custom/a.jpg"), "val"), "val")


if __name__ == "__main__":
    unittest.main()
