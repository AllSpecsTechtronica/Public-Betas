from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

try:
    from PyQt6.QtWidgets import QApplication
except Exception:
    QApplication = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class CvOpsSubmitPanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_submit_panel_initializes_scenarios_and_registry_models(self) -> None:
        from insight_local.cvops.ui.submit_panel import SubmitPanel

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "registry_model.pt"
            model_path.write_bytes(b"weights")

            def fake_get(path: str) -> dict:
                if path == "/models":
                    return {
                        "models": [
                            {
                                "name": "demo:v2",
                                "value": "demo:v2",
                                "path": str(model_path),
                            }
                        ]
                    }
                return {}

            panel = SubmitPanel(
                base_url="http://127.0.0.1:8787",
                scenarios_provider=lambda: [{"name": "demo"}, {"name": "fall_detection"}],
                http_get=fake_get,
            )
            try:
                scenarios = [panel._combo.itemText(i) for i in range(panel._combo.count())]
                self.assertEqual(scenarios, ["[REGISTRY]", "demo", "fall_detection"])
                self.assertEqual(panel.current_scenario(), "[REGISTRY]")
                self.assertGreater(panel._registry_model_combo.count(), 0)
                labels = [
                    panel._registry_model_combo.itemText(i)
                    for i in range(panel._registry_model_combo.count())
                ]
                self.assertIn("demo:v2", labels)
                idx = panel._registry_model_combo.findText("demo:v2")
                self.assertGreaterEqual(idx, 0)
                self.assertEqual(panel._registry_model_combo.itemData(idx), str(model_path.resolve()))
            finally:
                panel.deleteLater()

    def test_submit_panel_shows_scenario_related_registry_models(self) -> None:
        from insight_local.cvops.ui.submit_panel import SubmitPanel

        def fake_get(path: str) -> dict:
            if path == "/models":
                return {
                    "models": [
                        {"name": "demo:v2", "value": "demo:v2", "path": "/tmp/demo-v2.pt"},
                        {"name": "other:v1", "value": "other:v1", "path": "/tmp/other-v1.pt"},
                    ]
                }
            if path == "/scenarios/demo/history":
                return {"runs": []}
            if path == "/scenarios/demo/status":
                return {"backbone_type": "yolo_detection"}
            return {}

        panel = SubmitPanel(
            base_url="http://127.0.0.1:8787",
            scenarios_provider=lambda: [{"name": "demo"}],
            http_get=fake_get,
        )
        try:
            panel._combo.setCurrentText("demo")
            labels = [panel._model_combo.itemText(i) for i in range(panel._model_combo.count())]
            values = [panel._model_combo.itemData(i) for i in range(panel._model_combo.count())]
            self.assertIn("Registry: v2", labels)
            self.assertIn("demo:v2", values)
            self.assertNotIn("other:v1", values)
        finally:
            panel.deleteLater()


if __name__ == "__main__":
    unittest.main()
