from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

try:
    from PyQt6.QtWidgets import QApplication
except Exception:
    QApplication = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class CvOpsWindowIaTests(unittest.TestCase):
    """Lock down the post-rework Information Architecture of CvOpsWindow.

    The CV Ops UI was collapsed from nine flat tabs into five stage-based tabs
    (Data, Train, Verify, Improve, Diagnostics) that mirror the canonical
    pipeline. Downstream stages must be disabled until their prerequisites are
    met, and a successful train job must surface a Verify hand-off toast.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

        from insight_local.cvops import window as win_mod

        cls._patcher = mock.patch.object(win_mod, "CvOpsServerHandle")
        mock_server = cls._patcher.start()
        mock_server.return_value.start = lambda: None
        mock_server.return_value.stop = lambda **_kw: None

        cls._win_mod = win_mod
        cls._win = win_mod.CvOpsWindow(host="127.0.0.1", port=0)
        # Stub HTTP so any timer-driven refreshes don't hit the network.
        cls._win._http_json = lambda *_a, **_kw: {}  # type: ignore[assignment]
        cls._win._http_text = lambda *_a, **_kw: ""  # type: ignore[assignment]

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls._win.close()
        except Exception:
            pass
        cls._patcher.stop()

    def setUp(self) -> None:
        # Reset per-test mutable state.
        self._win._scenarios_cache = []
        self._win._toast.setVisible(False)
        self._win._toast_target_scenario = ""
        self._win._tabs.setCurrentIndex(self._win._tab_data)
        self._win._apply_stage_gating()

    @staticmethod
    def _is_shown(widget) -> bool:
        """isVisible() requires ancestors to be shown; in offscreen tests the
        QMainWindow is never shown, so check the inverse instead."""
        return not widget.isHidden()

    def test_five_stage_tabs(self) -> None:
        win = self._win
        self.assertEqual(win._tabs.count(), 5)
        self.assertEqual(win._tab_data, 0)
        self.assertEqual(win._tab_train, 1)
        self.assertEqual(win._tab_verify, 2)
        self.assertEqual(win._tab_improve, 3)
        self.assertEqual(win._tab_diagnostics, 4)
        self.assertEqual(win._tabs.tabText(win._tab_data), "Data")
        self.assertTrue(win._tabs.tabText(win._tab_diagnostics).startswith("Diagnostics"))

    def test_downstream_tabs_disabled_when_no_scenarios(self) -> None:
        win = self._win
        win._scenarios_cache = []
        win._apply_stage_gating()
        self.assertTrue(win._tabs.isTabEnabled(win._tab_data))
        self.assertFalse(win._tabs.isTabEnabled(win._tab_train))
        self.assertFalse(win._tabs.isTabEnabled(win._tab_verify))
        self.assertFalse(win._tabs.isTabEnabled(win._tab_improve))
        self.assertTrue(win._tabs.isTabEnabled(win._tab_diagnostics))
        # Disabled tabs annotate the missing prerequisite in the label.
        self.assertIn("needs", win._tabs.tabText(win._tab_train))

    def test_train_unlocks_when_scenario_exists(self) -> None:
        win = self._win
        win._scenarios_cache = [{"name": "demo", "status": "empty"}]
        win._apply_stage_gating()
        self.assertTrue(win._tabs.isTabEnabled(win._tab_train))
        self.assertEqual(win._tabs.tabText(win._tab_train), "Train")
        self.assertFalse(win._tabs.isTabEnabled(win._tab_verify))
        self.assertFalse(win._tabs.isTabEnabled(win._tab_improve))

    def test_verify_unlocks_when_scenario_is_trained(self) -> None:
        win = self._win
        win._scenarios_cache = [
            {"name": "demo", "status": "trained", "weights_ready": True}
        ]
        win._apply_stage_gating()
        self.assertTrue(win._tabs.isTabEnabled(win._tab_verify))
        self.assertFalse(win._tabs.isTabEnabled(win._tab_improve))

    def test_improve_unlocks_when_scenario_is_verified(self) -> None:
        win = self._win
        win._scenarios_cache = [
            {"name": "demo", "status": "trained", "weights_ready": True, "verified": True}
        ]
        win._apply_stage_gating()
        self.assertTrue(win._tabs.isTabEnabled(win._tab_improve))

    def test_train_success_shows_verify_toast(self) -> None:
        win = self._win
        win._scenarios_cache = [
            {"name": "demo", "status": "trained", "weights_ready": True}
        ]
        win._apply_stage_gating()
        self.assertFalse(self._is_shown(win._toast))

        win._on_ws_job_status(
            {
                "job_id": "j-1",
                "job_type": "train",
                "scenario": "demo",
                "state": "succeeded",
            }
        )
        self.assertTrue(self._is_shown(win._toast))
        self.assertIn("demo", win._toast.text())
        self.assertEqual(win._toast_target_scenario, "demo")

        win._on_toast_clicked(None)
        self.assertEqual(win._tabs.currentIndex(), win._tab_verify)
        self.assertFalse(self._is_shown(win._toast))

    def test_no_racing_metaphor_in_visible_strings(self) -> None:
        """The Marathon / Race Control copy theme is gone."""
        win = self._win
        haystacks = [
            win.windowTitle(),
            win._status.text(),
            win._ws_status.text(),
        ]
        for i in range(win._tabs.count()):
            haystacks.append(win._tabs.tabText(i))
        joined = " ".join(haystacks).lower()
        for needle in (
            "race control",
            "race entry",
            "race plans",
            "start corral",
            "finish results",
            "training grounds",
            "marathon",
            "issues log",
        ):
            self.assertNotIn(needle, joined, f"unexpected racing wording: {needle!r}")

    def test_human_ui_attribute_removed(self) -> None:
        """No code path should branch on the legacy _human_ui flag."""
        win = self._win
        self.assertFalse(hasattr(win, "_human_ui"))


if __name__ == "__main__":
    unittest.main()
