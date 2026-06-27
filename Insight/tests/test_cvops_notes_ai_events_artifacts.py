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
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
except Exception:  # pragma: no cover - import availability is environment-specific
    QApplication = None  # type: ignore[assignment]
    Qt = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class NotesAiEventsArtifactsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_workspace_persists_chat_jobs_and_artifacts(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace
        from insight_local.cvops.ui.notes_spaces import ensure_notes_spaces_layout

        with tempfile.TemporaryDirectory() as td:
            spaces_root = ensure_notes_spaces_layout(Path(td) / "notes")
            workspace = NotesAiWorkspace()
            try:
                workspace.set_space_root(spaces_root / "main")
                chat_id = workspace.current_chat_id()

                workspace.record_chat_job(
                    {
                        "job_id": "job-test",
                        "scenario": "fall_detection",
                        "job_type": "train",
                        "state": "queued",
                    }
                )
                workspace.apply_cvops_event(
                    {
                        "type": "job_status",
                        "job_id": "job-test",
                        "scenario": "fall_detection",
                        "job_type": "train",
                        "state": "running",
                    }
                )
                workspace.record_chat_artifact("metrics.json", str(Path(td) / "metrics.json"))

                assert workspace._chat_mgr is not None
                chat = workspace._chat_mgr.chats[chat_id]
                metadata = chat.get("metadata") or {}
                jobs = metadata.get("cvops_jobs") or []
                artifacts = metadata.get("cvops_artifacts") or []

                self.assertEqual(jobs[0]["job_id"], "job-test")
                self.assertEqual(jobs[0]["state"], "running")
                self.assertEqual(artifacts[0]["label"], "metrics.json")
                self.assertTrue((spaces_root / "main" / "events_artifacts.json").is_file())
            finally:
                workspace.close()
                workspace.deleteLater()

    def test_running_indicator_uses_measured_job_state(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace
        from insight_local.cvops.ui.notes_spaces import ensure_notes_spaces_layout

        with tempfile.TemporaryDirectory() as td:
            spaces_root = ensure_notes_spaces_layout(Path(td) / "notes")
            workspace = NotesAiWorkspace()
            try:
                workspace.set_space_root(spaces_root / "main")
                chat_id = workspace.current_chat_id()

                def fake_http(method: str, path: str, payload=None):
                    if method == "GET" and path == "/jobs/job-canceled":
                        return {
                            "job_id": "job-canceled",
                            "scenario": "demo",
                            "job_type": "train",
                            "state": "canceled",
                        }
                    raise RuntimeError(path)

                workspace._cvops_http_json = fake_http  # type: ignore[method-assign]
                workspace.record_chat_job(
                    {
                        "job_id": "job-canceled",
                        "scenario": "demo",
                        "job_type": "train",
                        "state": "queued",
                    }
                )

                target = workspace._encode_chat_ref("main", chat_id)
                host = None
                for listw in (workspace.chat_list_pinned, workspace.chat_list_recent):
                    for i in range(listw.count()):
                        item = listw.item(i)
                        if item and item.data(Qt.ItemDataRole.UserRole) == target:
                            host = listw.itemWidget(item)
                            break
                self.assertIsNotNone(host)
                self.assertFalse(host.has_active_jobs())
                self.assertFalse(workspace._activity_timer.isActive())

                chat = workspace._chat_mgr.chats[chat_id]  # type: ignore[union-attr]
                jobs = (chat.get("metadata") or {}).get("cvops_jobs") or []
                self.assertEqual(jobs[0]["state"], "canceled")
            finally:
                workspace.close()
                workspace.deleteLater()

    def test_events_artifacts_panel_toggle(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace

        workspace = NotesAiWorkspace()
        try:
            panel = workspace._events_artifacts_panel
            btn = workspace._btn_chat_header_artifacts
            # Shown by default.
            self.assertTrue(btn.isChecked())
            self.assertFalse(panel.isHidden())
            # Toggling off hides the panel; toggling on restores it.
            btn.setChecked(False)
            self.assertTrue(panel.isHidden())
            btn.setChecked(True)
            self.assertFalse(panel.isHidden())
        finally:
            workspace.close()
            workspace.deleteLater()


if __name__ == "__main__":
    unittest.main()
