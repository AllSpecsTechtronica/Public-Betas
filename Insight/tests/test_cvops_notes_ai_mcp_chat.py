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
except Exception:  # pragma: no cover - import availability is environment-specific
    QApplication = None  # type: ignore[assignment]


class NotesAiMcpPromptTests(unittest.TestCase):
    def test_ollama_prompt_can_include_compact_mcp_catalog(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import (
            _build_ollama_prompt,
            _compact_tacitus_mcp_catalog,
        )

        prompt = _build_ollama_prompt(
            [{"role": "user", "content": "show pipeline for @scenario demo"}],
            assistant_name="Tacitus",
            mcp_context={"active_scenario": "demo", "selected_dataset": "Tiny"},
            mcp_catalog=_compact_tacitus_mcp_catalog(),
        )

        self.assertIn("[Tacitus MCP tools]", prompt)
        self.assertIn("exactly one JSON object", prompt)
        self.assertIn("pipeline_get", prompt)
        self.assertIn("active_scenario", prompt)

    def test_structured_mcp_tool_call_extractor_is_strict(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import _extract_structured_mcp_tool_call

        self.assertIsNone(_extract_structured_mcp_tool_call("Use the pipeline tool."))
        parsed = _extract_structured_mcp_tool_call(
            '```json\n{"tool":"pipeline_get","arguments":{"scenario":"demo"}}\n```'
        )
        self.assertIsNotNone(parsed)
        self.assertEqual((parsed or {})["tool"], "pipeline_get")


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class NotesAiMcpChatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_ollama_structured_mcp_response_is_dispatched_and_saved(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace
        from insight_local.cvops.ui.notes_spaces import ensure_notes_spaces_layout

        with tempfile.TemporaryDirectory() as td:
            spaces_root = ensure_notes_spaces_layout(Path(td) / "notes")
            workspace = NotesAiWorkspace()
            try:
                workspace.set_space_root(spaces_root / "main")
                chat_id = workspace.current_chat_id()

                def fake_http(method: str, path: str, payload=None):
                    if method == "GET" and path == "/scenarios/demo/pipeline":
                        return {
                            "scenario": "demo",
                            "ci_cd": {"enabled": True, "promotion": "manual"},
                            "candidate": {"version_id": "demo:v2"},
                            "prod": {"version_id": "demo:v1"},
                            "latest_gate": {"gate_status": "passed"},
                        }
                    raise RuntimeError(path)

                workspace._cvops_http_json = fake_http  # type: ignore[method-assign]
                workspace._streaming_provider = "ollama"
                workspace._streaming_mcp_enabled = True
                workspace._streaming_model_label = "local-test"

                workspace._on_chat_done(
                    {"full_response": '{"tool":"pipeline_get","arguments":{"scenario":"demo"}}'}
                )

                assert workspace._chat_mgr is not None
                messages = workspace._chat_mgr.get_chat_messages(chat_id)
                self.assertEqual(messages[-1]["role"], "assistant")
                self.assertIn("Loaded pipeline for demo.", messages[-1]["content"])
                self.assertIn("Candidate: `demo:v2`", messages[-1]["content"])
                self.assertEqual(messages[-1]["metadata"]["model_label"], "Tacitus MCP")
                self.assertEqual(messages[-1]["metadata"]["mcp_model_label"], "local-test")
                self.assertEqual(messages[-1]["metadata"]["mcp_result"]["tool"], "pipeline.get")

                ledger_path = spaces_root / "main" / "events_artifacts.json"
                self.assertIn("mcp_tool_call", ledger_path.read_text(encoding="utf-8"))
            finally:
                workspace.close()
                workspace.deleteLater()


if __name__ == "__main__":
    unittest.main()
