"""Tests for ingested-chat -> RAG memory (knowledge transfer) in cvops notes AI.

These cover :mod:`insight_local.cvops.ui.notes_ai_memory`, which is Qt-free, so
no QApplication is required. Summarization is left off (it would call a local
model); the build path is exercised with ``summarize=False``.
"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.ui import notes_ai_memory as mem


def _make_vault(td: Path) -> tuple[Path, Path]:
    """Create ``<td>/notes`` with a single ``main`` space + chats dir."""
    vault = td / "notes"
    spaces_root = vault / "spaces"
    (spaces_root / "main").mkdir(parents=True, exist_ok=True)
    (vault / "chats" / "main").mkdir(parents=True, exist_ok=True)
    return vault, spaces_root


def _write_imported_chat(
    vault: Path, chat_id: str, *, source: str, title: str, messages: list[dict]
) -> None:
    chat = {
        "id": chat_id,
        "title": title,
        "created_at": "2024-01-01T00:00:00",
        "updated_at": "2024-01-01T00:00:00",
        "messages": messages,
        "metadata": {"imported": True, "import_source": source, "original_id": chat_id},
    }
    path = vault / "chats" / "main" / f"{chat_id}.json"
    path.write_text(json.dumps(chat), encoding="utf-8")


def _chatgpt_zip(path: Path, *, custom_instructions: str) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("conversations.json", json.dumps([]))
        zf.writestr("user.json", json.dumps({"custom_instructions": custom_instructions}))
    path.write_bytes(buf.getvalue())


class BuildMemoryDocsTests(unittest.TestCase):
    def test_build_writes_transcripts_and_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, spaces_root = _make_vault(Path(td))
            _write_imported_chat(
                vault,
                "chat_1",
                source="chatgpt",
                title="Quantum notes",
                messages=[
                    {"role": "user", "content": "What is superposition?"},
                    {"role": "assistant", "content": "A quantum state combination."},
                ],
            )

            result = mem.build_memory_docs(
                vault_root=vault, spaces_root=spaces_root, source="chatgpt"
            )

            self.assertEqual(result["transcripts"], 1)
            self.assertEqual(result["chats"], 1)
            doc = vault / mem.INGESTED_MEMORY_DIRNAME / "chatgpt" / "transcript_chat_1.md"
            self.assertTrue(doc.is_file())
            self.assertIn("superposition", doc.read_text(encoding="utf-8"))

            ledger = mem.read_ledger(vault)
            self.assertEqual(len(ledger["batches"]), 1)
            self.assertEqual(ledger["batches"][0]["source"], "chatgpt")
            self.assertTrue(mem.ledger_summary_text(vault))

    def test_build_is_idempotent_per_chat(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, spaces_root = _make_vault(Path(td))
            _write_imported_chat(
                vault, "chat_1", source="chatgpt", title="A",
                messages=[{"role": "user", "content": "hi"}],
            )
            mem.build_memory_docs(vault_root=vault, spaces_root=spaces_root, source="chatgpt")
            again = mem.build_memory_docs(vault_root=vault, spaces_root=spaces_root, source="chatgpt")
            # Already has a transcript -> nothing new written.
            self.assertEqual(again["transcripts"], 0)
            self.assertEqual(len(mem.memory_doc_paths(vault)), 1)

    def test_detected_memory_file_indexed(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, spaces_root = _make_vault(Path(td))
            _write_imported_chat(
                vault, "chat_1", source="chatgpt", title="A",
                messages=[{"role": "user", "content": "hi"}],
            )
            export = Path(td) / "chatgpt_export.zip"
            _chatgpt_zip(export, custom_instructions="Call me Ada. I prefer Python.")

            result = mem.build_memory_docs(
                vault_root=vault, spaces_root=spaces_root, source="chatgpt", export_path=export
            )
            self.assertEqual(result["memory_files"], 1)
            mfile = vault / mem.INGESTED_MEMORY_DIRNAME / "chatgpt" / "memory_chatgpt_instructions.md"
            self.assertTrue(mfile.is_file())
            self.assertIn("Ada", mfile.read_text(encoding="utf-8"))


class DetectMemoryFilesTests(unittest.TestCase):
    def test_chatgpt_zip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            export = Path(td) / "e.zip"
            _chatgpt_zip(export, custom_instructions="be terse")
            docs = mem.detect_memory_files(export)
            self.assertEqual(len(docs), 1)
            self.assertIn("be terse", docs[0]["content"])

    def test_claude_projects_folder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            export = Path(td) / "claude"
            export.mkdir()
            projects = [
                {
                    "uuid": "p1",
                    "name": "Thesis",
                    "instructions": "Write formally.",
                    "docs": [{"filename": "outline.md", "content": "Chapter 1: intro"}],
                }
            ]
            (export / "projects.json").write_text(json.dumps(projects), encoding="utf-8")
            docs = mem.detect_memory_files(export)
            self.assertEqual(len(docs), 1)
            body = docs[0]["content"]
            self.assertIn("Write formally.", body)
            self.assertIn("Chapter 1: intro", body)

    def test_missing_export_is_empty(self) -> None:
        self.assertEqual(mem.detect_memory_files(Path("/no/such/path.zip")), [])


class DeletionTests(unittest.TestCase):
    def _seed(self, td: Path) -> tuple[Path, Path]:
        vault, spaces_root = _make_vault(td)
        _write_imported_chat(
            vault, "chat_1", source="chatgpt", title="A",
            messages=[{"role": "user", "content": "hi"}],
        )
        _write_imported_chat(
            vault, "chat_2", source="claude", title="B",
            messages=[{"role": "user", "content": "yo"}],
        )
        mem.build_memory_docs(vault_root=vault, spaces_root=spaces_root, source="chatgpt")
        mem.build_memory_docs(vault_root=vault, spaces_root=spaces_root, source="claude")
        return vault, spaces_root

    def test_keep_memory_marks_flag(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, _ = self._seed(Path(td))
            self.assertEqual(mem.memory_doc_count(vault, source="chatgpt"), 1)

            n = mem.mark_chats_deleted(vault, source="chatgpt")
            self.assertEqual(n, 1)
            # Docs survive; the batch is flagged.
            self.assertEqual(mem.memory_doc_count(vault, source="chatgpt"), 1)
            ledger = mem.read_ledger(vault)
            flagged = [b for b in ledger["batches"] if b["source"] == "chatgpt"]
            self.assertTrue(all(b["chats_deleted"] for b in flagged))
            self.assertIn("retained", mem.ledger_summary_text(vault))

    def test_delete_memory_removes_source_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, _ = self._seed(Path(td))
            removed = mem.delete_memory_docs(vault, source="chatgpt")
            self.assertEqual(removed, 1)
            self.assertEqual(mem.memory_doc_count(vault, source="chatgpt"), 0)
            # Claude memory is untouched and remains available for a rebuild.
            self.assertEqual(mem.memory_doc_count(vault, source="claude"), 1)
            self.assertEqual(len(mem.memory_doc_paths(vault)), 1)
            ledger = mem.read_ledger(vault)
            self.assertTrue(all(b["source"] != "chatgpt" for b in ledger["batches"]))

    def test_delete_all_memory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, _ = self._seed(Path(td))
            mem.delete_memory_docs(vault, source=None)
            self.assertEqual(mem.memory_doc_paths(vault), [])
            self.assertEqual(mem.read_ledger(vault)["batches"], [])


if __name__ == "__main__":
    unittest.main()
