"""Tests for reindexing already-ingested chats into transferable memory.

Covers the ``force`` / all-sources / progress additions to
:func:`insight_local.cvops.ui.notes_ai_memory.build_memory_docs`, which let a
user rebuild memory from chats that are already on disk (no re-import needed).
Qt-free; summarization is left off so no local model is called.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.ui import notes_ai_import as imp
from insight_local.cvops.ui import notes_ai_memory as mem


def _make_vault(td: Path) -> tuple[Path, Path, Path]:
    vault = td / "notes"
    spaces = vault / "spaces"
    (spaces / "main").mkdir(parents=True, exist_ok=True)
    chats_dir = vault / "chats" / "main"
    chats_dir.mkdir(parents=True, exist_ok=True)
    return vault, spaces, chats_dir


def _seed_mixed(chats_dir: Path, n_claude: int, n_gpt: int) -> None:
    claude = [
        {
            "uuid": f"c{i}",
            "name": f"cl {i}",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:00:00Z",
            "chat_messages": [
                {"sender": "human", "text": "hi", "created_at": "2025-01-01T00:00:00Z"},
                {"sender": "assistant", "text": "yo", "created_at": "2025-01-01T00:00:01Z"},
            ],
        }
        for i in range(n_claude)
    ]
    imp.write_chats_to_dir(imp.parse_claude(claude), chats_dir, source="claude")
    gpt = []
    for i in range(n_gpt):
        gpt.append(
            {
                "id": f"g{i}",
                "title": f"gpt {i}",
                "create_time": 1_700_000_000 + i,
                "update_time": 1_700_000_100 + i,
                "mapping": {
                    "a": {
                        "id": "a",
                        "message": {
                            "author": {"role": "user"},
                            "content": {"content_type": "text", "parts": ["hello"]},
                            "create_time": 1_700_000_000 + i,
                        },
                        "parent": None,
                    },
                    "b": {
                        "id": "b",
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"content_type": "text", "parts": ["hi there"]},
                            "create_time": 1_700_000_001 + i,
                        },
                        "parent": "a",
                    },
                },
                "current_node": "b",
            }
        )
    imp.write_chats_to_dir(imp.parse_chatgpt(gpt), chats_dir, source="chatgpt")


class ReindexTests(unittest.TestCase):
    def test_reindex_all_sources_with_progress(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, spaces, chats_dir = _make_vault(Path(td))
            _seed_mixed(chats_dir, 5, 3)

            seen: list[tuple[int, int]] = []
            res = mem.build_memory_docs(
                vault_root=vault,
                spaces_root=spaces,
                source="all",
                summarize=False,
                force=True,
                on_progress=lambda d, t: seen.append((d, t)),
            )
            self.assertEqual(res["source"], "all")
            self.assertEqual(res["transcripts"], 8)
            self.assertEqual(res["chats"], 8)
            # Docs land under their own source dirs.
            self.assertEqual(mem.memory_doc_count(vault, source="claude"), 5)
            self.assertEqual(mem.memory_doc_count(vault, source="chatgpt"), 3)
            # Progress is cumulative and reaches the total.
            self.assertEqual(seen[0], (0, 8))
            self.assertEqual(seen[-1], (8, 8))

    def test_force_regenerates_but_does_not_duplicate_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, spaces, chats_dir = _make_vault(Path(td))
            _seed_mixed(chats_dir, 4, 0)

            mem.build_memory_docs(
                vault_root=vault, spaces_root=spaces, source="all",
                summarize=False, force=True,
            )
            batches_1 = len(mem.read_ledger(vault)["batches"])

            res2 = mem.build_memory_docs(
                vault_root=vault, spaces_root=spaces, source="all",
                summarize=False, force=True,
            )
            batches_2 = len(mem.read_ledger(vault)["batches"])
            # Transcripts are rewritten, but no duplicate provenance batches.
            self.assertEqual(res2["transcripts"], 4)
            self.assertEqual(batches_1, batches_2)

    def test_non_force_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vault, spaces, chats_dir = _make_vault(Path(td))
            _seed_mixed(chats_dir, 3, 0)

            first = mem.build_memory_docs(
                vault_root=vault, spaces_root=spaces, source="claude", summarize=False,
            )
            self.assertEqual(first["transcripts"], 3)
            second = mem.build_memory_docs(
                vault_root=vault, spaces_root=spaces, source="claude", summarize=False,
            )
            # Already on disk -> nothing rewritten, no new batch.
            self.assertEqual(second["transcripts"], 0)
            self.assertEqual(len(mem.read_ledger(vault)["batches"]), 1)


if __name__ == "__main__":
    unittest.main()
