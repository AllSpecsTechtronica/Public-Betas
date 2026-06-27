from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))


class NotesAiMentionTests(unittest.TestCase):
    def test_extract_file_mentions_supports_paths_and_quoted_names(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import _extract_file_mentions

        mentions = _extract_file_mentions(
            'Compare @README.md with @"docs/label rules.md" and ignore mail@example.com.'
        )

        self.assertEqual(mentions, ["README.md", "docs/label rules.md"])

    def test_read_text_excerpt_bounds_large_files(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import _read_text_excerpt

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "note.md"
            path.write_text("abcdef", encoding="utf-8")

            text, truncated = _read_text_excerpt(path, max_chars=3)

        self.assertEqual(text, "abc")
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
