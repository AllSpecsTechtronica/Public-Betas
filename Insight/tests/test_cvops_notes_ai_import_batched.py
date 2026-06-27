"""Tests for batched, progress-reporting chat ingestion in cvops notes AI.

Covers the pacing + progress-callback additions to
:mod:`insight_local.cvops.ui.notes_ai_import`, which is Qt-free so no
QApplication is required. These guarantee a large export is written in batches
(reporting cumulative ``(done, total)`` so the UI can show an ETA) without
changing what lands on disk.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.ui import notes_ai_import as ai


def _claude_export(n: int) -> list[dict]:
    """A Claude-shaped export payload with ``n`` two-message conversations."""
    convs = []
    for i in range(n):
        convs.append(
            {
                "uuid": f"u{i}",
                "name": f"chat {i}",
                "created_at": "2025-01-01T00:00:00Z",
                "updated_at": "2025-01-01T00:00:01Z",
                "chat_messages": [
                    {"sender": "human", "text": "hi", "created_at": "2025-01-01T00:00:00Z"},
                    {"sender": "assistant", "text": "yo", "created_at": "2025-01-01T00:00:01Z"},
                ],
            }
        )
    return convs


class BatchedIngestTests(unittest.TestCase):
    def test_progress_is_cumulative_and_complete(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "conversations.json"
            src.write_text(json.dumps(_claude_export(120)), encoding="utf-8")
            chats_dir = Path(td) / "chats"

            seen: list[tuple[int, int]] = []
            res = ai.ingest_export_file(
                src,
                chats_dir,
                source="claude",
                batch_size=25,
                batch_pause=0.0,
                on_progress=lambda d, t: seen.append((d, t)),
            )

            self.assertEqual(res.chats_written, 120)
            self.assertEqual(res.messages_written, 240)
            # One initial (0, total) plus one per conversation.
            self.assertEqual(seen[0], (0, 120))
            self.assertEqual(seen[-1], (120, 120))
            self.assertTrue(all(t == 120 for _, t in seen))
            dones = [d for d, _ in seen]
            self.assertEqual(dones, sorted(dones))  # monotonic non-decreasing
            self.assertEqual(len(list(chats_dir.glob("chat_*.json"))), 120)

    def test_batch_pause_paces_the_work(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "conversations.json"
            src.write_text(json.dumps(_claude_export(100)), encoding="utf-8")
            chats_dir = Path(td) / "chats"

            t0 = time.monotonic()
            ai.ingest_export_file(
                src, chats_dir, source="claude", batch_size=25, batch_pause=0.02
            )
            elapsed = time.monotonic() - t0
            # 100 chats / 25 per batch -> 3 internal batch boundaries -> >= ~0.06s.
            self.assertGreaterEqual(elapsed, 0.05)

    def test_on_chat_fires_for_skipped_and_empty(self) -> None:
        # Empty + duplicate conversations still advance progress so the bar
        # reaches 100% even when nothing is written for those entries.
        chats = [
            ai._make_chat(
                title="empty",
                created_at="2025-01-01T00:00:00",
                updated_at="2025-01-01T00:00:00",
                messages=[],
                source="claude",
                original_id="e1",
            ),
            ai._make_chat(
                title="real",
                created_at="2025-01-01T00:00:00",
                updated_at="2025-01-01T00:00:00",
                messages=[ai._make_message("user", "hi", "2025-01-01T00:00:00")],
                source="claude",
                original_id="r1",
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            ticks = []
            res = ai.write_chats_to_dir(
                chats, Path(td) / "chats", source="claude",
                on_chat=lambda: ticks.append(1),
            )
            self.assertEqual(len(ticks), 2)  # both processed
            self.assertEqual(res.chats_written, 1)
            self.assertEqual(res.skipped_empty, 1)


if __name__ == "__main__":
    unittest.main()
