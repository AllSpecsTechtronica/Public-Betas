from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))


class CvOpsNotesTranscriptionTests(unittest.TestCase):
    def test_format_transcript_markdown_preserves_text_and_segments(self) -> None:
        from insight_local.cvops.notes_transcription import format_transcript_markdown

        with tempfile.TemporaryDirectory() as tmp:
            notes_root = Path(tmp) / "notes" / "spaces" / "main"
            audio = notes_root / "recordings" / "voice.wav"
            payload = {
                "source_path": str(audio),
                "source_name": "voice.wav",
                "provider": "audio_asr",
                "model": "vosk",
                "capability": "ok",
                "created_at": "2026-06-05T12:00:00+00:00",
                "text": "First note.\n\nSecond note.",
                "blocks": [
                    {
                        "text": "First note.",
                        "raw_region": {"kind": "audio_segment", "start_sec": 0.0, "end_sec": 1.25},
                    },
                    {
                        "text": "Second note.",
                        "metadata": {"start_sec": 1.25, "end_sec": 2.5},
                    },
                ],
            }

            markdown = format_transcript_markdown(payload, notes_root=notes_root)

        self.assertIn("# Transcript: voice.wav", markdown)
        self.assertIn("- Source: `recordings/voice.wav`", markdown)
        self.assertIn("First note.\n\nSecond note.", markdown)
        self.assertIn("- [00:00.000 - 00:01.250] First note.", markdown)
        self.assertIn("- [00:01.250 - 00:02.500] Second note.", markdown)

    def test_transcribe_audio_note_uses_vosk_by_default(self) -> None:
        from insight_local.cvops import notes_transcription
        from insight_local.cvops.notes_transcription import transcribe_audio_note

        blocks = [
            {
                "block_kind": "audio_transcript",
                "text": "meeting note",
                "raw_region": {"kind": "audio_segment", "start_sec": 0.0, "end_sec": 0.75},
            }
        ]
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "voice.wav"
            audio.write_bytes(b"fake wav")
            with mock.patch.object(notes_transcription, "_transcribe_vosk", return_value=(blocks, "ok")) as asr:
                payload = transcribe_audio_note(audio)

        asr.assert_called_once()
        self.assertEqual(payload["capability"], "ok")
        self.assertEqual(payload["source_name"], "voice.wav")
        self.assertEqual(payload["model"], "vosk")
        self.assertEqual(payload["text"], "meeting note")
        self.assertEqual(payload["blocks"][0]["block_kind"], "audio_transcript")

    def test_transcribe_audio_note_can_still_route_to_whisper(self) -> None:
        from insight_local.cvops import notes_transcription
        from insight_local.cvops.notes_transcription import transcribe_audio_note

        blocks = [{"block_kind": "audio_transcript", "text": "cloud fallback"}]
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "voice.wav"
            audio.write_bytes(b"fake wav")
            with mock.patch.object(notes_transcription, "_transcribe_whisper", return_value=(blocks, "ok")) as asr:
                payload = transcribe_audio_note(audio, provider="whisper_tiny")

        asr.assert_called_once()
        self.assertEqual(payload["model"], "whisper_tiny")
        self.assertEqual(payload["text"], "cloud fallback")

    def test_rejects_non_audio_note_suffix(self) -> None:
        from insight_local.cvops.notes_transcription import transcribe_audio_note

        with tempfile.TemporaryDirectory() as tmp:
            note = Path(tmp) / "note.txt"
            note.write_text("hello", encoding="utf-8")
            with self.assertRaises(ValueError):
                transcribe_audio_note(note)


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    unittest.main()
