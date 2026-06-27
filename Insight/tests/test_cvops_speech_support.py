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
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]


class VoiceProfileSchemaTests(unittest.TestCase):
    """Qt-free: voice-profile schema, presets, and ffmpeg filter builder."""

    def test_default_profile_is_tuned_tacitus(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import default_voice_profile

        p = default_voice_profile()
        self.assertEqual(p["name"], "Tacitus")
        self.assertEqual(p["base_voice"], "Rishi")      # the en_IN base we settled on
        self.assertLess(p["pitch_semitones"], 0)        # lower than natural
        self.assertGreater(p["warmth_db"], 0)           # body
        self.assertLessEqual(p["room"], 0.4)            # present but not theatrical
        # Smoothing/naturalness defaults are engaged.
        self.assertGreater(p["sibilance"], 0)           # de-ess on
        self.assertGreater(p["smoothing"], 0)           # leveling on

    def test_presets_present(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import VOICE_PRESETS

        self.assertEqual(set(VOICE_PRESETS), {"Tacitus", "Minimal", "Spartan-comms"})

    def test_normalize_clamps_and_drops_unknown(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import _normalized_voice_profile

        out = _normalized_voice_profile(
            {
                "name": "x" * 200,
                "base_voice": "  Daniel  ",
                "rate_wpm": 99999,
                "pitch_semitones": "-3",
                "room": 5.0,
                "comms_bandpass": 1,
                "bogus": "drop me",
            }
        )
        self.assertEqual(out["rate_wpm"], 320.0)        # clamped to max
        self.assertEqual(out["pitch_semitones"], -3.0)  # coerced from str
        self.assertEqual(out["room"], 1.0)              # clamped to max
        self.assertEqual(out["base_voice"], "Daniel")   # trimmed
        self.assertIs(out["comms_bandpass"], True)
        self.assertNotIn("bogus", out)
        self.assertLessEqual(len(out["name"]), 60)

    def test_build_ffmpeg_filter_tacitus_and_noop(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import default_voice_profile
        from insight_local.cvops.ui.speech_support import build_ffmpeg_filter

        chain = build_ffmpeg_filter(default_voice_profile())
        # Duration-preserving pitch shift + warmth + de-harsh + faint room.
        self.assertIn("atempo=", chain)
        self.assertIn("bass=g=", chain)
        self.assertIn("lowpass=f=", chain)
        self.assertIn("aecho=", chain)
        self.assertNotIn("highpass=f=450", chain)       # no comms band-pass
        # Fine-tuning / smoothing stages are present in the Tacitus default.
        self.assertIn("deesser=", chain)                # tame sibilance
        self.assertIn("dynaudnorm=", chain)             # loudness leveling
        self.assertIn("equalizer=f=3000", chain)        # presence/clarity
        self.assertIn("treble=g=", chain)               # air
        self.assertIn("chorus=", chain)                 # depth/richness

        flat = {
            "pitch_semitones": 0, "warmth_db": 0, "high_cut_hz": 0,
            "room": 0, "comms_bandpass": False,
            "low_cut_hz": 0, "presence_db": 0, "air_db": 0,
            "sibilance": 0, "smoothing": 0, "depth": 0,
        }
        self.assertEqual(build_ffmpeg_filter(flat), "")  # no-op -> no ffmpeg pass

    def test_build_ffmpeg_filter_comms_bandpass(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import VOICE_PRESETS
        from insight_local.cvops.ui.speech_support import build_ffmpeg_filter

        chain = build_ffmpeg_filter(VOICE_PRESETS["Spartan-comms"])
        self.assertIn("highpass=f=450", chain)
        self.assertIn("lowpass=f=3000", chain)

    def test_settings_round_trip_preserves_voice(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as k

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            orig = k.ai_settings_path
            k.ai_settings_path = lambda: path  # type: ignore[assignment]
            try:
                settings = k.default_ai_settings()
                settings[k.KEY_VOICE_PROFILE] = {
                    "name": "Custom", "base_voice": "Daniel",
                    "rate_wpm": 150, "pitch_semitones": -3.5,
                    "warmth_db": 4, "high_cut_hz": 8000, "room": 0.2,
                    "comms_bandpass": True,
                    "low_cut_hz": 75, "presence_db": 2.5, "air_db": 1.0,
                    "sibilance": 0.5, "smoothing": 0.6, "depth": 0.2,
                }
                settings[k.KEY_OPENAI] = "sk-test"
                k.save_ai_settings(settings)
                loaded = k.load_ai_settings()
            finally:
                k.ai_settings_path = orig  # type: ignore[assignment]

        vp = loaded[k.KEY_VOICE_PROFILE]
        self.assertEqual(vp["base_voice"], "Daniel")
        self.assertEqual(vp["pitch_semitones"], -3.5)
        self.assertIs(vp["comms_bandpass"], True)
        # Fine-tuning fields survive the round-trip too.
        self.assertEqual(vp["smoothing"], 0.6)
        self.assertEqual(vp["presence_db"], 2.5)
        # API-key handling is untouched by the new branch.
        self.assertEqual(loaded[k.KEY_OPENAI], "sk-test")

    def test_list_system_voices_returns_list(self) -> None:
        from insight_local.cvops.ui.speech_support import list_system_voices

        self.assertIsInstance(list_system_voices(), list)


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class CvOpsSpeechSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_tts_playback_bar_starts_hidden(self) -> None:
        from insight_local.cvops.ui.speech_support import TtsPlaybackBar

        bar = TtsPlaybackBar()
        try:
            self.assertFalse(bar.isVisible())
            # Empty text yields no synthesis and must not raise.
            bar.speak("")
            self.assertFalse(bar.isVisible())
        finally:
            bar.deleteLater()

    def test_dictation_controller_idle_by_default(self) -> None:
        from insight_local.cvops.ui.speech_support import SpeechDictationController

        ctrl = SpeechDictationController()
        self.assertFalse(ctrl.is_recording())
        self.assertFalse(ctrl.is_busy())
        # Stopping when not recording is a no-op.
        ctrl.stop()
        self.assertFalse(ctrl.is_recording())

    def test_synthesize_speech_empty_returns_none(self) -> None:
        from insight_local.cvops.ui.speech_support import synthesize_speech

        self.assertIsNone(synthesize_speech(""))
        self.assertIsNone(synthesize_speech("   "))

    def test_speak_action_link_renders_when_tts_enabled(self) -> None:
        from insight_local.cvops.ui import notes_ai_workspace as mod
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace

        original = mod.text_to_speech_available
        mod.text_to_speech_available = lambda: True  # type: ignore[assignment]
        try:
            w = NotesAiWorkspace()
        finally:
            mod.text_to_speech_available = original  # type: ignore[assignment]
        try:
            w._tts_enabled = True
            html = w._render_chat_html(
                [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello there"},
                ]
            )
            self.assertIn("cvops-action://speak/1", html)
            self.assertIn("Play", html)

            # Disabled: the speak link disappears.
            w._tts_enabled = False
            html_off = w._render_chat_html(
                [{"role": "assistant", "content": "hello there"}]
            )
            self.assertNotIn("cvops-action://speak", html_off)
        finally:
            w.deleteLater()

    def test_composer_exposes_dictation_controls(self) -> None:
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace

        w = NotesAiWorkspace()
        try:
            self.assertTrue(hasattr(w, "_btn_compose_dictate"))
            self.assertTrue(hasattr(w, "_tts_bar"))
            self.assertTrue(hasattr(w, "_dictation"))
        finally:
            w.deleteLater()

    def test_voice_designer_form_round_trips(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import (
            VOICE_PRESETS,
            _normalized_voice_profile,
        )
        from insight_local.cvops.ui.notes_ai_workspace import NotesAiWorkspace

        w = NotesAiWorkspace()
        try:
            # The voice section is built and exposes its controls.
            self.assertTrue(hasattr(w, "_voice_preset_combo"))
            self.assertTrue(hasattr(w, "_voice_sliders"))

            # Applying a preset and reading the form back matches the preset.
            w._apply_voice_profile_to_form(VOICE_PRESETS["Spartan-comms"])
            got = w._voice_profile_from_form()
            want = _normalized_voice_profile(VOICE_PRESETS["Spartan-comms"])
            for key in (
                "pitch_semitones", "warmth_db", "high_cut_hz", "comms_bandpass",
                "low_cut_hz", "sibilance", "smoothing",
            ):
                self.assertEqual(got[key], want[key], key)

            # Editing a slider flips the preset selector to "Custom".
            w._voice_preset_combo.setCurrentText("Tacitus")
            w._voice_sliders["warmth_db"].setValue(95)
            self.assertEqual(w._voice_preset_combo.currentText(), "Custom")
        finally:
            w.deleteLater()


if __name__ == "__main__":
    unittest.main()
