from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))


class NotesAiSettingsTests(unittest.TestCase):
    def test_assistant_name_defaults_to_tacitus(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import assistant_display_name

        self.assertEqual(assistant_display_name({}), "Tacitus")

    def test_assistant_name_persists_custom_value(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            with patch.object(mod, "ai_settings_path", return_value=path):
                mod.save_ai_settings({mod.KEY_ASSISTANT_NAME: "Tacitus Prime"})

                settings = mod.load_ai_settings()

        self.assertEqual(mod.assistant_display_name(settings), "Tacitus Prime")

    def test_local_gguf_models_persist_and_dedupe(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            gguf = str(Path(td) / "tiny.gguf")
            with patch.object(mod, "ai_settings_path", return_value=path):
                mod.save_ai_settings({mod.KEY_LOCAL_GGUF_MODELS: [gguf, gguf, ""]})

                settings = mod.load_ai_settings()

        self.assertEqual(mod.local_gguf_models(settings), [gguf])

    def test_system_prompt_defaults_empty(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import system_prompt

        self.assertEqual(system_prompt({}), "")

    def test_system_prompt_persists_and_trims(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            with patch.object(mod, "ai_settings_path", return_value=path):
                mod.save_ai_settings({mod.KEY_SYSTEM_PROMPT: "  Be terse.  "})

                settings = mod.load_ai_settings()

        self.assertEqual(mod.system_prompt(settings), "Be terse.")

    def test_system_prompt_clamped_to_max_chars(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            with patch.object(mod, "ai_settings_path", return_value=path):
                mod.save_ai_settings({mod.KEY_SYSTEM_PROMPT: "x" * (mod.SYSTEM_PROMPT_MAX_CHARS + 500)})

                settings = mod.load_ai_settings()

        self.assertEqual(len(mod.system_prompt(settings)), mod.SYSTEM_PROMPT_MAX_CHARS)

    def test_saving_other_keys_preserves_system_prompt(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            with patch.object(mod, "ai_settings_path", return_value=path):
                mod.save_ai_settings({mod.KEY_SYSTEM_PROMPT: "Stay grounded."})
                # A later save that omits the prompt should not silently keep it;
                # the form always passes the field, so omission means cleared.
                mod.save_ai_settings({mod.KEY_ASSISTANT_NAME: "Tacitus"})
                settings = mod.load_ai_settings()

        self.assertEqual(mod.system_prompt(settings), "")

    def test_keys_persist_to_keyring_not_json_when_available(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as mod

        store: dict[tuple[str, str], str] = {}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            with patch.object(mod, "ai_settings_path", return_value=path), \
                 patch.object(mod, "keyring_available", return_value=True), \
                 patch.object(mod, "_keyring_get", side_effect=lambda n: store.get((mod._KEYRING_SERVICE, n), "")), \
                 patch.object(mod, "_keyring_set", side_effect=lambda n, v: (store.__setitem__((mod._KEYRING_SERVICE, n), v) or True)):
                mod.save_ai_settings({mod.KEY_OPENAI: "sk-secret-123"})

                # The secret must NOT be written to the JSON file...
                on_disk = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(on_disk[mod.KEY_OPENAI], "")
                # ...but it round-trips through the keyring.
                settings = mod.load_ai_settings()

        self.assertEqual(settings[mod.KEY_OPENAI], "sk-secret-123")

    def test_keys_fall_back_to_json_without_keyring(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as mod

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            with patch.object(mod, "ai_settings_path", return_value=path), \
                 patch.object(mod, "keyring_available", return_value=False):
                mod.save_ai_settings({mod.KEY_GROK: "grok-key"})
                settings = mod.load_ai_settings()

        self.assertEqual(settings[mod.KEY_GROK], "grok-key")

    def test_legacy_plaintext_key_migrates_into_keyring(self) -> None:
        from insight_local.cvops.ui import notes_ai_keys as mod

        store: dict[tuple[str, str], str] = {}
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ai_settings.json"
            path.write_text(json.dumps({mod.KEY_ANTHROPIC: "sk-ant-legacy"}), encoding="utf-8")
            with patch.object(mod, "ai_settings_path", return_value=path), \
                 patch.object(mod, "keyring_available", return_value=True), \
                 patch.object(mod, "_keyring_get", side_effect=lambda n: store.get((mod._KEYRING_SERVICE, n), "")), \
                 patch.object(mod, "_keyring_set", side_effect=lambda n, v: (store.__setitem__((mod._KEYRING_SERVICE, n), v) or True)):
                settings = mod.load_ai_settings()
                # Migrated into the keyring and stripped from the file.
                self.assertEqual(store[(mod._KEYRING_SERVICE, mod.KEY_ANTHROPIC)], "sk-ant-legacy")
                on_disk = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(settings[mod.KEY_ANTHROPIC], "sk-ant-legacy")
        self.assertEqual(on_disk[mod.KEY_ANTHROPIC], "")

    def test_model_catalog_entries_include_local_ggufs(self) -> None:
        from insight_local.cvops.ui.notes_ai_keys import KEY_LOCAL_GGUF_MODELS, model_catalog_entries

        rows = model_catalog_entries(
            {KEY_LOCAL_GGUF_MODELS: ["/models/tacitus.gguf"]},
            ollama_installed=["gemma3:4b"],
        )

        self.assertIn(("tacitus.gguf (GGUF)", "ollama:/models/tacitus.gguf"), rows)


if __name__ == "__main__":
    unittest.main()
