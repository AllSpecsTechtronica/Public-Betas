from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))


class OllamaModelDiscoveryTests(unittest.TestCase):
    def test_fetch_ollama_model_names_parses_tags(self) -> None:
        from insight_local.cvops.ollama_model_discovery import fetch_ollama_model_names

        payload = {"models": [{"name": "m:latest", "size": 1}, {"name": "a:7b", "size": 2}]}
        raw = json.dumps(payload).encode("utf-8")

        class _Resp:
            def read(self) -> bytes:
                return raw

            def __enter__(self) -> _Resp:
                return self

            def __exit__(self, *args: object) -> None:
                return None

        with patch("insight_local.cvops.ollama_model_discovery.urllib.request.urlopen", return_value=_Resp()):
            names = fetch_ollama_model_names(base_url="http://127.0.0.1:11434", timeout=1.0)
        self.assertEqual(names, ["a:7b", "m:latest"])

    def test_choose_ollama_embedding_model_keeps_installed_current(self) -> None:
        from insight_local.cvops.ollama_model_discovery import choose_ollama_embedding_model

        chosen = choose_ollama_embedding_model(
            ["gemma3:4b", "mxbai-embed-large:latest"],
            current="mxbai-embed-large",
        )

        self.assertEqual(chosen, "mxbai-embed-large:latest")

    def test_choose_ollama_embedding_model_replaces_missing_default(self) -> None:
        from insight_local.cvops.ollama_model_discovery import choose_ollama_embedding_model

        chosen = choose_ollama_embedding_model(
            ["llama3.2:latest", "bge-m3:latest", "gemma3:4b"],
            current="nomic-embed-text",
        )

        self.assertEqual(chosen, "bge-m3:latest")

    def test_choose_ollama_embedding_model_falls_back_to_local_tag(self) -> None:
        from insight_local.cvops.ollama_model_discovery import choose_ollama_embedding_model

        chosen = choose_ollama_embedding_model(["gemma3:4b"], current="nomic-embed-text")

        self.assertEqual(chosen, "gemma3:4b")

    def test_discover_local_gguf_respects_repo_root(self) -> None:
        from insight_local.cvops.ollama_model_discovery import discover_local_gguf_files

        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            nested = repo / "weights"
            nested.mkdir(parents=True)
            gguf = nested / "tiny.gguf"
            gguf.write_bytes(b"x")
            found = discover_local_gguf_files(repo_root=repo, max_files=50, per_root_depth=6)
        self.assertIn(str(gguf.resolve()), found)

    def test_discover_honors_insight_gguf_scan_roots(self) -> None:
        from insight_local.cvops import ollama_model_discovery as mod

        with tempfile.TemporaryDirectory() as td:
            extra = Path(td) / "extra"
            extra.mkdir()
            (extra / "x.gguf").write_bytes(b"y")
            old = os.environ.get("INSIGHT_GGUF_SCAN_ROOTS")
            os.environ["INSIGHT_GGUF_SCAN_ROOTS"] = str(extra)
            try:
                found = mod.discover_local_gguf_files(repo_root=None, max_files=50, per_root_depth=2)
            finally:
                if old is None:
                    os.environ.pop("INSIGHT_GGUF_SCAN_ROOTS", None)
                else:
                    os.environ["INSIGHT_GGUF_SCAN_ROOTS"] = old
        self.assertIn(str((extra / "x.gguf").resolve()), found)


if __name__ == "__main__":
    unittest.main()
