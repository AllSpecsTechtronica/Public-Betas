from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.ui import theme as theme_mod
from insight_local.ui import timeline_card as timeline_card_mod


class FireThemePolishTests(unittest.TestCase):
    def test_fire_black_text_uses_ivory_and_sans_stack(self) -> None:
        theme_mod.configure_color_scheme("fire")
        theme_mod.configure_text_mode("black")
        self.assertEqual(theme_mod.text_hex(), "#f2e7cf")
        css = theme_mod.get_global_stylesheet()
        self.assertIn('font-family: "IBM Plex Sans"', css)
        self.assertIn("background: #002b36;", css)
        self.assertIn("#f8a43f", css)
        self.assertIn("rgba(248, 164, 64, 0.20)", css)
        self.assertIn("qlineargradient(", css)

    def test_fire_overlay_sources_do_not_embed_legacy_red_literals(self) -> None:
        loading_gate_src = (ROOT / "Insight" / "insight_local" / "ui" / "loading_gate.py").read_text(encoding="utf-8")
        timeline_src = (ROOT / "Insight" / "insight_local" / "ui" / "timeline_card.py").read_text(encoding="utf-8")

        self.assertIsNone(re.search(r"rgba\(\s*20\s*,\s*8\s*,\s*8", loading_gate_src))
        self.assertNotIn("#ffd0d0", loading_gate_src.lower())
        self.assertIsNone(re.search(r"QColor\(\s*198\s*,\s*40\s*,\s*40", timeline_src))
        self.assertIsNone(re.search(r"QColor\(\s*138\s*,\s*20\s*,\s*20", timeline_src))

    def test_timeline_card_exposes_theme_refresh_hook(self) -> None:
        self.assertTrue(callable(getattr(timeline_card_mod.TimelineCardWidget, "refresh_theme", None)))


if __name__ == "__main__":
    unittest.main()
