from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

try:
    from PyQt6.QtWidgets import QApplication
except Exception:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None, "PyQt6 is not available")
class CiCdLifecycleBarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def _bar(self, aliases: dict, posts: list):
        from insight_local.cvops.ui.ci_cd_lifecycle_bar import CiCdLifecycleBar

        def http_get(path: str) -> dict:
            return {"scenario": "demo", "aliases": aliases}

        def http_post(path: str, body: dict | None = None) -> dict:
            posts.append((path, body))
            return {"reverted": True, "version_id": "demo:v1"}

        return CiCdLifecycleBar(http_get=http_get, http_post=http_post)

    def test_chips_and_button_state_reflect_aliases(self) -> None:
        posts: list = []
        aliases = {
            "candidate": {"version_id": "demo:v3", "history": ["demo:v3"]},
            "staging": {"version_id": "demo:v3", "history": ["demo:v3"]},
            "prod": {"version_id": "demo:v2", "history": ["demo:v1", "demo:v2"]},
        }
        bar = self._bar(aliases, posts)
        try:
            bar.set_scenario("demo")
            self.assertIn("v3", bar._chips["staging"].text())
            self.assertIn("v2", bar._chips["prod"].text())
            # staging set -> promote enabled; prod history >=2 -> revert enabled.
            self.assertTrue(bar._promote_btn.isEnabled())
            self.assertTrue(bar._revert_btn.isEnabled())
        finally:
            bar.deleteLater()

    def test_revert_disabled_without_prior_prod(self) -> None:
        posts: list = []
        aliases = {
            "candidate": {"version_id": "demo:v1", "history": ["demo:v1"]},
            "staging": {"version_id": "", "history": []},
            "prod": {"version_id": "demo:v1", "history": ["demo:v1"]},
        }
        bar = self._bar(aliases, posts)
        try:
            bar.set_scenario("demo")
            self.assertFalse(bar._promote_btn.isEnabled())  # nothing staged
            self.assertFalse(bar._revert_btn.isEnabled())   # no prior prod
        finally:
            bar.deleteLater()

    def test_promote_posts_staging_run_to_prod(self) -> None:
        posts: list = []
        aliases = {
            "candidate": {"version_id": "demo:v3", "history": ["demo:v3"]},
            "staging": {"version_id": "demo:v3", "history": ["demo:v3"]},
            "prod": {"version_id": "demo:v2", "history": ["demo:v1", "demo:v2"]},
        }
        bar = self._bar(aliases, posts)
        try:
            bar.set_scenario("demo")
            bar._on_promote()
            self.assertEqual(posts[0][0], "/scenarios/demo/runs/v3/promote")
            self.assertEqual(posts[0][1]["target_alias"], "prod")
        finally:
            bar.deleteLater()

    def test_revert_posts_to_prod_alias_endpoint(self) -> None:
        posts: list = []
        aliases = {
            "candidate": {"version_id": "demo:v2", "history": ["demo:v1", "demo:v2"]},
            "staging": {"version_id": "", "history": []},
            "prod": {"version_id": "demo:v2", "history": ["demo:v1", "demo:v2"]},
        }
        bar = self._bar(aliases, posts)
        try:
            bar.set_scenario("demo")
            bar._on_revert()
            self.assertEqual(posts[0][0], "/scenarios/demo/aliases/prod/revert")
        finally:
            bar.deleteLater()


if __name__ == "__main__":
    unittest.main()
