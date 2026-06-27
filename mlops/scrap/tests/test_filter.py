from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mlops.scrap.filter import dedupe_and_stage  # noqa: E402


def _make_image(path: Path, size: tuple[int, int], color: tuple[int, int, int]) -> None:
    from PIL import Image
    Image.new("RGB", size, color).save(path, format="JPEG", quality=92)


class TestDedupeAndStage(unittest.TestCase):
    def test_default_keeps_small_images(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            staged = Path(td) / "staged"
            raw.mkdir()

            _make_image(raw / "big_red.jpg", (640, 640), (200, 0, 0))
            _make_image(raw / "tiny_green.jpg", (64, 64), (0, 200, 0))

            result = dedupe_and_stage(raw, staged)

            self.assertEqual(result.skipped_small, 0)
            self.assertEqual(len(result.staged), 2, msg=f"staged={[p.name for p in result.staged]}")

    def test_drops_small_dup_and_unreadable(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            staged = Path(td) / "staged"
            raw.mkdir()

            _make_image(raw / "big_red_a.jpg", (640, 640), (200, 0, 0))
            _make_image(raw / "big_red_b.jpg", (640, 640), (200, 0, 0))
            _make_image(raw / "big_blue.jpg", (640, 640), (0, 0, 200))
            _make_image(raw / "tiny.jpg", (64, 64), (0, 200, 0))
            (raw / "garbage.jpg").write_bytes(b"\x00\x01not-an-image\x02")

            result = dedupe_and_stage(raw, staged, min_size=320)

            self.assertEqual(len(result.staged), 2, msg=f"staged={[p.name for p in result.staged]}")
            self.assertEqual(result.skipped_small, 1)
            self.assertGreaterEqual(result.skipped_dup, 1)
            self.assertGreaterEqual(result.skipped_unreadable, 1)

    def test_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw"
            staged = Path(td) / "staged"
            raw.mkdir()
            _make_image(raw / "a.jpg", (640, 640), (200, 0, 0))
            _make_image(raw / "b.jpg", (640, 640), (0, 200, 0))

            r1 = dedupe_and_stage(raw, staged, min_size=320)
            r2 = dedupe_and_stage(raw, staged, min_size=320)

            self.assertEqual(len(r1.staged), 2)
            self.assertEqual(len(r2.staged), 2)
            self.assertEqual({p.name for p in r1.staged}, {p.name for p in r2.staged})


if __name__ == "__main__":
    unittest.main()
