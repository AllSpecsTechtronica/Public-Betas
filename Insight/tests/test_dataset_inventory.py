from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mlops.pipeline import registry as reg  # noqa: E402


class TestDatasetInventory(unittest.TestCase):
    def test_inventory_counts_and_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "nested").mkdir()
            (root / "a.jpg").write_bytes(b"x")
            (root / "b.JPG").write_bytes(b"yy")
            (root / "c.txt").write_text("hello", encoding="utf-8")
            (root / "noext").write_bytes(b"1234")
            (root / ".hidden.csv").write_text("a,b\n1,2\n", encoding="utf-8")
            (root / "nested" / "d.json").write_text("{}", encoding="utf-8")

            payload = reg.inventory_folder_types_at(root, include_hidden=False)
            self.assertEqual(int(payload["total_files"]), 5)

            counts = {t["ext"]: int(t["count"]) for t in payload["types"]}
            self.assertEqual(counts.get(".jpg"), 2)
            self.assertEqual(counts.get(".txt"), 1)
            self.assertEqual(counts.get(".json"), 1)
            self.assertEqual(counts.get(reg._INVENTORY_EXT_NONE), 1)
            self.assertNotIn(".csv", counts)

            payload2 = reg.inventory_folder_types_at(root, include_hidden=True)
            self.assertEqual(int(payload2["total_files"]), 6)
            counts2 = {t["ext"]: int(t["count"]) for t in payload2["types"]}
            self.assertEqual(counts2.get(".csv"), 1)

    def test_move_and_delete_by_extension_scoped(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = root / "data"
            data.mkdir()
            (data / "keep.jpg").write_bytes(b"x")
            (data / "move.txt").write_text("t", encoding="utf-8")

            moved = reg.move_files_by_extension_at(
                root,
                ext=".txt",
                dest_relative_dir="_q",
                relative_dir="data",
                preserve_tree=True,
            )
            self.assertEqual(int(moved["moved"]), 1)
            self.assertFalse((data / "move.txt").exists())
            self.assertTrue((root / "_q" / "data" / "move.txt").exists())

            deleted = reg.delete_files_by_extension_at(root, ext=".jpg", relative_dir="data")
            self.assertEqual(int(deleted["deleted"]), 1)
            self.assertFalse((data / "keep.jpg").exists())


if __name__ == "__main__":
    unittest.main()
