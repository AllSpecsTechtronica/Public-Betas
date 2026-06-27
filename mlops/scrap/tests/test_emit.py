from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mlops.pipeline import registry as reg  # noqa: E402
from mlops.scrap.emit import LabeledItem, emit_yolo_dataset  # noqa: E402


def _make_image(path: Path) -> None:
    from PIL import Image
    Image.new("RGB", (640, 480), (123, 45, 67)).save(path, format="JPEG", quality=85)


class TestEmitYoloDataset(unittest.TestCase):
    def test_writes_split_labels_and_classes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_db = Path(td) / "database"
            tmp_db.mkdir()
            staged = Path(td) / "staged"
            staged.mkdir()

            slug = "scrap_test_topic"
            with patch.object(reg, "DATABASE_ROOT", tmp_db):
                reg.create_library_dataset_root(slug)

                imgs = []
                for i in range(4):
                    p = staged / f"img{i}.jpg"
                    _make_image(p)
                    imgs.append(p)

                items = [
                    LabeledItem(image_path=imgs[0], boxes=((0, 0.5, 0.5, 0.4, 0.4),)),
                    LabeledItem(image_path=imgs[1], boxes=((1, 0.25, 0.25, 0.2, 0.2), (0, 0.7, 0.7, 0.1, 0.1))),
                    LabeledItem(image_path=imgs[2], boxes=((0, 0.5, 0.5, 0.6, 0.6),)),
                    LabeledItem(image_path=imgs[3], boxes=((1, 0.5, 0.5, 0.3, 0.3),)),
                ]

                base = emit_yolo_dataset(
                    slug=slug,
                    classes=["red", "blue"],
                    items=items,
                    val_frac=0.25,
                    seed=0,
                )

                self.assertEqual(base.resolve(), (tmp_db / slug).resolve())
                self.assertTrue((base / "images" / "train").is_dir())
                self.assertTrue((base / "images" / "val").is_dir())
                self.assertTrue((base / "labels" / "train").is_dir())
                self.assertTrue((base / "labels" / "val").is_dir())

                classes_txt = (base / "classes.txt").read_text(encoding="utf-8").splitlines()
                self.assertEqual(classes_txt, ["red", "blue"])

                train_imgs = list((base / "images" / "train").iterdir())
                val_imgs = list((base / "images" / "val").iterdir())
                self.assertEqual(len(train_imgs) + len(val_imgs), 4)
                self.assertGreaterEqual(len(val_imgs), 1)
                self.assertGreaterEqual(len(train_imgs), 1)

                for split in ("train", "val"):
                    for img in (base / "images" / split).iterdir():
                        lbl = base / "labels" / split / (img.stem + ".txt")
                        self.assertTrue(lbl.exists(), msg=f"missing label for {img.name}")
                        for line in lbl.read_text(encoding="utf-8").splitlines():
                            parts = line.split()
                            self.assertEqual(len(parts), 5)
                            cls = int(parts[0])
                            self.assertIn(cls, (0, 1))
                            for v in parts[1:]:
                                f = float(v)
                                self.assertGreaterEqual(f, 0.0)
                                self.assertLessEqual(f, 1.0)

                fmt = reg.detect_library_dataset_format(base)
                self.assertEqual(fmt, reg.LIBRARY_DATASET_FORMAT_YOLO)

    def test_rejects_empty_input(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp_db = Path(td) / "database"
            tmp_db.mkdir()
            slug = "scrap_test_empty"
            with patch.object(reg, "DATABASE_ROOT", tmp_db):
                reg.create_library_dataset_root(slug)
                with self.assertRaises(ValueError):
                    emit_yolo_dataset(slug=slug, classes=["a"], items=[])
                with self.assertRaises(ValueError):
                    emit_yolo_dataset(slug=slug, classes=[], items=[
                        LabeledItem(image_path=Path("x.jpg"), boxes=((0, 0.5, 0.5, 0.1, 0.1),))
                    ])


if __name__ == "__main__":
    unittest.main()
