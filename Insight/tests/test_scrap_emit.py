from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from mlops.scrap.emit import LabeledItem, emit_yolo_dataset


class ScrapEmitTests(unittest.TestCase):
    def test_emit_rebuilds_yolo_output_and_writes_class_map(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            staged = root / "staged"
            staged.mkdir()
            image_a = staged / "a.jpg"
            image_b = staged / "b.jpg"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")
            stale_label = root / "labels" / "train" / "stale.txt"
            stale_label.parent.mkdir(parents=True)
            stale_label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")
            (root / "raw").mkdir()
            (root / "scrap.json").write_text("{}", encoding="utf-8")

            with patch("mlops.scrap.emit.reg.resolve_library_dataset_path", return_value=root):
                emit_yolo_dataset(
                    slug="scrap_test",
                    classes=["scratch", "dent"],
                    items=[
                        LabeledItem(image_a, ((1, 0.5, 0.5, 0.2, 0.2),)),
                        LabeledItem(image_b, ((0, 0.4, 0.4, 0.1, 0.1),)),
                    ],
                    val_frac=0.5,
                    seed=0,
                )

            self.assertFalse(stale_label.exists())
            self.assertTrue((root / "raw").is_dir())
            self.assertTrue((root / "scrap.json").is_file())
            self.assertEqual((root / "classes.txt").read_text(encoding="utf-8"), "scratch\ndent\n")

            data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data["train"], "images/train")
            self.assertEqual(data["val"], "images/val")
            self.assertEqual(data["names"], {0: "scratch", 1: "dent"})

            images = sorted((root / "images").glob("*/*.jpg"))
            labels = sorted((root / "labels").glob("*/*.txt"))
            self.assertEqual(len(images), 2)
            self.assertEqual(len(labels), 2)
            self.assertEqual(
                {p.relative_to(root / "images").with_suffix(".txt") for p in images},
                {p.relative_to(root / "labels") for p in labels},
            )

    def test_emit_rejects_labels_outside_class_map(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            staged = root / "staged"
            staged.mkdir()
            image = staged / "a.jpg"
            image.write_bytes(b"a")

            with patch("mlops.scrap.emit.reg.resolve_library_dataset_path", return_value=root):
                with self.assertRaisesRegex(ValueError, "class index 1"):
                    emit_yolo_dataset(
                        slug="scrap_test",
                        classes=["scratch"],
                        items=[LabeledItem(image, ((1, 0.5, 0.5, 0.2, 0.2),))],
                    )

    def test_emit_preserves_existing_generated_images_when_rebuilding(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            existing_img = root / "images" / "train" / "old.jpg"
            existing_img.parent.mkdir(parents=True)
            existing_img.write_bytes(b"old")
            new_img = root / "staged" / "new.jpg"
            new_img.parent.mkdir()
            new_img.write_bytes(b"new")

            with patch("mlops.scrap.emit.reg.resolve_library_dataset_path", return_value=root):
                emit_yolo_dataset(
                    slug="scrap_test",
                    classes=["tiger"],
                    items=[
                        LabeledItem(existing_img, ((0, 0.5, 0.5, 1.0, 1.0),)),
                        LabeledItem(new_img, ((0, 0.5, 0.5, 1.0, 1.0),)),
                    ],
                    val_frac=0.5,
                    seed=0,
                )

            emitted = sorted(p.name for p in (root / "images").glob("*/*.jpg"))
            self.assertEqual(emitted, ["new.jpg", "old.jpg"])
            self.assertFalse((root / ".emit-tmp").exists())


if __name__ == "__main__":
    unittest.main()
