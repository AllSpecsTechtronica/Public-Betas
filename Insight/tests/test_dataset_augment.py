from __future__ import annotations

import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.service import CvOpsService
from mlops.pipeline import registry as mlops_registry


class DatasetAugmentTests(unittest.TestCase):
    def _service(self, td: str) -> CvOpsService:
        root = Path(td)
        return CvOpsService(
            db_path=root / "jobs.db",
            catalog_db_path=root / "catalog.db",
            catalog_assets_root=root / "assets",
            snapshot_db_path=root / "snapshots.db",
            snapshot_weights_root=root / "weights",
            lineage_db_path=root / "lineage.db",
            range_db_path=root / "range.db",
        )

    def test_copy_augmented_train_to_val_copies_image_and_label(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "database"
            ds = db / "toy"
            (ds / "images" / "train").mkdir(parents=True)
            (ds / "labels" / "train").mkdir(parents=True)
            (ds / "labels" / "val").mkdir(parents=True)
            image = np.full((40, 60, 3), 180, dtype=np.uint8)
            image_path = ds / "images" / "train" / "a.jpg"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            (ds / "labels" / "train" / "a.txt").write_text("0 0.500000 0.500000 0.500000 0.500000\n")
            (ds / "classes.txt").write_text("thing\n", encoding="utf-8")

            with patch.object(mlops_registry, "DATABASE_ROOT", db):
                svc = self._service(td)
                with TestClient(svc.app) as client:
                    resp = client.post(
                        "/database/toy/copy_augmented_to_split",
                        json={
                            "relative_paths": ["images/train/a.jpg"],
                            "target_split": "val",
                            "copies_per_image": 1,
                            "scale_pct": 50,
                            "angle_deg": 15,
                            "jpeg_quality": 80,
                            "grayscale": True,
                        },
                    )
                    resp2 = client.post(
                        "/database/toy/copy_augmented_to_split",
                        json={
                            "relative_paths": ["images/train/a.jpg"],
                            "target_split": "val",
                            "copies_per_image": 1,
                            "scale_pct": 50,
                            "angle_deg": 15,
                            "jpeg_quality": 80,
                            "grayscale": True,
                        },
                    )

            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertEqual(resp.json()["copied"], 1)
            self.assertEqual(resp2.status_code, 200, resp2.text)
            self.assertEqual(resp2.json()["copied"], 1)
            val_images = list((ds / "images" / "val").glob("*.jpg"))
            val_labels = list((ds / "labels" / "val").glob("*.txt"))
            self.assertEqual(len(val_images), 2)
            self.assertEqual(len(val_labels), 2)
            augmented = cv2.imread(str(val_images[0]))
            self.assertIsNotNone(augmented)
            assert augmented is not None
            self.assertEqual(augmented.shape[:2], (20, 30))
            label_parts = val_labels[0].read_text(encoding="utf-8").split()
            self.assertEqual(label_parts[0], "0")
            self.assertEqual(len(label_parts), 5)

    def test_auto_augment_to_target_total_preserves_train_val_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "database"
            ds = db / "toy"
            for split in ("train", "val"):
                (ds / "images" / split).mkdir(parents=True)
                (ds / "labels" / split).mkdir(parents=True)
            for idx, split in enumerate(("train", "train", "val"), start=1):
                image = np.full((32, 48, 3), 80 + idx, dtype=np.uint8)
                image_path = ds / "images" / split / f"{idx}.jpg"
                self.assertTrue(cv2.imwrite(str(image_path), image))
                (ds / "labels" / split / f"{idx}.txt").write_text(
                    "0 0.500000 0.500000 0.500000 0.500000\n",
                    encoding="utf-8",
                )
            (ds / "classes.txt").write_text("thing\n", encoding="utf-8")

            with patch.object(mlops_registry, "DATABASE_ROOT", db):
                svc = self._service(td)
                with TestClient(svc.app) as client:
                    resp = client.post(
                        "/database/toy/auto_augment",
                        json={
                            "target_total": 6,
                            "min_scale_pct": 90,
                            "max_scale_pct": 110,
                            "max_angle_deg": 10,
                            "min_jpeg_quality": 75,
                            "max_jpeg_quality": 95,
                            "grayscale_probability": 0.5,
                            "bgr_shuffle_probability": 0.5,
                            "seed": 7,
                        },
                    )

            self.assertEqual(resp.status_code, 200, resp.text)
            self.assertEqual(resp.json()["copied"], 3)
            self.assertEqual(resp.json()["additions_by_split"], {"train": 2, "val": 1})
            train_images = list((ds / "images" / "train").glob("*.jpg"))
            val_images = list((ds / "images" / "val").glob("*.jpg"))
            train_labels = list((ds / "labels" / "train").glob("*.txt"))
            val_labels = list((ds / "labels" / "val").glob("*.txt"))
            self.assertEqual(len(train_images), 4)
            self.assertEqual(len(val_images), 2)
            self.assertEqual(len(train_labels), 4)
            self.assertEqual(len(val_labels), 2)

    def test_auto_augment_train_only_creates_val_split(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "database"
            ds = db / "toy"
            (ds / "images" / "train").mkdir(parents=True)
            (ds / "labels" / "train").mkdir(parents=True)
            for idx in range(1, 6):
                image = np.full((32, 48, 3), 40 + idx, dtype=np.uint8)
                image_path = ds / "images" / "train" / f"{idx}.jpg"
                self.assertTrue(cv2.imwrite(str(image_path), image))
                (ds / "labels" / "train" / f"{idx}.txt").write_text(
                    "0 0.500000 0.500000 0.500000 0.500000\n",
                    encoding="utf-8",
                )
            (ds / "classes.txt").write_text("thing\n", encoding="utf-8")
            (ds / "data.yaml").write_text(
                f"path: {ds.resolve().as_posix()}\ntrain: images/train\nnc: 1\nnames:\n  0: thing\n",
                encoding="utf-8",
            )

            with patch.object(mlops_registry, "DATABASE_ROOT", db):
                svc = self._service(td)
                with TestClient(svc.app) as client:
                    resp = client.post(
                        "/database/toy/auto_augment",
                        json={
                            "target_total": 10,
                            "ensure_val": True,
                            "val_frac": 0.2,
                            "seed": 3,
                        },
                    )

            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertGreater(int(body["additions_by_split"]["val"]), 0)
            self.assertTrue(body["val_layout_updated"])
            self.assertTrue((ds / "images" / "val").is_dir())
            self.assertTrue((ds / "labels" / "val").is_dir())
            val_images = list((ds / "images" / "val").glob("*.jpg"))
            self.assertGreaterEqual(len(val_images), 1)
            yaml_text = (ds / "data.yaml").read_text(encoding="utf-8")
            self.assertIn("val: images/val", yaml_text)

    def test_even_dataset_balances_train_and_val(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "database"
            ds = db / "toy"
            for split in ("train", "val"):
                (ds / "images" / split).mkdir(parents=True)
                (ds / "labels" / split).mkdir(parents=True)
            for idx in range(1, 4):
                image = np.full((24, 36, 3), 50 + idx, dtype=np.uint8)
                image_path = ds / "images" / "train" / f"{idx}.jpg"
                self.assertTrue(cv2.imwrite(str(image_path), image))
                (ds / "labels" / "train" / f"{idx}.txt").write_text(
                    "0 0.500000 0.500000 0.500000 0.500000\n",
                    encoding="utf-8",
                )
            image = np.full((24, 36, 3), 90, dtype=np.uint8)
            val_path = ds / "images" / "val" / "1.jpg"
            self.assertTrue(cv2.imwrite(str(val_path), image))
            (ds / "labels" / "val" / "1.txt").write_text(
                "0 0.500000 0.500000 0.500000 0.500000\n",
                encoding="utf-8",
            )
            (ds / "classes.txt").write_text("thing\n", encoding="utf-8")

            with patch.object(mlops_registry, "DATABASE_ROOT", db):
                svc = self._service(td)
                with TestClient(svc.app) as client:
                    resp = client.post(
                        "/database/toy/even_dataset",
                        json={"seed": 11, "max_angle_deg": 0.0, "grayscale_probability": 0.0, "bgr_shuffle_probability": 0.0},
                    )

            self.assertEqual(resp.status_code, 200, resp.text)
            body = resp.json()
            self.assertEqual(body["target_per_bucket"], 3)
            self.assertEqual(body["copied"], 2)
            self.assertEqual(len(list((ds / "images" / "train").glob("*.jpg"))), 3)
            self.assertEqual(len(list((ds / "images" / "val").glob("*.jpg"))), 3)


if __name__ == "__main__":
    unittest.main()
