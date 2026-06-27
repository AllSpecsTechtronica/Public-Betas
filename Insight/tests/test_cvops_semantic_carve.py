from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))


def _make_images(folder: Path, n: int) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        p = folder / f"img{i:03d}.jpg"
        p.write_bytes(b"fake-jpeg-bytes")
        paths.append(p)
    return paths


class _FakeEmbedder:
    """Deterministic embeddings: image i gets a 1-hot-ish vector; the query
    aligns with the first half of images so scores are separable."""

    def __init__(self, paths: list[Path]) -> None:
        self._paths = paths

    def embed_images(self, paths, *, batch_size=32, progress_cb=None):
        paths = list(paths)
        feats = []
        for p in paths:
            idx = int(p.stem.replace("img", ""))
            v = np.array([1.0, 0.0] if idx < len(paths) // 2 else [0.0, 1.0], dtype=np.float32)
            feats.append(v / np.linalg.norm(v))
        return np.stack(feats), paths

    def embed_texts(self, prompts):
        # Query aligns with the [1,0] cluster (the "first half").
        v = np.array([1.0, 0.0], dtype=np.float32)
        return (v / np.linalg.norm(v))[None, :]


class SemanticCarveTests(unittest.TestCase):
    def test_select_splits_positives_and_negatives(self) -> None:
        from insight_local.cvops import semantic_carve as sc

        scores = np.array([0.95, 0.92, 0.10, 0.05, 0.50], dtype=np.float32)
        pos, neg = sc.select(scores, threshold=0.6)
        self.assertEqual(sorted(pos), [0, 1])
        self.assertTrue(set(neg).issubset({2, 3}))  # clearly-below items only

    def test_build_index_and_query_scores(self) -> None:
        from insight_local.cvops import semantic_carve as sc

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            paths = _make_images(src, 10)
            emb = _FakeEmbedder(paths)
            index = sc.build_index(src, emb)
            self.assertEqual(len(index), 10)
            scores = sc.query_scores(index, "thing", emb)
            # First half (1-hot [1,0]) should score ~1, second half ~0.
            self.assertGreater(scores[:5].mean(), 0.9)
            self.assertLess(scores[5:].mean(), 0.1)

    def test_materialize_creates_imagefolder(self) -> None:
        from insight_local.cvops import semantic_carve as sc
        from mlops.pipeline.registry import detect_library_dataset_format, LIBRARY_DATASET_FORMAT_IMAGEFOLDER

        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "src"
            paths = _make_images(src, 10)
            emb = _FakeEmbedder(paths)
            index = sc.build_index(src, emb)
            scores = sc.query_scores(index, "thing", emb)
            pos, neg = sc.select(scores, threshold=0.6)

            registry = Path(td) / "dataset_registry"
            result = sc.materialize_imagefolder(
                registry_dir=registry,
                slug="alhambra_thing",
                class_name="thing",
                positive_paths=[index.paths[i] for i in pos],
                negative_paths=[index.paths[i] for i in neg],
                query="thing",
                threshold=0.6,
            )
            ds_root = Path(result["dataset_path"])
            self.assertTrue((ds_root / "thing").is_dir())
            self.assertTrue((ds_root / "not_thing").is_dir())
            self.assertTrue((ds_root / "classes.txt").is_file())
            self.assertEqual(result["counts"]["thing"], len(pos))
            # The emitted layout is recognized by the training pipeline.
            self.assertEqual(
                detect_library_dataset_format(ds_root),
                LIBRARY_DATASET_FORMAT_IMAGEFOLDER,
            )

    def test_materialize_refuses_existing(self) -> None:
        from insight_local.cvops import semantic_carve as sc

        with tempfile.TemporaryDirectory() as td:
            registry = Path(td) / "reg"
            (registry / "dup").mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                sc.materialize_imagefolder(
                    registry_dir=registry, slug="dup", class_name="x",
                    positive_paths=[], negative_paths=[],
                )


class SemanticCarveServiceTests(unittest.TestCase):
    def test_carve_preview_and_create_endpoints(self) -> None:
        from fastapi.testclient import TestClient
        from insight_local.cvops import service as service_mod
        from insight_local.cvops.service import CvOpsService
        from insight_local.cvops import semantic_carve as sc
        from mlops.pipeline.registry import (
            detect_library_dataset_format,
            LIBRARY_DATASET_FORMAT_IMAGEFOLDER,
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            src = root / "src"
            paths = _make_images(src, 10)
            emb = _FakeEmbedder(paths)

            svc = CvOpsService(
                db_path=root / "jobs.db",
                catalog_db_path=root / "catalog.db",
                catalog_assets_root=root / "assets",
            )
            # Inject a prebuilt index + embedder (skip the CLIP background job).
            svc._carve_index = sc.build_index(src, emb)
            svc._carve_embedder = emb

            registry = root / "dataset_registry"
            registry.mkdir()
            orig = service_mod.DATASET_REGISTRY_DIR
            service_mod.DATASET_REGISTRY_DIR = registry
            try:
                with TestClient(svc.app) as client:
                    pv = client.post("/carve/preview", json={"query": "thing", "threshold": 0.6})
                    self.assertEqual(pv.status_code, 200, pv.text)
                    self.assertEqual(pv.json()["positive_count"], 5)
                    self.assertEqual(len(pv.json()["sample"]) > 0, True)

                    cr = client.post("/carve/create", json={
                        "slug": "carved_thing", "class_name": "thing",
                        "query": "thing", "threshold": 0.6,
                    })
                    self.assertEqual(cr.status_code, 200, cr.text)
                    ds_root = Path(cr.json()["dataset_path"])
                    self.assertEqual(cr.json()["counts"]["thing"], 5)
                    self.assertEqual(
                        detect_library_dataset_format(ds_root),
                        LIBRARY_DATASET_FORMAT_IMAGEFOLDER,
                    )

                    # Re-create with same slug -> 409.
                    dup = client.post("/carve/create", json={
                        "slug": "carved_thing", "class_name": "thing",
                        "query": "thing", "threshold": 0.6,
                    })
                    self.assertEqual(dup.status_code, 409, dup.text)
            finally:
                service_mod.DATASET_REGISTRY_DIR = orig

    def test_carve_preview_without_index_returns_409(self) -> None:
        from fastapi.testclient import TestClient
        from insight_local.cvops.service import CvOpsService

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            svc = CvOpsService(
                db_path=root / "jobs.db",
                catalog_db_path=root / "catalog.db",
                catalog_assets_root=root / "assets",
            )
            with TestClient(svc.app) as client:
                r = client.post("/carve/preview", json={"query": "x"})
                self.assertEqual(r.status_code, 409, r.text)


if __name__ == "__main__":
    unittest.main()
