"""Tests for replication input staging (images / COLMAP paths)."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Insight"))


class GaussianReplicationTests(unittest.TestCase):
    def test_list_images_and_validate_folder(self) -> None:
        from mlops.gaussian_splat.replication import (
            list_images_in_folder,
            validate_image_folder,
        )

        base = Path(tempfile.mkdtemp(prefix="gsrepl_"))
        try:
            (base / "a.jpg").write_bytes(b"")
            (base / "skip.txt").write_text("no")
            imgs = list_images_in_folder(base)
            self.assertEqual(len(imgs), 1)
            ok, err = validate_image_folder(base)
            self.assertTrue(ok)
            self.assertEqual(err, "")
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_resolve_colmap_sparse_dir(self) -> None:
        from mlops.gaussian_splat.replication import resolve_colmap_sparse_dir

        base = Path(tempfile.mkdtemp(prefix="gsrepl_colmap_"))
        try:
            sparse0 = base / "sparse" / "0"
            sparse0.mkdir(parents=True, exist_ok=True)
            (sparse0 / "cameras.txt").write_text("# c")
            (sparse0 / "images.txt").write_text("# i")
            resolved = resolve_colmap_sparse_dir(base / "sparse")
            self.assertEqual(resolved, sparse0.resolve())
            resolved2 = resolve_colmap_sparse_dir(sparse0)
            self.assertEqual(resolved2, sparse0.resolve())
        finally:
            shutil.rmtree(base, ignore_errors=True)

    def test_prepare_workspace_images_only(self) -> None:
        from mlops.gaussian_splat.replication import prepare_replication_workspace

        src = Path(tempfile.mkdtemp(prefix="gsrepl_src_"))
        ws = Path(tempfile.mkdtemp(prefix="gsrepl_ws_"))
        try:
            (src / "one.jpg").write_bytes(b"")
            man = prepare_replication_workspace(
                ws,
                source_kind="image_folder",
                source_path=src,
                calibration="none",
                colmap_sparse_user=None,
                video_fps=1.0,
                video_max_frames=100,
                prefer_symlink=False,
                on_status=None,
            )
            self.assertTrue((ws / "images").is_dir())
            self.assertTrue((ws / "replication_manifest.json").is_file())
            self.assertEqual(len(man.image_paths), 1)
        finally:
            shutil.rmtree(src, ignore_errors=True)
            shutil.rmtree(ws, ignore_errors=True)

    def test_true_gaussian_rejects_single_image_depth_fallback(self) -> None:
        from mlops.gaussian_splat.replication import prepare_replication_workspace
        from mlops.gaussian_splat.true_gaussian import run_true_gaussian_pipeline

        src = Path(tempfile.mkdtemp(prefix="gsrepl_single_"))
        ws = Path(tempfile.mkdtemp(prefix="gsrepl_ws_single_"))
        try:
            (src / "one.jpg").write_bytes(b"")
            prepare_replication_workspace(
                ws,
                source_kind="single_image",
                source_path=src / "one.jpg",
                calibration="none",
                colmap_sparse_user=None,
                video_fps=1.0,
                video_max_frames=1,
                prefer_symlink=False,
                on_status=None,
            )
            with self.assertRaisesRegex(RuntimeError, "overlapping multi-view"):
                run_true_gaussian_pipeline(ws, max_num_iterations=100)
        finally:
            shutil.rmtree(src, ignore_errors=True)
            shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
