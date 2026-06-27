from __future__ import annotations

import base64
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.service import CvOpsService
from insight_local.cvops import archive_engine
from mlops.pipeline import registry as mlops_registry


class ArchivalRegistryTests(unittest.TestCase):
    def test_archival_scenario_profile_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp
            mlops_root = repo_root / "mlops"
            (mlops_root / "scenarios").mkdir(parents=True)
            (mlops_root / "datasets").mkdir(parents=True)
            (mlops_root / "registry.json").write_text('{"version": 1, "scenarios": []}', encoding="utf-8")

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_audio_root = mlops_registry.ML_AUDIO_ROOT
            old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = repo_root / "database"
                mlops_registry.ML_AUDIO_ROOT = repo_root / "assets" / "ml_audio"
                mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                status = mlops_registry.create_scenario_profile(
                    name="alhambra_archive",
                    display_name="Alhambra Archive",
                    description="Archive ingestion scenario",
                    backbone_type="archival_ingestion",
                    backbone_config={"domain_profile": {"geography": "SGV"}},
                )
                self.assertEqual(status["backbone_type"], "archival_ingestion")
                cfg = mlops_registry.get_scenario_config("alhambra_archive")
                self.assertEqual(cfg.backbone_type, "archival_ingestion")
                self.assertEqual(cfg.backbone_config.get("domain_profile"), {"geography": "SGV"})
                self.assertIn("archive_storage_root", cfg.backbone_config)
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.ML_AUDIO_ROOT = old_audio_root
                mlops_registry.TABULAR_DATASETS_ROOT = old_tabular
                mlops_registry.REGISTRY_PATH = old_registry


class ArchivalServiceTests(unittest.TestCase):
    def test_archive_phase3_review_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "src"
            src.mkdir()
            (src / "batz_note_1923.txt").write_text(
                "Marguerita Batz at age 26 in Spring 1923 near Alhambra",
                encoding="utf-8",
            )

            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
                snapshot_db_path=tmp / "snapshots.db",
                snapshot_weights_root=tmp / "snapshot_weights",
                lineage_db_path=tmp / "lineages.db",
                provenance_db_path=tmp / "provenance.db",
                range_db_path=tmp / "ranges.db",
                archive_db_path=tmp / "archives.db",
                archive_storage_root=tmp / "archive_corpora",
            )
            with TestClient(svc.app) as client:
                imported = client.post(
                    "/archives/import",
                    json={"source_paths": [str(src)], "name": "Phase3 Smoke"},
                )
                self.assertEqual(imported.status_code, 200, imported.text)
                imp = imported.json()
                corpus_id = imp["corpus"]["corpus_id"]
                dataset_version_id = imp["dataset_version_id"]

                parent_snapshot_id = ""
                for phase in ("archive_phase0", "archive_phase1", "archive_phase2", "archive_phase3"):
                    kicked = client.post(
                        f"/archives/{corpus_id}/jobs",
                        json={
                            "dataset_version_id": dataset_version_id,
                            "phase": phase,
                            "parent_snapshot_id": parent_snapshot_id,
                            "write_run_artifacts": False,
                        },
                    )
                    self.assertEqual(kicked.status_code, 200, kicked.text)
                    job_id = kicked.json()["job_id"]
                    final_job = {}
                    for _ in range(80):
                        final_job = client.get(f"/archives/jobs/{job_id}").json()
                        if final_job.get("state") in {"done", "error"}:
                            break
                        time.sleep(0.1)
                    self.assertEqual(final_job.get("state"), "done", final_job)
                    parent_snapshot_id = str((final_job.get("result") or {}).get("snapshot_id") or "")
                    self.assertTrue(parent_snapshot_id.startswith("snapshot-"))

                phase3_review = client.get(f"/archives/snapshots/{parent_snapshot_id}/phase3_review").json()
                self.assertEqual(phase3_review.get("phase"), "archive_phase3")
                self.assertIn("objects", phase3_review)
                self.assertIn("anchors", phase3_review)
                self.assertIn("mentions", phase3_review)
                self.assertIn("relationships", phase3_review)
                summary = phase3_review.get("summary") or {}
                self.assertGreaterEqual(int(summary.get("anchor_count") or 0), 1)
                self.assertGreaterEqual(int(summary.get("mention_count") or 0), 1)
                self.assertGreaterEqual(int(summary.get("structured_object_count") or 0), 1)
                self.assertTrue(bool(phase3_review.get("anchor_groups")))
                self.assertTrue(bool(phase3_review.get("entity_groups")))

                row = list(phase3_review.get("objects") or [])[0]
                detail = client.get(
                    f"/archives/snapshots/{parent_snapshot_id}/objects/{row['object_id']}"
                ).json()
                self.assertTrue(detail.get("anchors"))
                self.assertTrue(detail.get("mentions"))
                self.assertTrue(
                    any(str(assertion.get("field") or "").startswith("temporal_") for assertion in detail.get("assertions") or [])
                )

    def test_archive_phase2_review_and_detail_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "src"
            src.mkdir()
            (src / "typed_note_1934.txt").write_text(
                "Typed meeting notes for April 12, 1934 in Alhambra",
                encoding="utf-8",
            )
            png_bytes = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnU2r8AAAAASUVORK5CYII="
            )
            (src / "photo_back.png").write_bytes(png_bytes)

            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
                snapshot_db_path=tmp / "snapshots.db",
                snapshot_weights_root=tmp / "snapshot_weights",
                lineage_db_path=tmp / "lineages.db",
                provenance_db_path=tmp / "provenance.db",
                range_db_path=tmp / "ranges.db",
                archive_db_path=tmp / "archives.db",
                archive_storage_root=tmp / "archive_corpora",
            )
            with TestClient(svc.app) as client:
                imported = client.post(
                    "/archives/import",
                    json={"source_paths": [str(src)], "name": "Phase2 Smoke"},
                )
                self.assertEqual(imported.status_code, 200, imported.text)
                imp = imported.json()
                corpus_id = imp["corpus"]["corpus_id"]
                dataset_version_id = imp["dataset_version_id"]

                parent_snapshot_id = ""
                for phase in ("archive_phase0", "archive_phase1", "archive_phase2"):
                    kicked = client.post(
                        f"/archives/{corpus_id}/jobs",
                        json={
                            "dataset_version_id": dataset_version_id,
                            "phase": phase,
                            "parent_snapshot_id": parent_snapshot_id,
                            "write_run_artifacts": False,
                        },
                    )
                    self.assertEqual(kicked.status_code, 200, kicked.text)
                    job_id = kicked.json()["job_id"]
                    final_job = {}
                    for _ in range(80):
                        final_job = client.get(f"/archives/jobs/{job_id}").json()
                        if final_job.get("state") in {"done", "error"}:
                            break
                        time.sleep(0.1)
                    self.assertEqual(final_job.get("state"), "done", final_job)
                    parent_snapshot_id = str((final_job.get("result") or {}).get("snapshot_id") or "")
                    self.assertTrue(parent_snapshot_id.startswith("snapshot-"))

                phase2_review = client.get(f"/archives/snapshots/{parent_snapshot_id}/phase2_review").json()
                self.assertEqual(phase2_review.get("phase"), "archive_phase2")
                self.assertIn("groups", phase2_review)
                self.assertIn("summary", phase2_review)
                self.assertGreaterEqual(int((phase2_review.get("summary") or {}).get("object_count") or 0), 1)
                rows = list(phase2_review.get("objects") or [])
                self.assertTrue(rows)
                self.assertTrue(any(isinstance(row.get("extraction_summary"), dict) for row in rows))

                image_row = next((row for row in rows if str(row.get("media_family") or "") == "image"), rows[0])
                detail = client.get(
                    f"/archives/snapshots/{parent_snapshot_id}/objects/{image_row['object_id']}"
                ).json()
                self.assertIn("preview", detail)
                self.assertIn("extraction_summary", detail)
                self.assertIn("text_blocks", detail)
                self.assertIn("segmentation", detail)
                self.assertIn("summary", detail.get("segmentation") or {})
                self.assertTrue(isinstance(detail.get("text_blocks"), list))
                self.assertIn(str((detail.get("preview") or {}).get("kind") or ""), {"image", "none", "pdf_page", "audio"})

    def test_archive_phase1_classification_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "src"
            src.mkdir()
            (src / "photo_front_1923.txt").write_text(
                "Portrait of Marguerita Batz, Spring 1923",
                encoding="utf-8",
            )
            (src / "city_map_1910.txt").write_text(
                "San Gabriel map sheet 1910",
                encoding="utf-8",
            )

            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
                snapshot_db_path=tmp / "snapshots.db",
                snapshot_weights_root=tmp / "snapshot_weights",
                lineage_db_path=tmp / "lineages.db",
                provenance_db_path=tmp / "provenance.db",
                range_db_path=tmp / "ranges.db",
                archive_db_path=tmp / "archives.db",
                archive_storage_root=tmp / "archive_corpora",
            )
            with TestClient(svc.app) as client:
                imported = client.post(
                    "/archives/import",
                    json={"source_paths": [str(src)], "name": "Phase1 Smoke"},
                )
                self.assertEqual(imported.status_code, 200, imported.text)
                imp = imported.json()
                corpus_id = imp["corpus"]["corpus_id"]
                dataset_version_id = imp["dataset_version_id"]

                phase0 = client.post(
                    f"/archives/{corpus_id}/jobs",
                    json={
                        "dataset_version_id": dataset_version_id,
                        "phase": "archive_phase0",
                        "write_run_artifacts": False,
                    },
                )
                self.assertEqual(phase0.status_code, 200, phase0.text)
                phase0_job_id = phase0.json()["job_id"]

                final_phase0 = {}
                for _ in range(80):
                    final_phase0 = client.get(f"/archives/jobs/{phase0_job_id}").json()
                    if final_phase0.get("state") in {"done", "error"}:
                        break
                    time.sleep(0.1)
                self.assertEqual(final_phase0.get("state"), "done", final_phase0)
                phase0_snapshot_id = str((final_phase0.get("result") or {}).get("snapshot_id") or "")
                self.assertTrue(phase0_snapshot_id.startswith("snapshot-"))

                phase1 = client.post(
                    f"/archives/{corpus_id}/jobs",
                    json={
                        "dataset_version_id": dataset_version_id,
                        "phase": "archive_phase1",
                        "parent_snapshot_id": phase0_snapshot_id,
                        "write_run_artifacts": False,
                    },
                )
                self.assertEqual(phase1.status_code, 200, phase1.text)
                phase1_job_id = phase1.json()["job_id"]

                final_phase1 = {}
                for _ in range(80):
                    final_phase1 = client.get(f"/archives/jobs/{phase1_job_id}").json()
                    if final_phase1.get("state") in {"done", "error"}:
                        break
                    time.sleep(0.1)
                self.assertEqual(final_phase1.get("state"), "done", final_phase1)
                phase1_snapshot_id = str((final_phase1.get("result") or {}).get("snapshot_id") or "")
                self.assertTrue(phase1_snapshot_id.startswith("snapshot-"))

                timeline = client.get(f"/archives/snapshots/{phase1_snapshot_id}/timeline").json()
                self.assertEqual(timeline.get("phase"), "archive_phase1")
                self.assertIn("classification_summary", timeline)
                classification_summary = timeline.get("classification_summary") or {}
                self.assertGreaterEqual(int(classification_summary.get("classified_count") or 0), 1)
                self.assertTrue(bool(classification_summary.get("object_types")))
                rows = list(timeline.get("items") or []) + list(timeline.get("holding_pen") or [])
                self.assertTrue(rows)
                self.assertTrue(any(list((row.get("classification") or {}).get("routes") or []) for row in rows))

                detail = client.get(
                    f"/archives/snapshots/{phase1_snapshot_id}/objects/{rows[0]['object_id']}"
                ).json()
                self.assertIn("related", detail)
                self.assertIn("by_classification", detail.get("related") or {})

    def test_archive_import_pipeline_and_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            src = tmp / "src"
            src.mkdir()
            (src / "photo_front_1923.txt").write_text(
                "Marguerita Batz at age 26 in Spring 1923",
                encoding="utf-8",
            )
            (src / "photo_back_1923.txt").write_text(
                "Alhambra Historical Society April 12, 1987",
                encoding="utf-8",
            )
            (src / "notes.aup3").write_bytes(b"audacity-project")
            (src / "._noise").write_text("ignore", encoding="utf-8")

            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
                snapshot_db_path=tmp / "snapshots.db",
                snapshot_weights_root=tmp / "snapshot_weights",
                lineage_db_path=tmp / "lineages.db",
                provenance_db_path=tmp / "provenance.db",
                range_db_path=tmp / "ranges.db",
                archive_db_path=tmp / "archives.db",
                archive_storage_root=tmp / "archive_corpora",
            )
            with TestClient(svc.app) as client:
                imported = client.post(
                    "/archives/import",
                    json={"source_paths": [str(src)], "name": "Alhambra Smoke"},
                )
                self.assertEqual(imported.status_code, 200, imported.text)
                imp = imported.json()
                self.assertEqual(imp["corpus"]["name"], "Alhambra Smoke")
                self.assertGreaterEqual(int(imp["noise_file_count"]), 1)
                self.assertTrue(str(imp.get("catalog_asset_id") or "").startswith("asset-"))
                corpus_id = imp["corpus"]["corpus_id"]
                dataset_version_id = imp["dataset_version_id"]
                file_count = int(imp.get("file_count") or 0)

                phase0_review = client.get(
                    f"/archives/{corpus_id}/versions/{dataset_version_id}/phase0_review"
                )
                self.assertEqual(phase0_review.status_code, 200, phase0_review.text)
                phase0_payload = phase0_review.json()
                self.assertGreaterEqual(int((phase0_payload.get("summary") or {}).get("object_count") or 0), 1)
                self.assertTrue(bool(phase0_payload.get("phase_goals")))

                kicked = client.post(
                    f"/archives/{corpus_id}/jobs",
                    json={
                        "dataset_version_id": dataset_version_id,
                        "phase": "archive_pipeline",
                        "write_run_artifacts": False,
                    },
                )
                self.assertEqual(kicked.status_code, 200, kicked.text)
                job_id = kicked.json()["job_id"]

                final_job = {}
                for _ in range(80):
                    final_job = client.get(f"/archives/jobs/{job_id}").json()
                    if final_job.get("state") in {"done", "error"}:
                        break
                    time.sleep(0.1)
                self.assertEqual(final_job.get("state"), "done", final_job)

                result = client.get(f"/jobs/{job_id}/result").json()
                snapshot_id = str(result.get("snapshot_id") or "")
                self.assertTrue(snapshot_id.startswith("snapshot-"))

                timeline = client.get(f"/archives/snapshots/{snapshot_id}/timeline").json()
                summary = timeline.get("summary") or {}
                self.assertLess(int(summary.get("object_count") or 0), file_count)
                self.assertGreaterEqual(int(summary.get("timeline_count") or 0), 1)
                self.assertGreaterEqual(int(summary.get("holding_pen_count") or 0), 1)
                self.assertGreaterEqual(int(summary.get("assertion_count") or 0), 1)
                self.assertIn("phase_goals", timeline)
                self.assertIn("classification_summary", timeline)

                objects = client.get(f"/archives/snapshots/{snapshot_id}/objects").json()
                self.assertGreaterEqual(int(objects.get("count") or 0), 1)
                object_rows = list(objects.get("objects") or [])
                self.assertTrue(object_rows)

                target_object = next(
                    (
                        row for row in object_rows
                        if str(row.get("media_family") or "") == "document" or str(row.get("earliest") or "")
                    ),
                    object_rows[0],
                )
                detail = client.get(
                    f"/archives/snapshots/{snapshot_id}/objects/{target_object['object_id']}"
                ).json()
                self.assertIn("object", detail)
                self.assertIn("anchors", detail)
                self.assertIn("assertions", detail)
                self.assertGreaterEqual(len(detail.get("assertions") or []), 1)
                self.assertIn("health", detail)
                self.assertIn("related", detail)


class ArchivalPhase2EngineTests(unittest.TestCase):
    def test_phase2_audio_segments_preserved_in_metadata(self) -> None:
        objects = [
            {
                "object_id": "obj-audio",
                "object_type": "audio_recording",
                "title": "Interview Tape",
                "media_family": "audio",
                "content_complexity": "single",
                "metadata": {},
                "files": [
                    {
                        "file_id": "file-audio",
                        "role": "recording",
                        "file": {
                            "file_id": "file-audio",
                            "stored_path": "/tmp/fake.wav",
                            "processable": True,
                            "media_family": "audio",
                        },
                    }
                ],
            }
        ]
        mocked_blocks = [
            {
                "block_id": "block-audio-0",
                "provider": "audio_asr",
                "capability": "ok",
                "block_kind": "audio_transcript",
                "source_file_id": "fake-name.wav",
                "page_index": None,
                "segment_id": "audio-0",
                "text": "Hello archive",
                "confidence": 0.73,
                "raw_region": {"kind": "audio_segment", "start_sec": 1.0, "end_sec": 3.5},
                "metadata": {"start_sec": 1.0, "end_sec": 3.5},
            }
        ]
        with patch.object(archive_engine, "_transcribe_audio", return_value=(mocked_blocks, "ok")):
            out = archive_engine._phase2_extract_text(objects)
        metadata = dict(out[0].get("metadata") or {})
        text_blocks = [dict(block) for block in (metadata.get("text_blocks") or []) if isinstance(block, dict)]
        audio_blocks = [block for block in text_blocks if str(block.get("block_kind") or "") == "audio_transcript"]
        self.assertEqual(len(audio_blocks), 1)
        self.assertEqual(audio_blocks[0].get("segment_id"), "audio-0")
        self.assertEqual((audio_blocks[0].get("raw_region") or {}).get("start_sec"), 1.0)
        self.assertEqual((audio_blocks[0].get("raw_region") or {}).get("end_sec"), 3.5)
        segmentation = dict(metadata.get("segmentation") or {})
        self.assertGreaterEqual(int((segmentation.get("summary") or {}).get("region_segments") or 0), 1)
        extraction_summary = dict(metadata.get("extraction_summary") or {})
        self.assertEqual(extraction_summary.get("status"), "extracted")

    def test_phase2_handwriting_capability_unavailable_is_explicit(self) -> None:
        objects = [
            {
                "object_id": "obj-photo",
                "object_type": "photograph",
                "title": "Verso Note",
                "media_family": "image",
                "content_complexity": "single",
                "metadata": {},
                "files": [
                    {
                        "file_id": "file-back",
                        "role": "back",
                        "file": {
                            "file_id": "file-back",
                            "stored_path": "/tmp/fake.png",
                            "processable": False,
                            "media_family": "image",
                        },
                    }
                ],
            }
        ]
        out = archive_engine._phase2_extract_text(objects, {"handwriting_ocr": "none"})
        metadata = dict(out[0].get("metadata") or {})
        handwriting_blocks = [
            dict(block)
            for block in (metadata.get("text_blocks") or [])
            if isinstance(block, dict) and str(block.get("block_kind") or "") == "handwriting_ocr"
        ]
        self.assertEqual(len(handwriting_blocks), 1)
        self.assertEqual(handwriting_blocks[0].get("capability"), "capability_unavailable")


try:
    from PyQt6.QtWidgets import QApplication
except Exception:
    QApplication = None  # type: ignore[assignment]


@unittest.skipIf(QApplication is None or sys.platform == "darwin", "PyQt6 archival widget tests are unstable in headless macOS")
class ArchivalQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QApplication.instance() or QApplication([])

    def test_new_scenario_archival_backbone_visible(self) -> None:
        from insight_local.cvops.ui.new_scenario_dialog import NewScenarioDialog

        def _get(path: str) -> dict[str, object]:
            del path
            return {"datasets": [], "categories": {}, "tabular_datasets": [], "text_datasets": []}

        dlg = NewScenarioDialog(http_get=_get, http_post=lambda _p, _b: {"name": "x"}, models=[])
        try:
            dlg._set_backbone_type("archival_ingestion")
            dlg._on_backbone_changed()
            dlg.show()
            dlg._name.setText("archive_case")
            dlg._on_dataset_changed()
            self._app.processEvents()
            self.assertEqual(dlg._current_backbone(), "archival_ingestion")
            self.assertFalse(dlg._backbone_config_edit.isHidden())
            self.assertTrue(dlg._create_btn.isEnabled())
        finally:
            dlg.deleteLater()

    def test_data_viz_hub_exposes_tabular_and_archival_tabs(self) -> None:
        from insight_local.cvops.ui.data_viz_hub import DataVizHub

        hub = DataVizHub(http_get=lambda _p: {"count": 0, "corpora": []}, http_post=lambda _p, _b: {})
        try:
            self.assertEqual(hub._tabs.count(), 2)
            self.assertEqual(hub._tabs.tabText(0), "Tabular")
            self.assertEqual(hub._tabs.tabText(1), "Archival")
        finally:
            hub.deleteLater()


if __name__ == "__main__":
    unittest.main()
