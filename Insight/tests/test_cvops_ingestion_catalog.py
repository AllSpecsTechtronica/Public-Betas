from __future__ import annotations

import sys
import shutil
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.service import CvOpsService, CvOpsServerHandle
from mlops.pipeline import audio_ops as mlops_audio_ops
from mlops.pipeline import integration as mlops_integration
from mlops.pipeline import registry as mlops_registry
from mlops.pipeline.backbone import BackboneContext
from mlops.pipeline.backbones import get_backbone
from mlops.pipeline.backbones import audio_recognition as audio_backbone


class CvOpsIngestionCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._events_tmp = tempfile.TemporaryDirectory()
        self._old_events_path = mlops_integration.EVENTS_PATH
        mlops_integration.EVENTS_PATH = Path(self._events_tmp.name) / "events.jsonl"

    def tearDown(self) -> None:
        mlops_integration.EVENTS_PATH = self._old_events_path
        self._events_tmp.cleanup()

    def test_cvops_server_handle_disables_transport_ws_keepalive_ping(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            handle = CvOpsServerHandle(host="127.0.0.1", port=8787, db_path=Path(td) / "jobs.db")
            self.assertIsNone(handle.config.ws_ping_interval)
            self.assertIsNone(handle.config.ws_ping_timeout)

    @staticmethod
    def _write_tone(path: Path, *, amplitude: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(8000)
            frames = bytearray()
            for i in range(800):
                sample = amplitude if i % 20 < 10 else -amplitude
                frames.extend(int(sample).to_bytes(2, "little", signed=True))
            wf.writeframes(bytes(frames))

    def test_audiofolder_scenario_creation_and_training_backbone(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp
            database_root = repo_root / "database"
            mlops_root = repo_root / "mlops"
            (mlops_root / "scenarios").mkdir(parents=True)
            (mlops_root / "datasets").mkdir(parents=True)
            (mlops_root / "registry.json").write_text(
                '{"version": 1, "scenarios": []}',
                encoding="utf-8",
            )
            ds = database_root / "AudioTiny"
            self._write_tone(ds / "train" / "alarm" / "a.wav", amplitude=18000)
            self._write_tone(ds / "train" / "speech" / "b.wav", amplitude=6000)

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_audio_root = mlops_registry.ML_AUDIO_ROOT
            old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            old_audio_repo = audio_backbone.REPO_ROOT
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = database_root
                mlops_registry.ML_AUDIO_ROOT = repo_root / "assets" / "ml_audio"
                mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"
                audio_backbone.REPO_ROOT = repo_root

                self.assertEqual(
                    mlops_registry.detect_library_dataset_format(ds),
                    mlops_registry.LIBRARY_DATASET_FORMAT_AUDIOFOLDER,
                )
                audio_dest = mlops_registry.ensure_ml_audio_root() / "AudioTiny"
                audio_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(ds, audio_dest)
                status = mlops_registry.create_scenario_profile(
                    name="audio_tiny",
                    display_name="Audio Tiny",
                    description="Tiny audio recognition test",
                    dataset="AudioTiny",
                    backbone_type="audio_recognition",
                )
                self.assertEqual(status["backbone_type"], "audio_recognition")
                cfg = mlops_registry.get_scenario_config("audio_tiny")
                backbone = get_backbone(cfg.backbone_type, cfg)
                result = backbone.run(
                    BackboneContext(
                        scenario_config=cfg,
                        job_id="job-audio-test",
                        job_type="train",
                        image_bgr=None,
                        payload={},
                        cell_callback=lambda _payload: None,
                    )
                )
                self.assertFalse(result.get("error"), result)
                weights = repo_root / str(result.get("weights") or "")
                self.assertTrue(weights.exists(), result)
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.ML_AUDIO_ROOT = old_audio_root
                mlops_registry.TABULAR_DATASETS_ROOT = old_tabular
                mlops_registry.REGISTRY_PATH = old_registry
                audio_backbone.REPO_ROOT = old_audio_repo

    def test_audio_analysis_and_cleanup_for_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            source = tmp / "noisy.wav"
            self._write_tone(source, amplitude=9000)

            metrics = mlops_audio_ops.analyze_wav(source)
            self.assertEqual(metrics["format"], "wav")
            self.assertGreater(metrics["duration_s"], 0)
            self.assertGreater(metrics["rms"], 0)

            cleaned = tmp / "cleaned.wav"
            result = mlops_audio_ops.clean_wav(source, cleaned, noise_reduction_strength=0.75)
            self.assertTrue(cleaned.exists())
            self.assertEqual(result["output_path"], str(cleaned))
            self.assertGreater(result["after"]["peak"], result["before"]["peak"])
            self.assertLessEqual(result["after"]["duration_s"], result["before"]["duration_s"])

    def test_cvops_audio_analyze_and_clean_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            source = tmp / "voice.wav"
            self._write_tone(source, amplitude=7000)

            svc = CvOpsService(db_path=tmp / "jobs.db", catalog_db_path=tmp / "catalog.db")
            with TestClient(svc.app) as client:
                analyzed = client.post("/audio/analyze", json={"path": str(source)})
                self.assertEqual(analyzed.status_code, 200, analyzed.text)
                self.assertGreater(analyzed.json()["metrics"]["rms"], 0)

                output_name = f"voice_clean_{tmp.name}.wav"
                cleaned = client.post(
                    "/audio/clean",
                    json={"path": str(source), "output_name": output_name},
                )
                self.assertEqual(cleaned.status_code, 200, cleaned.text)
                payload = cleaned.json()
                cleaned_path = Path(payload["cleaned_path"])
                self.assertTrue(cleaned_path.exists())
                self.assertGreater(payload["result"]["after"]["peak"], 0)
                cleaned_path.unlink(missing_ok=True)

    def test_cvops_audio_assets_list_root_media_and_dataset_clips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp / "repo"
            database_root = repo_root / "database"
            audio_root = repo_root / "assets" / "ml_audio"
            mlops_root = repo_root / "mlops"
            (mlops_root / "datasets").mkdir(parents=True)
            (mlops_root / "registry.json").write_text('{"version": 1, "scenarios": []}', encoding="utf-8")
            (audio_root).mkdir(parents=True)
            (audio_root / "loose_source.mp4").write_bytes(b"not-a-real-video")
            self._write_tone(audio_root / "AudioRecognition" / "train" / "alarm" / "a.wav", amplitude=7000)

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_audio_root = mlops_registry.ML_AUDIO_ROOT
            old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = database_root
                mlops_registry.ML_AUDIO_ROOT = audio_root
                mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                svc = CvOpsService(db_path=tmp / "jobs.db", catalog_db_path=tmp / "catalog.db")
                with TestClient(svc.app) as client:
                    listed = client.get("/audio/assets")
                    self.assertEqual(listed.status_code, 200, listed.text)
                    payload = listed.json()
                    rels = {item["relative_path"]: item for item in payload["items"]}
                    self.assertIn("loose_source.mp4", rels)
                    self.assertIn("AudioRecognition/train/alarm/a.wav", rels)
                    self.assertFalse(rels["loose_source.mp4"]["training_ready"])
                    self.assertTrue(rels["AudioRecognition/train/alarm/a.wav"]["training_ready"])
                    self.assertEqual(rels["AudioRecognition/train/alarm/a.wav"]["classification_label"], "alarm")
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.ML_AUDIO_ROOT = old_audio_root
                mlops_registry.TABULAR_DATASETS_ROOT = old_tabular
                mlops_registry.REGISTRY_PATH = old_registry

    def test_audio_clip_collector_cuts_labels_and_builds_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp / "repo"
            database_root = repo_root / "database"
            audio_root = repo_root / "assets" / "ml_audio"
            mlops_root = repo_root / "mlops"
            (mlops_root / "datasets").mkdir(parents=True)
            (mlops_root / "registry.json").write_text(
                '{"version": 1, "scenarios": []}',
                encoding="utf-8",
            )
            source = tmp / "source.wav"
            self._write_tone(source, amplitude=10000)

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_audio_root = mlops_registry.ML_AUDIO_ROOT
            old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = database_root
                mlops_registry.ML_AUDIO_ROOT = audio_root
                mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                svc = CvOpsService(db_path=tmp / "jobs.db", catalog_db_path=tmp / "catalog.db")
                with TestClient(svc.app) as client:
                    created = client.post("/audio/datasets", json={"name": "CollectedAudio"})
                    self.assertEqual(created.status_code, 200, created.text)
                    self.assertEqual(created.json()["path"], str((audio_root / "CollectedAudio").resolve()))

                    clip = client.post(
                        "/audio/collect_clip",
                        json={
                            "dataset": "CollectedAudio",
                            "source_path": str(source),
                            "label": "alarm",
                            "split": "train",
                            "start_ms": 10,
                            "end_ms": 70,
                            "clean": True,
                        },
                    )
                    self.assertEqual(clip.status_code, 200, clip.text)
                    payload = clip.json()
                    clip_path = Path(payload["clip_path"])
                    self.assertTrue(clip_path.exists())
                    self.assertEqual(
                        clip_path.parent.resolve(),
                        (audio_root / "CollectedAudio" / "train" / "alarm").resolve(),
                    )
                    self.assertEqual(payload["dataset_summary"]["count"], 1)
                    self.assertEqual(payload["dataset_summary"]["classes"], ["alarm"])
                    self.assertLessEqual(payload["result"]["after"]["duration_s"], 0.12)

                    listed = client.get("/database/CollectedAudio")
                    self.assertEqual(listed.status_code, 200, listed.text)
                    self.assertEqual(listed.json()["audio_files"][0]["classification_label"], "alarm")
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.ML_AUDIO_ROOT = old_audio_root
                mlops_registry.TABULAR_DATASETS_ROOT = old_tabular
                mlops_registry.REGISTRY_PATH = old_registry

    def test_database_import_folder_copies_complete_dataset_tree(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp / "repo"
            database_root = repo_root / "database"
            mlops_root = repo_root / "mlops"
            (mlops_root / "datasets").mkdir(parents=True)
            (mlops_root / "registry.json").write_text(
                '{"version": 1, "scenarios": []}',
                encoding="utf-8",
            )
            source = tmp / "ExternalTiny"
            (source / "images" / "train").mkdir(parents=True)
            (source / "labels" / "train").mkdir(parents=True)
            (source / "images" / "train" / "part.jpg").write_bytes(b"fake-image")
            (source / "labels" / "train" / "part.txt").write_text(
                "0 0.5 0.5 1 1\n",
                encoding="utf-8",
            )

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = database_root
                mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                svc = CvOpsService(db_path=tmp / "jobs.db", catalog_db_path=tmp / "catalog.db")
                with TestClient(svc.app) as client:
                    imported = client.post(
                        "/database/import_folder",
                        json={"source_path": str(source)},
                    )
                    self.assertEqual(imported.status_code, 200, imported.text)
                    payload = imported.json()
                    self.assertEqual(payload["slug"], "ExternalTiny")
                    self.assertEqual(payload["format"], mlops_registry.LIBRARY_DATASET_FORMAT_YOLO)
                    copied = database_root / "ExternalTiny" / "labels" / "train" / "part.txt"
                    self.assertTrue(copied.exists())

                    listed = client.get("/database/ExternalTiny")
                    self.assertEqual(listed.status_code, 200, listed.text)
                    self.assertEqual(listed.json()["count"], 1)
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.TABULAR_DATASETS_ROOT = old_tabular
                mlops_registry.REGISTRY_PATH = old_registry

    def test_audiofolder_import_copies_to_ml_audio_assets_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp / "repo"
            database_root = repo_root / "database"
            audio_root = repo_root / "assets" / "ml_audio"
            mlops_root = repo_root / "mlops"
            (mlops_root / "datasets").mkdir(parents=True)
            (mlops_root / "registry.json").write_text(
                '{"version": 1, "scenarios": []}',
                encoding="utf-8",
            )
            source = tmp / "ExternalAudio"
            self._write_tone(source / "train" / "alarm" / "a.wav", amplitude=12000)
            self._write_tone(source / "train" / "speech" / "b.wav", amplitude=5000)

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_audio_root = mlops_registry.ML_AUDIO_ROOT
            old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = database_root
                mlops_registry.ML_AUDIO_ROOT = audio_root
                mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                svc = CvOpsService(db_path=tmp / "jobs.db", catalog_db_path=tmp / "catalog.db")
                with TestClient(svc.app) as client:
                    imported = client.post(
                        "/database/import_folder",
                        json={"source_path": str(source)},
                    )
                    self.assertEqual(imported.status_code, 200, imported.text)
                    payload = imported.json()
                    self.assertEqual(payload["slug"], "ExternalAudio")
                    self.assertEqual(payload["category"], mlops_registry.DATASET_CATEGORY_AUDIO)
                    self.assertEqual(payload["format"], mlops_registry.LIBRARY_DATASET_FORMAT_AUDIOFOLDER)
                    self.assertTrue((audio_root / "ExternalAudio" / "train" / "alarm" / "a.wav").exists())
                    self.assertFalse((database_root / "ExternalAudio").exists())

                    listed = client.get("/database/ExternalAudio")
                    self.assertEqual(listed.status_code, 200, listed.text)
                    listed_payload = listed.json()
                    self.assertEqual(listed_payload["category"], mlops_registry.DATASET_CATEGORY_AUDIO)
                    self.assertEqual(listed_payload["count"], 2)

                    all_datasets = client.get("/database")
                    self.assertEqual(all_datasets.status_code, 200, all_datasets.text)
                    self.assertEqual(
                        all_datasets.json()["categories"]["ExternalAudio"],
                        mlops_registry.DATASET_CATEGORY_AUDIO,
                    )
                    self.assertEqual(all_datasets.json()["audio_root"], str(audio_root.resolve()))
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.ML_AUDIO_ROOT = old_audio_root
                mlops_registry.TABULAR_DATASETS_ROOT = old_tabular
                mlops_registry.REGISTRY_PATH = old_registry

    def test_split_first_yolo_import_is_not_misdetected_as_imagefolder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp / "repo"
            database_root = repo_root / "database"
            mlops_root = repo_root / "mlops"
            (mlops_root / "datasets").mkdir(parents=True)
            (mlops_root / "registry.json").write_text(
                '{"version": 1, "scenarios": []}',
                encoding="utf-8",
            )
            source = tmp / "SplitFirstYolo"
            (source / "train" / "images").mkdir(parents=True)
            (source / "train" / "labels").mkdir(parents=True)
            (source / "train" / "images" / "part.jpg").write_bytes(b"fake-image")
            (source / "train" / "labels" / "part.txt").write_text(
                "0 0.5 0.5 1 1\n",
                encoding="utf-8",
            )
            (source / "data.yaml").write_text(
                "train: ../train/images\nval: ../valid/images\nnames: ['Animals']\n",
                encoding="utf-8",
            )

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = database_root
                mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                svc = CvOpsService(db_path=tmp / "jobs.db", catalog_db_path=tmp / "catalog.db")
                with TestClient(svc.app) as client:
                    imported = client.post(
                        "/database/import_folder",
                        json={"source_path": str(source)},
                    )
                    self.assertEqual(imported.status_code, 200, imported.text)
                    payload = imported.json()
                    self.assertEqual(payload["format"], mlops_registry.LIBRARY_DATASET_FORMAT_YOLO)
                    self.assertEqual(payload["classes"], ["Animals"])

                    listed = client.get("/database/SplitFirstYolo")
                    self.assertEqual(listed.status_code, 200, listed.text)
                    listed_payload = listed.json()
                    self.assertEqual(listed_payload["count"], 1)
                    self.assertTrue(listed_payload["images"][0]["has_label"])

                from mlops.pipeline import train as train_mod

                generated = train_mod._build_data_yaml(
                    database_root / "SplitFirstYolo",
                    ["Animals"],
                    tmp / "data.generated.yaml",
                )
                self.assertEqual(generated, tmp / "data.generated.yaml")
                text = generated.read_text(encoding="utf-8")
                self.assertIn("train: train/images", text)
                self.assertIn("val: train/images", text)
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.TABULAR_DATASETS_ROOT = old_tabular
                mlops_registry.REGISTRY_PATH = old_registry

    def test_nested_images_class_yolo_label_path_resolves_to_dataset_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            ds = tmp / "NestedImagesClassYolo"
            image = ds / "images" / "train" / "images" / "part.jpg"
            label = ds / "labels" / "train" / "images" / "part.txt"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            image.write_bytes(b"fake-image")
            label.write_text("0 0.5 0.5 1 1\n", encoding="utf-8")

            entries = mlops_registry.list_dataset_entries_at(ds)
            self.assertEqual(len(entries), 1)
            self.assertTrue(entries[0]["has_label"])
            self.assertEqual(
                Path(entries[0]["label_path"]).resolve(),
                label.resolve(),
            )
            self.assertEqual(
                mlops_registry.resolve_dataset_label_path(image),
                label.resolve(),
            )

    def test_imagefolder_conversion_imports_existing_yolo_sidecar_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo_root = tmp / "repo"
            database_root = repo_root / "database"
            mlops_root = repo_root / "mlops"
            (mlops_root / "datasets").mkdir(parents=True)
            (mlops_root / "registry.json").write_text(
                '{"version": 1, "scenarios": []}',
                encoding="utf-8",
            )
            source = tmp / "ExternalClassified"
            labeled_dir = source / "train" / "defect"
            labeled_dir.mkdir(parents=True)
            (labeled_dir / "part.jpg").write_bytes(b"fake-image")
            (labeled_dir / "part.txt").write_text("0.5 0.5 0.25 0.25\n", encoding="utf-8")
            (labeled_dir / "missing.jpg").write_bytes(b"fake-image")
            mirrored_dir = source / "valid" / "good"
            mirrored_dir.mkdir(parents=True)
            (mirrored_dir / "ok.jpg").write_bytes(b"fake-image")
            mirrored_label = source / "labels" / "valid" / "good" / "ok.txt"
            mirrored_label.parent.mkdir(parents=True)
            mirrored_label.write_text("0.5 0.5 0.5 0.5\n", encoding="utf-8")

            old_repo = mlops_registry.REPO_ROOT
            old_mlops = mlops_registry.MLOPS_ROOT
            old_db = mlops_registry.DATABASE_ROOT
            old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
            old_registry = mlops_registry.REGISTRY_PATH
            try:
                mlops_registry.REPO_ROOT = repo_root
                mlops_registry.MLOPS_ROOT = mlops_root
                mlops_registry.DATABASE_ROOT = database_root
                mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
                mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"

                svc = CvOpsService(db_path=tmp / "jobs.db", catalog_db_path=tmp / "catalog.db")
                with TestClient(svc.app) as client:
                    imported = client.post(
                        "/database/import_folder",
                        json={"source_path": str(source)},
                    )
                    self.assertEqual(imported.status_code, 200, imported.text)
                    self.assertEqual(
                        imported.json()["format"],
                        mlops_registry.LIBRARY_DATASET_FORMAT_IMAGEFOLDER,
                    )
                    inspected = client.get("/database/ExternalClassified")
                    self.assertEqual(inspected.status_code, 200, inspected.text)
                    self.assertEqual(inspected.json()["detection_label_count"], 2)

                    converted = client.post(
                        "/database/ExternalClassified/convert/imagefolder_to_yolo",
                        json={"mode": "import_labels"},
                    )
                    self.assertEqual(converted.status_code, 200, converted.text)
                    payload = converted.json()
                    self.assertEqual(payload["mode"], "import_labels")
                    self.assertEqual(payload["converted"], 3)
                    self.assertEqual(payload["imported_labels"], 2)
                    self.assertEqual(payload["normalized_label_lines"], 2)
                    self.assertEqual(payload["missing_labels"], 1)

                    out_slug = payload["output_slug"]
                    copied = database_root / out_slug / "labels" / "train" / "defect" / "part.txt"
                    self.assertTrue(copied.exists())
                    self.assertEqual(copied.read_text(encoding="utf-8"), "0 0.5 0.5 0.25 0.25\n")
                    mirrored = database_root / out_slug / "labels" / "val" / "good" / "ok.txt"
                    self.assertTrue(mirrored.exists())
                    self.assertEqual(mirrored.read_text(encoding="utf-8"), "1 0.5 0.5 0.5 0.5\n")
            finally:
                mlops_registry.REPO_ROOT = old_repo
                mlops_registry.MLOPS_ROOT = old_mlops
                mlops_registry.DATABASE_ROOT = old_db
                mlops_registry.TABULAR_DATASETS_ROOT = old_tabular
                mlops_registry.REGISTRY_PATH = old_registry

    def test_document_ingestion_managed_copy_and_reference_health(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
            )
            with TestClient(svc.app) as client:
                managed = client.post(
                    "/ingest/assets",
                    data={
                        "name": "notes-managed",
                        "source_type": "txt",
                        "storage_mode": "managed_copy",
                        "sector_id": "sector-root",
                    },
                    files={"file": ("notes.txt", b"hello world\nline2\n", "text/plain")},
                )
                self.assertEqual(managed.status_code, 200, managed.text)
                managed_payload = managed.json()
                self.assertEqual(managed_payload["source_type"], "txt")
                self.assertEqual(managed_payload["storage_mode"], "managed_copy")
                self.assertTrue(Path(managed_payload["managed_path"]).exists())
                text_stats = (managed_payload.get("metadata") or {}).get("text_stats") or {}
                self.assertEqual(int(text_stats.get("line_count", 0)), 2)

                ref_path = tmp / "ref.txt"
                ref_path.write_text("alpha beta gamma", encoding="utf-8")
                referenced = client.post(
                    "/ingest/assets",
                    data={
                        "name": "notes-ref",
                        "source_type": "txt",
                        "storage_mode": "reference",
                        "source_uri": str(ref_path),
                        "sector_id": "sector-root",
                    },
                )
                self.assertEqual(referenced.status_code, 200, referenced.text)
                ref_asset = referenced.json()
                asset_id = str(ref_asset["asset_id"])

                got_1 = client.get(f"/ingest/assets/{asset_id}")
                self.assertEqual(got_1.status_code, 200, got_1.text)
                self.assertEqual(got_1.json()["availability_status"], "ok")

                ref_path.unlink()
                got_2 = client.get(f"/ingest/assets/{asset_id}")
                self.assertEqual(got_2.status_code, 200, got_2.text)
                self.assertEqual(got_2.json()["availability_status"], "missing")

    def test_sector_tree_assign_search_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
            )
            with TestClient(svc.app) as client:
                s1 = client.post("/sectors", json={"name": "Finance", "parent_id": "sector-root"})
                self.assertEqual(s1.status_code, 200, s1.text)
                finance = s1.json()

                s2 = client.post("/sectors", json={"name": "Ops", "parent_id": "sector-root"})
                self.assertEqual(s2.status_code, 200, s2.text)
                ops = s2.json()

                renamed = client.patch(f"/sectors/{finance['sector_id']}", json={"name": "Banking"})
                self.assertEqual(renamed.status_code, 200, renamed.text)
                self.assertEqual(renamed.json()["path"], "/Banking")

                moved = client.post(
                    f"/sectors/{ops['sector_id']}/move",
                    json={"parent_id": finance["sector_id"]},
                )
                self.assertEqual(moved.status_code, 200, moved.text)
                self.assertEqual(moved.json()["path"], "/Banking/Ops")

                ingest = client.post(
                    "/ingest/assets",
                    data={
                        "name": "team-doc",
                        "source_type": "md",
                        "storage_mode": "managed_copy",
                        "sector_id": "sector-root",
                    },
                    files={"file": ("team.md", b"# Team\nOperations notes", "text/markdown")},
                )
                self.assertEqual(ingest.status_code, 200, ingest.text)
                asset = ingest.json()

                assign = client.post(
                    f"/assets/{asset['asset_id']}/assign_sector",
                    json={"sector_id": ops["sector_id"]},
                )
                self.assertEqual(assign.status_code, 200, assign.text)
                self.assertEqual(assign.json()["sector_path"], "/Banking/Ops")

                search = client.get(
                    "/catalog/search",
                    params={"q": "team", "sector_path": "/Banking", "include_descendants": 1},
                )
                self.assertEqual(search.status_code, 200, search.text)
                search_payload = search.json()
                self.assertEqual(int(search_payload["count"]), 1)
                self.assertEqual(search_payload["items"][0]["asset_id"], asset["asset_id"])

                summary = client.get("/catalog/sectors/Banking/summary")
                self.assertEqual(summary.status_code, 200, summary.text)
                summary_payload = summary.json()
                self.assertEqual(int(summary_payload["assets_total"]), 1)
                by_type = summary_payload.get("counts_by_type") or {}
                self.assertEqual(int(by_type.get("md", 0)), 1)

    def test_existing_health_endpoint_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
            )
            with TestClient(svc.app) as client:
                health = client.get("/health")
                self.assertEqual(health.status_code, 200, health.text)
                payload = health.json()
                self.assertEqual(payload.get("status"), "ok")

    def test_train_request_persists_final_model_name_in_job_payload(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
            )
            with TestClient(svc.app) as client:
                from insight_local.cvops import service as cvops_service

                class _Cfg:
                    name = "demo"
                    base_model = "assets/models/yolo26n.pt"
                    hyperparams = {}

                old_get = cvops_service.mlops_registry.get_scenario_config
                try:
                    cvops_service.mlops_registry.get_scenario_config = lambda _name: _Cfg()
                    res = client.post(
                        "/scenarios/demo/train",
                        json={
                            "final_model_name": "Factory Line Final",
                            "base_model_override": "assets/models/yolo26s.pt",
                        },
                    )
                    self.assertEqual(res.status_code, 200, res.text)
                    job_id = res.json()["job_id"]
                    job = svc.store.get_job(job_id)
                    self.assertEqual(job.payload.get("final_model_name"), "Factory Line Final")
                    self.assertEqual(job.payload.get("base_model_override"), "assets/models/yolo26s.pt")
                finally:
                    cvops_service.mlops_registry.get_scenario_config = old_get

    def test_set_scenario_dataset_route_uses_registry_helper(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
            )
            with TestClient(svc.app) as client:
                with patch(
                    "insight_local.cvops.service.mlops_registry.set_scenario_dataset",
                    return_value={"name": "demo", "dataset": "tiger111_subset"},
                ) as set_dataset_mock:
                    res = client.post(
                        "/scenarios/demo/dataset",
                        json={"dataset": "tiger111_subset"},
                    )
                    self.assertEqual(res.status_code, 200, res.text)
                    self.assertEqual(res.json()["dataset"], "tiger111_subset")
                    set_dataset_mock.assert_called_once_with("demo", "tiger111_subset")

    def test_run_train_job_passes_base_model_override_to_training(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
            )

            class _Cfg:
                name = "demo"
                backbone_type = "yolo_detection"

            job = svc.store.create_job(
                job_id="job-train-override",
                scenario="demo",
                job_type="train",
                source="test",
                image_path="",
                payload={
                    "base_model_override": "assets/models/yolo26s.pt",
                    "resume": True,
                    "save_period": 1,
                },
            )

            with patch("insight_local.cvops.service.mlops_registry.get_scenario_config", return_value=_Cfg()):
                with patch("insight_local.cvops.service.run_training") as run_training_mock:
                    run_training_mock.return_value = {
                        "output": "",
                        "weights": "",
                        "data_yaml": "",
                        "map50": "",
                        "map50_95": "",
                        "final_model_name": "",
                        "final_model_file": "",
                        "resumed_from": "",
                        "save_period": 1,
                    }

                    result = svc._run_train_job(job)

            self.assertEqual(result.get("error"), "")
            self.assertTrue(run_training_mock.called)
            self.assertEqual(run_training_mock.call_args.kwargs.get("base_model_override"), "assets/models/yolo26s.pt")


if __name__ == "__main__":
    unittest.main()
