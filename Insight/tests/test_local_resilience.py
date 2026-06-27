from __future__ import annotations

import base64
import sys
import tempfile
import threading
import time
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local import config as config_mod
from insight_local.config import RuntimeConfig
from insight_local.engine.action_runner import SingleFlightActionRunner
from insight_local.engine import detector as detector_mod
from insight_local.engine import local_session as session_mod
from insight_local.engine import segmentation_engine as segmentation_mod
from insight_local import export_apple as export_apple_mod
from insight_local.engine.gallery_db import GalleryDB
from insight_local.engine.perception import InsightPerceptionEngine
from insight_local.filtering import matches_detection_filter, matches_detection_view, normalized_query_tokens
from insight_local.privacy import detect_privacy_status
from insight_local.runtime_profile import RuntimeProfile


def make_runtime_profile(
    *,
    system: str = "Darwin",
    machine: str = "arm64",
    torch_device: str = "cpu",
    accelerator: str = "cpu",
    has_cuda: bool = False,
    has_mps: bool = False,
    onnx_providers: tuple[str, ...] = (),
    preferred_model_suffixes: tuple[str, ...] = (".pt", ".engine", ".onnx", ".mlpackage"),
) -> RuntimeProfile:
    return RuntimeProfile(
        system=system,
        machine=machine,
        torch_device=torch_device,
        accelerator=accelerator,
        has_cuda=has_cuda,
        has_mps=has_mps,
        onnx_providers=onnx_providers,
        preferred_model_suffixes=preferred_model_suffixes,
    )


class FakeThread:
    def __init__(self, target=None, daemon=None, name=None, args=(), kwargs=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}

    def start(self) -> None:
        return

    def is_alive(self) -> bool:
        return False

    def join(self, timeout=None) -> None:
        return


class FakeGalleryDB:
    integrity_ok = True
    integrity_message = "ok"
    quarantines: list[Path] = []
    face_backend_ready = True
    face_backend_error_text = "recognizer unavailable"

    def __init__(self, db_path: Path, embedder=None) -> None:
        self.db_path = db_path
        self.matrix = np.zeros((0, 576), dtype=np.float32)
        self.matrix_labels: list[str] = []
        self.matrix_groups: list[str] = []
        self.matrix_sources: list[str] = []
        self.profile_matrix = np.zeros((0, 576), dtype=np.float32)
        self.profile_labels: list[str] = []
        self.profile_groups: list[str] = []
        self.profile_sources: list[str] = []
        self.profile_sample_counts: list[int] = []
        self.similarity_items: list[dict[str, object]] = []
        self.similarity_item_paths: dict[int, str] = {}

    @property
    def has_gallery(self) -> bool:
        return True

    def ensure_face_backend(self) -> bool:
        return self.face_backend_ready

    @property
    def face_backend_error(self) -> str:
        return "" if self.face_backend_ready else self.face_backend_error_text

    def ensure_similarity_backend(self) -> bool:
        return self.face_backend_ready

    @property
    def similarity_backend_error(self) -> str:
        return "" if self.face_backend_ready else "similarity unavailable"

    @staticmethod
    def verify_integrity(db_path: Path) -> tuple[bool, str]:
        return FakeGalleryDB.integrity_ok, FakeGalleryDB.integrity_message

    @staticmethod
    def quarantine_database(db_path: Path, reason: str = "") -> Path:
        quarantine = db_path.with_suffix(".corrupt.db")
        FakeGalleryDB.quarantines.append(quarantine)
        return quarantine

    def build_matrix(self) -> int:
        return 0

    def get_stats(self):
        return session_mod.GalleryStats(identity_count=0, image_count=0, group_names=[], last_rebuild=0.0, similarity_item_count=len(self.similarity_items))

    def list_identities(self, group_filter: str = "") -> list:
        return []

    def get_identity_images(self, identity_name: str) -> list[str]:
        return []

    def list_similarity_items(self) -> list:
        return [
            type(
                "SimilarityItem",
                (),
                {
                    "item_id": int(item["item_id"]),
                    "display_name": str(item["display_name"]),
                    "batch_label": str(item["batch_label"]),
                    "source_path": str(item["source_path"]),
                    "thumb_png": bytes(item.get("thumb_png", b"")),
                },
            )()
            for item in self.similarity_items
        ]

    def get_similarity_item_path(self, item_id: int) -> str:
        return self.similarity_item_paths.get(int(item_id), "")

    def find_similar_items(self, item_id: int, top_k: int = 12) -> list[dict[str, object]]:
        return [
            {
                "item_id": 2,
                "display_name": "match",
                "batch_label": "batch",
                "similarity": 0.91,
                "source_path": "/tmp/match.jpg",
            }
        ] if int(item_id) in self.similarity_item_paths else []

    def ingest_folder(self, folder: Path, identity_name: str, group_name: str = "", progress_cb=None):
        return 0, []

    def ingest_single(self, image_path: Path, identity_name: str, group_name: str = ""):
        return True, ""

    def ingest_similarity_image(self, image_path: Path, batch_label: str = ""):
        item_id = len(self.similarity_items) + 1
        self.similarity_item_paths[item_id] = str(image_path)
        self.similarity_items.append(
            {
                "item_id": item_id,
                "display_name": image_path.stem,
                "batch_label": batch_label or image_path.parent.name,
                "source_path": str(image_path),
                "thumb_png": b"",
            }
        )
        return True, ""

    def ingest_similarity_folder(self, folder: Path, progress_cb=None):
        item_id = len(self.similarity_items) + 1
        fake_path = str(folder / "one.jpg")
        self.similarity_item_paths[item_id] = fake_path
        self.similarity_items.append(
            {
                "item_id": item_id,
                "display_name": "one",
                "batch_label": folder.name,
                "source_path": fake_path,
                "thumb_png": b"",
            }
        )
        if progress_cb:
            progress_cb(0, 1, "one.jpg")
        return 1, []

    def ingest_bgr(self, bgr, identity_name: str, group_name: str = "", source_label: str = "crop"):
        return True, ""

    def delete_identity(self, identity_name: str) -> int:
        return 0

    def delete_similarity_item(self, item_id: int) -> int:
        before = len(self.similarity_items)
        self.similarity_items = [item for item in self.similarity_items if int(item["item_id"]) != int(item_id)]
        self.similarity_item_paths.pop(int(item_id), None)
        return before - len(self.similarity_items)

    def rename_identity(self, old_name: str, new_name: str) -> None:
        return

    def close(self) -> None:
        return


class FakeEmbedder:
    ready = True
    error = "recognizer unavailable"

    def __init__(self) -> None:
        self.load_error = None

    def ensure_loaded(self) -> bool:
        self.load_error = None if self.ready else self.error
        return self.ready

    def embed(self, bgr_crop):
        return np.ones((576,), dtype=np.float32)


class FakeRecognitionWorker:
    def __init__(
        self,
        embedder,
        gallery_db,
        broadcaster,
        threshold=0.72,
        margin_threshold=0.045,
        top_k=5,
    ) -> None:
        self._top_k = top_k
        self.enqueues: list[dict[str, object]] = []

    def set_threshold(self, threshold: float) -> None:
        return

    def enqueue(self, entry_id, crop, label="", source="auto", track_id=0) -> bool:
        self.enqueues.append(
            {
                "entry_id": entry_id,
                "crop_shape": getattr(crop, "shape", None),
                "label": label,
                "source": source,
                "track_id": track_id,
            }
        )
        return True

    def stop(self) -> None:
        return


class FakeDetector:
    ready = True
    reload_success = True
    last_error_value = "detector unavailable"

    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._ready = self.ready
        self._last_error = "" if self.ready else self.last_error_value

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def last_error(self) -> str:
        return self._last_error

    def ensure_ready(self) -> bool:
        self._ready = self.ready
        self._last_error = "" if self._ready else self.last_error_value
        return self._ready

    def reload(self) -> bool:
        self._ready = self.reload_success
        self._last_error = "" if self._ready else self.last_error_value
        return self._ready

    def predict(self, frame, *, image_size: int, confidence: float, iou: float, max_det: int):
        return []


class FakeFrameSource:
    source_status: dict[str, bool] = {"camera": True, "video": True}

    def __init__(self, config: RuntimeConfig, status_callback) -> None:
        self.config = config
        self.status_callback = status_callback
        self.current_source = config.source
        self.prepared_source = None
        self.last_shape = (720, 1280)
        self.failure_count = 0
        self.last_error = ""

    def prepare_switch(self, requested=None) -> str:
        target = requested or ("video" if self.current_source == "camera" else "camera")
        if not self.source_status.get(target, False):
            self.failure_count += 1
            self.last_error = f"{target} unavailable"
            raise RuntimeError(self.last_error)
        self.prepared_source = target
        return target

    def commit_prepared_switch(self, expected_source=None) -> str:
        if self.prepared_source is None:
            raise RuntimeError("no prepared source")
        self.current_source = self.prepared_source
        self.prepared_source = None
        self.failure_count = 0
        self.last_error = ""
        return self.current_source

    def cancel_prepared_switch(self) -> None:
        self.prepared_source = None

    def describe_source(self) -> str:
        return self.current_source

    def reopen_current_source(self) -> None:
        if not self.source_status.get(self.current_source, False):
            self.failure_count += 1
            self.last_error = f"{self.current_source} unavailable"
            raise RuntimeError(self.last_error)
        self.failure_count = 0
        self.last_error = ""

    def cleanup(self) -> None:
        return

    def read(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)


class FakePerception:
    def __init__(
        self,
        config: RuntimeConfig,
        broadcaster,
        source_label_getter,
        detector=None,
        recognition_worker=None,
        detector_state_callback=None,
        detector_recovery_callback=None,
        recognition_control_callback=None,
    ) -> None:
        self.config = config
        self.broadcaster = broadcaster
        self.source_label_getter = source_label_getter
        self.recognition_worker = recognition_worker
        self.detector = detector
        self.detector_latched = False
        self.live_overlays = []
        self._roi = None
        self.status_messages: list[tuple[str, str]] = []

    def start(self) -> None:
        return

    def publish_state(self, force: bool = False) -> None:
        return

    def get_runtime_metrics(self) -> dict:
        return {"frames_dropped": 0, "frames_processed": 0}

    def get_runtime_snapshot(self) -> dict:
        return {"settings": {}, "roi": self._roi, "metrics": self.get_runtime_metrics()}

    def attach_detector(self, detector) -> None:
        self.detector = detector
        self.detector_latched = False

    def set_roi(self, x1, y1, x2, y2, shape="rect") -> None:
        self._roi = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "shape": shape}

    def clear_roi(self) -> None:
        self._roi = None

    def set_status(self, message: str, level: str = "info") -> None:
        self.status_messages.append((message, level))

    def reset_scene(self) -> None:
        return

    def clear_history(self) -> None:
        return

    def delete_history_entry(self, entry_id: int) -> None:
        return

    def update_settings(self, settings: dict) -> None:
        return

    def set_focus(self, track_id: int) -> None:
        return

    def clear_focus(self) -> None:
        return

    def capture_roi_snapshot(self) -> None:
        return

    def get_last_roi_capture_context(self):
        return {"image": "x", "scan_results": [], "captured_at": 1}

    def stop(self) -> None:
        return

    def apply_recognition_result(self, result: dict) -> None:
        return


class ActionRunnerTests(unittest.TestCase):
    def test_rejects_duplicate_group_while_active(self) -> None:
        statuses: list[tuple[str, str]] = []
        runner = SingleFlightActionRunner(lambda message, level: statuses.append((message, level)), max_workers=1)
        started = threading.Event()
        release = threading.Event()

        def blocking() -> None:
            started.set()
            release.wait(timeout=1.0)

        self.assertTrue(runner.submit("roi_ai", "ROI AI analysis", blocking, "ROI AI already running"))
        self.assertTrue(started.wait(0.5))
        self.assertFalse(runner.submit("roi_ai", "ROI AI analysis", lambda: None, "ROI AI already running"))
        self.assertIn(("ROI AI already running", "info"), statuses)
        release.set()
        deadline = time.monotonic() + 0.5
        while runner.is_active("roi_ai") and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(runner.is_active("roi_ai"))
        self.assertTrue(runner.submit("roi_ai", "ROI AI analysis", lambda: None, "ROI AI already running"))
        runner.shutdown(timeout_sec=0.5)

    def test_clears_active_group_after_exception(self) -> None:
        statuses: list[tuple[str, str]] = []
        runner = SingleFlightActionRunner(lambda message, level: statuses.append((message, level)), max_workers=1)

        def boom() -> None:
            raise RuntimeError("boom")

        self.assertTrue(runner.submit("gallery_write", "Gallery operation", boom, "Gallery operation already running"))
        deadline = time.monotonic() + 0.5
        while runner.is_active("gallery_write") and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(runner.is_active("gallery_write"))
        self.assertTrue(any(message == "Gallery operation failed: boom" for message, _ in statuses))
        self.assertTrue(runner.submit("gallery_write", "Gallery operation", lambda: None, "Gallery operation already running"))
        runner.shutdown(timeout_sec=0.5)


class ConfigModelDiscoveryTests(unittest.TestCase):
    def test_normalize_text_color_accepts_only_supported_modes(self) -> None:
        self.assertEqual(config_mod.normalize_text_color("black"), "black")
        self.assertEqual(config_mod.normalize_text_color("bright-cyan"), "bright-cyan")
        self.assertEqual(config_mod.normalize_text_color("cyan"), "black")

    def test_normalize_fps_supports_uncapped_and_high_values(self) -> None:
        self.assertEqual(config_mod.normalize_fps(0), 0)
        self.assertEqual(config_mod.normalize_fps(-1), 0)
        self.assertEqual(config_mod.normalize_fps(240), 240)
        self.assertEqual(config_mod.normalize_fps(5), 10)

    def test_parse_args_accepts_text_color_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "yolo26n.pt"
            model_path.write_text("", encoding="utf-8")
            with mock.patch.object(config_mod, "discover_model_choices", return_value={"yolo26n": model_path}):
                with mock.patch.object(config_mod, "DEFAULT_MODEL_PATH", model_path):
                    with mock.patch.object(sys, "argv", ["insight-local", "--model", "yolo26n", "--text-color", "bright-cyan"]):
                        config = config_mod.parse_args()

        self.assertEqual(config.text_color, "bright-cyan")

    def test_discover_model_choices_reads_yolo_files_from_folder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "yolo11n.pt").write_text("", encoding="utf-8")
            (models_dir / "yolov10n.pt").write_text("", encoding="utf-8")
            (models_dir / "face_detection_yunet_2023mar.onnx").write_text("", encoding="utf-8")
            (models_dir / "notes.txt").write_text("", encoding="utf-8")

            choices = config_mod.discover_model_choices(models_dir)

        self.assertEqual(sorted(choices.keys()), ["yolo11n", "yolov10n"])
        self.assertTrue(all(path.suffix == ".pt" for path in choices.values()))

    def test_discover_model_choices_prefers_apple_variants_and_detects_mlpackage_dirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "yolo26n.pt").write_text("", encoding="utf-8")
            (models_dir / "yolo26n.onnx").write_text("", encoding="utf-8")
            (models_dir / "yolo26n.mlpackage").mkdir()

            runtime = make_runtime_profile(
                system="Darwin",
                machine="arm64",
                torch_device="mps",
                accelerator="apple-silicon",
                has_mps=True,
                onnx_providers=("CoreMLExecutionProvider", "CPUExecutionProvider"),
                preferred_model_suffixes=(".mlpackage", ".onnx", ".pt", ".engine"),
            )
            with mock.patch.object(config_mod, "profile_runtime", return_value=runtime):
                choices = config_mod.discover_model_choices(models_dir)
                self.assertEqual(choices["yolo26n"].suffix, ".mlpackage")
                self.assertTrue(choices["yolo26n"].is_dir())

    def test_discover_model_choices_prefers_cuda_variants_on_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "yolo26n.pt").write_text("", encoding="utf-8")
            (models_dir / "yolo26n.onnx").write_text("", encoding="utf-8")
            (models_dir / "yolo26n.engine").write_text("", encoding="utf-8")

            runtime = make_runtime_profile(
                system="Windows",
                machine="AMD64",
                torch_device="cuda",
                accelerator="nvidia-cuda",
                has_cuda=True,
                onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                preferred_model_suffixes=(".engine", ".onnx", ".pt", ".mlpackage"),
            )
            with mock.patch.object(config_mod, "profile_runtime", return_value=runtime):
                choices = config_mod.discover_model_choices(models_dir)

        self.assertEqual(choices["yolo26n"].suffix, ".engine")

    def test_list_model_names_returns_discovered_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "yolo26n.pt").write_text("", encoding="utf-8")
            (models_dir / "yolo26s.onnx").write_text("", encoding="utf-8")

            names = config_mod.list_model_names(models_dir)

        self.assertEqual(names, ["yolo26n", "yolo26s"])

    def test_model_catalog_lists_all_supported_yolo_variants(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "yolo26n.pt").write_text("", encoding="utf-8")
            (models_dir / "yolo26n.onnx").write_text("", encoding="utf-8")
            (models_dir / "yolo26n.mlpackage").mkdir()
            (models_dir / "face_detection_yunet_2023mar.onnx").write_text("", encoding="utf-8")

            catalog = config_mod.discover_model_catalog(models_dir)

        self.assertEqual(
            sorted(catalog.keys()),
            ["yolo26n.mlpackage", "yolo26n.onnx", "yolo26n.pt"],
        )

    def test_resolve_model_path_accepts_exact_catalog_filename_without_alias_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            pt_path = models_dir / "yolo26n.pt"
            onnx_path = models_dir / "yolo26n.onnx"
            pt_path.write_text("", encoding="utf-8")
            onnx_path.write_text("", encoding="utf-8")

            with mock.patch.object(
                config_mod,
                "discover_model_catalog",
                return_value={"yolo26n.pt": pt_path.resolve(), "yolo26n.onnx": onnx_path.resolve()},
            ):
                resolved = config_mod.resolve_model_path("yolo26n.onnx")

        self.assertEqual(resolved, onnx_path.resolve())

    def test_model_choice_name_returns_exact_catalog_filename_for_variant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            onnx_path = models_dir / "yolo26n.onnx"
            onnx_path.write_text("", encoding="utf-8")

            with mock.patch.object(
                config_mod,
                "discover_model_catalog",
                return_value={"yolo26n.onnx": onnx_path.resolve()},
            ):
                choice = config_mod.model_choice_name(onnx_path)

        self.assertEqual(choice, "yolo26n.onnx")


class DetectorRuntimeTests(unittest.TestCase):
    def test_exported_backends_skip_explicit_device_dispatch(self) -> None:
        class FakeYoloModel:
            def __init__(self, path: str) -> None:
                self.path = path
                self.to_calls: list[str] = []
                self.predict_calls: list[dict[str, object]] = []

            def to(self, device: str) -> None:
                self.to_calls.append(device)

            def predict(self, **kwargs):
                self.predict_calls.append(kwargs)
                return []

        created_models: list[FakeYoloModel] = []

        def fake_yolo(path: str) -> FakeYoloModel:
            model = FakeYoloModel(path)
            created_models.append(model)
            return model

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "yolo26n.onnx"
            model_path.write_text("", encoding="utf-8")
            fake_ultralytics = types.SimpleNamespace(YOLO=fake_yolo)

            with mock.patch.object(detector_mod, "pick_device", return_value="mps"):
                with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                    detector = detector_mod.YoloDetector(model_path)
                    self.assertTrue(detector.reload())
                    detector.predict(
                        np.zeros((10, 10, 3), dtype=np.uint8),
                        image_size=640,
                        confidence=0.25,
                        iou=0.2,
                        max_det=100,
                    )

        self.assertEqual(len(created_models), 1)
        self.assertEqual(created_models[0].to_calls, [])
        self.assertNotIn("device", created_models[0].predict_calls[0])

    def test_pt_backend_keeps_explicit_device_dispatch(self) -> None:
        class FakeYoloModel:
            def __init__(self, path: str) -> None:
                self.path = path
                self.to_calls: list[str] = []
                self.predict_calls: list[dict[str, object]] = []

            def to(self, device: str) -> None:
                self.to_calls.append(device)

            def predict(self, **kwargs):
                self.predict_calls.append(kwargs)
                return []

        created_models: list[FakeYoloModel] = []

        def fake_yolo(path: str) -> FakeYoloModel:
            model = FakeYoloModel(path)
            created_models.append(model)
            return model

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "yolo26n.pt"
            model_path.write_text("", encoding="utf-8")
            fake_ultralytics = types.SimpleNamespace(YOLO=fake_yolo)

            with mock.patch.object(detector_mod, "pick_device", return_value="mps"):
                with mock.patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
                    detector = detector_mod.YoloDetector(model_path)
                    self.assertTrue(detector.reload())
                    detector.predict(
                        np.zeros((10, 10, 3), dtype=np.uint8),
                        image_size=640,
                        confidence=0.25,
                        iou=0.2,
                        max_det=100,
                    )

        self.assertEqual(len(created_models), 1)
        self.assertEqual(created_models[0].to_calls, ["mps"])
        self.assertEqual(created_models[0].predict_calls[0]["device"], "mps")


class RuntimeExportTests(unittest.TestCase):
    def test_ensure_runtime_model_exports_missing_apple_variants_from_pt(self) -> None:
        exported: list[dict[str, object]] = []

        def fake_export(model_path: Path, **kwargs):
            exported.append({"model_path": model_path, **kwargs})
            model_path.with_suffix(".mlpackage").mkdir()
            model_path.with_suffix(".onnx").write_text("", encoding="utf-8")
            return [model_path.with_suffix(".mlpackage"), model_path.with_suffix(".onnx")]

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "yolo26n.pt"
            model_path.write_text("", encoding="utf-8")
            runtime = make_runtime_profile(
                system="Darwin",
                machine="arm64",
                torch_device="mps",
                accelerator="apple-silicon",
                has_mps=True,
                preferred_model_suffixes=(".mlpackage", ".onnx", ".pt", ".engine"),
            )
            with mock.patch.object(export_apple_mod, "export_runtime_variants", side_effect=fake_export):
                prepared = export_apple_mod.ensure_runtime_model(model_path, runtime=runtime)

        self.assertEqual(prepared.suffix, ".mlpackage")
        self.assertEqual(len(exported), 1)
        self.assertTrue(exported[0]["export_coreml"])
        self.assertTrue(exported[0]["export_onnx"])

    def test_ensure_runtime_model_exports_only_missing_apple_variant_when_partial_set_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "yolo26n.pt"
            model_path.write_text("", encoding="utf-8")
            model_path.with_suffix(".mlpackage").mkdir()
            runtime = make_runtime_profile(
                system="Darwin",
                machine="arm64",
                torch_device="mps",
                accelerator="apple-silicon",
                has_mps=True,
                preferred_model_suffixes=(".mlpackage", ".onnx", ".pt", ".engine"),
            )
            with mock.patch.object(export_apple_mod, "export_runtime_variants") as export_mock:
                prepared = export_apple_mod.ensure_runtime_model(model_path, runtime=runtime)

        self.assertEqual(prepared.suffix, ".mlpackage")
        export_mock.assert_called_once()
        self.assertTrue(export_mock.call_args.kwargs["export_onnx"])
        self.assertFalse(export_mock.call_args.kwargs["export_coreml"])

    def test_ensure_runtime_model_exports_onnx_for_windows_cuda_hosts(self) -> None:
        exported: list[dict[str, object]] = []

        def fake_export(model_path: Path, **kwargs):
            exported.append({"model_path": model_path, **kwargs})
            model_path.with_suffix(".onnx").write_text("", encoding="utf-8")
            return [model_path.with_suffix(".onnx")]

        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "yolo26n.pt"
            model_path.write_text("", encoding="utf-8")
            runtime = make_runtime_profile(
                system="Windows",
                machine="AMD64",
                torch_device="cuda",
                accelerator="nvidia-cuda",
                has_cuda=True,
                onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
                preferred_model_suffixes=(".engine", ".onnx", ".pt", ".mlpackage"),
            )
            with mock.patch.object(export_apple_mod, "export_runtime_variants", side_effect=fake_export):
                prepared = export_apple_mod.ensure_runtime_model(model_path, runtime=runtime)

        self.assertEqual(prepared.suffix, ".onnx")
        self.assertEqual(len(exported), 1)
        self.assertTrue(exported[0]["export_onnx"])
        self.assertFalse(exported[0]["export_coreml"])
        self.assertFalse(exported[0]["export_engine"])


class LocalSessionAppleRuntimeTests(unittest.TestCase):
    def test_session_prepares_runtime_models_for_active_detector_and_seg_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            state_dir = Path(tmp) / "state"
            state_dir.mkdir(parents=True, exist_ok=True)
            model_path = state_dir / "yolo26n.pt"
            seg_model_path = state_dir / "yolo26n-seg.pt"
            model_path.write_text("", encoding="utf-8")
            seg_model_path.write_text("", encoding="utf-8")

            config = RuntimeConfig(
                source="camera",
                state_dir=state_dir,
                model_path=model_path,
                video_path=state_dir / "demo.mp4",
                offline_only=True,
            )
            stack = ExitStack()
            stack.enter_context(mock.patch.object(session_mod, "FrameSource", FakeFrameSource))
            stack.enter_context(mock.patch.object(session_mod, "GalleryDB", FakeGalleryDB))
            stack.enter_context(mock.patch.object(session_mod, "RecognitionWorker", FakeRecognitionWorker))
            stack.enter_context(mock.patch.object(session_mod, "YoloDetector", FakeDetector))
            stack.enter_context(mock.patch.object(session_mod, "InsightPerceptionEngine", FakePerception))
            stack.enter_context(mock.patch.object(session_mod.threading, "Thread", FakeThread))
            stack.enter_context(mock.patch.object(session_mod.JsonStateStore, "load_snapshot", return_value={}))
            stack.enter_context(mock.patch.object(session_mod.LocalInsightSession, "_probe_local_ai", return_value=False))
            stack.enter_context(mock.patch.object(session_mod, "GALLERY_DB_PATH", state_dir / "gallery.db"))
            prepared_calls: list[Path] = []

            def fake_prepare(path: Path, *, image_size: int = 640, half: bool = True, simplify: bool = True) -> Path:
                prepared_calls.append(Path(path))
                return Path(path).with_suffix(".mlpackage")

            stack.enter_context(mock.patch.object(session_mod, "ensure_runtime_model", side_effect=fake_prepare))
            self.addCleanup(stack.close)

            session = session_mod.LocalInsightSession(config)
            self.addCleanup(session.close)

            self.assertEqual(session.config.model_path.suffix, ".mlpackage")
            self.assertEqual(session.detector.model_path.suffix, ".mlpackage")
            self.assertTrue(any(path.name == "yolo26n.pt" for path in prepared_calls))
            self.assertTrue(any(path.name == "yolo26n-seg.pt" for path in prepared_calls))


class SegmentationLiveMaskTests(unittest.TestCase):
    def test_draw_overlays_refreshes_masks_from_tracker(self) -> None:
        class FakeTracker:
            def update(self, frame):
                return True, (30, 30, 80, 80)

        engine = segmentation_mod.SegmentationEngine()
        self.addCleanup(engine.stop)

        frame = np.zeros((160, 160, 3), dtype=np.uint8)
        cv2.circle(frame, (70, 70), 15, (255, 255, 255), thickness=2)
        zero_mask = np.zeros((160, 160), dtype=np.uint8)

        with engine._regions_lock:
            engine._tracked_regions = [
                {
                    "entity_id": 1,
                    "edge_mask": zero_mask.copy(),
                    "filled_mask": zero_mask.copy(),
                    "color": engine.HIGHLIGHT_COLOR,
                    "bbox": (10, 10, 60, 60),
                    "ts": time.time(),
                    "tracker": FakeTracker(),
                    "start_time": time.monotonic(),
                }
            ]

        rendered = engine.draw_overlays(frame.copy())

        with engine._regions_lock:
            refreshed = engine._tracked_regions[0]

        self.assertEqual(refreshed["bbox"], (30, 30, 110, 110))
        self.assertGreater(int(np.sum(refreshed["edge_mask"] > 0)), 0)
        self.assertGreater(int(np.sum(refreshed["filled_mask"] > 0)), 0)
        self.assertEqual(rendered.shape, frame.shape)

    def test_draw_overlays_removes_region_and_warns_when_tracking_fails(self) -> None:
        engine = segmentation_mod.SegmentationEngine()
        self.addCleanup(engine.stop)

        statuses: list[tuple[str, str]] = []
        engine.status_changed.connect(lambda msg, lvl: statuses.append((msg, lvl)))

        frame = np.zeros((120, 120, 3), dtype=np.uint8)
        zero_mask = np.zeros((120, 120), dtype=np.uint8)

        with engine._regions_lock:
            engine._tracked_regions = [
                {
                    "entity_id": 1,
                    "edge_mask": zero_mask.copy(),
                    "filled_mask": zero_mask.copy(),
                    "color": engine.HIGHLIGHT_COLOR,
                    "bbox": (10, 10, 60, 60),
                    "ts": time.time(),
                    "tracker": None,
                    "start_time": time.monotonic(),
                }
            ]

        rendered = engine.draw_overlays(frame.copy())

        with engine._regions_lock:
            self.assertEqual(engine._tracked_regions, [])
        self.assertEqual(rendered.shape, frame.shape)
        self.assertIn(("[SEG] Tracking failed", "warn"), statuses)


class PrivacyStatusTests(unittest.TestCase):
    def test_detects_icloud_risk_when_storage_path_is_inside_clouddocs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            cloud_root = home / "Library" / "Mobile Documents" / "com~apple~CloudDocs"
            cloud_root.mkdir(parents=True, exist_ok=True)
            status = detect_privacy_status(
                {
                    "Gallery DB": cloud_root / "Insight" / "gallery.db",
                    "State Folder": home / "Projects" / "Insight" / "state",
                },
                home=home,
            )

        self.assertFalse(status.protected)
        icloud = next(provider for provider in status.providers if provider.name == "iCloud Drive")
        self.assertTrue(icloud.detected)
        self.assertEqual(icloud.matched_paths, ("Gallery DB",))

    def test_reports_protected_when_paths_are_outside_cloud_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            (home / "Library" / "CloudStorage" / "OneDrive-Personal").mkdir(parents=True, exist_ok=True)
            status = detect_privacy_status(
                {
                    "Gallery DB": home / "Projects" / "Insight" / "gallery.db",
                    "State Folder": home / "Projects" / "Insight" / "state",
                },
                home=home,
            )

        self.assertTrue(status.protected)
        onedrive = next(provider for provider in status.providers if provider.name == "Microsoft OneDrive")
        self.assertTrue(onedrive.detected)
        self.assertEqual(onedrive.matched_paths, ())


class DetectionFilteringTests(unittest.TestCase):
    def test_plural_terms_normalize(self) -> None:
        self.assertEqual(normalized_query_tokens("cats humans"), ["cat", "human"])

    def test_matches_label_or_category_alias(self) -> None:
        self.assertTrue(matches_detection_filter("cats", "cat", "animal"))
        self.assertTrue(matches_detection_filter("humans", "person", "human"))
        self.assertFalse(matches_detection_filter("dogs", "cat", "animal"))

    def test_quick_filters_match_category_groups(self) -> None:
        self.assertTrue(matches_detection_view("", {"people"}, "person", "human"))
        self.assertTrue(matches_detection_view("", {"animals"}, "cat", "animal"))
        self.assertTrue(matches_detection_view("", {"objects"}, "backpack", "inorganic"))
        self.assertFalse(matches_detection_view("", {"tech"}, "cat", "animal"))


class LocalSessionBootTests(unittest.TestCase):
    def _build_session(
        self,
        *,
        requested_source: str = "camera",
        frame_status: dict[str, bool] | None = None,
        detector_ready: bool = True,
        recognizer_ready: bool = True,
        integrity_ok: bool = True,
        local_ai: bool = False,
        snapshot: dict | None = None,
        patch_threads: bool = True,
    ):
        FakeFrameSource.source_status = frame_status or {"camera": True, "video": True}
        FakeDetector.ready = detector_ready
        FakeDetector.reload_success = detector_ready
        FakeEmbedder.ready = recognizer_ready
        FakeGalleryDB.integrity_ok = integrity_ok
        FakeGalleryDB.integrity_message = "ok" if integrity_ok else "corrupt"
        FakeGalleryDB.quarantines = []
        FakeGalleryDB.face_backend_ready = recognizer_ready
        tempdir = tempfile.TemporaryDirectory()
        state_dir = Path(tempdir.name) / "state"
        config = RuntimeConfig(
            source=requested_source,
            state_dir=state_dir,
            model_path=state_dir / "model.pt",
            video_path=state_dir / "demo.mp4",
            offline_only=True,
        )
        stack = ExitStack()
        stack.enter_context(mock.patch.object(session_mod, "FrameSource", FakeFrameSource))
        stack.enter_context(mock.patch.object(session_mod, "GalleryDB", FakeGalleryDB))
        stack.enter_context(mock.patch.object(session_mod, "RecognitionWorker", FakeRecognitionWorker))
        stack.enter_context(mock.patch.object(session_mod, "YoloDetector", FakeDetector))
        stack.enter_context(mock.patch.object(session_mod, "InsightPerceptionEngine", FakePerception))
        if patch_threads:
            stack.enter_context(mock.patch.object(session_mod.threading, "Thread", FakeThread))
        stack.enter_context(mock.patch.object(session_mod.JsonStateStore, "load_snapshot", return_value=snapshot or {}))
        stack.enter_context(mock.patch.object(session_mod.LocalInsightSession, "_probe_local_ai", return_value=local_ai))
        stack.enter_context(mock.patch.object(session_mod, "GALLERY_DB_PATH", state_dir / "gallery.db"))
        stack.enter_context(
            mock.patch.object(session_mod, "ensure_runtime_model", side_effect=lambda path, **kwargs: Path(path))
        )
        self.addCleanup(stack.close)
        self.addCleanup(tempdir.cleanup)
        return session_mod.LocalInsightSession(config)

    def test_missing_model_enters_degraded_manual(self) -> None:
        session = self._build_session(detector_ready=False, frame_status={"camera": True, "video": True})
        self.assertEqual(session.operating_mode, "degraded_manual")
        self.assertFalse(session.capabilities.detector)
        self.assertTrue(session.capabilities.frame_source)
        session.close()

    def test_camera_falls_back_to_video(self) -> None:
        session = self._build_session(frame_status={"camera": False, "video": True}, detector_ready=True, recognizer_ready=True)
        self.assertEqual(session.frame_source.current_source, "video")
        self.assertTrue(session.capabilities.frame_source)
        self.assertIn(session.operating_mode, {"full", "degraded_cv"})
        session.close()

    def test_both_sources_unavailable_enters_safe_idle(self) -> None:
        session = self._build_session(frame_status={"camera": False, "video": False}, detector_ready=True)
        self.assertEqual(session.operating_mode, "safe_idle")
        self.assertFalse(session.capabilities.frame_source)
        session.close()

    def test_recognizer_unavailable_enters_degraded_cv(self) -> None:
        session = self._build_session(recognizer_ready=False, detector_ready=True, frame_status={"camera": True, "video": True})
        self.assertEqual(session.operating_mode, "degraded_cv")
        self.assertFalse(session.capabilities.recognizer)
        session.close()

    def test_corrupt_gallery_is_quarantined(self) -> None:
        session = self._build_session(integrity_ok=False, detector_ready=True, frame_status={"camera": True, "video": True})
        self.assertTrue(FakeGalleryDB.quarantines)
        self.assertTrue(any(event["action"] == "gallery_quarantine" for event in session._recovery_log))
        session.close()

    def test_snapshot_restores_last_source(self) -> None:
        session = self._build_session(
            requested_source="camera",
            frame_status={"camera": False, "video": True},
            detector_ready=True,
            snapshot={"last_source": "video"},
        )
        self.assertEqual(session.config.source, "video")
        self.assertEqual(session.frame_source.current_source, "video")
        session.close()


class PerceptionDegradationTests(unittest.TestCase):
    def test_detector_fault_latches_after_threshold(self) -> None:
        class FailingDetector:
            def __init__(self) -> None:
                self._last_error = "boom"

            @property
            def is_ready(self) -> bool:
                return True

            @property
            def last_error(self) -> str:
                return self._last_error

            def ensure_ready(self) -> bool:
                return True

            def reload(self) -> bool:
                return False

            def predict(self, frame, *, image_size: int, confidence: float, iou: float, max_det: int):
                raise RuntimeError("boom")

        events: list[tuple[str, str]] = []
        states: list[tuple[str, str]] = []
        config = RuntimeConfig(detector_error_threshold=3, fault_latch_threshold=10)
        engine = InsightPerceptionEngine(
            config=config,
            broadcaster=lambda payload: None,
            source_label_getter=lambda: "camera:0",
            detector=FailingDetector(),
            recognition_worker=None,
            detector_state_callback=lambda state, detail: states.append((state, detail)),
            detector_recovery_callback=lambda action, detail: events.append((action, detail)),
        )
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for _ in range(10):
            engine._process_frame(frame)
        self.assertTrue(engine.detector_latched)
        self.assertIn(("detector_latched", "boom"), events)
        self.assertTrue(any(state == "failed" for state, _ in states))


class LocalSessionActionDispatchTests(LocalSessionBootTests):
    def test_prepare_switch_waits_for_explicit_confirmation(self) -> None:
        session = self._build_session(patch_threads=False)
        emitted: list[dict[str, object]] = []
        session.hud_payload.connect(lambda payload: emitted.append(payload))

        session._prepare_switch_worker("video")

        self.assertEqual(session.frame_source.current_source, "camera")
        self.assertEqual(session.frame_source.prepared_source, "video")
        self.assertTrue(any(payload.get("stage") == "prepared" for payload in emitted))
        self.assertFalse(any(payload.get("stage") == "committing" for payload in emitted))
        self.assertFalse(any(payload.get("stage") == "ready" for payload in emitted))

        session.confirm_prepared_switch("video")

        self.assertEqual(session.frame_source.current_source, "video")
        self.assertIsNone(session.frame_source.prepared_source)
        self.assertTrue(any(payload.get("stage") == "ready" for payload in emitted))
        session.close()

    def test_repeated_ask_ai_roi_is_single_flight(self) -> None:
        session = self._build_session(patch_threads=False)
        started = threading.Event()
        release = threading.Event()
        calls: list[str] = []

        def blocking(*args, **kwargs) -> None:
            calls.append("run")
            started.set()
            release.wait(timeout=1.0)

        with mock.patch.object(session, "_run_roi_ai_analysis", side_effect=blocking):
            session.handle_client_message({"type": "ask_ai_roi", "provider": "auto", "prompt": "check"})
            self.assertTrue(started.wait(0.5))
            session.handle_client_message({"type": "ask_ai_roi", "provider": "auto", "prompt": "check"})
            time.sleep(0.05)
            self.assertEqual(calls, ["run"])
            self.assertIn(("ROI AI already running", "info"), session.perception.status_messages)
            release.set()
        session.close()

    def test_repeated_switch_source_is_single_flight(self) -> None:
        session = self._build_session(patch_threads=False)
        started = threading.Event()
        release = threading.Event()
        calls: list[str | None] = []

        def blocking(requested) -> None:
            calls.append(requested)
            started.set()
            release.wait(timeout=1.0)

        with mock.patch.object(session, "_prepare_switch_worker", side_effect=blocking):
            session.handle_client_message({"type": "switch_source", "source": "video"})
            self.assertTrue(started.wait(0.5))
            session.handle_client_message({"type": "switch_source", "source": "camera"})
            time.sleep(0.05)
            self.assertEqual(calls, ["video"])
            self.assertIn(("Source switch already in progress", "info"), session.perception.status_messages)
            release.set()
        session.close()

    def test_gallery_write_actions_are_serialized(self) -> None:
        session = self._build_session(patch_threads=False)
        started = threading.Event()
        release = threading.Event()
        ingest_calls: list[tuple[str, str]] = []
        rebuild_calls: list[str] = []

        def blocking_ingest(folder: str, identity: str, group: str) -> None:
            ingest_calls.append((folder, identity))
            started.set()
            release.wait(timeout=1.0)

        def rebuild() -> None:
            rebuild_calls.append("run")

        with mock.patch.object(session, "_run_ingest_folder", side_effect=blocking_ingest), mock.patch.object(
            session,
            "_run_rebuild_index",
            side_effect=rebuild,
        ):
            session.handle_client_message(
                {
                    "type": "ingest_gallery_folder",
                    "folder": "/tmp/faces",
                    "identity": "alice",
                    "group": "ops",
                }
            )
            self.assertTrue(started.wait(0.5))
            session.handle_client_message({"type": "rebuild_gallery_index"})
            time.sleep(0.05)
            self.assertEqual(ingest_calls, [("/tmp/faces", "alice")])
            self.assertEqual(rebuild_calls, [])
            self.assertIn(("Gallery operation already running", "info"), session.perception.status_messages)
            release.set()
        session.close()

    def test_manual_recognize_enqueues_without_ad_hoc_thread(self) -> None:
        session = self._build_session(patch_threads=False)
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        image_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")

        session.handle_client_message(
            {
                "type": "recognize_entry",
                "entry_id": 7,
                "track_id": 11,
                "image_b64": image_b64,
            }
        )

        assert session.recognition_worker is not None
        self.assertEqual(len(session.recognition_worker.enqueues), 1)
        self.assertEqual(session.recognition_worker.enqueues[0]["source"], "manual")
        self.assertEqual(session.recognition_worker.enqueues[0]["track_id"], 11)
        session.close()

    def test_unified_similarity_folder_ingest_uses_gallery_write(self) -> None:
        session = self._build_session(patch_threads=False)
        started = threading.Event()
        release = threading.Event()
        calls: list[tuple[str, str]] = []

        def blocking(folder: str) -> None:
            calls.append(("folder", folder))
            started.set()
            release.wait(timeout=1.0)

        with mock.patch.object(session, "_run_ingest_similarity_folder", side_effect=blocking):
            session.handle_client_message(
                {
                    "type": "ingest_gallery_media",
                    "mode": "similarity",
                    "source_kind": "folder",
                    "path": "/tmp/library",
                }
            )
            self.assertTrue(started.wait(0.5))
            session.handle_client_message({"type": "rebuild_gallery_index"})
            time.sleep(0.05)
            self.assertEqual(calls, [("folder", "/tmp/library")])
            self.assertIn(("Gallery operation already running", "info"), session.perception.status_messages)
            release.set()
        session.close()

    def test_find_similar_gallery_item_broadcasts_results(self) -> None:
        session = self._build_session(patch_threads=False)
        session.gallery.similarity_item_paths[5] = "/tmp/query.jpg"

        emitted: list[dict[str, object]] = []
        session.hud_payload.connect(lambda payload: emitted.append(payload))
        session.handle_client_message({"type": "find_similar_gallery_item", "item_id": 5})

        result = next(payload for payload in emitted if payload.get("type") == "similarity_search_result")
        self.assertEqual(result["item_id"], 5)
        self.assertEqual(len(result["results"]), 1)
        session.close()

    def test_close_shuts_down_action_runner(self) -> None:
        session = self._build_session(patch_threads=False)
        with mock.patch.object(session._action_runner, "shutdown", wraps=session._action_runner.shutdown) as shutdown:
            session.close()
        self.assertTrue(shutdown.called)


class GalleryDBSimilarityTests(unittest.TestCase):
    class _FakeSimilarityEmbedder:
        def __init__(self) -> None:
            self.load_error = ""

        def ensure_loaded(self) -> bool:
            return True

        def embed(self, bgr):
            h, w = bgr.shape[:2]
            vec = np.array([float(h), float(w), float(np.mean(bgr))], dtype=np.float32)
            norm = float(np.linalg.norm(vec))
            return vec / norm if norm > 0 else None

    def test_similarity_schema_ingest_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = GalleryDB(tmp_path / "gallery.db", embedder=self._FakeSimilarityEmbedder())
            image_a = np.full((16, 16, 3), 30, dtype=np.uint8)
            image_b = np.full((16, 16, 3), 32, dtype=np.uint8)
            image_c = np.full((32, 16, 3), 220, dtype=np.uint8)
            path_a = tmp_path / "batch" / "a.jpg"
            path_b = tmp_path / "batch" / "b.jpg"
            path_c = tmp_path / "other" / "c.jpg"
            path_a.parent.mkdir(parents=True, exist_ok=True)
            path_c.parent.mkdir(parents=True, exist_ok=True)
            self.assertTrue(cv2.imwrite(str(path_a), image_a))
            self.assertTrue(cv2.imwrite(str(path_b), image_b))
            self.assertTrue(cv2.imwrite(str(path_c), image_c))

            ok_a, err_a = db.ingest_similarity_image(path_a)
            ok_b, err_b = db.ingest_similarity_image(path_b)
            ok_c, err_c = db.ingest_similarity_image(path_c)
            self.assertTrue(ok_a, err_a)
            self.assertTrue(ok_b, err_b)
            self.assertTrue(ok_c, err_c)

            db.build_matrix()
            stats = db.get_stats()
            self.assertEqual(stats.similarity_item_count, 3)
            items = db.list_similarity_items()
            self.assertEqual(len(items), 3)
            self.assertTrue(items[0].thumb_png is not None)

            query_item = next(item for item in items if item.source_path == str(path_a))
            results = db.find_similar_items(query_item.item_id, top_k=2)
            self.assertEqual(len(results), 2)
            self.assertNotEqual(results[0]["item_id"], query_item.item_id)
            self.assertEqual(results[0]["source_path"], str(path_b))

            removed = db.delete_similarity_item(query_item.item_id)
            self.assertEqual(removed, 1)
            db.build_matrix()
            self.assertEqual(db.get_stats().similarity_item_count, 2)
            db.close()

    def test_similarity_folder_ingest_is_recursive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            folder = tmp_path / "collection"
            nested = folder / "nested"
            nested.mkdir(parents=True, exist_ok=True)
            self.assertTrue(cv2.imwrite(str(folder / "one.jpg"), np.full((8, 8, 3), 10, dtype=np.uint8)))
            self.assertTrue(cv2.imwrite(str(nested / "two.png"), np.full((8, 8, 3), 12, dtype=np.uint8)))
            db = GalleryDB(tmp_path / "gallery.db", embedder=self._FakeSimilarityEmbedder())

            added, errors = db.ingest_similarity_folder(folder)

            self.assertEqual(added, 2)
            self.assertEqual(errors, [])
            db.close()


if __name__ == "__main__":
    unittest.main()
