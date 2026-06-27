from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from mlops.image_to_3d import JobStatus, JobStore, Scene, default_intrinsics
from mlops.image_to_3d.capability import coreml_depth_model_path, coreml_tmp_dir
from mlops.image_to_3d.models import depth as depth_model
from mlops.image_to_3d.models.depth import DepthModel
from mlops.image_to_3d.pipeline import PipelineConfig
from mlops.image_to_3d.stages import enhance_trellis, export, geometry, pose, preprocess, texture, walkability


def _write_fake_coreml_package(path: Path) -> None:
    (path / "com.apple.CoreML" / "weights").mkdir(parents=True)
    (path / "Manifest.json").write_text("{}", encoding="utf-8")
    (path / "com.apple.CoreML" / "model.mlmodel").write_bytes(b"model")
    (path / "com.apple.CoreML" / "weights" / "weight.bin").write_bytes(b"weights")


def test_scene_provenance_round_trip(tmp_path: Path) -> None:
    scene = Scene(job_id="job_test", root=str(tmp_path))
    artifact = tmp_path / "mesh.glb"
    artifact.write_bytes(b"glTF")
    scene.add_artifact(
        artifact,
        kind="mesh",
        source="image_depth",
        confidence=0.5,
        stage_version="test@v1",
        depends_on=["depth.npy"],
    )
    scene.save()

    loaded = Scene.load(tmp_path / "provenance.json")
    assert loaded.job_id == "job_test"
    assert loaded.provenance[0].artifact == "mesh.glb"
    assert loaded.provenance[0].depends_on == ["depth.npy"]


def test_job_store_round_trip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job = store.create({"mesh_stride": 2})
    job.status = JobStatus.RUNNING
    job.stage = "depth"
    job.progress = 2.0
    store.save(job)

    loaded = store.load(job.job_id)
    assert loaded is not None
    assert loaded.status == JobStatus.RUNNING
    assert loaded.progress == 1.0
    assert store.list_recent(limit=1)[0].job_id == job.job_id


def test_default_intrinsics() -> None:
    intr = default_intrinsics(640, 480)
    assert intr.width == 640
    assert intr.height == 480
    assert intr.fx > 0
    assert intr.cx == 319.5
    assert intr.cy == 239.5


def test_geometry_stage_exports_artifacts_without_trimesh_requirement(tmp_path: Path) -> None:
    scene = Scene(job_id="job_test", root=str(tmp_path))
    input_path = tmp_path / "input.png"
    Image.new("RGB", (32, 24), (100, 80, 60)).save(input_path)
    preprocessed = preprocess.run(scene, input_path, max_size=32)
    intrinsics = pose.run(scene, preprocessed)
    depth_path = tmp_path / "depth.npy"
    np.save(depth_path, np.linspace(0, 1, 32 * 24, dtype="float32").reshape(24, 32))

    points, mesh = geometry.run(scene, preprocessed, depth_path, intrinsics, mesh_stride=4)
    texture_path = texture.run(scene, preprocessed)
    floor_path = walkability.run(scene, depth_path)
    scene_path = export.run(scene, mesh)

    assert points.stat().st_size > 0
    assert mesh.stat().st_size > 0
    assert texture_path.exists()
    assert floor_path.exists()
    assert scene_path.exists()


def test_pipeline_config_progress_clamp_and_trellis_failure_record(tmp_path: Path) -> None:
    config = PipelineConfig.from_dict({"mesh_stride": 3, "depth_backend": "coreml", "unknown": "ignored"})
    assert config.mesh_stride == 3
    assert config.depth_backend == "coreml"

    scene = Scene(job_id="job_test", root=str(tmp_path))
    enhance_trellis.record_failure(scene, "network down")
    assert scene.provenance[0].status == "failed"
    assert scene.provenance[0].source == "trellis"


def test_coreml_depth_model_path_env_override(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "DepthAnythingSmallF16.mlpackage"
    monkeypatch.setenv("IMAGE_TO_3D_DEPTH_MODEL_PATH", str(model_path))
    assert coreml_depth_model_path() == model_path.resolve()


def test_coreml_tmp_dir_env_override(tmp_path: Path, monkeypatch) -> None:
    tmp_dir = tmp_path / "coreml_tmp"
    monkeypatch.setenv("IMAGE_TO_3D_COREML_TMPDIR", str(tmp_dir))
    assert coreml_tmp_dir() == tmp_dir.resolve()


def test_depth_model_auto_prefers_coreml_when_package_exists(tmp_path: Path) -> None:
    model_path = tmp_path / "DepthAnythingSmallF16.mlpackage"
    _write_fake_coreml_package(model_path)
    model = DepthModel(backend="auto", model_path=model_path)
    assert model._select_backend() == "coreml"


def test_coreml_depth_prediction_normalizes_to_original_size(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (6, 4), (120, 100, 80)).save(image_path)
    model_path = tmp_path / "DepthAnythingSmallF16.mlpackage"
    _write_fake_coreml_package(model_path)

    class FakeShape:
        shape = [1, 3, 2, 2]

    class FakeType:
        multiArrayType = FakeShape()

    class FakeInput:
        name = "image"
        type = FakeType()

    class FakeDescription:
        input = [FakeInput()]

    class FakeSpec:
        description = FakeDescription()

    class FakeCoreMLModel:
        def get_spec(self):
            return FakeSpec()

        def predict(self, _inputs):
            return {"predicted_depth": np.array([[[[0.0, 0.25], [0.5, 1.0]]]], dtype="float32")}

    monkeypatch.setattr(depth_model.DepthModel, "_load_coreml", lambda self: FakeCoreMLModel())
    depth = DepthModel(backend="coreml", model_path=model_path).predict(image_path)

    assert depth.shape == (4, 6)
    assert depth.dtype == np.float32
    assert 0.0 <= float(depth.min()) <= float(depth.max()) <= 1.0


def test_auto_depth_backend_raises_clear_error_when_coreml_prediction_fails(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "input.png"
    Image.new("RGB", (3, 2), (120, 100, 80)).save(image_path)
    model_path = tmp_path / "DepthAnythingSmallF16.mlpackage"
    _write_fake_coreml_package(model_path)

    def fail_coreml(self, _image_path):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(depth_model.DepthModel, "_predict_coreml", fail_coreml)

    model = DepthModel(backend="auto", model_path=model_path)
    try:
        model.predict(image_path)
    except RuntimeError as exc:
        msg = str(exc)
    else:
        raise AssertionError("expected RuntimeError")

    assert "CoreML depth inference failed" in msg
    assert str(model_path) in msg
