from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.service import CvOpsService
from mlops.pipeline import ci_cd
from mlops.pipeline import model_registry
from mlops.pipeline import registry as mlops_registry


@contextmanager
def _patched_mlops_roots(root: Path) -> Iterator[None]:
    mlops_root = root / "mlops"
    database_root = root / "database"
    model_root = root / "assets" / "models"
    for path in (mlops_root / "scenarios", mlops_root / "datasets", database_root, model_root):
        path.mkdir(parents=True, exist_ok=True)
    (mlops_root / "registry.json").write_text('{"version": 1, "scenarios": []}', encoding="utf-8")
    (model_root / "base.pt").write_bytes(b"base model")

    old_values = {
        "repo": mlops_registry.REPO_ROOT,
        "mlops": mlops_registry.MLOPS_ROOT,
        "database": mlops_registry.DATABASE_ROOT,
        "audio": mlops_registry.ML_AUDIO_ROOT,
        "tabular": mlops_registry.TABULAR_DATASETS_ROOT,
        "registry": mlops_registry.REGISTRY_PATH,
        "search_roots": list(mlops_registry.MODEL_SEARCH_ROOTS),
        "model_registry": model_registry.MODEL_REGISTRY_PATH,
    }
    try:
        mlops_registry.REPO_ROOT = root
        mlops_registry.MLOPS_ROOT = mlops_root
        mlops_registry.DATABASE_ROOT = database_root
        mlops_registry.ML_AUDIO_ROOT = root / "assets" / "ml_audio"
        mlops_registry.TABULAR_DATASETS_ROOT = mlops_root / "datasets"
        mlops_registry.REGISTRY_PATH = mlops_root / "registry.json"
        mlops_registry.MODEL_SEARCH_ROOTS = [model_root]
        model_registry.MODEL_REGISTRY_PATH = mlops_root / "model_registry.json"
        try:
            with mlops_registry._SCENARIO_STATUS_CACHE_LOCK:  # type: ignore[attr-defined]
                mlops_registry._SCENARIO_STATUS_CACHE.clear()  # type: ignore[attr-defined]
        except Exception:
            pass
        yield
    finally:
        mlops_registry.REPO_ROOT = old_values["repo"]
        mlops_registry.MLOPS_ROOT = old_values["mlops"]
        mlops_registry.DATABASE_ROOT = old_values["database"]
        mlops_registry.ML_AUDIO_ROOT = old_values["audio"]
        mlops_registry.TABULAR_DATASETS_ROOT = old_values["tabular"]
        mlops_registry.REGISTRY_PATH = old_values["registry"]
        mlops_registry.MODEL_SEARCH_ROOTS = old_values["search_roots"]
        model_registry.MODEL_REGISTRY_PATH = old_values["model_registry"]
        try:
            with mlops_registry._SCENARIO_STATUS_CACHE_LOCK:  # type: ignore[attr-defined]
                mlops_registry._SCENARIO_STATUS_CACHE.clear()  # type: ignore[attr-defined]
        except Exception:
            pass


def _write_yolo_dataset(root: Path, name: str = "Tiny") -> None:
    image = root / "database" / name / "train" / "images" / "sample.jpg"
    label = root / "database" / name / "train" / "labels" / "sample.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"fake-image")
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")


def _create_demo_scenario(root: Path) -> None:
    _write_yolo_dataset(root)
    mlops_registry.create_scenario_profile(
        name="demo",
        display_name="Demo",
        description="CI/CD demo",
        base_model="base.pt",
        dataset="Tiny",
        classes=["thing"],
        hyperparams={"epochs": 1, "imgsz": 64, "quality_stop_threshold": 0.90},
    )


def _write_run(root: Path, scenario: str, version: str, metric: float) -> Path:
    run_dir = root / "mlops" / "models" / scenario / version
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = run_dir / "weights.pt"
    weights.write_bytes(b"candidate weights")
    snapshot = run_dir / "dataset_snapshot.json"
    snapshot.write_text("{}", encoding="utf-8")
    repro = run_dir / "repro_manifest.json"
    repro.write_text("{}", encoding="utf-8")
    metrics = {
        "scenario": scenario,
        "dataset": "Tiny",
        "map50": metric,
        "map50_95": metric,
        "weights": str(weights),
        "dataset_snapshot_id": f"snap-{version}",
        "dataset_snapshot_path": str(snapshot),
        "dataset_contract": {"status": "ok", "issues": []},
        "dataset_quality": {"quality_score": 100.0},
        "quality_stop": {"verdict": "viable"},
        "repro_manifest": str(repro),
        "metrics": {"map50_95": metric},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    model_registry.register_model_version(
        scenario=scenario,
        run_version=version,
        artifacts={"run_dir": str(run_dir), "weights": str(weights), "metrics_path": str(run_dir / "metrics.json")},
        lineage={"dataset_snapshot_id": f"snap-{version}"},
        metrics={"map50_95": metric, "map50": metric},
        ci_cd={"gate_status": "pending"},
        set_candidate=True,
    )
    return run_dir


class MlopsCiCdTests(unittest.TestCase):
    def test_new_scenario_policy_defaults_and_patch_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)

                cfg = mlops_registry.get_scenario_config("demo")
                self.assertTrue(cfg.raw["ci_cd"]["enabled"])
                policy = mlops_registry.get_scenario_ci_cd_policy("demo")
                self.assertTrue(policy["enabled"])
                self.assertEqual(policy["metric"], "map50_95")
                self.assertEqual(policy["promotion"], "manual")

                status = mlops_registry.patch_scenario_ci_cd_policy(
                    "demo",
                    {"threshold": 0.75, "promotion": "auto"},
                )
                self.assertEqual(status["ci_cd"]["threshold"], 0.75)
                self.assertEqual(status["ci_cd"]["promotion"], "auto")

                cfg_path = mlops_registry.get_scenario_config("demo").config_path
                raw = cfg_path.read_text(encoding="utf-8")
                self.assertIn("ci_cd:", raw)
                self.assertIn("promotion: auto", raw)

    def test_gate_pass_and_regression_fail_update_model_registry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)
                _write_run(root, "demo", "v1", 0.98)
                model_registry.set_alias("demo", "prod", "demo:v1")
                _write_run(root, "demo", "v2", 0.97)

                passed = ci_cd.evaluate_run_gate("demo", "v2", update_registry=True)
                self.assertEqual(passed["gate_status"], "passed")
                self.assertTrue((root / "mlops" / "models" / "demo" / "v2" / "ci_cd_report.json").is_file())
                entry = model_registry.get_model_version("demo", "demo:v2")
                self.assertEqual((entry or {}).get("ci_cd", {}).get("gate_status"), "passed")

                _write_run(root, "demo", "v3", 0.95)
                failed = ci_cd.evaluate_run_gate("demo", "v3", update_registry=True)
                self.assertEqual(failed["gate_status"], "failed")
                self.assertTrue(any("regressed" in str(item) for item in failed["failures"]))

    def test_promotion_requires_passed_gate_and_sets_prod_alias(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)
                run_dir = _write_run(root, "demo", "v2", 0.95)
                report = ci_cd.evaluate_run_gate("demo", "v2", update_registry=True)
                self.assertEqual(report["gate_status"], "passed")

                result = ci_cd.promote_run("demo", "v2", actor="test", reason="ship it")
                self.assertTrue(result["ok"])
                prod = model_registry.resolve_alias("demo", "prod")
                self.assertEqual((prod or {}).get("version_id"), "demo:v2")
                target = mlops_registry.get_scenario_config("demo").weights_path
                self.assertTrue(target.is_file())
                self.assertEqual(target.read_bytes(), (run_dir / "weights.pt").read_bytes())

    def test_staging_promotion_does_not_overwrite_live_weights(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)
                _write_run(root, "demo", "v2", 0.95)
                ci_cd.evaluate_run_gate("demo", "v2", update_registry=True)

                live = mlops_registry.get_scenario_config("demo").weights_path
                self.assertFalse(live.is_file())  # not yet promoted to prod

                result = ci_cd.promote_run("demo", "v2", target_alias="staging")
                self.assertEqual(result["alias"], "staging")
                staging = model_registry.resolve_alias("demo", "staging")
                self.assertEqual((staging or {}).get("version_id"), "demo:v2")
                # Staging is a challenger tier: live serving weights untouched.
                self.assertFalse(live.is_file())
                self.assertIsNone(model_registry.resolve_alias("demo", "prod"))

    def test_revert_alias_returns_to_prior_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)
                _write_run(root, "demo", "v1", 0.93)
                _write_run(root, "demo", "v2", 0.96)
                ci_cd.evaluate_run_gate("demo", "v1", update_registry=True)
                ci_cd.evaluate_run_gate("demo", "v2", update_registry=True)

                ci_cd.promote_run("demo", "v1", target_alias="prod")
                ci_cd.promote_run("demo", "v2", target_alias="prod")
                self.assertEqual(
                    (model_registry.resolve_alias("demo", "prod") or {}).get("version_id"),
                    "demo:v2",
                )
                self.assertEqual(
                    model_registry.alias_history("demo", "prod"), ["demo:v1", "demo:v2"]
                )

                reverted = model_registry.revert_alias("demo", "prod", actor="test")
                self.assertTrue(reverted["reverted"])
                self.assertEqual(reverted["version_id"], "demo:v1")
                self.assertEqual(reverted["from_version_id"], "demo:v2")
                self.assertEqual(
                    (model_registry.resolve_alias("demo", "prod") or {}).get("version_id"),
                    "demo:v1",
                )

    def test_revert_alias_no_history_is_safe_noop(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)
                _write_run(root, "demo", "v1", 0.93)
                ci_cd.evaluate_run_gate("demo", "v1", update_registry=True)
                ci_cd.promote_run("demo", "v1", target_alias="prod")

                reverted = model_registry.revert_alias("demo", "prod")
                self.assertFalse(reverted["reverted"])
                # Prod pointer is left untouched when there is nothing prior.
                self.assertEqual(
                    (model_registry.resolve_alias("demo", "prod") or {}).get("version_id"),
                    "demo:v1",
                )

    def test_service_aliases_and_revert_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)
                _write_run(root, "demo", "v1", 0.93)
                _write_run(root, "demo", "v2", 0.96)
                svc = CvOpsService(
                    db_path=root / "jobs.db",
                    catalog_db_path=root / "catalog.db",
                    catalog_assets_root=root / "catalog_assets",
                )
                with TestClient(svc.app) as client:
                    client.post("/scenarios/demo/runs/v1/gate")
                    client.post("/scenarios/demo/runs/v2/gate")
                    client.post("/scenarios/demo/runs/v1/promote", json={"target_alias": "prod"})
                    client.post("/scenarios/demo/runs/v2/promote", json={"target_alias": "prod"})

                    aliases = client.get("/scenarios/demo/aliases")
                    self.assertEqual(aliases.status_code, 200, aliases.text)
                    body = aliases.json()["aliases"]
                    self.assertEqual(body["prod"]["version_id"], "demo:v2")
                    self.assertEqual(body["prod"]["history"], ["demo:v1", "demo:v2"])

                    reverted = client.post(
                        "/scenarios/demo/aliases/prod/revert", json={"actor": "test"}
                    )
                    self.assertEqual(reverted.status_code, 200, reverted.text)
                    self.assertTrue(reverted.json()["reverted"])
                    self.assertEqual(reverted.json()["version_id"], "demo:v1")

                    after = client.get("/scenarios/demo/aliases").json()["aliases"]
                    self.assertEqual(after["prod"]["version_id"], "demo:v1")

    def test_service_pipeline_gate_and_promote_endpoints(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)
                _write_run(root, "demo", "v2", 0.96)
                svc = CvOpsService(
                    db_path=root / "jobs.db",
                    catalog_db_path=root / "catalog.db",
                    catalog_assets_root=root / "catalog_assets",
                )
                with TestClient(svc.app) as client:
                    pipeline = client.get("/scenarios/demo/pipeline")
                    self.assertEqual(pipeline.status_code, 200, pipeline.text)
                    self.assertTrue(pipeline.json()["ci_cd"]["enabled"])

                    patch = client.patch("/scenarios/demo/pipeline", json={"updates": {"threshold": 0.90}})
                    self.assertEqual(patch.status_code, 200, patch.text)
                    self.assertEqual(patch.json()["ci_cd"]["threshold"], 0.90)

                    gate = client.post("/scenarios/demo/runs/v2/gate")
                    self.assertEqual(gate.status_code, 200, gate.text)
                    self.assertEqual(gate.json()["gate_status"], "passed")

                    fetched = client.get("/scenarios/demo/runs/v2/gate")
                    self.assertEqual(fetched.status_code, 200, fetched.text)
                    self.assertEqual(fetched.json()["gate_status"], "passed")

                    promoted = client.post(
                        "/scenarios/demo/runs/v2/promote",
                        json={"actor": "test", "reason": "service"},
                    )
                    self.assertEqual(promoted.status_code, 200, promoted.text)
                    self.assertEqual(promoted.json()["version_id"], "demo:v2")

    def test_service_train_completion_writes_gate_result(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with _patched_mlops_roots(root):
                _create_demo_scenario(root)
                run_dir = _write_run(root, "demo", "v2", 0.96)
                svc = CvOpsService(
                    db_path=root / "jobs.db",
                    catalog_db_path=root / "catalog.db",
                    catalog_assets_root=root / "catalog_assets",
                )
                job = svc.store.create_job(
                    job_id="job-ci",
                    scenario="demo",
                    job_type="train",
                    source="test",
                    image_path="",
                    payload={},
                )

                original = svc._run_train_job
                try:
                    svc._run_train_job = lambda _job: {  # type: ignore[assignment]
                        "job_id": _job.job_id,
                        "scenario": "demo",
                        "run_version": "v2",
                        "result_path": str(run_dir),
                        "weights": str(run_dir / "weights.pt"),
                        "error": "",
                    }
                    svc._execute_job(job)
                finally:
                    svc._run_train_job = original  # type: ignore[assignment]

                stored = svc.store.get_result("job-ci")
                self.assertIsNotNone(stored)
                self.assertEqual((stored or {}).get("ci_cd", {}).get("gate_status"), "passed")
                self.assertTrue((run_dir / "ci_cd_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
