from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.service import CvOpsService
from mlops.pipeline import integration as mlops_integration
from mlops.pipeline import registry as mlops_registry
from mlops.pipeline.backbone import BackboneContext
from mlops.pipeline.backbones import get_backbone
from mlops.pipeline.backbones import llm_fine_tuning as llm_backbone


class _RegistryPatch:
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.mlops_root = repo_root / "mlops"
        self.database_root = repo_root / "database"
        self.old_repo = mlops_registry.REPO_ROOT
        self.old_mlops = mlops_registry.MLOPS_ROOT
        self.old_db = mlops_registry.DATABASE_ROOT
        self.old_audio = mlops_registry.ML_AUDIO_ROOT
        self.old_tabular = mlops_registry.TABULAR_DATASETS_ROOT
        self.old_registry = mlops_registry.REGISTRY_PATH
        self.old_events = mlops_integration.EVENTS_PATH

    def __enter__(self) -> None:
        (self.mlops_root / "scenarios").mkdir(parents=True, exist_ok=True)
        (self.mlops_root / "datasets").mkdir(parents=True, exist_ok=True)
        self.database_root.mkdir(parents=True, exist_ok=True)
        (self.mlops_root / "registry.json").write_text('{"version": 1, "scenarios": []}', encoding="utf-8")
        mlops_registry.REPO_ROOT = self.repo_root
        mlops_registry.MLOPS_ROOT = self.mlops_root
        mlops_registry.DATABASE_ROOT = self.database_root
        mlops_registry.ML_AUDIO_ROOT = self.repo_root / "assets" / "ml_audio"
        mlops_registry.TABULAR_DATASETS_ROOT = self.mlops_root / "datasets"
        mlops_registry.REGISTRY_PATH = self.mlops_root / "registry.json"
        mlops_integration.EVENTS_PATH = self.mlops_root / "integration" / "events.jsonl"

    def __exit__(self, *_exc: object) -> None:
        mlops_registry.REPO_ROOT = self.old_repo
        mlops_registry.MLOPS_ROOT = self.old_mlops
        mlops_registry.DATABASE_ROOT = self.old_db
        mlops_registry.ML_AUDIO_ROOT = self.old_audio
        mlops_registry.TABULAR_DATASETS_ROOT = self.old_tabular
        mlops_registry.REGISTRY_PATH = self.old_registry
        mlops_integration.EVENTS_PATH = self.old_events


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


class CvOpsLlmFineTuningTests(unittest.TestCase):
    def test_jsonl_shapes_and_feedback_are_prepared(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            with _RegistryPatch(repo):
                data_path = repo / "mlops" / "datasets" / "instructions.jsonl"
                _write_jsonl(
                    data_path,
                    [
                        {"messages": [{"role": "user", "content": "A?"}, {"role": "assistant", "content": "B."}]},
                        {"prompt": "Summarize", "response": "Short summary."},
                        {"instruction": "Classify", "input": "alarm", "output": "urgent"},
                    ],
                )
                feedback_path = repo / "feedback.jsonl"
                _write_jsonl(
                    feedback_path,
                    [
                        {
                            "scenario": "detector",
                            "issue_type": "miss",
                            "severity": "medium",
                            "notes": "Missed reflective object.",
                            "recommendation": "Ask for reflective material context.",
                        }
                    ],
                )
                mlops_registry.create_scenario_profile(
                    name="llm_demo",
                    display_name="LLM Demo",
                    description="test",
                    dataset="mlops/datasets/instructions.jsonl",
                    backbone_type="llm_fine_tuning",
                    backbone_config={
                        "base_model": "local/tiny",
                        "ollama_base_model": "llama3.2",
                        "feedback_path": str(feedback_path),
                        "sources": ["jsonl", "feedback"],
                        "dry_run": True,
                    },
                )
                cfg = mlops_registry.get_scenario_config("llm_demo")
                examples, manifest = llm_backbone.load_training_examples(cfg)
                self.assertEqual(len(examples), 4)
                self.assertEqual(sum(int(f["accepted"]) for f in manifest["jsonl_files"]), 3)
                self.assertEqual(sum(int(f["accepted"]) for f in manifest["feedback_files"]), 1)

    def test_registry_creation_and_dry_run_backbone_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td)
            with _RegistryPatch(repo):
                data_path = repo / "mlops" / "datasets" / "instructions.jsonl"
                _write_jsonl(data_path, [{"prompt": "Hello", "response": "Hi"}])

                status = mlops_registry.create_scenario_profile(
                    name="llm_demo",
                    display_name="LLM Demo",
                    description="test",
                    dataset="mlops/datasets/instructions.jsonl",
                    backbone_type="llm_fine_tuning",
                    backbone_config={
                        "base_model": "local/tiny",
                        "ollama_base_model": "llama3.2",
                        "sources": ["jsonl"],
                        "dry_run": True,
                    },
                )
                self.assertEqual(status["backbone_type"], "llm_fine_tuning")
                self.assertEqual(status["dataset_count"], 1)

                cfg = mlops_registry.get_scenario_config("llm_demo")
                result = get_backbone(cfg.backbone_type, cfg).run(
                    BackboneContext(
                        scenario_config=cfg,
                        job_id="job-llm-test",
                        job_type="train",
                        image_bgr=None,
                        payload={},
                        cell_callback=lambda _payload: None,
                    )
                )
                self.assertFalse(result.get("error"), result)
                run_dir = Path(str(result.get("result_path") or ""))
                self.assertTrue((run_dir / "adapter" / "adapter_model.safetensors").exists())
                self.assertTrue((run_dir / "Modelfile").exists())
                self.assertTrue((run_dir / "metrics.json").exists())

                refreshed = mlops_registry.get_scenario_status("llm_demo")
                self.assertEqual(refreshed["status"], "trained")
                history = mlops_registry.list_scenario_runs("llm_demo")
                self.assertEqual(history[0]["status"], "trained")
                self.assertTrue(str(history[0]["weights"]).endswith("adapter_model.safetensors"))

    def test_service_train_history_and_artifacts_for_dry_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            repo = tmp / "repo"
            with _RegistryPatch(repo):
                data_path = repo / "mlops" / "datasets" / "instructions.jsonl"
                _write_jsonl(data_path, [{"prompt": "Hello", "response": "Hi"}])
                mlops_registry.create_scenario_profile(
                    name="llm_service",
                    display_name="LLM Service",
                    description="test",
                    dataset="mlops/datasets/instructions.jsonl",
                    backbone_type="llm_fine_tuning",
                    backbone_config={
                        "base_model": "local/tiny",
                        "ollama_base_model": "llama3.2",
                        "sources": ["jsonl"],
                        "dry_run": True,
                    },
                )
                svc = CvOpsService(
                    db_path=tmp / "jobs.db",
                    catalog_db_path=tmp / "catalog.db",
                    snapshot_db_path=tmp / "snapshots.db",
                    lineage_db_path=tmp / "lineages.db",
                    range_db_path=tmp / "ranges.db",
                )
                with TestClient(svc.app) as client:
                    kicked = client.post("/scenarios/llm_service/train")
                    self.assertEqual(kicked.status_code, 200, kicked.text)
                    job_id = kicked.json()["job_id"]
                    final_state = ""
                    for _ in range(100):
                        job = client.get(f"/jobs/{job_id}")
                        self.assertEqual(job.status_code, 200, job.text)
                        final_state = str(job.json().get("state") or "")
                        if final_state in {"done", "error"}:
                            break
                        time.sleep(0.05)
                    self.assertEqual(final_state, "done")

                    history = client.get("/scenarios/llm_service/history")
                    self.assertEqual(history.status_code, 200, history.text)
                    runs = history.json()["runs"]
                    self.assertEqual(runs[0]["version"], "v1")
                    artifacts = client.get("/scenarios/llm_service/runs/v1/artifacts")
                    self.assertEqual(artifacts.status_code, 200, artifacts.text)
                    names = {item["name"] for item in artifacts.json()["items"]}
                    self.assertIn("adapter/adapter_model.safetensors", names)
                    self.assertIn("Modelfile", names)
                    self.assertIn("metrics.json", names)


if __name__ == "__main__":
    unittest.main()
