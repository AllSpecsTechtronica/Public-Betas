from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.service import CvOpsService
from mlops.pipeline import registry as mlops_registry


class CvOpsTrainingCancelHistoryTests(unittest.TestCase):
    def test_cancelled_yolo_train_writes_partial_history_entry(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            mlops_root = tmp / "mlops"
            models_root = mlops_root / "models" / "demo"
            run_dir = models_root / "v1"
            run_dir.mkdir(parents=True, exist_ok=True)

            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
                snapshot_db_path=tmp / "snapshots.db",
                lineage_db_path=tmp / "lineages.db",
                provenance_db_path=tmp / "provenance.db",
                range_db_path=tmp / "ranges.db",
                archive_db_path=tmp / "archive.db",
                archive_storage_root=tmp / "archive_storage",
            )
            try:
                job = svc.store.create_job(
                    job_id="job-cancel-history",
                    scenario="demo",
                    job_type="train",
                    source="test",
                    image_path="",
                    payload={"save_period": 1, "resume": True},
                )

                def _cancelled_worker(_job, _spec, on_progress):
                    on_progress(
                        {
                            "event": "log",
                            "line": "[trainer] base model: assets/models/yolo26s.pt",
                            "stream": "stdout",
                        }
                    )
                    on_progress(
                        {
                            "event": "start",
                            "epoch": -1,
                            "epochs": 20,
                            "progress": 0.0,
                            "run_dir": str(run_dir),
                            "asset_root": str(models_root),
                            "save_period": 1,
                            "dataset_snapshot_id": "snap-1",
                            "dataset_snapshot_path": str(tmp / "snapshots" / "snap-1.json"),
                        }
                    )
                    on_progress(
                        {
                            "event": "epoch",
                            "epoch": 2,
                            "epochs": 20,
                            "progress": 10.0,
                            "map50": 0.125,
                            "map50_95": 0.055,
                        }
                    )
                    raise RuntimeError("training cancelled by operator")

                cfg = SimpleNamespace(name="demo", backbone_type="yolo_detection")
                with (
                    mock.patch.object(mlops_registry, "MLOPS_ROOT", mlops_root),
                    mock.patch.object(mlops_registry, "get_scenario_config", return_value=cfg),
                    mock.patch("insight_local.cvops.service.register_model_version") as register_mock,
                    mock.patch.object(svc, "_run_yolo_train_subprocess", side_effect=_cancelled_worker),
                ):
                    register_mock.side_effect = lambda **kwargs: {
                        "version_id": f"{kwargs['scenario']}:{kwargs['run_version']}",
                        "status": kwargs.get("initial_status"),
                    }

                    result = svc._run_train_job(job)
                    history = mlops_registry.list_scenario_runs("demo")

                self.assertEqual(result["status"], "canceled")
                self.assertEqual(result["result_path"], str(run_dir))
                self.assertEqual(result["model_version_id"], "demo:v1")
                self.assertEqual(history[0]["version"], "v1")
                self.assertEqual(history[0]["status"], "canceled")
                self.assertEqual(history[0]["job_id"], "job-cancel-history")
                self.assertEqual(history[0]["error"], "training cancelled by operator")
                self.assertEqual(history[0]["map50"], 0.125)

                metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                self.assertEqual(metrics["run_status"], "canceled")
                self.assertEqual(metrics["dataset_snapshot_id"], "snap-1")
                register_mock.assert_called_once()
                self.assertEqual(register_mock.call_args.kwargs["initial_status"], "canceled")
                self.assertFalse(register_mock.call_args.kwargs["set_candidate"])
            finally:
                svc._stop.set()
                svc._executor.shutdown(wait=False, cancel_futures=True)
                svc.store.close()


if __name__ == "__main__":
    unittest.main()
