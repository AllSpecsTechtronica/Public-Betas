from __future__ import annotations

import os
import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))


class CvOpsDirectServiceReadsTests(unittest.TestCase):
    def _service(self, tmp: Path):
        from insight_local.cvops.service import CvOpsService

        return CvOpsService(
            db_path=tmp / "jobs.db",
            catalog_db_path=tmp / "catalog.db",
            catalog_assets_root=tmp / "catalog_assets",
            snapshot_db_path=tmp / "snapshots.db",
            snapshot_weights_root=tmp / "snapshot_weights",
            lineage_db_path=tmp / "lineage.db",
            provenance_db_path=tmp / "provenance.db",
            range_db_path=tmp / "ranges.db",
            archive_db_path=tmp / "archives.db",
            archive_storage_root=tmp / "archive_storage",
        )

    def test_startup_resync_payload_reads_jobs_progress_and_scenarios_in_process(self) -> None:
        from insight_local.cvops import service as service_mod

        with tempfile.TemporaryDirectory() as raw_tmp:
            svc = self._service(Path(raw_tmp))
            try:
                job = svc.store.create_job(
                    job_id="job-1",
                    scenario="demo",
                    job_type="train",
                    source="manual",
                    image_path="",
                    payload={},
                )
                svc._record_training_progress(
                    job.job_id,
                    {"event": "epoch", "epoch": 1, "progress": 0.5},
                )

                with mock.patch.object(
                    service_mod.mlops_registry,
                    "list_enabled_scenarios",
                    return_value=[{"name": "demo", "display_name": "Demo", "description": "Demo scenario"}],
                ), mock.patch.object(
                    service_mod.mlops_registry,
                    "get_scenario_status",
                    return_value={"name": "demo", "status": "ready"},
                ):
                    payload = svc.startup_resync_payload()
            finally:
                svc.store.close()
                svc.catalog.close()
                svc.archives.close()

        self.assertEqual(payload["errors"], [])
        self.assertEqual(payload["jobs"][0]["job_id"], "job-1")
        self.assertEqual(payload["scenarios"][0]["name"], "demo")
        self.assertEqual(payload["scenarios"][0]["status"], "training")
        self.assertEqual(payload["training_events"][0]["scenario"], "demo")
        self.assertEqual(payload["training_events"][0]["job_id"], "job-1")

    def test_ecosystem_read_routes_run_in_threadpool(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            svc = self._service(Path(raw_tmp))
            try:
                endpoints = {
                    getattr(route, "path", ""): getattr(route, "endpoint", None)
                    for route in svc.app.routes
                }
            finally:
                svc.store.close()
                svc.catalog.close()
                svc.archives.close()

        for path in (
            "/diagnostics/summary",
            "/ecosystem/summary",
            "/ecosystem/graph_view",
            "/ontology/graph",
            "/ontology/entity/{entity_type}/{entity_id:path}",
            "/ecosystem/impact/{entity_id:path}",
            "/ecosystem/path",
            "/ecosystem/orphans",
        ):
            self.assertIn(path, endpoints)
            self.assertFalse(inspect.iscoroutinefunction(endpoints[path]), path)


if __name__ == "__main__":
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    unittest.main()
