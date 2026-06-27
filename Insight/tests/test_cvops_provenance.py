from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.backfill_provenance import run_backfill
from insight_local.cvops.service import CvOpsService


class CvOpsProvenanceTests(unittest.TestCase):
    def test_ontology_links_legacy_train_job_to_model_version_from_result_path(self) -> None:
        from insight_local.cvops import ontology
        from insight_local.cvops.catalog_store import CatalogStore
        from insight_local.cvops.jobs import JobStore
        from insight_local.cvops.lineage_store import LineageStore
        from insight_local.cvops.range_store import RangeStore
        from insight_local.cvops.snapshot_store import SnapshotStore

        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            jobs = JobStore(tmp / "jobs.db")
            snapshots = SnapshotStore(tmp / "snapshots.db", tmp / "snapshot_weights")
            lineages = LineageStore(tmp / "lineages.db")
            ranges = RangeStore(tmp / "ranges.db")
            catalog = CatalogStore(tmp / "catalog.db")
            try:
                run_dir = tmp / "mlops" / "models" / "demo" / "v2"
                run_dir.mkdir(parents=True)
                job = jobs.create_job(
                    job_id="job-train-1",
                    scenario="demo",
                    job_type="train",
                    source="unit",
                    image_path="",
                    payload={},
                )
                jobs.write_result(
                    job.job_id,
                    {
                        "scenario": "demo",
                        "result_path": str(run_dir),
                        "weights": str(run_dir / "weights" / "best.pt"),
                    },
                )
                jobs.set_job_state(job.job_id, "done", result_ref=job.job_id)

                fake_registry = {
                    "version": 1,
                    "models": {
                        "demo": {
                            "versions": [
                                {
                                    "version_id": "demo:v2",
                                    "scenario": "demo",
                                    "run_version": "v2",
                                    "metrics": {},
                                    "lineage": {},
                                    "artifacts": {},
                                }
                            ],
                            "aliases": {"candidate": "demo:v2"},
                        }
                    },
                }

                with mock.patch.object(ontology, "_load_model_registry", return_value=fake_registry):
                    graph = ontology.build_graph(
                        job_store=jobs,
                        snapshots=snapshots,
                        lineages=lineages,
                        ranges=ranges,
                        catalog=catalog,
                    )
                    detail = ontology.get_entity(
                        "job",
                        job.job_id,
                        job_store=jobs,
                        snapshots=snapshots,
                        lineages=lineages,
                        ranges=ranges,
                        catalog=catalog,
                    )

                expected = ("job:job-train-1", "model_version:demo:v2", "produces")
                graph_edges = {
                    (str(e.get("source")), str(e.get("target")), str(e.get("type")))
                    for e in graph.get("edges", [])
                }
                detail_edges = {
                    (str(e.get("source")), str(e.get("target")), str(e.get("type")))
                    for e in detail.get("edges", [])
                }
                self.assertIn(expected, graph_edges)
                self.assertIn(expected, detail_edges)
            finally:
                jobs.close()
                snapshots.close()
                lineages.close()
                ranges.close()
                catalog.close()

    def test_lineage_provenance_prov_json_relations(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            w0 = tmp / "m0.pt"
            w0.write_bytes(b"weights0")
            w1 = tmp / "m1.pt"
            w1.write_bytes(b"weights1-different")

            svc = CvOpsService(
                db_path=tmp / "jobs.db",
                catalog_db_path=tmp / "catalog.db",
                catalog_assets_root=tmp / "catalog_assets",
                snapshot_db_path=tmp / "snapshots.db",
                snapshot_weights_root=tmp / "snapshot_weights",
                lineage_db_path=tmp / "lineages.db",
                provenance_db_path=tmp / "provenance.db",
                range_db_path=tmp / "ranges.db",
            )
            with TestClient(svc.app) as client:
                r0 = client.post(
                    "/snapshots",
                    json={
                        "weights_path": str(w0),
                        "model_type": "yolov8n",
                        "storage_mode": "reference",
                        "origin": "imported",
                    },
                )
                self.assertEqual(r0.status_code, 200, r0.text)
                snap0 = r0.json()["snapshot_id"]

                lin = client.post(
                    "/lineages",
                    json={
                        "name": "prov-test",
                        "sector_id": "s1",
                        "sector_path": "/test",
                        "base_snapshot_id": snap0,
                    },
                )
                self.assertEqual(lin.status_code, 200, lin.text)
                lineage_id = lin.json()["lineage_id"]

                r1 = client.post(
                    "/snapshots",
                    json={
                        "weights_path": str(w1),
                        "model_type": "yolov8n",
                        "storage_mode": "reference",
                        "origin": "lineage",
                        "lineage_id": lineage_id,
                        "parent_snapshot_id": snap0,
                    },
                )
                self.assertEqual(r1.status_code, 200, r1.text)
                snap1 = r1.json()["snapshot_id"]

                drop = client.post(
                    f"/lineages/{lineage_id}/drops",
                    json={"snapshot_id": snap1, "source": {}},
                )
                self.assertEqual(drop.status_code, 200, drop.text)

                prov = client.get(f"/lineages/{lineage_id}/provenance")
                self.assertEqual(prov.status_code, 200, prov.text)
                doc = prov.json()["prov"]
                self.assertIn("prefix", doc)
                self.assertTrue(doc["wasGeneratedBy"])
                self.assertTrue(doc["used"])
                self.assertTrue(doc["wasDerivedFrom"])
                self.assertTrue(doc["wasInformedBy"])
                self.assertTrue(doc["hadMember"])
                self.assertTrue(doc["wasAssociatedWith"])

                og = client.get("/ontology/graph")
                self.assertEqual(og.status_code, 200, og.text)
                graph = og.json()
                ntypes = {n.get("type") for n in graph.get("nodes", [])}
                etypes = {e.get("type") for e in graph.get("edges", [])}
                self.assertIn("prov_activity", ntypes)
                self.assertIn("prov_generates", etypes)
                self.assertIn("had_member", etypes)

                reg = client.get("/lineages/registry:nonexistent-scenario/provenance")
                self.assertEqual(reg.status_code, 404)

            svc.snapshots.close()
            svc.lineages.close()
            svc.provenance.close()
            svc.ranges.close()
            svc.catalog.close()
            svc.store.close()

    def test_provenance_backfill_single_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            w0 = tmp / "b0.pt"
            w0.write_bytes(b"bb0")
            w1 = tmp / "b1.pt"
            w1.write_bytes(b"bb1")

            svc = CvOpsService(
                db_path=tmp / "jobs2.db",
                catalog_db_path=tmp / "catalog2.db",
                catalog_assets_root=tmp / "catalog_assets2",
                snapshot_db_path=tmp / "snapshots2.db",
                snapshot_weights_root=tmp / "snapshot_weights2",
                lineage_db_path=tmp / "lineages2.db",
                provenance_db_path=tmp / "provenance2.db",
                range_db_path=tmp / "ranges2.db",
            )
            with TestClient(svc.app) as client:
                snap0 = client.post(
                    "/snapshots",
                    json={
                        "weights_path": str(w0),
                        "model_type": "yolov8n",
                        "storage_mode": "reference",
                    },
                ).json()["snapshot_id"]
                lineage_id = client.post(
                    "/lineages",
                    json={
                        "name": "bf",
                        "sector_id": "s1",
                        "sector_path": "/bf",
                        "base_snapshot_id": snap0,
                    },
                ).json()["lineage_id"]
                snap1 = client.post(
                    "/snapshots",
                    json={
                        "weights_path": str(w1),
                        "model_type": "yolov8n",
                        "storage_mode": "reference",
                        "lineage_id": lineage_id,
                        "parent_snapshot_id": snap0,
                        "origin": "lineage",
                    },
                ).json()["snapshot_id"]
                client.post(
                    f"/lineages/{lineage_id}/drops",
                    json={"snapshot_id": snap1},
                )

                svc2 = CvOpsService(
                    db_path=tmp / "jobs2.db",
                    catalog_db_path=tmp / "catalog2.db",
                    catalog_assets_root=tmp / "catalog_assets2",
                    snapshot_db_path=tmp / "snapshots2.db",
                    snapshot_weights_root=tmp / "snapshot_weights2",
                    lineage_db_path=tmp / "lineages2.db",
                    provenance_db_path=tmp / "provenance_empty.db",
                    range_db_path=tmp / "ranges2.db",
                )
                bf = TestClient(svc2.app).post(
                    "/provenance/backfill",
                    json={"lineage_id": lineage_id},
                )
                self.assertEqual(bf.status_code, 200, bf.text)
                bf_body = bf.json()
                self.assertEqual(bf_body.get("status"), "ok")
                self.assertGreaterEqual(int(bf_body.get("prov_nodes_after") or 0), 1)
                self.assertTrue(int(bf_body.get("prov_edges_after") or 0) > 0)
                prov = TestClient(svc2.app).get(f"/lineages/{lineage_id}/provenance")
                self.assertEqual(prov.status_code, 200)
                self.assertTrue(prov.json()["prov"]["wasGeneratedBy"])

            svc.snapshots.close()
            svc.lineages.close()
            svc.provenance.close()
            svc.ranges.close()
            svc.catalog.close()
            svc.store.close()
            svc2.snapshots.close()
            svc2.lineages.close()
            svc2.provenance.close()
            svc2.ranges.close()
            svc2.catalog.close()
            svc2.store.close()

    def test_run_backfill_cli_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            w0 = tmp / "c0.pt"
            w0.write_bytes(b"c0")
            w1 = tmp / "c1.pt"
            w1.write_bytes(b"c1")
            state = tmp / "state"
            state.mkdir()

            svc = CvOpsService(
                db_path=state / "jobs.db",
                catalog_db_path=state / "catalog.db",
                catalog_assets_root=state / "catalog_assets",
                snapshot_db_path=state / "snapshots.db",
                snapshot_weights_root=state / "snapshot_weights",
                lineage_db_path=state / "lineages.db",
                provenance_db_path=state / "provenance.db",
                range_db_path=state / "ranges.db",
            )
            with TestClient(svc.app) as client:
                snap0 = client.post(
                    "/snapshots",
                    json={
                        "weights_path": str(w0),
                        "model_type": "yolov8n",
                        "storage_mode": "reference",
                    },
                ).json()["snapshot_id"]
                lineage_id = client.post(
                    "/lineages",
                    json={
                        "name": "cli-bf",
                        "sector_id": "s1",
                        "sector_path": "/cli",
                        "base_snapshot_id": snap0,
                    },
                ).json()["lineage_id"]
                snap1 = client.post(
                    "/snapshots",
                    json={
                        "weights_path": str(w1),
                        "model_type": "yolov8n",
                        "storage_mode": "reference",
                        "lineage_id": lineage_id,
                        "parent_snapshot_id": snap0,
                        "origin": "lineage",
                    },
                ).json()["snapshot_id"]
                client.post(
                    f"/lineages/{lineage_id}/drops",
                    json={"snapshot_id": snap1},
                )

            (state / "provenance.db").unlink()
            summary = run_backfill(state_dir=state)
            self.assertEqual(summary["status"], "ok")
            self.assertEqual(summary["lineages_processed"], 1)
            self.assertGreater(int(summary["prov_edges_after"]), 0)
            summary2 = run_backfill(state_dir=state)
            self.assertEqual(summary2["prov_nodes_after"], summary["prov_nodes_after"])
            self.assertEqual(summary2["prov_edges_after"], summary["prov_edges_after"])

            svc.snapshots.close()
            svc.lineages.close()
            svc.provenance.close()
            svc.ranges.close()
            svc.catalog.close()
            svc.store.close()
