from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Insight"))

from insight_local.cvops.tacitus_mcp import TacitusMcpSurface, parse_controlled_run_request


class _FakeCvOpsBridge:
    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self.posts: list[tuple[str, dict[str, Any] | None]] = []

    def get(self, path: str) -> dict[str, Any]:
        if path == "/scenarios":
            return {
                "scenarios": [
                    {"name": "demo", "display_name": "Demo", "status": "ready", "dataset": "Tiny"},
                    {"name": "archive", "display_name": "Archive", "status": "dataset", "dataset": "ArchiveSet"},
                ]
            }
        if path == "/scenarios/demo/status":
            return {"name": "demo", "status": "ready", "dataset": "Tiny", "dataset_count": 3}
        if path == "/database":
            return {"datasets": ["Tiny"], "categories": {"Tiny": "vision"}}
        if path == "/database/Tiny":
            return {
                "slug": "Tiny",
                "path": str(self.dataset_path),
                "format": "yolo_detection",
                "category": "image",
                "count": 3,
                "split_counts": {"train": 2, "val": 1},
                "classes": ["widget", "defect"],
                "images": [{"name": "images/train/a.jpg"}, {"name": "images/train/b.jpg"}],
                "folders": [{"name": "images/train"}, {"name": "labels/train"}],
                "detection_label_count": 2,
            }
        if path == "/database/Tiny/classes":
            return {"slug": "Tiny", "classes": ["widget", "defect"]}
        if path.startswith("/database/Tiny/inventory"):
            return {"slug": "Tiny", "files": [{"name": "images/train/a.jpg"}], "dirs": ["images", "labels"]}
        if path == "/scenarios/demo/pipeline":
            return {
                "scenario": "demo",
                "ci_cd": {"enabled": True, "promotion": "manual"},
                "candidate": {
                    "version_id": "demo:v2",
                    "run_version": "v2",
                    "metrics": {"map50": 0.71, "precision": 0.8},
                },
                "prod": {
                    "version_id": "demo:v1",
                    "run_version": "v1",
                    "metrics": {"map50": 0.66, "precision": 0.82},
                },
                "latest_gate": {"run_version": "v2"},
                "active_jobs": [],
                "runs": [
                    {"version": "v2", "model_version_id": "demo:v2", "metrics": {"map50": 0.71}},
                    {"version": "v1", "model_version_id": "demo:v1", "metrics": {"map50": 0.66}},
                ],
            }
        if path == "/jobs/job-1":
            return {"job_id": "job-1", "scenario": "demo", "job_type": "train", "state": "done"}
        if path == "/jobs/job-1/result":
            return {
                "job_id": "job-1",
                "scenario": "demo",
                "run_version": "v2",
                "result_path": "/runs/demo/v2",
                "ci_cd": {"gate_status": "passed", "report_path": "/runs/demo/v2/ci_cd_report.json"},
            }
        if path == "/scenarios/demo/runs/v2/gate":
            return {
                "scenario": "demo",
                "run_version": "v2",
                "gate_status": "passed",
                "report_path": "/runs/demo/v2/ci_cd_report.json",
            }
        raise RuntimeError(f"unexpected GET {path}")

    def post(self, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        self.posts.append((path, body))
        if path == "/database/import_folder":
            return {
                "slug": "Imported",
                "path": str(self.dataset_path),
                "format": "yolo_detection",
                "count": 3,
            }
        if path == "/scenarios/demo/dataset":
            return {"name": "demo", "dataset": str((body or {}).get("dataset") or "")}
        if path == "/scenarios/demo/train":
            return {"job_id": "job-1", "scenario": "demo", "job_type": "train", "state": "queued"}
        if path == "/scenarios/demo/runs/v2/promote":
            return {"ok": True, "scenario": "demo", "version_id": "demo:v2"}
        raise RuntimeError(f"unexpected POST {path}")


class TacitusMcpTests(unittest.TestCase):
    def test_tools_and_resources_include_required_surface(self) -> None:
        tool_names = {item["name"] for item in TacitusMcpSurface.tools()}
        self.assertTrue(
            {
                "scenario.list",
                "scenario.resolve",
                "dataset.read",
                "dataset.resolve",
                "dataset.import_or_bind",
                "pipeline.get",
                "run.launch",
                "job.status",
                "gate.get",
                "artifact.record",
                "experiment.record",
                "experiment.search",
                "promotion.request",
                "scenario.checkup",
                "model.compare",
            }.issubset(tool_names)
        )
        resource_uris = {item["uri"] for item in TacitusMcpSurface.resources()}
        self.assertTrue(
            {
                "tacitus://active_project",
                "tacitus://active_scenario",
                "tacitus://selected_dataset",
                "tacitus://model_registry_entry",
                "tacitus://run_artifacts",
                "tacitus://events_artifacts",
            }.issubset(resource_uris)
        )

    def test_provider_catalogs_use_safe_names_and_keep_mcp_mapping(self) -> None:
        manifest = TacitusMcpSurface.manifest()
        self.assertEqual(manifest["tool_name_map"]["scenario_list"], "scenario.list")
        self.assertEqual(manifest["tool_name_map"]["scenario_resolve"], "scenario.resolve")
        self.assertEqual(manifest["tool_name_map"]["dataset_read"], "dataset.read")
        self.assertEqual(manifest["tool_name_map"]["experiment_record"], "experiment.record")
        self.assertEqual(manifest["tool_name_map"]["experiment_search"], "experiment.search")

        openai_tools = TacitusMcpSurface.openai_tools()
        openai_by_name = {item["function"]["name"]: item for item in openai_tools}
        self.assertIn("scenario_list", openai_by_name)
        self.assertIn("scenario_resolve", openai_by_name)
        self.assertIn("dataset_read", openai_by_name)
        self.assertIn("experiment_record", openai_by_name)
        self.assertIn("experiment_search", openai_by_name)
        self.assertEqual(openai_by_name["scenario_resolve"]["type"], "function")
        self.assertIn("parameters", openai_by_name["scenario_resolve"]["function"])

        anthropic_tools = TacitusMcpSurface.anthropic_tools()
        anthropic_by_name = {item["name"]: item for item in anthropic_tools}
        self.assertIn("run_launch", anthropic_by_name)
        self.assertIn("input_schema", anthropic_by_name["run_launch"])

        catalog = TacitusMcpSurface.structured_json_tool_catalog()
        structured_by_name = {item["name"]: item for item in catalog["tools"]}
        self.assertEqual(structured_by_name["scenario_list"]["mcp_name"], "scenario.list")
        self.assertEqual(structured_by_name["dataset_read"]["mcp_name"], "dataset.read")
        self.assertEqual(structured_by_name["experiment_record"]["mcp_name"], "experiment.record")
        self.assertEqual(structured_by_name["experiment_search"]["mcp_name"], "experiment.search")
        self.assertEqual(structured_by_name["promotion_request"]["mcp_name"], "promotion.request")
        self.assertEqual(structured_by_name["scenario_checkup"]["mcp_name"], "scenario.checkup")
        self.assertEqual(structured_by_name["model_compare"]["mcp_name"], "model.compare")
        self.assertEqual(TacitusMcpSurface.mcp_tool_name("dataset_import_or_bind"), "dataset.import_or_bind")
        self.assertEqual(TacitusMcpSurface.mcp_tool_name("gate.get"), "gate.get")

    def test_provider_tool_dispatch_maps_safe_names_back_to_mcp_tools(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = _FakeCvOpsBridge(Path(td))
            surface = TacitusMcpSurface(http_get=bridge.get)

            resolved = surface.call_provider_tool("scenario_resolve", {"scenario": "demo"})
            self.assertTrue(resolved["ok"], resolved)
            self.assertEqual(resolved["tool"], "scenario.resolve")

            unknown = surface.call_provider_tool("missing_tool", {})
            self.assertFalse(unknown["ok"])
            self.assertIn("unknown MCP tool", unknown["error"])

    def test_structured_provider_tool_call_validation_and_dispatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = _FakeCvOpsBridge(Path(td))
            surface = TacitusMcpSurface(http_get=bridge.get)

            local_call = '{"tool": "pipeline_get", "arguments": {"scenario": "demo"}}'
            parsed = TacitusMcpSurface.validate_provider_tool_call(local_call)
            self.assertTrue(parsed["ok"], parsed)
            self.assertEqual(parsed["mcp_tool"], "pipeline.get")

            result = surface.dispatch_provider_tool_call(
                {"type": "tool_use", "name": "pipeline_get", "input": {"scenario": "demo"}}
            )
            self.assertTrue(result["ok"], result)
            self.assertEqual(result["tool"], "pipeline.get")

            malformed = surface.dispatch_provider_tool_call({"tool": "pipeline_get", "arguments": "[1, 2]"})
            self.assertFalse(malformed["ok"])
            self.assertIn("arguments must be a JSON object", malformed["error"])

    def test_experiment_record_and_search_use_project_ledger(self) -> None:
        events: list[dict[str, Any]] = []
        artifacts = [
            {
                "artifact_id": "weights:v14",
                "label": "v14 checkpoint",
                "path": "/runs/tiger/v14/best.pt",
                "scenario": "tiger-id",
                "dataset": "tigers",
                "tags": ["reid"],
            }
        ]
        surface = TacitusMcpSurface(
            context_provider=lambda: {"events_artifacts": {"events": events, "artifacts": artifacts}},
            event_recorder=events.append,
        )

        recorded = surface.call_provider_tool(
            "experiment_record",
            {
                "experiment_id": "tiger-v14-reid",
                "scenario": "tiger-id",
                "dataset": "tigers",
                "run_version": "v14",
                "model_version": "tiger-id:v14",
                "checkpoint_path": "/runs/tiger/v14/best.pt",
                "hypothesis": "One-source tiger checkpoint may be reusable for instance recognition.",
                "knob": "source diversity",
                "evidence": ["precision stayed high", "recall collapsed on augmented variants"],
                "outcome": "overfit, needs cross-individual rejection test",
                "reuse_notes": "reuse as template recognizer candidate",
                "tags": ["reid", "reuse"],
            },
        )
        search = surface.call_provider_tool("experiment_search", {"query": "overfit", "tags": ["reid"]})

        self.assertTrue(recorded["ok"], recorded)
        self.assertEqual(recorded["tool"], "experiment.record")
        self.assertEqual(events[0]["event_id"], "experiment:tiger-v14-reid")
        self.assertTrue(search["ok"], search)
        self.assertEqual(search["tool"], "experiment.search")
        self.assertEqual(search["data"]["count"], 1)
        self.assertEqual(search["data"]["matches"][0]["experiment_id"], "tiger-v14-reid")

    def test_dataset_read_reads_catalog_metadata_classes_and_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = _FakeCvOpsBridge(Path(td))
            surface = TacitusMcpSurface(http_get=bridge.get)

            catalog = surface.call_provider_tool("dataset_read", {})
            result = surface.call_provider_tool(
                "dataset_read",
                {"dataset": "Tiny", "limit": 1, "include_inventory": True},
            )

        self.assertTrue(catalog["ok"], catalog)
        self.assertEqual(catalog["tool"], "dataset.read")
        self.assertEqual(catalog["data"]["datasets"], ["Tiny"])
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "dataset.read")
        self.assertEqual(result["data"]["metadata"]["format"], "yolo_detection")
        self.assertEqual(result["data"]["metadata"]["images_total"], 2)
        self.assertEqual(len(result["data"]["metadata"]["images"]), 1)
        self.assertEqual(result["data"]["classes"]["classes"], ["widget", "defect"])
        self.assertEqual(result["data"]["inventory"]["slug"], "Tiny")

    def test_scenario_list_lists_and_filters_available_scenarios(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = _FakeCvOpsBridge(Path(td))
            surface = TacitusMcpSurface(http_get=bridge.get)

            result = surface.call_provider_tool("scenario_list", {})
            filtered = surface.call_provider_tool("scenario_list", {"query": "arch"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "scenario.list")
        self.assertEqual(result["data"]["count"], 2)
        self.assertEqual([item["name"] for item in result["data"]["scenarios"]], ["demo", "archive"])
        self.assertTrue(filtered["ok"], filtered)
        self.assertEqual(filtered["data"]["count"], 1)
        self.assertEqual(filtered["data"]["scenarios"][0]["name"], "archive")

    def test_scenario_checkup_reads_status_pipeline_and_latest_gate(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = _FakeCvOpsBridge(Path(td))
            surface = TacitusMcpSurface(http_get=bridge.get)

            result = surface.call_provider_tool("scenario_checkup", {"scenario": "demo"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "scenario.checkup")
        self.assertEqual(result["data"]["status"]["dataset"], "Tiny")
        self.assertEqual(result["data"]["gate"]["gate_status"], "passed")
        self.assertIn("Candidate and production point at different", " ".join(result["data"]["findings"]))

    def test_model_compare_compares_candidate_to_prod_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = _FakeCvOpsBridge(Path(td))
            surface = TacitusMcpSurface(http_get=bridge.get)

            result = surface.call_provider_tool("model_compare", {"scenario": "demo"})

        self.assertTrue(result["ok"], result)
        self.assertEqual(result["tool"], "model.compare")
        self.assertAlmostEqual(result["data"]["metric_deltas"]["map50"], 0.05)
        self.assertAlmostEqual(result["data"]["metric_deltas"]["precision"], -0.02)
        self.assertIn("Compared candidate", result["summary"])

    def test_parse_controlled_run_requires_dataset_marker_and_run_intent(self) -> None:
        self.assertIsNone(parse_controlled_run_request("What is @dataset Tiny?"))
        parsed = parse_controlled_run_request("run @dataset Tiny for me @scenario demo")
        self.assertIsNotNone(parsed)
        self.assertEqual((parsed or {})["dataset"], "Tiny")
        self.assertEqual((parsed or {})["scenario"], "demo")

    def test_controlled_run_imports_binds_launches_and_records(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset_dir = Path(td) / "drop_dataset"
            dataset_dir.mkdir()
            bridge = _FakeCvOpsBridge(dataset_dir)
            jobs: list[dict[str, Any]] = []
            artifacts: list[tuple[str, str, str, dict[str, Any]]] = []

            surface = TacitusMcpSurface(
                http_get=bridge.get,
                http_post=bridge.post,
                context_provider=lambda: {"active_scenario": "demo", "selected_dataset": "Tiny"},
                job_recorder=jobs.append,
                artifact_recorder=lambda label, path, kind, metadata: artifacts.append((label, path, kind, metadata)),
            )

            result = surface.controlled_run(source_path=str(dataset_dir))

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["data"]["job_id"], "job-1")
            self.assertEqual(jobs[0]["job_id"], "job-1")
            self.assertEqual(artifacts[0][2], "dataset")
            self.assertIn(("/database/import_folder", {"source_path": str(dataset_dir), "name": "drop_dataset"}), bridge.posts)
            self.assertIn(("/scenarios/demo/dataset", {"dataset": "Imported"}), bridge.posts)
            self.assertIn(("/scenarios/demo/train", None), bridge.posts)

    def test_collect_job_result_fetches_gate_and_records_report_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = _FakeCvOpsBridge(Path(td))
            artifacts: list[tuple[str, str, str, dict[str, Any]]] = []
            surface = TacitusMcpSurface(
                http_get=bridge.get,
                http_post=bridge.post,
                artifact_recorder=lambda label, path, kind, metadata: artifacts.append((label, path, kind, metadata)),
            )

            result = surface.collect_job_result({"job_id": "job-1"})

            self.assertTrue(result["ok"], result)
            self.assertEqual(result["data"]["gate"]["gate_status"], "passed")
            self.assertEqual(artifacts[0][1], "/runs/demo/v2/ci_cd_report.json")
            self.assertIn("gate=passed", result["summary"])

    def test_promotion_request_requires_explicit_confirmation(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bridge = _FakeCvOpsBridge(Path(td))
            events: list[dict[str, Any]] = []
            surface = TacitusMcpSurface(
                http_get=bridge.get,
                http_post=bridge.post,
                event_recorder=events.append,
            )

            pending = surface.call_tool("promotion.request", {"scenario": "demo", "version": "v2"})
            self.assertTrue(pending["ok"], pending)
            self.assertEqual(pending["data"]["state"], "confirmation_required")
            self.assertFalse(any(path.endswith("/promote") for path, _body in bridge.posts))
            self.assertEqual(events[0]["type"], "promotion_requested")

            promoted = surface.call_tool(
                "promotion.request",
                {"scenario": "demo", "version": "v2", "confirmed": True, "reason": "manual"},
            )
            self.assertTrue(promoted["ok"], promoted)
            self.assertTrue(any(path.endswith("/promote") for path, _body in bridge.posts))


if __name__ == "__main__":
    unittest.main()
