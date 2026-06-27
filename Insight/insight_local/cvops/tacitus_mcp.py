"""MCP-style tool/resource surface for Tacitus controlled CV Ops actions.

This is intentionally transport-neutral. A future MCP server, an OpenAI/Claude
tool bridge, or an Ollama structured-output bridge can all call the same
dispatcher. CV Ops service policy remains the authority for validation and
mutation.
"""

from __future__ import annotations

import json
import re
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional


JsonDict = dict[str, Any]
HttpGet = Callable[[str], JsonDict]
HttpPost = Callable[[str, Optional[JsonDict]], JsonDict]
ContextProvider = Callable[[], JsonDict]
JobRecorder = Callable[[JsonDict], None]
ArtifactRecorder = Callable[[str, str, str, JsonDict], None]
EventRecorder = Callable[[JsonDict], None]


_FINAL_JOB_STATES = {"done", "completed", "complete", "succeeded", "success", "error", "failed", "canceled", "cancelled"}
_SURFACE_NAME = "tacitus-cvops"
_SURFACE_VERSION = 1
_PROVIDER_BRIDGES = ("openai-compatible", "anthropic", "structured-json")


@dataclass(frozen=True)
class TacitusMcpTool:
    name: str
    description: str
    input_schema: JsonDict

    def to_dict(self) -> JsonDict:
        return asdict(self)


@dataclass(frozen=True)
class TacitusMcpResource:
    uri: str
    name: str
    description: str

    def to_dict(self) -> JsonDict:
        return asdict(self)


def _object_schema(properties: JsonDict, required: tuple[str, ...] = ()) -> JsonDict:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required),
        "additionalProperties": True,
    }


def _string_schema(description: str = "") -> JsonDict:
    payload: JsonDict = {"type": "string"}
    if description:
        payload["description"] = description
    return payload


def _bool_schema(description: str = "") -> JsonDict:
    payload: JsonDict = {"type": "boolean"}
    if description:
        payload["description"] = description
    return payload


def _quote(value: str) -> str:
    return urllib.parse.quote(str(value or "").strip(), safe="")


def _first_text(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _clean_args(args: Optional[JsonDict]) -> JsonDict:
    return dict(args or {}) if isinstance(args, dict) else {}


def _json_object(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(str(exc)) from exc
        if isinstance(loaded, dict):
            return dict(loaded)
    raise ValueError("expected a JSON object")


def _is_final_state(state: str) -> bool:
    return str(state or "").strip().lower() in _FINAL_JOB_STATES


def _positive_int(value: Any, *, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(1, min(maximum, parsed))


def _coerce_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    if not text:
        return []
    return [item.strip() for item in re.split(r"[,;\n]+", text) if item.strip()]


def _trim_dataset_detail(detail: JsonDict, limit: int) -> JsonDict:
    trimmed = dict(detail)
    for key in ("images", "folders", "audio_files", "csv_files"):
        value = trimmed.get(key)
        if isinstance(value, list):
            trimmed[key] = value[:limit]
            trimmed[f"{key}_total"] = len(value)
    return trimmed


class TacitusMcpSurface:
    """Validated MCP-style dispatcher for controlled Tacitus CV Ops actions."""

    def __init__(
        self,
        *,
        http_get: Optional[HttpGet] = None,
        http_post: Optional[HttpPost] = None,
        context_provider: Optional[ContextProvider] = None,
        job_recorder: Optional[JobRecorder] = None,
        artifact_recorder: Optional[ArtifactRecorder] = None,
        event_recorder: Optional[EventRecorder] = None,
    ) -> None:
        self._http_get = http_get
        self._http_post = http_post
        self._context_provider = context_provider
        self._job_recorder = job_recorder
        self._artifact_recorder = artifact_recorder
        self._event_recorder = event_recorder

    @staticmethod
    def tools() -> list[JsonDict]:
        return [tool.to_dict() for tool in _TOOLS]

    @staticmethod
    def resources() -> list[JsonDict]:
        return [resource.to_dict() for resource in _RESOURCES]

    @staticmethod
    def manifest() -> JsonDict:
        return {
            "name": _SURFACE_NAME,
            "version": _SURFACE_VERSION,
            "tools": TacitusMcpSurface.tools(),
            "resources": TacitusMcpSurface.resources(),
            "provider_bridges": list(_PROVIDER_BRIDGES),
            "tool_name_map": TacitusMcpSurface.provider_tool_map(),
        }

    @staticmethod
    def provider_tool_name(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name or "").strip()).strip("_")
        if not safe:
            return "tool"
        if not re.match(r"^[A-Za-z]", safe):
            safe = f"tool_{safe}"
        return safe

    @staticmethod
    def provider_tool_map() -> dict[str, str]:
        return {TacitusMcpSurface.provider_tool_name(tool.name): tool.name for tool in _TOOLS}

    @staticmethod
    def mcp_tool_name(name: str) -> str:
        raw = str(name or "").strip()
        if raw in {tool.name for tool in _TOOLS}:
            return raw
        return TacitusMcpSurface.provider_tool_map().get(raw, raw)

    @staticmethod
    def openai_tools() -> list[JsonDict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": TacitusMcpSurface.provider_tool_name(tool.name),
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in _TOOLS
        ]

    @staticmethod
    def anthropic_tools() -> list[JsonDict]:
        return [
            {
                "name": TacitusMcpSurface.provider_tool_name(tool.name),
                "description": tool.description,
                "input_schema": tool.input_schema,
            }
            for tool in _TOOLS
        ]

    @staticmethod
    def structured_json_tool_catalog() -> JsonDict:
        return {
            "name": _SURFACE_NAME,
            "version": _SURFACE_VERSION,
            "tools": [
                {
                    "name": TacitusMcpSurface.provider_tool_name(tool.name),
                    "mcp_name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in _TOOLS
            ],
            "call_shape": {"tool": "provider_safe_tool_name", "arguments": {}},
            "result_envelope": {"ok": "boolean", "tool": "mcp_tool_name", "data": {}, "summary": "string", "error": "string"},
        }

    @staticmethod
    def validate_provider_tool_call(payload: Any) -> JsonDict:
        try:
            call = _json_object(payload)
        except ValueError as exc:
            return {"ok": False, "tool": "", "mcp_tool": "", "arguments": {}, "error": str(exc)}

        source = call.get("function") if isinstance(call.get("function"), dict) else call
        raw_name = _first_text(source.get("name"), source.get("tool"), call.get("tool"))
        if not raw_name:
            return {"ok": False, "tool": "", "mcp_tool": "", "arguments": {}, "error": "tool name is required"}

        raw_args = source.get("arguments", source.get("input", call.get("arguments", call.get("input", {}))))
        try:
            arguments = _json_object(raw_args) if not isinstance(raw_args, dict) else dict(raw_args)
        except ValueError as exc:
            return {"ok": False, "tool": raw_name, "mcp_tool": "", "arguments": {}, "error": f"arguments must be a JSON object: {exc}"}

        mcp_name = TacitusMcpSurface.mcp_tool_name(raw_name)
        if mcp_name not in {tool.name for tool in _TOOLS}:
            return {"ok": False, "tool": raw_name, "mcp_tool": mcp_name, "arguments": arguments, "error": f"unknown MCP tool: {raw_name}"}
        return {"ok": True, "tool": raw_name, "mcp_tool": mcp_name, "arguments": arguments, "error": ""}

    def context(self) -> JsonDict:
        if self._context_provider is None:
            return {}
        payload = self._context_provider()
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def call_provider_tool(self, name: str, args: Optional[JsonDict] = None) -> JsonDict:
        return self.call_tool(TacitusMcpSurface.mcp_tool_name(name), args)

    def dispatch_provider_tool_call(self, payload: Any) -> JsonDict:
        parsed = TacitusMcpSurface.validate_provider_tool_call(payload)
        if not parsed.get("ok"):
            return self._fail("provider.tool_call", str(parsed.get("error") or "invalid provider tool call"), dict(parsed))
        return self.call_tool(str(parsed.get("mcp_tool") or ""), parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else {})

    def call_tool(self, name: str, args: Optional[JsonDict] = None) -> JsonDict:
        tool = str(name or "").strip()
        fn = {
            "scenario.list": self._tool_scenario_list,
            "scenario.resolve": self._tool_scenario_resolve,
            "dataset.read": self._tool_dataset_read,
            "dataset.resolve": self._tool_dataset_resolve,
            "dataset.import_or_bind": self._tool_dataset_import_or_bind,
            "dataset.tabular_fix": self._tool_dataset_tabular_fix,
            "dataset.tabular_split": self._tool_dataset_tabular_split,
            "dataset.tabular_history": self._tool_dataset_tabular_history,
            "dataset.tabular_undo": self._tool_dataset_tabular_undo,
            "dataset.tabular_import_folder": self._tool_dataset_tabular_import_folder,
            "dataset.tabular_target": self._tool_dataset_tabular_target,
            "dataset.tabular_score": self._tool_dataset_tabular_score,
            "pipeline.get": self._tool_pipeline_get,
            "scenario.checkup": self._tool_scenario_checkup,
            "model.compare": self._tool_model_compare,
            "run.launch": self._tool_run_launch,
            "job.status": self._tool_job_status,
            "gate.get": self._tool_gate_get,
            "artifact.record": self._tool_artifact_record,
            "experiment.record": self._tool_experiment_record,
            "experiment.search": self._tool_experiment_search,
            "promotion.request": self._tool_promotion_request,
        }.get(tool)
        if fn is None:
            return self._fail(tool or "unknown", f"unknown MCP tool: {tool}")
        try:
            return fn(_clean_args(args))
        except Exception as exc:
            return self._fail(tool, str(exc))

    def read_resource(self, uri: str, params: Optional[JsonDict] = None) -> JsonDict:
        target = str(uri or "").strip()
        args = _clean_args(params)
        try:
            if target == "tacitus://active_project":
                return self._resource_ok(target, self.context().get("active_project") or {})
            if target == "tacitus://active_scenario":
                return self._resource_ok(target, {"scenario": str(self.context().get("active_scenario") or "")})
            if target == "tacitus://selected_dataset":
                return self._resource_ok(target, {"dataset": str(self.context().get("selected_dataset") or "")})
            if target == "tacitus://events_artifacts":
                return self._resource_ok(target, self.context().get("events_artifacts") or {})
            if target == "tacitus://ingested_memory":
                return self._resource_ok(target, {"summary": str(self.context().get("ingested_memory") or "")})
            if target == "tacitus://model_registry_entry":
                scenario = self._scenario_name(args)
                if not scenario:
                    return self._resource_fail(target, "scenario is required")
                pipeline = self._get(f"/scenarios/{_quote(scenario)}/pipeline")
                alias = _first_text(args.get("alias"), "candidate")
                data = pipeline.get(alias) if alias in {"candidate", "prod"} else {}
                return self._resource_ok(target, data if isinstance(data, dict) else {})
            if target == "tacitus://run_artifacts":
                return self._resource_ok(target, self.collect_job_result(args).get("data") or {})
            return self._resource_fail(target, f"unknown MCP resource: {target}")
        except Exception as exc:
            return self._resource_fail(target, str(exc))

    def controlled_run(
        self,
        *,
        scenario: str = "",
        dataset: str = "",
        source_path: str = "",
        mode: str = "train",
        bind_dataset: bool = True,
    ) -> JsonDict:
        """Resolve context, optionally bind/import a dataset, and queue a train/update job."""
        steps: list[JsonDict] = []
        scenario_res = self.call_tool("scenario.resolve", {"scenario": scenario})
        steps.append({"step": "scenario.resolve", "result": scenario_res})
        if not scenario_res.get("ok"):
            return self._ok("tacitus.controlled_run", {"steps": steps}, "Could not resolve scenario.", ok=False, error=str(scenario_res.get("error") or "scenario resolution failed"))

        scenario_name = str((scenario_res.get("data") or {}).get("scenario") or "").strip()
        if dataset or source_path:
            dataset_res = self.call_tool(
                "dataset.resolve",
                {"scenario": scenario_name, "dataset": dataset, "source_path": source_path},
            )
            steps.append({"step": "dataset.resolve", "result": dataset_res})
            import_res = self.call_tool(
                "dataset.import_or_bind",
                {
                    "scenario": scenario_name,
                    "dataset": dataset or str((dataset_res.get("data") or {}).get("dataset") or ""),
                    "source_path": source_path,
                    "bind": bool(bind_dataset),
                },
            )
            steps.append({"step": "dataset.import_or_bind", "result": import_res})
            if not import_res.get("ok"):
                return self._ok("tacitus.controlled_run", {"scenario": scenario_name, "steps": steps}, "Could not bind dataset.", ok=False, error=str(import_res.get("error") or "dataset bind failed"))

        pipeline_res = self.call_tool("pipeline.get", {"scenario": scenario_name})
        steps.append({"step": "pipeline.get", "result": pipeline_res})

        launch_res = self.call_tool(
            "run.launch",
            {"scenario": scenario_name, "mode": mode, "record": True},
        )
        steps.append({"step": "run.launch", "result": launch_res})
        if not launch_res.get("ok"):
            return self._ok("tacitus.controlled_run", {"scenario": scenario_name, "steps": steps}, "Could not launch run.", ok=False, error=str(launch_res.get("error") or "run launch failed"))

        launch_data = dict(launch_res.get("data") or {})
        job_id = str(launch_data.get("job_id") or "").strip()
        summary = f"Queued {str(mode or 'train').strip() or 'train'} for {scenario_name}"
        if job_id:
            summary += f" as {job_id}"
        return self._ok(
            "tacitus.controlled_run",
            {
                "scenario": scenario_name,
                "job_id": job_id,
                "mode": str(mode or "train").strip() or "train",
                "steps": steps,
                "pipeline": pipeline_res.get("data") or {},
            },
            summary + ".",
        )

    def collect_job_result(self, args: Optional[JsonDict] = None) -> JsonDict:
        clean = _clean_args(args)
        status_res = self.call_tool("job.status", clean)
        if not status_res.get("ok"):
            return status_res
        status_data = dict(status_res.get("data") or {})
        job = dict(status_data.get("job") or status_data)
        result = dict(status_data.get("result") or {})
        scenario = _first_text(clean.get("scenario"), job.get("scenario"), result.get("scenario"))
        run_version = _first_text(clean.get("version"), clean.get("run_version"), result.get("run_version"))
        gate: JsonDict = {}
        if scenario and run_version:
            gate_res = self.call_tool("gate.get", {"scenario": scenario, "version": run_version})
            if gate_res.get("ok"):
                gate = dict(gate_res.get("data") or {})
                report_path = str(gate.get("report_path") or "").strip()
                if report_path and self._artifact_recorder is not None:
                    self._artifact_recorder(
                        "CI/CD gate report",
                        report_path,
                        "ci_cd_report",
                        {"job_id": str(job.get("job_id") or clean.get("job_id") or ""), "scenario": scenario, "run_version": run_version},
                    )
        summary = self._job_summary(job, result, gate)
        return self._ok(
            "tacitus.collect_job_result",
            {"job": job, "result": result, "gate": gate},
            summary,
        )

    def _tool_scenario_list(self, args: JsonDict) -> JsonDict:
        query = _first_text(args.get("query"), args.get("filter"))
        payload = self._get("/scenarios")
        scenarios = [dict(item) for item in payload.get("scenarios") or [] if isinstance(item, dict)]
        if query:
            needle = query.lower()
            fields = ("name", "display_name", "description", "status", "dataset")
            scenarios = [
                item
                for item in scenarios
                if any(needle in str(item.get(field) or "").lower() for field in fields)
            ]
        data: JsonDict = {"scenarios": scenarios, "count": len(scenarios)}
        if query:
            data["query"] = query
        return self._ok("scenario.list", data, f"Listed {len(scenarios)} scenario(s).")

    def _tool_scenario_resolve(self, args: JsonDict) -> JsonDict:
        query = _first_text(args.get("scenario"), args.get("name"), args.get("query"), self.context().get("active_scenario"))
        if not query:
            return self._fail("scenario.resolve", "scenario is required")
        payload = self._get("/scenarios")
        scenarios = [dict(item) for item in payload.get("scenarios") or [] if isinstance(item, dict)]
        exact = [item for item in scenarios if str(item.get("name") or "").lower() == query.lower()]
        matches = exact or [
            item for item in scenarios
            if query.lower() in str(item.get("name") or "").lower()
            or query.lower() in str(item.get("display_name") or "").lower()
        ]
        if len(matches) == 1:
            item = dict(matches[0])
            item["scenario"] = str(item.get("name") or "")
            return self._ok("scenario.resolve", item, f"Resolved scenario {item['scenario']}.")
        if len(matches) > 1:
            return self._fail(
                "scenario.resolve",
                f"scenario is ambiguous: {', '.join(str(item.get('name') or '') for item in matches[:8])}",
                {"matches": matches[:8]},
            )
        return self._fail("scenario.resolve", f"scenario not found: {query}", {"query": query})

    def _tool_dataset_read(self, args: JsonDict) -> JsonDict:
        dataset = _first_text(args.get("dataset"), args.get("slug"), args.get("name"), args.get("query"))
        scenario = self._scenario_name(args)
        if not dataset and scenario:
            resolved = self.call_tool("dataset.resolve", {"scenario": scenario})
            if resolved.get("ok"):
                data = resolved.get("data") if isinstance(resolved.get("data"), dict) else {}
                dataset = _first_text(data.get("dataset"), data.get("slug"))
        if not dataset:
            catalog = self._get("/database")
            names = [str(item) for item in catalog.get("datasets") or [] if str(item).strip()]
            data = {
                "datasets": names,
                "categories": catalog.get("categories") if isinstance(catalog.get("categories"), dict) else {},
                "tabular_datasets": catalog.get("tabular_datasets") if isinstance(catalog.get("tabular_datasets"), list) else [],
                "text_datasets": catalog.get("text_datasets") if isinstance(catalog.get("text_datasets"), list) else [],
            }
            return self._ok("dataset.read", data, f"Read dataset catalog with {len(names)} image/audio dataset(s).")

        limit = _positive_int(args.get("limit"), default=12, maximum=50)
        include_inventory = bool(args.get("include_inventory") or args.get("inventory"))
        include_profile = bool(args.get("include_profile") or args.get("profile"))
        detail = dict(self._get(f"/database/{_quote(dataset)}"))
        detail.setdefault("dataset", dataset)
        detail.setdefault("slug", dataset)
        data: JsonDict = {"dataset": dataset, "metadata": _trim_dataset_detail(detail, limit)}
        if include_inventory:
            try:
                data["inventory"] = self._get(f"/database/{_quote(dataset)}/inventory?max_files={limit}")
            except Exception as exc:
                data["inventory_error"] = str(exc)
        try:
            data["classes"] = self._get(f"/database/{_quote(dataset)}/classes")
        except Exception:
            if isinstance(detail.get("classes"), list):
                data["classes"] = {"slug": dataset, "classes": list(detail.get("classes") or [])}
        if include_profile:
            try:
                data["tabular_profile"] = self._get(f"/database/{_quote(dataset)}/tabular_profile?max_rows=5000")
            except Exception as exc:
                data["tabular_profile_error"] = str(exc)
        fmt = str(detail.get("format") or "unknown").strip() or "unknown"
        count = str(detail.get("count") or "").strip()
        count_bit = f" with {count} item(s)" if count else ""
        return self._ok("dataset.read", data, f"Read dataset {dataset} ({fmt}){count_bit}.")

    def _tool_dataset_resolve(self, args: JsonDict) -> JsonDict:
        dataset = _first_text(args.get("dataset"), args.get("slug"), args.get("name"), args.get("query"), self.context().get("selected_dataset"))
        source_path = _first_text(args.get("source_path"), args.get("path"))
        scenario = self._scenario_name(args)
        if source_path:
            path = Path(source_path).expanduser()
            data = {
                "dataset": dataset,
                "source_path": str(path),
                "exists": path.exists(),
                "is_dir": path.is_dir(),
                "import_needed": path.exists() and path.is_dir() and not dataset,
            }
            return self._ok("dataset.resolve", data, f"Resolved dataset source {path}.")
        if not dataset and scenario:
            try:
                status = self._get(f"/scenarios/{_quote(scenario)}/status")
                dataset = _first_text(status.get("dataset"), status.get("dataset_name"), status.get("dataset_slug"))
            except Exception:
                dataset = ""
        if not dataset:
            return self._fail("dataset.resolve", "dataset or source_path is required")
        try:
            info = self._get(f"/database/{_quote(dataset)}")
            data = dict(info)
            data.setdefault("dataset", dataset)
            data.setdefault("slug", dataset)
            return self._ok("dataset.resolve", data, f"Resolved dataset {dataset}.")
        except Exception:
            payload = self._get("/database")
            names = [str(item) for item in payload.get("datasets") or []]
            matches = [name for name in names if name.lower() == dataset.lower()] or [name for name in names if dataset.lower() in name.lower()]
            if len(matches) == 1:
                return self._ok("dataset.resolve", {"dataset": matches[0], "slug": matches[0]}, f"Resolved dataset {matches[0]}.")
            if len(matches) > 1:
                return self._fail("dataset.resolve", f"dataset is ambiguous: {', '.join(matches[:8])}", {"matches": matches[:8]})
            return self._fail("dataset.resolve", f"dataset not found: {dataset}", {"query": dataset})

    def _tool_dataset_import_or_bind(self, args: JsonDict) -> JsonDict:
        scenario = self._scenario_name(args)
        dataset = _first_text(args.get("dataset"), args.get("slug"), args.get("name"))
        source_path = _first_text(args.get("source_path"), args.get("path"))
        bind = bool(args.get("bind", True))
        data: JsonDict = {"scenario": scenario, "dataset": dataset, "imported": False, "bound": False}
        if source_path:
            path = Path(source_path).expanduser()
            if not path.is_dir():
                return self._fail("dataset.import_or_bind", f"source_path must be an existing directory: {path}")
            imported = self._post(
                "/database/import_folder",
                {"source_path": str(path), "name": _first_text(args.get("import_name"), args.get("name"), path.name)},
            )
            data["import_result"] = imported
            data["imported"] = True
            dataset = str(imported.get("slug") or dataset or "").strip()
            data["dataset"] = dataset
            if self._artifact_recorder is not None:
                self._artifact_recorder(
                    f"Dataset {dataset or path.name}",
                    str(imported.get("path") or path),
                    "dataset",
                    {"scenario": scenario, "source_path": str(path), "imported": True},
                )
        if bind:
            if not scenario:
                return self._fail("dataset.import_or_bind", "scenario is required to bind a dataset")
            if not dataset:
                return self._fail("dataset.import_or_bind", "dataset is required to bind")
            bound = self._post(f"/scenarios/{_quote(scenario)}/dataset", {"dataset": dataset})
            data["bind_result"] = bound
            data["bound"] = True
        if not data["imported"] and not data["bound"]:
            return self._fail("dataset.import_or_bind", "nothing to import or bind")
        return self._ok("dataset.import_or_bind", data, f"Dataset {dataset or source_path} ready for {scenario or 'scenario'}.")

    def _resolve_tabular_dataset(self, args: JsonDict) -> str:
        dataset = _first_text(
            args.get("dataset"), args.get("slug"), args.get("name"),
            self.context().get("selected_dataset"),
        )
        scenario = self._scenario_name(args)
        if not dataset and scenario:
            resolved = self.call_tool("dataset.resolve", {"scenario": scenario})
            if resolved.get("ok"):
                data = resolved.get("data") if isinstance(resolved.get("data"), dict) else {}
                dataset = _first_text(data.get("dataset"), data.get("slug"))
        return dataset

    def _tool_dataset_tabular_fix(self, args: JsonDict) -> JsonDict:
        """Apply profile-driven cleaning ops to a tabular dataset CSV (drop dup rows,
        drop columns, drop high-missing/constant columns, impute missing)."""
        dataset = self._resolve_tabular_dataset(args)
        if not dataset:
            return self._fail("dataset.tabular_fix", "dataset is required")

        ops = args.get("ops")
        if not isinstance(ops, list) or not ops:
            # Convenience: a single op can be given via flat fields.
            single = _first_text(args.get("op"))
            if not single:
                return self._fail("dataset.tabular_fix", "ops (list) or op (string) is required")
            op: JsonDict = {"op": single}
            if isinstance(args.get("columns"), list):
                op["columns"] = list(args.get("columns") or [])
            for key in ("strategy", "fill_value"):
                if args.get(key) is not None:
                    op[key] = args.get(key)
            if args.get("threshold_pct") is not None:
                op["threshold_pct"] = args.get("threshold_pct")
            ops = [op]

        payload: JsonDict = {"ops": ops}
        if _first_text(args.get("csv_name"), args.get("file")):
            payload["name"] = _first_text(args.get("csv_name"), args.get("file"))
        result = self._post(f"/database/{_quote(dataset)}/tabular_transform", payload)
        before = result.get("before") if isinstance(result.get("before"), dict) else {}
        after = result.get("after") if isinstance(result.get("after"), dict) else {}
        summary = (
            f"Applied {len(result.get('ops_applied') or [])} fix op(s) to {dataset}: "
            f"{before.get('rows', '?')}x{before.get('cols', '?')} -> "
            f"{after.get('rows', '?')}x{after.get('cols', '?')}."
        )
        return self._ok("dataset.tabular_fix", dict(result), summary)

    def _tool_dataset_tabular_split(self, args: JsonDict) -> JsonDict:
        """Write reproducible, optionally stratified train/val/test split assignments
        for a tabular dataset."""
        dataset = self._resolve_tabular_dataset(args)
        if not dataset:
            return self._fail("dataset.tabular_split", "dataset is required")

        payload: JsonDict = {}
        for key in ("val_frac", "test_frac", "seed"):
            if args.get(key) is not None:
                payload[key] = args.get(key)
        stratify = _first_text(args.get("stratify_col"), args.get("stratify"))
        if stratify:
            payload["stratify_col"] = stratify
        if args.get("write_column") is not None:
            payload["write_column"] = bool(args.get("write_column"))
        if _first_text(args.get("csv_name"), args.get("file")):
            payload["name"] = _first_text(args.get("csv_name"), args.get("file"))
        result = self._post(f"/database/{_quote(dataset)}/tabular_split", payload)
        counts = result.get("counts") if isinstance(result.get("counts"), dict) else {}
        strat_bit = " (stratified)" if result.get("stratified") else ""
        summary = (
            f"Split {dataset}{strat_bit}: train={counts.get('train', '?')}, "
            f"val={counts.get('val', '?')}, test={counts.get('test', '?')}."
        )
        return self._ok("dataset.tabular_split", dict(result), summary)

    def _tool_dataset_tabular_history(self, args: JsonDict) -> JsonDict:
        """Read the provenance log of cleaning transforms applied to a tabular dataset."""
        dataset = self._resolve_tabular_dataset(args)
        if not dataset:
            return self._fail("dataset.tabular_history", "dataset is required")
        path = f"/database/{_quote(dataset)}/tabular_history"
        csv_name = _first_text(args.get("csv_name"), args.get("file"))
        if csv_name:
            path = f"{path}?name={_quote(csv_name)}"
        result = self._get(path)
        summary = (
            f"{dataset} has {result.get('count', 0)} recorded transform(s); "
            f"undo {'available' if result.get('can_undo') else 'unavailable'}."
        )
        return self._ok("dataset.tabular_history", dict(result), summary)

    def _tool_dataset_tabular_undo(self, args: JsonDict) -> JsonDict:
        """Undo the most recent tabular transform by restoring its backup."""
        dataset = self._resolve_tabular_dataset(args)
        if not dataset:
            return self._fail("dataset.tabular_undo", "dataset is required")
        payload: JsonDict = {}
        if _first_text(args.get("csv_name"), args.get("file")):
            payload["name"] = _first_text(args.get("csv_name"), args.get("file"))
        result = self._post(f"/database/{_quote(dataset)}/tabular_undo", payload)
        after = result.get("after") if isinstance(result.get("after"), dict) else {}
        summary = (
            f"Reverted {dataset} to {after.get('rows', '?')}x{after.get('cols', '?')} "
            f"(revision {result.get('revision', '?')})."
        )
        return self._ok("dataset.tabular_undo", dict(result), summary)

    def _tool_dataset_tabular_import_folder(self, args: JsonDict) -> JsonDict:
        """Batch-import every supported tabular file (csv/tsv/xlsx/parquet/json/jsonl)
        in a local folder as datasets."""
        source = _first_text(args.get("source_path"), args.get("path"), args.get("folder"))
        if not source:
            return self._fail("dataset.tabular_import_folder", "source_path is required")
        payload: JsonDict = {"source_path": source, "recursive": bool(args.get("recursive"))}
        result = self._post("/database/import_tabular_folder", payload)
        summary = (
            f"Imported {result.get('imported_count', 0)} of {result.get('found', 0)} "
            f"tabular file(s) from {source}."
        )
        return self._ok("dataset.tabular_import_folder", dict(result), summary)

    def _tool_dataset_tabular_target(self, args: JsonDict) -> JsonDict:
        """Analyze a tabular target column: task type, class balance, leakage, and a
        train-readiness gate (blockers + warnings)."""
        dataset = self._resolve_tabular_dataset(args)
        if not dataset:
            return self._fail("dataset.tabular_target", "dataset is required")
        label = _first_text(args.get("label_col"), args.get("label"), args.get("target"))
        if not label:
            return self._fail("dataset.tabular_target", "label_col is required")
        path = f"/database/{_quote(dataset)}/tabular_target?label_col={_quote(label)}"
        feats = args.get("feature_cols")
        if isinstance(feats, list) and feats:
            path += f"&feature_cols={_quote(','.join(str(f) for f in feats))}"
        elif _first_text(feats):
            path += f"&feature_cols={_quote(_first_text(feats))}"
        if _first_text(args.get("csv_name"), args.get("file")):
            path += f"&name={_quote(_first_text(args.get('csv_name'), args.get('file')))}"
        result = self._get(path)
        readiness = result.get("readiness") if isinstance(result.get("readiness"), dict) else {}
        ready = bool(readiness.get("ready"))
        n_block = len(readiness.get("blockers") or [])
        n_warn = len(readiness.get("warnings") or [])
        summary = (
            f"{dataset} target '{label}': task={result.get('task', '?')}, "
            f"{'READY' if ready else 'NOT READY'} ({n_block} blocker(s), {n_warn} warning(s))."
        )
        return self._ok("dataset.tabular_target", dict(result), summary)

    def _tool_dataset_tabular_score(self, args: JsonDict) -> JsonDict:
        """Batch-score a tabular dataset against a trained tabular model (resolved from a
        scenario/version or an explicit model_path)."""
        dataset = self._resolve_tabular_dataset(args)
        if not dataset:
            return self._fail("dataset.tabular_score", "dataset (input to score) is required")
        scenario = self._scenario_name(args)
        model_path = _first_text(args.get("model_path"), args.get("weights_path"))
        if not scenario and not model_path:
            return self._fail("dataset.tabular_score", "provide scenario or model_path")
        payload: JsonDict = {
            "scenario": scenario,
            "version": _first_text(args.get("version")),
            "model_path": model_path,
            "write_dataset": bool(args.get("write_dataset")),
            "output_name": _first_text(args.get("output_name")),
        }
        if _first_text(args.get("csv_name"), args.get("file")):
            payload["name"] = _first_text(args.get("csv_name"), args.get("file"))
        result = self._post(f"/database/{_quote(dataset)}/tabular_score", payload)
        written = result.get("written_slug")
        summary = (
            f"Scored {result.get('n_rows', 0)} row(s) of {dataset} "
            f"(task={result.get('task', '?')})"
            + (f"; wrote dataset '{written}'." if written else ".")
        )
        return self._ok("dataset.tabular_score", dict(result), summary)

    def _tool_pipeline_get(self, args: JsonDict) -> JsonDict:
        scenario = self._scenario_name(args)
        if not scenario:
            return self._fail("pipeline.get", "scenario is required")
        return self._ok(
            "pipeline.get",
            self._get(f"/scenarios/{_quote(scenario)}/pipeline"),
            f"Loaded pipeline for {scenario}.",
        )

    def _tool_scenario_checkup(self, args: JsonDict) -> JsonDict:
        scenario = self._scenario_name(args)
        if not scenario:
            return self._fail("scenario.checkup", "scenario is required")
        status = self._get(f"/scenarios/{_quote(scenario)}/status")
        pipeline = self._get(f"/scenarios/{_quote(scenario)}/pipeline")
        latest_gate = pipeline.get("latest_gate") if isinstance(pipeline.get("latest_gate"), dict) else {}
        gate: JsonDict = {}
        run_version = _first_text(latest_gate.get("run_version"), args.get("version"), args.get("run_version"))
        if run_version:
            gate_res = self.call_tool("gate.get", {"scenario": scenario, "version": run_version})
            if gate_res.get("ok"):
                gate = dict(gate_res.get("data") or {})
        candidate = pipeline.get("candidate") if isinstance(pipeline.get("candidate"), dict) else {}
        prod = pipeline.get("prod") if isinstance(pipeline.get("prod"), dict) else {}
        active_jobs = pipeline.get("active_jobs") if isinstance(pipeline.get("active_jobs"), list) else []
        findings = self._scenario_checkup_findings(status, pipeline, gate)
        summary = self._scenario_checkup_summary(scenario, status, candidate, prod, gate, active_jobs)
        return self._ok(
            "scenario.checkup",
            {
                "scenario": scenario,
                "status": status,
                "pipeline": pipeline,
                "gate": gate,
                "findings": findings,
            },
            summary,
        )

    def _tool_model_compare(self, args: JsonDict) -> JsonDict:
        scenario = self._scenario_name(args)
        if not scenario:
            return self._fail("model.compare", "scenario is required")
        left_ref = _first_text(args.get("left"), args.get("a"), args.get("candidate"), "candidate")
        right_ref = _first_text(args.get("right"), args.get("b"), args.get("baseline"), "prod")
        pipeline = self._get(f"/scenarios/{_quote(scenario)}/pipeline")
        left = self._resolve_pipeline_model_ref(pipeline, left_ref)
        right = self._resolve_pipeline_model_ref(pipeline, right_ref)
        if not left:
            return self._fail("model.compare", f"model reference not found: {left_ref}", {"scenario": scenario})
        if not right:
            return self._fail("model.compare", f"model reference not found: {right_ref}", {"scenario": scenario})
        left_metrics = self._numeric_metrics(left)
        right_metrics = self._numeric_metrics(right)
        deltas: JsonDict = {}
        for key in sorted(set(left_metrics) & set(right_metrics)):
            deltas[key] = left_metrics[key] - right_metrics[key]
        summary = self._model_compare_summary(left_ref, left, right_ref, right, deltas)
        return self._ok(
            "model.compare",
            {
                "scenario": scenario,
                "left_ref": left_ref,
                "right_ref": right_ref,
                "left": left,
                "right": right,
                "left_metrics": left_metrics,
                "right_metrics": right_metrics,
                "metric_deltas": deltas,
            },
            summary,
        )

    def _tool_run_launch(self, args: JsonDict) -> JsonDict:
        scenario = self._scenario_name(args)
        if not scenario:
            return self._fail("run.launch", "scenario is required")
        mode = str(args.get("mode") or "train").strip().lower()
        if mode not in {"train", "update"}:
            return self._fail("run.launch", "mode must be train or update")
        payload = args.get("payload") if isinstance(args.get("payload"), dict) else None
        launched = self._post(f"/scenarios/{_quote(scenario)}/{mode}", payload)
        data = dict(launched)
        data.setdefault("scenario", scenario)
        data.setdefault("job_type", "train")
        data.setdefault("source", "tacitus")
        if bool(args.get("record", True)) and self._job_recorder is not None:
            self._job_recorder(data)
        return self._ok("run.launch", data, f"Launched {mode} for {scenario}.")

    def _tool_job_status(self, args: JsonDict) -> JsonDict:
        job_id = _first_text(args.get("job_id"), args.get("id"))
        if not job_id:
            return self._fail("job.status", "job_id is required")
        job = self._get(f"/jobs/{_quote(job_id)}")
        data: JsonDict = {"job": job}
        state = str(job.get("state") or "").strip()
        if bool(args.get("include_result", True)) and _is_final_state(state):
            try:
                data["result"] = self._get(f"/jobs/{_quote(job_id)}/result")
            except Exception:
                data["result"] = {}
        return self._ok("job.status", data, f"Job {job_id}: {state or 'unknown'}.")

    def _tool_gate_get(self, args: JsonDict) -> JsonDict:
        scenario = self._scenario_name(args)
        version = _first_text(args.get("version"), args.get("run_version"))
        job_id = _first_text(args.get("job_id"), args.get("id"))
        result: JsonDict = {}
        if job_id and (not scenario or not version):
            try:
                result = self._get(f"/jobs/{_quote(job_id)}/result")
            except Exception:
                result = {}
            scenario = scenario or str(result.get("scenario") or "").strip()
            version = version or str(result.get("run_version") or "").strip()
        if not scenario:
            return self._fail("gate.get", "scenario is required")
        if not version:
            pipeline = self._get(f"/scenarios/{_quote(scenario)}/pipeline")
            latest_gate = pipeline.get("latest_gate") if isinstance(pipeline.get("latest_gate"), dict) else {}
            version = str((latest_gate or {}).get("run_version") or "").strip()
        if not version:
            return self._fail("gate.get", "run version is required")
        try:
            report = self._get(f"/scenarios/{_quote(scenario)}/runs/{_quote(version)}/gate")
        except Exception:
            ci_cd = result.get("ci_cd") if isinstance(result.get("ci_cd"), dict) else {}
            if ci_cd:
                report = {"scenario": scenario, "run_version": version, **ci_cd}
            else:
                raise
        return self._ok("gate.get", dict(report), f"Loaded gate report for {scenario}:{version}.")

    def _tool_artifact_record(self, args: JsonDict) -> JsonDict:
        label = _first_text(args.get("label"), args.get("name"), args.get("path"), "Artifact")
        path = _first_text(args.get("path"), args.get("uri"))
        kind = _first_text(args.get("kind"), "file")
        metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
        if self._artifact_recorder is None:
            return self._fail("artifact.record", "artifact recorder is not available")
        self._artifact_recorder(label, path, kind, dict(metadata or {}))
        return self._ok("artifact.record", {"label": label, "path": path, "kind": kind}, f"Recorded artifact {label}.")

    def _tool_experiment_record(self, args: JsonDict) -> JsonDict:
        if self._event_recorder is None:
            return self._fail("experiment.record", "event recorder is not available")
        scenario = self._scenario_name(args)
        dataset = _first_text(args.get("dataset"))
        run_version = _first_text(args.get("run_version"), args.get("version"))
        model_version = _first_text(args.get("model_version"), args.get("model_version_id"), args.get("version_id"))
        hypothesis = _first_text(args.get("hypothesis"), args.get("claim"), args.get("question"))
        experiment_id = _first_text(args.get("experiment_id"), args.get("id"))
        if not experiment_id:
            basis = _first_text(scenario, dataset, run_version, model_version, hypothesis, "experiment")
            experiment_id = re.sub(r"[^A-Za-z0-9_.:-]+", "-", basis).strip("-")[:80] or "experiment"
        event: JsonDict = {
            "event_id": f"experiment:{experiment_id}",
            "type": "experiment_record",
            "experiment_id": experiment_id,
            "scenario": scenario,
            "dataset": dataset,
            "run_version": run_version,
            "model_version": model_version,
            "checkpoint_path": _first_text(args.get("checkpoint_path"), args.get("checkpoint"), args.get("weights_path")),
            "status": _first_text(args.get("status"), "noted"),
            "hypothesis": hypothesis,
            "knob": _first_text(args.get("knob")),
            "evidence": _coerce_list(args.get("evidence")),
            "outcome": _first_text(args.get("outcome"), args.get("finding")),
            "reuse_notes": _first_text(args.get("reuse_notes"), args.get("reuse")),
            "next_experiment": _first_text(args.get("next_experiment"), args.get("next_step")),
            "tags": _coerce_list(args.get("tags")),
        }
        metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
        if metadata:
            event["metadata"] = dict(metadata)
        self._event_recorder(event)
        return self._ok("experiment.record", event, f"Recorded experiment {experiment_id}.")

    def _tool_experiment_search(self, args: JsonDict) -> JsonDict:
        query = _first_text(args.get("query"), args.get("q"), args.get("hypothesis"))
        scenario = _first_text(args.get("scenario"))
        dataset = _first_text(args.get("dataset"))
        tags = [item.lower() for item in _coerce_list(args.get("tags"))]
        limit = _positive_int(args.get("limit"), default=8, maximum=25)
        ledger = self.context().get("events_artifacts")
        ledger = ledger if isinstance(ledger, dict) else {}
        events = [dict(item) for item in ledger.get("events") or [] if isinstance(item, dict)]
        artifacts = [dict(item) for item in ledger.get("artifacts") or [] if isinstance(item, dict)]
        rows: list[JsonDict] = []
        for item in events:
            if str(item.get("type") or "") == "experiment_record" and self._experiment_matches(item, query, scenario, dataset, tags):
                rows.append({"kind": "experiment", **item})
        for item in artifacts:
            if self._experiment_matches(item, query, scenario, dataset, tags):
                rows.append({"kind": "artifact", **item})
        rows = list(reversed(rows))[:limit]
        summary = f"Found {len(rows)} reusable experiment record(s)." if rows else "No matching experiments found."
        return self._ok(
            "experiment.search",
            {"matches": rows, "count": len(rows), "query": query, "scenario": scenario, "dataset": dataset, "tags": tags},
            summary,
        )

    def _tool_promotion_request(self, args: JsonDict) -> JsonDict:
        scenario = self._scenario_name(args)
        version = _first_text(args.get("version"), args.get("run_version"))
        if not scenario or not version:
            return self._fail("promotion.request", "scenario and version are required")
        actor = _first_text(args.get("actor"), "tacitus")
        reason = _first_text(args.get("reason"), "manual Tacitus promotion request")
        confirmed = bool(args.get("confirmed") or args.get("manual_confirmed"))
        if not confirmed:
            data = {"scenario": scenario, "version": version, "state": "confirmation_required", "reason": reason}
            if self._event_recorder is not None:
                self._event_recorder({"type": "promotion_requested", **data})
            return self._ok("promotion.request", data, f"Promotion for {scenario}:{version} requires explicit confirmation.")
        result = self._post(
            f"/scenarios/{_quote(scenario)}/runs/{_quote(version)}/promote",
            {"actor": actor, "reason": reason, "override": bool(args.get("override", False))},
        )
        if self._event_recorder is not None:
            self._event_recorder({"type": "promotion_completed", "scenario": scenario, "version": version, "result": result})
        return self._ok("promotion.request", result, f"Promotion completed for {scenario}:{version}.")

    def _scenario_name(self, args: JsonDict) -> str:
        return _first_text(args.get("scenario"), args.get("name"), self.context().get("active_scenario"))

    def _experiment_matches(self, item: JsonDict, query: str, scenario: str, dataset: str, tags: list[str]) -> bool:
        if scenario and str(item.get("scenario") or "").lower() != scenario.lower():
            return False
        if dataset and str(item.get("dataset") or "").lower() != dataset.lower():
            return False
        item_tags = [tag.lower() for tag in _coerce_list(item.get("tags"))]
        if tags and not all(tag in item_tags for tag in tags):
            return False
        if not query:
            return True
        haystack = " ".join(
            str(item.get(key) or "")
            for key in (
                "experiment_id",
                "label",
                "path",
                "hypothesis",
                "claim",
                "outcome",
                "reuse_notes",
                "next_experiment",
                "checkpoint_path",
                "model_version",
                "run_version",
            )
        )
        return query.lower() in haystack.lower()

    def _scenario_checkup_findings(self, status: JsonDict, pipeline: JsonDict, gate: JsonDict) -> list[str]:
        findings: list[str] = []
        state = str(status.get("status") or "").strip()
        dataset = _first_text(status.get("dataset"), status.get("dataset_name"), status.get("dataset_slug"))
        if not dataset:
            findings.append("No dataset is bound to this scenario.")
        try:
            dataset_count = int(status.get("dataset_count") or 0)
        except Exception:
            dataset_count = 0
        if dataset and dataset_count <= 0:
            findings.append("Dataset is bound but has no counted samples.")
        if state == "error" or status.get("error"):
            findings.append(f"Scenario status reports an error: {status.get('error') or 'unknown error'}.")
        active_jobs = pipeline.get("active_jobs") if isinstance(pipeline.get("active_jobs"), list) else []
        if active_jobs:
            findings.append(f"{len(active_jobs)} training job(s) are currently queued or running.")
        candidate = pipeline.get("candidate") if isinstance(pipeline.get("candidate"), dict) else {}
        prod = pipeline.get("prod") if isinstance(pipeline.get("prod"), dict) else {}
        if candidate and not prod:
            findings.append("A candidate model exists but no production alias is set.")
        if candidate and prod and str(candidate.get("version_id") or "") != str(prod.get("version_id") or ""):
            findings.append("Candidate and production point at different model versions; compare before promotion.")
        gate_status = _first_text(gate.get("gate_status"), (gate.get("ci_cd") or {}).get("gate_status") if isinstance(gate.get("ci_cd"), dict) else "")
        if gate_status and gate_status.lower() not in {"passed", "pass", "ok", "success"}:
            findings.append(f"Latest gate status is {gate_status}.")
        if not findings:
            findings.append("No obvious scenario blockers found from status, pipeline, and latest gate reads.")
        return findings

    def _scenario_checkup_summary(
        self,
        scenario: str,
        status: JsonDict,
        candidate: JsonDict,
        prod: JsonDict,
        gate: JsonDict,
        active_jobs: list[Any],
    ) -> str:
        bits = [f"Scenario {scenario} checkup"]
        state = str(status.get("status") or "").strip()
        dataset = _first_text(status.get("dataset"), status.get("dataset_name"), status.get("dataset_slug"))
        if state:
            bits.append(f"status={state}")
        if dataset:
            bits.append(f"dataset={dataset}")
        if candidate.get("version_id"):
            bits.append(f"candidate={candidate.get('version_id')}")
        if prod.get("version_id"):
            bits.append(f"prod={prod.get('version_id')}")
        gate_status = _first_text(gate.get("gate_status"), (gate.get("ci_cd") or {}).get("gate_status") if isinstance(gate.get("ci_cd"), dict) else "")
        if gate_status:
            bits.append(f"gate={gate_status}")
        if active_jobs:
            bits.append(f"active_jobs={len(active_jobs)}")
        return "; ".join(bits) + "."

    def _resolve_pipeline_model_ref(self, pipeline: JsonDict, ref: str) -> JsonDict:
        wanted = str(ref or "").strip()
        if not wanted:
            return {}
        lower = wanted.lower()
        if lower in {"candidate", "staging", "prod"}:
            entry = pipeline.get(lower)
            return dict(entry) if isinstance(entry, dict) else {}
        for alias in ("candidate", "staging", "prod"):
            entry = pipeline.get(alias)
            if not isinstance(entry, dict):
                continue
            if wanted in {
                str(entry.get("version_id") or ""),
                str(entry.get("run_version") or ""),
                str(entry.get("model_version_id") or ""),
            }:
                return dict(entry)
        for run in pipeline.get("runs") or []:
            if not isinstance(run, dict):
                continue
            if wanted in {
                str(run.get("version") or ""),
                str(run.get("run_version") or ""),
                str(run.get("version_id") or ""),
                str(run.get("model_version_id") or ""),
            }:
                return dict(run)
        return {}

    def _numeric_metrics(self, payload: JsonDict) -> dict[str, float]:
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else payload
        out: dict[str, float] = {}

        def walk(prefix: str, obj: Any) -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    next_prefix = f"{prefix}.{key}" if prefix else str(key)
                    walk(next_prefix, value)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                out[prefix] = float(obj)

        walk("", metrics)
        return out

    def _model_compare_summary(
        self,
        left_ref: str,
        left: JsonDict,
        right_ref: str,
        right: JsonDict,
        deltas: JsonDict,
    ) -> str:
        left_id = _first_text(left.get("version_id"), left.get("model_version_id"), left.get("run_version"), left_ref)
        right_id = _first_text(right.get("version_id"), right.get("model_version_id"), right.get("run_version"), right_ref)
        if not deltas:
            return f"Compared {left_ref} ({left_id}) to {right_ref} ({right_id}); no shared numeric metrics found."
        ranked = sorted(deltas.items(), key=lambda item: abs(float(item[1])), reverse=True)[:5]
        parts = [f"{key}={float(value):+.4g}" for key, value in ranked]
        return f"Compared {left_ref} ({left_id}) to {right_ref} ({right_id}); deltas: {', '.join(parts)}."

    def _get(self, path: str) -> JsonDict:
        if self._http_get is None:
            raise RuntimeError("MCP HTTP GET bridge is not available")
        payload = self._http_get(path)
        return dict(payload or {}) if isinstance(payload, dict) else {}

    def _post(self, path: str, payload: Optional[JsonDict] = None) -> JsonDict:
        if self._http_post is None:
            raise RuntimeError("MCP HTTP POST bridge is not available")
        data = self._http_post(path, dict(payload or {}) if payload is not None else None)
        return dict(data or {}) if isinstance(data, dict) else {}

    def _job_summary(self, job: JsonDict, result: JsonDict, gate: JsonDict) -> str:
        job_id = str(job.get("job_id") or "").strip()
        scenario = _first_text(job.get("scenario"), result.get("scenario"))
        state = str(job.get("state") or "").strip()
        bits = [f"Job {job_id or 'unknown'}"]
        if scenario:
            bits.append(f"for {scenario}")
        if state:
            bits.append(f"is {state}")
        if result.get("error"):
            bits.append(f"with error: {result.get('error')}")
        ci_cd = result.get("ci_cd") if isinstance(result.get("ci_cd"), dict) else {}
        gate_status = str(gate.get("gate_status") or ci_cd.get("gate_status") or "").strip()
        if gate_status:
            bits.append(f"gate={gate_status}")
        return " ".join(bits) + "."

    @staticmethod
    def _ok(tool: str, data: JsonDict, summary: str = "", *, ok: bool = True, error: str = "") -> JsonDict:
        return {"ok": bool(ok), "tool": tool, "data": data, "summary": summary, "error": str(error or "")}

    @staticmethod
    def _fail(tool: str, error: str, data: Optional[JsonDict] = None) -> JsonDict:
        return {"ok": False, "tool": tool, "data": dict(data or {}), "summary": "", "error": str(error or "")}

    @staticmethod
    def _resource_ok(uri: str, data: Any) -> JsonDict:
        return {"ok": True, "uri": uri, "data": data, "error": ""}

    @staticmethod
    def _resource_fail(uri: str, error: str) -> JsonDict:
        return {"ok": False, "uri": uri, "data": {}, "error": str(error or "")}


def parse_controlled_run_request(text: str, attachments: Optional[list[str]] = None) -> Optional[JsonDict]:
    """Parse an explicit Tacitus controlled-run request from composer text."""
    body = str(text or "")
    paths = [str(p or "").strip() for p in list(attachments or []) if str(p or "").strip()]
    dataset_dirs = [p for p in paths if Path(p).expanduser().is_dir()]
    lower = body.lower()
    has_dataset_marker = "@dataset" in lower or bool(dataset_dirs)
    if not has_dataset_marker:
        return None
    if not re.search(r"\b(run|train|update|launch|return results|start)\b", lower):
        return None

    scenario = _extract_marker_value(body, "scenario")
    dataset_value = _extract_marker_value(body, "dataset")
    source_path = dataset_dirs[0] if dataset_dirs else ""
    dataset = ""
    if dataset_value:
        possible = Path(dataset_value).expanduser()
        if possible.exists() and possible.is_dir():
            source_path = str(possible)
        else:
            dataset = dataset_value
    mode = "update" if re.search(r"\b(update|retrain)\b", lower) else "train"
    return {
        "scenario": scenario,
        "dataset": dataset,
        "source_path": source_path,
        "mode": mode,
        "bind_dataset": True,
    }


def _extract_marker_value(text: str, marker: str) -> str:
    pattern = rf"@{re.escape(marker)}(?:\s*[:=]\s*|\s+)(?:\"([^\"]+)\"|'([^']+)'|([^\s,;]+))"
    match = re.search(pattern, str(text or ""), flags=re.IGNORECASE)
    if not match:
        return ""
    return _first_text(*match.groups()).strip().rstrip(".,!?")


_TOOLS: tuple[TacitusMcpTool, ...] = (
    TacitusMcpTool(
        "scenario.list",
        "List available CV Ops scenarios before resolving, checking, or comparing models.",
        _object_schema({"query": _string_schema("Optional text filter for scenario fields.")}),
    ),
    TacitusMcpTool(
        "scenario.resolve",
        "Resolve a scenario name from explicit input or active CV Ops context.",
        _object_schema({"scenario": _string_schema(), "query": _string_schema()}),
    ),
    TacitusMcpTool(
        "dataset.resolve",
        "Resolve a managed dataset slug or local dataset folder.",
        _object_schema({"dataset": _string_schema(), "source_path": _string_schema(), "scenario": _string_schema()}),
    ),
    TacitusMcpTool(
        "dataset.read",
        "Read dataset catalog metadata, dataset detail, classes, and bounded associated files.",
        _object_schema(
            {
                "dataset": _string_schema("Dataset slug. Omit to list the dataset catalog."),
                "scenario": _string_schema("Optional scenario used to resolve the bound dataset."),
                "limit": {"type": "integer", "description": "Maximum associated entries to return, default 12."},
                "include_inventory": _bool_schema("Include bounded folder inventory."),
                "include_profile": _bool_schema("Include tabular profile when the dataset is CSV."),
            }
        ),
    ),
    TacitusMcpTool(
        "dataset.import_or_bind",
        "Import a local dataset folder and/or bind a dataset slug to a scenario.",
        _object_schema({"scenario": _string_schema(), "dataset": _string_schema(), "source_path": _string_schema(), "bind": _bool_schema()}),
    ),
    TacitusMcpTool(
        "dataset.tabular_fix",
        "Apply profile-driven cleaning ops to a tabular (CSV) dataset: drop_duplicate_rows, "
        "drop_columns, drop_high_missing_columns, drop_constant_columns, impute_missing. "
        "Use dataset.read with include_profile first to find issues, then fix them here.",
        _object_schema(
            {
                "dataset": _string_schema("Tabular dataset slug. Omit to resolve from scenario."),
                "scenario": _string_schema("Optional scenario used to resolve the bound dataset."),
                "ops": {
                    "type": "array",
                    "description": (
                        "Ordered list of cleaning ops. op is one of: drop_duplicate_rows, "
                        "drop_columns, drop_high_missing_columns, drop_constant_columns, "
                        "impute_missing (strategy mean|median|mode|zero|constant), "
                        "rename_columns ({rename:{old:new}}), coerce_numeric, "
                        "normalize ({method:minmax|zscore}), clip_outliers ({factor:1.5}), "
                        "filter_rows ({where_col, where_op: ==|!=|>|>=|<|<=|contains|missing|not_missing, where_value}), "
                        "balance_classes ({label_col, strategy: oversample|undersample, max_ratio}). "
                        "Each op may carry columns?, threshold_pct?, strategy?, fill_value?."
                    ),
                    "items": {"type": "object"},
                },
                "op": _string_schema("Single op shorthand when ops is omitted."),
                "columns": {"type": "array", "description": "Columns for the single-op shorthand.", "items": {"type": "string"}},
                "strategy": _string_schema("Imputation strategy for the single-op shorthand."),
                "csv_name": _string_schema("CSV file name within a directory dataset (optional)."),
            }
        ),
    ),
    TacitusMcpTool(
        "dataset.tabular_split",
        "Write reproducible, optionally stratified train/val/test split assignments for a "
        "tabular dataset (sibling <slug>.splits.json; optional split column).",
        _object_schema(
            {
                "dataset": _string_schema("Tabular dataset slug. Omit to resolve from scenario."),
                "scenario": _string_schema("Optional scenario used to resolve the bound dataset."),
                "val_frac": {"type": "number", "description": "Validation fraction, default 0.2."},
                "test_frac": {"type": "number", "description": "Test fraction, default 0.0."},
                "stratify_col": _string_schema("Column to stratify on (preserves class proportions)."),
                "seed": {"type": "integer", "description": "Random seed, default 42."},
                "write_column": _bool_schema("Also append a `split` column to the CSV."),
                "csv_name": _string_schema("CSV file name within a directory dataset (optional)."),
            }
        ),
    ),
    TacitusMcpTool(
        "dataset.tabular_history",
        "Read the provenance log of cleaning transforms applied to a tabular dataset "
        "(revision, timestamp, ops, before/after, backup) and whether an undo is available.",
        _object_schema(
            {
                "dataset": _string_schema("Tabular dataset slug. Omit to resolve from scenario."),
                "scenario": _string_schema("Optional scenario used to resolve the bound dataset."),
                "csv_name": _string_schema("CSV file name within a directory dataset (optional)."),
            }
        ),
    ),
    TacitusMcpTool(
        "dataset.tabular_undo",
        "Undo the most recent tabular transform by restoring its backup (single-level undo).",
        _object_schema(
            {
                "dataset": _string_schema("Tabular dataset slug. Omit to resolve from scenario."),
                "scenario": _string_schema("Optional scenario used to resolve the bound dataset."),
                "csv_name": _string_schema("CSV file name within a directory dataset (optional)."),
            }
        ),
    ),
    TacitusMcpTool(
        "dataset.tabular_target",
        "Analyze a tabular target column: task type (classification/regression), class "
        "balance, target-leakage flags, and a train-readiness gate (blockers + warnings).",
        _object_schema(
            {
                "dataset": _string_schema("Tabular dataset slug. Omit to resolve from scenario."),
                "scenario": _string_schema("Optional scenario used to resolve the bound dataset."),
                "label_col": _string_schema("Target/label column to analyze."),
                "feature_cols": {
                    "type": "array",
                    "description": "Feature columns (default: all non-label columns).",
                    "items": {"type": "string"},
                },
                "csv_name": _string_schema("CSV file name within a directory dataset (optional)."),
            },
            ("label_col",),
        ),
    ),
    TacitusMcpTool(
        "dataset.tabular_score",
        "Batch-score a tabular dataset against a trained tabular model (resolved from a "
        "scenario + optional version, or an explicit model_path). Optionally writes a new "
        "dataset with a 'prediction' column.",
        _object_schema(
            {
                "dataset": _string_schema("Input tabular dataset slug to score."),
                "scenario": _string_schema("Scenario to resolve the model from."),
                "version": _string_schema("'', 'candidate', 'prod', or an explicit run version."),
                "model_path": _string_schema("Explicit path to a model.pkl artifact (overrides scenario)."),
                "write_dataset": _bool_schema("Store predictions as a new tabular dataset."),
                "output_name": _string_schema("Slug stem for the written dataset."),
                "csv_name": _string_schema("CSV file name within a directory dataset (optional)."),
            }
        ),
    ),
    TacitusMcpTool(
        "dataset.tabular_import_folder",
        "Batch-import every supported tabular file (csv/tsv/xlsx/xls/parquet/pq/json/jsonl) "
        "in a local folder as datasets, normalizing each to CSV.",
        _object_schema(
            {
                "source_path": _string_schema("Local folder to scan for tabular files."),
                "recursive": _bool_schema("Recurse into subfolders."),
            },
            ("source_path",),
        ),
    ),
    TacitusMcpTool(
        "pipeline.get",
        "Read per-scenario CI/CD pipeline policy and current registry aliases.",
        _object_schema({"scenario": _string_schema()}, ("scenario",)),
    ),
    TacitusMcpTool(
        "scenario.checkup",
        "Read scenario status, pipeline aliases, active jobs, and latest gate to summarize blockers.",
        _object_schema({"scenario": _string_schema(), "version": _string_schema()}),
    ),
    TacitusMcpTool(
        "model.compare",
        "Compare two model versions or aliases for a scenario using shared numeric metrics.",
        _object_schema(
            {
                "scenario": _string_schema(),
                "left": _string_schema("Left model alias or version, default candidate."),
                "right": _string_schema("Right model alias or version, default prod."),
            },
            ("scenario",),
        ),
    ),
    TacitusMcpTool(
        "run.launch",
        "Launch a controlled train/update run for a scenario.",
        _object_schema({"scenario": _string_schema(), "mode": _string_schema(), "record": _bool_schema()}, ("scenario",)),
    ),
    TacitusMcpTool(
        "job.status",
        "Read a CV Ops job status and optional final result.",
        _object_schema({"job_id": _string_schema(), "include_result": _bool_schema()}, ("job_id",)),
    ),
    TacitusMcpTool(
        "gate.get",
        "Read a CI/CD gate report for a scenario run.",
        _object_schema({"scenario": _string_schema(), "version": _string_schema(), "job_id": _string_schema()}),
    ),
    TacitusMcpTool(
        "artifact.record",
        "Attach an artifact to the active Tacitus chat/project ledger.",
        _object_schema({"label": _string_schema(), "path": _string_schema(), "kind": _string_schema(), "metadata": {"type": "object"}}),
    ),
    TacitusMcpTool(
        "experiment.record",
        "Record a reusable experiment hypothesis, result, checkpoint, and next-step evidence in the project ledger.",
        _object_schema(
            {
                "experiment_id": _string_schema("Stable ID for upserting this experiment note."),
                "scenario": _string_schema(),
                "dataset": _string_schema(),
                "run_version": _string_schema(),
                "model_version": _string_schema(),
                "checkpoint_path": _string_schema(),
                "hypothesis": _string_schema(),
                "knob": _string_schema("Controlled variable, for example source diversity."),
                "evidence": {"type": "array", "items": {"type": "string"}},
                "outcome": _string_schema(),
                "reuse_notes": _string_schema(),
                "next_experiment": _string_schema(),
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object"},
            }
        ),
    ),
    TacitusMcpTool(
        "experiment.search",
        "Search reusable experiment records and artifacts from the active project ledger.",
        _object_schema(
            {
                "query": _string_schema("Text to match against hypothesis, outcome, checkpoint, and reuse notes."),
                "scenario": _string_schema(),
                "dataset": _string_schema(),
                "tags": {"type": "array", "items": {"type": "string"}},
                "limit": {"type": "integer", "description": "Maximum matches, default 8."},
            }
        ),
    ),
    TacitusMcpTool(
        "promotion.request",
        "Request or explicitly confirm promotion of a passing candidate.",
        _object_schema({"scenario": _string_schema(), "version": _string_schema(), "reason": _string_schema(), "confirmed": _bool_schema()}),
    ),
)


_RESOURCES: tuple[TacitusMcpResource, ...] = (
    TacitusMcpResource("tacitus://active_project", "active project", "Current Tacitus notes project."),
    TacitusMcpResource("tacitus://active_scenario", "active scenario", "Current CV Ops scenario selection."),
    TacitusMcpResource("tacitus://selected_dataset", "selected dataset", "Dataset bound to the active scenario when known."),
    TacitusMcpResource("tacitus://model_registry_entry", "model registry entry", "Candidate or production model registry entry."),
    TacitusMcpResource("tacitus://run_artifacts", "run artifacts", "Job result, gate report, and run artifact summary."),
    TacitusMcpResource("tacitus://events_artifacts", "events and artifacts", "Chat/project Tacitus Events & artifacts ledger."),
    TacitusMcpResource("tacitus://ingested_memory", "ingested memory", "Summary of knowledge transferred from other AI chat exports."),
)
