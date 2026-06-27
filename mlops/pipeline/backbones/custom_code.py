"""custom_code backbone — Colab-style Python cells with ``ctx.datasets`` / ``ctx.active_cell``."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import registry as _reg
from ..backbone import BackboneBase, BackboneCell, BackboneContext, CellResult
from ..python_cells import PythonFileCell, parse_cell_specs, resolve_repo_path


def _resolve_dataset_entry(ref: dict[str, Any], scenario_name: str) -> dict[str, Any]:
    out = dict(ref)
    path = str(out.get("path") or "").strip()
    resolved = ""
    if path:
        p = resolve_repo_path(path)
        try:
            resolved = str(p.resolve())
        except Exception:
            resolved = str(p)
    out["resolved_path"] = resolved
    out["scenario"] = scenario_name
    return out


def _active_cell_payload(scenario: str, spec: dict[str, Any]) -> dict[str, Any]:
    cell_id = str(spec.get("id") or "").strip() or _sanitize_for_path(str(spec.get("name") or "cell"))
    rel = f"mlops/custom_cells/{scenario}/draft/data/{cell_id}"
    pasted_dir = resolve_repo_path(rel)
    raw_ds = spec.get("datasets")
    cell_datasets: list[dict[str, Any]] = []
    if isinstance(raw_ds, list):
        for d in raw_ds:
            if isinstance(d, dict):
                cell_datasets.append(_resolve_dataset_entry(d, scenario))
    return {
        "id": cell_id,
        "name": str(spec.get("name") or ""),
        "path": str(spec.get("path") or ""),
        "entry": str(spec.get("entry") or "run"),
        "datasets": cell_datasets,
        "pasted_data_dir": str(pasted_dir),
    }


def _sanitize_for_path(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name or "").strip())[:64] or "cell"


class _CustomCodePythonCell(BackboneCell):
    """Runs a file-backed cell after wiring ``ctx.datasets`` and ``ctx.active_cell``."""

    def __init__(self, spec: dict[str, Any], scenario_datasets: list[dict[str, Any]], scenario_name: str) -> None:
        self._spec = dict(spec)
        self._scenario_datasets = [dict(x) for x in scenario_datasets if isinstance(x, dict)]
        self._scenario_name = scenario_name
        py_spec = {
            "path": str(spec.get("path") or ""),
            "entry": str(spec.get("entry") or "run"),
            "name": str(spec.get("name") or Path(str(spec.get("path") or "")).stem or "cell"),
            "description": str(spec.get("description") or ""),
        }
        self._inner = PythonFileCell(py_spec)
        self.name = self._inner.name
        self.description = self._inner.description

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        ctx.datasets = [_resolve_dataset_entry(d, self._scenario_name) for d in self._scenario_datasets]
        ctx.active_cell = _active_cell_payload(self._scenario_name, self._spec)
        try:
            return self._inner.run(ctx, prev)
        except AttributeError as exc:
            # Colab-style: the cell file has no `run(ctx, prev)` entrypoint, just
            # top-level statements. The module body already executed inside
            # PythonFileCell.run via load_module_from_file, so its stdout is
            # already in BackboneBase's redirect_stdout buffer. Treat as done.
            if "does not export entrypoint" not in str(exc):
                raise
            return CellResult(
                cell_name=self._inner.name,
                status="done",
                output="",
                elapsed_ms=0,
                data={},
            )


class _MissingCustomCellsCell(BackboneCell):
    name = "Configure Custom Cells"
    description = "No custom_code cells configured"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        scen = ctx.scenario_config.name
        msg = (
            f"Scenario '{scen}' uses backbone_type: custom_code but no runnable cells are configured.\n"
            f"Use the Custom Cells editor to save a draft, or set backbone_config.cells."
        )
        print(msg)
        return CellResult(cell_name=self.name, status="error", output=msg, elapsed_ms=0)


def _next_run_dir(models_root: Path) -> Path:
    runs = [p for p in models_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
    if not runs:
        return models_root / "v1"
    latest = max(int(p.name[1:]) for p in runs)
    return models_root / f"v{latest + 1}"


def _ensure_default_run_dir(cfg: Any, merged_data: dict[str, Any]) -> dict[str, Any]:
    """If cells did not populate ``result_path``, create a versioned run with metrics."""
    if str(merged_data.get("result_path") or "").strip():
        return merged_data
    models_root = _reg.MLOPS_ROOT / "models" / str(cfg.name)
    models_root.mkdir(parents=True, exist_ok=True)
    run_dir = _next_run_dir(models_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "weights.pth"
    weights_path.write_bytes(b"\0" * 64)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "backbone_type": "custom_code",
        "scenario": str(cfg.name),
        "summary": str((merged_data.get("signal") or {}).get("summary") or "custom_code run"),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8")
    try:
        rel_run = run_dir.resolve().relative_to(_reg.REPO_ROOT).as_posix()
        rel_weights = weights_path.resolve().relative_to(_reg.REPO_ROOT).as_posix()
    except Exception:
        rel_run = str(run_dir.resolve())
        rel_weights = str(weights_path.resolve())
    out = dict(merged_data)
    out["result_path"] = rel_run
    out["weights"] = rel_weights
    out.setdefault("weights_path", rel_weights)
    return out


class CustomCodeBackbone(BackboneBase):
    backbone_type = "custom_code"

    def __init__(self, config: Any) -> None:
        self._config = config
        self._cells: list[BackboneCell] | None = None
        self._job_type: str = "infer"

    @property
    def cells(self) -> list[BackboneCell]:
        if self._cells is not None:
            return self._cells
        backbone_cfg = self._config.backbone_config or {}
        scen = str(self._config.name or "")
        scenario_ds_raw = backbone_cfg.get("datasets")
        scenario_datasets = [dict(x) for x in scenario_ds_raw if isinstance(x, dict)] if isinstance(
            scenario_ds_raw, list
        ) else []

        if self._job_type == "train":
            raw = backbone_cfg.get("train_cells")
            specs = parse_cell_specs(raw) or parse_cell_specs(backbone_cfg.get("cells"))
        else:
            raw = backbone_cfg.get("infer_cells")
            specs = parse_cell_specs(raw) or parse_cell_specs(backbone_cfg.get("cells"))

        if specs:
            self._cells = [_CustomCodePythonCell(s, scenario_datasets, scen) for s in specs]
            return self._cells

        cells_module_path = str(backbone_cfg.get("cells_module") or "").strip()
        if cells_module_path:
            import importlib

            mod = importlib.import_module(cells_module_path)
            attr = "TRAIN_CELLS" if self._job_type == "train" else "INFER_CELLS"
            raw_cells = getattr(mod, attr, None)
            if raw_cells is None:
                raise AttributeError(f"cells_module '{cells_module_path}' missing {attr}")
            if not isinstance(raw_cells, list):
                raise TypeError(f"{attr} must be a list[BackboneCell]")
            self._cells = raw_cells
            return self._cells

        self._cells = [_MissingCustomCellsCell()]
        return self._cells

    def run(self, ctx: BackboneContext) -> dict[str, Any]:
        self._job_type = ctx.job_type
        self._cells = None
        return super().run(ctx)

    def _build_result(
        self,
        ctx: BackboneContext,
        cell_results: list[CellResult],
    ) -> dict[str, Any]:
        error = ""
        for r in cell_results:
            if r.status == "error":
                error = r.output or f"Cell '{r.cell_name}' failed"
                break

        merged_data: dict[str, Any] = {}
        for r in cell_results:
            if isinstance(r.data, dict):
                merged_data.update(r.data)

        if not error:
            merged_data = _ensure_default_run_dir(ctx.scenario_config, merged_data)

        signal = merged_data.get("signal")
        if not isinstance(signal, dict):
            signal = {"flag": bool(error), "summary": error or "completed", "metrics": {}}

        return {
            "scenario": ctx.scenario_config.name,
            "model_version": str(merged_data.get("model_version") or ""),
            "weights": str(merged_data.get("weights") or merged_data.get("weights_path") or ""),
            "summary": str(signal.get("summary") or ""),
            "detections": merged_data.get("detections") or [],
            "elapsed_ms": sum(r.elapsed_ms for r in cell_results),
            "overlay_image": str(merged_data.get("overlay_image") or ""),
            "signal": signal,
            "error": error,
            "artifact_policy": "inline_overlay_optional",
            "result_path": str(merged_data.get("result_path") or ""),
            "backbone_data": {
                k: v for k, v in merged_data.items()
                if k not in {"signal", "detections", "overlay_image", "weights", "weights_path", "model_version"}
            },
        }
