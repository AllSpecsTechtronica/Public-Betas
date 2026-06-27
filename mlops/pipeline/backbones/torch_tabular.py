"""torch_tabular.py — ML backbone: custom Python/PyTorch logic on tabular/signal data.

This backbone executes a list of "cells" (Colab-style) for both training and
inference. For tabular workflows we support two configuration styles:

1) **File-based cells (recommended)** — scenario YAML:

      backbone_config:
        cells:
          - path: mlops/algos/my_train_cell.py
          - path: mlops/algos/metrics_cell.py

   Each referenced file should export a callable named ``run`` (default) with
   signature ``run(ctx, prev)`` (ctx-only and no-arg also supported). The return
   value may be:
     - ``CellResult`` (advanced), or
     - ``dict`` (interpreted as ``data``), or
     - ``str`` (interpreted as ``output``), or
     - ``None`` (treated as success).

   Optional per-job overrides:
     - ``train_cells``: list of cell specs for job_type == "train"
     - ``infer_cells``: list of cell specs for job_type == "infer"

2) **Legacy cells module** — scenario YAML:

      backbone_config:
        cells_module: mlops.scenarios.my_model_cells

   That module exports ``TRAIN_CELLS`` / ``INFER_CELLS`` (list[BackboneCell]).

Each cell receives ``BackboneContext``; the scenario's backbone_config is
available on ``ctx.scenario_config.backbone_config``.
"""
from __future__ import annotations

import importlib
from typing import Any

from ..backbone import BackboneBase, BackboneCell, BackboneContext, CellResult
from ..python_cells import PythonFileCell, parse_cell_specs


class _MissingCellsConfigCell(BackboneCell):
    name = "Configure Cells"
    description = "No tabular cells configured — cannot proceed"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        cfg = ctx.scenario_config.backbone_config
        scen = ctx.scenario_config.name
        msg = (
            f"Scenario '{scen}' uses backbone_type: torch_tabular but "
            f"no runnable cells are configured.\n"
            f"Current backbone_config: {cfg}\n\n"
            f"Recommended: set backbone_config.cells to a list of Python files "
            f"that export a 'run' function.\n\n"
            f"  backbone_config:\n"
            f"    cells:\n"
            f"      - path: mlops/algos/{scen}_train.py\n"
            f"    dataset_csv: database/my_data.csv\n"
            f"    num_classes: 5\n"
            f"    epochs: 100"
        )
        print(msg)
        return CellResult(
            cell_name=self.name,
            status="error",
            output=msg,
            elapsed_ms=0,
        )


class TorchTabularBackbone(BackboneBase):
    """Tabular backbone that runs configured cells (file-based or legacy module-based)."""

    backbone_type = "torch_tabular"

    def __init__(self, config: Any) -> None:
        self._config = config
        self._cells: list[BackboneCell] | None = None
        self._job_type: str = "infer"

    # cells is resolved lazily so job_type is known at run() time.
    @property
    def cells(self) -> list[BackboneCell]:
        if self._cells is not None:
            return self._cells
        backbone_cfg = self._config.backbone_config or {}
        # Prefer file-based cells config.
        if self._job_type == "train":
            raw = backbone_cfg.get("train_cells")
            specs = parse_cell_specs(raw) or parse_cell_specs(backbone_cfg.get("cells"))
        else:
            raw = backbone_cfg.get("infer_cells")
            specs = parse_cell_specs(raw) or parse_cell_specs(backbone_cfg.get("cells"))
        if specs:
            self._cells = [PythonFileCell(s) for s in specs]
            return self._cells

        # Legacy: import a cells module that exports TRAIN_CELLS / INFER_CELLS.
        cells_module_path = str(backbone_cfg.get("cells_module") or "").strip()
        if not cells_module_path:
            self._cells = [_MissingCellsConfigCell()]
            return self._cells
        try:
            mod = importlib.import_module(cells_module_path)
        except ImportError as exc:
            raise ImportError(
                f"Cannot import cells_module '{cells_module_path}': {exc}"
            ) from exc
        if self._job_type == "train":
            raw_cells = getattr(mod, "TRAIN_CELLS", None)
            attr = "TRAIN_CELLS"
        else:
            raw_cells = getattr(mod, "INFER_CELLS", None)
            attr = "INFER_CELLS"
        if raw_cells is None:
            raise AttributeError(
                f"cells_module '{cells_module_path}' does not export '{attr}'"
            )
        if not isinstance(raw_cells, list):
            raise TypeError(
                f"'{attr}' in '{cells_module_path}' must be a list[BackboneCell]"
            )
        self._cells = raw_cells
        return self._cells

    def run(self, ctx: BackboneContext) -> dict[str, Any]:
        # Set job_type so cells property resolves the right list.
        self._job_type = ctx.job_type
        self._cells = None  # force re-resolution with correct job_type
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

        # Collect any result data the cells chose to expose.
        merged_data: dict[str, Any] = {}
        for r in cell_results:
            if isinstance(r.data, dict):
                merged_data.update(r.data)

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
            "backbone_data": {
                k: v for k, v in merged_data.items()
                if k not in {"signal", "detections", "overlay_image", "weights", "weights_path", "model_version"}
            },
        }
