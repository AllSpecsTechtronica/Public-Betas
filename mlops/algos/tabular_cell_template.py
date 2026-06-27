"""Template for a tabular "execution cell" used by the torch_tabular backbone.

Usage (scenario YAML):

  backbone_type: torch_tabular
  backbone_config:
    cells:
      - path: mlops/algos/tabular_cell_template.py

This file is meant to be copied/edited. The backbone reloads the file on each
run, so you can iterate quickly and re-kick training.
"""

from __future__ import annotations

from typing import Any

from mlops.pipeline.backbone import BackboneContext


def run(ctx: BackboneContext, prev: list[Any]) -> dict[str, Any] | None:
    cfg = ctx.scenario_config
    bcfg = getattr(cfg, "backbone_config", {}) or {}
    dataset_slug = str(getattr(cfg, "dataset", "") or "")
    dataset_csv = str(bcfg.get("dataset_csv") or "")

    print(f"[cell] scenario={cfg.name} job_type={ctx.job_type}")
    print(f"[cell] dataset_slug={dataset_slug} dataset_csv={dataset_csv}")
    print(f"[cell] previous_cells={len(prev)}")

    # Do training / evaluation here. You can:
    # - print progress (captured in UI)
    # - return {"data": {...}} to pass structured values to later cells
    return {"data": {"template_ran": True}}

