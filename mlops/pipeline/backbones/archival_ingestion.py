"""archival_ingestion backbone placeholder.

Archive execution is orchestrated through CV Ops archive job routes rather than
the generic train/infer path. This backbone exists so archival scenarios are
first-class registry entries and fail clearly if invoked through the wrong
endpoint.
"""
from __future__ import annotations

from typing import Any

from ..backbone import BackboneBase, BackboneCell, BackboneContext, CellResult


class _ArchiveRouteHintCell(BackboneCell):
    name = "Archive Job Router"
    description = "Archival ingestion runs through archive phase jobs"

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        msg = (
            "archival_ingestion scenarios do not run through generic train/infer jobs.\n"
            "Use the /archives import + /archives/{corpus_id}/jobs endpoints or the Data Viz archival mode."
        )
        print(msg)
        return CellResult(
            cell_name=self.name,
            status="error",
            output=msg,
            elapsed_ms=0.0,
        )


class ArchivalIngestionBackbone(BackboneBase):
    backbone_type = "archival_ingestion"

    def __init__(self, config: Any) -> None:
        self._config = config

    @property
    def cells(self) -> list[BackboneCell]:
        return [_ArchiveRouteHintCell()]

    def _build_result(
        self,
        ctx: BackboneContext,
        cell_results: list[CellResult],
    ) -> dict[str, Any]:
        error = ""
        if cell_results:
            error = str(cell_results[-1].output or "")
        return {
            "scenario": ctx.scenario_config.name,
            "summary": "",
            "error": error or "archival_ingestion requires archive routes",
            "artifact_policy": "path_only",
            "backbone_type": self.backbone_type,
            "result_path": "",
            "weights": "",
        }
