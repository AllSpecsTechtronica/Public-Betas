"""Repair W3C PROV graph in provenance.db from lineages.db + snapshots.db.

Use after upgrading CV Ops, deleting provenance.db, or if lineage rows predate
provenance mirroring. Does not create lineages from train jobs or model_registry.

From the Insight package root:

    python -m insight_local.cvops.backfill_provenance
    python -m insight_local.cvops.backfill_provenance --lineage-id line-abc123
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ..config import ROOT_DIR
from .lineage_store import LineageStore
from .provenance_store import ProvenanceStore
from .snapshot_store import SnapshotStore

_DEFAULT_STATE = ROOT_DIR / "state" / "insight_local" / "cvops"


def run_backfill(
    *,
    state_dir: Path,
    lineage_id: str = "",
) -> dict[str, object]:
    """Replay local lineages into provenance.db. Returns summary counts."""
    state_dir = Path(state_dir).resolve()
    lineages = LineageStore(state_dir / "lineages.db")
    snapshots = SnapshotStore(state_dir / "snapshots.db", state_dir / "snapshot_weights")
    provenance = ProvenanceStore(state_dir / "provenance.db")
    try:
        before = provenance.graph_counts()
        lid = str(lineage_id or "").strip()
        local_lineages = lineages.list_lineages()
        if lid:
            if lineages.get_lineage(lid) is None:
                raise SystemExit(f"lineage not found in {state_dir / 'lineages.db'}: {lid}")
            provenance.backfill_lineage(lid, lineages, snapshots)
            lineages_processed = 1
        else:
            lineages_processed = provenance.backfill_all(lineages, snapshots)
        after = provenance.graph_counts()
        drops_total = 0
        for rec in local_lineages:
            if lid and rec.lineage_id != lid:
                continue
            drops_total += len(lineages.list_drops(rec.lineage_id))
        return {
            "status": "ok",
            "state_dir": str(state_dir),
            "lineage_id": lid or None,
            "local_lineages": len(local_lineages),
            "lineages_processed": lineages_processed,
            "drops_in_scope": drops_total,
            "prov_nodes_before": before["prov_nodes"],
            "prov_edges_before": before["prov_edges"],
            "prov_nodes_after": after["prov_nodes"],
            "prov_edges_after": after["prov_edges"],
        }
    finally:
        lineages.close()
        snapshots.close()
        provenance.close()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Backfill provenance.db from local lineages and snapshots.",
    )
    p.add_argument(
        "--state-dir",
        type=Path,
        default=_DEFAULT_STATE,
        help=f"CV Ops state directory (default: {_DEFAULT_STATE})",
    )
    p.add_argument(
        "--lineage-id",
        default="",
        help="Backfill one lineage only (default: all local lineages)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = run_backfill(state_dir=args.state_dir, lineage_id=args.lineage_id)
    print("[cvops] provenance backfill complete")
    print(f"  state_dir: {summary['state_dir']}")
    if summary.get("lineage_id"):
        print(f"  lineage_id: {summary['lineage_id']}")
    print(f"  local_lineages: {summary['local_lineages']}")
    print(f"  lineages_processed: {summary['lineages_processed']}")
    print(f"  drops_in_scope: {summary['drops_in_scope']}")
    print(
        "  prov_nodes: "
        f"{summary['prov_nodes_before']} -> {summary['prov_nodes_after']}"
    )
    print(
        "  prov_edges: "
        f"{summary['prov_edges_before']} -> {summary['prov_edges_after']}"
    )
    if int(summary["local_lineages"] or 0) == 0:
        print(
            "[cvops] note: no rows in lineages.db — create lineages via "
            "Continuous Learning or POST /lineages first."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
