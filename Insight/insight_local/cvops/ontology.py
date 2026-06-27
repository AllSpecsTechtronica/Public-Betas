"""Cross-store entity graph builder for the Ontology Surface (Ecosystem panel).

Returns a serializable {nodes, edges} dict that powers the Cytoscape.js graph.
Node id format: "{type}:{entity_id}"

Node types: scenario, backbone, cell, dataset, model_version, job,
            dataset_snapshot, model_snapshot, lineage, range, catalog_asset,
            prov_activity, prov_agent, prov_entity,
            identity, correction_event, sector, collection

Edge types: belongs_to, uses_backbone, contains_cell, trains_on,
            governed_by, produces, branched_from, has_head, derived_from,
            evaluates, prov_generates, prov_used, prov_informed_by,
            prov_associated, had_member, specialization_of, prov_invalidated,
            prov_attributed,
            flagged_in, flagged_by_model, contains_sector, organized_in,
            catalogued_in, shares_backbone_with, shares_dataset_with
"""
from __future__ import annotations

import json
import sqlite3
import sys
from base64 import b64encode
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .catalog_store import CatalogStore
    from .provenance_store import ProvenanceStore

from ..config import GALLERY_DB_PATH, ROOT_DIR
from .corrections_store import load_corrections

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mlops.pipeline import registry as mlops_registry
from mlops.pipeline.model_registry import _load_registry as _load_model_registry
from mlops.pipeline.custom_cells_store import read_draft
from mlops.pipeline.governance import DATASET_REGISTRY_DIR

from .jobs import JobStore
from .snapshot_store import SnapshotStore
from .lineage_store import LineageStore
from .range_store import RangeStore
from .catalog_store import CatalogStore


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _node(ntype: str, nid: str, label: str, meta: dict[str, Any]) -> dict[str, Any]:
    return {"id": f"{ntype}:{nid}", "type": ntype, "label": label, "meta": meta}


def _edge(src: str, tgt: str, etype: str) -> dict[str, Any]:
    return {"source": src, "target": tgt, "type": etype}


# ---------------------------------------------------------------------------
# Unified relationship graph extension builders
# ---------------------------------------------------------------------------

def _build_identity_nodes(gallery_db_path: Path) -> tuple[list[dict], list[dict]]:
    """Read face identities from gallery.db. Returns (nodes, edges)."""
    nodes: list[dict] = []
    edges: list[dict] = []
    try:
        if not Path(str(gallery_db_path)).exists():
            return nodes, edges
        uri = f"file:{gallery_db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT MIN(i.id) AS identity_id, i.name, i.group_name,
                       MIN(i.source_path) AS source_path, COUNT(e.id) AS face_count
                FROM identities i
                JOIN embeddings e ON e.identity_id = i.id
                GROUP BY i.name, i.group_name
                ORDER BY i.name
                LIMIT 200
                """
            ).fetchall()
            for row in rows:
                iid = str(int(row["identity_id"]))
                name = str(row["name"])
                thumb_b64 = ""
                try:
                    thumb_row = conn.execute(
                        "SELECT face_png FROM embeddings WHERE identity_id = ? "
                        "AND face_png IS NOT NULL AND length(face_png) > 0 LIMIT 1",
                        (int(row["identity_id"]),),
                    ).fetchone()
                    if thumb_row and thumb_row["face_png"]:
                        thumb_b64 = b64encode(bytes(thumb_row["face_png"])).decode("ascii")
                except Exception:
                    pass
                nodes.append(_node("identity", iid, name, {
                    "group": str(row["group_name"] or ""),
                    "face_count": int(row["face_count"]),
                    "source_path": str(row["source_path"] or ""),
                    "thumbnail_b64": thumb_b64,
                }))
        finally:
            conn.close()
    except Exception:
        pass
    return nodes, edges


def _build_correction_nodes(
    snap_weights: dict[str, str],
    asset_paths: dict[str, list[str]],
) -> tuple[list[dict], list[dict]]:
    """Load human corrections and link them to catalog assets and model snapshots.

    snap_weights: snapshot_id → weights_uri
    asset_paths:  asset_id   → [source_uri, managed_path]
    Returns (nodes, edges).
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    try:
        corrections = load_corrections()[-50:]
        for c in corrections:
            nodes.append(_node("correction_event", c.id, f"corr:{c.id[:8]}", {
                "video_path": c.video_path,
                "frame_ts_ms": c.frame_ts_ms,
                "created_at": c.created_at,
                "kind": c.kind or "",
                "model_path": c.model_path,
                "label_count": len(c.ground_truth),
            }))
            cnode_id = f"correction_event:{c.id}"
            # flagged_in: match correction video_path → catalog_asset
            vpath = str(c.video_path or "").strip()
            if vpath:
                for aid, paths in asset_paths.items():
                    if any(
                        p and (p == vpath or vpath.endswith(p.lstrip("/")) or p.endswith(vpath.lstrip("/")))
                        for p in paths
                    ):
                        edges.append(_edge(cnode_id, f"catalog_asset:{aid}", "flagged_in"))
                        break
            # flagged_by_model: match correction model_path → model_snapshot weights_uri
            mpath = str(c.model_path or "").strip()
            if mpath:
                for snap_id, wuri in snap_weights.items():
                    if wuri and (
                        mpath == wuri
                        or mpath.endswith(wuri.lstrip("/"))
                        or wuri.endswith(mpath.lstrip("/"))
                    ):
                        edges.append(_edge(cnode_id, f"model_snapshot:{snap_id}", "flagged_by_model"))
                        break
    except Exception:
        pass
    return nodes, edges


def _build_sector_collection_nodes(
    catalog: "CatalogStore",
    asset_collection_map: dict[str, str],
) -> tuple[list[dict], list[dict]]:
    """Add catalog sector and collection hierarchy to the graph.

    asset_collection_map: asset_id → collection_id (from existing catalog_asset nodes)
    Returns (nodes, edges).
    """
    nodes: list[dict] = []
    edges: list[dict] = []
    try:
        for sec in catalog.list_sectors():
            if sec.path == "/":
                continue
            nodes.append(_node("sector", sec.sector_id, sec.name, {
                "path": sec.path,
                "parent_id": sec.parent_id,
            }))
            if sec.parent_id and sec.parent_id != "sector-root":
                edges.append(_edge(f"sector:{sec.parent_id}", f"sector:{sec.sector_id}", "contains_sector"))
    except Exception:
        pass
    try:
        for coll in catalog.list_collections():
            nodes.append(_node("collection", coll["collection_id"], coll["name"], {
                "source_type": coll["source_type"],
                "sector_path": coll["sector_path"],
                "description": coll["description"],
            }))
            edges.append(_edge(
                f"collection:{coll['collection_id']}",
                f"sector:{coll['sector_id']}",
                "organized_in",
            ))
    except Exception:
        pass
    for aid, cid in asset_collection_map.items():
        if cid:
            edges.append(_edge(f"catalog_asset:{aid}", f"collection:{cid}", "catalogued_in"))
    return nodes, edges


def _build_cross_entity_edges(nodes: list[dict]) -> list[dict]:
    """Emit shared-resource edges between scenarios and lineages.

    Scenarios sharing the same backbone_type get shares_backbone_with edges.
    Lineages sharing the same base_snapshot_id get shares_dataset_with edges.
    """
    edges: list[dict] = []
    backbone_to_scenarios: dict[str, list[str]] = {}
    base_to_lineages: dict[str, list[str]] = {}
    for n in nodes:
        ntype = n.get("type", "")
        meta = n.get("meta") or {}
        if ntype == "scenario":
            bt = str(meta.get("backbone_type") or "").strip()
            if bt:
                backbone_to_scenarios.setdefault(bt, []).append(n["id"])
        elif ntype == "lineage":
            bsid = str(meta.get("base_snapshot_id") or "").strip()
            if bsid:
                base_to_lineages.setdefault(bsid, []).append(n["id"])
    for sids in backbone_to_scenarios.values():
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                edges.append(_edge(sids[i], sids[j], "shares_backbone_with"))
    for lids in base_to_lineages.values():
        for i in range(len(lids)):
            for j in range(i + 1, len(lids)):
                edges.append(_edge(lids[i], lids[j], "shares_dataset_with"))
    return edges


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_graph(
    *,
    job_store: JobStore,
    snapshots: SnapshotStore,
    lineages: LineageStore,
    ranges: RangeStore,
    catalog: CatalogStore,
    entity_types: Optional[list[str]] = None,
    scenario: Optional[str] = None,
    depth: int = 2,
    since_ts: Optional[float] = None,
    job_limit: int = 150,
    extra_lineages: Optional[list[dict[str, Any]]] = None,
    provenance: Optional["ProvenanceStore"] = None,
) -> dict[str, Any]:
    """Build the cross-store entity graph.

    Returns {"nodes": [...], "edges": [...]}. Nodes already in the graph are
    deduplicated; edges are not (a pair can have multiple typed relationships).
    """
    want: Optional[set[str]] = set(entity_types) if entity_types else None

    def include(t: str) -> bool:
        return want is None or t in want

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add_node(n: dict[str, Any]) -> None:
        if n["id"] not in seen:
            seen.add(n["id"])
            nodes.append(n)

    # ---- scenarios -------------------------------------------------------
    scenario_names: list[str] = []
    try:
        reg = mlops_registry.load_registry()
        raw_scenarios = reg.get("scenarios") or []
    except Exception:
        raw_scenarios = []

    for item in raw_scenarios:
        if not isinstance(item, dict):
            continue
        sname = str(item.get("name") or "").strip()
        if not sname:
            continue
        if scenario and sname != scenario:
            continue
        scenario_names.append(sname)
        if not include("scenario"):
            continue
        try:
            cfg = mlops_registry.get_scenario_config(sname)
            add_node(_node("scenario", sname, cfg.display_name or sname, {
                "description": cfg.description,
                "backbone_type": cfg.backbone_type,
                "dataset": cfg.dataset,
                "enabled": item.get("enabled", True),
            }))
        except Exception:
            add_node(_node("scenario", sname, sname, {"enabled": item.get("enabled", True)}))

    # ---- backbones -------------------------------------------------------
    if include("backbone"):
        for sname in scenario_names:
            try:
                cfg = mlops_registry.get_scenario_config(sname)
                bt = cfg.backbone_type or "yolo_detection"
                bn_id = f"{sname}/{bt}"
                hp = cfg.hyperparams or {}
                add_node(_node("backbone", bn_id, bt, {
                    "scenario":    sname,
                    "num_classes": len(cfg.classes or []),
                    "imgsz":       hp.get("imgsz"),
                    "epochs":      hp.get("epochs"),
                    "base_model":  cfg.base_model or "",
                    "weights":     cfg.weights or "",
                }))
                if include("scenario"):
                    edges.append(_edge(f"scenario:{sname}", f"backbone:{bn_id}", "uses_backbone"))
            except Exception:
                pass

    # ---- datasets (from scenario config references) ----------------------
    if include("dataset"):
        _seen_ds: set[str] = set()
        for sname in scenario_names:
            try:
                cfg = mlops_registry.get_scenario_config(sname)
                ds_name = str(cfg.dataset or "").strip()
                if not ds_name:
                    continue
                if ds_name not in _seen_ds:
                    _seen_ds.add(ds_name)
                    add_node(_node("dataset", ds_name, ds_name, {
                        "dataset_path": str(cfg.dataset_path or ""),
                        "num_classes":  len(cfg.classes or []),
                        "classes":      (cfg.classes or [])[:20],
                    }))
                if include("scenario"):
                    edges.append(_edge(f"scenario:{sname}", f"dataset:{ds_name}", "trains_on"))
            except Exception:
                pass

    # ---- custom cells ----------------------------------------------------
    if include("cell"):
        for sname in scenario_names:
            try:
                draft = read_draft(sname)
                for cell in draft.get("cells") or []:
                    cid = str(cell.get("id") or cell.get("name") or "").strip()
                    if not cid:
                        continue
                    cnode_id = f"{sname}/{cid}"
                    add_node(_node("cell", cnode_id, cell.get("name") or cid, {
                        "scenario":    sname,
                        "entry":       cell.get("entry", ""),
                        "cell_type":   cell.get("cell_type", ""),
                        "description": cell.get("description", ""),
                        "enabled":     cell.get("enabled", True),
                    }))
                    try:
                        cfg = mlops_registry.get_scenario_config(sname)
                        bt = cfg.backbone_type or "yolo_detection"
                        bn_id = f"{sname}/{bt}"
                        edges.append(_edge(f"backbone:{bn_id}", f"cell:{cnode_id}", "contains_cell"))
                    except Exception:
                        pass
            except Exception:
                pass

    # ---- model versions --------------------------------------------------
    try:
        mreg = _load_model_registry()
    except Exception:
        mreg = {}

    model_snap_links: list[tuple[str, str]] = []  # (version_id, snapshot_id)

    if include("model_version"):
        for sname, sdata in (mreg.get("models") or {}).items():
            if not isinstance(sdata, dict):
                continue
            if scenario and sname != scenario:
                continue
            aliases_map: dict[str, str] = sdata.get("aliases") or {}
            for ver in sdata.get("versions") or []:
                if not isinstance(ver, dict):
                    continue
                vid = str(ver.get("version_id") or "").strip()
                if not vid:
                    continue
                run_ver = str(ver.get("run_version") or "").strip()
                metrics = ver.get("metrics") or {}
                alias_labels = [k for k, v in aliases_map.items() if v == vid]
                raw_m = metrics.get("raw") or {}
                add_node(_node("model_version", vid, f"{sname} v{run_ver}", {
                    "scenario":    sname,
                    "run_version": run_ver,
                    "status":      ver.get("status", "active"),
                    "created_at":  ver.get("created_at"),
                    "map50":       metrics.get("map50"),
                    "map50_95":    raw_m.get("metrics/mAP50-95(B)"),
                    "precision":   raw_m.get("metrics/precision(B)"),
                    "recall":      raw_m.get("metrics/recall(B)"),
                    "train_epoch": raw_m.get("epoch"),
                    "train_time_s": raw_m.get("time"),
                    "aliases":     alias_labels,
                }))
                if include("scenario") and sname in scenario_names:
                    edges.append(_edge(f"model_version:{vid}", f"scenario:{sname}", "belongs_to"))
                lin_meta = ver.get("lineage") or {}
                ds_snap_id = str(lin_meta.get("dataset_snapshot_id") or "").strip()
                if ds_snap_id:
                    model_snap_links.append((vid, ds_snap_id))
                    if include("dataset_snapshot"):
                        edges.append(_edge(f"model_version:{vid}", f"dataset_snapshot:{ds_snap_id}", "governed_by"))

    # ---- dataset snapshots (governance JSON files) -----------------------
    if include("dataset_snapshot"):
        try:
            snap_dir = Path(DATASET_REGISTRY_DIR)
            for snap_file in sorted(snap_dir.rglob("*.json")):
                try:
                    data = json.loads(snap_file.read_text(encoding="utf-8"))
                    if not isinstance(data, dict):
                        continue
                    ds_snap_id = str(data.get("snapshot_id") or snap_file.stem).strip()
                    scen_name = snap_file.parent.name
                    if scenario and scen_name != scenario:
                        continue
                    add_node(_node("dataset_snapshot", ds_snap_id, f"{scen_name} data-snap", {
                        "scenario": scen_name,
                        "total_files": data.get("total_files"),
                        "fingerprint_mode": data.get("fingerprint_mode"),
                        "created_at": data.get("created_at"),
                    }))
                    # Only emit the belongs_to edge if the scenario node
                    # actually exists — snapshot directories can outlive
                    # scenarios in the registry.
                    if include("scenario") and scen_name in scenario_names:
                        edges.append(_edge(
                            f"dataset_snapshot:{ds_snap_id}",
                            f"scenario:{scen_name}",
                            "belongs_to",
                        ))
                except Exception:
                    pass
        except Exception:
            pass

    # ---- jobs ------------------------------------------------------------
    if include("job"):
        try:
            jobs_raw = job_store.list_jobs(limit=job_limit)
        except Exception:
            jobs_raw = []
        jobs = [
            j for j in jobs_raw
            if not (scenario and j.scenario != scenario)
            and not (since_ts is not None and j.created_at < since_ts)
        ]
        for job in jobs:
            add_node(_node("job", job.job_id, f"{job.scenario} [{job.job_type}]", {
                "scenario": job.scenario,
                "job_type": job.job_type,
                "state": job.state,
                "source": job.source,
                "created_at": job.created_at,
                "finished_at": job.finished_at,
                "error": job.error or None,
            }))
        # Resolve produces edges: direct match on result_ref first, then fall
        # back to batch result lookup for legacy jobs where result_ref is still
        # the job_id rather than the "{scenario}:{run_version}" version_id.
        if include("model_version"):
            legacy_ids = [
                j.job_id for j in jobs
                if j.job_type in ("train", "update")
                and (
                    not str(j.result_ref or "").strip()
                    or f"model_version:{str(j.result_ref or '').strip()}" not in seen
                )
            ]
            legacy_results: dict[str, dict] = {}
            if legacy_ids:
                try:
                    legacy_results = job_store.batch_results(legacy_ids)
                except Exception:
                    pass
            for job in jobs:
                if job.job_type not in ("train", "update"):
                    continue
                result_ref = str(job.result_ref or "").strip()
                mv_target = f"model_version:{result_ref}" if result_ref else ""
                if (not mv_target or mv_target not in seen) and job.job_id in legacy_results:
                    r = legacy_results[job.job_id]
                    _scen = str(r.get("scenario") or job.scenario or "").strip()
                    _out = str(r.get("output") or r.get("result_path") or "").strip()
                    if not _out:
                        _weights = str(r.get("weights") or "").strip()
                        _out = str(Path(_weights).parent) if _weights else ""
                    _rv = Path(_out).name if _out else ""
                    if _scen and _rv:
                        mv_target = f"model_version:{_scen}:{_rv}"
                if mv_target:
                    edges.append(_edge(f"job:{job.job_id}", mv_target, "produces"))

    # ---- model snapshots (SQLite cvops store) ----------------------------
    if include("model_snapshot"):
        try:
            snap_records = snapshots.list(limit=200)
        except Exception:
            snap_records = []
        for s in snap_records:
            add_node(_node("model_snapshot", s.snapshot_id, f"snap:{s.snapshot_id[:8]}", {
                "origin": s.origin,
                "model_type": s.model_type,
                "lineage_id": s.lineage_id,
                "created_at": s.created_at,
                "size_bytes": s.size_bytes,
            }))
            if s.lineage_id and include("lineage"):
                edges.append(_edge(
                    f"model_snapshot:{s.snapshot_id}",
                    f"lineage:{s.lineage_id}",
                    "belongs_to",
                ))
            if s.parent_snapshot_id:
                edges.append(_edge(
                    f"model_snapshot:{s.snapshot_id}",
                    f"model_snapshot:{s.parent_snapshot_id}",
                    "derived_from",
                ))

    # ---- lineages --------------------------------------------------------
    if include("lineage"):
        try:
            lineage_records = lineages.list_lineages()
        except Exception:
            lineage_records = []
        for lin in lineage_records:
            add_node(_node("lineage", lin.lineage_id, lin.name, {
                "state": lin.state,
                "update_strategy": lin.update_strategy,
                "sector_path": lin.sector_path,
                "created_at": lin.created_at,
                "base_snapshot_id": lin.base_snapshot_id,
                "head_snapshot_id": lin.head_snapshot_id,
            }))
            if lin.head_snapshot_id and include("model_snapshot"):
                edges.append(_edge(
                    f"lineage:{lin.lineage_id}",
                    f"model_snapshot:{lin.head_snapshot_id}",
                    "has_head",
                ))
        # Synthetic registry-derived lineages (model_registry.json).
        for extra in extra_lineages or []:
            if not isinstance(extra, dict):
                continue
            lid = str(extra.get("lineage_id") or "").strip()
            if not lid:
                continue
            add_node(_node("lineage", lid, str(extra.get("name") or lid), {
                "state": str(extra.get("state") or "frozen"),
                "update_strategy": str(extra.get("update_strategy") or "head_only"),
                "sector_path": str(extra.get("sector_path") or "/"),
                "created_at": extra.get("created_at"),
                "base_snapshot_id": str(extra.get("base_snapshot_id") or ""),
                "head_snapshot_id": str(extra.get("head_snapshot_id") or ""),
                "source": "model_registry",
            }))

    # ---- W3C PROV overlay (per local lineage) ----------------------------
    if (
        provenance is not None
        and include("lineage")
        and include("model_snapshot")
    ):
        try:
            for lin_rec in lineages.list_lineages():
                lid = str(lin_rec.lineage_id or "").strip()
                if not lid or lid.startswith("registry:"):
                    continue
                drops_ov = lineages.list_drops(lid)
                if not drops_ov:
                    continue
                frag = provenance.build_ontology_overlay_fragments(lid, drops_ov)
                for n in nodes:
                    if n.get("id") == f"lineage:{lid}":
                        mm = n.setdefault("meta", {})
                        mm["w3c_prov_overlay_edges"] = len(frag.get("edges") or [])
                        mm["w3c_prov_overlay_nodes"] = len(frag.get("nodes") or [])
                for n in frag.get("nodes") or []:
                    add_node(n)
                for e in frag.get("edges") or []:
                    edges.append(e)
        except Exception:
            pass

    # ---- ranges ----------------------------------------------------------
    if include("range"):
        try:
            range_records = ranges.list_ranges()
        except Exception:
            range_records = []
        for r in range_records:
            add_node(_node("range", r.range_id, r.name, {
                "mode": r.mode,
                "sector_path": r.sector_path,
                "created_at": r.created_at,
            }))
            try:
                subjects = ranges.list_subjects(r.range_id)
                for subj in subjects:
                    if include("model_snapshot"):
                        edges.append(_edge(
                            f"range:{r.range_id}",
                            f"model_snapshot:{subj.snapshot_id}",
                            "evaluates",
                        ))
            except Exception:
                pass

    # ---- catalog assets (top 100) ----------------------------------------
    if include("catalog_asset"):
        try:
            assets = catalog.search_assets(sector_path="/", limit=100)
            for asset in assets:
                add_node(_node("catalog_asset", asset.asset_id, asset.name, {
                    "source_type": asset.source_type,
                    "status": asset.status,
                    "sector_path": asset.sector_path,
                    "size_bytes": asset.size_bytes,
                }))
        except Exception:
            pass

    # ---- identity nodes (face gallery) -----------------------------------
    if include("identity"):
        id_nodes, id_edges = _build_identity_nodes(GALLERY_DB_PATH)
        for n in id_nodes:
            add_node(n)
        edges.extend(id_edges)

    # ---- correction event nodes ------------------------------------------
    if include("correction_event"):
        _snap_weights: dict[str, str] = {}
        try:
            for s in snapshots.list(limit=200):
                if s.weights_uri:
                    _snap_weights[s.snapshot_id] = s.weights_uri
        except Exception:
            pass
        _asset_paths: dict[str, list[str]] = {}
        try:
            for asset in catalog.search_assets(sector_path="/", limit=100):
                paths = [p for p in (str(asset.source_uri or ""), str(asset.managed_path or "")) if p]
                if paths:
                    _asset_paths[asset.asset_id] = paths
        except Exception:
            pass
        corr_nodes, corr_edges = _build_correction_nodes(_snap_weights, _asset_paths)
        for n in corr_nodes:
            add_node(n)
        edges.extend(corr_edges)

    # ---- sector + collection hierarchy ------------------------------------
    if include("sector") or include("collection"):
        _asset_coll_map: dict[str, str] = {}
        if include("catalog_asset") and include("collection"):
            try:
                for asset in catalog.search_assets(sector_path="/", limit=100):
                    if asset.collection_id:
                        _asset_coll_map[asset.asset_id] = asset.collection_id
            except Exception:
                pass
        sc_nodes, sc_edges = _build_sector_collection_nodes(catalog, _asset_coll_map)
        for n in sc_nodes:
            add_node(n)
        edges.extend(sc_edges)

    # ---- cross-entity shared-resource edges ------------------------------
    edges.extend(_build_cross_entity_edges(nodes))

    # ---- Final pass: drop any edges whose endpoints didn't make it into
    # the node set. The cvops registries are loosely coupled and may
    # reference entities that have been deleted, filtered out by
    # ``entity_types``/``scenario``, or live in a store that was skipped
    # due to an exception. Cytoscape.js throws on the first such edge and
    # aborts the whole graph render, so we sanitize here.
    valid_edges = [e for e in edges if e["source"] in seen and e["target"] in seen]

    return {"nodes": nodes, "edges": valid_edges}


# ---------------------------------------------------------------------------
# Single-entity detail lookup
# ---------------------------------------------------------------------------

def get_entity(
    entity_type: str,
    entity_id: str,
    *,
    job_store: JobStore,
    snapshots: SnapshotStore,
    lineages: LineageStore,
    ranges: RangeStore,
    catalog: CatalogStore,
    provenance: Optional["ProvenanceStore"] = None,
) -> dict[str, Any]:
    """Return full entity dict + direct edges for a single entity.

    Always returns a dict; "error" key is set if the entity could not be
    found or resolved.
    """
    etype = str(entity_type or "").strip()
    eid = str(entity_id or "").strip()
    entity: dict[str, Any] = {"id": f"{etype}:{eid}", "type": etype, "entity_id": eid}
    direct_edges: list[dict[str, Any]] = []

    if etype == "scenario":
        try:
            cfg = mlops_registry.get_scenario_config(eid)
            entity.update({
                "name": cfg.name,
                "display_name": cfg.display_name,
                "description": cfg.description,
                "backbone_type": cfg.backbone_type,
                "dataset": cfg.dataset,
                "classes": cfg.classes,
                "hyperparams": dict(cfg.hyperparams or {}),
            })
        except Exception as exc:
            entity["error"] = str(exc)

    elif etype == "job":
        try:
            job = job_store.get_job(eid)
            entity.update(job.to_dict())
            mv_target = ""
            if job.job_type in ("train", "update"):
                result_ref = str(job.result_ref or "").strip()
                mv_target = f"model_version:{result_ref}" if result_ref and result_ref != job.job_id else ""
            if job.job_type in ("train", "update") and not mv_target:
                try:
                    result = job_store.get_result(eid) or {}
                except Exception:
                    result = {}
                if isinstance(result, dict):
                    _scen = str(result.get("scenario") or job.scenario or "").strip()
                    _out = str(result.get("output") or result.get("result_path") or "").strip()
                    if not _out:
                        _weights = str(result.get("weights") or "").strip()
                        _out = str(Path(_weights).parent) if _weights else ""
                    _rv = Path(_out).name if _out else ""
                    if _scen and _rv:
                        mv_target = f"model_version:{_scen}:{_rv}"
            if mv_target:
                direct_edges.append(_edge(f"job:{eid}", mv_target, "produces"))
        except Exception as exc:
            entity["error"] = str(exc)

    elif etype == "model_version":
        try:
            mreg = _load_model_registry()
            found = False
            for sname, sdata in (mreg.get("models") or {}).items():
                if not isinstance(sdata, dict):
                    continue
                for ver in sdata.get("versions") or []:
                    if not isinstance(ver, dict):
                        continue
                    if str(ver.get("version_id") or "") == eid:
                        entity.update(ver)
                        aliases_map = sdata.get("aliases") or {}
                        entity["aliases"] = [k for k, v in aliases_map.items() if v == eid]
                        lin_meta = ver.get("lineage") or {}
                        ds_snap_id = str(lin_meta.get("dataset_snapshot_id") or "").strip()
                        if ds_snap_id:
                            direct_edges.append(_edge(
                                f"model_version:{eid}",
                                f"dataset_snapshot:{ds_snap_id}",
                                "governed_by",
                            ))
                        direct_edges.append(_edge(f"model_version:{eid}", f"scenario:{sname}", "belongs_to"))
                        found = True
                        break
                if found:
                    break
            if not found:
                entity["error"] = "model_version not found"
        except Exception as exc:
            entity["error"] = str(exc)

    elif etype == "dataset_snapshot":
        try:
            snap_dir = Path(DATASET_REGISTRY_DIR)
            found = False
            for snap_file in snap_dir.rglob("*.json"):
                try:
                    data = json.loads(snap_file.read_text(encoding="utf-8"))
                    snap_id_check = str(data.get("snapshot_id") or snap_file.stem).strip()
                    if snap_id_check == eid:
                        entity.update(data)
                        found = True
                        break
                except Exception:
                    pass
            if not found:
                entity["error"] = "dataset_snapshot not found"
        except Exception as exc:
            entity["error"] = str(exc)

    elif etype == "model_snapshot":
        snap = snapshots.get(eid)
        if snap:
            entity.update({
                "origin": snap.origin,
                "model_type": snap.model_type,
                "lineage_id": snap.lineage_id,
                "weights_sha256": snap.weights_sha256,
                "size_bytes": snap.size_bytes,
                "created_at": snap.created_at,
                "tags": snap.tags,
                "metadata": snap.metadata,
            })
            if snap.lineage_id:
                direct_edges.append(_edge(f"model_snapshot:{eid}", f"lineage:{snap.lineage_id}", "belongs_to"))
            if snap.parent_snapshot_id:
                direct_edges.append(_edge(
                    f"model_snapshot:{eid}",
                    f"model_snapshot:{snap.parent_snapshot_id}",
                    "derived_from",
                ))
        else:
            entity["error"] = "model_snapshot not found"

    elif etype == "lineage":
        lin = lineages.get_lineage(eid)
        if lin:
            entity.update(lin.to_dict())
            direct_edges.append(_edge(
                f"lineage:{eid}",
                f"model_snapshot:{lin.head_snapshot_id}",
                "has_head",
            ))
            if provenance is not None and not str(eid).startswith("registry:"):
                try:
                    drops_ov = lineages.list_drops(eid)
                    if drops_ov:
                        frag = provenance.build_ontology_overlay_fragments(eid, drops_ov)
                        entity["w3c_prov_overlay"] = {
                            "extra_nodes": len(frag.get("nodes") or []),
                            "extra_edges": len(frag.get("edges") or []),
                            "provenance_json": f"/lineages/{eid}/provenance",
                        }
                except Exception:
                    pass
        else:
            entity["error"] = "lineage not found"

    elif etype == "range":
        r = ranges.get_range(eid)
        if r:
            entity.update({
                "range_id": r.range_id,
                "name": r.name,
                "mode": r.mode,
                "sector_path": r.sector_path,
                "description": r.description,
                "created_at": r.created_at,
                "updated_at": r.updated_at,
                "tags": r.tags,
            })
            try:
                subjects = ranges.list_subjects(eid)
                for subj in subjects:
                    direct_edges.append(_edge(
                        f"range:{eid}",
                        f"model_snapshot:{subj.snapshot_id}",
                        "evaluates",
                    ))
            except Exception:
                pass
        else:
            entity["error"] = "range not found"

    elif etype == "catalog_asset":
        try:
            asset = catalog.get_asset(eid)
            if asset:
                entity.update({
                    "asset_id": asset.asset_id,
                    "name": asset.name,
                    "source_type": asset.source_type,
                    "storage_mode": asset.storage_mode,
                    "sector_path": asset.sector_path,
                    "status": asset.status,
                    "size_bytes": asset.size_bytes,
                    "tags": asset.tags,
                    "metadata": asset.metadata,
                    "created_at": asset.created_at,
                })
            else:
                entity["error"] = "catalog_asset not found"
        except Exception as exc:
            entity["error"] = str(exc)

    else:
        entity["error"] = f"unknown entity_type: {etype!r}"

    entity["edges"] = direct_edges
    return entity
