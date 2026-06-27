from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from .lineage_store import DropRecord, LineageRecord, LineageStore
from .snapshot_store import SnapshotRecord, SnapshotStore

PROV_PREFIX = "http://www.w3.org/ns/prov#"
CVOPS_PROV_ROOT = "urn:cvops:prov"
# PROV-JSON uses QName-style keys (see W3C PROV-JSON); values can use cvops: prefix.
TYPE_MODEL_SNAPSHOT = "cvops:ModelSnapshot"
TYPE_LINEAGE_COLLECTION = "cvops:LineageCollection"
TYPE_SOFTWARE_AGENT = "prov:SoftwareAgent"
TYPE_EXTERNAL_ENTITY = "cvops:ExternalEntity"
TYPE_AGENT = "cvops:Agent"

VALID_RELATIONS = frozenset(
    {
        "wasGeneratedBy",
        "wasInvalidatedBy",
        "used",
        "wasInformedBy",
        "wasDerivedFrom",
        "wasAttributedTo",
        "wasAssociatedWith",
        "specializationOf",
        "hadMember",
    }
)

_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS prov_nodes (
    node_id    TEXT PRIMARY KEY,
    kind       TEXT NOT NULL,
    label      TEXT NOT NULL DEFAULT '',
    cvops_ref  TEXT UNIQUE,
    attrs_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS prov_edges (
    rel        TEXT NOT NULL,
    subject    TEXT NOT NULL,
    object     TEXT NOT NULL,
    attrs_json TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    PRIMARY KEY (rel, subject, object)
);

CREATE INDEX IF NOT EXISTS idx_prov_edges_subject ON prov_edges(subject);
CREATE INDEX IF NOT EXISTS idx_prov_edges_object ON prov_edges(object);
"""


def entity_snapshot_uri(snapshot_id: str) -> str:
    return f"{CVOPS_PROV_ROOT}:entity:snapshot:{snapshot_id}"


def entity_lineage_uri(lineage_id: str) -> str:
    return f"{CVOPS_PROV_ROOT}:entity:lineage:{lineage_id}"


def activity_drop_uri(drop_id: str) -> str:
    return f"{CVOPS_PROV_ROOT}:activity:drop:{drop_id}"


def agent_cvops_service_uri() -> str:
    return f"{CVOPS_PROV_ROOT}:agent:cvops-service"


def activity_invalidate_uri(prefix: str) -> str:
    return f"{CVOPS_PROV_ROOT}:activity:{prefix}:{uuid.uuid4().hex[:12]}"


def _merge_dict(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    out = dict(a)
    out.update(b)
    return out


def _load_json_obj(raw: str) -> dict[str, Any]:
    try:
        v = json.loads(raw or "{}")
    except Exception:
        return {}
    return v if isinstance(v, dict) else {}


def _attributed_agent_uri(metadata: dict[str, Any]) -> Optional[str]:
    for key in ("prov_agent", "prov:agent", "attributed_to", "attributedTo"):
        v = metadata.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _used_entity_uris_from_source(source: dict[str, Any]) -> list[str]:
    raw = source.get("prov_used_entities")
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


class ProvenanceStore:
    """Persisted PROV-DM-aligned graph for cvops lineage and snapshots."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._ensure_agent_cvops_service()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_agent_cvops_service(self) -> None:
        uri = agent_cvops_service_uri()
        ref = "agent:cvops-service"
        attrs = {
            "prov:type": TYPE_SOFTWARE_AGENT,
        }
        self._upsert_node(
            node_id=uri,
            kind="agent",
            label="CV Ops service",
            cvops_ref=ref,
            attrs=attrs,
        )

    def _upsert_node(
        self,
        *,
        node_id: str,
        kind: str,
        label: str,
        cvops_ref: str,
        attrs: dict[str, Any],
    ) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT attrs_json FROM prov_nodes WHERE cvops_ref = ?", (cvops_ref,)
            ).fetchone()
            if row is not None:
                merged = _merge_dict(_load_json_obj(str(row["attrs_json"])), attrs)
                self._conn.execute(
                    """
                    UPDATE prov_nodes
                       SET node_id = ?, kind = ?, label = ?, attrs_json = ?
                     WHERE cvops_ref = ?
                    """,
                    (node_id, kind, label, json.dumps(merged), cvops_ref),
                )
            else:
                self._conn.execute(
                    """
                    INSERT INTO prov_nodes (node_id, kind, label, cvops_ref, attrs_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (node_id, kind, label, cvops_ref, json.dumps(attrs)),
                )
            self._conn.commit()

    def _insert_edge(
        self,
        rel: str,
        subject: str,
        object: str,
        attrs: Optional[dict[str, Any]] = None,
    ) -> None:
        if rel not in VALID_RELATIONS:
            raise ValueError(f"unsupported PROV relation: {rel}")
        blob = json.dumps(dict(attrs or {}))
        now = time.time()
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO prov_edges (rel, subject, object, attrs_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (rel, subject, object, blob, now),
            )
            self._conn.commit()

    def record_snapshot_registered(self, rec: SnapshotRecord) -> None:
        """Entity for a snapshot; wasDerivedFrom from parent; optional wasAttributedTo."""
        sid = str(rec.snapshot_id or "").strip()
        if not sid:
            return
        e_uri = entity_snapshot_uri(sid)
        meta = dict(rec.metadata or {})
        attrs: dict[str, Any] = {
            "prov:type": TYPE_MODEL_SNAPSHOT,
            "cvops:snapshot_id": sid,
            "cvops:model_type": rec.model_type,
            "cvops:weights_sha256": rec.weights_sha256,
            "cvops:weights_uri": rec.weights_uri,
            "cvops:origin": rec.origin,
            "cvops:lineage_id": rec.lineage_id or "",
            "cvops:adapter_only": rec.adapter_only,
        }
        self._upsert_node(
            node_id=e_uri,
            kind="entity",
            label=f"snapshot {sid}",
            cvops_ref=f"snapshot:{sid}",
            attrs=attrs,
        )
        parent = str(rec.parent_snapshot_id or "").strip()
        if parent:
            p_uri = entity_snapshot_uri(parent)
            self._insert_edge("wasDerivedFrom", e_uri, p_uri)
        agent_uri = _attributed_agent_uri(meta)
        if agent_uri:
            self._ensure_agent_uri(agent_uri)
            self._insert_edge("wasAttributedTo", e_uri, agent_uri)

    def _ensure_agent_uri(self, uri: str) -> None:
        ref = f"agent:external:{hash(uri) & 0xFFFFFFFF}"
        self._upsert_node(
            node_id=uri,
            kind="agent",
            label=uri,
            cvops_ref=ref,
            attrs={"prov:type": TYPE_AGENT},
        )

    def record_lineage_created(
        self,
        lineage: LineageRecord,
        base_drop: DropRecord,
        base_snap: SnapshotRecord,
    ) -> None:
        """Lineage collection entity, base drop activity, wasGeneratedBy, hadMember."""
        lid = str(lineage.lineage_id or "").strip()
        if not lid:
            return
        l_uri = entity_lineage_uri(lid)
        self._upsert_node(
            node_id=l_uri,
            kind="entity",
            label=lineage.name,
            cvops_ref=f"lineage:{lid}",
            attrs={
                "prov:type": TYPE_LINEAGE_COLLECTION,
                "cvops:lineage_id": lid,
                "cvops:sector_path": lineage.sector_path,
            },
        )
        drop = base_drop
        a_uri = activity_drop_uri(drop.drop_id)
        self._upsert_node(
            node_id=a_uri,
            kind="activity",
            label=f"drop {drop.drop_index} (base)",
            cvops_ref=f"drop_activity:{drop.drop_id}",
            attrs=_drop_activity_attrs(drop),
        )
        s_uri = entity_snapshot_uri(base_snap.snapshot_id)
        self.record_snapshot_registered(base_snap)
        self._insert_edge("wasGeneratedBy", s_uri, a_uri)
        self._insert_edge("wasAssociatedWith", a_uri, agent_cvops_service_uri())
        self._insert_edge("hadMember", l_uri, s_uri)

    def record_drop_added(
        self,
        *,
        lineage: LineageRecord,
        drop: DropRecord,
        prev_drop: DropRecord,
        snap_rec: SnapshotRecord,
        prev_snap_rec: SnapshotRecord,
        job_id: str = "",
    ) -> None:
        """Activity for a non-base drop: used, wasGeneratedBy, wasDerivedFrom, wasInformedBy."""
        lid = str(lineage.lineage_id or "").strip()
        if not lid or drop.drop_index <= 0:
            return
        l_uri = entity_lineage_uri(lid)
        a_uri = activity_drop_uri(drop.drop_id)
        prev_a_uri = activity_drop_uri(prev_drop.drop_id)
        s_uri = entity_snapshot_uri(snap_rec.snapshot_id)
        prev_s_uri = entity_snapshot_uri(prev_snap_rec.snapshot_id)

        self._upsert_node(
            node_id=a_uri,
            kind="activity",
            label=f"drop {drop.drop_index}",
            cvops_ref=f"drop_activity:{drop.drop_id}",
            attrs=_merge_dict(
                _drop_activity_attrs(drop),
                {"cvops:job_id": str(job_id)} if job_id else {},
            ),
        )
        self.record_snapshot_registered(snap_rec)
        self._insert_edge("used", a_uri, prev_s_uri)
        self._insert_edge("wasGeneratedBy", s_uri, a_uri)
        self._insert_edge("wasDerivedFrom", s_uri, prev_s_uri)
        self._insert_edge("wasInformedBy", a_uri, prev_a_uri)
        self._insert_edge("wasAssociatedWith", a_uri, agent_cvops_service_uri())
        self._insert_edge("hadMember", l_uri, s_uri)

        for extra_uri in _used_entity_uris_from_source(dict(drop.source or {})):
            ext_ref = "external:" + hashlib.sha256(extra_uri.encode("utf-8")).hexdigest()[:16]
            self._upsert_node(
                node_id=extra_uri,
                kind="entity",
                label=extra_uri,
                cvops_ref=ext_ref,
                attrs={"prov:type": TYPE_EXTERNAL_ENTITY},
            )
            self._insert_edge("used", a_uri, extra_uri)

    def record_lineage_fork(self, lineage: LineageRecord) -> None:
        """specializationOf new lineage entity -> source lineage entity."""
        meta = dict(lineage.metadata or {})
        fork = meta.get("forked_from")
        if not isinstance(fork, dict):
            return
        src_lid = str(fork.get("lineage_id") or "").strip()
        if not src_lid:
            return
        new_uri = entity_lineage_uri(lineage.lineage_id)
        old_uri = entity_lineage_uri(src_lid)
        self._insert_edge("specializationOf", new_uri, old_uri)

    def record_snapshot_invalidated(self, rec: SnapshotRecord) -> None:
        sid = str(rec.snapshot_id or "").strip()
        if not sid:
            return
        e_uri = entity_snapshot_uri(sid)
        inv_uri = activity_invalidate_uri("invalidate_snapshot")
        self._upsert_node(
            node_id=inv_uri,
            kind="activity",
            label=f"invalidate snapshot {sid}",
            cvops_ref=f"invalidate_snapshot:{sid}:{inv_uri[-12:]}",
            attrs={
                "prov:startTime": rec.created_at,
                "prov:endTime": time.time(),
                "cvops:snapshot_id": sid,
            },
        )
        self._insert_edge("wasInvalidatedBy", e_uri, inv_uri)

    def record_lineage_deleted(self, lineage_id: str, *, label: Optional[str] = None) -> None:
        """Invalidate lineage collection entity only (snapshots may be shared)."""
        lid = str(lineage_id or "").strip()
        if not lid:
            return
        l_uri = entity_lineage_uri(lid)
        disp = str(label or "").strip() or f"lineage {lid}"
        self._upsert_node(
            node_id=l_uri,
            kind="entity",
            label=f"{disp} (deleted)",
            cvops_ref=f"lineage:{lid}",
            attrs={
                "prov:type": TYPE_LINEAGE_COLLECTION,
                "cvops:lineage_id": lid,
                "cvops:deleted": True,
            },
        )
        inv_uri = activity_invalidate_uri("invalidate_lineage")
        self._upsert_node(
            node_id=inv_uri,
            kind="activity",
            label=f"delete lineage {lid}",
            cvops_ref=f"invalidate_lineage:{lid}:{inv_uri[-12:]}",
            attrs={
                "prov:endTime": time.time(),
                "cvops:lineage_id": lid,
            },
        )
        self._insert_edge("wasInvalidatedBy", l_uri, inv_uri)

    def backfill_lineage(
        self,
        lineage_id: str,
        lineages: LineageStore,
        snapshots: SnapshotStore,
    ) -> None:
        lid = str(lineage_id or "").strip()
        if not lid:
            return
        lin = lineages.get_lineage(lid)
        if lin is None:
            return
        drops = lineages.list_drops(lid)
        if not drops:
            return
        base = drops[0]
        bs = snapshots.get(base.snapshot_id)
        if bs is not None:
            self.record_lineage_created(lin, base, bs)
        for i in range(1, len(drops)):
            d = drops[i]
            prev = drops[i - 1]
            sn = snapshots.get(d.snapshot_id)
            psn = snapshots.get(prev.snapshot_id)
            if sn is None or psn is None:
                continue
            self.record_drop_added(
                lineage=lin,
                drop=d,
                prev_drop=prev,
                snap_rec=sn,
                prev_snap_rec=psn,
            )
        self.record_lineage_fork(lin)

    def backfill_all(self, lineages: LineageStore, snapshots: SnapshotStore) -> int:
        n = 0
        for rec in lineages.list_lineages():
            self.backfill_lineage(rec.lineage_id, lineages, snapshots)
            n += 1
        return n

    def graph_counts(self) -> dict[str, int]:
        """Return row counts for the persisted PROV graph."""
        with self._lock:
            nodes = int(self._conn.execute("SELECT COUNT(*) FROM prov_nodes").fetchone()[0])
            edges = int(self._conn.execute("SELECT COUNT(*) FROM prov_edges").fetchone()[0])
        return {"prov_nodes": nodes, "prov_edges": edges}

    def _prov_closure_bundle(
        self, lineage_id: str, drops: list[DropRecord]
    ) -> tuple[set[str], list[sqlite3.Row], list[sqlite3.Row]]:
        """Expand PROV subgraph from lineage + drops; return seeds, nodes, internal edges."""
        lid = str(lineage_id or "").strip()
        if not lid or not drops:
            return set(), [], []

        seeds: set[str] = {entity_lineage_uri(lid)}
        for d in drops:
            seeds.add(entity_snapshot_uri(d.snapshot_id))
            seeds.add(activity_drop_uri(d.drop_id))

        changed = True
        while changed:
            changed = False
            with self._lock:
                rows = self._conn.execute(
                    "SELECT rel, subject, object FROM prov_edges"
                ).fetchall()
            for r in rows:
                subj = str(r["subject"])
                obj = str(r["object"])
                if subj in seeds and obj not in seeds:
                    seeds.add(obj)
                    changed = True
                if obj in seeds and subj not in seeds:
                    seeds.add(subj)
                    changed = True

        if not seeds:
            return set(), [], []

        placeholders = ",".join("?" * len(seeds))
        tup = tuple(seeds)
        with self._lock:
            node_rows = self._conn.execute(
                f"SELECT * FROM prov_nodes WHERE node_id IN ({placeholders})",
                tup,
            ).fetchall()
            edge_rows = self._conn.execute(
                f"SELECT rel, subject, object, attrs_json FROM prov_edges "
                f"WHERE subject IN ({placeholders}) AND object IN ({placeholders})",
                tup + tup,
            ).fetchall()
        return seeds, list(node_rows), list(edge_rows)

    def export_lineage_closure_prov_json(
        self,
        lineage_id: str,
        lineages: LineageStore,
    ) -> dict[str, Any]:
        """PROV-JSON document for all nodes/edges reachable from a lineage subgraph."""
        lid = str(lineage_id or "").strip()
        if not lid:
            return _empty_prov_document()
        drops = lineages.list_drops(lid)
        if not drops:
            return _empty_prov_document()
        _seeds, node_rows, edge_rows = self._prov_closure_bundle(lid, drops)
        if not _seeds:
            return _empty_prov_document()
        return _rows_to_prov_json(node_rows, edge_rows)

    def build_ontology_overlay_fragments(
        self,
        lineage_id: str,
        drops: list[DropRecord],
    ) -> dict[str, list[Any]]:
        """Nodes and edges in ontology ``build_graph`` shape for Cytoscape (W3C PROV overlay)."""
        from .ontology import _edge as ont_edge
        from .ontology import _node as ont_node

        lid = str(lineage_id or "").strip()
        out_nodes: list[dict[str, Any]] = []
        out_edges: list[dict[str, Any]] = []
        if not lid or not drops:
            return {"nodes": out_nodes, "edges": out_edges}

        seeds, node_rows, edge_rows = self._prov_closure_bundle(lid, drops)
        if not seeds:
            return {"nodes": out_nodes, "edges": out_edges}

        snap_p = f"{CVOPS_PROV_ROOT}:entity:snapshot:"
        lin_p = f"{CVOPS_PROV_ROOT}:entity:lineage:"
        drop_act_p = f"{CVOPS_PROV_ROOT}:activity:drop:"
        agent_p = f"{CVOPS_PROV_ROOT}:agent:"

        def uri_to_graph_id(uri: str) -> Optional[str]:
            if uri.startswith(snap_p):
                return "model_snapshot:" + uri[len(snap_p) :]
            if uri.startswith(lin_p):
                return "lineage:" + uri[len(lin_p) :]
            if uri.startswith(drop_act_p):
                return "prov_activity:" + uri[len(drop_act_p) :]
            if uri.startswith(agent_p):
                tail = uri[len(agent_p) :].replace(":", "_")
                return "prov_agent:" + tail
            if CVOPS_PROV_ROOT + ":activity:" in uri:
                h = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:14]
                return "prov_activity:inv_" + h
            if uri.startswith("http://") or uri.startswith("https://") or uri.startswith("urn:"):
                h = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:12]
                return "prov_entity_ext:" + h
            return None

        seen_out_node: set[str] = set()

        def emit_node(n: dict[str, Any]) -> None:
            nid = str(n.get("id") or "")
            if nid and nid not in seen_out_node:
                seen_out_node.add(nid)
                out_nodes.append(n)

        for r in node_rows:
            uri = str(r["node_id"])
            kind = str(r["kind"])
            label = str(r["label"] or "")
            attrs = _load_json_obj(str(r["attrs_json"]))
            attrs["prov_uri"] = uri
            attrs["w3c_prov"] = True
            if kind == "entity" and uri.startswith(snap_p):
                continue
            if kind == "entity" and uri.startswith(lin_p):
                continue
            if kind == "entity":
                gid = uri_to_graph_id(uri)
                if gid is None:
                    h = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:12]
                    gid = "prov_entity_ext:" + h
                emit_node(
                    ont_node(
                        "prov_entity",
                        gid.split(":", 1)[-1],
                        label or gid,
                        attrs,
                    )
                )
                continue
            if kind == "activity":
                gid = uri_to_graph_id(uri)
                if gid is None or not gid.startswith("prov_activity:"):
                    h = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:14]
                    gid = "prov_activity:inv_" + h
                short = gid.split(":", 1)[-1]
                emit_node(
                    ont_node(
                        "prov_activity",
                        short,
                        label or short,
                        attrs,
                    )
                )
                continue
            if kind == "agent":
                gid = uri_to_graph_id(uri)
                if gid is None:
                    gid = "prov_agent:unknown"
                short = gid.split(":", 1)[-1]
                emit_node(
                    ont_node("prov_agent", short, label or short, attrs)
                )

        edge_dedup: set[tuple[str, str, str]] = set()

        def emit_edge(e: dict[str, Any]) -> None:
            k = (str(e["type"]), str(e["source"]), str(e["target"]))
            if k in edge_dedup:
                return
            edge_dedup.add(k)
            out_edges.append(e)

        for er in edge_rows:
            rel = str(er["rel"])
            su = str(er["subject"])
            ob = str(er["object"])
            sid = uri_to_graph_id(su)
            tid = uri_to_graph_id(ob)
            if sid is None or tid is None:
                continue
            if rel == "wasGeneratedBy":
                # subject=entity, object=activity -> activity generates entity
                emit_edge(ont_edge(tid, sid, "prov_generates"))
            elif rel == "used":
                emit_edge(ont_edge(sid, tid, "prov_used"))
            elif rel == "wasDerivedFrom":
                emit_edge(ont_edge(sid, tid, "derived_from"))
            elif rel == "wasInformedBy":
                # subject=informed activity, object=informant -> informant -> informed
                emit_edge(ont_edge(tid, sid, "prov_informed_by"))
            elif rel == "wasAssociatedWith":
                emit_edge(ont_edge(sid, tid, "prov_associated"))
            elif rel == "hadMember":
                emit_edge(ont_edge(sid, tid, "had_member"))
            elif rel == "specializationOf":
                emit_edge(ont_edge(sid, tid, "specialization_of"))
            elif rel == "wasInvalidatedBy":
                emit_edge(ont_edge(sid, tid, "prov_invalidated"))
            elif rel == "wasAttributedTo":
                emit_edge(ont_edge(sid, tid, "prov_attributed"))

        return {"nodes": out_nodes, "edges": out_edges}


def _drop_activity_attrs(drop: DropRecord) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prov:type": "cvops:DropActivity",
        "cvops:drop_id": drop.drop_id,
        "cvops:drop_index": drop.drop_index,
        "cvops:lineage_id": drop.lineage_id,
    }
    if drop.started_at:
        out["prov:startTime"] = drop.started_at
    if drop.finished_at is not None:
        out["prov:endTime"] = drop.finished_at
    return out


def _empty_prov_document() -> dict[str, Any]:
    return {
        "prefix": {
            "prov": PROV_PREFIX,
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "cvops": f"{CVOPS_PROV_ROOT}:",
        },
        "entity": {},
        "activity": {},
        "agent": {},
        "wasGeneratedBy": {},
        "wasInvalidatedBy": {},
        "used": {},
        "wasInformedBy": {},
        "wasDerivedFrom": {},
        "wasAttributedTo": {},
        "wasAssociatedWith": {},
        "specializationOf": {},
        "hadMember": {},
    }


def _rows_to_prov_json(
    node_rows: list[sqlite3.Row],
    edge_rows: list[sqlite3.Row],
) -> dict[str, Any]:
    doc = _empty_prov_document()
    entities: dict[str, Any] = doc["entity"]
    activities: dict[str, Any] = doc["activity"]
    agents: dict[str, Any] = doc["agent"]

    for r in node_rows:
        nid = str(r["node_id"])
        kind = str(r["kind"])
        attrs = _load_json_obj(str(r["attrs_json"]))
        label = str(r["label"] or "")
        if label:
            attrs["prov:label"] = label
        if kind == "entity":
            entities[nid] = attrs
        elif kind == "activity":
            activities[nid] = attrs
        elif kind == "agent":
            agents[nid] = attrs

    counters: dict[str, int] = {}

    def _rel_bucket(rel: str) -> dict[str, Any]:
        b = doc.get(rel)
        if isinstance(b, dict):
            return b
        return {}

    def _next_id(rel: str) -> str:
        counters[rel] = counters.get(rel, 0) + 1
        return f"_:b{rel}{counters[rel]}"

    for r in edge_rows:
        rel = str(r["rel"])
        subj = str(r["subject"])
        obj = str(r["object"])
        attrs = _load_json_obj(str(r["attrs_json"]))
        bid = _next_id(rel)
        bucket = _rel_bucket(rel)
        if rel == "wasGeneratedBy":
            bucket[bid] = _merge_dict({"prov:entity": subj, "prov:activity": obj}, attrs)
        elif rel == "wasInvalidatedBy":
            bucket[bid] = _merge_dict({"prov:entity": subj, "prov:activity": obj}, attrs)
        elif rel == "used":
            bucket[bid] = _merge_dict({"prov:activity": subj, "prov:entity": obj}, attrs)
        elif rel == "wasInformedBy":
            bucket[bid] = _merge_dict({"prov:informed": subj, "prov:informant": obj}, attrs)
        elif rel == "wasDerivedFrom":
            bucket[bid] = _merge_dict(
                {"prov:generatedEntity": subj, "prov:usedEntity": obj},
                attrs,
            )
        elif rel == "wasAttributedTo":
            bucket[bid] = _merge_dict({"prov:entity": subj, "prov:agent": obj}, attrs)
        elif rel == "wasAssociatedWith":
            bucket[bid] = _merge_dict({"prov:activity": subj, "prov:agent": obj}, attrs)
        elif rel == "specializationOf":
            bucket[bid] = _merge_dict(
                {"prov:specificEntity": subj, "prov:generalEntity": obj},
                attrs,
            )
        elif rel == "hadMember":
            bucket[bid] = _merge_dict({"prov:collection": subj, "prov:entity": obj}, attrs)

    return doc
