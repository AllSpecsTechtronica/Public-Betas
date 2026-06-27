from __future__ import annotations

import asyncio
import base64
import copy
import csv
import hashlib
import io
import json
import math
import mimetypes
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Any, Callable, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..config import ROOT_DIR
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from mlops.pipeline import cards as mlops_cards
from mlops.pipeline import audio_ops as mlops_audio_ops
# NOTE: mlops.pipeline.export and .infer pull in ultralytics -> torch -> torchvision
# (~thousands of files; the dominant cold-start cost). They are imported lazily at
# their single call sites so the app window can appear without paying for torch up
# front; the warmup thread (see CvOpsService._warm_inference_stack) preloads them in
# the background so the first real inference/export is still fast.
from mlops.pipeline.ci_cd import evaluate_run_gate, load_gate_report, promote_run
from mlops.pipeline.governance import DATASET_REGISTRY_DIR
from mlops.pipeline.integration import append_integration_event
from mlops.pipeline.model_registry import (
    MODEL_REGISTRY_PATH,
    alias_history,
    list_model_versions,
    register_model_version,
    resolve_alias,
    revert_alias,
)
from mlops.pipeline import registry as mlops_registry
from mlops.pipeline.custom_cells_store import promote_draft, read_draft, write_draft
from mlops.pipeline.train import run_training

from .jobs import JobRecord, JobStore
from .ui.settings_panel import CvOpsSettings, load_cvops_settings, save_cvops_settings


# Path constants live in the dependency-free paths module so window.py and the
# lazy panels can read them without importing this (heavy) module. Re-exported
# here for backward-compatible ``from .service import CVOPS_STATE_DIR`` imports.
from .paths import (  # noqa: E402
    MLOPS_ROOT,
    CVOPS_STATE_DIR,
    CVOPS_DB_PATH,
    CVOPS_CATALOG_DB_PATH,
)

JOB_IMAGES_DIR = MLOPS_ROOT / "jobs" / "images"
CVOPS_CATALOG_ASSETS_DIR = CVOPS_STATE_DIR / "catalog_assets"
CVOPS_SNAPSHOT_DB_PATH = CVOPS_STATE_DIR / "snapshots.db"
CVOPS_SNAPSHOT_WEIGHTS_DIR = CVOPS_STATE_DIR / "snapshot_weights"
CVOPS_LINEAGE_DB_PATH = CVOPS_STATE_DIR / "lineages.db"
CVOPS_PROVENANCE_DB_PATH = CVOPS_STATE_DIR / "provenance.db"
CVOPS_RANGE_DB_PATH = CVOPS_STATE_DIR / "ranges.db"
CVOPS_ARCHIVE_DB_PATH = CVOPS_STATE_DIR / "archives.db"
CVOPS_ARCHIVE_STORAGE_ROOT = CVOPS_STATE_DIR / "archive_corpora"
CVOPS_WEB_DIST_DIR = Path(__file__).resolve().parent / "web" / "dist"
ROOT_SECTOR_ID = "sector-root"
ROOT_SECTOR_PATH = "/"


_REGISTRY_LINEAGE_PREFIX = "registry:"
_GRAPH_CORE_ENTITY_TYPES: tuple[str, ...] = (
    "scenario",
    "backbone",
    "dataset",
    "model_version",
    "job",
)
_GRAPH_DEFAULT_LAYERS: tuple[str, ...] = ("core", "full")


def _browser_ecosystem_html(*, base_url: str, graph: dict[str, Any]) -> str:
    from .ui.ontology_panel import _CYTOSCAPE_FALLBACK, _CYTOSCAPE_LOCAL_PATH, _build_html

    html = _build_html(
        graph,
        base_url,
        base_url.rstrip("/") + _CYTOSCAPE_LOCAL_PATH,
        _CYTOSCAPE_FALLBACK,
    )
    bridge = """
function _cvopsPost(type, payload) {
  if (window.parent && window.parent !== window) {
    window.parent.postMessage(Object.assign({ type: type }, payload || {}), window.location.origin);
    return true;
  }
  return false;
}
function _cvopsReload() {
  if (!_cvopsPost('cvops-eco-reload', {})) window.location.href = 'appbridge://reload';
}
function _cvopsGoto(target, focusId, scenarioHint) {
  if (_cvopsPost('cvops-eco-goto', {
    target: String(target || ''),
    focusId: String(focusId || ''),
    scenarioHint: String(scenarioHint || ''),
  })) return;
  const url = 'appbridge://goto/' + encodeURIComponent(target || '')
    + '/' + encodeURIComponent(focusId || '')
    + (scenarioHint ? ('?scenario=' + encodeURIComponent(scenarioHint)) : '');
  window.location.href = url;
}
function _cvopsInspect(actionType, actionId) {
  if (!_cvopsPost('cvops-eco-inspect', {
    entityType: String(actionType || ''),
    entityId: String(actionId || ''),
  })) {
    window.location.href = 'appbridge://inspect/' + encodeURIComponent(actionType || '')
      + '/' + encodeURIComponent(actionId || '');
  }
}
function _cvopsEntity(entityType, entityId) {
  if (!_cvopsPost('cvops-eco-entity', {
    entityType: String(entityType || ''),
    entityId: String(entityId || ''),
  })) {
    window.location.href = 'appbridge://entity/' + encodeURIComponent(entityType || '')
      + '/' + encodeURIComponent(entityId || '');
  }
}
function _cvopsOpenAppbridge(url) {
  if (typeof url !== 'string' || !url) return;
  if (url === 'appbridge://reload') {
    _cvopsReload();
    return;
  }
  if (url.startsWith('appbridge://entity/scenario/')) {
    _cvopsEntity('scenario', decodeURIComponent(url.slice('appbridge://entity/scenario/'.length)));
    return;
  }
  if (url.startsWith('appbridge://entity/')) {
    const raw = url.slice('appbridge://entity/'.length);
    const slash = raw.indexOf('/');
    if (slash >= 0) {
      _cvopsEntity(
        decodeURIComponent(raw.slice(0, slash)),
        decodeURIComponent(raw.slice(slash + 1)),
      );
      return;
    }
  }
  if (url.startsWith('appbridge://inspect/')) {
    const raw = url.slice('appbridge://inspect/'.length);
    const slash = raw.indexOf('/');
    if (slash >= 0) {
      _cvopsInspect(
        decodeURIComponent(raw.slice(0, slash)),
        decodeURIComponent(raw.slice(slash + 1)),
      );
      return;
    }
  }
  if (url.startsWith('appbridge://goto/')) {
    const raw = url.slice('appbridge://goto/'.length);
    const parts = raw.split('?');
    const pathParts = parts[0].split('/');
    const query = new URLSearchParams(parts[1] || '');
    _cvopsGoto(
      decodeURIComponent(pathParts[0] || ''),
      decodeURIComponent(pathParts[1] || ''),
      query.get('scenario') || '',
    );
    return;
  }
  window.location.href = url;
}
"""
    html = html.replace(
        "<script>\nfunction _bootShowErr(msg) {",
        "<script>\n" + bridge + "\nfunction _bootShowErr(msg) {",
        1,
    )
    html = html.replace(
        "        const url = 'appbridge://goto/' + encodeURIComponent(t.target)\n"
        "                  + '/' + encodeURIComponent(t.fid)\n"
        "                  + (scenarioHint ? ('?scenario=' + encodeURIComponent(scenarioHint)) : '');\n"
        "        window.location.href = url;",
        "        _cvopsGoto(t.target, t.fid, scenarioHint);",
    )
    html = html.replace(
        "        setTimeout(() => { window.location.href = 'appbridge://reload'; }, 600);",
        "        setTimeout(() => { _cvopsReload(); }, 600);",
    )
    html = html.replace(
        "    window.location.href = 'appbridge://inspect/' + encodeURIComponent(actionType)\n"
        "                       + '/' + encodeURIComponent(actionId);",
        "    _cvopsInspect(actionType, actionId);",
    )
    html = html.replace(
        "    if (_currentNavTarget) window.location.href = _currentNavTarget;",
        "    if (_currentNavTarget) _cvopsOpenAppbridge(_currentNavTarget);",
    )
    html = html.replace(
        "    window.location.href = 'appbridge://entity/' + encodeURIComponent(etype) + '/' + encodeURIComponent(eid);",
        "    _cvopsEntity(etype, eid);",
    )
    return html


def _fallback_nice_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>CV Ops Nice</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #071013;
      --panel: rgba(13, 28, 32, 0.82);
      --line: rgba(125, 239, 219, 0.22);
      --text: #e7fffb;
      --muted: #8fbab3;
      --accent: #6fffe9;
      --hot: #ffb86b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at 12% 20%, rgba(111,255,233,0.18), transparent 34rem),
        radial-gradient(circle at 88% 12%, rgba(255,184,107,0.14), transparent 30rem),
        linear-gradient(135deg, #071013, #0b1d22 55%, #061116);
      color: var(--text);
    }
    main { max-width: 1180px; margin: 0 auto; padding: 56px 24px; }
    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.2fr) minmax(320px, 0.8fr);
      gap: 24px;
      align-items: stretch;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 28px;
      background: var(--panel);
      box-shadow: 0 24px 80px rgba(0,0,0,0.35);
      backdrop-filter: blur(18px);
      padding: 28px;
    }
    h1 { margin: 0 0 16px; font-size: clamp(42px, 7vw, 86px); line-height: 0.9; letter-spacing: -0.07em; }
    h2 { margin: 0 0 16px; font-size: 22px; }
    p { color: var(--muted); font-size: 17px; line-height: 1.55; margin: 0; }
    .actions { display: flex; flex-wrap: wrap; gap: 12px; margin-top: 28px; }
    a.button, button {
      appearance: none;
      border: 1px solid rgba(111,255,233,0.38);
      background: rgba(111,255,233,0.12);
      color: var(--text);
      border-radius: 999px;
      padding: 12px 18px;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
    }
    a.button.primary {
      background: var(--accent);
      border-color: var(--accent);
      color: #001817;
    }
    .grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; margin-top: 24px; }
    .metric { border: 1px solid var(--line); border-radius: 18px; padding: 16px; background: rgba(255,255,255,0.04); }
    .metric b { display: block; font-size: 26px; margin-bottom: 6px; }
    .metric span { color: var(--muted); font-size: 13px; }
    .event { padding: 12px 0; border-bottom: 1px solid rgba(125,239,219,0.14); color: var(--muted); font-size: 14px; }
    .event:first-child { color: var(--text); }
    code { color: var(--hot); }
    @media (max-width: 840px) {
      .hero, .grid { grid-template-columns: 1fr; }
      main { padding: 28px 16px; }
    }
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <div class="card">
        <h1>CV Ops gateway</h1>
        <p>This is the new <code>--nice</code> entry point. The backend is live; drop a React build into <code>cvops/web/dist</code> and this route will serve it automatically.</p>
        <div class="actions">
          <a class="button primary" href="/docs">API docs</a>
          <a class="button" href="/jobs">Jobs JSON</a>
          <a class="button" href="/scenarios">Scenarios JSON</a>
        </div>
        <div class="grid">
          <div class="metric"><b id="health">...</b><span>service</span></div>
          <div class="metric"><b id="jobs">...</b><span>jobs tracked</span></div>
          <div class="metric"><b id="scenarios">...</b><span>scenarios</span></div>
        </div>
      </div>
      <aside class="card">
        <h2>Live events</h2>
        <div id="events"><div class="event">Connecting to /events...</div></div>
      </aside>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    async function refresh() {
      try {
        const [health, jobs, scenarios] = await Promise.all([
          fetch('/health').then(r => r.json()),
          fetch('/jobs').then(r => r.json()),
          fetch('/scenarios').then(r => r.json())
        ]);
        $('health').textContent = health.status || 'ok';
        $('jobs').textContent = Array.isArray(jobs.jobs) ? jobs.jobs.length : (Array.isArray(jobs) ? jobs.length : 0);
        $('scenarios').textContent = Array.isArray(scenarios.scenarios) ? scenarios.scenarios.length : (Array.isArray(scenarios) ? scenarios.length : 0);
      } catch (err) {
        $('health').textContent = 'offline';
      }
    }
    function addEvent(text) {
      const row = document.createElement('div');
      row.className = 'event';
      row.textContent = text;
      $('events').prepend(row);
      while ($('events').children.length > 8) $('events').lastChild.remove();
    }
    function connectEvents() {
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      const ws = new WebSocket(`${proto}//${location.host}/events`);
      ws.onopen = () => addEvent('Connected.');
      ws.onmessage = (event) => addEvent(event.data);
      ws.onclose = () => setTimeout(connectEvents, 1500);
      ws.onerror = () => ws.close();
    }
    refresh();
    setInterval(refresh, 5000);
    connectEvents();
  </script>
</body>
</html>"""


def _ts_from_iso(value: Any) -> float:
    """Parse an ISO 8601 timestamp into a unix epoch; return 0.0 on failure."""
    s = str(value or "").strip()
    if not s:
        return 0.0
    try:
        from datetime import datetime
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return 0.0


def _registry_lineage_descriptor(scenario: str, versions: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a synthetic lineage record + drop list from model-registry versions.

    Versions are sorted oldest -> newest so drop_index advances monotonically.
    Returned shape mirrors LineageStore.to_dict() + DropRecord.to_dict() so the
    UI can render it through the same code path.
    """
    sorted_versions = sorted(
        (v for v in versions if isinstance(v, dict)),
        key=lambda v: str(v.get("created_at") or ""),
    )
    if not sorted_versions:
        return {}

    first = sorted_versions[0]
    last = sorted_versions[-1]
    first_ts = _ts_from_iso(first.get("created_at"))
    last_ts = _ts_from_iso(last.get("created_at"))
    base_snap = str(first.get("version_id") or "")
    head_snap = str(last.get("version_id") or "")

    lineage_dict = {
        "lineage_id": f"{_REGISTRY_LINEAGE_PREFIX}{scenario}",
        "name": f"{scenario} (model registry)",
        "sector_id": "registry",
        "sector_path": f"/registry/{scenario}",
        "description": "Synthesized from mlops/model_registry.json (read-only).",
        "base_snapshot_id": base_snap,
        "head_snapshot_id": head_snap,
        "update_strategy": "head_only",
        "replay_config": {},
        "state": "frozen",
        "tags": ["registry"],
        "metadata": {"source": "model_registry", "version_count": len(sorted_versions)},
        "created_at": first_ts,
        "updated_at": last_ts,
    }

    drops: list[dict[str, Any]] = []
    prev_drop_id: str = ""
    for idx, ver in enumerate(sorted_versions):
        ver_id = str(ver.get("version_id") or "")
        run_v = str(ver.get("run_version") or "")
        lin_meta = ver.get("lineage") if isinstance(ver.get("lineage"), dict) else {}
        metrics = ver.get("metrics") if isinstance(ver.get("metrics"), dict) else {}
        started = _ts_from_iso(ver.get("created_at"))
        finished = _ts_from_iso(ver.get("completed_at") or ver.get("created_at"))
        dur_ms = max(0, int((finished - started) * 1000)) if finished and started else 0
        drop_id = f"reg-drop-{scenario}-{idx}"
        drops.append({
            "drop_id": drop_id,
            "lineage_id": lineage_dict["lineage_id"],
            "drop_index": idx,
            "snapshot_id": ver_id,
            "parent_drop_id": prev_drop_id,
            "source": {
                "kind": "base" if idx == 0 else "training",
                "dataset_snapshot_id": str(lin_meta.get("dataset_snapshot_id") or ""),
                "base_model": str(lin_meta.get("base_model") or ""),
                "parent_version_id": str(lin_meta.get("parent_version_id") or ""),
            },
            "replay": {},
            "training_delta": {
                "source_weights": str(lin_meta.get("source_weights") or ""),
                "metrics": metrics.get("raw") if isinstance(metrics.get("raw"), dict) else {},
                "map50": metrics.get("map50"),
            },
            "sample_count": int(metrics.get("samples") or 0) if isinstance(metrics.get("samples"), (int, float)) else 0,
            "data_sha256": str(lin_meta.get("repro_manifest") or ""),
            "started_at": started,
            "finished_at": finished,
            "duration_ms": dur_ms,
            "notes": f"{run_v}  status={ver.get('status','active')}",
        })
        prev_drop_id = drop_id

    return {"lineage": lineage_dict, "drops": drops}


def _load_registry_lineages() -> dict[str, dict[str, Any]]:
    """Return {lineage_id: descriptor} for every scenario in the model registry."""
    try:
        with MODEL_REGISTRY_PATH.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    models = payload.get("models")
    if not isinstance(models, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for scenario, node in models.items():
        if not isinstance(node, dict):
            continue
        versions = node.get("versions")
        if not isinstance(versions, list):
            continue
        desc = _registry_lineage_descriptor(str(scenario), versions)
        if desc:
            out[desc["lineage"]["lineage_id"]] = desc
    return out


def _registry_lineage_matches_filters(
    lineage: dict[str, Any],
    *,
    sector_path: Optional[str],
    include_subtree: bool,
    state: Optional[str],
) -> bool:
    if state is not None and str(state).strip().lower() != str(lineage.get("state") or "").lower():
        return False
    if sector_path:
        sp = str(sector_path).strip()
        lp = str(lineage.get("sector_path") or "")
        if include_subtree:
            if sp == "/":
                return True
            if not (lp == sp or lp.startswith(sp + "/")):
                return False
        elif lp != sp:
            return False
    return True


class JobSubmitRequest(BaseModel):
    scenario: str
    version: Optional[str] = None
    model_artifact: Optional[str] = None
    image_b64: str
    source: str
    track_id: Optional[int] = None
    entry_id: Optional[int] = None
    captured_at: Optional[float] = None
    infer_overrides: Optional[dict] = None
    backbone_config_override: Optional[dict] = None


class VerifyRequest(BaseModel):
    note: Optional[str] = ""


class ModelSelectRequest(BaseModel):
    model: str


class DatasetSelectRequest(BaseModel):
    dataset: str


class GuardProfileRequest(BaseModel):
    profile: str


class HyperparamsPatchRequest(BaseModel):
    updates: dict
    reset: bool = False


class PipelinePatchRequest(BaseModel):
    updates: dict = Field(default_factory=dict)
    reset: bool = False


class PromoteRunRequest(BaseModel):
    actor: str = "cvops"
    reason: str = ""
    override: bool = False
    target_alias: str = "prod"


class AliasRevertRequest(BaseModel):
    actor: str = "cvops"
    reason: str = ""


class CarveIndexRequest(BaseModel):
    folder: str
    max_images: int = 4000


class CarvePreviewRequest(BaseModel):
    query: str
    threshold: float = 0.22
    sample: int = 24


class CarveCreateRequest(BaseModel):
    slug: str
    class_name: str
    query: str
    threshold: float = 0.22
    max_positive: int = 0
    max_negative: int = 0


class TrainKickRequest(BaseModel):
    backbone_config_override: Optional[dict] = None
    final_model_name: str = ""
    base_model_override: str = ""
    auto_fresh_on_completed_resume: bool = True
    training_assets_root: str = ""
    asset_save_root: str = ""
    save_root: str = ""
    device: str = ""


class CustomCellsDraftRequest(BaseModel):
    cells: list[Any] = Field(default_factory=list)
    scenario_datasets: list[Any] = Field(default_factory=list)


class PromoteCustomCellsRequest(BaseModel):
    template_name: str
    cell_ids: Optional[list[str]] = None


class BackboneConfigPatchRequest(BaseModel):
    patch: dict


class ScenarioCreateRequest(BaseModel):
    name: str
    display_name: str = ""
    description: str = ""
    base_model: str = ""
    dataset: str = ""
    classes: Optional[list[str]] = None
    epochs: int = 20
    imgsz: int = 640
    postproc: str = "mlops.pipeline.postproc.generic_detection:run"
    guard_profile: str = "balanced"
    backbone_type: str = "yolo_detection"
    backbone_config: dict = {}


class ArchiveImportRequest(BaseModel):
    source_paths: list[str] = Field(default_factory=list)
    name: str = ""
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    corpus_id: str = ""
    sector_id: str = ROOT_SECTOR_ID
    scenario: str = ""
    correlation_id: str = ""


class ArchiveJobKickRequest(BaseModel):
    dataset_version_id: str
    phase: str = "archive_pipeline"
    parent_snapshot_id: str = ""
    scenario: str = ""
    write_run_artifacts: bool = True
    provider_config: dict[str, Any] = Field(default_factory=dict)


class ArchiveAssemblyOverrideRequest(BaseModel):
    corpus_id: str
    dataset_version_id: str
    scope_key: str = ""
    action: str
    payload: dict[str, Any] = Field(default_factory=dict)


class ArchiveEntityMergeOverrideRequest(BaseModel):
    corpus_id: str
    snapshot_id: str
    other_entity_ids: list[str] = Field(default_factory=list)
    canonical_name: str = ""
    reject: bool = False


class ArchiveAnchorResolveOverrideRequest(BaseModel):
    corpus_id: str
    snapshot_id: str
    earliest: str = ""
    latest: str = ""
    note: str = ""


class ArchiveProposalDecisionRequest(BaseModel):
    snapshot_id: str
    decision: str
    decided_by: str = ""
    reason: str = ""


class ImageFolderConvertRequest(BaseModel):
    output_slug: Optional[str] = None
    mode: str = "full_frame"  # full_frame | empty | import_labels
    include_test: bool = True


class DatasetFolderImportRequest(BaseModel):
    source_path: str
    name: str = ""


class AudioAnalyzeRequest(BaseModel):
    path: str
    start_ms: int = 0
    end_ms: Optional[int] = None


class AudioCleanRequest(BaseModel):
    path: str
    output_name: str = ""
    noise_reduce: bool = True
    trim_silence: bool = True
    normalize: bool = True
    noise_reduction_strength: float = 0.65


class YoloDatasetTemplateCreateRequest(BaseModel):
    """Scaffold ``database/<slug>/`` as YOLO detection (images + labels + data.yaml)."""

    name: str = Field(..., min_length=1)
    classes: Optional[list[str]] = None
    unique: bool = True


class AudioDatasetCreateRequest(BaseModel):
    name: str


class AudioCollectClipRequest(BaseModel):
    dataset: str
    source_path: str
    label: str
    split: str = "train"
    start_ms: int = 0
    end_ms: Optional[int] = None
    clean: bool = True
    noise_reduce: bool = True
    trim_silence: bool = False
    normalize: bool = True
    noise_reduction_strength: float = 0.65


class AudioCopyClipRequest(BaseModel):
    source_path: str
    dest_path: str
    start_ms: int = 0
    end_ms: Optional[int] = None


_AUDIO_SOURCE_SUFFIXES = set(mlops_registry.DATASET_AUDIO_SUFFIXES) | {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
}


class LabelWriteRequest(BaseModel):
    text: str


class ClassesWriteRequest(BaseModel):
    classes: list[str]


class BulkLabelApplyRequest(BaseModel):
    class_id: int
    geometry: str = "full_image"  # full_image | center
    center_w: float = 0.5
    center_h: float = 0.5
    scope: str = "all"  # all | split | class_folder
    split: Optional[str] = None  # any split name from the dataset listing
    class_folder_name: Optional[str] = None
    only_missing: bool = True
    replace: bool = True
    limit: Optional[int] = None


class MoveToSplitRequest(BaseModel):
    relative_paths: list[str]
    target_split: str


class DatasetSubsetCloneRequest(BaseModel):
    """Create a new YOLO dataset from selected images in an existing database dataset."""

    name: str = Field(..., min_length=1)
    relative_paths: list[str] = []
    max_images: int = 0
    target_split: str = "train"
    preserve_splits: bool = False
    include_labels: bool = True
    only_labeled: bool = False
    unique: bool = True


class CopyAugmentToSplitRequest(BaseModel):
    relative_paths: list[str]
    target_split: str = "val"
    copies_per_image: int = 1
    balance_to_train: bool = False
    scale_pct: int = 100
    angle_deg: float = 0.0
    jpeg_quality: int = 90
    grayscale: bool = False
    suffix: str = "aug"


class AutoAugmentDatasetRequest(BaseModel):
    target_total: int = 1000
    folders: list[str] = []
    min_scale_pct: int = 80
    max_scale_pct: int = 120
    max_angle_deg: float = 15.0
    min_jpeg_quality: int = 70
    max_jpeg_quality: int = 100
    grayscale_probability: float = 0.15
    bgr_shuffle_probability: float = 0.15
    seed: Optional[int] = None
    val_frac: float = 0.2
    ensure_val: bool = True


SUPPORTED_TABULAR_SUFFIXES = (
    ".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pq", ".json", ".jsonl", ".ndjson",
)


def _tabular_to_csv_bytes(raw: bytes, suffix: str, filename: str = "") -> bytes:
    """Normalize an uploaded tabular file of any supported format to UTF-8 CSV bytes.

    csv/tsv are handled with the stdlib; xlsx/xls/parquet/json/jsonl go through pandas.
    Raises ValueError with an actionable message on unsupported formats or missing
    optional engines (e.g. openpyxl for .xlsx).
    """
    suffix = str(suffix or "").lower()
    if suffix == ".csv":
        # Pass through, but ensure it decodes as UTF-8 text.
        return raw
    if suffix == ".tsv":
        text = raw.decode("utf-8", errors="replace")
        rows = list(csv.reader(io.StringIO(text), delimiter="\t"))
        out = io.StringIO()
        csv.writer(out).writerows(rows)
        return out.getvalue().encode("utf-8")

    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - pandas is expected to be present
        raise ValueError(f"pandas is required to ingest {suffix} files: {exc}") from exc

    buffer = io.BytesIO(raw)
    try:
        if suffix in (".xlsx", ".xls"):
            try:
                frame = pd.read_excel(buffer)
            except ImportError as exc:
                raise ValueError(
                    "reading Excel files requires the 'openpyxl' package (pip install openpyxl)"
                ) from exc
        elif suffix in (".parquet", ".pq"):
            frame = pd.read_parquet(buffer)
        elif suffix in (".jsonl", ".ndjson"):
            frame = pd.read_json(io.BytesIO(raw), lines=True)
        elif suffix == ".json":
            try:
                frame = pd.read_json(io.BytesIO(raw))
            except ValueError:
                # Fall back to line-delimited JSON.
                frame = pd.read_json(io.BytesIO(raw), lines=True)
        else:
            raise ValueError(f"unsupported tabular extension: {suffix or '(none)'}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"failed to parse {suffix or filename} as tabular data: {exc}") from exc

    out = io.StringIO()
    frame.to_csv(out, index=False)
    return out.getvalue().encode("utf-8")


def _score_tabular_with_artifact(model_path: Path, csv_path: Path, *, max_sample: int = 50) -> dict[str, Any]:
    """Run a trained tabular model (sklearn baseline artifact) over an input CSV.

    Reproduces the baselines' preprocessing: select the saved feature_cols (numeric,
    inf->nan->0.0, float32), apply the saved StandardScaler if present, predict, and
    map integer class indices back to label_classes for classification.

    Returns the full prediction list plus metadata. Raises ValueError on unsupported
    artifacts (e.g. torch_tabular custom cells that do not pickle an sklearn predictor).
    """
    import pickle  # noqa: PLC0415

    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"pandas is required for tabular scoring: {exc}") from exc

    try:
        with model_path.open("rb") as fh:
            artifact = pickle.load(fh)
    except Exception as exc:
        raise ValueError(f"failed to load model artifact: {exc}") from exc
    if not isinstance(artifact, dict) or not hasattr(artifact.get("model"), "predict"):
        raise ValueError(
            "unsupported model artifact for batch scoring "
            "(expected a pickled sklearn baseline with a 'model' predictor)"
        )

    model = artifact["model"]
    feature_cols = [str(c) for c in (artifact.get("feature_cols") or [])]
    label_classes = [str(c) for c in (artifact.get("label_classes") or [])]
    task = str(artifact.get("task") or ("classification" if label_classes else "")).strip().lower()
    scaler_mean = artifact.get("scaler_mean")
    scaler_scale = artifact.get("scaler_scale")

    df = pd.read_csv(csv_path)
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if feature_cols:
        present = [c for c in feature_cols if c in df.columns]
        feats = df[present] if present else df.select_dtypes(include=[np.number])
    else:
        feats = df.select_dtypes(include=[np.number])
    feats = feats.select_dtypes(include=[np.number]).copy()
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if feats.shape[1] == 0:
        raise ValueError("no numeric feature columns available in the input CSV")
    x = feats.to_numpy(dtype=np.float32, copy=True)

    if isinstance(scaler_mean, list) and isinstance(scaler_scale, list) and len(scaler_mean) == x.shape[1]:
        mean = np.asarray(scaler_mean, dtype=np.float32)
        scale = np.asarray(scaler_scale, dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        x = (x - mean) / scale

    raw_preds = model.predict(x)
    predictions: list[Any] = []
    for p in raw_preds.tolist() if hasattr(raw_preds, "tolist") else list(raw_preds):
        if task != "regression" and label_classes:
            try:
                predictions.append(label_classes[int(p)])
            except (ValueError, IndexError, TypeError):
                predictions.append(p)
        else:
            predictions.append(p)

    return {
        "n_rows": int(len(predictions)),
        "task": task or ("classification" if label_classes else "regression"),
        "label_col": str(artifact.get("label_col") or ""),
        "feature_cols": feats.columns.tolist(),
        "missing_feature_cols": missing_cols,
        "label_classes": label_classes,
        "predictions": predictions,
        "sample": predictions[:max_sample],
        "model_path": str(model_path),
    }


class TabularTransformOp(BaseModel):
    """A single profile-driven fix / cleaning operation applied to a tabular dataset CSV."""

    op: str = Field(..., min_length=1)
    columns: Optional[list[str]] = None
    threshold_pct: float = 50.0
    strategy: str = "median"  # mean | median | mode | zero | constant
    fill_value: str = ""
    rename: Optional[dict[str, str]] = None  # for rename_columns: {old: new}
    method: str = "minmax"  # for normalize: minmax | zscore
    factor: float = 1.5  # for clip_outliers: IQR multiplier
    where_col: str = ""  # for filter_rows
    where_op: str = "=="  # == | != | > | >= | < | <= | contains | missing | not_missing
    where_value: str = ""
    label_col: str = ""  # for balance_classes
    max_ratio: float = 1.0  # for balance_classes: target majority:minority ratio cap


class TabularTransformRequest(BaseModel):
    """Ordered list of fix operations to apply to mlops/datasets/<slug>.csv."""

    ops: list[TabularTransformOp] = Field(default_factory=list)
    name: str = ""  # selected csv within a directory dataset (optional)


class TabularScoreRequest(BaseModel):
    """Score the rows of a tabular dataset against a trained tabular model."""

    scenario: str = ""        # resolve the model from a scenario (+ optional version)
    version: str = ""         # "", "candidate", "prod", or an explicit run version
    model_path: str = ""      # explicit path to a model.pkl artifact (overrides scenario)
    write_dataset: bool = False  # store predictions as a new tabular dataset
    output_name: str = ""     # slug stem for the written dataset
    name: str = ""            # csv selector within a directory dataset (optional)


class TabularFolderImportRequest(BaseModel):
    """Batch-import every supported tabular file found in a local folder."""

    source_path: str = Field(..., min_length=1)
    recursive: bool = False


class TabularSplitRequest(BaseModel):
    """Write reproducible train/val/test split assignments for a tabular dataset."""

    val_frac: float = 0.2
    test_frac: float = 0.0
    stratify_col: str = ""
    seed: int = 42
    write_column: bool = False
    name: str = ""  # selected csv within a directory dataset (optional)


class EvenDatasetRequest(BaseModel):
    folders: list[str] = []
    min_scale_pct: int = 80
    max_scale_pct: int = 120
    max_angle_deg: float = 15.0
    min_jpeg_quality: int = 70
    max_jpeg_quality: int = 100
    grayscale_probability: float = 0.15
    bgr_shuffle_probability: float = 0.15
    seed: Optional[int] = None
    max_copies: int = 5000


class InventoryMoveByExtRequest(BaseModel):
    ext: str
    dest_relative_dir: str
    relative_dir: str = ""
    include_hidden: bool = False
    preserve_tree: bool = True
    dry_run: bool = False


class InventoryDeleteByExtRequest(BaseModel):
    ext: str
    relative_dir: str = ""
    include_hidden: bool = False
    dry_run: bool = False


class ClearLabelsToPathsRequest(BaseModel):
    relative_paths: list[str]


class BulkLabelApplyToPathsRequest(BaseModel):
    relative_paths: list[str]
    class_id: int
    geometry: str = "full_image"  # full_image | center
    center_w: float = 0.5
    center_h: float = 0.5
    only_missing: bool = True
    replace: bool = True
    limit: Optional[int] = None


class RemapLabelsByNameRequest(BaseModel):
    old_classes: list[str]
    new_classes: list[str]
    drop_unmapped: bool = False
    limit_files: Optional[int] = None


class IngestAssetRequest(BaseModel):
    name: str = ""
    source_type: str = ""
    storage_mode: str = "reference"
    sector_id: str = ROOT_SECTOR_ID
    sector_path: str = ""
    source_uri: str = ""
    collection_id: str = ""
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    lineage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateSectorRequest(BaseModel):
    name: str
    parent_id: str = ROOT_SECTOR_ID
    parent_path: str = ""


class RenameSectorRequest(BaseModel):
    name: str


class MoveSectorRequest(BaseModel):
    parent_id: str = ROOT_SECTOR_ID


class AssignAssetSectorRequest(BaseModel):
    sector_id: str = ""
    sector_path: str = ""


class SnapshotRegisterRequest(BaseModel):
    weights_path: str
    model_type: str
    storage_mode: str = "managed_copy"
    lineage_id: Optional[str] = None
    parent_snapshot_id: Optional[str] = None
    origin: str = "imported"
    adapter_only: bool = False
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class CreateLineageRequest(BaseModel):
    name: str
    sector_id: str
    sector_path: str
    base_snapshot_id: str
    update_strategy: str = "head_only"
    replay_config: Optional[dict[str, Any]] = None
    description: str = ""
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class AddDropRequest(BaseModel):
    snapshot_id: str
    source: dict[str, Any] = Field(default_factory=dict)
    training_delta: Optional[dict[str, Any]] = None
    sample_count: int = 0
    data_sha256: str = ""
    replay: Optional[dict[str, Any]] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    notes: str = ""


class ForkLineageRequest(BaseModel):
    at_drop_index: int
    new_name: str
    description: str = ""
    update_strategy: Optional[str] = None
    replay_config: Optional[dict[str, Any]] = None


class SetLineageStateRequest(BaseModel):
    state: str


class ProvenanceBackfillRequest(BaseModel):
    lineage_id: Optional[str] = None


class CreateRangeRequest(BaseModel):
    name: str
    sector_id: str
    sector_path: str
    mode: str = "single"
    description: str = ""
    config: Optional[dict[str, Any]] = None
    tags: Optional[list[str]] = None
    metadata: Optional[dict[str, Any]] = None


class AttachSubjectRequest(BaseModel):
    snapshot_id: str
    label: str = ""


class SealGoldenSetRequest(BaseModel):
    name: str
    split_spec: dict[str, Any]
    storage_uri: str
    row_count: int
    content_sha256: str
    description: str = ""


class AddDriftRequest(BaseModel):
    name: str
    kind: str
    params: Optional[dict[str, Any]] = None


class RecordEvaluationRequest(BaseModel):
    snapshot_id: str
    golden_id: str
    metrics: dict[str, Any]
    drift_id: Optional[str] = None
    predictions_uri: str = ""
    ran_at: Optional[float] = None
    duration_ms: int = 0


class AddGateRequest(BaseModel):
    metric: str
    threshold_type: str
    threshold_value: float
    golden_id: Optional[str] = None
    baseline_snapshot_id: Optional[str] = None
    action: str = "warn"


class SettingsUpdateRequest(BaseModel):
    color_scheme: Optional[str] = None
    button_shape: Optional[str] = None
    ui_scale_pct: Optional[int] = None
    time_format: Optional[str] = None
    auto_start_dashboard: Optional[bool] = None
    dashboard_port: Optional[int] = None
    health_poll_ms: Optional[int] = None
    gallery_poll_ms: Optional[int] = None
    dashboard_poll_ms: Optional[int] = None
    show_event_pulse: Optional[bool] = None


class ScrapeCreateRequest(BaseModel):
    topic: str
    query: str = ""
    target_count: int = 50


class ScrapeTargetUpdateRequest(BaseModel):
    target_count: int


class ScrapeClassesWriteRequest(BaseModel):
    classes: list[str] = Field(default_factory=list)


class ScrapeLabelWriteRequest(BaseModel):
    boxes: list[list[float]] = Field(default_factory=list)


class ScrapeEmitRequest(BaseModel):
    val_frac: float = 0.2
    epochs: int = 20
    base_model: str = "assets/models/yolov10n.pt"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


class CvOpsService:
    def __init__(
        self,
        db_path: Path = CVOPS_DB_PATH,
        catalog_db_path: Path = CVOPS_CATALOG_DB_PATH,
        catalog_assets_root: Path = CVOPS_CATALOG_ASSETS_DIR,
        snapshot_db_path: Path = CVOPS_SNAPSHOT_DB_PATH,
        snapshot_weights_root: Path = CVOPS_SNAPSHOT_WEIGHTS_DIR,
        lineage_db_path: Path = CVOPS_LINEAGE_DB_PATH,
        provenance_db_path: Path = CVOPS_PROVENANCE_DB_PATH,
        range_db_path: Path = CVOPS_RANGE_DB_PATH,
        archive_db_path: Path = CVOPS_ARCHIVE_DB_PATH,
        archive_storage_root: Path = CVOPS_ARCHIVE_STORAGE_ROOT,
    ) -> None:
        self.store = JobStore(db_path)
        self._catalog_db_path = Path(catalog_db_path)
        self.catalog_assets_root = Path(catalog_assets_root).resolve()
        self._archive_db_path = Path(archive_db_path)
        self._archive_storage_root = Path(archive_storage_root)
        self._snapshot_db_path = Path(snapshot_db_path)
        self.snapshot_weights_root = Path(snapshot_weights_root).resolve()
        self._lineage_db_path = Path(lineage_db_path)
        self._provenance_db_path = Path(provenance_db_path)
        self._range_db_path = Path(range_db_path)
        self._catalog_store: Any = None
        self._archives_store: Any = None
        self._snapshots_store: Any = None
        self._lineages_store: Any = None
        self._provenance_store: Any = None
        self._ranges_store: Any = None
        self._archive_engine_module: Any = None
        self._lazy_store_lock = threading.RLock()
        self.app = FastAPI(title="Insight CV Ops", version="0.1.0")
        self._stop = threading.Event()
        self._worker_threads = max(1, min(16, int(os.environ.get("CVOPS_WORKER_THREADS", "2"))))
        self._executor = ThreadPoolExecutor(
            max_workers=self._worker_threads,
            thread_name_prefix="CvOpsJob",
        )
        self._slot_sem = threading.BoundedSemaphore(self._worker_threads)
        self._dispatcher = threading.Thread(target=self._dispatcher_loop, daemon=True, name="CvOpsDispatcher")
        # Live training subprocesses (job_id -> Popen). Used so a cancel can
        # SIGKILL the process group instead of waiting for a Python callback.
        self._train_procs: dict[str, subprocess.Popen] = {}
        self._train_procs_lock = threading.Lock()
        self._ws_clients: set[WebSocket] = set()
        # In-process event subscribers (e.g. the embedded Qt UI). These receive
        # the same payloads as websocket clients, but synchronously and without
        # JSON/TCP overhead. The websocket path is kept for browser clients.
        self._event_sinks: list[Callable[[dict[str, Any]], None]] = []
        self._event_bus_lock = threading.RLock()
        self._event_sinks_lock = threading.Lock()
        self._event_seq = 0
        self._event_log: list[dict[str, Any]] = []
        self._event_log_limit = max(200, int(os.environ.get("CVOPS_EVENT_REPLAY_LIMIT", "1500")))
        self._import_progress: dict[str, dict[str, Any]] = {}
        # Semantic carve (folder of images -> ImageFolder dataset) state.
        self._carve_progress: dict[str, dict[str, Any]] = {}
        self._carve_index: Any = None  # cached semantic_carve.CarveIndex
        self._carve_embedder: Any = None  # lazily-loaded clip_embed.ClipEmbedder
        self._carve_lock = threading.Lock()
        self._ws_lock = threading.RLock()
        self._ws_broadcast_lock: Optional[asyncio.Lock] = None
        self._ws_send_timeout_s = max(0.5, float(os.environ.get("CVOPS_WS_SEND_TIMEOUT_SECONDS", "2.0")))
        self._scrape_live: dict[str, threading.Event] = {}
        self._scrape_lock = threading.RLock()
        self._train_history_lock = threading.RLock()
        self._train_history: dict[str, list[dict[str, Any]]] = {}
        self._startup_resync_lock = threading.RLock()
        self._startup_resync_generation = 0
        self._startup_resync_cache: Optional[tuple[int, dict[str, Any]]] = None
        self._graph_cache_lock = threading.RLock()
        self._graph_cache_generation = 0
        self._graph_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._graph_building: set[tuple[Any, ...]] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._heartbeat_task: Optional[asyncio.Task[Any]] = None
        self._heartbeat_interval_s = max(
            2.0,
            float(os.environ.get("CVOPS_HEARTBEAT_SECONDS", "5")),
        )
        self._forecasting_runtime = None
        self._forecasting_started = False
        self._include_forecasting_router_lazy()
        self._mount_nice_frontend()
        self._register_routes()
        if os.environ.get("CVOPS_PROVENANCE_BACKFILL_ON_START", "").strip().lower() in (
            "1",
            "true",
            "yes",
        ):
            self.provenance.backfill_all(self.lineages, self.snapshots)

    @property
    def catalog(self) -> Any:
        store = self._catalog_store
        if store is None:
            with self._lazy_store_lock:
                store = self._catalog_store
                if store is None:
                    from .catalog_store import CatalogStore  # noqa: PLC0415
                    self.catalog_assets_root.mkdir(parents=True, exist_ok=True)
                    store = CatalogStore(self._catalog_db_path)
                    self._catalog_store = store
        return store

    @property
    def archives(self) -> Any:
        store = self._archives_store
        if store is None:
            with self._lazy_store_lock:
                store = self._archives_store
                if store is None:
                    from .archive_store import ArchiveStore  # noqa: PLC0415
                    store = ArchiveStore(self._archive_db_path, self._archive_storage_root)
                    self._archives_store = store
        return store

    @property
    def snapshots(self) -> Any:
        store = self._snapshots_store
        if store is None:
            with self._lazy_store_lock:
                store = self._snapshots_store
                if store is None:
                    from .snapshot_store import SnapshotStore  # noqa: PLC0415
                    self.snapshot_weights_root.mkdir(parents=True, exist_ok=True)
                    store = SnapshotStore(self._snapshot_db_path, self.snapshot_weights_root)
                    self._snapshots_store = store
        return store

    @property
    def lineages(self) -> Any:
        store = self._lineages_store
        if store is None:
            with self._lazy_store_lock:
                store = self._lineages_store
                if store is None:
                    from .lineage_store import LineageStore  # noqa: PLC0415
                    store = LineageStore(self._lineage_db_path)
                    self._lineages_store = store
        return store

    @property
    def provenance(self) -> Any:
        store = self._provenance_store
        if store is None:
            with self._lazy_store_lock:
                store = self._provenance_store
                if store is None:
                    from .provenance_store import ProvenanceStore  # noqa: PLC0415
                    store = ProvenanceStore(self._provenance_db_path)
                    self._provenance_store = store
        return store

    @property
    def ranges(self) -> Any:
        store = self._ranges_store
        if store is None:
            with self._lazy_store_lock:
                store = self._ranges_store
                if store is None:
                    from .range_store import RangeStore  # noqa: PLC0415
                    store = RangeStore(self._range_db_path)
                    self._ranges_store = store
        return store

    def _include_forecasting_router_lazy(self) -> None:
        try:
            from mlops.forecasting.api import build_router as _build_forecasting_router  # noqa: PLC0415

            service = self

            class _RuntimeProxy:
                def __getattr__(self, name: str) -> Any:
                    return getattr(service._forecasting_runtime_lazy(), name)

            self.app.include_router(_build_forecasting_router(_RuntimeProxy()))
        except Exception:
            self._forecasting_runtime = None

    def _forecasting_runtime_lazy(self) -> Any:
        runtime = self._forecasting_runtime
        if runtime is None:
            with self._lazy_store_lock:
                runtime = self._forecasting_runtime
                if runtime is None:
                    from mlops.forecasting.runtime import get_runtime as _get_forecasting_runtime  # noqa: PLC0415
                    runtime = _get_forecasting_runtime()
                    self._forecasting_runtime = runtime
                    if self._loop is not None and not self._forecasting_started:
                        try:
                            runtime.start()
                            self._forecasting_started = True
                        except Exception:
                            pass
        return runtime

    def _archive_engine(self) -> Any:
        module = self._archive_engine_module
        if module is None:
            with self._lazy_store_lock:
                module = self._archive_engine_module
                if module is None:
                    from . import archive_engine as module  # noqa: PLC0415
                    self._archive_engine_module = module
        return module

    def _mount_nice_frontend(self) -> None:
        if CVOPS_WEB_DIST_DIR.is_dir() and (CVOPS_WEB_DIST_DIR / "index.html").is_file():
            self.app.mount(
                "/nice",
                StaticFiles(directory=str(CVOPS_WEB_DIST_DIR), html=True),
                name="cvops-nice",
            )
            return

        @self.app.get("/nice", response_class=HTMLResponse)
        @self.app.get("/nice/", response_class=HTMLResponse)
        async def nice_gateway() -> str:
            return _fallback_nice_html()

    def _resolve_audio_path(self, value: str) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("audio path is required")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"audio file not found: {path}")
        if path.suffix.lower() != ".wav":
            raise ValueError("audio analysis and cleanup currently require PCM .wav input")
        return path

    def _audio_clean_output_path(self, source: Path, output_name: str = "") -> Path:
        clean_root = (CVOPS_STATE_DIR / "audio_cleaned").resolve()
        clean_root.mkdir(parents=True, exist_ok=True)
        name = str(output_name or "").strip()
        if name:
            safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name)
            if not safe.lower().endswith(".wav"):
                safe += ".wav"
        else:
            safe = f"{source.stem}.clean.wav"
        return clean_root / safe

    def _resolve_media_source_path(self, value: str) -> Path:
        raw = str(value or "").strip()
        if not raw:
            raise ValueError("media source path is required")
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = (ROOT_DIR / path).resolve()
        else:
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"media source file not found: {path}")
        return path

    def _resolve_audio_source_path(self, value: str) -> Path:
        path = self._resolve_media_source_path(value)
        if path.suffix.lower() not in _AUDIO_SOURCE_SUFFIXES:
            raise ValueError(
                "audio source must be an audio/video file "
                f"({', '.join(sorted(_AUDIO_SOURCE_SUFFIXES))})"
            )
        return path

    def _list_audio_asset_files(self) -> list[dict[str, Any]]:
        root = mlops_registry.ensure_ml_audio_root().resolve()
        items: list[dict[str, Any]] = []
        if not root.exists():
            return items
        try:
            candidates = sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix().lower())
        except Exception:
            candidates = []
        for path in candidates:
            if not path.is_file() or path.name.startswith("."):
                continue
            suffix = path.suffix.lower()
            if suffix not in _AUDIO_SOURCE_SUFFIXES:
                continue
            try:
                rel = path.relative_to(root).as_posix()
            except Exception:
                rel = path.name
            parts = Path(rel).parts
            split = ""
            label = ""
            dataset = ""
            if len(parts) >= 4 and str(parts[1]).lower() in {"train", "val", "valid", "test"}:
                dataset = str(parts[0])
                split = "val" if str(parts[1]).lower() in {"val", "valid", "test"} else "train"
                label = str(parts[2])
            elif len(parts) >= 3:
                dataset = str(parts[0])
                label = str(parts[1])
            elif len(parts) >= 2:
                label = str(parts[0])
            try:
                size = path.stat().st_size
            except Exception:
                size = 0
            items.append(
                {
                    "name": path.name,
                    "stem": path.stem,
                    "suffix": suffix,
                    "path": str(path),
                    "relative_path": rel,
                    "size": size,
                    "split": split or "source",
                    "classification_label": label,
                    "dataset": dataset,
                    "training_ready": bool(label and suffix in mlops_registry.DATASET_AUDIO_SUFFIXES),
                }
            )
        return items

    def _analysis_wav_path(self, source: Path) -> Path:
        analysis_root = (CVOPS_STATE_DIR / "audio_analysis").resolve()
        analysis_root.mkdir(parents=True, exist_ok=True)
        safe_stem = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in source.stem).strip("_")
        if not safe_stem:
            safe_stem = "audio"
        return analysis_root / f"{safe_stem}.{uuid.uuid4().hex[:10]}.analysis.wav"

    def _scrape_worker_live(self, slug: str) -> bool:
        with self._scrape_lock:
            ev = self._scrape_live.get(str(slug or ""))
            return bool(ev is not None and ev.is_set())

    def _scrape_serialize_job(self, job: Any, *, include_log: bool = False) -> dict[str, Any]:
        labels = getattr(job, "labels", {}) or {}
        processing_log = list(getattr(job, "processing_log", []) or [])
        payload = {
            "slug": str(getattr(job, "slug", "") or ""),
            "topic": str(getattr(job, "topic", "") or ""),
            "target_count": int(getattr(job, "target_count", 0) or 0),
            "state": str(getattr(job, "state", "") or ""),
            "message": str(getattr(job, "message", "") or ""),
            "raw_count": int(getattr(job, "raw_count", 0) or 0),
            "staged_count": int(getattr(job, "staged_count", 0) or 0),
            "classes": [str(item) for item in list(getattr(job, "classes", []) or [])],
            "labels": labels if include_log else {},
            "labeled_images": sum(1 for value in labels.values() if value),
            "scrape_paused": bool(getattr(job, "scrape_paused", False)),
            "scrape_generation": int(getattr(job, "scrape_generation", 0) or 0),
            "last_scrape_query": str(getattr(job, "last_scrape_query", "") or ""),
            "created_at": float(getattr(job, "created_at", 0.0) or 0.0),
            "updated_at": float(getattr(job, "updated_at", 0.0) or 0.0),
            "scrape_live": self._scrape_worker_live(str(getattr(job, "slug", "") or "")),
        }
        if include_log:
            payload["processing_log"] = processing_log
        else:
            payload["log_tail"] = processing_log[-60:]
        return payload

    def _scrape_list_jobs(self) -> list[dict[str, Any]]:
        try:
            from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415
        except Exception:
            return []
        rows: list[dict[str, Any]] = []
        for slug in mlops_registry.list_library_dataset_names():
            job = ScrapJobStore.load(slug)
            if job is None:
                continue
            rows.append(self._scrape_serialize_job(job, include_log=False))
        rows.sort(key=lambda item: float(item.get("updated_at") or 0.0), reverse=True)
        return rows

    def _scrape_dataset_root(self, slug: str) -> Path:
        return mlops_registry.resolve_library_dataset_path(mlops_registry.sanitize_library_dataset_slug(slug)).resolve()

    def _scrape_resolve_media_path(self, slug: str, kind: str, name: str) -> Path:
        root = self._scrape_dataset_root(slug)
        folder = str(kind or "").strip().lower()
        if folder not in {"raw", "staged"}:
            raise ValueError("kind must be raw or staged")
        clean_name = Path(str(name or "").strip())
        if clean_name.is_absolute() or ".." in clean_name.parts:
            raise ValueError("invalid scrape image path")
        path = (root / folder / clean_name).resolve()
        try:
            path.relative_to((root / folder).resolve())
        except ValueError as exc:
            raise ValueError("scrape image escapes dataset root") from exc
        if not path.is_file():
            raise FileNotFoundError(f"scrape image not found: {clean_name.as_posix()}")
        return path

    def _scrape_gallery_items(self, slug: str) -> list[dict[str, Any]]:
        from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

        root = self._scrape_dataset_root(slug)
        job = ScrapJobStore.load(slug)
        labels = dict(getattr(job, "labels", {}) or {})
        items: list[dict[str, Any]] = []
        for kind in ("raw", "staged"):
            folder = root / kind
            if not folder.is_dir():
                continue
            for path in sorted(folder.iterdir(), key=lambda p: p.name.lower()):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                    continue
                items.append(
                    {
                        "id": f"{kind}:{path.name}",
                        "name": path.name,
                        "kind": kind,
                        "relative_path": f"{kind}/{path.name}",
                        "size": int(path.stat().st_size if path.exists() else 0),
                        "has_label": bool(labels.get(path.name)),
                        "label_count": len(list(labels.get(path.name) or [])),
                    }
                )
        return items

    def _scrape_label_payload(self, slug: str, name: str) -> dict[str, Any]:
        from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

        job = ScrapJobStore.load(slug)
        if job is None:
            raise FileNotFoundError(f"scrape job not found: {slug}")
        boxes_raw = list((job.labels or {}).get(str(name or ""), []) or [])
        cleaned_boxes: list[list[float]] = []
        lines: list[str] = []
        for box in boxes_raw:
            if not isinstance(box, (list, tuple)) or len(box) < 5:
                continue
            try:
                row = [
                    float(box[0]),
                    float(box[1]),
                    float(box[2]),
                    float(box[3]),
                    float(box[4]),
                ]
            except Exception:
                continue
            cleaned_boxes.append(row)
            lines.append(
                f"{int(row[0])} {row[1]:.6f} {row[2]:.6f} {row[3]:.6f} {row[4]:.6f}"
            )
        return {
            "slug": slug,
            "name": name,
            "boxes": cleaned_boxes,
            "text": "\n".join(lines) + ("\n" if lines else ""),
            "has_label": bool(cleaned_boxes),
            "classes": [str(item) for item in list(job.classes or [])],
        }

    def _scrape_write_boxes(self, slug: str, name: str, boxes: list[list[float]]) -> dict[str, Any]:
        from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

        job = ScrapJobStore.load(slug)
        if job is None:
            raise FileNotFoundError(f"scrape job not found: {slug}")
        cleaned: list[list[float]] = []
        for index, box in enumerate(list(boxes or [])):
            if not isinstance(box, (list, tuple)) or len(box) < 5:
                raise ValueError(f"box {index + 1} must have 5 values")
            try:
                class_idx = int(float(box[0]))
                cx = float(box[1])
                cy = float(box[2])
                bw = float(box[3])
                bh = float(box[4])
            except Exception as exc:
                raise ValueError(f"box {index + 1} contains non-numeric values") from exc
            if class_idx < 0:
                raise ValueError(f"box {index + 1} has invalid class index")
            for label, value in (("cx", cx), ("cy", cy), ("w", bw), ("h", bh)):
                if value < 0.0 or value > 1.0:
                    raise ValueError(f"box {index + 1} has {label} outside [0, 1]")
            cleaned.append([float(class_idx), cx, cy, bw, bh])
        labels = dict(job.labels or {})
        if cleaned:
            labels[str(name)] = cleaned
        else:
            labels.pop(str(name), None)
        ScrapJobStore.update(slug, labels=labels, state="labeling", message=f"saved labels for {name}")
        ScrapJobStore.append_log(slug, f"Saved {len(cleaned)} box(es) for {name}")
        updated = ScrapJobStore.load(slug)
        return self._scrape_label_payload(str(getattr(updated, "slug", slug) or slug), name)

    def _scrape_start_thread(self, slug: str, query: str, *, clear_raw: bool = False) -> None:
        from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

        with self._scrape_lock:
            existing = self._scrape_live.get(slug)
            if existing is not None and existing.is_set():
                raise ValueError("a scrape worker is already running for this job")

        preload = ScrapJobStore.load(slug)
        if preload is None:
            raise FileNotFoundError(f"scrape job not found: {slug}")

        worker_gen = ScrapJobStore.bump_scrape_generation(slug)
        if worker_gen is None:
            raise FileNotFoundError(f"could not start scrape worker for: {slug}")

        ev = threading.Event()
        ev.set()
        with self._scrape_lock:
            self._scrape_live[slug] = ev

        def poll_continue() -> bool:
            while True:
                live_job = ScrapJobStore.load(slug)
                if live_job is None:
                    return False
                if int(live_job.scrape_generation or 0) != int(worker_gen):
                    return False
                if not bool(live_job.scrape_paused):
                    return True
                time.sleep(0.25)

        def _run() -> None:
            try:
                ScrapJobStore.append_log(
                    slug,
                    "Worker thread entered — importing Selenium/Chrome stack (first run can take a while).",
                )
                from mlops.scrap.selenium_search import search_google_images  # noqa: PLC0415
                from mlops.scrap import filter as scrap_filter  # noqa: PLC0415

                def jlog(msg: str) -> None:
                    ScrapJobStore.append_log(slug, msg)

                def still_owner() -> bool:
                    current = ScrapJobStore.load(slug)
                    return current is not None and int(current.scrape_generation or 0) == int(worker_gen)

                snapshot = ScrapJobStore.load(slug)
                if snapshot is None:
                    return
                target_count = int(snapshot.target_count or 0)
                ScrapJobStore.update(
                    slug,
                    state="scraping",
                    last_scrape_query=query,
                    message=f"searching '{query}'",
                    scrape_paused=False,
                )
                base = self._scrape_dataset_root(slug)
                raw_dir = base / "raw"
                staged_dir = base / "staged"
                raw_dir.mkdir(parents=True, exist_ok=True)
                staged_dir.mkdir(parents=True, exist_ok=True)
                jlog(f"Worker thread started for slug={slug!r} (generation {worker_gen})")
                jlog(f"Parameters: query={query!r}, restart_raw={clear_raw}")
                jlog(f"Target raw count={target_count}")
                jlog(f"raw_dir={raw_dir}")
                jlog(f"staged_dir={staged_dir}")

                if clear_raw:
                    jlog("Clearing raw/ (restart downloads)")
                    shutil.rmtree(raw_dir, ignore_errors=True)
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    ScrapJobStore.update(slug, raw_count=0, message="cleared raw/; downloading")

                if not still_owner():
                    jlog("Aborted — job superseded before download phase.")
                    return

                raw_now = sum(1 for path in raw_dir.iterdir() if path.is_file())
                remaining = max(0, target_count - raw_now)
                jlog(f"Raw files on disk={raw_now}; requesting up to {remaining} new download(s).")
                result = search_google_images(
                    query,
                    remaining,
                    raw_dir,
                    on_progress=jlog,
                    poll_continue=poll_continue,
                )

                if not still_owner():
                    return

                if bool(getattr(result, "cancelled", False)):
                    raw_after = sum(1 for path in raw_dir.iterdir() if path.is_file())
                    jlog(
                        f"Download phase cancelled/superseded (saved this session={len(getattr(result, 'saved', []) or [])}; raw_total={raw_after})."
                    )
                    ScrapJobStore.update(
                        slug,
                        state="paused_downloads",
                        raw_count=raw_after,
                        message="downloads interrupted — use Continue or Restart",
                    )
                    return

                raw_after_done = sum(1 for path in raw_dir.iterdir() if path.is_file())
                ScrapJobStore.update(
                    slug,
                    raw_count=raw_after_done,
                    message=(
                        f"downloaded to raw_total={raw_after_done}"
                        f" (attempted {int(getattr(result, 'attempted', 0) or 0)}, skipped {int(getattr(result, 'skipped', 0) or 0)}); staging"
                    ),
                )
                jlog(
                    f"Download phase finished: saved={len(getattr(result, 'saved', []) or [])} attempted={int(getattr(result, 'attempted', 0) or 0)} skipped={int(getattr(result, 'skipped', 0) or 0)} raw_total={raw_after_done}"
                )

                if not poll_continue():
                    return

                jlog("Beginning dedupe_and_stage (perceptual hash, keeping small images)…")
                stage = scrap_filter.dedupe_and_stage(
                    raw_dir,
                    staged_dir,
                    min_size=0,
                    on_progress=jlog,
                    poll_continue=poll_continue,
                )

                if not still_owner():
                    return

                staged_count = len(list(getattr(stage, "staged", []) or []))
                jlog(
                    f"Staging finished: staged_files={staged_count} skipped_small={int(getattr(stage, 'skipped_small', 0) or 0)} skipped_dup={int(getattr(stage, 'skipped_dup', 0) or 0)} skipped_unreadable={int(getattr(stage, 'skipped_unreadable', 0) or 0)}"
                )
                ScrapJobStore.update(
                    slug,
                    state="staged",
                    staged_count=staged_count,
                    message=(
                        f"staged {staged_count}; kept readable small images; dup={int(getattr(stage, 'skipped_dup', 0) or 0)} unreadable={int(getattr(stage, 'skipped_unreadable', 0) or 0)}"
                    ),
                )
                jlog("Job state set to staged. You can label images on the Label tab.")
            except Exception as exc:
                try:
                    current = ScrapJobStore.load(slug)
                    if current is not None and int(current.scrape_generation or 0) != int(worker_gen):
                        return
                    ScrapJobStore.append_log(slug, f"ERROR: {exc}")
                    ScrapJobStore.update(slug, state="error", message=f"scrape failed: {exc}")
                except Exception:
                    pass
            finally:
                ev.clear()
                with self._scrape_lock:
                    live = self._scrape_live.get(slug)
                    if live is ev:
                        self._scrape_live.pop(slug, None)

        threading.Thread(target=_run, name=f"ScrapeJob-{slug}", daemon=True).start()

    def _recover_orphaned_jobs(self) -> None:
        """Mark any queued/running jobs from a prior session as error.

        If the process was killed mid-training those jobs stay in 'queued' or
        'running' state in the DB forever, making scenarios appear stuck in
        'training' and blocking re-training on every restart.
        """
        try:
            jobs = self.store.list_jobs(limit=500)
        except Exception:
            return
        for job in jobs:
            if job.state in ("queued", "running"):
                try:
                    self.store.set_job_state(
                        job.job_id,
                        "error",
                        error="interrupted: service restarted while job was active",
                    )
                except Exception:
                    pass

    def scenarios_payload(self) -> dict[str, Any]:
        try:
            base = mlops_registry.list_enabled_scenarios()
        except Exception as exc:
            return {"scenarios": [], "error": str(exc)}
        enriched: list[dict[str, Any]] = []
        active = self._scenarios_with_active_training()
        for item in base:
            name = str(item.get("name") or "")
            if not name:
                continue
            status_payload = mlops_registry.get_scenario_status(name)
            if name in active:
                status_payload["status"] = "training"
            # Preserve description/display from enabled listing in case
            # status helper swallowed them on error.
            status_payload.setdefault("display_name", item.get("display_name"))
            status_payload.setdefault("description", item.get("description"))
            enriched.append(status_payload)
        return {"scenarios": enriched, "error": ""}

    def scenario_status_payload(self, scenario: str) -> dict[str, Any]:
        payload = mlops_registry.get_scenario_status(scenario)
        if scenario in self._scenarios_with_active_training():
            payload["status"] = "training"
        return payload

    def jobs_payload(self) -> dict[str, Any]:
        return {"jobs": [job.to_dict() for job in self.store.list_jobs(limit=500)]}

    def training_progress_payload(self, job_id: str) -> dict[str, Any]:
        with self._train_history_lock:
            events = list(self._train_history.get(job_id, []))
        epoch_events = [e for e in events if str(e.get("event") or "") != "log"]
        log_events = [e for e in events if str(e.get("event") or "") == "log"]
        return {
            "job_id": job_id,
            "events": events,
            "epoch_events": epoch_events,
            "log_events": log_events,
        }

    # ------------------------------------------------------------------
    # Reusable read payloads. These back both the FastAPI GET routes and the
    # Qt UI's in-process direct-read path (window._direct_service_json), so a
    # desktop read never has to round-trip the loopback socket. Each is a pure
    # read and may raise on error/not-found; the HTTP handlers wrap them in the
    # appropriate HTTPException, and the in-process caller falls back to HTTP on
    # any exception so error/status semantics stay identical.
    def models_payload(self) -> dict[str, Any]:
        return {"models": mlops_registry.list_available_models()}

    def sectors_payload(self) -> dict[str, Any]:
        items = [s.to_dict() for s in self.catalog.list_sectors()]
        return {"count": len(items), "sectors": items}

    def scenario_history_payload(self, scenario: str) -> dict[str, Any]:
        runs = mlops_registry.list_scenario_runs(scenario)
        return {"scenario": scenario, "runs": runs, "count": len(runs)}

    def scenario_pipeline_payload(self, scenario: str) -> dict[str, Any]:
        policy = mlops_registry.get_scenario_ci_cd_policy(scenario)
        candidate = resolve_alias(scenario, "candidate")
        staging = resolve_alias(scenario, "staging")
        prod = resolve_alias(scenario, "prod")
        runs = mlops_registry.list_scenario_runs(scenario)
        latest_gate = None
        for run in sorted(runs, key=lambda item: int(item.get("version_number") or -1), reverse=True):
            ci_cd_state = run.get("ci_cd") if isinstance(run.get("ci_cd"), dict) else {}
            if ci_cd_state.get("gate_status"):
                latest_gate = {
                    "run_version": str(run.get("version") or ""),
                    "model_version_id": str(run.get("model_version_id") or ""),
                    "ci_cd": dict(ci_cd_state),
                }
                break
        active_jobs = [
            job.to_dict()
            for job in self.store.list_jobs(limit=500)
            if job.scenario == scenario
            and job.job_type == "train"
            and job.state in {"queued", "running"}
        ]
        return {
            "scenario": scenario,
            "ci_cd": policy,
            "candidate": candidate or {},
            "staging": staging or {},
            "prod": prod or {},
            "latest_gate": latest_gate or {},
            "active_jobs": active_jobs,
            "runs": runs,
        }

    def scenario_cards_payload(self, scenario: str) -> dict[str, Any]:
        return mlops_cards.build_scenario_cards(scenario)

    def custom_cells_payload(self, scenario: str) -> dict[str, Any]:
        return read_draft(scenario)

    def job_payload(self, job_id: str) -> dict[str, Any]:
        return self.store.get_job(job_id).to_dict()

    def job_result_payload(self, job_id: str) -> dict[str, Any]:
        result = self.store.get_result(job_id)
        if result is None:
            raise KeyError(job_id)
        return result

    def database_list_payload(self) -> dict[str, Any]:
        names = mlops_registry.list_library_dataset_names()
        root = mlops_registry.DATABASE_ROOT.resolve()
        audio_root = mlops_registry.ensure_ml_audio_root().resolve()
        categories: dict[str, str] = {}
        for name in names:
            try:
                ds_path = mlops_registry.resolve_library_dataset_path(name)
                fmt = mlops_registry.detect_library_dataset_format(ds_path)
                categories[name] = mlops_registry.dataset_category(fmt)
            except Exception:
                categories[name] = mlops_registry.DATASET_CATEGORY_IMAGE
        return {
            "datasets": names,
            "categories": categories,
            "tabular_datasets": mlops_registry.list_tabular_dataset_entries(),
            "text_datasets": mlops_registry.list_text_dataset_entries(),
            "root": str(root),
            "audio_root": str(audio_root),
        }

    def database_dataset_payload(self, slug: str) -> dict[str, Any]:
        dataset_root = mlops_registry.resolve_library_dataset_path(slug)
        payload = mlops_registry.inspect_library_dataset_at(dataset_root)
        items = payload.get("images") if isinstance(payload, dict) else []
        folders = payload.get("folders") if isinstance(payload, dict) else []
        audio_files = payload.get("audio_files") if isinstance(payload, dict) else []
        csv_files = payload.get("csv_files") if isinstance(payload, dict) else []
        split_counts = payload.get("split_counts") if isinstance(payload, dict) else {}
        fmt = str((payload or {}).get("format") or mlops_registry.LIBRARY_DATASET_FORMAT_UNKNOWN)
        classes = (payload or {}).get("classes") if isinstance(payload, dict) else []
        count = int((payload or {}).get("count") or len(items or []) or len(audio_files or []))
        return {
            "slug": slug,
            "path": str(dataset_root.resolve()),
            "format": fmt,
            "category": mlops_registry.dataset_category(fmt),
            "count": count,
            "content_sha256": self._dataset_fingerprint(
                dataset_root,
                fmt=fmt,
                count=count,
                split_counts=split_counts if isinstance(split_counts, dict) else {},
                classes=classes if isinstance(classes, list) else [],
            ),
            "images": items,
            "folders": folders or [],
            "audio_files": audio_files or [],
            "csv_files": csv_files or [],
            "split_counts": split_counts,
            "classes": classes,
            "detection_label_count": (payload or {}).get("detection_label_count", 0) if isinstance(payload, dict) else 0,
            "missing_detection_label_count": (payload or {}).get("missing_detection_label_count", 0) if isinstance(payload, dict) else 0,
        }

    def _mark_startup_resync_dirty(self) -> None:
        with self._startup_resync_lock:
            self._startup_resync_generation += 1
            self._startup_resync_cache = None

    def startup_resync_payload(self) -> dict[str, Any]:
        with self._startup_resync_lock:
            cached = self._startup_resync_cache
            generation = self._startup_resync_generation
            if cached is not None and cached[0] == generation:
                return copy.deepcopy(cached[1])

        payload = self._build_startup_resync_payload()
        with self._startup_resync_lock:
            if self._startup_resync_generation == generation:
                self._startup_resync_cache = (generation, copy.deepcopy(payload))
        return payload

    def _build_startup_resync_payload(self) -> dict[str, Any]:
        errors: list[str] = []
        jobs: list[dict[str, Any]] = []
        training_events: list[dict[str, Any]] = []
        scenarios: list[dict[str, Any]] = []

        try:
            payload = self.jobs_payload()
            jobs = [j for j in list(payload.get("jobs") or []) if isinstance(j, dict)]
        except Exception as exc:
            errors.append(f"jobs: {exc}")

        for job in [j for j in jobs if str(j.get("job_type") or "") == "train"][:18]:
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            try:
                payload = self.training_progress_payload(job_id)
            except Exception as exc:
                errors.append(f"training_progress/{job_id}: {exc}")
                continue
            events = payload.get("events") if isinstance(payload, dict) else []
            if not isinstance(events, list):
                continue
            scenario = str(job.get("scenario") or "")
            for event in events:
                if not isinstance(event, dict):
                    continue
                merged = dict(event)
                if not merged.get("scenario"):
                    merged["scenario"] = scenario
                if not merged.get("job_id"):
                    merged["job_id"] = job_id
                training_events.append(merged)

        try:
            payload = self.scenarios_payload()
            scenarios = [s for s in list(payload.get("scenarios") or []) if isinstance(s, dict)]
            err = str(payload.get("error") or "").strip()
            if err:
                errors.append(f"scenarios: {err}")
        except Exception as exc:
            errors.append(f"scenarios: {exc}")

        return {
            "jobs": jobs,
            "training_events": training_events,
            "scenarios": scenarios,
            "errors": errors[:8],
            "event_seq": self.latest_event_seq(),
        }

    def _json_cache_response(
        self,
        payload: dict[str, Any],
        request: Request,
        *,
        max_age: int = 2,
        stale_while_revalidate: int = 20,
    ) -> Response:
        raw = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        ).encode("utf-8")
        etag = f'W/"cvops-{hashlib.sha256(raw).hexdigest()[:24]}"'
        headers = {
            "Cache-Control": (
                f"private, max-age={max(0, int(max_age))}, "
                f"stale-while-revalidate={max(0, int(stale_while_revalidate))}"
            ),
            "ETag": etag,
        }
        candidates = [tag.strip() for tag in str(request.headers.get("if-none-match") or "").split(",")]
        if etag in candidates or "*" in candidates:
            return Response(status_code=304, headers=headers)
        return Response(content=raw, media_type="application/json", headers=headers)

    @staticmethod
    def _normalize_graph_layer(layer: str) -> str:
        value = str(layer or "full").strip().lower()
        return "core" if value == "core" else "full"

    @staticmethod
    def _normalize_graph_entity_types(entity_types: str) -> tuple[str, ...]:
        return tuple(sorted({t.strip() for t in str(entity_types or "").split(",") if t.strip()}))

    def _graph_cache_key(
        self,
        *,
        layer: str,
        entity_types: str = "",
        scenario: str = "",
        depth: int = 2,
        since_ts: Optional[float] = None,
    ) -> tuple[Any, ...]:
        since_key = None if since_ts is None else round(float(since_ts), 3)
        return (
            self._normalize_graph_layer(layer),
            self._normalize_graph_entity_types(entity_types),
            str(scenario or "").strip(),
            int(depth),
            since_key,
        )

    def _graph_entity_types_for_layer(self, layer: str, entity_types: str) -> Optional[list[str]]:
        explicit = self._normalize_graph_entity_types(entity_types)
        if explicit:
            return list(explicit)
        if self._normalize_graph_layer(layer) == "core":
            return list(_GRAPH_CORE_ENTITY_TYPES)
        return None

    def _build_ontology_graph_payload(
        self,
        *,
        layer: str,
        entity_types: str = "",
        scenario: str = "",
        depth: int = 2,
        since_ts: Optional[float] = None,
    ) -> dict[str, Any]:
        from .ontology import build_graph as _build_graph

        normalized_layer = self._normalize_graph_layer(layer)
        types_list = self._graph_entity_types_for_layer(normalized_layer, entity_types)
        scen = str(scenario or "").strip() or None
        extra = [d["lineage"] for d in _load_registry_lineages().values()]
        graph = _build_graph(
            job_store=self.store,
            snapshots=self.snapshots,
            lineages=self.lineages,
            ranges=self.ranges,
            catalog=self.catalog,
            entity_types=types_list,
            scenario=scen,
            depth=depth,
            since_ts=since_ts,
            job_limit=80 if normalized_layer == "core" else 150,
            extra_lineages=extra,
            provenance=self.provenance,
        )
        graph.setdefault("cache", {})
        graph["cache"].update(
            {
                "layer": normalized_layer,
                "generated_at": time.time(),
                "stale": False,
                "pending": False,
            }
        )
        return graph

    def _store_graph_cache(self, key: tuple[Any, ...], generation: int, graph: dict[str, Any]) -> dict[str, Any]:
        should_reschedule = False
        with self._graph_cache_lock:
            graph = dict(graph)
            meta = dict(graph.get("cache") or {})
            meta["generation"] = generation
            should_reschedule = generation != self._graph_cache_generation
            meta["stale"] = should_reschedule
            graph["cache"] = meta
            self._graph_cache[key] = {
                "generation": generation,
                "built_at": time.time(),
                "graph": graph,
            }
            self._graph_building.discard(key)
        if should_reschedule:
            layer, entity_types, scenario, depth, since_ts = key
            self._schedule_graph_rebuild(
                layer=str(layer),
                entity_types=",".join(entity_types),
                scenario=str(scenario or ""),
                depth=int(depth),
                since_ts=since_ts,
            )
        return graph

    def _schedule_graph_rebuild(
        self,
        *,
        layer: str,
        entity_types: str = "",
        scenario: str = "",
        depth: int = 2,
        since_ts: Optional[float] = None,
    ) -> None:
        key = self._graph_cache_key(
            layer=layer,
            entity_types=entity_types,
            scenario=scenario,
            depth=depth,
            since_ts=since_ts,
        )
        with self._graph_cache_lock:
            if key in self._graph_building:
                return
            self._graph_building.add(key)
            generation = self._graph_cache_generation

        def _worker() -> None:
            try:
                graph = self._build_ontology_graph_payload(
                    layer=layer,
                    entity_types=entity_types,
                    scenario=scenario,
                    depth=depth,
                    since_ts=since_ts,
                )
                self._store_graph_cache(key, generation, graph)
            except Exception:
                with self._graph_cache_lock:
                    self._graph_building.discard(key)

        threading.Thread(
            target=_worker,
            name=f"CvOpsGraphCache-{self._normalize_graph_layer(layer)}",
            daemon=True,
        ).start()

    def _prewarm_graph_cache(self) -> None:
        for layer in _GRAPH_DEFAULT_LAYERS:
            self._schedule_graph_rebuild(layer=layer)

    def _mark_graph_cache_dirty(self, reason: str = "") -> None:
        with self._graph_cache_lock:
            self._graph_cache_generation += 1
        self._prewarm_graph_cache()

    def _get_ontology_graph_cached(
        self,
        *,
        layer: str,
        entity_types: str = "",
        scenario: str = "",
        depth: int = 2,
        since_ts: Optional[float] = None,
        stale_ok: bool = True,
    ) -> dict[str, Any]:
        key = self._graph_cache_key(
            layer=layer,
            entity_types=entity_types,
            scenario=scenario,
            depth=depth,
            since_ts=since_ts,
        )
        with self._graph_cache_lock:
            entry = self._graph_cache.get(key)
            generation = self._graph_cache_generation
            building = key in self._graph_building
        if entry is not None:
            graph = dict(entry.get("graph") or {})
            meta = dict(graph.get("cache") or {})
            is_stale = int(entry.get("generation", -1)) != generation
            meta.update(
                {
                    "stale": is_stale,
                    "pending": bool(building),
                    "generation": int(entry.get("generation", -1)),
                    "served_at": time.time(),
                }
            )
            graph["cache"] = meta
            if is_stale:
                self._schedule_graph_rebuild(
                    layer=layer,
                    entity_types=entity_types,
                    scenario=scenario,
                    depth=depth,
                    since_ts=since_ts,
                )
            return graph

        normalized_layer = self._normalize_graph_layer(layer)
        if normalized_layer == "full" and stale_ok:
            self._schedule_graph_rebuild(
                layer=layer,
                entity_types=entity_types,
                scenario=scenario,
                depth=depth,
                since_ts=since_ts,
            )
            core_key = self._graph_cache_key(
                layer="core",
                entity_types="",
                scenario=scenario,
                depth=depth,
                since_ts=since_ts,
            )
            with self._graph_cache_lock:
                core_entry = self._graph_cache.get(core_key)
            if core_entry is not None:
                graph = dict(core_entry.get("graph") or {})
                meta = dict(graph.get("cache") or {})
                meta.update(
                    {
                        "layer": "core",
                        "requested_layer": "full",
                        "pending": True,
                        "stale": True,
                        "served_at": time.time(),
                    }
                )
                graph["cache"] = meta
                return graph
            return {
                "nodes": [],
                "edges": [],
                "cache": {
                    "layer": normalized_layer,
                    "pending": True,
                    "stale": True,
                    "served_at": time.time(),
                },
            }

        graph = self._build_ontology_graph_payload(
            layer=layer,
            entity_types=entity_types,
            scenario=scenario,
            depth=depth,
            since_ts=since_ts,
        )
        return self._store_graph_cache(key, generation, graph)

    def _register_routes(self) -> None:
        @self.app.on_event("startup")
        async def _startup() -> None:
            try:
                from insight_local.cvops.__main__ import _boot_step as _bstep  # noqa: PLC0415
                _bstep("server accepting connections")
            except Exception:
                pass
            self._recover_orphaned_jobs()
            self._loop = asyncio.get_running_loop()
            self._ws_broadcast_lock = asyncio.Lock()
            if not self._dispatcher.is_alive():
                self._dispatcher.start()
            if self._heartbeat_task is None or self._heartbeat_task.done():
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            if self._forecasting_runtime is not None:
                try:
                    self._forecasting_runtime.start()
                    self._forecasting_started = True
                except Exception:
                    pass
            self._warm_inference_stack()
            if str(os.environ.get("CVOPS_PREWARM_GRAPH_ON_START", "0")).strip().lower() in {"1", "true", "yes", "on"}:
                self._prewarm_graph_cache()

        @self.app.on_event("shutdown")
        async def _shutdown() -> None:
            self._stop.set()
            if self._heartbeat_task is not None:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                except Exception:
                    pass
                self._heartbeat_task = None
            self._ws_broadcast_lock = None
            if self._dispatcher.is_alive():
                self._dispatcher.join(timeout=1.0)
            try:
                self._executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            if self._forecasting_runtime is not None and self._forecasting_started:
                try:
                    self._forecasting_runtime.stop()
                except Exception:
                    pass
            self.store.close()
            for store in (
                self._catalog_store,
                self._archives_store,
                self._snapshots_store,
                self._lineages_store,
                self._provenance_store,
                self._ranges_store,
            ):
                close = getattr(store, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception:
                        pass

        @self.app.get("/health")
        def health(request: Request) -> Response:
            # Sync (threadpool) on purpose — see the /scenarios handler. These
            # read endpoints are polled by the UI on startup; as blocking
            # `async def` they serialize on the single event loop and starve
            # each other, which is what makes /scenarios miss its client
            # deadline during startup / active training.
            return self._json_cache_response(self._health_snapshot(), request, max_age=1)

        @self.app.get("/system/metrics")
        async def system_metrics() -> dict[str, Any]:
            out: dict[str, Any] = {
                "cpu_pct": None,
                "ram_used_gb": None,
                "ram_total_gb": None,
                "ram_pct": None,
                "gpus": [],
            }
            try:
                import psutil as _psutil
                out["cpu_pct"] = float(_psutil.cpu_percent(interval=None))
                vm = _psutil.virtual_memory()
                gib = 1024 ** 3
                out["ram_used_gb"] = round((vm.total - vm.available) / gib, 2)
                out["ram_total_gb"] = round(vm.total / gib, 2)
                out["ram_pct"] = float(vm.percent)
            except Exception:
                pass
            try:
                from mlops.pipeline.system_guard import detect_system_specs
                specs = detect_system_specs()
                gib = 1024 ** 3
                for entry in specs.gpus:
                    if not isinstance(entry, dict):
                        continue
                    idx = int(entry.get("index", 0) or 0)
                    backend = str(entry.get("backend") or "").lower()
                    gpu_out: dict[str, Any] = {
                        "index": idx,
                        "name": str(entry.get("name") or ""),
                        "backend": backend,
                        "memory_gb": entry.get("memory_gb"),
                        "util_pct": None,
                        "mem_used_gb": None,
                        "mem_total_gb": None,
                        "temp_c": None,
                    }
                    if backend == "cuda":
                        try:
                            import pynvml as _pynvml
                            _pynvml.nvmlInit()
                            handle = _pynvml.nvmlDeviceGetHandleByIndex(idx)
                            util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
                            gpu_out["util_pct"] = float(util.gpu)
                            mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
                            gpu_out["mem_used_gb"] = round(mem.used / gib, 2)
                            gpu_out["mem_total_gb"] = round(mem.total / gib, 2)
                            gpu_out["temp_c"] = float(
                                _pynvml.nvmlDeviceGetTemperature(handle, _pynvml.NVML_TEMPERATURE_GPU)
                            )
                        except Exception:
                            pass
                    out["gpus"].append(gpu_out)
            except Exception:
                pass
            return out

        @self.app.get("/diagnostics/summary")
        def diagnostics_summary() -> dict[str, Any]:
            summary = self._ecosystem_summary()
            summary["settings"] = self._settings_payload()["settings"]
            summary["recent_errors"] = self._recent_errors_payload()["errors"][:12]
            return summary

        @self.app.get("/diagnostics/errors")
        async def diagnostics_errors() -> dict[str, Any]:
            return self._recent_errors_payload()

        @self.app.get("/settings")
        async def get_settings() -> dict[str, Any]:
            return self._settings_payload()

        @self.app.put("/settings")
        async def update_settings(req: SettingsUpdateRequest) -> dict[str, Any]:
            return self._apply_settings_patch(req)

        @self.app.get("/scrape/jobs")
        async def scrape_jobs() -> dict[str, Any]:
            return {"jobs": self._scrape_list_jobs()}

        @self.app.post("/scrape/jobs")
        async def scrape_create_job(req: ScrapeCreateRequest) -> dict[str, Any]:
            topic = str(req.topic or "").strip()
            if not topic:
                raise HTTPException(status_code=400, detail="topic is required")
            query = str(req.query or "").strip() or topic
            target_count = max(10, min(5000, int(req.target_count or 50)))
            try:
                from mlops.scrap.jobs import JobState as ScrapJobState  # noqa: PLC0415
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                raw_slug = topic.lower().replace(" ", "_").replace("-", "_")
                raw_slug = "".join(ch for ch in raw_slug if ch.isalnum() or ch == "_") or "topic"
                slug = mlops_registry.pick_unique_library_dataset_slug(f"scrap_{raw_slug}")
                mlops_registry.create_library_dataset_root(slug)
                ScrapJobStore.save(
                    ScrapJobState(
                        slug=slug,
                        topic=topic,
                        target_count=target_count,
                        state="pending",
                        message="job created",
                        processing_log=[],
                    )
                )
                self._scrape_start_thread(slug, query, clear_raw=False)
                job = ScrapJobStore.load(slug)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"could not create scrape job: {exc}") from exc
            return self._scrape_serialize_job(job, include_log=True) if job is not None else {"slug": slug}

        @self.app.get("/scrape/jobs/{slug}")
        async def scrape_job_status(slug: str) -> dict[str, Any]:
            try:
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                job = ScrapJobStore.load(slug)
                if job is None:
                    raise FileNotFoundError(f"scrape job not found: {slug}")
                payload = self._scrape_serialize_job(job, include_log=True)
                payload["gallery"] = self._scrape_gallery_items(slug)
                payload["status"] = {
                    "slug": payload["slug"],
                    "topic": payload["topic"],
                    "state": payload["state"],
                    "message": payload["message"],
                    "raw_count": payload["raw_count"],
                    "staged_count": payload["staged_count"],
                    "labeled_images": payload["labeled_images"],
                    "classes": payload["classes"],
                }
                return payload
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"scrape status failed: {exc}") from exc

        @self.app.put("/scrape/jobs/{slug}/target")
        async def scrape_update_target(slug: str, req: ScrapeTargetUpdateRequest) -> dict[str, Any]:
            try:
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                target_count = max(10, min(5000, int(req.target_count or 50)))
                job = ScrapJobStore.update(slug, target_count=target_count, message=f"target set to {target_count}")
                if job is None:
                    raise FileNotFoundError(f"scrape job not found: {slug}")
                ScrapJobStore.append_log(slug, f"Target count updated to {target_count}")
                return self._scrape_serialize_job(job, include_log=True)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"target update failed: {exc}") from exc

        @self.app.post("/scrape/jobs/{slug}/pause")
        async def scrape_toggle_pause(slug: str) -> dict[str, Any]:
            try:
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                job = ScrapJobStore.load(slug)
                if job is None:
                    raise FileNotFoundError(f"scrape job not found: {slug}")
                paused = not bool(job.scrape_paused)
                updated = ScrapJobStore.update(slug, scrape_paused=paused)
                ScrapJobStore.append_log(slug, "Paused scrape worker" if paused else "Resumed scrape worker")
                final = updated or ScrapJobStore.load(slug)
                return self._scrape_serialize_job(final, include_log=True) if final is not None else {}
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"pause toggle failed: {exc}") from exc

        @self.app.post("/scrape/jobs/{slug}/continue")
        async def scrape_continue(slug: str) -> dict[str, Any]:
            try:
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                job = ScrapJobStore.load(slug)
                if job is None:
                    raise FileNotFoundError(f"scrape job not found: {slug}")
                query = str(job.last_scrape_query or job.topic or "").strip()
                if not query:
                    raise ValueError("job has no saved query to continue")
                self._scrape_start_thread(slug, query, clear_raw=False)
                fresh = ScrapJobStore.load(slug)
                return self._scrape_serialize_job(fresh, include_log=True) if fresh is not None else {}
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"continue scrape failed: {exc}") from exc

        @self.app.post("/scrape/jobs/{slug}/restart")
        async def scrape_restart(slug: str) -> dict[str, Any]:
            try:
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                job = ScrapJobStore.load(slug)
                if job is None:
                    raise FileNotFoundError(f"scrape job not found: {slug}")
                query = str(job.last_scrape_query or job.topic or "").strip()
                if not query:
                    raise ValueError("job has no saved query to restart")
                self._scrape_start_thread(slug, query, clear_raw=True)
                fresh = ScrapJobStore.load(slug)
                return self._scrape_serialize_job(fresh, include_log=True) if fresh is not None else {}
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"restart scrape failed: {exc}") from exc

        @self.app.get("/scrape/jobs/{slug}/gallery")
        async def scrape_gallery(slug: str) -> dict[str, Any]:
            try:
                return {"slug": slug, "items": self._scrape_gallery_items(slug)}
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"scrape gallery failed: {exc}") from exc

        @self.app.get("/scrape/jobs/{slug}/thumb/{kind}/{name:path}")
        async def scrape_thumb(slug: str, kind: str, name: str, max_side: int = Query(160)) -> Response:
            try:
                match = self._scrape_resolve_media_path(slug, kind, name)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            image = cv2.imread(str(match))
            if image is None:
                raise HTTPException(status_code=400, detail="unable to decode scrape image")
            h, w = image.shape[:2]
            max_side = max(32, min(512, int(max_side or 160)))
            scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
            if scale < 1.0:
                image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                raise HTTPException(status_code=500, detail="scrape thumbnail encode failed")
            return Response(content=buf.tobytes(), media_type="image/jpeg")

        @self.app.get("/scrape/jobs/{slug}/image/{kind}/{name:path}")
        async def scrape_image(slug: str, kind: str, name: str, max_side: int = Query(1280)) -> Response:
            try:
                match = self._scrape_resolve_media_path(slug, kind, name)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            image = cv2.imread(str(match))
            if image is None:
                raise HTTPException(status_code=400, detail="unable to decode scrape image")
            h, w = image.shape[:2]
            max_side = max(128, min(2400, int(max_side or 1280)))
            scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
            if scale < 1.0:
                image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise HTTPException(status_code=500, detail="scrape image encode failed")
            return Response(content=buf.tobytes(), media_type="image/jpeg")

        @self.app.get("/scrape/jobs/{slug}/labels/{name:path}")
        async def scrape_labels(slug: str, name: str) -> dict[str, Any]:
            try:
                return self._scrape_label_payload(slug, name)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"scrape label load failed: {exc}") from exc

        @self.app.put("/scrape/jobs/{slug}/labels/{name:path}")
        async def scrape_save_labels(slug: str, name: str, req: ScrapeLabelWriteRequest) -> dict[str, Any]:
            try:
                return self._scrape_write_boxes(slug, name, list(req.boxes or []))
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"scrape label save failed: {exc}") from exc

        @self.app.put("/scrape/jobs/{slug}/classes")
        async def scrape_save_classes(slug: str, req: ScrapeClassesWriteRequest) -> dict[str, Any]:
            try:
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                job = ScrapJobStore.load(slug)
                if job is None:
                    raise FileNotFoundError(f"scrape job not found: {slug}")
                classes: list[str] = []
                seen: set[str] = set()
                for item in list(req.classes or []):
                    name = str(item or "").strip()
                    if not name:
                        continue
                    key = name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    classes.append(name)
                updated = ScrapJobStore.update(slug, classes=classes, state="labeling", message="updated scrape classes")
                ScrapJobStore.append_log(slug, f"Saved {len(classes)} class(es)")
                final = updated or ScrapJobStore.load(slug)
                return self._scrape_serialize_job(final, include_log=True) if final is not None else {"slug": slug, "classes": classes}
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"scrape class save failed: {exc}") from exc

        @self.app.delete("/scrape/jobs/{slug}/staged/{name:path}")
        async def scrape_delete_staged_image(slug: str, name: str) -> dict[str, Any]:
            try:
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                match = self._scrape_resolve_media_path(slug, "staged", name)
                match.unlink(missing_ok=False)
                job = ScrapJobStore.load(slug)
                if job is None:
                    raise FileNotFoundError(f"scrape job not found: {slug}")
                labels = dict(job.labels or {})
                labels.pop(match.name, None)
                staged_dir = self._scrape_dataset_root(slug) / "staged"
                staged_count = sum(
                    1 for path in staged_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
                ) if staged_dir.is_dir() else 0
                updated = ScrapJobStore.update(
                    slug,
                    labels=labels,
                    staged_count=staged_count,
                    message=f"deleted staged image {match.name}",
                )
                ScrapJobStore.append_log(slug, f"Deleted staged image {match.name}")
                final = updated or ScrapJobStore.load(slug)
                return self._scrape_serialize_job(final, include_log=True) if final is not None else {"slug": slug}
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"staged delete failed: {exc}") from exc

        @self.app.post("/scrape/jobs/{slug}/emit")
        async def scrape_emit(slug: str, req: ScrapeEmitRequest) -> dict[str, Any]:
            try:
                from mlops.scrap.emit import LabeledItem, emit_yolo_dataset  # noqa: PLC0415
                from mlops.scrap.jobs import JobStore as ScrapJobStore  # noqa: PLC0415

                root = self._scrape_dataset_root(slug)
                job = ScrapJobStore.load(slug)
                if job is None:
                    raise FileNotFoundError(f"scrape job not found: {slug}")
                staged_dir = root / "staged"
                staged_images = [
                    path for path in staged_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_EXTS
                ] if staged_dir.is_dir() else []
                labeled_images = [path for path in staged_images if path.name in (job.labels or {}) and (job.labels or {}).get(path.name)]
                if len(labeled_images) < 2:
                    raise ValueError("need at least 2 labeled staged images for emit")
                items = [
                    LabeledItem(
                        image_path=path,
                        boxes=tuple(
                            (
                                int(float(box[0])),
                                float(box[1]),
                                float(box[2]),
                                float(box[3]),
                                float(box[4]),
                            )
                            for box in list((job.labels or {}).get(path.name, []) or [])
                        ),
                    )
                    for path in labeled_images
                ]
                ds_root = emit_yolo_dataset(
                    slug=slug,
                    classes=list(job.classes or []),
                    items=items,
                    val_frac=max(0.05, min(0.5, float(req.val_frac or 0.2))),
                )
                mlops_registry.create_scenario_profile(
                    name=slug,
                    display_name=str(job.topic or slug).title(),
                    description=f"Scrap-built scenario for topic '{job.topic}'.",
                    base_model=str(req.base_model or "").strip() or "assets/models/yolov10n.pt",
                    dataset=slug,
                    classes=list(job.classes or []),
                    hyperparams={"epochs": max(1, min(300, int(req.epochs or 20))), "imgsz": 640},
                    guard_profile="balanced",
                    backbone_type="yolo_detection",
                )
                updated = ScrapJobStore.update(slug, state="emitted", message="dataset + scenario emitted")
                ScrapJobStore.append_log(slug, f"Emitted dataset and scenario for {slug}")
                self._emit_scenario_updated(slug)
                final = updated or ScrapJobStore.load(slug)
                return {
                    "job": self._scrape_serialize_job(final, include_log=True) if final is not None else {"slug": slug},
                    "dataset_root": str(ds_root),
                    "scenario": slug,
                }
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"scrape emit failed: {exc}") from exc

        @self.app.get("/ecosystem/summary")
        def ecosystem_summary(request: Request) -> Response:
            # Sync (threadpool): this is registry + SQLite filesystem work and
            # is polled by the Ecosystem panel while training progress events
            # are active. Keeping it off the uvicorn event loop prevents the
            # graph view from freezing behind long-running training updates.
            return self._json_cache_response(self._ecosystem_summary(), request, max_age=2)

        @self.app.get("/ecosystem/graph_view", response_class=HTMLResponse)
        def ecosystem_graph_view(
            request: Request,
            entity_types: str = Query(default=""),
            scenario: str = Query(default=""),
            depth: int = Query(default=2, ge=1, le=3),
            since_ts: Optional[float] = Query(default=None),
            layer: str = Query(default="core"),
        ) -> HTMLResponse:
            try:
                graph = self._get_ontology_graph_cached(
                    layer=layer,
                    entity_types=entity_types,
                    scenario=scenario,
                    depth=depth,
                    since_ts=since_ts,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            base_url = str(request.base_url).rstrip("/")
            return HTMLResponse(content=_browser_ecosystem_html(base_url=base_url, graph=graph))

        @self.app.post("/ingest/assets")
        async def ingest_asset(
            req: Optional[IngestAssetRequest] = None,
            file: Optional[UploadFile] = File(None),
            name: str = Form(""),
            source_type: str = Form(""),
            storage_mode: str = Form(""),
            sector_id: str = Form(""),
            sector_path: str = Form(""),
            source_uri: str = Form(""),
            collection_id: str = Form(""),
            tags: str = Form(""),
            keywords: str = Form(""),
            lineage_json: str = Form(""),
            metadata_json: str = Form(""),
        ) -> dict[str, Any]:
            payload = self._resolve_ingest_request(
                req=req,
                name=name,
                source_type=source_type,
                storage_mode=storage_mode,
                sector_id=sector_id,
                sector_path=sector_path,
                source_uri=source_uri,
                collection_id=collection_id,
                tags=tags,
                keywords=keywords,
                lineage_json=lineage_json,
                metadata_json=metadata_json,
            )
            try:
                out = await self._ingest_asset(payload, file=file)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"sector not found: {exc}") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            self._record_catalog_event("asset_ingested", out)
            return out

        @self.app.get("/ingest/assets/{asset_id}")
        async def get_ingested_asset(asset_id: str) -> dict[str, Any]:
            try:
                payload = self._get_asset_payload(asset_id, refresh_reference=True)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="asset not found") from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return payload

        @self.app.post("/sectors")
        async def create_sector(req: CreateSectorRequest) -> dict[str, Any]:
            try:
                sector = self.catalog.create_sector(
                    name=req.name,
                    parent_id=req.parent_id or ROOT_SECTOR_ID,
                    parent_path=req.parent_path,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"parent sector not found: {exc}") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            payload = sector.to_dict()
            self._record_catalog_event("sector_created", payload)
            return payload

        @self.app.get("/sectors")
        async def list_sectors() -> dict[str, Any]:
            return self.sectors_payload()

        @self.app.patch("/sectors/{sector_id}")
        async def rename_sector(sector_id: str, req: RenameSectorRequest) -> dict[str, Any]:
            try:
                sector = self.catalog.rename_sector(sector_id, req.name)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"sector not found: {exc}") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            payload = sector.to_dict()
            self._record_catalog_event("sector_renamed", payload)
            return payload

        @self.app.post("/sectors/{sector_id}/move")
        async def move_sector(sector_id: str, req: MoveSectorRequest) -> dict[str, Any]:
            try:
                sector = self.catalog.move_sector(sector_id, req.parent_id or ROOT_SECTOR_ID)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"sector not found: {exc}") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            payload = sector.to_dict()
            self._record_catalog_event("sector_moved", payload)
            return payload

        @self.app.post("/assets/{asset_id}/assign_sector")
        async def assign_asset_sector(asset_id: str, req: AssignAssetSectorRequest) -> dict[str, Any]:
            if not str(req.sector_id or "").strip() and not str(req.sector_path or "").strip():
                raise HTTPException(status_code=400, detail="sector_id or sector_path is required")
            try:
                asset = self.catalog.assign_asset_sector(
                    asset_id,
                    sector_id=str(req.sector_id or "").strip(),
                    sector_path=str(req.sector_path or "").strip(),
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"asset/sector not found: {exc}") from exc
            payload = asset.to_dict()
            self._record_catalog_event("asset_sector_assigned", payload)
            return payload

        @self.app.get("/catalog/search")
        async def catalog_search(
            q: str = Query("", alias="q"),
            source_type: str = Query("", alias="source_type"),
            status: str = Query("", alias="status"),
            storage_mode: str = Query("", alias="storage_mode"),
            sector_path: str = Query("", alias="sector_path"),
            include_descendants: int = Query(1, alias="include_descendants"),
            limit: int = Query(100, alias="limit"),
        ) -> dict[str, Any]:
            try:
                items = self.catalog.search_assets(
                    query=str(q or ""),
                    source_type=str(source_type or ""),
                    status=str(status or ""),
                    storage_mode=str(storage_mode or ""),
                    sector_path=str(sector_path or ""),
                    include_descendants=bool(int(include_descendants or 0)),
                    limit=int(limit or 100),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            out = [self._asset_with_reference_health(item.to_dict()) for item in items]
            return {"count": len(out), "items": out}

        @self.app.get("/catalog/sectors/{sector_path:path}/summary")
        async def catalog_sector_summary(sector_path: str) -> dict[str, Any]:
            normalized_path = "/" + str(sector_path or "").strip().lstrip("/")
            if normalized_path == "//":
                normalized_path = ROOT_SECTOR_PATH
            try:
                payload = self.catalog.sector_summary(normalized_path)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=f"sector not found: {exc}") from exc
            return payload

        @self.app.get("/catalog/sectors/summary")
        async def catalog_root_sector_summary() -> dict[str, Any]:
            return self.catalog.sector_summary(ROOT_SECTOR_PATH)

        @self.app.post("/archives/import")
        async def import_archive(req: ArchiveImportRequest) -> dict[str, Any]:
            source_paths = [Path(str(item or "").strip()).expanduser() for item in req.source_paths if str(item or "").strip()]
            if not source_paths:
                raise HTTPException(status_code=400, detail="source_paths is required")
            try:
                _req_name = req.name
                _req_desc = req.description
                _req_meta = dict(req.metadata or {})
                _req_corpus = str(req.corpus_id or "").strip()
                _corr_id = str(req.correlation_id or "").strip() or uuid.uuid4().hex
                _src_label = " ".join(str(p) for p in source_paths[:2])
                if len(source_paths) > 2:
                    _src_label += f" (+{len(source_paths) - 2} more)"
                _lines: list[str] = [
                    f"> archive import {_src_label}",
                    f"[STARTING] storage root: {self.archives.storage_root}",
                ]
                self._import_progress[_corr_id] = {"phase": "starting", "current": 0, "total": 0, "correlation_id": _corr_id, "lines": _lines}

                def _fmt_line(state: dict[str, Any]) -> Optional[str]:
                    phase = state.get("phase", "")
                    cur = int(state.get("current") or 0)
                    tot = int(state.get("total") or 0)
                    fname = str(state.get("current_filename") or "")
                    fsize = state.get("file_size_bytes")
                    hpfx = str(state.get("hash_prefix") or "")
                    workers = int(state.get("workers") or 1)
                    size_str = f"  {_human_bytes(int(fsize))}" if fsize else ""
                    count_str = f"{cur:>5}/{tot}" if tot > 0 else ""
                    hash_str = f"  {hpfx[:8]}…" if hpfx else ""
                    if phase == "scanning":
                        return "[SCAN]   Scanning source paths…"
                    if phase == "copying" and cur == 0:
                        w_str = f" ({workers} worker{'s' if workers != 1 else ''})" if workers > 1 else ""
                        return f"[SCAN]   Found {tot} file(s) — starting copy+hash{w_str}…"
                    if phase == "copying":
                        return f"[COPY]   {count_str}  {fname}{size_str}{hash_str}"
                    if phase == "indexing":
                        return f"[INDEX]  Writing {cur} file records to database…"
                    if phase == "done":
                        return f"[DONE]   {tot} file(s) processed"
                    return None

                def _progress_cb(state: dict[str, Any]) -> None:
                    line = _fmt_line(state)
                    if line:
                        _lines.append(line)
                    self._import_progress[_corr_id] = {**state, "correlation_id": _corr_id, "lines": _lines}

                _import_result: list[dict[str, Any]] = []
                _import_error: list[BaseException] = []
                _import_done = threading.Event()

                def _run_import() -> None:
                    try:
                        r = self.archives.import_paths(
                            source_paths=source_paths,
                            name=_req_name,
                            description=_req_desc,
                            metadata=_req_meta,
                            corpus_id=_req_corpus,
                            progress_cb=_progress_cb,
                        )
                        _import_result.append(r)
                    except BaseException as _exc:
                        _import_error.append(_exc)
                    finally:
                        _import_done.set()

                threading.Thread(target=_run_import, daemon=True, name=f"ArchiveImport-{_corr_id[:8]}").start()
                while not _import_done.is_set():
                    await asyncio.sleep(0.25)
                if _import_error:
                    raise _import_error[0]
                payload = _import_result[0]
                corpus = payload.get("corpus") if isinstance(payload.get("corpus"), dict) else {}
                corpus_id = str(corpus.get("corpus_id") or "")
                dataset_version_id = str(payload.get("dataset_version_id") or "")
                sector_id = str(req.sector_id or "").strip() or ROOT_SECTOR_ID

                collection_id = str(corpus.get("collection_id") or "")
                if not collection_id:
                    collection = self.catalog.create_collection(
                        name=str(corpus.get("name") or req.name or "Archive Corpus"),
                        source_type="archival_corpus",
                        sector_id=sector_id,
                        description=str(corpus.get("description") or req.description or ""),
                        metadata={
                            "corpus_id": corpus_id,
                            "slug": str(corpus.get("slug") or ""),
                            "managed_root": str(corpus.get("managed_root") or ""),
                        },
                    )
                    collection_id = str(collection.get("collection_id") or "")
                    self.archives.set_corpus_collection(corpus_id, collection_id)
                    corpus["collection_id"] = collection_id
                    payload["corpus"] = corpus

                source_uri = ""
                try:
                    raw_source_paths = json.loads(str(payload.get("source_path") or "[]"))
                    if isinstance(raw_source_paths, list) and raw_source_paths:
                        source_uri = str(raw_source_paths[0] or "")
                except Exception:
                    source_uri = str(payload.get("source_path") or "")
                size_bytes = int(payload.get("total_size_bytes") or 0)
                asset = self.catalog.create_asset(
                    name=f"{str(corpus.get('name') or req.name or 'Archive Corpus')} {str(payload.get('label') or '')}".strip(),
                    source_type="archival_corpus",
                    storage_mode="managed_copy",
                    sector_id=sector_id,
                    source_uri=source_uri,
                    managed_path=str(payload.get("raw_root") or ""),
                    status="ingested",
                    schema_status="structured",
                    extraction_status="pending",
                    availability_status="available",
                    size_bytes=size_bytes,
                    tags=["archive", "archival_ingestion"],
                    keywords=[
                        str(corpus.get("slug") or ""),
                        str(payload.get("label") or ""),
                        str(corpus.get("name") or ""),
                    ],
                    lineage={
                        "corpus_id": corpus_id,
                        "dataset_version_id": dataset_version_id,
                        "source_type": "archival_corpus",
                    },
                    metadata={
                        "corpus_id": corpus_id,
                        "dataset_version_id": dataset_version_id,
                        "version_index": payload.get("version_index"),
                        "file_count": int(payload.get("file_count") or 0),
                        "noise_file_count": int(payload.get("noise_file_count") or 0),
                        "processable_file_count": int(payload.get("processable_file_count") or 0),
                    },
                    collection_id=collection_id,
                )
                self.archives.set_dataset_version_catalog_asset(dataset_version_id, asset.asset_id)
                payload["catalog_asset_id"] = asset.asset_id
                payload["catalog_collection_id"] = collection_id

                _noise_n = int(payload.get("noise_file_count") or 0)
                _proc_n = int(payload.get("processable_file_count") or 0)
                _total_n = int(payload.get("file_count") or 0)
                _retained_n = max(0, _total_n - _proc_n - _noise_n)
                _lines.extend([
                    f"[INDEX]  catalog collection {collection_id}",
                    f"[DONE]   corpus_id    {corpus_id}",
                    f"[DONE]   dataset_id   {dataset_version_id}",
                    f"[DONE]   version      {str(payload.get('label') or '')}",
                    f"[DONE]   processable  {_proc_n} file(s)",
                    f"[DONE]   noise        {_noise_n} file(s)",
                    f"[DONE]   retained     {_retained_n} file(s)",
                    f"[DONE]   total        {_total_n} file(s) indexed",
                ])
                self._import_progress[_corr_id] = {**self._import_progress.get(_corr_id, {}), "phase": "complete", "lines": _lines}

                scenario_name = str(req.scenario or "").strip()
                if scenario_name:
                    cfg = mlops_registry.get_scenario_config(scenario_name)
                    if str(cfg.backbone_type or "") != "archival_ingestion":
                        raise ValueError("scenario must use backbone_type=archival_ingestion")
                    try:
                        archive_storage_root = self.archives.storage_root.resolve().relative_to(ROOT_DIR.resolve()).as_posix()
                    except Exception:
                        archive_storage_root = str(self.archives.storage_root.resolve())
                    mlops_registry.patch_scenario_backbone_config(
                        scenario_name,
                        {
                            "corpus_id": corpus_id,
                            "dataset_version_id": dataset_version_id,
                            "latest_snapshot_id": "",
                            "archive_storage_root": archive_storage_root,
                        },
                    )
                    self._emit_scenario_updated(scenario_name)
                    payload["scenario"] = scenario_name
                return payload
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                import traceback as _tb
                _tb.print_exc()
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @self.app.get("/archives/import_progress/{correlation_id}")
        async def get_import_progress(correlation_id: str) -> dict[str, Any]:
            entry = self._import_progress.get(str(correlation_id or "").strip())
            if entry is None:
                return {"phase": "unknown", "correlation_id": correlation_id, "message": "No active import for this id"}
            return dict(entry)

        @self.app.get("/archives")
        async def list_archives() -> dict[str, Any]:
            items = self.archives.list_corpora()
            return {"count": len(items), "corpora": items}

        @self.app.get("/archives/jobs/{job_id}")
        async def get_archive_job(job_id: str) -> dict[str, Any]:
            try:
                payload = self.store.get_job(job_id).to_dict()
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job not found") from exc
            result = self.store.get_result(job_id)
            if isinstance(result, dict):
                payload["result"] = result
            return payload

        @self.app.get("/archives/snapshots/{snapshot_id}/timeline")
        async def archive_timeline(
            snapshot_id: str,
            q: str = Query("", alias="q"),
            unresolved_only: int = Query(0, alias="unresolved_only"),
        ) -> dict[str, Any]:
            try:
                payload = self._archive_engine().build_timeline_payload(
                    self.archives,
                    snapshot_id,
                    query=str(q or ""),
                    unresolved_only=bool(unresolved_only),
                )
                payload["snapshot"] = self.archives.get_snapshot(snapshot_id)
                return payload
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.get("/archives/{corpus_id}/versions/{dataset_version_id}/phase0_review")
        async def archive_phase0_review(
            corpus_id: str,
            dataset_version_id: str,
            q: str = Query("", alias="q"),
        ) -> dict[str, Any]:
            try:
                payload = self._archive_engine().build_phase0_review_payload(
                    self.archives,
                    corpus_id,
                    dataset_version_id,
                    query=str(q or ""),
                )
                payload["version"] = self.archives.get_dataset_version(dataset_version_id)
                return payload
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.get("/archives/snapshots/{snapshot_id}/phase2_review")
        async def archive_phase2_review(
            snapshot_id: str,
            q: str = Query("", alias="q"),
        ) -> dict[str, Any]:
            try:
                payload = self._archive_engine().build_phase2_review_payload(
                    self.archives,
                    snapshot_id,
                    query=str(q or ""),
                )
                payload["snapshot"] = self.archives.get_snapshot(snapshot_id)
                return payload
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.get("/archives/snapshots/{snapshot_id}/phase3_review")
        async def archive_phase3_review(
            snapshot_id: str,
            q: str = Query("", alias="q"),
        ) -> dict[str, Any]:
            try:
                payload = self._archive_engine().build_phase3_review_payload(
                    self.archives,
                    snapshot_id,
                    query=str(q or ""),
                )
                payload["snapshot"] = self.archives.get_snapshot(snapshot_id)
                return payload
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.get("/archives/snapshots/{snapshot_id}/phase4_review")
        async def archive_phase4_review(
            snapshot_id: str,
            q: str = Query("", alias="q"),
        ) -> dict[str, Any]:
            try:
                payload = self._archive_engine().build_phase4_review_payload(
                    self.archives,
                    snapshot_id,
                    query=str(q or ""),
                )
                payload["snapshot"] = self.archives.get_snapshot(snapshot_id)
                return payload
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.get("/archives/snapshots/{snapshot_id}/phase5_review")
        async def archive_phase5_review(
            snapshot_id: str,
            q: str = Query("", alias="q"),
        ) -> dict[str, Any]:
            try:
                payload = self._archive_engine().build_phase5_review_payload(
                    self.archives,
                    snapshot_id,
                    query=str(q or ""),
                )
                payload["snapshot"] = self.archives.get_snapshot(snapshot_id)
                return payload
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.post("/archives/proposals/{proposal_id}/decision")
        async def archive_proposal_decision(proposal_id: str, req: ArchiveProposalDecisionRequest) -> dict[str, Any]:
            try:
                return self._archive_engine().apply_proposal_decision(
                    self.archives,
                    str(req.snapshot_id or "").strip(),
                    proposal_id,
                    str(req.decision or "").strip(),
                    decided_by=str(req.decided_by or ""),
                    reason=str(req.reason or ""),
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.get("/archives/snapshots/{snapshot_id}/objects")
        async def archive_objects(
            snapshot_id: str,
            q: str = Query("", alias="q"),
            unresolved_only: int = Query(0, alias="unresolved_only"),
        ) -> dict[str, Any]:
            try:
                objects = self.archives.list_objects(snapshot_id)
                timeline = self._archive_engine().build_timeline_payload(
                    self.archives,
                    snapshot_id,
                    query=str(q or ""),
                    unresolved_only=bool(unresolved_only),
                )
                allowed_ids = {str(item.get("object_id") or "") for item in (timeline.get("items") or []) + (timeline.get("holding_pen") or [])}
                filtered = [
                    item for item in objects
                    if str(item.get("object_id") or "") in allowed_ids
                ]
                return {
                    "snapshot": self.archives.get_snapshot(snapshot_id),
                    "count": len(filtered),
                    "objects": filtered,
                }
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        @self.app.get("/archives/snapshots/{snapshot_id}/objects/{object_id}")
        async def archive_object_detail(snapshot_id: str, object_id: str) -> dict[str, Any]:
            try:
                detail = self.archives.build_object_detail(snapshot_id, object_id)
                detail["snapshot"] = self.archives.get_snapshot(snapshot_id)
                return detail
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        @self.app.post("/archives/objects/{object_id}/assembly_override")
        async def archive_assembly_override(object_id: str, req: ArchiveAssemblyOverrideRequest) -> dict[str, Any]:
            try:
                return self.archives.add_assembly_override(
                    corpus_id=req.corpus_id,
                    dataset_version_id=req.dataset_version_id,
                    scope_key=str(req.scope_key or object_id).strip() or object_id,
                    action=req.action,
                    payload=dict(req.payload or {}),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.post("/archives/entities/{entity_id}/merge_override")
        async def archive_entity_override(entity_id: str, req: ArchiveEntityMergeOverrideRequest) -> dict[str, Any]:
            try:
                self.archives.get_entity(req.snapshot_id, entity_id)
                action = "reject_entity_merge" if req.reject else "merge_entities"
                payload: dict[str, Any] = {"other_entity_ids": [str(item) for item in req.other_entity_ids if str(item).strip()]}
                if not req.reject and str(req.canonical_name or "").strip():
                    payload["canonical_name"] = str(req.canonical_name).strip()
                return self.archives.add_resolution_override(
                    corpus_id=req.corpus_id,
                    snapshot_id=req.snapshot_id,
                    target_type="entity",
                    target_id=entity_id,
                    action=action,
                    payload=payload,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        @self.app.post("/archives/anchors/{anchor_id}/resolve_override")
        async def archive_anchor_override(anchor_id: str, req: ArchiveAnchorResolveOverrideRequest) -> dict[str, Any]:
            try:
                anchors = self.archives.list_anchors(req.snapshot_id)
                if anchor_id not in {str(item.get("anchor_id") or "") for item in anchors}:
                    raise KeyError(anchor_id)
                return self.archives.add_resolution_override(
                    corpus_id=req.corpus_id,
                    snapshot_id=req.snapshot_id,
                    target_type="anchor",
                    target_id=anchor_id,
                    action="pin_date",
                    payload={
                        "earliest": str(req.earliest or "").strip(),
                        "latest": str(req.latest or "").strip(),
                        "note": str(req.note or "").strip(),
                    },
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        @self.app.post("/archives/{corpus_id}/jobs")
        async def kick_archive_job(corpus_id: str, req: ArchiveJobKickRequest) -> dict[str, Any]:
            valid_job_types = set(self._archive_engine().PHASE_SEQUENCE) | {"archive_pipeline", "archive_reconcile"}
            phase = str(req.phase or "archive_pipeline").strip()
            if phase not in valid_job_types:
                raise HTTPException(status_code=400, detail=f"unsupported archive phase: {phase}")
            try:
                corpus = self.archives.get_corpus(corpus_id)
                version = self.archives.get_dataset_version(req.dataset_version_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if str(version.get("corpus_id") or "") != str(corpus.get("corpus_id") or ""):
                raise HTTPException(status_code=400, detail="dataset version does not belong to corpus")

            scenario_name = str(req.scenario or "").strip() or f"archive:{str(corpus.get('slug') or corpus_id)}"
            provider_config: dict[str, Any] = {}
            if str(req.scenario or "").strip():
                try:
                    cfg = mlops_registry.get_scenario_config(scenario_name)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                if str(cfg.backbone_type or "") != "archival_ingestion":
                    raise HTTPException(status_code=400, detail="scenario must use backbone_type=archival_ingestion")
                if isinstance(getattr(cfg, "backbone_config", None), dict):
                    provider_config.update(
                        dict((getattr(cfg, "backbone_config", {}) or {}).get("providers") or {})
                    )
                self._emit_scenario_updated(scenario_name)
            if isinstance(req.provider_config, dict) and req.provider_config:
                provider_config.update(dict(req.provider_config))

            job_id = f"job-{uuid.uuid4().hex[:12]}"
            payload = {
                "corpus_id": corpus_id,
                "dataset_version_id": req.dataset_version_id,
                "phase": phase,
                "parent_snapshot_id": str(req.parent_snapshot_id or "").strip(),
                "write_run_artifacts": bool(req.write_run_artifacts),
                "provider_config": provider_config,
            }
            job = self.store.create_job(
                job_id=job_id,
                scenario=scenario_name,
                job_type=phase,
                source="archive",
                image_path="",
                payload=payload,
            )
            self._emit_job_status(job)
            return {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "state": job.state,
                "scenario": job.scenario,
                "corpus_id": corpus_id,
                "dataset_version_id": req.dataset_version_id,
                "phase": phase,
            }

        @self.app.get("/archives/{corpus_id}/versions/{dataset_version_id}")
        async def get_archive_dataset_version(corpus_id: str, dataset_version_id: str) -> dict[str, Any]:
            try:
                payload = self.archives.get_dataset_version(dataset_version_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            if str(payload.get("corpus_id") or "") != str(corpus_id or ""):
                raise HTTPException(status_code=404, detail="dataset version not found for corpus")
            return payload

        @self.app.get("/archives/{corpus_id}")
        async def get_archive_corpus(corpus_id: str) -> dict[str, Any]:
            try:
                return self.archives.get_corpus(corpus_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc

        @self.app.get("/scenarios")
        def scenarios(request: Request) -> Response:
            # Declared sync (not async) on purpose: the body is entirely
            # blocking work — registry scan, per-scenario filesystem status,
            # and a sqlite list_jobs() under a lock. FastAPI runs sync route
            # handlers in its anyio threadpool, so this no longer blocks (or
            # gets starved on) the single uvicorn event loop. As an `async def`
            # it would serialize behind the heartbeat task, WS broadcasts, and
            # every other blocking handler, which is what made it time out on
            # the 6s client deadline during startup / active training.
            return self._json_cache_response(self.scenarios_payload(), request, max_age=2)

        @self.app.post("/scenarios")
        async def create_scenario(req: ScenarioCreateRequest) -> dict[str, Any]:
            try:
                status = mlops_registry.create_scenario_profile(
                    name=req.name,
                    display_name=req.display_name,
                    description=req.description,
                    base_model=req.base_model,
                    dataset=req.dataset,
                    classes=req.classes,
                    postproc=req.postproc,
                    hyperparams={"epochs": int(req.epochs or 20), "imgsz": int(req.imgsz or 640)},
                    guard_profile=req.guard_profile,
                    backbone_type=req.backbone_type,
                    backbone_config=dict(req.backbone_config) if req.backbone_config else {},
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(str(status.get("name") or req.name))
            return status

        @self.app.get("/scenarios/{scenario}/status")
        async def scenario_status(scenario: str) -> dict[str, Any]:
            return self.scenario_status_payload(scenario)

        @self.app.get("/scenarios/{scenario}/history")
        async def scenario_history(scenario: str) -> dict[str, Any]:
            try:
                return self.scenario_history_payload(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.get("/scenarios/{scenario}/pipeline")
        async def scenario_pipeline(scenario: str) -> dict[str, Any]:
            try:
                return self.scenario_pipeline_payload(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.patch("/scenarios/{scenario}/pipeline")
        async def patch_scenario_pipeline(scenario: str, req: PipelinePatchRequest) -> dict[str, Any]:
            try:
                status = mlops_registry.patch_scenario_ci_cd_policy(
                    scenario,
                    dict(req.updates or {}),
                    reset=bool(req.reset),
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return {
                "scenario": scenario,
                "ci_cd": status.get("ci_cd") or mlops_registry.get_scenario_ci_cd_policy(scenario),
                "status": status,
            }

        @self.app.post("/scenarios/{scenario}/runs/{version}/gate")
        async def run_scenario_gate(scenario: str, version: str) -> dict[str, Any]:
            try:
                report = evaluate_run_gate(
                    scenario,
                    version,
                    policy=mlops_registry.get_scenario_ci_cd_policy(scenario),
                    update_registry=True,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return report

        @self.app.get("/scenarios/{scenario}/runs/{version}/gate")
        async def get_scenario_gate(scenario: str, version: str) -> dict[str, Any]:
            try:
                return load_gate_report(scenario, version)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.post("/scenarios/{scenario}/runs/{version}/promote")
        async def promote_scenario_run(scenario: str, version: str, req: PromoteRunRequest) -> dict[str, Any]:
            try:
                result = promote_run(
                    scenario,
                    version,
                    target_alias=str(req.target_alias or "prod"),
                    actor=str(req.actor or "cvops"),
                    reason=str(req.reason or ""),
                    override=bool(req.override),
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return result

        @self.app.get("/scenarios/{scenario}/aliases")
        async def scenario_aliases(scenario: str) -> dict[str, Any]:
            """Current champion-challenger pointers + each alias's revert trail."""
            aliases: dict[str, Any] = {}
            for name in ("candidate", "staging", "prod"):
                version = resolve_alias(scenario, name)
                aliases[name] = {
                    "version_id": str((version or {}).get("version_id") or ""),
                    "history": alias_history(scenario, name),
                }
            return {"scenario": scenario, "aliases": aliases}

        @self.app.post("/scenarios/{scenario}/aliases/{alias}/revert")
        async def revert_scenario_alias(scenario: str, alias: str, req: AliasRevertRequest) -> dict[str, Any]:
            """Roll an alias (e.g. prod) back to the version it previously pointed at."""
            try:
                result = revert_alias(
                    scenario,
                    alias,
                    actor=str(req.actor or "cvops"),
                    reason=str(req.reason or ""),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return result

        @self.app.post("/carve/index")
        async def carve_index(req: CarveIndexRequest) -> dict[str, Any]:
            """Embed a source folder with CLIP in the background; reusable across
            preview queries. Returns a correlation id for progress polling."""
            folder = Path(str(req.folder or "").strip()).expanduser()
            if not folder.is_dir():
                raise HTTPException(status_code=400, detail=f"folder not found: {folder}")
            cid = uuid.uuid4().hex
            self._carve_progress[cid] = {"phase": "starting", "current": 0, "total": 0,
                                         "folder": str(folder), "ready": False}

            def _run() -> None:
                try:
                    from .clip_embed import ClipEmbedder
                    from . import semantic_carve as sc

                    with self._carve_lock:
                        if self._carve_embedder is None:
                            self._carve_progress[cid].update(phase="loading_model")
                            self._carve_embedder = ClipEmbedder(device="auto")
                        embedder = self._carve_embedder

                    def _cb(done: int, total: int) -> None:
                        self._carve_progress[cid].update(phase="embedding", current=done, total=total)

                    index = sc.build_index(folder, embedder, max_images=int(req.max_images), progress_cb=_cb)
                    self._carve_index = index
                    self._carve_progress[cid].update(
                        phase="done", ready=True, current=len(index), total=len(index),
                        indexed=len(index),
                    )
                except Exception as exc:  # pragma: no cover - runtime/env dependent
                    self._carve_progress[cid].update(phase="error", error=str(exc), ready=False)

            threading.Thread(target=_run, daemon=True, name=f"carve-index-{cid[:6]}").start()
            return {"correlation_id": cid}

        @self.app.get("/carve/index_progress/{correlation_id}")
        async def carve_index_progress(correlation_id: str) -> dict[str, Any]:
            entry = self._carve_progress.get(str(correlation_id or "").strip())
            if entry is None:
                raise HTTPException(status_code=404, detail="unknown carve correlation id")
            return entry

        @self.app.post("/carve/preview")
        async def carve_preview(req: CarvePreviewRequest) -> dict[str, Any]:
            index = self._carve_index
            if index is None or len(index) == 0:
                raise HTTPException(status_code=409, detail="no carve index; POST /carve/index first")
            from . import semantic_carve as sc
            import numpy as _np

            scores = sc.query_scores(index, req.query, self._carve_embedder)
            pos, neg = sc.select(scores, threshold=float(req.threshold))
            order = _np.argsort(-scores)
            sample = [
                {"path": str(index.paths[int(i)]), "name": index.paths[int(i)].name,
                 "score": round(float(scores[int(i)]), 3)}
                for i in order[: max(0, int(req.sample))]
            ]
            return {
                "query": req.query, "threshold": req.threshold, "total": int(len(index)),
                "positive_count": len(pos), "negative_count": len(neg),
                "score_max": round(float(scores.max()), 3) if len(scores) else 0.0,
                "score_mean": round(float(scores.mean()), 3) if len(scores) else 0.0,
                "sample": sample,
            }

        @self.app.post("/carve/create")
        async def carve_create(req: CarveCreateRequest) -> dict[str, Any]:
            index = self._carve_index
            if index is None or len(index) == 0:
                raise HTTPException(status_code=409, detail="no carve index; POST /carve/index first")
            from . import semantic_carve as sc

            scores = sc.query_scores(index, req.query, self._carve_embedder)
            pos, neg = sc.select(
                scores, threshold=float(req.threshold),
                max_positive=int(req.max_positive), max_negative=int(req.max_negative),
            )
            if not pos:
                raise HTTPException(status_code=400, detail="no images matched; lower the threshold")
            try:
                result = sc.materialize_imagefolder(
                    registry_dir=DATASET_REGISTRY_DIR,
                    slug=req.slug, class_name=req.class_name,
                    positive_paths=[index.paths[i] for i in pos],
                    negative_paths=[index.paths[i] for i in neg],
                    query=req.query, threshold=float(req.threshold),
                )
            except FileExistsError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return result

        @self.app.get("/scenarios/{scenario}/metrics")
        async def scenario_metrics(scenario: str) -> dict[str, Any]:
            metrics = mlops_registry.latest_run_metrics(scenario)
            if metrics is None:
                raise HTTPException(status_code=404, detail="no metrics")
            return metrics

        @self.app.get("/scenarios/{scenario}/cards")
        async def scenario_cards(scenario: str) -> dict[str, Any]:
            try:
                return self.scenario_cards_payload(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.get("/cvops/state/export")
        async def cvops_state_export() -> StreamingResponse:
            mem = SpooledTemporaryFile(max_size=96 * 1024 * 1024)

            def _write() -> None:
                with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                    manifest = {
                        "version": 1,
                        "format": "cvops_state_bundle",
                        "exported_at": time.time(),
                        "entries": ["manifest.json", "jobs.db", "model_registry.json", "dataset_registry/"],
                    }
                    zf.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=True))
                    if CVOPS_DB_PATH.is_file():
                        zf.write(CVOPS_DB_PATH, arcname="jobs.db")
                    if MODEL_REGISTRY_PATH.is_file():
                        zf.write(MODEL_REGISTRY_PATH, arcname="model_registry.json")
                    if DATASET_REGISTRY_DIR.exists():
                        for path in DATASET_REGISTRY_DIR.rglob("*"):
                            if path.is_file():
                                arc = Path("dataset_registry") / path.relative_to(DATASET_REGISTRY_DIR)
                                zf.write(path, arcname=str(arc).replace("\\", "/"))

            await asyncio.to_thread(_write)
            mem.seek(0)

            def iterfile():
                try:
                    while True:
                        chunk = mem.read(1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
                finally:
                    try:
                        mem.close()
                    except Exception:
                        pass

            headers = {"Content-Disposition": 'attachment; filename="cvops_state_export.zip"'}
            return StreamingResponse(iterfile(), media_type="application/zip", headers=headers)

        @self.app.post("/cvops/state/import")
        async def cvops_state_import(bundle: UploadFile = File(...)) -> dict[str, Any]:
            raw = await bundle.read()
            if not raw:
                raise HTTPException(status_code=400, detail="empty bundle")
            import tempfile

            with tempfile.TemporaryDirectory() as td:
                tdir = Path(td).resolve()
                try:
                    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                        for info in zf.infolist():
                            rel = Path(info.filename)
                            if rel.is_absolute() or ".." in rel.parts:
                                raise HTTPException(status_code=400, detail="illegal path in archive")
                            target = (tdir / rel).resolve()
                            if not str(target).startswith(str(tdir)):
                                raise HTTPException(status_code=400, detail="zip entry escapes extract directory")
                            if info.is_dir():
                                target.mkdir(parents=True, exist_ok=True)
                                continue
                            target.parent.mkdir(parents=True, exist_ok=True)
                            with zf.open(info) as src, open(target, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                except zipfile.BadZipFile as exc:
                    raise HTTPException(status_code=400, detail=f"invalid zip: {exc}") from exc
                man_path = tdir / "manifest.json"
                if man_path.is_file():
                    try:
                        man = json.loads(man_path.read_text(encoding="utf-8"))
                    except Exception:
                        man = {}
                    if str(man.get("format") or "") != "cvops_state_bundle":
                        raise HTTPException(status_code=400, detail="manifest.format must be cvops_state_bundle")
                jobs_src = tdir / "jobs.db"
                if jobs_src.is_file():
                    CVOPS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(jobs_src, CVOPS_DB_PATH)
                reg_src = tdir / "model_registry.json"
                if reg_src.is_file():
                    MODEL_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(reg_src, MODEL_REGISTRY_PATH)
                ds_root = tdir / "dataset_registry"
                if ds_root.is_dir():
                    DATASET_REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
                    for path in ds_root.rglob("*"):
                        if path.is_file():
                            rel = path.relative_to(ds_root)
                            dest = DATASET_REGISTRY_DIR / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(path, dest)
            return {
                "imported": True,
                "note": "Jobs DB replaced on disk; restart Insight CV Ops if the queue looks inconsistent.",
            }

        @self.app.get("/models")
        async def list_models(request: Request) -> Response:
            try:
                return self._json_cache_response(self.models_payload(), request, max_age=10, stale_while_revalidate=60)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @self.app.post("/models/upload")
        async def upload_model_to_registry(
            scenario: str = Form(...),
            run_version: str = Form(""),
            file: UploadFile = File(...),
        ) -> dict[str, Any]:
            """Store weights under assets/models/registry_uploads and register in model_registry.json."""
            from mlops.pipeline.registry import MODEL_SUFFIXES, _is_model_candidate  # noqa: PLC0415

            scen = str(scenario or "").strip()
            if not scen:
                raise HTTPException(status_code=400, detail="scenario is required")
            try:
                mlops_registry.get_scenario_config(scen)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            suffix = Path(str(file.filename or "")).suffix.lower()
            if suffix not in MODEL_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported model extension {suffix!r}; allowed: {sorted(MODEL_SUFFIXES)}",
                )
            if suffix == ".mlpackage":
                raise HTTPException(
                    status_code=400,
                    detail="Directory bundles (.mlpackage) cannot be uploaded through this endpoint.",
                )

            rv = str(run_version or "").strip()
            if not rv:
                rv = f"upload_{uuid.uuid4().hex[:12]}"
            if not re.match(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$", rv):
                raise HTTPException(
                    status_code=400,
                    detail="run_version must be 1–64 characters: letters, digits, underscore, or hyphen.",
                )

            for existing in list_model_versions(scen):
                if str(existing.get("run_version") or "").strip() == rv:
                    raise HTTPException(
                        status_code=409,
                        detail=f"run_version {rv!r} already exists for scenario {scen!r}; pick another id.",
                    )

            dest_root = ROOT_DIR / "assets" / "models" / "registry_uploads" / scen
            dest_root.mkdir(parents=True, exist_ok=True)
            dest_path = dest_root / f"{rv}{suffix}"
            if dest_path.exists():
                raise HTTPException(status_code=409, detail=f"destination already exists: {dest_path}")

            try:
                with dest_path.open("wb") as out_f:
                    shutil.copyfileobj(file.file, out_f)
            except Exception as exc:
                if dest_path.exists():
                    try:
                        dest_path.unlink()
                    except Exception:
                        pass
                raise HTTPException(status_code=500, detail=f"failed to save upload: {exc}") from exc

            try:
                if not dest_path.is_file() or dest_path.stat().st_size == 0:
                    raise ValueError("empty upload")
                if not _is_model_candidate(dest_path):
                    raise ValueError("not a valid model file")
            except Exception as exc:
                try:
                    dest_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            try:
                entry = register_model_version(
                    scenario=scen,
                    run_version=rv,
                    artifacts={
                        "weights": str(dest_path.resolve()),
                        "final_model_file": dest_path.name,
                        "final_model_name": dest_path.name,
                        "final_model_path": str(dest_path.resolve()),
                    },
                    lineage={
                        "source": "cvops_manual_upload",
                        "original_filename": str(file.filename or ""),
                    },
                    metrics={"source": "manual_upload"},
                    set_candidate=True,
                )
            except Exception as exc:
                try:
                    dest_path.unlink(missing_ok=True)
                except Exception:
                    pass
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            self._emit_scenario_updated(scen)
            return {
                "ok": True,
                "scenario": scen,
                "run_version": rv,
                "model_ref": f"{scen}:{rv}",
                "weights_path": str(dest_path.resolve()),
                "entry": entry,
            }

        @self.app.post("/scenarios/{scenario}/model")
        async def set_scenario_model(scenario: str, req: ModelSelectRequest) -> dict[str, Any]:
            try:
                status = mlops_registry.set_scenario_base_model(scenario, req.model)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return status

        @self.app.post("/scenarios/{scenario}/dataset")
        async def set_scenario_dataset(scenario: str, req: DatasetSelectRequest) -> dict[str, Any]:
            try:
                status = mlops_registry.set_scenario_dataset(scenario, req.dataset)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return status

        @self.app.post("/scenarios/{scenario}/guard_profile")
        async def set_scenario_guard_profile(scenario: str, req: GuardProfileRequest) -> dict[str, Any]:
            try:
                status = mlops_registry.set_scenario_guard_profile(scenario, req.profile)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return status

        @self.app.get("/scenarios/{scenario}/guard")
        async def get_scenario_guard(scenario: str) -> dict[str, Any]:
            try:
                cfg = mlops_registry.get_scenario_config(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            hyperparams = dict(cfg.hyperparams or {})
            try:
                from mlops.pipeline.system_guard import build_training_guard as _build_guard
                return _build_guard(cfg.base_model, hyperparams, scenario=cfg.name)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @self.app.get("/scenarios/{scenario}/hyperparams")
        async def get_scenario_hyperparams(scenario: str) -> dict[str, Any]:
            try:
                cfg = mlops_registry.get_scenario_config(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            schema = mlops_registry.hyperparam_schema()
            schema_out: dict[str, dict[str, Any]] = {}
            for key, (kind, validator) in schema.items():
                entry: dict[str, Any] = {"kind": kind}
                if isinstance(validator, tuple):
                    if kind == "str_choices":
                        entry["choices"] = list(validator)
                    else:
                        entry["min"], entry["max"] = validator
                schema_out[key] = entry
            return {
                "scenario": cfg.name,
                "hyperparams": dict(cfg.hyperparams or {}),
                "schema": schema_out,
            }

        @self.app.post("/scenarios/{scenario}/hyperparams")
        async def post_scenario_hyperparams(scenario: str, req: HyperparamsPatchRequest) -> dict[str, Any]:
            try:
                status = mlops_registry.set_scenario_hyperparams(
                    scenario, req.updates or {}, reset=bool(req.reset)
                )
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return status

        @self.app.post("/scenarios/{scenario}/train")
        async def kick_training(
            scenario: str,
            req: Optional[TrainKickRequest] = None,
            resume: bool = Query(True, description="Resume from latest checkpoint if available"),
            save_period: int = Query(1, ge=1, le=100, description="Checkpoint frequency in epochs"),
        ) -> dict[str, Any]:
            try:
                cfg = mlops_registry.get_scenario_config(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            merged_hp = dict(cfg.hyperparams or {})
            if req is not None:
                for key in ("training_assets_root", "asset_save_root", "save_root"):
                    val = str(getattr(req, key, "") or "").strip()
                    if val:
                        merged_hp["training_assets_root"] = val
                        break
                dev = str(getattr(req, "device", "") or "").strip()
                if dev:
                    merged_hp["device"] = dev
            guard_model = str(getattr(req, "base_model_override", "") or "").strip() if req is not None else ""
            if not guard_model:
                guard_model = str(cfg.base_model or "")
            try:
                from mlops.pipeline.system_guard import build_training_guard as _build_guard
                guard = _build_guard(guard_model, merged_hp, scenario=cfg.name)
            except Exception:
                guard = {"status": "ok", "blocking_reasons": []}
            if str(guard.get("status") or "") == "blocked":
                reasons = guard.get("blocking_reasons") or []
                detail = {
                    "detail": "; ".join(reasons) if reasons else "training blocked by system guard",
                    "training_guard": guard,
                }
                raise HTTPException(status_code=400, detail=detail)
            job = self._queue_train_like_job(
                cfg=cfg,
                req=req,
                resume=bool(resume),
                save_period=int(save_period),
                trigger="manual",
                update_mode=False,
            )
            self._emit_job_status(job)
            self._emit_scenario_updated(cfg.name)
            return {
                "job_id": job.job_id,
                "state": job.state,
                "scenario": cfg.name,
                "training_guard": guard,
                "resume": bool(resume),
                "save_period": int(save_period),
                "trigger": "manual",
                "update_mode": False,
            }

        @self.app.get("/scenarios/{scenario}/custom_cells")
        async def get_custom_cells(scenario: str) -> dict[str, Any]:
            try:
                return self.custom_cells_payload(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.put("/scenarios/{scenario}/custom_cells")
        async def put_custom_cells(scenario: str, req: CustomCellsDraftRequest) -> dict[str, Any]:
            try:
                return write_draft(
                    scenario,
                    {
                        "cells": list(req.cells or []),
                        "scenario_datasets": list(req.scenario_datasets or []),
                    },
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.post("/scenarios/{scenario}/custom_cells/promote")
        async def promote_custom_cells(scenario: str, req: PromoteCustomCellsRequest) -> dict[str, Any]:
            try:
                out = promote_draft(scenario, req.template_name, cell_ids=req.cell_ids)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return out

        @self.app.post("/scenarios/{scenario}/update")
        async def kick_update(
            scenario: str,
            req: Optional[TrainKickRequest] = None,
            resume: bool = Query(False, description="Resume from latest checkpoint (default false for update runs)"),
            save_period: int = Query(1, ge=1, le=100, description="Checkpoint frequency in epochs"),
        ) -> dict[str, Any]:
            try:
                cfg = mlops_registry.get_scenario_config(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if not self._scenario_has_existing_model(cfg):
                raise HTTPException(
                    status_code=400,
                    detail="Update requires an existing trained model. Run Start Training first.",
                )
            merged_hp = dict(cfg.hyperparams or {})
            if req is not None:
                for key in ("training_assets_root", "asset_save_root", "save_root"):
                    val = str(getattr(req, key, "") or "").strip()
                    if val:
                        merged_hp["training_assets_root"] = val
                        break
                dev = str(getattr(req, "device", "") or "").strip()
                if dev:
                    merged_hp["device"] = dev
            guard_model = str(getattr(req, "base_model_override", "") or "").strip() if req is not None else ""
            if not guard_model:
                guard_model = str(cfg.base_model or "")
            try:
                from mlops.pipeline.system_guard import build_training_guard as _build_guard
                guard = _build_guard(guard_model, merged_hp, scenario=cfg.name)
            except Exception:
                guard = {"status": "ok", "blocking_reasons": []}
            if str(guard.get("status") or "") == "blocked":
                reasons = guard.get("blocking_reasons") or []
                detail = {
                    "detail": "; ".join(reasons) if reasons else "training blocked by system guard",
                    "training_guard": guard,
                }
                raise HTTPException(status_code=400, detail=detail)
            job = self._queue_train_like_job(
                cfg=cfg,
                req=req,
                resume=bool(resume),
                save_period=int(save_period),
                trigger="update",
                update_mode=True,
            )
            self._emit_job_status(job)
            self._emit_scenario_updated(cfg.name)
            return {
                "job_id": job.job_id,
                "state": job.state,
                "scenario": cfg.name,
                "training_guard": guard,
                "resume": bool(resume),
                "save_period": int(save_period),
                "trigger": "update",
                "update_mode": True,
            }

        @self.app.post("/scenarios/{scenario}/backbone_config")
        async def patch_backbone_config(scenario: str, req: BackboneConfigPatchRequest) -> dict[str, Any]:
            try:
                status = mlops_registry.patch_scenario_backbone_config(scenario, dict(req.patch or {}))
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return status

        @self.app.post("/scenarios/{scenario}/verify")
        async def verify_scenario(scenario: str, req: VerifyRequest) -> dict[str, Any]:
            try:
                payload = mlops_registry.mark_verified(scenario, note=req.note or "")
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return payload

        @self.app.delete("/scenarios/{scenario}/verify")
        async def unverify_scenario(scenario: str) -> dict[str, Any]:
            removed = mlops_registry.clear_verified(scenario)
            self._emit_scenario_updated(scenario)
            return {"cleared": removed}

        @self.app.get("/datasets/{scenario}")
        async def list_dataset(scenario: str) -> dict[str, Any]:
            try:
                items = mlops_registry.list_dataset_entries(scenario)
                split_counts = mlops_registry.dataset_split_counts(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            return {
                "scenario": scenario,
                "count": len(items),
                "images": items,
                "split_counts": split_counts,
            }

        @self.app.get("/datasets/{scenario}/thumb/{name:path}")
        async def dataset_thumb(scenario: str, name: str) -> dict[str, Any]:
            try:
                match = mlops_registry.resolve_dataset_image_path(scenario, name)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if match is None:
                raise HTTPException(status_code=404, detail="image not found")
            image = cv2.imread(str(match))
            if image is None:
                raise HTTPException(status_code=400, detail="unable to decode")
            h, w = image.shape[:2]
            max_side = 160
            scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
            if scale < 1.0:
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                raise HTTPException(status_code=500, detail="encode failed")
            return {"name": name, "thumb_b64": base64.b64encode(buf.tobytes()).decode("ascii")}

        @self.app.get("/datasets/{scenario}/label/{name:path}")
        async def dataset_label_text(scenario: str, name: str) -> dict[str, Any]:
            try:
                match = mlops_registry.resolve_dataset_image_path(scenario, name)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if match is None:
                raise HTTPException(status_code=404, detail="image not found")
            label_path = mlops_registry.resolve_dataset_label_path(match)
            if label_path is None or not label_path.exists():
                return {
                    "relative_path": name,
                    "has_label": False,
                    "text": "",
                    "line_count": 0,
                }
            try:
                text = label_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            nonempty_lines = [ln for ln in text.splitlines() if ln.strip()]
            return {
                "relative_path": name,
                "has_label": True,
                "text": text,
                "line_count": len(nonempty_lines),
            }

        @self.app.post("/database/create_yolo_template")
        async def database_create_yolo_template(req: YoloDatasetTemplateCreateRequest) -> dict[str, Any]:
            try:
                payload = mlops_registry.create_yolo_detection_dataset_template(
                    req.name.strip(),
                    classes=req.classes,
                    unique_slug=bool(req.unique),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return dict(payload)

        @self.app.get("/database")
        async def database_list(request: Request) -> Response:
            try:
                return self._json_cache_response(self.database_list_payload(), request, max_age=5, stale_while_revalidate=30)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        @self.app.post("/database/import_folder")
        async def database_import_folder(req: DatasetFolderImportRequest) -> dict[str, Any]:
            src = Path(str(req.source_path or "")).expanduser()
            if not src.exists():
                raise HTTPException(status_code=400, detail=f"source path does not exist: {src}")
            if not src.is_dir():
                raise HTTPException(status_code=400, detail="source path must be a directory")
            try:
                source = src.resolve()
                source_payload = mlops_registry.inspect_library_dataset_at(source)
                source_fmt = str(source_payload.get("format") or mlops_registry.LIBRARY_DATASET_FORMAT_UNKNOWN)
                preferred = str(req.name or "").strip() or source.name
                if source_fmt == mlops_registry.LIBRARY_DATASET_FORMAT_AUDIOFOLDER:
                    slug = mlops_registry.pick_unique_audio_dataset_slug(preferred)
                    dest = (mlops_registry.ensure_ml_audio_root() / slug).resolve()
                else:
                    slug = mlops_registry.pick_unique_library_dataset_slug(preferred)
                    dest = (mlops_registry.ensure_database_root() / slug).resolve()
                try:
                    dest.relative_to(source)
                except ValueError:
                    pass
                else:
                    raise ValueError("destination would be created inside the source folder")
                shutil.copytree(source, dest, symlinks=True)
                payload = mlops_registry.inspect_library_dataset_at(dest)
                fmt = str(payload.get("format") or mlops_registry.LIBRARY_DATASET_FORMAT_UNKNOWN)
                count = int(payload.get("count") or 0)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"dataset import failed: {exc}") from exc
            return {
                "slug": slug,
                "source_path": str(source),
                "path": str(dest),
                "format": fmt,
                "category": mlops_registry.dataset_category(fmt),
                "count": count,
                "split_counts": payload.get("split_counts", {}) if isinstance(payload, dict) else {},
                "classes": payload.get("classes", []) if isinstance(payload, dict) else [],
                "detection_label_count": payload.get("detection_label_count", 0) if isinstance(payload, dict) else 0,
                "missing_detection_label_count": payload.get("missing_detection_label_count", 0) if isinstance(payload, dict) else 0,
            }

        def _store_tabular_csv_bytes(csv_bytes: bytes, preferred_name: str) -> dict[str, Any]:
            """Write normalized CSV bytes to TABULAR_DATASETS_ROOT/<unique-slug>.csv."""
            preferred = str(preferred_name or "").strip() or "dataset"
            safe = "".join(c if (c.isalnum() or c in ("-", "_")) else "-" for c in preferred)
            safe = "-".join(part for part in safe.split("-") if part)[:60] or "dataset"
            try:
                base_slug = mlops_registry.sanitize_library_dataset_slug(safe)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            existing = {str(entry.get("name") or "") for entry in mlops_registry.list_tabular_dataset_entries()}
            slug = base_slug
            counter = 2
            while slug in existing:
                slug = f"{base_slug}-{counter}"
                counter += 1
            root = mlops_registry.TABULAR_DATASETS_ROOT
            try:
                root.mkdir(parents=True, exist_ok=True)
                target = (root / f"{slug}.csv").resolve()
                target.relative_to(root.resolve())  # guard against path traversal
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid dataset path: {exc}") from exc
            try:
                target.write_bytes(csv_bytes)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to save upload: {exc}") from exc
            try:
                rel = str(target.relative_to(mlops_registry.REPO_ROOT))
            except Exception:
                rel = str(target)
            return {
                "slug": slug,
                "name": slug,
                "filename": target.name,
                "path": rel,
                "size_bytes": target.stat().st_size,
                "category": mlops_registry.DATASET_CATEGORY_TABULAR,
                "format": mlops_registry.LIBRARY_DATASET_FORMAT_CSV,
            }

        async def _ingest_tabular_upload(file: UploadFile, name: str) -> dict[str, Any]:
            filename = str(file.filename or "")
            suffix = Path(filename).suffix.lower()
            if suffix not in SUPPORTED_TABULAR_SUFFIXES:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"unsupported tabular extension: {suffix or '(none)'} "
                        f"(expected one of {', '.join(SUPPORTED_TABULAR_SUFFIXES)})"
                    ),
                )
            raw = await file.read()
            if not raw:
                raise HTTPException(status_code=400, detail="empty upload")
            try:
                csv_bytes = _tabular_to_csv_bytes(raw, suffix, filename)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            preferred = str(name or "").strip() or Path(filename).stem or "dataset"
            result = _store_tabular_csv_bytes(csv_bytes, preferred)
            result["source_format"] = suffix.lstrip(".")
            return result

        @self.app.post("/database/upload_csv")
        async def database_upload_csv(
            file: UploadFile = File(...),
            name: str = Form(""),
        ) -> dict[str, Any]:
            """Onboard a single tabular file as a library dataset under mlops/datasets/.

            Accepts csv/tsv/xlsx/xls/parquet/pq/json/jsonl/ndjson; all are normalized to
            TABULAR_DATASETS_ROOT/<slug>.csv so they are auto-discovered by
            list_tabular_dataset_entries() and immediately usable by tabular_profile +
            torch_tabular. (Name kept for back-compat; see also /database/upload_tabular.)
            """
            return await _ingest_tabular_upload(file, name)

        @self.app.post("/database/upload_tabular")
        async def database_upload_tabular(
            file: UploadFile = File(...),
            name: str = Form(""),
        ) -> dict[str, Any]:
            """Onboard a single tabular file of any supported format (csv/tsv/xlsx/xls/
            parquet/pq/json/jsonl/ndjson), normalizing it to CSV."""
            return await _ingest_tabular_upload(file, name)

        @self.app.post("/database/import_tabular_folder")
        async def database_import_tabular_folder(req: TabularFolderImportRequest) -> dict[str, Any]:
            """Batch-import every supported tabular file in a local folder as datasets."""
            source = str(req.source_path or "").strip()
            if not source:
                raise HTTPException(status_code=400, detail="source_path is required")
            folder = Path(source).expanduser()
            if not folder.exists() or not folder.is_dir():
                raise HTTPException(status_code=400, detail=f"source_path is not a directory: {folder}")
            globber = folder.rglob("*") if bool(req.recursive) else folder.glob("*")
            candidates = sorted(
                p for p in globber
                if p.is_file() and p.suffix.lower() in SUPPORTED_TABULAR_SUFFIXES
            )
            imported: list[dict[str, Any]] = []
            errors: list[str] = []
            for path in candidates:
                try:
                    raw = path.read_bytes()
                    if not raw:
                        errors.append(f"{path.name}: empty file")
                        continue
                    csv_bytes = _tabular_to_csv_bytes(raw, path.suffix.lower(), path.name)
                    result = _store_tabular_csv_bytes(csv_bytes, path.stem)
                    result["source_format"] = path.suffix.lower().lstrip(".")
                    result["source"] = str(path)
                    imported.append(result)
                except HTTPException as exc:
                    errors.append(f"{path.name}: {exc.detail}")
                except ValueError as exc:
                    errors.append(f"{path.name}: {exc}")
                except Exception as exc:
                    errors.append(f"{path.name}: {exc}")
            return {
                "source_path": str(folder),
                "recursive": bool(req.recursive),
                "found": len(candidates),
                "imported": imported,
                "imported_count": len(imported),
                "errors": errors,
            }

        @self.app.get("/database/{slug}")
        async def database_list_dataset(slug: str) -> dict[str, Any]:
            try:
                return self.database_dataset_payload(slug)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @self.app.post("/database/{slug}/clone_subset")
        async def database_clone_subset(slug: str, req: DatasetSubsetCloneRequest) -> dict[str, Any]:
            def _split_name(value: object, *, default: str = "train") -> str:
                raw = str(value or "").strip().lower()
                if raw in {"valid", "validation"}:
                    raw = "val"
                if raw not in {"train", "val", "test"}:
                    raw = default
                return raw

            def _rel_after_images_root(img_path: Path, source_root: Path, fallback_rel: str) -> Path:
                fallback = Path(str(fallback_rel or img_path.name).strip().lstrip("/"))
                for parent in img_path.resolve().parents:
                    if parent.name != "images":
                        continue
                    try:
                        parent.relative_to(source_root)
                    except Exception:
                        continue
                    try:
                        rel = img_path.resolve().relative_to(parent)
                    except Exception:
                        continue
                    parts = list(rel.parts)
                    if parts and str(parts[0]).lower() in {"train", "val", "valid", "validation", "test"}:
                        parts = parts[1:]
                    return Path(*parts) if parts else Path(img_path.name)
                parts = list(fallback.parts)
                if parts and str(parts[0]).lower() == "images":
                    parts = parts[1:]
                if parts and str(parts[0]).lower() in {"train", "val", "valid", "validation", "test"}:
                    parts = parts[1:]
                return Path(*parts) if parts else Path(img_path.name)

            def _unique_dest(image_root: Path, label_root: Path, rel_after: Path, suffix: str) -> tuple[Path, Path]:
                folder_parts = [
                    self._sanitize_dataset_stem(part)
                    for part in rel_after.parts[:-1]
                    if str(part or "").strip()
                ]
                rel_parent = Path(*folder_parts) if folder_parts else Path()
                stem = self._sanitize_dataset_stem(rel_after.stem or "image")
                img = image_root / rel_parent / f"{stem}{suffix}"
                lbl = label_root / rel_parent / f"{stem}.txt"
                idx = 1
                while img.exists() or lbl.exists():
                    img = image_root / rel_parent / f"{stem}-{idx:02d}{suffix}"
                    lbl = label_root / rel_parent / f"{stem}-{idx:02d}.txt"
                    idx += 1
                return img.resolve(), lbl.resolve()

            def _entry_has_coordinate_label(entry: dict[str, Any], source_fmt: str) -> bool:
                if bool(entry.get("has_detection_label")):
                    return True
                if str(entry.get("label_path") or entry.get("detection_label_path") or "").strip():
                    return True
                return source_fmt == mlops_registry.LIBRARY_DATASET_FORMAT_YOLO and bool(entry.get("has_label"))

            def _entry_coordinate_label_path(entry: dict[str, Any], img_path: Path, source_root: Path) -> Optional[Path]:
                for key in ("label_path", "detection_label_path"):
                    raw = str(entry.get(key) or "").strip()
                    if not raw:
                        continue
                    candidate = Path(raw)
                    if not candidate.is_absolute():
                        candidate = source_root / candidate
                    try:
                        resolved = candidate.resolve()
                        resolved.relative_to(source_root)
                    except Exception:
                        continue
                    if resolved.exists() and resolved.is_file():
                        return resolved
                label_path = mlops_registry.resolve_dataset_label_path(img_path)
                if label_path is not None and label_path.exists() and label_path.is_file():
                    return label_path
                return None

            try:
                source_slug = mlops_registry.sanitize_library_dataset_slug(slug)
                source_root = mlops_registry.resolve_library_dataset_path(source_slug).resolve()
                source_payload = mlops_registry.inspect_library_dataset_at(source_root)
                source_fmt = str((source_payload or {}).get("format") or "")
                entries = [
                    e for e in list(source_payload.get("images") or [])
                    if isinstance(e, dict) and str(e.get("relative_path") or e.get("name") or "").strip()
                ]
                if not entries:
                    raise ValueError("source dataset has no image entries to copy")

                by_rel = {
                    str(e.get("relative_path") or e.get("name") or "").strip().replace("\\", "/"): e
                    for e in entries
                }
                requested = [
                    str(p or "").strip().lstrip("/").replace("\\", "/")
                    for p in (req.relative_paths or [])
                    if str(p or "").strip()
                ]
                selected = [by_rel[p] for p in requested if p in by_rel] if requested else list(entries)
                if req.only_labeled:
                    selected = [e for e in selected if _entry_has_coordinate_label(e, source_fmt)]
                limit = int(req.max_images or 0)
                if limit > 0:
                    selected = selected[:limit]
                if not selected:
                    raise ValueError("no matching images selected for clone")

                raw_classes = source_payload.get("classes") if isinstance(source_payload, dict) else []
                classes = [str(c).strip() for c in (raw_classes or []) if str(c).strip()] if isinstance(raw_classes, list) else []
                template = mlops_registry.create_yolo_detection_dataset_template(
                    req.name.strip(),
                    classes=classes or ["object"],
                    unique_slug=bool(req.unique),
                )
                out_slug = str(template.get("slug") or "").strip()
                dest_root = mlops_registry.resolve_library_dataset_path(out_slug).resolve()
                copied: list[dict[str, Any]] = []
                errors: list[str] = []
                target_default_split = _split_name(req.target_split)

                for entry in selected:
                    rel = str(entry.get("relative_path") or entry.get("name") or "").strip().lstrip("/").replace("\\", "/")
                    img_path = mlops_registry.resolve_dataset_image_path_at(source_root, rel)
                    if img_path is None:
                        errors.append(f"{rel}: image not found")
                        continue
                    source_split = _split_name(entry.get("split"), default=target_default_split)
                    dest_split = source_split if bool(req.preserve_splits) else target_default_split
                    image_root = (dest_root / "images" / dest_split).resolve()
                    label_root = (dest_root / "labels" / dest_split).resolve()
                    image_root.mkdir(parents=True, exist_ok=True)
                    label_root.mkdir(parents=True, exist_ok=True)
                    rel_after = _rel_after_images_root(img_path, source_root, rel)
                    dest_img, dest_label = _unique_dest(image_root, label_root, rel_after, img_path.suffix.lower())
                    try:
                        dest_img.parent.mkdir(parents=True, exist_ok=True)
                        dest_label.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(img_path, dest_img)
                        label_path = _entry_coordinate_label_path(entry, img_path, source_root)
                        copied_label = False
                        if bool(req.include_labels) and label_path is not None and label_path.exists():
                            shutil.copy2(label_path, dest_label)
                            copied_label = True
                        else:
                            dest_label.write_text("", encoding="utf-8")
                        copied.append(
                            {
                                "source_relative_path": rel,
                                "source_path": str(img_path),
                                "source_label_path": str(label_path) if label_path is not None and label_path.exists() else "",
                                "source_split": source_split,
                                "dest_split": dest_split,
                                "dest_image_relative_path": dest_img.relative_to(dest_root).as_posix(),
                                "dest_label_relative_path": dest_label.relative_to(dest_root).as_posix(),
                                "copied_label_coordinates": copied_label,
                            }
                        )
                    except Exception as exc:
                        errors.append(f"{rel}: {exc}")

                if not copied:
                    shutil.rmtree(dest_root, ignore_errors=True)
                    raise ValueError("clone did not copy any images: " + "; ".join(errors[:3]))

                manifest_path = dest_root / "cvops_subset_manifest.json"
                manifest = {
                    "kind": "cvops.dataset_subset_clone",
                    "created_at_unix": time.time(),
                    "source_slug": source_slug,
                    "source_path": str(source_root),
                    "output_slug": out_slug,
                    "output_path": str(dest_root),
                    "requested_count": len(requested) if requested else len(entries),
                    "copied_count": len(copied),
                    "target_split": target_default_split,
                    "preserve_splits": bool(req.preserve_splits),
                    "include_labels": bool(req.include_labels),
                    "only_labeled": bool(req.only_labeled),
                    "classes": classes or ["object"],
                    "entries": copied,
                    "errors": errors,
                }
                manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                payload = mlops_registry.inspect_library_dataset_at(dest_root)
                return {
                    "source_slug": source_slug,
                    "output_slug": out_slug,
                    "path": str(dest_root),
                    "copied": len(copied),
                    "error_count": len(errors),
                    "errors": errors,
                    "manifest": str(manifest_path.resolve()),
                    "format": payload.get("format") if isinstance(payload, dict) else "",
                    "split_counts": payload.get("split_counts", {}) if isinstance(payload, dict) else {},
                    "classes": payload.get("classes", []) if isinstance(payload, dict) else [],
                }
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @self.app.get("/database/{slug}/tabular_profile")
        async def database_tabular_profile(
            slug: str,
            name: str = Query("", alias="name"),
            max_rows: int = Query(5000, alias="max_rows", ge=200, le=50000),
        ) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                payload = mlops_registry.inspect_library_dataset_at(dataset_root)
                fmt = str((payload or {}).get("format") or "")
                if fmt not in {
                    mlops_registry.LIBRARY_DATASET_FORMAT_CSV,
                    mlops_registry.LIBRARY_DATASET_FORMAT_FACE_CSV,
                }:
                    raise ValueError("dataset is not tabular (.csv)")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            csv_files = payload.get("csv_files") if isinstance(payload, dict) else []
            available = [str((entry or {}).get("name") or "").strip() for entry in (csv_files or []) if isinstance(entry, dict)]
            available = [entry for entry in available if entry]
            selected_name = str(name or "").strip()
            if not selected_name and available:
                selected_name = available[0]

            # Single-file tabular datasets (mlops/datasets/<slug>.csv) resolve to the
            # .csv file itself; directory datasets need a csv selected within them.
            try:
                if dataset_root.is_file():
                    csv_path = dataset_root.resolve()
                else:
                    if not selected_name:
                        raise HTTPException(status_code=404, detail="no csv file found for dataset")
                    csv_path = (dataset_root / selected_name).resolve()
                    csv_path.relative_to(dataset_root.resolve())
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid csv path: {selected_name}") from exc
            if not csv_path.exists() or not csv_path.is_file():
                raise HTTPException(status_code=404, detail=f"csv not found: {selected_name or csv_path.name}")
            if csv_path.suffix.lower() != ".csv":
                raise HTTPException(status_code=400, detail="requested file is not .csv")

            rows: list[dict[str, str]] = []
            columns: list[str] = []
            analyzed_rows = 0
            truncated = False
            try:
                with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.DictReader(handle)
                    columns = [str(col or "").strip() for col in (reader.fieldnames or []) if str(col or "").strip()]
                    for raw in reader:
                        if not isinstance(raw, dict):
                            continue
                        row: dict[str, str] = {}
                        for col in columns:
                            value = raw.get(col, "")
                            row[col] = "" if value is None else str(value).strip()
                        rows.append(row)
                        analyzed_rows += 1
                        if analyzed_rows >= int(max_rows):
                            truncated = True
                            break
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to read csv: {exc}") from exc

            if not columns:
                raise HTTPException(status_code=400, detail="csv has no header row")
            if analyzed_rows < 2:
                raise HTTPException(status_code=400, detail="csv has fewer than 2 rows")

            missing_tokens = {"", "na", "n/a", "null", "none", "nan", "?"}
            missing_counts: dict[str, int] = {col: 0 for col in columns}
            numeric_values: dict[str, list[float]] = {col: [] for col in columns}
            categorical_counts: dict[str, Counter[str]] = {col: Counter() for col in columns}
            non_missing_counts: dict[str, int] = {col: 0 for col in columns}

            def _is_missing(text: str) -> bool:
                return text.strip().lower() in missing_tokens

            def _parse_numeric(text: str) -> Optional[float]:
                value = text.strip()
                if _is_missing(value):
                    return None
                try:
                    parsed = float(value)
                except Exception:
                    return None
                if not math.isfinite(parsed):
                    return None
                return parsed

            for row in rows:
                for col in columns:
                    value = row.get(col, "")
                    if _is_missing(value):
                        missing_counts[col] += 1
                        continue
                    non_missing_counts[col] += 1
                    parsed = _parse_numeric(value)
                    if parsed is not None:
                        numeric_values[col].append(parsed)
                    else:
                        categorical_counts[col][value] += 1

            numeric_cols: list[str] = []
            categorical_cols: list[str] = []
            for col in columns:
                non_missing = max(1, non_missing_counts[col])
                numeric_count = len(numeric_values[col])
                if numeric_count >= 5 and (numeric_count / float(non_missing)) >= 0.6:
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)

            distributions: list[dict[str, Any]] = []
            for col in numeric_cols:
                vals = numeric_values[col]
                if not vals:
                    continue
                arr = np.asarray(vals, dtype=np.float64)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                arr.sort()
                missing = missing_counts[col]
                std = float(np.std(arr))
                mean = float(np.mean(arr))
                if std > 0:
                    z = (arr - mean) / std
                    skew = float(np.mean(z ** 3))
                    kurt = float(np.mean(z ** 4) - 3.0)
                else:
                    skew = 0.0
                    kurt = 0.0
                distributions.append(
                    {
                        "name": col,
                        "dtype": "float64",
                        "count": analyzed_rows,
                        "missing": missing,
                        "missing_pct": round((missing / float(max(1, analyzed_rows))) * 100.0, 2),
                        "unique": int(len(set(vals))),
                        "mean": round(mean, 3),
                        "std": round(std, 3),
                        "min": round(float(arr[0]), 6),
                        "q25": round(float(np.percentile(arr, 25)), 6),
                        "median": round(float(np.percentile(arr, 50)), 6),
                        "q75": round(float(np.percentile(arr, 75)), 6),
                        "max": round(float(arr[-1]), 6),
                        "skewness": round(skew, 3),
                        "kurtosis": round(kurt, 3),
                    }
                )

            cat_distributions: list[dict[str, Any]] = []
            for col in categorical_cols[:30]:
                missing = missing_counts[col]
                top = categorical_counts[col].most_common(12)
                den = float(max(1, analyzed_rows - missing))
                cat_distributions.append(
                    {
                        "name": col,
                        "dtype": "object",
                        "count": analyzed_rows,
                        "missing": missing,
                        "missing_pct": round((missing / float(max(1, analyzed_rows))) * 100.0, 2),
                        "unique": int(len(categorical_counts[col])),
                        "top_values": [
                            {"value": value, "count": int(count), "pct": round((count / den) * 100.0, 2)}
                            for value, count in top
                        ],
                    }
                )

            corr_pairs: list[dict[str, Any]] = []
            corr_cols = numeric_cols[:15]
            for idx, left in enumerate(corr_cols):
                for right in corr_cols[idx + 1 :]:
                    x_vals: list[float] = []
                    y_vals: list[float] = []
                    for row in rows:
                        x = _parse_numeric(row.get(left, ""))
                        y = _parse_numeric(row.get(right, ""))
                        if x is None or y is None:
                            continue
                        x_vals.append(x)
                        y_vals.append(y)
                    if len(x_vals) < 10:
                        continue
                    x_arr = np.asarray(x_vals, dtype=np.float64)
                    y_arr = np.asarray(y_vals, dtype=np.float64)
                    if x_arr.size < 10 or y_arr.size < 10:
                        continue
                    x_std = float(np.std(x_arr))
                    y_std = float(np.std(y_arr))
                    if x_std <= 0 or y_std <= 0:
                        continue
                    corr = float(np.corrcoef(x_arr, y_arr)[0, 1])
                    if not math.isfinite(corr):
                        continue
                    if abs(corr) >= 0.3:
                        corr_pairs.append(
                            {
                                "feature_1": left,
                                "feature_2": right,
                                "correlation": round(corr, 4),
                            }
                        )
            corr_pairs.sort(key=lambda item: abs(float(item.get("correlation") or 0.0)), reverse=True)

            issues: list[dict[str, Any]] = []
            for entry in distributions:
                missing_pct = float(entry.get("missing_pct") or 0.0)
                name_value = str(entry.get("name") or "")
                if missing_pct > 50.0:
                    issues.append(
                        {
                            "severity": "critical",
                            "category": "missing",
                            "message": f"'{name_value}' has {missing_pct:.2f}% missing values",
                        }
                    )
                elif missing_pct > 10.0:
                    issues.append(
                        {
                            "severity": "warning",
                            "category": "missing",
                            "message": f"'{name_value}' has {missing_pct:.2f}% missing values",
                        }
                    )

            duplicate_rows = max(0, analyzed_rows - len({tuple(row.get(col, "") for col in columns) for row in rows}))
            if duplicate_rows > 0:
                dup_ratio = duplicate_rows / float(max(1, analyzed_rows))
                issues.append(
                    {
                        "severity": "critical" if dup_ratio > 0.05 else "warning",
                        "category": "duplicate",
                        "message": f"{duplicate_rows} duplicate rows ({dup_ratio * 100.0:.1f}%) in analyzed sample",
                    }
                )

            for pair in corr_pairs:
                corr = float(pair.get("correlation") or 0.0)
                if abs(corr) > 0.95:
                    issues.append(
                        {
                            "severity": "warning",
                            "category": "leakage",
                            "message": f"'{pair['feature_1']}' and '{pair['feature_2']}' have r={corr:.3f}",
                        }
                    )

            embedding: list[dict[str, float]] = []
            if len(numeric_cols) >= 2:
                x_key = numeric_cols[0]
                y_key = numeric_cols[1]
                for row in rows:
                    x = _parse_numeric(row.get(x_key, ""))
                    y = _parse_numeric(row.get(y_key, ""))
                    if x is None or y is None:
                        continue
                    embedding.append({"x": float(x), "y": float(y)})
                    if len(embedding) >= 2000:
                        break

            quality_score = 1.0
            for issue in issues:
                severity = str(issue.get("severity") or "info")
                if severity == "critical":
                    quality_score -= 0.15
                elif severity == "warning":
                    quality_score -= 0.05
                else:
                    quality_score -= 0.01
            quality_score = max(0.0, min(1.0, quality_score))

            return {
                "slug": slug,
                "name": selected_name,
                "source": str(csv_path.resolve()),
                "archetype": "TABULAR",
                "n_samples": analyzed_rows,
                "n_features": len(columns),
                "numeric_features": len(distributions),
                "categorical_features": len(cat_distributions),
                "distributions": distributions,
                "cat_distributions": cat_distributions,
                "corr_pairs": corr_pairs[:60],
                "issues": issues[:120],
                "quality_score": round(float(quality_score), 3),
                "embedding": embedding,
                "num_cols": numeric_cols,
                "cat_cols": categorical_cols,
                "analyzed_rows": analyzed_rows,
                "truncated": bool(truncated),
            }

        def _resolve_tabular_csv(slug: str, name: str = "") -> Path:
            """Resolve a single .csv file for a tabular library dataset slug.

            Mirrors the resolution used by /database/{slug}/tabular_profile so the
            transform/split endpoints operate on the same file.
            """
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                payload = mlops_registry.inspect_library_dataset_at(dataset_root)
                fmt = str((payload or {}).get("format") or "")
                if fmt not in {
                    mlops_registry.LIBRARY_DATASET_FORMAT_CSV,
                    mlops_registry.LIBRARY_DATASET_FORMAT_FACE_CSV,
                }:
                    raise ValueError("dataset is not tabular (.csv)")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            selected = str(name or "").strip()
            try:
                if dataset_root.is_file():
                    csv_path = dataset_root.resolve()
                else:
                    csv_files = payload.get("csv_files") if isinstance(payload, dict) else []
                    available = [
                        str((entry or {}).get("name") or "").strip()
                        for entry in (csv_files or [])
                        if isinstance(entry, dict)
                    ]
                    available = [entry for entry in available if entry]
                    if not selected and available:
                        selected = available[0]
                    if not selected:
                        raise HTTPException(status_code=404, detail="no csv file found for dataset")
                    csv_path = (dataset_root / selected).resolve()
                    csv_path.relative_to(dataset_root.resolve())
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"invalid csv path: {selected}") from exc
            if not csv_path.exists() or not csv_path.is_file():
                raise HTTPException(status_code=404, detail=f"csv not found: {selected or csv_path.name}")
            if csv_path.suffix.lower() != ".csv":
                raise HTTPException(status_code=400, detail="requested file is not .csv")
            return csv_path

        _TABULAR_MISSING_TOKENS = {"", "na", "n/a", "null", "none", "nan", "?"}

        # Safety cap for full-file (mutating) reads so multi-GB CSVs fail with a clear
        # message instead of silently exhausting memory. Override via env.
        try:
            _TABULAR_MAX_ROWS = int(os.environ.get("CVOPS_TABULAR_MAX_ROWS", "2000000"))
        except ValueError:
            _TABULAR_MAX_ROWS = 2_000_000

        def _read_tabular_table(
            csv_path: Path,
            *,
            max_rows: Optional[int] = None,
            hard_cap: Optional[int] = None,
        ) -> tuple[list[str], list[list[str]]]:
            """Stream a CSV into (header, body), padding rows to header width.

            max_rows: stop after this many data rows (sample reads; analysis only).
            hard_cap: refuse files with more than this many data rows (full-file mutating
                      reads) with HTTP 413, rather than loading an unbounded file.
            """
            header: list[str] = []
            body: list[list[str]] = []
            width = 0
            try:
                with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle)
                    for i, row in enumerate(reader):
                        if i == 0:
                            header = [str(col or "").strip() for col in row]
                            width = len(header)
                            continue
                        if max_rows is not None and len(body) >= int(max_rows):
                            break
                        if hard_cap is not None and len(body) >= int(hard_cap):
                            raise HTTPException(
                                status_code=413,
                                detail=(
                                    f"dataset exceeds the in-memory transform cap "
                                    f"({hard_cap:,} rows). Trim it first or raise "
                                    f"CVOPS_TABULAR_MAX_ROWS."
                                ),
                            )
                        if len(row) < width:
                            row = list(row) + [""] * (width - len(row))
                        elif len(row) > width:
                            row = list(row[:width])
                        else:
                            row = list(row)
                        body.append(row)
            except HTTPException:
                raise
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to read csv: {exc}") from exc
            if not header:
                raise HTTPException(status_code=400, detail="csv is empty")
            return header, body

        def _write_tabular_table(csv_path: Path, header: list[str], body: list[list[str]]) -> Path:
            backup = csv_path.with_suffix(csv_path.suffix + ".bak")
            try:
                shutil.copy2(csv_path, backup)
            except Exception:
                backup = csv_path  # best-effort; do not block the write
            tmp = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}")
            try:
                with tmp.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(header)
                    writer.writerows(body)
                tmp.replace(csv_path)
            except Exception as exc:
                try:
                    tmp.unlink(missing_ok=True)
                except Exception:
                    pass
                raise HTTPException(status_code=500, detail=f"failed to write csv: {exc}") from exc
            return backup

        def _tabular_dims(csv_path: Path) -> dict[str, int]:
            """Stream a CSV to count rows/cols without loading it into memory."""
            rows = 0
            cols = 0
            try:
                with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
                    reader = csv.reader(handle)
                    for i, row in enumerate(reader):
                        if i == 0:
                            cols = len(row)
                        else:
                            rows += 1
            except Exception:
                return {"rows": 0, "cols": 0}
            return {"rows": rows, "cols": cols}

        def _is_tabular_missing(text: str) -> bool:
            return str(text or "").strip().lower() in _TABULAR_MISSING_TOKENS

        def _tabular_history_path(csv_path: Path) -> Path:
            return csv_path.with_name(f"{csv_path.stem}.transforms.json")

        def _read_tabular_history(csv_path: Path) -> dict[str, Any]:
            import json as _json

            path = _tabular_history_path(csv_path)
            if not path.exists():
                return {"version": 1, "csv": csv_path.name, "entries": []}
            try:
                data = _json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and isinstance(data.get("entries"), list):
                    return data
            except Exception:
                pass
            return {"version": 1, "csv": csv_path.name, "entries": []}

        def _append_tabular_history(csv_path: Path, entry: dict[str, Any]) -> int:
            """Append a provenance entry to <stem>.transforms.json; returns the revision."""
            import json as _json

            log = _read_tabular_history(csv_path)
            entries = log.get("entries") if isinstance(log.get("entries"), list) else []
            revision = len(entries) + 1
            record = dict(entry)
            record["revision"] = revision
            record["at"] = datetime.now(timezone.utc).isoformat()
            entries.append(record)
            log["entries"] = entries
            log["version"] = 1
            log["csv"] = csv_path.name
            log["updated_at"] = record["at"]
            path = _tabular_history_path(csv_path)
            try:
                tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
                tmp.write_text(_json.dumps(log, indent=2), encoding="utf-8")
                tmp.replace(path)
            except Exception:
                pass
            return revision

        @self.app.post("/database/{slug}/tabular_transform")
        async def database_tabular_transform(slug: str, req: TabularTransformRequest) -> dict[str, Any]:
            """Apply profile-driven fix operations to a tabular dataset CSV in place.

            Each op corresponds to an issue category emitted by tabular_profile, so the
            UI can render a one-click Fix per detected issue. A .bak sibling is kept for
            single-step undo.
            """
            csv_path = _resolve_tabular_csv(slug, req.name)
            header, body = _read_tabular_table(csv_path, hard_cap=_TABULAR_MAX_ROWS)
            if not header:
                raise HTTPException(status_code=400, detail="csv has no header row")

            before = {"rows": len(body), "cols": len(header)}
            applied: list[dict[str, Any]] = []
            col_index = {name: idx for idx, name in enumerate(header)}

            def _drop_columns(names: list[str]) -> int:
                keep = [i for i, col in enumerate(header) if col not in set(names)]
                dropped = len(header) - len(keep)
                if dropped <= 0:
                    return 0
                new_header = [header[i] for i in keep]
                header[:] = new_header
                for r_idx, row in enumerate(body):
                    body[r_idx] = [row[i] for i in keep]
                col_index.clear()
                col_index.update({name: idx for idx, name in enumerate(header)})
                return dropped

            for op_model in req.ops:
                op = str(op_model.op or "").strip().lower()
                if op == "drop_duplicate_rows":
                    seen: set[tuple[str, ...]] = set()
                    kept: list[list[str]] = []
                    removed = 0
                    for row in body:
                        key = tuple(row)
                        if key in seen:
                            removed += 1
                            continue
                        seen.add(key)
                        kept.append(row)
                    body[:] = kept
                    applied.append({"op": op, "rows_removed": removed})
                elif op == "drop_columns":
                    names = [str(c).strip() for c in (op_model.columns or []) if str(c).strip()]
                    dropped = _drop_columns(names)
                    applied.append({"op": op, "columns_dropped": dropped, "requested": names})
                elif op == "drop_high_missing_columns":
                    thresh = float(op_model.threshold_pct)
                    total = max(1, len(body))
                    targets: list[str] = []
                    for col, idx in list(col_index.items()):
                        missing = sum(1 for row in body if _is_tabular_missing(row[idx]))
                        if (missing / total) * 100.0 > thresh:
                            targets.append(col)
                    dropped = _drop_columns(targets)
                    applied.append(
                        {"op": op, "threshold_pct": thresh, "columns_dropped": dropped, "columns": targets}
                    )
                elif op == "drop_constant_columns":
                    targets = []
                    for col, idx in list(col_index.items()):
                        values = {row[idx].strip() for row in body if not _is_tabular_missing(row[idx])}
                        if len(values) <= 1:
                            targets.append(col)
                    dropped = _drop_columns(targets)
                    applied.append({"op": op, "columns_dropped": dropped, "columns": targets})
                elif op == "impute_missing":
                    names = [str(c).strip() for c in (op_model.columns or []) if str(c).strip()]
                    if not names:
                        names = list(header)
                    strategy = str(op_model.strategy or "median").strip().lower()
                    filled_total = 0
                    per_col: dict[str, Any] = {}
                    for col in names:
                        idx = col_index.get(col)
                        if idx is None:
                            continue
                        present = [row[idx].strip() for row in body if not _is_tabular_missing(row[idx])]
                        numeric_vals: list[float] = []
                        for v in present:
                            try:
                                f = float(v)
                            except Exception:
                                continue
                            if math.isfinite(f):
                                numeric_vals.append(f)
                        is_numeric = len(numeric_vals) >= 1 and len(numeric_vals) >= 0.6 * max(1, len(present))
                        fill: str
                        if strategy == "zero":
                            fill = "0"
                        elif strategy == "constant":
                            fill = str(op_model.fill_value)
                        elif strategy in ("mean", "median") and is_numeric and numeric_vals:
                            arr = sorted(numeric_vals)
                            if strategy == "mean":
                                fill = f"{sum(arr) / len(arr):.6g}"
                            else:
                                mid = len(arr) // 2
                                fill = (
                                    f"{arr[mid]:.6g}"
                                    if len(arr) % 2
                                    else f"{(arr[mid - 1] + arr[mid]) / 2.0:.6g}"
                                )
                        else:
                            # mode (also the fallback for non-numeric mean/median)
                            counts = Counter(present)
                            fill = counts.most_common(1)[0][0] if counts else str(op_model.fill_value)
                        col_filled = 0
                        for row in body:
                            if _is_tabular_missing(row[idx]):
                                row[idx] = fill
                                col_filled += 1
                        if col_filled:
                            per_col[col] = {"filled": col_filled, "value": fill}
                            filled_total += col_filled
                    applied.append(
                        {"op": op, "strategy": strategy, "cells_filled": filled_total, "columns": per_col}
                    )
                elif op == "rename_columns":
                    mapping = {
                        str(k).strip(): str(v).strip()
                        for k, v in (op_model.rename or {}).items()
                        if str(k).strip() and str(v).strip()
                    }
                    renamed: dict[str, str] = {}
                    for old, new in mapping.items():
                        idx = col_index.get(old)
                        if idx is None or new == old:
                            continue
                        header[idx] = new
                        renamed[old] = new
                    if renamed:
                        col_index.clear()
                        col_index.update({name: i for i, name in enumerate(header)})
                    applied.append({"op": op, "renamed": renamed})
                elif op == "coerce_numeric":
                    names = [str(c).strip() for c in (op_model.columns or []) if str(c).strip()] or list(header)
                    coerced: dict[str, int] = {}
                    for col in names:
                        idx = col_index.get(col)
                        if idx is None:
                            continue
                        changed_cells = 0
                        for row in body:
                            val = row[idx].strip()
                            if _is_tabular_missing(val):
                                continue
                            cleaned = val.replace(",", "").replace("$", "").replace("%", "").strip()
                            try:
                                num = float(cleaned)
                            except Exception:
                                continue
                            new_val = f"{num:.6g}"
                            if new_val != row[idx]:
                                row[idx] = new_val
                                changed_cells += 1
                        if changed_cells:
                            coerced[col] = changed_cells
                    applied.append({"op": op, "columns": coerced})
                elif op == "normalize":
                    names = [str(c).strip() for c in (op_model.columns or []) if str(c).strip()] or list(header)
                    method = str(op_model.method or "minmax").strip().lower()
                    scaled: dict[str, Any] = {}
                    for col in names:
                        idx = col_index.get(col)
                        if idx is None:
                            continue
                        nums: list[tuple[int, float]] = []
                        for r_idx, row in enumerate(body):
                            val = row[idx].strip()
                            if _is_tabular_missing(val):
                                continue
                            try:
                                f = float(val)
                            except Exception:
                                nums = []
                                break
                            if not math.isfinite(f):
                                continue
                            nums.append((r_idx, f))
                        if not nums:
                            continue
                        values = [f for _, f in nums]
                        if method == "zscore":
                            mean = sum(values) / len(values)
                            var = sum((v - mean) ** 2 for v in values) / len(values)
                            std = math.sqrt(var)
                            if std <= 0:
                                continue
                            for r_idx, f in nums:
                                body[r_idx][idx] = f"{(f - mean) / std:.6g}"
                            scaled[col] = {"method": "zscore", "mean": round(mean, 6), "std": round(std, 6)}
                        else:
                            lo, hi = min(values), max(values)
                            if hi <= lo:
                                continue
                            for r_idx, f in nums:
                                body[r_idx][idx] = f"{(f - lo) / (hi - lo):.6g}"
                            scaled[col] = {"method": "minmax", "min": round(lo, 6), "max": round(hi, 6)}
                    applied.append({"op": op, "columns": scaled})
                elif op == "clip_outliers":
                    names = [str(c).strip() for c in (op_model.columns or []) if str(c).strip()] or list(header)
                    factor = float(op_model.factor) if op_model.factor else 1.5
                    clipped: dict[str, Any] = {}
                    for col in names:
                        idx = col_index.get(col)
                        if idx is None:
                            continue
                        nums = []
                        for r_idx, row in enumerate(body):
                            val = row[idx].strip()
                            if _is_tabular_missing(val):
                                continue
                            try:
                                f = float(val)
                            except Exception:
                                nums = []
                                break
                            nums.append((r_idx, f))
                        if len(nums) < 4:
                            continue
                        arr = sorted(f for _, f in nums)
                        n_local = len(arr)
                        q1 = arr[int(0.25 * (n_local - 1))]
                        q3 = arr[int(0.75 * (n_local - 1))]
                        iqr = q3 - q1
                        if iqr <= 0:
                            continue
                        lo, hi = q1 - factor * iqr, q3 + factor * iqr
                        count = 0
                        for r_idx, f in nums:
                            if f < lo:
                                body[r_idx][idx] = f"{lo:.6g}"
                                count += 1
                            elif f > hi:
                                body[r_idx][idx] = f"{hi:.6g}"
                                count += 1
                        if count:
                            clipped[col] = {"clipped": count, "lower": round(lo, 6), "upper": round(hi, 6)}
                    applied.append({"op": op, "factor": factor, "columns": clipped})
                elif op == "filter_rows":
                    col = str(op_model.where_col or "").strip()
                    idx = col_index.get(col)
                    if idx is None:
                        raise HTTPException(status_code=400, detail=f"filter_rows: column not found: {col}")
                    where_op = str(op_model.where_op or "==").strip()
                    target = str(op_model.where_value)

                    def _keep(cell: str) -> bool:
                        """Keep rows where `column where_op value` is true."""
                        c = cell.strip()
                        if where_op == "missing":
                            return _is_tabular_missing(c)
                        if where_op == "not_missing":
                            return not _is_tabular_missing(c)
                        if where_op == "contains":
                            return target in c
                        try:
                            a, b = float(c), float(target)
                            numeric = True
                        except Exception:
                            a = b = 0.0
                            numeric = False
                        if where_op == "==":
                            return (a == b) if numeric else (c == target)
                        if where_op == "!=":
                            return (a != b) if numeric else (c != target)
                        if not numeric:
                            return True
                        if where_op == ">":
                            return a > b
                        if where_op == ">=":
                            return a >= b
                        if where_op == "<":
                            return a < b
                        if where_op == "<=":
                            return a <= b
                        return True

                    kept = [row for row in body if _keep(row[idx])]
                    removed = len(body) - len(kept)
                    body[:] = kept
                    applied.append(
                        {"op": op, "column": col, "where_op": where_op, "value": target, "rows_removed": removed}
                    )
                elif op == "balance_classes":
                    label_col = str(op_model.label_col or "").strip()
                    idx = col_index.get(label_col)
                    if idx is None:
                        raise HTTPException(
                            status_code=400, detail=f"balance_classes: label_col not found: {label_col}"
                        )
                    method = str(op_model.strategy or "oversample").strip().lower()
                    max_ratio = float(op_model.max_ratio) if op_model.max_ratio else 1.0
                    max_ratio = max(1.0, max_ratio)
                    groups: dict[str, list[list[str]]] = {}
                    for row in body:
                        groups.setdefault(row[idx].strip(), []).append(row)
                    if len(groups) < 2:
                        applied.append({"op": op, "label_col": label_col, "note": "fewer than 2 classes; no change"})
                    else:
                        sizes = {k: len(v) for k, v in groups.items()}
                        majority = max(sizes.values())
                        minority = min(sizes.values())
                        rng = random.Random(42)
                        new_body: list[list[str]] = []
                        if method == "undersample":
                            # Cap every class at max_ratio * minority.
                            cap = int(round(minority * max_ratio))
                            for members in groups.values():
                                if len(members) > cap:
                                    new_body.extend(rng.sample(members, cap))
                                else:
                                    new_body.extend(members)
                        else:  # oversample
                            # Raise every class up to majority / max_ratio (>= its own size).
                            floor = int(math.ceil(majority / max_ratio))
                            for members in groups.values():
                                new_body.extend(members)
                                deficit = floor - len(members)
                                if deficit > 0 and members:
                                    new_body.extend(rng.choices(members, k=deficit))
                        rng.shuffle(new_body)
                        before_n = len(body)
                        body[:] = new_body
                        applied.append(
                            {
                                "op": op,
                                "label_col": label_col,
                                "method": method,
                                "before_counts": sizes,
                                "rows_before": before_n,
                                "rows_after": len(body),
                            }
                        )
                else:
                    raise HTTPException(status_code=400, detail=f"unknown tabular transform op: {op}")

            after = {"rows": len(body), "cols": len(header)}
            changed = bool(applied) and (after != before or any(applied))
            backup = _write_tabular_table(csv_path, header, body) if applied else csv_path
            try:
                backup_rel = str(Path(backup).relative_to(mlops_registry.REPO_ROOT))
            except Exception:
                backup_rel = str(backup)
            revision = 0
            if applied:
                revision = _append_tabular_history(
                    csv_path,
                    {
                        "action": "transform",
                        "ops": applied,
                        "before": before,
                        "after": after,
                        "backup": backup_rel,
                    },
                )
            return {
                "slug": slug,
                "ops_applied": applied,
                "before": before,
                "after": after,
                "changed": changed,
                "backup": backup_rel if applied else "",
                "revision": revision,
            }

        @self.app.post("/database/{slug}/tabular_split")
        async def database_tabular_split(slug: str, req: TabularSplitRequest) -> dict[str, Any]:
            """Write reproducible, optionally stratified train/val/test split assignments.

            Produces a sibling <slug>.splits.json mapping split -> 0-based row indices so
            any algo cell can read a shared, seeded split instead of re-rolling its own.
            Optionally appends a `split` column to the CSV.
            """
            import json as _json
            import random as _random

            val_frac = min(max(float(req.val_frac), 0.0), 0.9)
            test_frac = min(max(float(req.test_frac), 0.0), 0.9)
            if val_frac + test_frac >= 1.0:
                raise HTTPException(status_code=400, detail="val_frac + test_frac must be < 1.0")

            csv_path = _resolve_tabular_csv(slug, req.name)
            header, body = _read_tabular_table(csv_path, hard_cap=_TABULAR_MAX_ROWS)
            n = len(body)
            if n < 3:
                raise HTTPException(status_code=400, detail="need at least 3 rows to split")

            stratify_col = str(req.stratify_col or "").strip()
            rng = _random.Random(int(req.seed))

            def _partition(indices: list[int]) -> tuple[list[int], list[int], list[int]]:
                shuffled = list(indices)
                rng.shuffle(shuffled)
                n_local = len(shuffled)
                n_test = int(round(n_local * test_frac))
                n_val = int(round(n_local * val_frac))
                # Guarantee train keeps at least one row when fractions are aggressive.
                n_val = min(n_val, max(0, n_local - n_test - 1))
                test_idx = shuffled[:n_test]
                val_idx = shuffled[n_test : n_test + n_val]
                train_idx = shuffled[n_test + n_val :]
                return train_idx, val_idx, test_idx

            stratified = False
            train_set: list[int] = []
            val_set: list[int] = []
            test_set: list[int] = []
            if stratify_col:
                if stratify_col not in header:
                    raise HTTPException(status_code=400, detail=f"stratify_col not found: {stratify_col}")
                cidx = header.index(stratify_col)
                groups: dict[str, list[int]] = {}
                for i, row in enumerate(body):
                    groups.setdefault(row[cidx].strip(), []).append(i)
                # Stratify only when every class is large enough to span the splits;
                # otherwise fall back to a plain random partition.
                min_needed = 1 + (1 if val_frac > 0 else 0) + (1 if test_frac > 0 else 0)
                if all(len(members) >= min_needed for members in groups.values()) and len(groups) > 1:
                    stratified = True
                    for members in groups.values():
                        tr, va, te = _partition(members)
                        train_set.extend(tr)
                        val_set.extend(va)
                        test_set.extend(te)
                else:
                    train_set, val_set, test_set = _partition(list(range(n)))
            else:
                train_set, val_set, test_set = _partition(list(range(n)))

            train_set.sort()
            val_set.sort()
            test_set.sort()
            counts = {"train": len(train_set), "val": len(val_set), "test": len(test_set)}

            splits_path = csv_path.with_name(f"{csv_path.stem}.splits.json")
            payload = {
                "version": 1,
                "slug": slug,
                "csv": csv_path.name,
                "seed": int(req.seed),
                "val_frac": val_frac,
                "test_frac": test_frac,
                "stratify_col": stratify_col,
                "stratified": stratified,
                "counts": counts,
                "splits": {"train": train_set, "val": val_set, "test": test_set},
            }
            try:
                tmp = splits_path.with_name(f"{splits_path.name}.tmp.{os.getpid()}")
                tmp.write_text(_json.dumps(payload, indent=2), encoding="utf-8")
                tmp.replace(splits_path)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to write splits: {exc}") from exc

            wrote_column = False
            if bool(req.write_column):
                assignment = {i: "train" for i in train_set}
                assignment.update({i: "val" for i in val_set})
                assignment.update({i: "test" for i in test_set})
                if "split" in header:
                    sidx = header.index("split")
                    for i, row in enumerate(body):
                        row[sidx] = assignment.get(i, "train")
                else:
                    header.append("split")
                    for i, row in enumerate(body):
                        row.append(assignment.get(i, "train"))
                _write_tabular_table(csv_path, header, body)
                wrote_column = True

            try:
                splits_rel = str(splits_path.relative_to(mlops_registry.REPO_ROOT))
            except Exception:
                splits_rel = str(splits_path)
            return {
                "slug": slug,
                "seed": int(req.seed),
                "counts": counts,
                "stratified": stratified,
                "wrote_column": wrote_column,
                "splits_path": splits_rel,
            }

        @self.app.get("/database/{slug}/tabular_history")
        async def database_tabular_history(slug: str, name: str = Query("", alias="name")) -> dict[str, Any]:
            """Read the provenance log of cleaning transforms applied to a tabular dataset."""
            csv_path = _resolve_tabular_csv(slug, name)
            log = _read_tabular_history(csv_path)
            entries = log.get("entries") if isinstance(log.get("entries"), list) else []
            backup = csv_path.with_suffix(csv_path.suffix + ".bak")
            return {
                "slug": slug,
                "csv": csv_path.name,
                "count": len(entries),
                "entries": entries,
                "can_undo": backup.exists(),
            }

        @self.app.post("/database/{slug}/tabular_undo")
        async def database_tabular_undo(slug: str, req: TabularTransformRequest) -> dict[str, Any]:
            """Restore the single most recent .bak (undo the last transform) and log it.

            Reuses TabularTransformRequest only for its optional `name` field (csv selector).
            """
            csv_path = _resolve_tabular_csv(slug, req.name)
            backup = csv_path.with_suffix(csv_path.suffix + ".bak")
            if not backup.exists():
                raise HTTPException(status_code=400, detail="no backup available to undo")
            before = _tabular_dims(csv_path)
            try:
                tmp = csv_path.with_name(f"{csv_path.name}.tmp.{os.getpid()}")
                shutil.copy2(backup, tmp)
                tmp.replace(csv_path)
                backup.unlink(missing_ok=True)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"failed to restore backup: {exc}") from exc
            after = _tabular_dims(csv_path)
            revision = _append_tabular_history(
                csv_path, {"action": "undo", "before": before, "after": after}
            )
            return {"slug": slug, "restored": True, "before": before, "after": after, "revision": revision}

        @self.app.get("/database/{slug}/tabular_target")
        async def database_tabular_target(
            slug: str,
            label_col: str = Query("", alias="label_col"),
            feature_cols: str = Query("", alias="feature_cols"),
            name: str = Query("", alias="name"),
            max_rows: int = Query(20000, alias="max_rows", ge=200, le=200000),
        ) -> dict[str, Any]:
            """Analyze a tabular dataset's target column: task type, class balance,
            leakage flags, and a train-readiness gate (blockers + warnings)."""
            csv_path = _resolve_tabular_csv(slug, name)
            header, body = _read_tabular_table(csv_path, max_rows=int(max_rows))
            if not header:
                raise HTTPException(status_code=400, detail="csv has no header row")
            n_rows = len(body)

            blockers: list[str] = []
            warnings: list[str] = []
            label = str(label_col or "").strip()
            if not label:
                blockers.append("no label column selected")
            elif label not in header:
                blockers.append(f"label column not found: {label}")

            # Resolve feature columns (default: all non-label, matching torch_tabular).
            requested_feats = [c.strip() for c in str(feature_cols or "").split(",") if c.strip()]
            if requested_feats:
                feats = [c for c in requested_feats if c in header and c != label]
            else:
                feats = [c for c in header if c != label]

            def _col_values(col: str) -> list[str]:
                idx = header.index(col)
                return [row[idx].strip() for row in body]

            task = "unknown"
            classes_out: list[dict[str, Any]] = []
            class_balance: dict[str, Any] = {}
            target_missing_pct = 0.0
            leakage: list[dict[str, Any]] = []

            if label and label in header:
                label_vals = _col_values(label)
                non_missing = [v for v in label_vals if not _is_tabular_missing(v)]
                missing_n = n_rows - len(non_missing)
                target_missing_pct = round((missing_n / float(max(1, n_rows))) * 100.0, 2)
                if not non_missing:
                    blockers.append(f"label column '{label}' is entirely missing")
                else:
                    # Task type: numeric+high-cardinality -> regression, else classification.
                    numeric_vals: list[float] = []
                    for v in non_missing:
                        try:
                            f = float(v)
                        except Exception:
                            continue
                        if math.isfinite(f):
                            numeric_vals.append(f)
                    numeric_frac = len(numeric_vals) / float(len(non_missing))
                    n_unique = len(set(non_missing))
                    is_numeric = numeric_frac >= 0.95
                    if is_numeric and n_unique > max(20, int(0.2 * len(non_missing))):
                        task = "regression"
                    else:
                        task = "classification"

                    if task == "classification":
                        counts = Counter(non_missing)
                        den = float(max(1, len(non_missing)))
                        classes_out = [
                            {"value": val, "count": int(c), "pct": round((c / den) * 100.0, 2)}
                            for val, c in counts.most_common()
                        ]
                        sizes = [c for _, c in counts.items()]
                        majority, minority = (max(sizes), min(sizes)) if sizes else (0, 0)
                        ratio = (majority / float(minority)) if minority else float("inf")
                        imbalanced = bool(minority and ratio > 10.0)
                        class_balance = {
                            "n_classes": len(counts),
                            "majority": majority,
                            "minority": minority,
                            "imbalance_ratio": round(ratio, 2) if minority else None,
                            "imbalanced": imbalanced,
                        }
                        if len(counts) < 2:
                            blockers.append(f"label '{label}' has only one class")
                        if imbalanced:
                            warnings.append(
                                f"class imbalance {ratio:.1f}:1 (consider balance_classes)"
                            )
                        # Tiny minority class breaks stratified splits.
                        if minority and minority < 2:
                            warnings.append("a class has fewer than 2 rows (stratified split will degrade)")

                    # Leakage: a feature that near-perfectly predicts the target.
                    label_idx = header.index(label)
                    for col in feats[:60]:
                        cidx = header.index(col)
                        pairs = [
                            (row[cidx].strip(), row[label_idx].strip())
                            for row in body
                            if not _is_tabular_missing(row[cidx].strip())
                            and not _is_tabular_missing(row[label_idx].strip())
                        ]
                        if len(pairs) < 10:
                            continue
                        if task == "regression":
                            xs, ys = [], []
                            ok = True
                            for fv, tv in pairs:
                                try:
                                    xs.append(float(fv))
                                    ys.append(float(tv))
                                except Exception:
                                    ok = False
                                    break
                            if not ok or len(xs) < 10:
                                continue
                            xa = np.asarray(xs)
                            ya = np.asarray(ys)
                            if float(np.std(xa)) <= 0 or float(np.std(ya)) <= 0:
                                continue
                            corr = float(np.corrcoef(xa, ya)[0, 1])
                            if math.isfinite(corr) and abs(corr) >= 0.98:
                                leakage.append({"feature": col, "metric": "corr", "value": round(corr, 4)})
                        else:
                            # Purity: fraction of rows whose target is determined by the feature value.
                            by_val: dict[str, Counter] = {}
                            for fv, tv in pairs:
                                by_val.setdefault(fv, Counter())[tv] += 1
                            if len(by_val) <= 1:
                                continue
                            pure = sum(max(c.values()) for c in by_val.values())
                            purity = pure / float(len(pairs))
                            if purity >= 0.999:
                                leakage.append({"feature": col, "metric": "purity", "value": round(purity, 4)})
                    if leakage:
                        names = ", ".join(str(item["feature"]) for item in leakage[:5])
                        warnings.append(f"possible target leakage from: {names}")
                    if target_missing_pct > 5.0:
                        warnings.append(f"label has {target_missing_pct:.1f}% missing values")

            # Feature-side readiness checks.
            if not feats:
                blockers.append("no feature columns available (all columns are the label)")
            else:
                all_missing_feats = []
                constant_feats = []
                for col in feats:
                    vals = _col_values(col)
                    present = [v for v in vals if not _is_tabular_missing(v)]
                    if not present:
                        all_missing_feats.append(col)
                    elif len(set(present)) <= 1:
                        constant_feats.append(col)
                if all_missing_feats:
                    warnings.append(f"{len(all_missing_feats)} feature column(s) entirely missing")
                if constant_feats:
                    warnings.append(f"{len(constant_feats)} constant feature column(s) (consider drop_constant_columns)")

            if n_rows < 10:
                blockers.append(f"only {n_rows} rows (need >= 10 to train)")

            ready = not blockers
            return {
                "slug": slug,
                "label_col": label,
                "task": task,
                "n_rows": n_rows,
                "n_features_selected": len(feats),
                "feature_cols": feats[:200],
                "classes": classes_out[:50],
                "class_balance": class_balance,
                "target_missing_pct": target_missing_pct,
                "leakage": leakage[:20],
                "readiness": {"ready": ready, "blockers": blockers, "warnings": warnings},
            }

        @self.app.post("/database/{slug}/tabular_score")
        async def database_tabular_score(slug: str, req: TabularScoreRequest) -> dict[str, Any]:
            """Batch-score a tabular dataset against a trained tabular model.

            Resolve the model from an explicit model_path, or from a scenario (+ optional
            version: "", "candidate", "prod", or a run version). Optionally write the
            predictions as a new tabular dataset.
            """
            csv_path = _resolve_tabular_csv(slug, req.name)

            model_path: Optional[Path] = None
            source = ""
            explicit = str(req.model_path or "").strip()
            if explicit:
                candidate = Path(explicit).expanduser()
                if not candidate.is_absolute():
                    candidate = (mlops_registry.REPO_ROOT / candidate).resolve()
                model_path = candidate
                source = "model_path"
            elif str(req.scenario or "").strip():
                try:
                    target = mlops_registry.resolve_inference_target(
                        str(req.scenario).strip(), str(req.version or "").strip()
                    )
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"failed to resolve model: {exc}") from exc
                if not target or not target.get("weights_path"):
                    raise HTTPException(
                        status_code=404,
                        detail=f"no trained model found for scenario '{req.scenario}'",
                    )
                model_path = Path(str(target.get("weights_path"))).resolve()
                source = str(target.get("source") or "scenario")
            else:
                raise HTTPException(status_code=400, detail="provide either scenario or model_path")

            if model_path is None or not model_path.exists():
                raise HTTPException(status_code=404, detail=f"model artifact not found: {model_path}")

            try:
                result = _score_tabular_with_artifact(model_path, csv_path)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"scoring failed: {exc}") from exc

            result["input_slug"] = slug
            result["model_source"] = source

            written_slug = ""
            if bool(req.write_dataset):
                header, body = _read_tabular_table(csv_path, hard_cap=_TABULAR_MAX_ROWS)
                preds = result.get("predictions") or []
                col = "prediction"
                if col in header:
                    pidx = header.index(col)
                    for i, row in enumerate(body):
                        row[pidx] = str(preds[i]) if i < len(preds) else ""
                else:
                    header.append(col)
                    for i, row in enumerate(body):
                        row.append(str(preds[i]) if i < len(preds) else "")
                out = io.StringIO()
                writer = csv.writer(out)
                writer.writerow(header)
                writer.writerows(body)
                stem = str(req.output_name or "").strip() or f"{slug}-scored"
                stored = _store_tabular_csv_bytes(out.getvalue().encode("utf-8"), stem)
                written_slug = str(stored.get("slug") or "")
                result["written_dataset"] = stored
            result["written_slug"] = written_slug
            # Trim the inline payload; full predictions live in the written dataset.
            result.pop("predictions", None)
            return result

        @self.app.get("/audio/assets")
        async def audio_assets() -> dict[str, Any]:
            try:
                root = mlops_registry.ensure_ml_audio_root().resolve()
                items = self._list_audio_asset_files()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"audio asset list failed: {exc}") from exc
            return {
                "root": str(root),
                "count": len(items),
                "items": items,
                "supported_suffixes": sorted(_AUDIO_SOURCE_SUFFIXES),
            }

        @self.app.post("/audio/analyze")
        async def audio_analyze(req: AudioAnalyzeRequest) -> dict[str, Any]:
            try:
                path = self._resolve_audio_source_path(req.path)
                if path.suffix.lower() == ".wav":
                    metrics = mlops_audio_ops.analyze_wav(path, max_seconds=30.0)
                    analysis_path = path
                    source_kind = "wav"
                else:
                    analysis_path = self._analysis_wav_path(path)
                    # Keep quick source checks responsive: analyze the selected
                    # range, or the first 30 seconds when the operator has not
                    # specified a clip boundary.
                    start_ms = int(req.start_ms or 0)
                    end_ms = req.end_ms
                    if end_ms is None or int(end_ms) <= start_ms:
                        end_ms = start_ms + 30000
                    result = mlops_audio_ops.extract_clip_to_wav(
                        path,
                        analysis_path,
                        start_ms=start_ms,
                        end_ms=end_ms,
                        clean=False,
                    )
                    metrics = result.get("after") if isinstance(result, dict) else {}
                    source_kind = "decoded_media"
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"audio analysis failed: {exc}") from exc
            return {
                "audio_path": str(path),
                "analysis_path": str(analysis_path),
                "source_kind": source_kind,
                "metrics": metrics,
            }

        @self.app.post("/audio/clean")
        async def audio_clean(req: AudioCleanRequest) -> dict[str, Any]:
            try:
                path = self._resolve_audio_path(req.path)
                output_path = self._audio_clean_output_path(path, req.output_name)
                result = mlops_audio_ops.clean_wav(
                    path,
                    output_path,
                    noise_reduce=bool(req.noise_reduce),
                    trim_silence=bool(req.trim_silence),
                    normalize=bool(req.normalize),
                    noise_reduction_strength=float(req.noise_reduction_strength),
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"audio cleanup failed: {exc}") from exc
            return {
                "audio_path": str(path),
                "cleaned_path": str(output_path),
                "result": result,
            }

        @self.app.post("/audio/datasets")
        async def audio_dataset_create(req: AudioDatasetCreateRequest) -> dict[str, Any]:
            try:
                slug = mlops_registry.pick_unique_audio_dataset_slug(req.name)
                root = (mlops_registry.ensure_ml_audio_root() / slug).resolve()
                root.mkdir(parents=True, exist_ok=False)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"audio dataset creation failed: {exc}") from exc
            return {
                "slug": slug,
                "path": str(root),
                "category": mlops_registry.DATASET_CATEGORY_AUDIO,
                "format": mlops_registry.LIBRARY_DATASET_FORMAT_AUDIOFOLDER,
            }

        @self.app.post("/audio/collect_clip")
        async def audio_collect_clip(req: AudioCollectClipRequest) -> dict[str, Any]:
            try:
                dataset_slug = mlops_registry.sanitize_library_dataset_slug(req.dataset)
                dataset_root = mlops_registry.ensure_ml_audio_root() / dataset_slug
                dataset_root.mkdir(parents=True, exist_ok=True)
                source_path = self._resolve_media_source_path(req.source_path)
                output_path = mlops_audio_ops.build_audio_dataset_clip_path(
                    dataset_root,
                    split=req.split,
                    label=req.label,
                    source_path=source_path,
                    start_ms=int(req.start_ms or 0),
                    end_ms=req.end_ms,
                )
                result = mlops_audio_ops.extract_clip_to_wav(
                    source_path,
                    output_path,
                    start_ms=int(req.start_ms or 0),
                    end_ms=req.end_ms,
                    clean=bool(req.clean),
                    noise_reduce=bool(req.noise_reduce),
                    trim_silence=bool(req.trim_silence),
                    normalize=bool(req.normalize),
                    noise_reduction_strength=float(req.noise_reduction_strength),
                )
                payload = mlops_registry.inspect_library_dataset_at(dataset_root)
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"audio clip collection failed: {exc}") from exc
            for scen in mlops_registry.scenario_names_for_dataset_folder(dataset_slug):
                self._emit_scenario_updated(scen)
            return {
                "dataset": dataset_slug,
                "dataset_path": str(dataset_root.resolve()),
                "clip_path": str(output_path.resolve()),
                "label": str(req.label or "").strip(),
                "split": "val" if str(req.split or "").lower() in {"val", "valid", "validation", "test"} else "train",
                "result": result,
                "dataset_summary": payload,
            }

        @self.app.post("/audio/copy_clip")
        async def audio_copy_clip(req: AudioCopyClipRequest) -> dict[str, Any]:
            """Extract a time region from a source audio/video file to an arbitrary WAV path."""
            try:
                source_path = self._resolve_media_source_path(req.source_path)
                dest_path = Path(req.dest_path).resolve()
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                result = mlops_audio_ops.extract_clip_to_wav(
                    source_path,
                    dest_path,
                    start_ms=int(req.start_ms or 0),
                    end_ms=req.end_ms,
                    clean=False,
                    noise_reduce=False,
                    trim_silence=False,
                    normalize=False,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"audio copy_clip failed: {exc}") from exc
            return {
                "clip_path": str(dest_path),
                "result": result,
            }

        @self.app.get("/database/{slug}/inventory")
        async def database_inventory(
            slug: str,
            rel: str = Query("", alias="rel"),
            include_hidden: int = Query(0, alias="include_hidden"),
            max_files: Optional[int] = Query(None, alias="max_files"),
        ) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                payload = mlops_registry.inventory_folder_types_at(
                    dataset_root,
                    relative_dir=str(rel or ""),
                    include_hidden=bool(int(include_hidden or 0)),
                    max_files=int(max_files) if max_files is not None else None,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return {"slug": slug, **payload}

        @self.app.post("/database/{slug}/inventory/move_by_ext")
        async def database_inventory_move_by_ext(slug: str, req: InventoryMoveByExtRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                payload = mlops_registry.move_files_by_extension_at(
                    dataset_root,
                    ext=req.ext,
                    dest_relative_dir=req.dest_relative_dir,
                    relative_dir=req.relative_dir,
                    include_hidden=bool(req.include_hidden),
                    preserve_tree=bool(req.preserve_tree),
                    dry_run=bool(req.dry_run),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            for scen in mlops_registry.scenario_names_for_dataset_folder(slug):
                self._emit_scenario_updated(scen)
            return {"slug": slug, **payload}

        @self.app.post("/database/{slug}/inventory/delete_by_ext")
        async def database_inventory_delete_by_ext(slug: str, req: InventoryDeleteByExtRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                payload = mlops_registry.delete_files_by_extension_at(
                    dataset_root,
                    ext=req.ext,
                    relative_dir=req.relative_dir,
                    include_hidden=bool(req.include_hidden),
                    dry_run=bool(req.dry_run),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            for scen in mlops_registry.scenario_names_for_dataset_folder(slug):
                self._emit_scenario_updated(scen)
            return {"slug": slug, **payload}

        @self.app.get("/database/{slug}/classes")
        async def database_get_classes(slug: str) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            classes_path = dataset_root.resolve() / "classes.txt"
            classes: list[str] = []
            if classes_path.exists():
                try:
                    classes = [
                        ln.strip()
                        for ln in classes_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if ln.strip()
                    ]
                except Exception:
                    classes = []
            return {"slug": slug, "classes": classes, "path": str(classes_path)}

        @self.app.put("/database/{slug}/classes")
        async def database_write_classes(slug: str, req: ClassesWriteRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            raw = req.classes if isinstance(req.classes, list) else []
            cleaned: list[str] = []
            seen: set[str] = set()
            for item in raw:
                name = str(item or "").strip()
                if not name:
                    continue
                # Keep file one-class-per-line; normalize embedded newlines.
                name = name.replace("\r", " ").replace("\n", " ").strip()
                if not name:
                    continue
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(name)
                if len(cleaned) >= 500:
                    break

            classes_path = dataset_root.resolve() / "classes.txt"
            try:
                classes_path.write_text(
                    "\n".join(cleaned) + ("\n" if cleaned else ""),
                    encoding="utf-8",
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            return {
                "slug": slug,
                "saved": str(classes_path),
                "count": len(cleaned),
                "classes": cleaned,
            }

        @self.app.get("/database/{slug}/thumb/{name:path}")
        async def database_thumb(slug: str, name: str) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                match = mlops_registry.resolve_dataset_image_path_at(dataset_root, name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if match is None:
                raise HTTPException(status_code=404, detail="image not found")
            image = cv2.imread(str(match))
            if image is None:
                raise HTTPException(status_code=400, detail="unable to decode")
            h, w = image.shape[:2]
            max_side = 160
            scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
            if scale < 1.0:
                image = cv2.resize(image, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if not ok:
                raise HTTPException(status_code=500, detail="encode failed")
            return {"name": name, "thumb_b64": base64.b64encode(buf.tobytes()).decode("ascii")}

        @self.app.get("/database/{slug}/image/{name:path}")
        async def database_image(slug: str, name: str, max_side: int = Query(1280)) -> dict[str, Any]:
            """Return a (possibly downscaled) JPEG for annotation/editing."""
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                match = mlops_registry.resolve_dataset_image_path_at(dataset_root, name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if match is None:
                raise HTTPException(status_code=404, detail="image not found")
            image = cv2.imread(str(match))
            if image is None:
                raise HTTPException(status_code=400, detail="unable to decode")
            h, w = image.shape[:2]
            max_side = int(max_side or 0)
            if max_side > 0:
                scale = max_side / float(max(h, w)) if max(h, w) > max_side else 1.0
                if scale < 1.0:
                    image = cv2.resize(image, (int(w * scale), int(h * scale)))
            oh, ow = image.shape[:2]
            ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            if not ok:
                raise HTTPException(status_code=500, detail="encode failed")
            return {
                "name": name,
                "width": int(ow),
                "height": int(oh),
                "image_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
            }

        @self.app.get("/database/{slug}/label/{name:path}")
        async def database_label_text(slug: str, name: str) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                match = mlops_registry.resolve_dataset_image_path_at(dataset_root, name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if match is None:
                raise HTTPException(status_code=404, detail="image not found")
            fmt = mlops_registry.detect_library_dataset_format(dataset_root)
            if fmt == mlops_registry.LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
                rel = Path(name)
                parts = rel.parts
                class_name = ""
                if parts and parts[0] in {"train", "valid", "val", "test"} and len(parts) >= 2:
                    class_name = str(parts[1])
                elif len(parts) >= 1:
                    class_name = str(parts[0])
                text = f"class: {class_name}\n" if class_name else ""
                return {
                    "relative_path": name,
                    "has_label": bool(class_name),
                    "text": text,
                    "line_count": 1 if class_name else 0,
                }
            label_path = mlops_registry.resolve_dataset_label_path(match)
            if label_path is None or not label_path.exists():
                return {
                    "relative_path": name,
                    "has_label": False,
                    "text": "",
                    "line_count": 0,
                }
            try:
                text = label_path.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            nonempty_lines = [ln for ln in text.splitlines() if ln.strip()]
            return {
                "relative_path": name,
                "has_label": True,
                "text": text,
                "line_count": len(nonempty_lines),
            }

        @self.app.put("/database/{slug}/label/{name:path}")
        async def database_write_label(slug: str, name: str, req: LabelWriteRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
                match = mlops_registry.resolve_dataset_image_path_at(dataset_root, name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if match is None:
                raise HTTPException(status_code=404, detail="image not found")
            label_path = mlops_registry.resolve_dataset_label_path(match)
            if label_path is None:
                raise HTTPException(status_code=500, detail="could not resolve label path")
            try:
                label_path.parent.mkdir(parents=True, exist_ok=True)
                text = str(req.text or "")
                label_path.write_text(text, encoding="utf-8")
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            nonempty_lines = [ln for ln in text.splitlines() if ln.strip()]
            return {
                "relative_path": name,
                "saved": str(label_path),
                "has_label": True,
                "line_count": len(nonempty_lines),
            }

        @self.app.post("/database/{slug}/labels/bulk_apply")
        async def database_bulk_apply_labels(slug: str, req: BulkLabelApplyRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            # Validate request.
            try:
                class_id = int(req.class_id)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="invalid class_id") from exc

            geom = str(req.geometry or "").strip().lower()
            if geom not in {"full_image", "center"}:
                raise HTTPException(status_code=400, detail="geometry must be full_image or center")
            cw = float(req.center_w or 0.0)
            ch = float(req.center_h or 0.0)
            if cw <= 0.0 or cw > 1.0 or ch <= 0.0 or ch > 1.0:
                raise HTTPException(status_code=400, detail="center_w/center_h must be in (0, 1]")

            scope = str(req.scope or "all").strip().lower()
            if scope not in {"all", "split", "class_folder"}:
                raise HTTPException(status_code=400, detail="scope must be all, split, or class_folder")
            split = str(req.split or "").strip()
            if scope == "split" and not split:
                raise HTTPException(status_code=400, detail="split is required when scope=split")
            split_lower = split.lower()

            class_folder = str(req.class_folder_name or "").strip()
            if scope == "class_folder" and not class_folder:
                raise HTTPException(status_code=400, detail="class_folder_name is required when scope=class_folder")
            class_folder_lower = class_folder.lower()

            only_missing = bool(req.only_missing)
            replace = bool(req.replace)
            limit = int(req.limit) if req.limit is not None else None

            # Precompute label line in normalized coordinates (no image decode required).
            if geom == "full_image":
                line = f"{class_id} 0.5 0.5 1.0 1.0\n"
            else:
                line = f"{class_id} 0.5 0.5 {cw:.6f} {ch:.6f}\n"

            entries = mlops_registry.list_dataset_entries_at(dataset_root)
            applied = 0
            skipped = 0
            errors: list[str] = []

            for entry in entries:
                if limit is not None and applied >= limit:
                    break
                try:
                    rel_path = str(entry.get("relative_path") or "")
                    if not rel_path:
                        continue
                    if scope == "split" and str(entry.get("split") or "").lower() != split_lower:
                        continue
                    if scope == "class_folder":
                        parts = Path(rel_path).parts
                        # Prefer the class folder immediately under images/<split>/ if present.
                        folder = ""
                        try:
                            images_idx = parts.index("images")
                        except ValueError:
                            images_idx = -1
                        if images_idx >= 0 and len(parts) > images_idx + 2:
                            folder = str(parts[images_idx + 2])
                        if folder and folder.lower() != class_folder_lower:
                            continue
                        if not folder:
                            if not any(str(p).lower() == class_folder_lower for p in parts):
                                continue

                    # Skip if label already exists when requested.
                    if only_missing and bool(entry.get("has_label")):
                        skipped += 1
                        continue

                    match = mlops_registry.resolve_dataset_image_path_at(dataset_root, rel_path)
                    if match is None:
                        skipped += 1
                        continue
                    label_path = mlops_registry.resolve_dataset_label_path(match)
                    if label_path is None:
                        skipped += 1
                        continue
                    label_path.parent.mkdir(parents=True, exist_ok=True)

                    if replace:
                        label_path.write_text(line, encoding="utf-8")
                    else:
                        prev = ""
                        if label_path.exists():
                            try:
                                prev = label_path.read_text(encoding="utf-8", errors="replace")
                            except Exception:
                                prev = ""
                        body = prev
                        if body and not body.endswith("\n"):
                            body += "\n"
                        body += line
                        label_path.write_text(body, encoding="utf-8")

                    applied += 1
                except Exception as exc:
                    errors.append(str(exc))
                    if len(errors) >= 8:
                        break

            return {
                "slug": slug,
                "geometry": geom,
                "class_id": class_id,
                "scope": scope,
                "split": split if split else "",
                "class_folder_name": class_folder,
                "only_missing": only_missing,
                "replace": replace,
                "applied": applied,
                "skipped": skipped,
                "error_count": len(errors),
                "errors": errors,
            }

        @self.app.post("/database/{slug}/move_to_split")
        async def database_move_to_split(slug: str, req: MoveToSplitRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
                target_split = self._normalize_dataset_split(req.target_split)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            rels = [str(p or "").strip().lstrip("/") for p in (req.relative_paths or []) if str(p or "").strip()]
            moved = 0
            errors: list[str] = []

            for rel in rels:
                try:
                    img_path = mlops_registry.resolve_dataset_image_path_at(dataset_root, rel)
                    if img_path is None:
                        raise ValueError("image not found")
                    label_path = mlops_registry.resolve_dataset_label_path(img_path)

                    images_dir: Optional[Path] = None
                    for parent in img_path.parents:
                        if parent.name == "images":
                            images_dir = parent
                            break
                    if images_dir is None:
                        raise ValueError("image is not under images/")
                    labels_dir = images_dir.parent / "labels"

                    rel_under_images = img_path.relative_to(images_dir)
                    rel_parts = list(rel_under_images.parts)
                    if rel_parts and rel_parts[0].lower() in {"train", "val", "test", "valid"}:
                        rel_parts = rel_parts[1:] if len(rel_parts) > 1 else [img_path.name]
                    rel_after = Path(*rel_parts)

                    dest_img = (images_dir / target_split / rel_after).resolve()
                    dest_img.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(img_path), str(dest_img))

                    if label_path is not None and label_path.exists():
                        dest_label = (labels_dir / target_split / rel_after).with_suffix(".txt").resolve()
                        dest_label.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(label_path), str(dest_label))

                    moved += 1
                except Exception as exc:
                    errors.append(f"{rel}: {exc}")
                    if len(errors) >= 8:
                        break

            return {"slug": slug, "target_split": target_split, "moved": moved, "error_count": len(errors), "errors": errors}

        @self.app.post("/database/{slug}/copy_augmented_to_split")
        async def database_copy_augmented_to_split(slug: str, req: CopyAugmentToSplitRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
                target_split = self._normalize_dataset_split(req.target_split)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            rels = [str(p or "").strip().lstrip("/") for p in (req.relative_paths or []) if str(p or "").strip()]
            copies_per_image = max(1, min(50, int(req.copies_per_image or 1)))
            scale_pct = max(10, min(300, int(req.scale_pct or 100)))
            angle_deg = max(-180.0, min(180.0, float(req.angle_deg or 0.0)))
            jpeg_quality = max(1, min(100, int(req.jpeg_quality or 90)))
            grayscale = bool(req.grayscale)
            suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(req.suffix or "aug")).strip("_") or "aug"

            entries = mlops_registry.list_dataset_entries_at(dataset_root)
            split_counts: dict[str, int] = {}
            for entry in entries:
                split_name = str(entry.get("split") or "").strip().lower()
                if split_name:
                    split_counts[split_name] = split_counts.get(split_name, 0) + 1
            requested_total = len(rels) * copies_per_image
            if bool(req.balance_to_train):
                deficit = max(0, int(split_counts.get("train", 0)) - int(split_counts.get(target_split, 0)))
                requested_total = min(max(0, deficit), 5000)
            if not rels or requested_total <= 0:
                return {
                    "slug": slug,
                    "target_split": target_split,
                    "copied": 0,
                    "skipped": 0,
                    "error_count": 0,
                    "errors": [],
                    "split_counts": split_counts,
                }

            def _label_lines(label_path: Optional[Path]) -> list[str]:
                if label_path is None or not label_path.exists():
                    return []
                try:
                    return [
                        line.strip()
                        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines()
                        if line.strip() and not line.strip().startswith("#")
                    ]
                except Exception:
                    return []

            def _rotate_label_line(line: str, width: int, height: int, matrix: np.ndarray) -> str | None:
                parts = line.split()
                if len(parts) < 5:
                    return None
                try:
                    cls_id = int(float(parts[0]))
                    cx, cy, bw, bh = [float(v) for v in parts[1:5]]
                except Exception:
                    return None
                x_c = cx * width
                y_c = cy * height
                half_w = bw * width / 2.0
                half_h = bh * height / 2.0
                corners = np.array(
                    [
                        [x_c - half_w, y_c - half_h, 1.0],
                        [x_c + half_w, y_c - half_h, 1.0],
                        [x_c + half_w, y_c + half_h, 1.0],
                        [x_c - half_w, y_c + half_h, 1.0],
                    ],
                    dtype=np.float32,
                )
                rotated = corners @ matrix.T
                xs = np.clip(rotated[:, 0], 0.0, float(width))
                ys = np.clip(rotated[:, 1], 0.0, float(height))
                x1, x2 = float(xs.min()), float(xs.max())
                y1, y2 = float(ys.min()), float(ys.max())
                new_w = x2 - x1
                new_h = y2 - y1
                if new_w <= 1.0 or new_h <= 1.0:
                    return None
                return (
                    f"{cls_id} {(x1 + x2) / 2.0 / width:.6f} {(y1 + y2) / 2.0 / height:.6f} "
                    f"{new_w / width:.6f} {new_h / height:.6f}"
                )

            def _augmented_labels(lines: list[str], width: int, height: int, matrix: np.ndarray | None) -> str:
                out: list[str] = []
                for line in lines:
                    if matrix is None:
                        parts = line.split()
                        if len(parts) >= 5:
                            out.append(" ".join(parts[:5]))
                    else:
                        rotated = _rotate_label_line(line, width, height, matrix)
                        if rotated is not None:
                            out.append(rotated)
                return "\n".join(out) + ("\n" if out else "")

            def _available_destination(base_img: Path, base_label: Path) -> tuple[Path, Path]:
                if not base_img.exists() and not base_label.exists():
                    return base_img, base_label
                for idx in range(2, 10000):
                    img = base_img.with_name(f"{base_img.stem}_v{idx}{base_img.suffix}")
                    label = base_label.with_name(f"{base_label.stem}_v{idx}{base_label.suffix}")
                    if not img.exists() and not label.exists():
                        return img, label
                raise ValueError("could not find unused augmented filename")

            copied = 0
            skipped = 0
            errors: list[str] = []
            source_idx = 0
            attempts = 0
            max_attempts = max(requested_total * 3, len(rels))
            while copied < requested_total and attempts < max_attempts:
                attempts += 1
                rel = rels[source_idx % len(rels)]
                source_idx += 1
                try:
                    img_path = mlops_registry.resolve_dataset_image_path_at(dataset_root, rel)
                    if img_path is None:
                        skipped += 1
                        continue
                    label_path = mlops_registry.resolve_dataset_label_path(img_path)
                    image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if image is None:
                        skipped += 1
                        continue
                    h, w = image.shape[:2]
                    matrix = None
                    if abs(angle_deg) > 0.001:
                        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
                        image = cv2.warpAffine(
                            image,
                            matrix,
                            (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REFLECT_101,
                        )
                    if scale_pct != 100:
                        new_w = max(1, int(round(w * scale_pct / 100.0)))
                        new_h = max(1, int(round(h * scale_pct / 100.0)))
                        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA if scale_pct < 100 else cv2.INTER_CUBIC)
                    if grayscale:
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    tag = (
                        f"{suffix}_s{scale_pct}_r{int(round(angle_deg))}_"
                        f"q{jpeg_quality}_{'gray' if grayscale else 'rgb'}_{copied + 1:04d}"
                    )
                    dest_name = f"{img_path.stem}_{tag}{img_path.suffix.lower()}"
                    dest_img, dest_label = self._augmented_sample_destination(
                        dataset_root, img_path, target_split, dest_name
                    )
                    dest_img.parent.mkdir(parents=True, exist_ok=True)
                    dest_label.parent.mkdir(parents=True, exist_ok=True)
                    dest_img, dest_label = _available_destination(dest_img, dest_label)
                    encode_params: list[int] = []
                    if dest_img.suffix.lower() in {".jpg", ".jpeg"}:
                        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                    elif dest_img.suffix.lower() == ".webp":
                        encode_params = [int(cv2.IMWRITE_WEBP_QUALITY), jpeg_quality]
                    ok = cv2.imwrite(str(dest_img), image, encode_params)
                    if not ok:
                        raise ValueError("image encode failed")

                    dest_label.write_text(_augmented_labels(_label_lines(label_path), w, h, matrix), encoding="utf-8")
                    copied += 1
                except Exception as exc:
                    errors.append(f"{rel}: {exc}")
                    if len(errors) >= 8:
                        break

            val_layout_updated = False
            if target_split == "val" and copied > 0:
                self._ensure_yolo_val_layout(dataset_root)
                val_layout_updated = True

            return {
                "slug": slug,
                "target_split": target_split,
                "copied": copied,
                "skipped": skipped,
                "error_count": len(errors),
                "errors": errors,
                "scale_pct": scale_pct,
                "angle_deg": angle_deg,
                "jpeg_quality": jpeg_quality,
                "grayscale": grayscale,
                "balance_to_train": bool(req.balance_to_train),
                "split_counts": split_counts,
                "val_layout_updated": val_layout_updated,
            }

        @self.app.post("/database/{slug}/auto_augment")
        async def database_auto_augment(slug: str, req: AutoAugmentDatasetRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            requested_folders: set[str] = set()
            for raw_folder in (req.folders or []):
                folder = str(raw_folder or "").strip().strip("/").replace("\\", "/").lower()
                if folder == "(root)":
                    folder = ""
                if folder or str(raw_folder or "").strip() == "(root)":
                    requested_folders.add(folder)

            entries = [
                e for e in mlops_registry.list_dataset_entries_at(dataset_root)
                if str(e.get("split") or "").strip().lower() in {"train", "val"}
            ]
            if requested_folders:
                def _entry_folder(entry: dict[str, Any]) -> str:
                    rel = str(entry.get("relative_path") or "").strip().replace("\\", "/")
                    parts = [p for p in rel.split("/") if p]
                    if len(parts) >= 3 and parts[0].lower() == "images":
                        return "/".join(parts[2:-1]).lower()
                    return "/".join(parts[:-1]).lower()

                entries = [e for e in entries if _entry_folder(e) in requested_folders]
            split_counts = {"train": 0, "val": 0}
            source_rels: dict[str, list[str]] = {"train": [], "val": []}
            for entry in entries:
                split_name = str(entry.get("split") or "").strip().lower()
                rel = str(entry.get("relative_path") or "").strip()
                if split_name in split_counts:
                    split_counts[split_name] += 1
                    if rel:
                        source_rels[split_name].append(rel)

            current_total = int(split_counts["train"] + split_counts["val"])
            target_total = max(1, min(1_000_000, int(req.target_total or 1)))
            requested_total = min(max(0, target_total - current_total), 5000)
            effective_total = current_total + requested_total
            if requested_total <= 0:
                return {
                    "slug": slug,
                    "target_total": target_total,
                    "current_total": current_total,
                    "copied": 0,
                    "skipped": 0,
                    "error_count": 0,
                    "errors": [],
                    "split_counts": split_counts,
                    "additions_by_split": {"train": 0, "val": 0},
                }
            if not source_rels["train"] and not source_rels["val"]:
                raise HTTPException(status_code=400, detail="dataset has no train/val images to augment")

            if split_counts["train"] > 0 and split_counts["val"] > 0:
                train_target = int(round(effective_total * (split_counts["train"] / max(1, current_total))))
                val_target = effective_total - train_target
                train_target = max(split_counts["train"], train_target)
                val_target = max(split_counts["val"], val_target)
                while train_target + val_target > effective_total:
                    if (train_target - split_counts["train"]) >= (val_target - split_counts["val"]) and train_target > split_counts["train"]:
                        train_target -= 1
                    elif val_target > split_counts["val"]:
                        val_target -= 1
                    else:
                        break
            elif split_counts["train"] > 0:
                ensure_val = bool(req.ensure_val)
                val_frac = max(0.05, min(0.5, float(req.val_frac or 0.2)))
                if ensure_val and val_frac > 0.0:
                    val_target = max(1, int(round(effective_total * val_frac)))
                    train_target = effective_total - val_target
                    train_target = max(split_counts["train"], train_target)
                    val_target = effective_total - train_target
                else:
                    train_target = effective_total
                    val_target = 0
            else:
                train_target = 0
                val_target = effective_total

            additions_by_split = {
                "train": max(0, train_target - split_counts["train"]),
                "val": max(0, val_target - split_counts["val"]),
            }
            missing = requested_total - int(additions_by_split["train"] + additions_by_split["val"])
            if missing > 0:
                if split_counts["train"] >= split_counts["val"] and source_rels["train"]:
                    additions_by_split["train"] += missing
                elif source_rels["val"] or source_rels["train"]:
                    additions_by_split["val"] += missing
                else:
                    additions_by_split["train"] += missing

            source_pools: dict[str, list[str]] = {
                "train": list(source_rels["train"]),
                "val": list(source_rels["val"]) or list(source_rels["train"]),
            }
            val_layout_updated = False
            if additions_by_split["val"] > 0 and source_pools["val"]:
                self._ensure_yolo_val_layout(dataset_root)
                val_layout_updated = True

            rng = random.Random(req.seed)
            min_scale = max(10, min(300, int(req.min_scale_pct or 80)))
            max_scale = max(10, min(300, int(req.max_scale_pct or 120)))
            if min_scale > max_scale:
                min_scale, max_scale = max_scale, min_scale
            max_angle = max(0.0, min(180.0, float(req.max_angle_deg or 0.0)))
            min_quality = max(1, min(100, int(req.min_jpeg_quality or 70)))
            max_quality = max(1, min(100, int(req.max_jpeg_quality or 100)))
            if min_quality > max_quality:
                min_quality, max_quality = max_quality, min_quality
            grayscale_probability = max(0.0, min(1.0, float(req.grayscale_probability or 0.0)))
            bgr_shuffle_probability = max(0.0, min(1.0, float(req.bgr_shuffle_probability or 0.0)))
            channel_orders = [(2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)]

            copied = 0
            skipped = 0
            errors: list[str] = []
            written: dict[str, int] = {"train": 0, "val": 0}
            for target_split, split_total in additions_by_split.items():
                attempts = 0
                max_attempts = max(1, split_total * 4)
                pool = source_pools.get(target_split) or []
                if not pool:
                    continue
                while written[target_split] < split_total and attempts < max_attempts:
                    attempts += 1
                    try:
                        rel = rng.choice(pool)
                        img_path = mlops_registry.resolve_dataset_image_path_at(dataset_root, rel)
                        if img_path is None:
                            skipped += 1
                            continue
                        scale_pct = rng.randint(min_scale, max_scale)
                        angle_deg = rng.uniform(-max_angle, max_angle) if max_angle > 0.0 else 0.0
                        jpeg_quality = rng.randint(min_quality, max_quality)
                        grayscale = rng.random() < grayscale_probability
                        channel_order = None
                        if (not grayscale) and rng.random() < bgr_shuffle_probability:
                            channel_order = rng.choice(channel_orders)
                        self._write_augmented_yolo_sample(
                            dataset_root=dataset_root,
                            img_path=img_path,
                            target_split=target_split,
                            scale_pct=scale_pct,
                            angle_deg=angle_deg,
                            jpeg_quality=jpeg_quality,
                            grayscale=grayscale,
                            channel_order=channel_order,
                            suffix="autoaug",
                            sequence=copied + 1,
                        )
                        copied += 1
                        written[target_split] += 1
                    except Exception as exc:
                        errors.append(f"{target_split}: {exc}")
                        if len(errors) >= 8:
                            break
                if len(errors) >= 8:
                    break

            return {
                "slug": slug,
                "target_total": target_total,
                "current_total": current_total,
                "copied": copied,
                "skipped": skipped,
                "error_count": len(errors),
                "errors": errors,
                "split_counts": split_counts,
                "additions_by_split": written,
                "val_layout_updated": val_layout_updated,
                "ensure_val": bool(req.ensure_val),
                "val_frac": max(0.05, min(0.5, float(req.val_frac or 0.2))),
                "randomization": {
                    "scale_pct": [min_scale, max_scale],
                    "angle_deg": [-max_angle, max_angle],
                    "jpeg_quality": [min_quality, max_quality],
                    "grayscale_probability": grayscale_probability,
                    "bgr_shuffle_probability": bgr_shuffle_probability,
                },
            }

        @self.app.post("/database/{slug}/even_dataset")
        async def database_even_dataset(slug: str, req: EvenDatasetRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            requested_folders: set[str] = set()
            for raw_folder in (req.folders or []):
                folder = str(raw_folder or "").strip().strip("/").replace("\\", "/").lower()
                if folder == "(root)":
                    folder = ""
                if folder or str(raw_folder or "").strip() == "(root)":
                    requested_folders.add(folder)

            entries = [
                e for e in mlops_registry.list_dataset_entries_at(dataset_root)
                if str(e.get("split") or "").strip().lower() in {"train", "val"}
            ]
            if requested_folders:
                entries = [
                    e for e in entries
                    if self._yolo_entry_subfolder(e) in requested_folders
                ]
            if not entries:
                raise HTTPException(status_code=400, detail="dataset has no train/val images in scope")

            buckets: dict[tuple[str, str], list[str]] = {}
            for entry in entries:
                split_name = str(entry.get("split") or "").strip().lower()
                if split_name not in {"train", "val"}:
                    continue
                rel = str(entry.get("relative_path") or "").strip()
                if not rel:
                    continue
                folder_name = self._yolo_entry_subfolder(entry)
                key = (split_name, folder_name)
                buckets.setdefault(key, []).append(rel)

            folder_names = sorted({folder_name for _split, folder_name in buckets})
            splits = ["train", "val"]
            full_buckets: dict[tuple[str, str], list[str]] = {
                (split_name, folder_name): list(buckets.get((split_name, folder_name), []))
                for split_name in splits
                for folder_name in folder_names
            }
            before_counts = {f"{split_name}/{folder_name or '(root)'}": len(rels) for (split_name, folder_name), rels in full_buckets.items()}
            target_per_bucket = max((len(rels) for rels in full_buckets.values()), default=0)
            if target_per_bucket <= 0:
                return {
                    "slug": slug,
                    "target_per_bucket": 0,
                    "before_counts": before_counts,
                    "after_counts": dict(before_counts),
                    "copied": 0,
                    "skipped": 0,
                    "error_count": 0,
                    "errors": [],
                    "additions_by_split": {"train": 0, "val": 0},
                }

            deficits: list[tuple[str, str, int]] = []
            for (split_name, folder_name), rels in full_buckets.items():
                need = target_per_bucket - len(rels)
                if need > 0:
                    deficits.append((split_name, folder_name, need))

            total_need = sum(need for _split, _folder, need in deficits)
            max_copies = max(1, min(5000, int(req.max_copies or 5000)))
            if total_need > max_copies:
                raise HTTPException(
                    status_code=400,
                    detail=f"even dataset requires {total_need} augmented copies (max {max_copies})",
                )
            if total_need <= 0:
                return {
                    "slug": slug,
                    "target_per_bucket": target_per_bucket,
                    "before_counts": before_counts,
                    "after_counts": dict(before_counts),
                    "copied": 0,
                    "skipped": 0,
                    "error_count": 0,
                    "errors": [],
                    "additions_by_split": {"train": 0, "val": 0},
                }

            split_pools: dict[str, list[str]] = {"train": [], "val": []}
            for (split_name, _folder_name), rels in full_buckets.items():
                split_pools.setdefault(split_name, []).extend(rels)
            for split_name in list(split_pools):
                if not split_pools[split_name]:
                    split_pools[split_name] = list(split_pools.get("train", []))

            val_layout_updated = False
            if any(split_name == "val" and need > 0 for split_name, _folder, need in deficits):
                self._ensure_yolo_val_layout(dataset_root)
                val_layout_updated = True

            rng = random.Random(req.seed)
            min_scale = max(10, min(300, int(req.min_scale_pct or 80)))
            max_scale = max(10, min(300, int(req.max_scale_pct or 120)))
            if min_scale > max_scale:
                min_scale, max_scale = max_scale, min_scale
            max_angle = max(0.0, min(180.0, float(req.max_angle_deg or 0.0)))
            min_quality = max(1, min(100, int(req.min_jpeg_quality or 70)))
            max_quality = max(1, min(100, int(req.max_jpeg_quality or 100)))
            if min_quality > max_quality:
                min_quality, max_quality = max_quality, min_quality
            grayscale_probability = max(0.0, min(1.0, float(req.grayscale_probability or 0.0)))
            bgr_shuffle_probability = max(0.0, min(1.0, float(req.bgr_shuffle_probability or 0.0)))
            channel_orders = [(2, 1, 0), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1)]

            copied = 0
            skipped = 0
            errors: list[str] = []
            written: dict[str, int] = {"train": 0, "val": 0}
            additions_by_bucket: dict[str, int] = {}

            for split_name, folder_name, need in deficits:
                pool = list(full_buckets.get((split_name, folder_name), []))
                if not pool:
                    pool = list(split_pools.get(split_name) or split_pools.get("train") or [])
                if not pool:
                    skipped += need
                    continue
                bucket_key = f"{split_name}/{folder_name or '(root)'}"
                attempts = 0
                max_attempts = max(1, need * 4)
                while additions_by_bucket.get(bucket_key, 0) < need and attempts < max_attempts:
                    attempts += 1
                    try:
                        rel = rng.choice(pool)
                        img_path = mlops_registry.resolve_dataset_image_path_at(dataset_root, rel)
                        if img_path is None:
                            skipped += 1
                            continue
                        scale_pct = rng.randint(min_scale, max_scale)
                        angle_deg = rng.uniform(-max_angle, max_angle) if max_angle > 0.0 else 0.0
                        jpeg_quality = rng.randint(min_quality, max_quality)
                        grayscale = rng.random() < grayscale_probability
                        channel_order = None
                        if (not grayscale) and rng.random() < bgr_shuffle_probability:
                            channel_order = rng.choice(channel_orders)
                        self._write_augmented_yolo_sample(
                            dataset_root=dataset_root,
                            img_path=img_path,
                            target_split=split_name,
                            scale_pct=scale_pct,
                            angle_deg=angle_deg,
                            jpeg_quality=jpeg_quality,
                            grayscale=grayscale,
                            channel_order=channel_order,
                            suffix="even",
                            sequence=copied + 1,
                        )
                        copied += 1
                        written[split_name] = written.get(split_name, 0) + 1
                        additions_by_bucket[bucket_key] = additions_by_bucket.get(bucket_key, 0) + 1
                    except Exception as exc:
                        errors.append(f"{bucket_key}: {exc}")
                        if len(errors) >= 8:
                            break
                if len(errors) >= 8:
                    break

            after_counts = dict(before_counts)
            for bucket_key, added in additions_by_bucket.items():
                after_counts[bucket_key] = int(after_counts.get(bucket_key, 0)) + added

            return {
                "slug": slug,
                "target_per_bucket": target_per_bucket,
                "before_counts": before_counts,
                "after_counts": after_counts,
                "copied": copied,
                "skipped": skipped,
                "error_count": len(errors),
                "errors": errors,
                "additions_by_split": written,
                "additions_by_bucket": additions_by_bucket,
                "val_layout_updated": val_layout_updated,
            }

        @self.app.post("/database/{slug}/labels/clear_to_paths")
        async def database_clear_labels_to_paths(slug: str, req: ClearLabelsToPathsRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            rels = [str(p or "").strip().lstrip("/") for p in (req.relative_paths or []) if str(p or "").strip()]
            cleared = 0
            errors: list[str] = []
            for rel in rels:
                try:
                    img_path = mlops_registry.resolve_dataset_image_path_at(dataset_root, rel)
                    if img_path is None:
                        raise ValueError("image not found")
                    label_path = mlops_registry.resolve_dataset_label_path(img_path)
                    if label_path is None:
                        raise ValueError("could not resolve label path")
                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    label_path.write_text("", encoding="utf-8")
                    cleared += 1
                except Exception as exc:
                    errors.append(f"{rel}: {exc}")
                    if len(errors) >= 8:
                        break
            return {"slug": slug, "cleared": cleared, "error_count": len(errors), "errors": errors}

        @self.app.post("/database/{slug}/labels/bulk_apply_to_paths")
        async def database_bulk_apply_to_paths(slug: str, req: BulkLabelApplyToPathsRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            try:
                class_id = int(req.class_id)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="invalid class_id") from exc

            geom = str(req.geometry or "").strip().lower()
            if geom not in {"full_image", "center"}:
                raise HTTPException(status_code=400, detail="geometry must be full_image or center")
            cw = float(req.center_w or 0.0)
            ch = float(req.center_h or 0.0)
            if cw <= 0.0 or cw > 1.0 or ch <= 0.0 or ch > 1.0:
                raise HTTPException(status_code=400, detail="center_w/center_h must be in (0, 1]")

            only_missing = bool(req.only_missing)
            replace = bool(req.replace)
            limit = int(req.limit) if req.limit is not None else None

            if geom == "full_image":
                line = f"{class_id} 0.5 0.5 1.0 1.0\n"
            else:
                line = f"{class_id} 0.5 0.5 {cw:.6f} {ch:.6f}\n"

            rels = [str(p or "").strip().lstrip("/") for p in (req.relative_paths or []) if str(p or "").strip()]
            applied = 0
            skipped = 0
            errors: list[str] = []

            for rel in rels:
                if limit is not None and applied >= limit:
                    break
                try:
                    img_path = mlops_registry.resolve_dataset_image_path_at(dataset_root, rel)
                    if img_path is None:
                        skipped += 1
                        continue
                    label_path = mlops_registry.resolve_dataset_label_path(img_path)
                    if label_path is None:
                        skipped += 1
                        continue

                    if only_missing and label_path.exists():
                        try:
                            body = label_path.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            body = ""
                        if any(ln.strip() and not ln.strip().startswith("#") for ln in (body or "").splitlines()):
                            skipped += 1
                            continue

                    label_path.parent.mkdir(parents=True, exist_ok=True)
                    if replace:
                        label_path.write_text(line, encoding="utf-8")
                    else:
                        prev = ""
                        if label_path.exists():
                            try:
                                prev = label_path.read_text(encoding="utf-8", errors="replace")
                            except Exception:
                                prev = ""
                        body = prev
                        if body and not body.endswith("\n"):
                            body += "\n"
                        body += line
                        label_path.write_text(body, encoding="utf-8")

                    applied += 1
                except Exception as exc:
                    errors.append(f"{rel}: {exc}")
                    if len(errors) >= 8:
                        break

            return {
                "slug": slug,
                "geometry": geom,
                "class_id": class_id,
                "only_missing": only_missing,
                "replace": replace,
                "applied": applied,
                "skipped": skipped,
                "error_count": len(errors),
                "errors": errors,
            }

        @self.app.post("/database/{slug}/labels/remap_by_name")
        async def database_remap_labels_by_name(slug: str, req: RemapLabelsByNameRequest) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    != mlops_registry.LIBRARY_DATASET_FORMAT_YOLO
                ):
                    raise ValueError("dataset is not YOLO detection format")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            def clean_classes(raw: list[str]) -> list[str]:
                out: list[str] = []
                seen: set[str] = set()
                for item in raw or []:
                    name = str(item or "").strip()
                    if not name:
                        continue
                    key = name.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(name)
                    if len(out) >= 500:
                        break
                return out

            old = clean_classes(req.old_classes)
            new = clean_classes(req.new_classes)
            new_map: dict[str, int] = {}
            for i, name in enumerate(new):
                key = name.lower()
                if key not in new_map:
                    new_map[key] = i
            mapping: dict[int, int] = {}
            for i, name in enumerate(old):
                j = new_map.get(name.lower())
                if j is not None:
                    mapping[i] = int(j)

            labels_root = dataset_root.resolve() / "labels"
            if not labels_root.exists():
                return {"slug": slug, "mapping_size": len(mapping), "files_touched": 0, "files_changed": 0}

            drop_unmapped = bool(req.drop_unmapped)
            limit_files = int(req.limit_files) if req.limit_files is not None else None

            files_touched = 0
            files_changed = 0
            errors: list[str] = []

            for p in sorted(labels_root.rglob("*.txt"), key=lambda x: x.as_posix().lower()):
                if limit_files is not None and files_touched >= limit_files:
                    break
                if not p.is_file():
                    continue
                files_touched += 1
                try:
                    text = p.read_text(encoding="utf-8", errors="replace")
                except Exception as exc:
                    errors.append(str(exc))
                    if len(errors) >= 8:
                        break
                    continue
                out_lines: list[str] = []
                changed = False
                for raw_ln in (text or "").splitlines():
                    ln = raw_ln.rstrip("\r\n")
                    stripped = ln.strip()
                    if not stripped or stripped.startswith("#"):
                        out_lines.append(ln)
                        continue
                    parts = stripped.split()
                    if len(parts) < 5:
                        out_lines.append(ln)
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except Exception:
                        out_lines.append(ln)
                        continue
                    if cid in mapping:
                        new_id = int(mapping[cid])
                        if new_id != cid:
                            changed = True
                        parts[0] = str(new_id)
                        out_lines.append(" ".join(parts))
                        continue
                    if drop_unmapped:
                        changed = True
                        continue
                    out_lines.append(ln)
                new_text = "\n".join(out_lines).rstrip() + ("\n" if out_lines else "")
                if new_text != (text or ""):
                    try:
                        p.write_text(new_text, encoding="utf-8")
                        files_changed += 1
                    except Exception as exc:
                        errors.append(str(exc))
                        if len(errors) >= 8:
                            break

            return {
                "slug": slug,
                "mapping_size": len(mapping),
                "drop_unmapped": drop_unmapped,
                "files_touched": files_touched,
                "files_changed": files_changed,
                "error_count": len(errors),
                "errors": errors,
            }

        @self.app.post("/database/{slug}/convert/imagefolder_to_yolo")
        async def convert_imagefolder_to_yolo(slug: str, req: ImageFolderConvertRequest) -> dict[str, Any]:
            try:
                src_root = mlops_registry.resolve_library_dataset_path(slug)
                if mlops_registry.detect_library_dataset_format(src_root) != mlops_registry.LIBRARY_DATASET_FORMAT_IMAGEFOLDER:
                    raise HTTPException(status_code=400, detail="dataset is not ImageFolder classification format")

                preferred = str(req.output_slug or "").strip() or f"{slug}-yolo"
                out_slug = mlops_registry.pick_unique_library_dataset_slug(preferred)
                dest_root = mlops_registry.ensure_database_root() / out_slug
                result = mlops_registry.convert_imagefolder_to_yolo(
                    src_root,
                    dest_root,
                    mode=str(req.mode or "full_frame"),
                    include_test=bool(req.include_test),
                )
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            for scen in mlops_registry.scenario_names_for_dataset_folder(out_slug):
                self._emit_scenario_updated(scen)
            return {"source_slug": slug, "output_slug": out_slug, **(result or {})}

        @self.app.post("/database/{slug}/add")
        async def database_add_dataset_image(
            slug: str,
            image: UploadFile = File(...),
            label: Optional[UploadFile] = File(None),
            split: str = Form("train"),
            target_folder: str = Form(""),
            storage_mode: str = Form("yolo"),
            label_name: str = Form(""),
            create_empty_label: int = Form(0),
        ) -> dict[str, Any]:
            try:
                mode = str(storage_mode or "yolo").strip().lower()
                if mode in {"loose", "folder", "raw"}:
                    safe_slug = mlops_registry.sanitize_library_dataset_slug(slug)
                    try:
                        dataset_root = mlops_registry.resolve_library_dataset_path(safe_slug)
                    except Exception:
                        db_root = mlops_registry.ensure_database_root().resolve()
                        dataset_root = (db_root / safe_slug).resolve()
                        try:
                            dataset_root.relative_to(db_root)
                        except Exception as exc:
                            raise ValueError("Invalid database folder.") from exc
                        dataset_root.mkdir(parents=True, exist_ok=True)
                    payload = await self._store_loose_database_sample(
                        dataset_root,
                        image=image,
                        label=label,
                        target_folder=target_folder,
                        label_name=label_name,
                    )
                    for scen in mlops_registry.scenario_names_for_dataset_folder(safe_slug):
                        self._emit_scenario_updated(scen)
                    return {"slug": safe_slug, **payload}

                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                if (
                    mlops_registry.detect_library_dataset_format(dataset_root)
                    == mlops_registry.LIBRARY_DATASET_FORMAT_IMAGEFOLDER
                ):
                    raise ValueError(
                        "dataset is ImageFolder classification format; convert to YOLO before adding labeled samples"
                    )
                payload = await self._store_dataset_sample_root(
                    dataset_root,
                    image=image,
                    label=label,
                    split=split,
                    target_folder=target_folder,
                    create_empty_label=bool(int(create_empty_label or 0)),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            for scen in mlops_registry.scenario_names_for_dataset_folder(slug):
                self._emit_scenario_updated(scen)
            return {"slug": slug, **payload}

        @self.app.delete("/database/{slug}")
        async def delete_database_dataset(slug: str, force: int = 0) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            db_root = mlops_registry.DATABASE_ROOT.resolve()
            try:
                dataset_root.resolve().relative_to(db_root)
            except Exception as exc:
                raise HTTPException(status_code=400, detail="refusing to delete path outside database/") from exc
            scenarios_using = list(mlops_registry.scenario_names_for_dataset_folder(slug))
            if scenarios_using and not force:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "dataset is in use",
                        "scenarios": scenarios_using,
                        "hint": "re-send with ?force=1 to delete anyway",
                    },
                )
            try:
                shutil.rmtree(dataset_root)
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"delete failed: {exc}") from exc
            link_path = mlops_registry.MLOPS_ROOT / "datasets" / slug
            link_removed = False
            try:
                if link_path.is_symlink() or link_path.exists():
                    if link_path.is_dir() and not link_path.is_symlink():
                        shutil.rmtree(link_path)
                    else:
                        link_path.unlink()
                    link_removed = True
            except Exception:
                link_removed = False
            for scen in scenarios_using:
                self._emit_scenario_updated(scen)
            return {
                "deleted": slug,
                "scenarios_affected": scenarios_using,
                "mlops_link_removed": link_removed,
            }

        @self.app.delete("/database/{slug}/{name:path}")
        async def delete_database_dataset_image(slug: str, name: str) -> dict[str, Any]:
            try:
                dataset_root = mlops_registry.resolve_library_dataset_path(slug)
                match = mlops_registry.resolve_dataset_image_path_at(dataset_root, name)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if match is None:
                raise HTTPException(status_code=404, detail="image not found")
            label_path = mlops_registry.resolve_dataset_label_path(match)
            try:
                match.unlink()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            label_deleted = False
            if label_path is not None and label_path.exists():
                try:
                    label_path.unlink()
                    label_deleted = True
                except Exception:
                    label_deleted = False
            for scen in mlops_registry.scenario_names_for_dataset_folder(slug):
                self._emit_scenario_updated(scen)
            return {"deleted": name, "label_deleted": label_deleted}

        @self.app.post("/datasets/{scenario}/upload")
        async def upload_dataset_image(
            scenario: str,
            file: UploadFile = File(...),
        ) -> dict[str, Any]:
            try:
                cfg = mlops_registry.get_scenario_config(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in IMAGE_EXTS:
                raise HTTPException(status_code=400, detail=f"unsupported extension: {suffix}")
            raw = await file.read()
            if not raw:
                raise HTTPException(status_code=400, detail="empty upload")
            arr = np.frombuffer(raw, dtype=np.uint8)
            decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if decoded is None:
                raise HTTPException(status_code=400, detail="invalid image")
            incoming = MLOPS_ROOT / "datasets" / cfg.dataset / "incoming"
            incoming.mkdir(parents=True, exist_ok=True)
            stem = Path(file.filename or "upload").stem or "upload"
            safe_stem = "".join(c for c in stem if c.isalnum() or c in ("-", "_"))[:40] or "upload"
            target = incoming / f"{safe_stem}-{uuid.uuid4().hex[:8]}{suffix}"
            target.write_bytes(raw)
            self._emit_scenario_updated(cfg.name)
            return {"scenario": cfg.name, "saved": str(target), "size": len(raw)}

        @self.app.post("/datasets/{scenario}/add")
        async def add_dataset_image(
            scenario: str,
            image: UploadFile = File(...),
            label: Optional[UploadFile] = File(None),
            split: str = Form("train"),
            create_empty_label: int = Form(0),
        ) -> dict[str, Any]:
            try:
                payload = await self._store_dataset_sample(
                    scenario,
                    image=image,
                    label=label,
                    split=split,
                    create_empty_label=bool(int(create_empty_label or 0)),
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            self._emit_scenario_updated(scenario)
            return payload

        @self.app.delete("/datasets/{scenario}/{name:path}")
        async def delete_dataset_image(scenario: str, name: str) -> dict[str, Any]:
            try:
                match = mlops_registry.resolve_dataset_image_path(scenario, name)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            if match is None:
                raise HTTPException(status_code=404, detail="image not found")
            label_path = mlops_registry.resolve_dataset_label_path(match)
            try:
                match.unlink()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            label_deleted = False
            if label_path is not None and label_path.exists():
                try:
                    label_path.unlink()
                    label_deleted = True
                except Exception:
                    label_deleted = False
            self._emit_scenario_updated(scenario)
            return {"deleted": name, "label_deleted": label_deleted}

        @self.app.post("/jobs")
        async def submit_job(req: JobSubmitRequest) -> dict[str, Any]:
            scenario = req.scenario.strip()
            if not scenario:
                raise HTTPException(status_code=400, detail="scenario is required")
            try:
                cfg = mlops_registry.get_scenario_config(scenario)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            image = self._decode_b64_image(req.image_b64)
            if image is None:
                raise HTTPException(status_code=400, detail="image_b64 is invalid or empty")

            job_id = f"job-{uuid.uuid4().hex[:12]}"
            image_path = self._store_job_image(job_id, image)
            source = (req.source or "focus").strip().lower() or "focus"
            requested_version = str(req.version or "").strip()
            raw_model = str(req.model_artifact or "").strip()
            is_registry_ref = bool(
                raw_model
                and ":" in raw_model
                and "/" not in raw_model
                and "\\" not in raw_model
            )
            inference_target: Optional[dict[str, Any]] = None
            requested_model_artifact = ""

            if is_registry_ref:
                reg_scenario, reg_run = raw_model.split(":", 1)
                reg_scenario = reg_scenario.strip()
                reg_run = reg_run.strip()
                if reg_scenario != scenario:
                    raise HTTPException(
                        status_code=400,
                        detail="Registry model reference must use the same scenario as the job",
                    )
                if not reg_run:
                    raise HTTPException(status_code=400, detail="Invalid registry model reference")
                try:
                    selected_weights = mlops_registry.resolve_model_reference(raw_model)
                except Exception as exc:
                    raise HTTPException(status_code=400, detail=f"Invalid model reference: {exc}") from exc
                inference_target = {
                    "scenario": scenario,
                    "version": requested_version or reg_run,
                    "weights_path": selected_weights,
                    "source": "model_registry",
                }
                requested_model_artifact = raw_model
            else:
                requested_model_artifact = Path(raw_model).name if raw_model else ""
                inference_target = mlops_registry.resolve_inference_target(scenario, requested_version)
                if requested_version and inference_target is None:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Version '{requested_version}' for scenario '{scenario}' is not ready for inference",
                    )
                if requested_model_artifact:
                    if not requested_version:
                        raise HTTPException(status_code=400, detail="Custom model selection requires a run version")
                    run_dir = mlops_registry.resolve_scenario_run_dir(scenario, requested_version)
                    if run_dir is None:
                        raise HTTPException(status_code=400, detail=f"Run '{requested_version}' is not available")
                    run_record = mlops_registry.get_scenario_run_record(scenario, requested_version) or {}
                    allowed_files = {
                        str(run_record.get("final_model_file") or "").strip(),
                        "weights.pt",
                        "weights.pth",
                        "model.pkl",
                    }
                    allowed_files = {item for item in allowed_files if item}
                    if requested_model_artifact not in allowed_files:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Model artifact '{requested_model_artifact}' is not available for {requested_version}",
                        )
                    selected_weights = (run_dir / requested_model_artifact).resolve()
                    if not selected_weights.is_file():
                        raise HTTPException(
                            status_code=400, detail=f"Model artifact is missing: {requested_model_artifact}"
                        )
                    if inference_target is None:
                        inference_target = {
                            "scenario": scenario,
                            "version": requested_version,
                            "source": "run_artifact",
                        }
                    inference_target = dict(inference_target)
                    inference_target["weights_path"] = selected_weights
                    inference_target["source"] = f"run_artifact:{requested_model_artifact}"
            payload = {
                "scenario": scenario,
                "version": str((inference_target or {}).get("version") or requested_version),
                "requested_version": requested_version,
                "model_artifact": requested_model_artifact,
                "weights_path": str((inference_target or {}).get("weights_path") or ""),
                "inference_source": str((inference_target or {}).get("source") or ""),
                "source": source,
                "track_id": req.track_id,
                "entry_id": req.entry_id,
                "captured_at": req.captured_at,
            }
            if isinstance(req.infer_overrides, dict):
                payload["infer_overrides"] = dict(req.infer_overrides)
            if isinstance(req.backbone_config_override, dict):
                payload["backbone_config_override"] = dict(req.backbone_config_override)

            job_type = "infer" if inference_target is not None else "train"
            if job_type == "train":
                self._store_incoming_training_image(scenario, job_id, image)

            job = self.store.create_job(
                job_id=job_id,
                scenario=scenario,
                job_type=job_type,
                source=source,
                image_path=str(image_path),
                payload=payload,
            )
            self._emit_job_status(job)
            return {
                "job_id": job.job_id,
                "job_type": job.job_type,
                "state": job.state,
                "scenario": job.scenario,
                "source": job.source,
                "version": payload.get("version", ""),
            }

        @self.app.get("/jobs")
        def list_jobs(request: Request) -> Response:
            # Sync (threadpool): blocking sqlite read under a lock, polled by
            # the UI's startup resync. See the /scenarios handler.
            return self._json_cache_response(self.jobs_payload(), request, max_age=1)

        @self.app.get("/jobs/{job_id}")
        async def get_job(job_id: str) -> dict[str, Any]:
            try:
                return self.job_payload(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job not found") from exc

        @self.app.post("/jobs/{job_id}/cancel")
        async def cancel_job(job_id: str) -> dict[str, Any]:
            try:
                job = self.store.request_cancel(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job not found") from exc
            self._emit_job_status(job)
            return job.to_dict()

        @self.app.post("/jobs/{job_id}/retry")
        async def retry_job(job_id: str) -> dict[str, Any]:
            try:
                job = self.store.enqueue_retry(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job not found") from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._emit_job_status(job)
            return job.to_dict()

        @self.app.get("/jobs/{job_id}/result")
        async def get_job_result(job_id: str) -> dict[str, Any]:
            try:
                return self.job_result_payload(job_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="job result not found") from exc

        @self.app.get("/jobs/{job_id}/training_progress")
        def get_training_progress(job_id: str) -> dict[str, Any]:
            # Sync (threadpool): blocking read under _train_history_lock,
            # fetched once per train job during the UI's startup resync. See
            # the /scenarios handler.
            return self.training_progress_payload(job_id)

        @self.app.get("/jobs/{job_id}/artifacts")
        async def list_job_artifacts(job_id: str) -> dict[str, Any]:
            run_dir = self._resolve_job_run_dir(job_id)
            if run_dir is None:
                raise HTTPException(status_code=404, detail="run directory not available")
            try:
                job = self.store.get_job(job_id)
                cfg = mlops_registry.get_scenario_config(job.scenario)
                btype = str(cfg.backbone_type or "yolo_detection")
            except Exception:
                btype = ""
            return self._artifacts_payload(run_dir, job_id=job_id, backbone_type=btype)

        @self.app.get("/jobs/{job_id}/artifacts/{name:path}")
        async def get_job_artifact(job_id: str, name: str, full: int = Query(0)) -> Response:
            run_dir = self._resolve_job_run_dir(job_id)
            if run_dir is None:
                raise HTTPException(status_code=404, detail="run directory not available")
            return self._artifact_response(run_dir, name, full=full)

        @self.app.get("/scenarios/{scenario}/runs/{version}/artifacts")
        async def list_scenario_run_artifacts(scenario: str, version: str) -> dict[str, Any]:
            run_dir = mlops_registry.resolve_scenario_run_dir(scenario, version)
            if run_dir is None:
                raise HTTPException(status_code=404, detail="run directory not available")
            try:
                cfg = mlops_registry.get_scenario_config(scenario)
                btype = str(cfg.backbone_type or "yolo_detection")
            except Exception:
                btype = ""
            return self._artifacts_payload(run_dir, scenario=scenario, version=version, backbone_type=btype)

        @self.app.get("/scenarios/{scenario}/runs/{version}/metrics")
        async def get_scenario_run_metrics(scenario: str, version: str) -> dict[str, Any]:
            run_dir = mlops_registry.resolve_scenario_run_dir(scenario, version)
            if run_dir is None:
                raise HTTPException(status_code=404, detail="run directory not available")
            path = run_dir / "metrics.json"
            if not path.is_file():
                raise HTTPException(status_code=404, detail="metrics.json not found")
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"invalid metrics.json: {exc}") from exc
            if not isinstance(data, dict):
                raise HTTPException(status_code=500, detail="metrics.json must be an object")
            return data

        @self.app.get("/scenarios/{scenario}/runs/{version}/export")
        async def export_scenario_run(
            scenario: str,
            version: str,
            export_format: str = Query("onnx", alias="format"),
        ) -> FileResponse:
            run_dir = mlops_registry.resolve_scenario_run_dir(scenario, version)
            if run_dir is None:
                raise HTTPException(status_code=404, detail="run directory not available")
            try:
                cfg = mlops_registry.get_scenario_config(scenario)
                hp = cfg.hyperparams if isinstance(cfg.hyperparams, dict) else {}
                imgsz = int(hp.get("imgsz", 640) or 640)
            except Exception:
                imgsz = 640

            backbone_type = ""
            try:
                backbone_type = str(cfg.backbone_type or "yolo_detection")  # type: ignore[name-defined]
            except Exception:
                backbone_type = "yolo_detection"

            # Non-YOLO runs: allow direct download of primary artifact.
            if backbone_type != "yolo_detection":
                key = str(export_format or "").strip().lower()
                if key not in {"raw", "pth", "pt", "pkl", "safetensors"}:
                    raise HTTPException(
                        status_code=400,
                        detail="Non-YOLO runs support format=raw|pth|pt|pkl|safetensors (direct download), not YOLO export formats.",
                    )
                candidates = [
                    ("safetensors", run_dir / "adapter" / "adapter_model.safetensors"),
                    ("safetensors", run_dir / "adapter_model.safetensors"),
                    ("pth", run_dir / "weights.pth"),
                    ("pt", run_dir / "weights.pt"),
                    ("pkl", run_dir / "model.pkl"),
                ]
                chosen = None
                if key == "raw":
                    chosen = next((p for _k, p in candidates if p.is_file()), None)
                else:
                    chosen = next((p for k, p in candidates if k == key and p.is_file()), None)
                if chosen is None:
                    raise HTTPException(status_code=404, detail="No downloadable weights artifact found for this run")
                filename = self._final_model_download_name(run_dir, chosen.name, chosen.suffix)
                return FileResponse(
                    str(chosen),
                    filename=filename,
                    media_type="application/octet-stream",
                )

            weights = run_dir / "weights.pt"
            if not weights.is_file():
                raise HTTPException(status_code=404, detail="weights.pt not found for this run")
            try:
                from mlops.pipeline import export as mlops_export  # lazy: pulls torch
                out_path = mlops_export.export_registered_run(
                    weights,
                    export_format,
                    out_dir=run_dir / "exports",
                    imgsz=imgsz,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc
            media = "application/octet-stream"
            if out_path.suffix.lower() == ".onnx":
                media = "application/octet-stream"
            filename = self._final_model_download_name(run_dir, out_path.name, out_path.suffix)
            return FileResponse(
                str(out_path),
                filename=filename,
                media_type=media,
            )

        @self.app.get("/scenarios/{scenario}/runs/{version}/artifacts/{name:path}")
        async def get_scenario_run_artifact(
            scenario: str,
            version: str,
            name: str,
            full: int = Query(0),
        ) -> Response:
            run_dir = mlops_registry.resolve_scenario_run_dir(scenario, version)
            if run_dir is None:
                raise HTTPException(status_code=404, detail="run directory not available")
            return self._artifact_response(run_dir, name, full=full)

        # -----------------------------------------------------------------
        # Snapshots / Continuous Learning / Range
        # -----------------------------------------------------------------

        @self.app.get("/snapshots")
        async def list_snapshots(
            lineage_id: Optional[str] = None,
            origin: Optional[str] = None,
            tag: Optional[str] = None,
            limit: int = Query(200),
        ) -> dict[str, Any]:
            items = self.snapshots.list(
                lineage_id=lineage_id,
                origin=origin,
                tag=tag,
                limit=max(1, min(1000, int(limit))),
            )
            return {"items": [s.to_dict() for s in items]}

        @self.app.get("/snapshots/{snapshot_id}")
        async def get_snapshot(snapshot_id: str) -> dict[str, Any]:
            rec = self.snapshots.get(snapshot_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="snapshot not found")
            return rec.to_dict()

        @self.app.post("/snapshots")
        async def register_snapshot(req: SnapshotRegisterRequest) -> dict[str, Any]:
            try:
                rec = self.snapshots.register(
                    weights_path=Path(req.weights_path),
                    model_type=req.model_type,
                    storage_mode=req.storage_mode,
                    lineage_id=req.lineage_id,
                    parent_snapshot_id=req.parent_snapshot_id,
                    origin=req.origin,
                    adapter_only=req.adapter_only,
                    tags=req.tags,
                    metadata=req.metadata,
                )
            except FileNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self.provenance.record_snapshot_registered(rec)
            self._mark_graph_cache_dirty("snapshot_registered")
            return rec.to_dict()

        @self.app.delete("/snapshots/{snapshot_id}")
        async def delete_snapshot(snapshot_id: str, delete_weights: int = Query(1)) -> dict[str, Any]:
            snap_before = self.snapshots.get(snapshot_id)
            ok = self.snapshots.delete(snapshot_id, delete_weights=bool(delete_weights))
            if ok and snap_before is not None:
                self.provenance.record_snapshot_invalidated(snap_before)
                self._mark_graph_cache_dirty("snapshot_deleted")
            return {"deleted": bool(ok)}

        @self.app.get("/lineages")
        async def list_lineages(
            sector_path: Optional[str] = None,
            include_subtree: int = Query(1),
            state: Optional[str] = None,
        ) -> dict[str, Any]:
            items = self.lineages.list_lineages(
                sector_path=sector_path,
                include_subtree=bool(include_subtree),
                state=state,
            )
            out = [x.to_dict() for x in items]
            registry = _load_registry_lineages()
            for desc in registry.values():
                lin = desc["lineage"]
                if _registry_lineage_matches_filters(
                    lin,
                    sector_path=sector_path,
                    include_subtree=bool(include_subtree),
                    state=state,
                ):
                    out.append(lin)
            out.sort(key=lambda r: float(r.get("updated_at") or 0.0), reverse=True)
            return {"items": out}

        @self.app.post("/lineages")
        async def create_lineage(req: CreateLineageRequest) -> dict[str, Any]:
            try:
                rec = self.lineages.create_lineage(
                    name=req.name,
                    sector_id=req.sector_id,
                    sector_path=req.sector_path,
                    base_snapshot_id=req.base_snapshot_id,
                    update_strategy=req.update_strategy,
                    replay_config=req.replay_config,
                    description=req.description,
                    tags=req.tags,
                    metadata=req.metadata,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            drops0 = self.lineages.list_drops(rec.lineage_id)
            if drops0:
                base_drop = drops0[0]
                bsnap = self.snapshots.get(base_drop.snapshot_id)
                if bsnap is not None:
                    self.provenance.record_lineage_created(rec, base_drop, bsnap)
            self._mark_graph_cache_dirty("lineage_created")
            return rec.to_dict()

        @self.app.get("/lineages/{lineage_id}")
        async def get_lineage(lineage_id: str) -> dict[str, Any]:
            if lineage_id.startswith(_REGISTRY_LINEAGE_PREFIX):
                desc = _load_registry_lineages().get(lineage_id)
                if desc is None:
                    raise HTTPException(status_code=404, detail="lineage not found")
                return {
                    "lineage": desc["lineage"],
                    "drops": desc["drops"],
                    "head_snapshot": None,
                }
            rec = self.lineages.get_lineage(lineage_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="lineage not found")
            drops = self.lineages.list_drops(lineage_id)
            head_snap = None
            if rec.head_snapshot_id:
                h = self.snapshots.get(rec.head_snapshot_id)
                head_snap = h.to_dict() if h else None
            return {
                "lineage": rec.to_dict(),
                "drops": [d.to_dict() for d in drops],
                "head_snapshot": head_snap,
            }

        @self.app.get("/lineages/{lineage_id}/provenance")
        async def get_lineage_provenance(lineage_id: str) -> dict[str, Any]:
            if lineage_id.startswith(_REGISTRY_LINEAGE_PREFIX):
                raise HTTPException(
                    status_code=404,
                    detail="provenance is not persisted for registry-derived lineages",
                )
            if self.lineages.get_lineage(lineage_id) is None:
                raise HTTPException(status_code=404, detail="lineage not found")
            prov_doc = self.provenance.export_lineage_closure_prov_json(
                lineage_id, self.lineages
            )
            return {"lineage_id": lineage_id, "prov": prov_doc}

        @self.app.post("/lineages/{lineage_id}/drops")
        async def add_lineage_drop(lineage_id: str, req: AddDropRequest) -> dict[str, Any]:
            if lineage_id.startswith(_REGISTRY_LINEAGE_PREFIX):
                raise HTTPException(status_code=400, detail="registry-derived lineages are read-only")
            try:
                rec = self.lineages.add_drop(
                    lineage_id=lineage_id,
                    snapshot_id=req.snapshot_id,
                    source=dict(req.source or {}),
                    training_delta=req.training_delta,
                    sample_count=req.sample_count,
                    data_sha256=req.data_sha256,
                    replay=req.replay,
                    started_at=req.started_at,
                    finished_at=req.finished_at,
                    notes=req.notes,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            drops_new = self.lineages.list_drops(lineage_id)
            lin = self.lineages.get_lineage(lineage_id)
            if lin is not None and len(drops_new) >= 2:
                prev_drop = drops_new[-2]
                cur_drop = drops_new[-1]
                sn = self.snapshots.get(cur_drop.snapshot_id)
                psn = self.snapshots.get(prev_drop.snapshot_id)
                if sn is not None and psn is not None:
                    self.provenance.record_drop_added(
                        lineage=lin,
                        drop=cur_drop,
                        prev_drop=prev_drop,
                        snap_rec=sn,
                        prev_snap_rec=psn,
                    )
            self._mark_graph_cache_dirty("lineage_drop_added")
            return rec.to_dict()

        @self.app.post("/lineages/{lineage_id}/fork")
        async def fork_lineage(lineage_id: str, req: ForkLineageRequest) -> dict[str, Any]:
            if lineage_id.startswith(_REGISTRY_LINEAGE_PREFIX):
                raise HTTPException(status_code=400, detail="registry-derived lineages are read-only")
            try:
                rec = self.lineages.fork_lineage(
                    source_lineage_id=lineage_id,
                    at_drop_index=req.at_drop_index,
                    new_name=req.new_name,
                    description=req.description,
                    update_strategy=req.update_strategy,
                    replay_config=req.replay_config,
                )
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            drops_f = self.lineages.list_drops(rec.lineage_id)
            if drops_f:
                base_drop = drops_f[0]
                bsnap = self.snapshots.get(base_drop.snapshot_id)
                if bsnap is not None:
                    self.provenance.record_lineage_created(rec, base_drop, bsnap)
            self.provenance.record_lineage_fork(rec)
            self._mark_graph_cache_dirty("lineage_forked")
            return rec.to_dict()

        @self.app.post("/lineages/{lineage_id}/state")
        async def set_lineage_state(lineage_id: str, req: SetLineageStateRequest) -> dict[str, Any]:
            if lineage_id.startswith(_REGISTRY_LINEAGE_PREFIX):
                raise HTTPException(status_code=400, detail="registry-derived lineages are read-only")
            try:
                rec = self.lineages.set_state(lineage_id, req.state)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._mark_graph_cache_dirty("lineage_state_changed")
            return rec.to_dict()

        @self.app.delete("/lineages/{lineage_id}")
        async def delete_lineage(lineage_id: str) -> dict[str, Any]:
            if lineage_id.startswith(_REGISTRY_LINEAGE_PREFIX):
                raise HTTPException(status_code=400, detail="registry-derived lineages are read-only")
            lin_before = self.lineages.get_lineage(lineage_id)
            ok = self.lineages.delete_lineage(lineage_id)
            if ok and lin_before is not None:
                self.provenance.record_lineage_deleted(
                    lineage_id, label=lin_before.name
                )
                self._mark_graph_cache_dirty("lineage_deleted")
            return {"deleted": bool(ok)}

        @self.app.post("/provenance/backfill")
        async def provenance_backfill(req: ProvenanceBackfillRequest) -> dict[str, Any]:
            lid = str(req.lineage_id or "").strip()
            before = self.provenance.graph_counts()
            if lid:
                if self.lineages.get_lineage(lid) is None:
                    raise HTTPException(status_code=404, detail="lineage not found")
                self.provenance.backfill_lineage(lid, self.lineages, self.snapshots)
                n = 1
            else:
                n = self.provenance.backfill_all(self.lineages, self.snapshots)
            after = self.provenance.graph_counts()
            self._mark_graph_cache_dirty("provenance_backfill")
            out: dict[str, Any] = {
                "status": "ok",
                "lineages_processed": n,
                "prov_nodes_before": before["prov_nodes"],
                "prov_edges_before": before["prov_edges"],
                "prov_nodes_after": after["prov_nodes"],
                "prov_edges_after": after["prov_edges"],
            }
            if lid:
                out["lineage_id"] = lid
            return out

        @self.app.get("/ranges")
        async def list_ranges(
            sector_path: Optional[str] = None,
            include_subtree: int = Query(1),
        ) -> dict[str, Any]:
            items = self.ranges.list_ranges(
                sector_path=sector_path,
                include_subtree=bool(include_subtree),
            )
            return {"items": [x.to_dict() for x in items]}

        @self.app.post("/ranges")
        async def create_range(req: CreateRangeRequest) -> dict[str, Any]:
            try:
                rec = self.ranges.create_range(
                    name=req.name,
                    sector_id=req.sector_id,
                    sector_path=req.sector_path,
                    mode=req.mode,
                    description=req.description,
                    config=req.config,
                    tags=req.tags,
                    metadata=req.metadata,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._mark_graph_cache_dirty("range_created")
            return rec.to_dict()

        @self.app.get("/ranges/{range_id}")
        async def get_range(range_id: str) -> dict[str, Any]:
            rec = self.ranges.get_range(range_id)
            if rec is None:
                raise HTTPException(status_code=404, detail="range not found")
            subjects = self.ranges.list_subjects(range_id)
            goldens = self.ranges.list_golden_sets(range_id)
            drifts = self.ranges.list_drifts(range_id)
            gates = self.ranges.list_gates(range_id)
            return {
                "range": rec.to_dict(),
                "subjects": [s.to_dict() for s in subjects],
                "golden_sets": [g.to_dict() for g in goldens],
                "drifts": [d.to_dict() for d in drifts],
                "gates": [g.to_dict() for g in gates],
            }

        @self.app.delete("/ranges/{range_id}")
        async def delete_range(range_id: str) -> dict[str, Any]:
            ok = self.ranges.delete_range(range_id)
            if ok:
                self._mark_graph_cache_dirty("range_deleted")
            return {"deleted": bool(ok)}

        @self.app.post("/ranges/{range_id}/subjects")
        async def attach_subject(range_id: str, req: AttachSubjectRequest) -> dict[str, Any]:
            try:
                rec = self.ranges.attach_subject(
                    range_id=range_id,
                    snapshot_id=req.snapshot_id,
                    label=req.label,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._mark_graph_cache_dirty("range_subject_attached")
            return rec.to_dict()

        @self.app.delete("/ranges/{range_id}/subjects/{snapshot_id}")
        async def remove_subject(range_id: str, snapshot_id: str) -> dict[str, Any]:
            ok = self.ranges.remove_subject(range_id, snapshot_id)
            if ok:
                self._mark_graph_cache_dirty("range_subject_removed")
            return {"removed": bool(ok)}

        @self.app.post("/ranges/{range_id}/golden_sets")
        async def seal_golden_set(range_id: str, req: SealGoldenSetRequest) -> dict[str, Any]:
            try:
                rec = self.ranges.seal_golden_set(
                    range_id=range_id,
                    name=req.name,
                    split_spec=req.split_spec,
                    storage_uri=req.storage_uri,
                    row_count=req.row_count,
                    content_sha256=req.content_sha256,
                    description=req.description,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._mark_graph_cache_dirty("golden_set_created")
            return rec.to_dict()

        @self.app.delete("/ranges/{range_id}/golden_sets/{golden_id}")
        async def delete_golden_set(range_id: str, golden_id: str) -> dict[str, Any]:
            ok = self.ranges.delete_golden_set(golden_id)
            if ok:
                self._mark_graph_cache_dirty("golden_set_deleted")
            return {"deleted": bool(ok)}

        @self.app.post("/ranges/{range_id}/drifts")
        async def add_drift(range_id: str, req: AddDriftRequest) -> dict[str, Any]:
            try:
                rec = self.ranges.add_drift(
                    range_id=range_id,
                    name=req.name,
                    kind=req.kind,
                    params=req.params,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._mark_graph_cache_dirty("drift_added")
            return rec.to_dict()

        @self.app.delete("/ranges/{range_id}/drifts/{drift_id}")
        async def delete_drift(range_id: str, drift_id: str) -> dict[str, Any]:
            ok = self.ranges.delete_drift(drift_id)
            if ok:
                self._mark_graph_cache_dirty("drift_deleted")
            return {"deleted": bool(ok)}

        @self.app.get("/ranges/{range_id}/evaluations")
        async def list_evaluations(
            range_id: str,
            snapshot_id: Optional[str] = None,
            golden_id: Optional[str] = None,
        ) -> dict[str, Any]:
            items = self.ranges.list_evaluations(
                range_id=range_id,
                snapshot_id=snapshot_id,
                golden_id=golden_id,
            )
            return {"items": [x.to_dict() for x in items]}

        @self.app.post("/ranges/{range_id}/evaluations")
        async def record_evaluation(range_id: str, req: RecordEvaluationRequest) -> dict[str, Any]:
            try:
                rec = self.ranges.record_evaluation(
                    range_id=range_id,
                    snapshot_id=req.snapshot_id,
                    golden_id=req.golden_id,
                    metrics=req.metrics,
                    drift_id=req.drift_id,
                    predictions_uri=req.predictions_uri,
                    ran_at=req.ran_at,
                    duration_ms=req.duration_ms,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._mark_graph_cache_dirty("range_evaluation_recorded")
            return rec.to_dict()

        @self.app.delete("/ranges/{range_id}/evaluations/{eval_id}")
        async def delete_evaluation(range_id: str, eval_id: str) -> dict[str, Any]:
            ok = self.ranges.delete_evaluation(eval_id)
            if ok:
                self._mark_graph_cache_dirty("range_evaluation_deleted")
            return {"deleted": bool(ok)}

        @self.app.post("/ranges/{range_id}/gates")
        async def add_gate(range_id: str, req: AddGateRequest) -> dict[str, Any]:
            try:
                rec = self.ranges.add_gate(
                    range_id=range_id,
                    metric=req.metric,
                    threshold_type=req.threshold_type,
                    threshold_value=req.threshold_value,
                    golden_id=req.golden_id,
                    baseline_snapshot_id=req.baseline_snapshot_id,
                    action=req.action,
                )
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            self._mark_graph_cache_dirty("gate_added")
            return rec.to_dict()

        @self.app.delete("/ranges/{range_id}/gates/{gate_id}")
        async def delete_gate(range_id: str, gate_id: str) -> dict[str, Any]:
            ok = self.ranges.delete_gate(gate_id)
            if ok:
                self._mark_graph_cache_dirty("gate_deleted")
            return {"deleted": bool(ok)}

        # ---- Ontology Surface routes -------------------------------------
        from .ontology import get_entity as _get_entity

        @self.app.get("/ontology/cytoscape.js")
        async def ontology_cytoscape_js() -> Response:
            """Serve a locally cached cytoscape.js bundle.

            Cached under ``CVOPS_STATE_DIR / cache / cytoscape.js``.
            Downloaded once from jsdelivr on first request, then served from disk.
            Avoids QtWebEngine network issues (CORS, TLS, proxy, offline).
            """
            import urllib.request as _urlreq
            cache_dir = CVOPS_STATE_DIR / "cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            cached = cache_dir / "cytoscape-3.30.2.min.js"
            if not cached.exists() or cached.stat().st_size < 50_000:
                try:
                    url = "https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js"
                    with _urlreq.urlopen(url, timeout=15) as resp:  # noqa: S310
                        data = resp.read()
                    cached.write_bytes(data)
                except Exception as exc:
                    raise HTTPException(
                        status_code=503,
                        detail=f"cytoscape fetch failed: {exc}",
                    ) from exc
            return FileResponse(
                str(cached),
                media_type="application/javascript",
                headers={"Cache-Control": "public, max-age=86400"},
            )

        @self.app.get("/ontology/graph")
        def ontology_graph(
            request: Request,
            entity_types: str = Query(default=""),
            scenario: str = Query(default=""),
            depth: int = Query(default=2, ge=1, le=3),
            since_ts: Optional[float] = Query(default=None),
            layer: str = Query(default="full"),
        ) -> Response:
            try:
                return self._json_cache_response(
                    self._get_ontology_graph_cached(
                        layer=layer,
                        entity_types=entity_types,
                        scenario=scenario,
                        depth=depth,
                        since_ts=since_ts,
                    ),
                    request,
                    max_age=3,
                    stale_while_revalidate=30,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        @self.app.get("/ontology/entity/{entity_type}/{entity_id:path}")
        def ontology_entity(entity_type: str, entity_id: str) -> dict[str, Any]:
            try:
                return _get_entity(
                    entity_type,
                    entity_id,
                    job_store=self.store,
                    snapshots=self.snapshots,
                    lineages=self.lineages,
                    ranges=self.ranges,
                    catalog=self.catalog,
                    provenance=self.provenance,
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

        # ---- Relationship graph traversal --------------------------------

        def _full_graph() -> dict[str, Any]:
            return self._get_ontology_graph_cached(layer="full")

        @self.app.get("/ecosystem/impact/{entity_id:path}")
        def ecosystem_impact(entity_id: str) -> dict[str, Any]:
            """BFS upstream and downstream from entity_id in the full graph.

            Returns upstream (deps of this node), downstream (consumers of this
            node), and the edges that connect them.
            """
            try:
                graph = _full_graph()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            nodes_by_id = {n["id"]: n for n in graph["nodes"]}
            if entity_id not in nodes_by_id:
                raise HTTPException(status_code=404, detail=f"entity not found: {entity_id!r}")

            # Build directed and undirected adjacency
            fwd: dict[str, list[str]] = {}  # source → [targets]
            rev: dict[str, list[str]] = {}  # target → [sources]
            edge_key: dict[tuple[str, str], list[dict]] = {}
            for e in graph["edges"]:
                src, tgt = e["source"], e["target"]
                fwd.setdefault(src, []).append(tgt)
                rev.setdefault(tgt, []).append(src)
                edge_key.setdefault((src, tgt), []).append(e)

            def bfs(start: str, adj: dict[str, list[str]]) -> set[str]:
                visited: set[str] = set()
                queue = [start]
                while queue:
                    cur = queue.pop(0)
                    for nxt in adj.get(cur, []):
                        if nxt not in visited and nxt != start:
                            visited.add(nxt)
                            queue.append(nxt)
                return visited

            upstream = bfs(entity_id, rev)
            downstream = bfs(entity_id, fwd)
            affected = upstream | downstream | {entity_id}
            impact_edges = [
                e for e in graph["edges"]
                if e["source"] in affected and e["target"] in affected
            ]
            return {
                "entity_id": entity_id,
                "upstream": sorted(upstream),
                "downstream": sorted(downstream),
                "edges": impact_edges,
            }

        @self.app.get("/ecosystem/path")
        def ecosystem_path(
            from_id: str = Query(..., description="source entity id (type:id)"),
            to_id: str = Query(..., description="target entity id (type:id)"),
        ) -> dict[str, Any]:
            """BFS shortest path between two entity IDs (direction-agnostic).

            Returns {path: [node_ids], edges: [edge_dicts]} or {path: null} if
            the entities are not connected.
            """
            try:
                graph = _full_graph()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            nodes_by_id = {n["id"]: n for n in graph["nodes"]}
            if from_id not in nodes_by_id:
                raise HTTPException(status_code=404, detail=f"from_id not found: {from_id!r}")
            if to_id not in nodes_by_id:
                raise HTTPException(status_code=404, detail=f"to_id not found: {to_id!r}")

            # Undirected adjacency for path finding
            undirected: dict[str, list[str]] = {}
            edge_lookup: dict[tuple[str, str], dict] = {}
            for e in graph["edges"]:
                src, tgt = e["source"], e["target"]
                undirected.setdefault(src, []).append(tgt)
                undirected.setdefault(tgt, []).append(src)
                edge_lookup[(src, tgt)] = e
                edge_lookup[(tgt, src)] = e

            # BFS
            parent: dict[str, Optional[str]] = {from_id: None}
            queue = [from_id]
            found = False
            while queue:
                cur = queue.pop(0)
                if cur == to_id:
                    found = True
                    break
                for nxt in undirected.get(cur, []):
                    if nxt not in parent:
                        parent[nxt] = cur
                        queue.append(nxt)

            if not found:
                return {"from_id": from_id, "to_id": to_id, "path": None, "edges": []}

            path: list[str] = []
            cur = to_id
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            path.reverse()

            path_edges: list[dict] = []
            for i in range(len(path) - 1):
                e = edge_lookup.get((path[i], path[i + 1]))
                if e:
                    path_edges.append(e)

            return {"from_id": from_id, "to_id": to_id, "path": path, "edges": path_edges}

        @self.app.get("/ecosystem/orphans")
        def ecosystem_orphans() -> dict[str, Any]:
            """Return entity nodes that have no edges (degree 0).

            Useful for finding stale/unreferenced entities.
            """
            try:
                graph = _full_graph()
            except Exception as exc:
                raise HTTPException(status_code=500, detail=str(exc)) from exc

            connected: set[str] = set()
            for e in graph["edges"]:
                connected.add(e["source"])
                connected.add(e["target"])
            orphans = [n for n in graph["nodes"] if n["id"] not in connected]
            return {"count": len(orphans), "nodes": orphans}

        @self.app.websocket("/events")
        async def events(ws: WebSocket) -> None:
            await ws.accept()
            try:
                try:
                    replay_since = int(str(ws.query_params.get("since") or "0"))
                except Exception:
                    replay_since = 0
                await self._send_ws_json(
                    ws,
                    {
                        "type": "hello",
                        "service": "cvops",
                        "event_seq": self.latest_event_seq(),
                    },
                )
                await self._send_ws_json(ws, self._build_heartbeat_payload())
                with self._event_bus_lock:
                    replay, complete = self._events_since_locked(replay_since)
                if replay_since > 0:
                    await self._send_ws_json(
                        ws,
                        {
                            "type": "replay_begin",
                            "since": replay_since,
                            "count": len(replay),
                            "complete": bool(complete),
                        },
                    )
                    for payload in replay:
                        await self._send_ws_json(ws, payload)
                    await self._send_ws_json(
                        ws,
                        {
                            "type": "replay_end",
                            "since": replay_since,
                            "event_seq": self.latest_event_seq(),
                            "complete": bool(complete),
                        },
                    )
                with self._ws_lock:
                    self._ws_clients.add(ws)
                while True:
                    await ws.receive_text()
            except WebSocketDisconnect:
                pass
            except Exception:
                pass
            finally:
                with self._ws_lock:
                    self._ws_clients.discard(ws)

    def _resolve_job_run_dir(self, job_id: str) -> Optional[Path]:
        try:
            result = self.store.get_result(job_id)
        except Exception:
            result = None
        if not isinstance(result, dict):
            return None
        rp = str(result.get("result_path") or "").strip()
        if not rp:
            return None
        p = Path(rp)
        if not p.is_absolute():
            p = (MLOPS_ROOT / p).resolve()
        if not p.exists() or not p.is_dir():
            return None
        return p

    @staticmethod
    def _final_model_download_name(run_dir: Path, fallback: str, suffix: str) -> str:
        filename = str(fallback or "").strip() or f"model{suffix or '.bin'}"
        try:
            metrics = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
            final_name = str((metrics if isinstance(metrics, dict) else {}).get("final_model_name") or "").strip()
            if final_name:
                safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in final_name).strip("._-")
                if safe_name:
                    filename = f"{safe_name}{suffix or Path(filename).suffix or '.bin'}"
        except Exception:
            pass
        return filename

    def _scenario_has_existing_model(self, cfg: Any) -> bool:
        try:
            runs = mlops_registry.list_scenario_runs(str(cfg.name))
            if isinstance(runs, list) and len(runs) > 0:
                return True
        except Exception:
            pass
        try:
            weights_path = Path(str(getattr(cfg, "weights_path", ""))).resolve()
            if weights_path.exists() and weights_path.stat().st_size > 1024:
                return True
        except Exception:
            pass
        return False

    def _queue_train_like_job(
        self,
        *,
        cfg: Any,
        req: Optional[TrainKickRequest],
        resume: bool,
        save_period: int,
        trigger: str,
        update_mode: bool,
    ) -> JobRecord:
        job_id = f"job-{uuid.uuid4().hex[:12]}"
        backbone_config_override = None
        final_model_name = ""
        base_model_override = ""
        if req is not None and isinstance(req.backbone_config_override, dict) and req.backbone_config_override:
            backbone_config_override = dict(req.backbone_config_override)
        if req is not None:
            final_model_name = str(getattr(req, "final_model_name", "") or "").strip()
            base_model_override = str(getattr(req, "base_model_override", "") or "").strip()
        auto_fresh_on_completed_resume = True
        if req is not None:
            auto_fresh_on_completed_resume = bool(
                getattr(req, "auto_fresh_on_completed_resume", True)
            )
        # Face-recognition update mode enables incremental gallery growth by default.
        if update_mode and str(getattr(cfg, "backbone_type", "") or "") == "face_recognition":
            merged = dict(backbone_config_override or {})
            merged["incremental"] = True
            backbone_config_override = merged
        asset_root_for_job = ""
        device_for_job = ""
        if req is not None:
            for key in ("training_assets_root", "asset_save_root", "save_root"):
                val = str(getattr(req, key, "") or "").strip()
                if val:
                    asset_root_for_job = val
                    break
            device_for_job = str(getattr(req, "device", "") or "").strip()
        payload: dict[str, Any] = {
                "scenario": cfg.name,
                "source": "cvops_ui",
                "trigger": str(trigger or "manual"),
                "update_mode": bool(update_mode),
                "resume": bool(resume),
                "auto_fresh_on_completed_resume": bool(auto_fresh_on_completed_resume),
                "save_period": int(save_period),
                "final_model_name": final_model_name,
                "base_model_override": base_model_override,
                "backbone_config_override": backbone_config_override,
        }
        if asset_root_for_job:
            payload["training_assets_root"] = asset_root_for_job
        if device_for_job:
            payload["device"] = device_for_job
        return self.store.create_job(
            job_id=job_id,
            scenario=cfg.name,
            job_type="train",
            source="cvops_ui",
            image_path="",
            payload=payload,
        )

    @staticmethod
    def _dataset_fingerprint(
        dataset_root: Path,
        *,
        fmt: str,
        count: int,
        split_counts: dict[str, Any],
        classes: list[Any],
    ) -> str:
        h = hashlib.sha256()
        root = Path(dataset_root).resolve()
        h.update(str(root).encode("utf-8", errors="replace"))
        h.update(str(fmt or "").encode("utf-8", errors="replace"))
        h.update(str(int(count or 0)).encode("ascii", errors="ignore"))
        try:
            h.update(json.dumps(split_counts or {}, sort_keys=True, ensure_ascii=True).encode("utf-8"))
            h.update(json.dumps([str(c) for c in (classes or [])], sort_keys=True, ensure_ascii=True).encode("utf-8"))
        except Exception:
            pass
        seen = 0
        try:
            for path in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
                if seen >= 5000:
                    break
                try:
                    rel = path.relative_to(root).as_posix()
                    st = path.stat()
                except Exception:
                    continue
                h.update(rel.encode("utf-8", errors="replace"))
                h.update(str(int(st.st_size)).encode("ascii", errors="ignore"))
                h.update(str(int(st.st_mtime_ns)).encode("ascii", errors="ignore"))
                seen += 1
        except Exception:
            pass
        return h.hexdigest()

    @staticmethod
    def _normalize_dataset_split(split: str) -> str:
        value = str(split or "").strip().lower()
        if value not in ("train", "val"):
            raise ValueError("dataset split must be 'train' or 'val'")
        return value

    @staticmethod
    def _yolo_entry_subfolder(entry: dict[str, Any]) -> str:
        rel = str(entry.get("relative_path") or "").strip().replace("\\", "/")
        parts = [p for p in rel.split("/") if p]
        if len(parts) >= 3 and parts[0].lower() == "images":
            folder = "/".join(parts[2:-1]).lower()
        else:
            folder = "/".join(parts[:-1]).lower()
        return folder

    @staticmethod
    def _is_split_first_yolo(dataset_root: Path) -> bool:
        """True for split-first YOLO layouts (base/<split>/images + base/<split>/labels).

        Mirrors the listing precedence in the registry: a real top-level base/images/
        always wins, so a dataset is only treated as split-first when base/images/ is
        absent and at least one base/<split>/images + base/<split>/labels pair exists.
        """
        root = dataset_root.resolve()
        if (root / "images").is_dir():
            return False
        for split_dir in ("train", "valid", "val", "test"):
            if (root / split_dir / "images").is_dir() and (root / split_dir / "labels").is_dir():
                return True
        return False

    @staticmethod
    def _resolve_split_dir_name(dataset_root: Path, target_split: str) -> str:
        """On-disk split directory name for a split-first dataset.

        Prefers an existing directory (e.g. reuse 'valid' rather than creating 'val').
        """
        root = dataset_root.resolve()
        target = str(target_split or "").strip().lower()
        if target in {"val", "valid"}:
            for cand in ("valid", "val"):
                if (root / cand).is_dir():
                    return cand
            return "valid"
        if target == "test":
            return "test"
        return "train"

    @staticmethod
    def _set_yolo_val_in_data_yaml(dataset_root: Path, val_value: str) -> None:
        """Point the `val:` key in data.yaml at ``val_value`` without disturbing train."""
        data_yaml = dataset_root.resolve() / "data.yaml"
        if not data_yaml.exists():
            return
        try:
            lines = data_yaml.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return
        out: list[str] = []
        saw_val = False
        for line in lines:
            if line.strip().startswith("val:"):
                out.append(f"val: {val_value}")
                saw_val = True
            else:
                out.append(line)
        if not saw_val:
            insert_at = len(out)
            for idx, line in enumerate(out):
                if line.strip().startswith("train:"):
                    insert_at = idx + 1
                    break
            out.insert(insert_at, f"val: {val_value}")
        data_yaml.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")

    @classmethod
    def _augmented_sample_destination(
        cls,
        dataset_root: Path,
        img_path: Path,
        target_split: str,
        dest_name: str,
    ) -> tuple[Path, Path]:
        """Resolve (dest_image, dest_label) for an augmented sample, matching the
        dataset's existing YOLO layout so the output is enumerated by the listing.

        - images-first (base/images/<split>/...): write base/images/<target_split>/...
        - split-first  (base/<split>/images/...): write base/<target_split_dir>/images/...

        Any nested subfolders under the source split are preserved in the destination.
        """
        root = dataset_root.resolve()
        img = img_path.resolve()
        images_dir: Optional[Path] = None
        for parent in img.parents:
            if parent.name == "images":
                images_dir = parent
                break
        if images_dir is None:
            raise ValueError("image is not under images/")

        rel_under_images = img.relative_to(images_dir)
        parts = list(rel_under_images.parts)
        if parts and parts[0].lower() in {"train", "val", "valid", "test"}:
            parts = parts[1:] if len(parts) > 1 else [img.name]
        rel_parent = Path(*parts).parent

        split_first = (
            images_dir.parent.name.lower() in {"train", "val", "valid", "test"}
            and images_dir.parent.parent == root
        )
        if split_first:
            split_dir = cls._resolve_split_dir_name(root, target_split)
            images_root = root / split_dir / "images"
            labels_root = root / split_dir / "labels"
        else:
            images_root = images_dir / target_split
            labels_root = (images_dir.parent / "labels") / target_split

        dest_img = (images_root / rel_parent / dest_name).resolve()
        dest_label = (labels_root / rel_parent / dest_name).with_suffix(".txt").resolve()
        return dest_img, dest_label

    @staticmethod
    def _ensure_yolo_val_layout(dataset_root: Path) -> None:
        """Create val folders and point data.yaml at the val split when only train existed.

        Layout-aware: images-first datasets (base/images/<split>) get base/images/val,
        while split-first datasets (base/<split>/images) get base/<valid>/images so we
        never inject a stray top-level base/images/ that would hide split-first images
        from the dataset listing.
        """
        root = dataset_root.resolve()
        if CvOpsService._is_split_first_yolo(root):
            split_dir = CvOpsService._resolve_split_dir_name(root, "val")
            (root / split_dir / "images").mkdir(parents=True, exist_ok=True)
            (root / split_dir / "labels").mkdir(parents=True, exist_ok=True)
            CvOpsService._set_yolo_val_in_data_yaml(root, f"{split_dir}/images")
            return
        (root / "images" / "val").mkdir(parents=True, exist_ok=True)
        (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
        data_yaml = root / "data.yaml"
        if not data_yaml.exists():
            return
        try:
            lines = data_yaml.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            return
        out: list[str] = []
        saw_val = False
        saw_train = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("val:"):
                out.append("val: images/val")
                saw_val = True
            elif stripped.startswith("train:"):
                current = stripped.split(":", 1)[1].strip()
                if current in {"", "images", "images/"}:
                    out.append("train: images/train")
                else:
                    out.append(line)
                saw_train = True
            else:
                out.append(line)
        if not saw_val:
            insert_at = 0
            for idx, line in enumerate(out):
                if line.strip().startswith("train:"):
                    insert_at = idx + 1
                    break
            out.insert(insert_at, "val: images/val")
        if not saw_train:
            path_idx = next((i for i, line in enumerate(out) if line.strip().startswith("path:")), -1)
            out.insert(path_idx + 1, "train: images/train")
        data_yaml.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")

    @staticmethod
    def _dataset_label_lines(label_path: Optional[Path]) -> list[str]:
        if label_path is None or not label_path.exists():
            return []
        try:
            return [
                line.strip()
                for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        except Exception:
            return []

    @staticmethod
    def _rotate_yolo_label_line(line: str, width: int, height: int, matrix: np.ndarray) -> str | None:
        parts = line.split()
        if len(parts) < 5:
            return None
        try:
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        except Exception:
            return None
        x_c = cx * width
        y_c = cy * height
        half_w = bw * width / 2.0
        half_h = bh * height / 2.0
        corners = np.array(
            [
                [x_c - half_w, y_c - half_h, 1.0],
                [x_c + half_w, y_c - half_h, 1.0],
                [x_c + half_w, y_c + half_h, 1.0],
                [x_c - half_w, y_c + half_h, 1.0],
            ],
            dtype=np.float32,
        )
        rotated = corners @ matrix.T
        xs = np.clip(rotated[:, 0], 0.0, float(width))
        ys = np.clip(rotated[:, 1], 0.0, float(height))
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        new_w = x2 - x1
        new_h = y2 - y1
        if new_w <= 1.0 or new_h <= 1.0:
            return None
        return (
            f"{cls_id} {(x1 + x2) / 2.0 / width:.6f} {(y1 + y2) / 2.0 / height:.6f} "
            f"{new_w / width:.6f} {new_h / height:.6f}"
        )

    @classmethod
    def _augmented_yolo_labels(cls, lines: list[str], width: int, height: int, matrix: np.ndarray | None) -> str:
        out: list[str] = []
        for line in lines:
            if matrix is None:
                parts = line.split()
                if len(parts) >= 5:
                    out.append(" ".join(parts[:5]))
            else:
                rotated = cls._rotate_yolo_label_line(line, width, height, matrix)
                if rotated is not None:
                    out.append(rotated)
        return "\n".join(out) + ("\n" if out else "")

    @staticmethod
    def _available_augmented_destination(base_img: Path, base_label: Path) -> tuple[Path, Path]:
        if not base_img.exists() and not base_label.exists():
            return base_img, base_label
        for idx in range(2, 10000):
            img = base_img.with_name(f"{base_img.stem}_v{idx}{base_img.suffix}")
            label = base_label.with_name(f"{base_label.stem}_v{idx}{base_label.suffix}")
            if not img.exists() and not label.exists():
                return img, label
        raise ValueError("could not find unused augmented filename")

    @classmethod
    def _write_augmented_yolo_sample(
        cls,
        *,
        dataset_root: Path,
        img_path: Path,
        target_split: str,
        scale_pct: int,
        angle_deg: float,
        jpeg_quality: int,
        grayscale: bool,
        channel_order: tuple[int, int, int] | None,
        suffix: str,
        sequence: int,
    ) -> tuple[Path, Path]:
        label_path = mlops_registry.resolve_dataset_label_path(img_path)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("image read failed")
        h, w = image.shape[:2]
        matrix = None
        if abs(angle_deg) > 0.001:
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
            image = cv2.warpAffine(
                image,
                matrix,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )
        if scale_pct != 100:
            new_w = max(1, int(round(w * scale_pct / 100.0)))
            new_h = max(1, int(round(h * scale_pct / 100.0)))
            interpolation = cv2.INTER_AREA if scale_pct < 100 else cv2.INTER_CUBIC
            image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        if grayscale:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif channel_order is not None:
            image = image[:, :, list(channel_order)]

        tag = (
            f"{suffix}_s{int(scale_pct)}_r{int(round(angle_deg))}_"
            f"q{int(jpeg_quality)}_{'gray' if grayscale else 'bgr'}_{int(sequence):05d}"
        )
        dest_name = f"{img_path.stem}_{tag}{img_path.suffix.lower()}"
        dest_img, dest_label = cls._augmented_sample_destination(
            dataset_root, img_path, target_split, dest_name
        )
        dest_img.parent.mkdir(parents=True, exist_ok=True)
        dest_label.parent.mkdir(parents=True, exist_ok=True)
        dest_img, dest_label = cls._available_augmented_destination(dest_img, dest_label)

        encode_params: list[int] = []
        if dest_img.suffix.lower() in {".jpg", ".jpeg"}:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        elif dest_img.suffix.lower() == ".webp":
            encode_params = [int(cv2.IMWRITE_WEBP_QUALITY), int(jpeg_quality)]
        ok = cv2.imwrite(str(dest_img), image, encode_params)
        if not ok:
            raise ValueError("image encode failed")
        dest_label.write_text(
            cls._augmented_yolo_labels(cls._dataset_label_lines(label_path), w, h, matrix),
            encoding="utf-8",
        )
        return dest_img, dest_label

    @staticmethod
    def _sanitize_dataset_stem(stem: str) -> str:
        cleaned = "".join(c for c in stem if c.isalnum() or c in ("-", "_", " ")).strip()
        return cleaned[:80] or "sample"

    @classmethod
    def _sanitize_dataset_subfolder(cls, folder: str) -> list[str]:
        value = str(folder or "").strip().strip("/")
        if not value or value.lower() in {"root", "."}:
            return []
        parts: list[str] = []
        for raw in re.split(r"[\\/]+", value)[:4]:
            raw = str(raw or "").strip()
            if not raw or raw in {".", ".."}:
                continue
            cleaned = cls._sanitize_dataset_stem(raw)
            if cleaned and cleaned not in {".", ".."}:
                parts.append(cleaned)
        return parts

    @staticmethod
    def _parse_json_field(raw: str, *, field_name: str) -> dict[str, Any]:
        value = str(raw or "").strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except Exception as exc:
            raise ValueError(f"{field_name} must be valid JSON object") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{field_name} must be a JSON object")
        return parsed

    @staticmethod
    def _parse_csv_tokens(raw: str) -> list[str]:
        value = str(raw or "").strip()
        if not value:
            return []
        return [tok.strip() for tok in value.split(",") if tok.strip()]

    def _resolve_ingest_request(
        self,
        *,
        req: Optional[IngestAssetRequest],
        name: str,
        source_type: str,
        storage_mode: str,
        sector_id: str,
        sector_path: str,
        source_uri: str,
        collection_id: str,
        tags: str,
        keywords: str,
        lineage_json: str,
        metadata_json: str,
    ) -> IngestAssetRequest:
        if req is not None and str(req.source_type or "").strip():
            return req
        return IngestAssetRequest(
            name=str(name or "").strip(),
            source_type=str(source_type or "").strip(),
            storage_mode=str(storage_mode or "").strip() or "reference",
            sector_id=str(sector_id or "").strip() or ROOT_SECTOR_ID,
            sector_path=str(sector_path or "").strip(),
            source_uri=str(source_uri or "").strip(),
            collection_id=str(collection_id or "").strip(),
            tags=self._parse_csv_tokens(tags),
            keywords=self._parse_csv_tokens(keywords),
            lineage=self._parse_json_field(lineage_json, field_name="lineage_json"),
            metadata=self._parse_json_field(metadata_json, field_name="metadata_json"),
        )

    @staticmethod
    def _normalize_doc_source_type(raw: str, *, filename: str = "", source_uri: str = "") -> str:
        value = str(raw or "").strip().lower()
        if value in {"pdf", "txt", "md", "yolo_dataset", "csv", "imagefolder", "audiofolder"}:
            return value
        probe = filename or source_uri
        ext = Path(str(probe or "")).suffix.lower()
        if ext == ".pdf":
            return "pdf"
        if ext == ".txt":
            return "txt"
        if ext in {".md", ".markdown"}:
            return "md"
        if ext == ".csv":
            return "csv"
        raise ValueError("source_type must be one of: pdf, txt, md, yolo_dataset, csv, imagefolder, audiofolder")

    def _asset_with_reference_health(self, payload: dict[str, Any]) -> dict[str, Any]:
        if str(payload.get("storage_mode") or "") != "reference":
            return payload
        source_uri = str(payload.get("source_uri") or "").strip()
        if not source_uri:
            return payload
        candidate = Path(source_uri).expanduser()
        if not candidate.is_absolute():
            candidate = (ROOT_DIR / candidate).resolve()
        exists = candidate.exists()
        payload = dict(payload)
        payload["availability_status"] = "ok" if exists else "missing"
        payload["last_checked_at"] = time.time()
        try:
            payload.setdefault("metadata", {})
            if isinstance(payload["metadata"], dict):
                payload["metadata"]["reference_exists"] = bool(exists)
                payload["metadata"]["reference_path"] = str(candidate)
        except Exception:
            pass
        try:
            self.catalog.update_asset_availability(
                str(payload.get("asset_id") or ""),
                availability_status=str(payload.get("availability_status") or "unknown"),
                metadata_patch=payload.get("metadata") if isinstance(payload.get("metadata"), dict) else None,
            )
        except Exception:
            pass
        return payload

    def _get_asset_payload(self, asset_id: str, *, refresh_reference: bool) -> dict[str, Any]:
        asset = self.catalog.get_asset(asset_id)
        payload = asset.to_dict()
        if refresh_reference:
            payload = self._asset_with_reference_health(payload)
        return payload

    async def _ingest_asset(
        self,
        req: IngestAssetRequest,
        *,
        file: Optional[UploadFile],
    ) -> dict[str, Any]:
        source_uri = str(req.source_uri or "").strip()
        filename = str(getattr(file, "filename", "") or "")
        source_type = self._normalize_doc_source_type(
            req.source_type,
            filename=filename,
            source_uri=source_uri,
        )
        storage_mode = self.catalog.normalize_storage_mode(req.storage_mode)
        sector = self.catalog.get_sector(
            sector_path=str(req.sector_path or "").strip(),
        ) if str(req.sector_path or "").strip() else self.catalog.get_sector(
            sector_id=str(req.sector_id or "").strip() or ROOT_SECTOR_ID
        )

        if source_type in {"yolo_dataset", "csv", "imagefolder", "audiofolder"}:
            return self._ingest_existing_dataset_type(
                req=req,
                source_type=source_type,
                storage_mode=storage_mode,
                sector_id=sector.sector_id,
            )
        return await self._ingest_document_asset(
            req=req,
            source_type=source_type,
            storage_mode=storage_mode,
            sector_id=sector.sector_id,
            file=file,
        )

    def _ingest_existing_dataset_type(
        self,
        *,
        req: IngestAssetRequest,
        source_type: str,
        storage_mode: str,
        sector_id: str,
    ) -> dict[str, Any]:
        slug = str(req.source_uri or req.name or "").strip()
        if not slug:
            raise ValueError("dataset ingestion requires source_uri as dataset slug")
        ds_root = mlops_registry.resolve_library_dataset_path(slug)
        if storage_mode == "managed_copy":
            raise ValueError("managed_copy is not supported for dataset-slug ingestion; use reference")
        fmt = mlops_registry.detect_library_dataset_format(ds_root)
        expected = {
            "yolo_dataset": mlops_registry.LIBRARY_DATASET_FORMAT_YOLO,
            "csv": mlops_registry.LIBRARY_DATASET_FORMAT_CSV,
            "imagefolder": mlops_registry.LIBRARY_DATASET_FORMAT_IMAGEFOLDER,
            "audiofolder": mlops_registry.LIBRARY_DATASET_FORMAT_AUDIOFOLDER,
        }[source_type]
        if fmt != expected:
            raise ValueError(f"dataset '{slug}' has format '{fmt}', expected '{expected}'")
        info = mlops_registry.inspect_library_dataset_at(ds_root)
        metadata = dict(req.metadata or {})
        metadata.update(
            {
                "dataset_slug": slug,
                "dataset_path": str(ds_root.resolve()),
                "dataset_format": fmt,
                "dataset_category": mlops_registry.dataset_category(fmt),
                "split_counts": info.get("split_counts", {}),
                "classes": info.get("classes", []),
                "count": int(info.get("count") or 0),
            }
        )
        asset = self.catalog.create_asset(
            name=str(req.name or slug),
            source_type=source_type,
            storage_mode="reference",
            sector_id=sector_id,
            source_uri=str(ds_root.resolve()),
            managed_path="",
            status="ingested",
            schema_status="inferred",
            extraction_status="complete",
            availability_status="ok",
            size_bytes=0,
            tags=list(req.tags or []),
            keywords=list(req.keywords or []),
            lineage=dict(req.lineage or {}),
            metadata=metadata,
            collection_id=str(req.collection_id or "").strip(),
        )
        return asset.to_dict()

    async def _ingest_document_asset(
        self,
        *,
        req: IngestAssetRequest,
        source_type: str,
        storage_mode: str,
        sector_id: str,
        file: Optional[UploadFile],
    ) -> dict[str, Any]:
        raw_data = b""
        source_uri = str(req.source_uri or "").strip()
        basename = ""
        if file is not None:
            basename = Path(str(file.filename or "")).name or f"ingest-{uuid.uuid4().hex[:8]}"
            raw_data = await file.read()
            if not raw_data:
                raise ValueError("uploaded file is empty")
        if not raw_data and source_uri:
            p = Path(source_uri).expanduser()
            if not p.is_absolute():
                p = (ROOT_DIR / p).resolve()
            if not p.exists() or not p.is_file():
                raise ValueError(f"source_uri file not found: {p}")
            basename = p.name
            if storage_mode == "managed_copy":
                raw_data = p.read_bytes()
            source_uri = str(p)
        if storage_mode == "managed_copy" and not raw_data:
            raise ValueError("managed_copy requires uploaded file or readable source_uri file")
        if not basename:
            basename = str(req.name or f"ingest-{uuid.uuid4().hex[:8]}")

        ext = Path(basename).suffix.lower()
        expected_ext = {"pdf": ".pdf", "txt": ".txt", "md": ".md"}[source_type]
        if ext and source_type in {"pdf", "txt"} and ext != expected_ext:
            raise ValueError(f"source_type '{source_type}' expects '{expected_ext}' extension")
        if source_type == "md" and ext not in {".md", ".markdown", ""}:
            raise ValueError("source_type 'md' expects .md/.markdown extension")

        managed_path = ""
        if storage_mode == "managed_copy":
            asset_folder = self.catalog_assets_root / f"asset-{uuid.uuid4().hex[:12]}"
            asset_folder.mkdir(parents=True, exist_ok=True)
            safe_name = Path(basename).name or f"asset{expected_ext}"
            target = asset_folder / safe_name
            target.write_bytes(raw_data)
            managed_path = str(target.resolve())
            if not source_uri:
                source_uri = managed_path
        elif not source_uri:
            raise ValueError("reference storage requires source_uri")

        metadata = dict(req.metadata or {})
        schema_status = "not_applicable"
        extraction_status = "pending"
        availability_status = "ok"
        size_bytes = 0
        text_probe = ""
        candidate_path = Path(managed_path or source_uri)
        if candidate_path.exists() and candidate_path.is_file():
            try:
                size_bytes = int(candidate_path.stat().st_size)
            except Exception:
                size_bytes = 0
            if source_type in {"txt", "md"}:
                try:
                    text_probe = candidate_path.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    text_probe = ""
            extraction_status = "complete" if source_type in {"txt", "md"} else "pending"
        else:
            availability_status = "missing"
            extraction_status = "pending"
        mime_guess = mimetypes.guess_type(str(candidate_path))[0] or ""
        metadata.update(
            {
                "filename": candidate_path.name,
                "extension": ext or expected_ext,
                "mime_type": mime_guess,
                "reference_exists": availability_status == "ok",
            }
        )
        if text_probe:
            lines = text_probe.splitlines()
            metadata["text_stats"] = {
                "line_count": len(lines),
                "char_count": len(text_probe),
                "word_count": len([t for t in text_probe.split() if t.strip()]),
            }
        asset = self.catalog.create_asset(
            name=str(req.name or candidate_path.stem or candidate_path.name),
            source_type=source_type,
            storage_mode=storage_mode,
            sector_id=sector_id,
            source_uri=str(source_uri),
            managed_path=str(managed_path),
            status="ingested" if availability_status == "ok" else "error",
            schema_status=schema_status,
            extraction_status=extraction_status,
            availability_status=availability_status,
            size_bytes=size_bytes,
            tags=list(req.tags or []),
            keywords=list(req.keywords or []),
            lineage=dict(req.lineage or {}),
            metadata=metadata,
            collection_id=str(req.collection_id or "").strip(),
        )
        return asset.to_dict()

    async def _store_dataset_sample_root(
        self,
        dataset_root: Path,
        *,
        image: UploadFile,
        label: Optional[UploadFile],
        split: str,
        target_folder: str = "",
        create_empty_label: bool,
    ) -> dict[str, Any]:
        root = dataset_root.resolve()
        split_name = self._normalize_dataset_split(split)
        folder_parts = self._sanitize_dataset_subfolder(target_folder)
        image_name = str(image.filename or "").strip()
        suffix = Path(image_name).suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise ValueError(f"unsupported image extension: {suffix}")
        raw_image = await image.read()
        if not raw_image:
            raise ValueError("empty image upload")
        arr = np.frombuffer(raw_image, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("invalid image")

        raw_label = b""
        if label is not None:
            label_name = str(label.filename or "").strip()
            if Path(label_name).suffix.lower() not in ("", ".txt"):
                raise ValueError("label file must use .txt")
            raw_label = await label.read()
        elif not create_empty_label:
            raise ValueError("label file is required unless creating an empty label")

        image_dir = root / "images" / split_name
        label_dir = root / "labels" / split_name
        for part in folder_parts:
            image_dir = image_dir / part
            label_dir = label_dir / part
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        base_stem = self._sanitize_dataset_stem(Path(image_name).stem)
        target_stem = base_stem
        image_target = image_dir / f"{target_stem}{suffix}"
        label_target = label_dir / f"{target_stem}.txt"
        index = 1
        while image_target.exists() or label_target.exists():
            target_stem = f"{base_stem}-{index:02d}"
            image_target = image_dir / f"{target_stem}{suffix}"
            label_target = label_dir / f"{target_stem}.txt"
            index += 1

        image_target.write_bytes(raw_image)
        label_target.write_bytes(raw_label if label is not None else b"")
        return {
            "split": split_name,
            "target_folder": "/".join(folder_parts),
            "image_path": str(image_target),
            "image_relative_path": image_target.relative_to(root).as_posix(),
            "label_path": str(label_target),
            "label_relative_path": label_target.relative_to(root).as_posix(),
            "label_created_empty": label is None,
        }

    async def _store_loose_database_sample(
        self,
        dataset_root: Path,
        *,
        image: UploadFile,
        label: Optional[UploadFile],
        target_folder: str = "",
        label_name: str = "",
    ) -> dict[str, Any]:
        root = dataset_root.resolve()
        folder_parts = self._sanitize_dataset_subfolder(target_folder)
        target_dir = root
        for part in folder_parts:
            target_dir = target_dir / part
        target_dir.mkdir(parents=True, exist_ok=True)

        image_name = str(image.filename or "").strip()
        suffix = Path(image_name).suffix.lower()
        if suffix not in IMAGE_EXTS:
            raise ValueError(f"unsupported image extension: {suffix}")
        raw_image = await image.read()
        if not raw_image:
            raise ValueError("empty image upload")
        arr = np.frombuffer(raw_image, dtype=np.uint8)
        decoded = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("invalid image")

        raw_label = b""
        label_created_empty = False
        label_text = str(label_name or "").strip()
        if label is not None:
            label_file_name = str(label.filename or "").strip()
            if Path(label_file_name).suffix.lower() not in ("", ".txt"):
                raise ValueError("label file must use .txt")
            raw_label = await label.read()
        elif label_text:
            raw_label = (label_text + "\n").encode("utf-8")
        else:
            label_created_empty = True

        base_stem = self._sanitize_dataset_stem(Path(image_name).stem)
        target_stem = base_stem
        image_target = target_dir / f"{target_stem}{suffix}"
        label_target = target_dir / f"{target_stem}.txt"
        index = 1
        while image_target.exists() or label_target.exists():
            target_stem = f"{base_stem}-{index:02d}"
            image_target = target_dir / f"{target_stem}{suffix}"
            label_target = target_dir / f"{target_stem}.txt"
            index += 1

        image_target.write_bytes(raw_image)
        label_target.write_bytes(raw_label)
        return {
            "storage_mode": "loose",
            "target_folder": "/".join(folder_parts),
            "image_path": str(image_target),
            "image_relative_path": image_target.relative_to(root).as_posix(),
            "label_path": str(label_target),
            "label_relative_path": label_target.relative_to(root).as_posix(),
            "label_created_empty": label_created_empty,
        }

    async def _store_dataset_sample(
        self,
        scenario: str,
        *,
        image: UploadFile,
        label: Optional[UploadFile],
        split: str,
        create_empty_label: bool,
    ) -> dict[str, Any]:
        cfg = mlops_registry.get_scenario_config(scenario)
        payload = await self._store_dataset_sample_root(
            cfg.dataset_path,
            image=image,
            label=label,
            split=split,
            create_empty_label=create_empty_label,
        )
        return {"scenario": scenario, **payload}

    def _artifacts_payload(self, run_dir: Path, **extra: Any) -> dict[str, Any]:
        payload = {"run_dir": str(run_dir), "items": self._enumerate_run_artifacts(run_dir)}
        payload.update(extra)
        return payload

    def _artifact_response(self, run_dir: Path, name: str, *, full: int = 0) -> Response:
        items = self._enumerate_run_artifacts(run_dir)
        allowed = {it["name"] for it in items}
        if name not in allowed:
            raise HTTPException(status_code=404, detail="artifact not found")
        path = (run_dir / name).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except Exception as exc:
            raise HTTPException(status_code=400, detail="invalid artifact path") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="artifact missing")
        kind = self._classify_artifact(name)
        if kind == "image":
            try:
                img = cv2.imread(str(path))
            except Exception:
                img = None
            if img is None:
                return FileResponse(str(path))
            if not full:
                h, w = img.shape[:2]
                max_side = 360
                if max(h, w) > max_side:
                    scale = max_side / float(max(h, w))
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
            ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                return FileResponse(str(path))
            return Response(
                content=base64.b64encode(buf.tobytes()).decode("ascii"),
                media_type="text/plain",
            )
        return FileResponse(str(path))

    @staticmethod
    def _classify_artifact(name: str) -> str:
        ext = Path(name).suffix.lower()
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
            return "image"
        if ext == ".csv":
            return "csv"
        if ext == ".json":
            return "json"
        if ext == ".jsonl":
            return "jsonl"
        if ext in {".yaml", ".yml"}:
            return "yaml"
        if Path(name).name == "Modelfile":
            return "modelfile"
        if ext in {".pt", ".pth", ".pkl", ".onnx", ".engine", ".safetensors"}:
            return "weights"
        return "other"

    @staticmethod
    def _artifact_content_type(name: str) -> str:
        if Path(name).name == "Modelfile":
            return "text/plain"
        ext = Path(name).suffix.lower()
        return {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".csv": "text/csv",
            ".json": "application/json",
            ".jsonl": "application/x-jsonlines",
            ".yaml": "text/yaml",
            ".yml": "text/yaml",
        }.get(ext, "application/octet-stream")

    def _enumerate_run_artifacts(self, run_dir: Path) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        try:
            entries = sorted(run_dir.iterdir(), key=lambda p: p.name.lower())
        except Exception:
            return items
        for entry in entries:
            if not entry.is_file():
                # include weights/ and adapter/ subdir contents as top-level files for convenience
                if entry.is_dir() and entry.name in {"weights", "adapter"}:
                    try:
                        for sub in sorted(entry.iterdir(), key=lambda p: p.name.lower()):
                            if sub.is_file():
                                try:
                                    size = sub.stat().st_size
                                except Exception:
                                    size = 0
                                items.append({
                                    "name": f"{entry.name}/{sub.name}",
                                    "kind": self._classify_artifact(sub.name),
                                    "size_bytes": size,
                                    "content_type": self._artifact_content_type(sub.name),
                                })
                    except Exception:
                        pass
                continue
            try:
                size = entry.stat().st_size
            except Exception:
                size = 0
            items.append({
                "name": entry.name,
                "kind": self._classify_artifact(entry.name),
                "size_bytes": size,
                "content_type": self._artifact_content_type(entry.name),
            })
        return items

    @staticmethod
    def _decode_b64_image(image_b64: str) -> Optional[np.ndarray]:
        value = (image_b64 or "").strip()
        if not value:
            return None
        if "," in value and value.lower().startswith("data:"):
            value = value.split(",", 1)[1]
        try:
            raw = base64.b64decode(value)
        except Exception:
            return None
        arr = np.frombuffer(raw, dtype=np.uint8)
        if arr.size == 0:
            return None
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    @staticmethod
    def _store_job_image(job_id: str, image_bgr: np.ndarray) -> Path:
        JOB_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        path = JOB_IMAGES_DIR / f"{job_id}.jpg"
        ok = cv2.imwrite(str(path), image_bgr)
        if not ok:
            raise RuntimeError("failed to store job image")
        return path

    @staticmethod
    def _store_incoming_training_image(scenario: str, job_id: str, image_bgr: np.ndarray) -> Path:
        incoming_dir = MLOPS_ROOT / "datasets" / scenario / "incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        path = incoming_dir / f"{job_id}.jpg"
        ok = cv2.imwrite(str(path), image_bgr)
        if not ok:
            raise RuntimeError("failed to store incoming training image")
        return path

    def add_event_sink(
        self,
        sink: Callable[[dict[str, Any]], None],
        *,
        replay_since: Optional[int] = None,
    ) -> None:
        """Register an in-process subscriber for live events (e.g. the Qt UI).

        The sink is called from worker/dispatcher threads with the same payloads
        delivered to websocket clients; it is responsible for marshalling onto
        its own thread if needed.
        """
        with self._event_bus_lock:
            replay = self._events_since_locked(replay_since or 0)[0] if replay_since is not None else []
            if sink not in self._event_sinks:
                self._event_sinks.append(sink)
            for payload in replay:
                try:
                    sink(dict(payload))
                except Exception:
                    pass

    def remove_event_sink(self, sink: Callable[[dict[str, Any]], None]) -> None:
        with self._event_bus_lock:
            try:
                self._event_sinks.remove(sink)
            except ValueError:
                pass

    def _dispatch_local(self, payload: dict[str, Any]) -> None:
        with self._event_bus_lock:
            sinks = list(self._event_sinks)
        for sink in sinks:
            try:
                sink(payload)
            except Exception:
                pass

    def latest_event_seq(self) -> int:
        with self._event_bus_lock:
            return int(self._event_seq)

    def _events_since_locked(self, since: int) -> tuple[list[dict[str, Any]], bool]:
        seq = max(0, int(since or 0))
        if not self._event_log:
            return [], True
        first_seq = int(self._event_log[0].get("seq") or 0)
        complete = seq == 0 or seq >= first_seq - 1
        return [dict(item) for item in self._event_log if int(item.get("seq") or 0) > seq], complete

    def replay_events(self, since: int, sink: Callable[[dict[str, Any]], None]) -> bool:
        with self._event_bus_lock:
            replay, complete = self._events_since_locked(since)
            for payload in replay:
                try:
                    sink(dict(payload))
                except Exception:
                    pass
        return complete

    def _record_event_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._event_seq += 1
        stamped = dict(payload)
        stamped["seq"] = self._event_seq
        self._event_log.append(stamped)
        if len(self._event_log) > self._event_log_limit:
            del self._event_log[: len(self._event_log) - self._event_log_limit]
        return stamped

    def _warm_inference_stack(self) -> None:
        """Preload the torch-backed inference/export modules in the background.

        They are lazy-imported so the window appears fast; warming them a few
        seconds after startup (off the request path, daemon thread) means the
        first real inference/export does not pay the cold torch import. Opt out
        with CVOPS_WARM_INFERENCE=0."""
        if str(os.environ.get("CVOPS_WARM_INFERENCE", "1")).strip().lower() in {"0", "false", "no", "off"}:
            return
        if getattr(self, "_warm_started", False):
            return
        self._warm_started = True

        def _warm() -> None:
            try:
                self._stop.wait(3.0)  # let the UI settle before pulling in torch
                if self._stop.is_set():
                    return
                import importlib  # noqa: PLC0415
                importlib.import_module("mlops.pipeline.infer")
                importlib.import_module("mlops.pipeline.export")
            except Exception:
                pass

        threading.Thread(target=_warm, name="CvOpsWarmInfer", daemon=True).start()

    def _emit(self, payload: dict[str, Any]) -> None:
        self._mark_startup_resync_dirty()
        with self._event_bus_lock:
            payload = self._record_event_locked(payload)
            sinks = list(self._event_sinks)
        # In-process subscribers first: independent of the asyncio loop / sockets.
        for sink in sinks:
            try:
                sink(payload)
            except Exception:
                pass
        loop = self._loop
        if loop is None:
            return
        with self._ws_lock:
            has_ws_clients = bool(self._ws_clients)
        if not has_ws_clients:
            return
        coro = self._broadcast(payload)
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
            try:
                coro.close()
            except Exception:
                pass
            return

    def _health_snapshot(self) -> dict[str, Any]:
        jobs = self.store.list_jobs(limit=400)
        try:
            slots_free = self._slot_sem._value  # type: ignore[attr-defined]
        except Exception:
            slots_free = 0
        return {
            "status": "ok",
            "worker_alive": self._dispatcher.is_alive(),
            "max_workers": self._worker_threads,
            "slots_free": int(slots_free),
            "queued": sum(1 for j in jobs if j.state == "queued"),
            "running": sum(1 for j in jobs if j.state == "running"),
            "done": sum(1 for j in jobs if j.state == "done"),
            "error": sum(1 for j in jobs if j.state == "error"),
        }

    def _ecosystem_summary(self) -> dict[str, Any]:
        """Aggregate the cross-store state needed by the Ecosystem command deck."""
        health = self._health_snapshot()
        now = time.time()
        try:
            jobs = [job.to_dict() for job in self.store.list_jobs(limit=80)]
        except Exception:
            jobs = []

        active = self._scenarios_with_active_training()
        scenarios: list[dict[str, Any]] = []
        scenario_error = ""
        try:
            base = mlops_registry.list_enabled_scenarios()
        except Exception as exc:
            base = []
            scenario_error = str(exc)
        for item in base:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            try:
                status = mlops_registry.get_scenario_status(name)
                cfg = mlops_registry.get_scenario_config(name)
                status.setdefault("display_name", item.get("display_name") or cfg.display_name)
                status.setdefault("description", item.get("description") or cfg.description)
                status["backbone_type"] = status.get("backbone_type") or cfg.backbone_type
                status["base_model"] = cfg.base_model
                status["weights"] = cfg.weights
                status["hyperparams"] = dict(cfg.hyperparams or {})
                status["class_count"] = len(cfg.classes or [])
                try:
                    status["split_counts"] = mlops_registry.dataset_split_counts(name)
                except Exception:
                    status["split_counts"] = {}
                if name in active:
                    status["status"] = "training"
                scenario_jobs = [j for j in jobs if str(j.get("scenario") or "") == name]
                status["latest_job"] = scenario_jobs[0] if scenario_jobs else None
                latest_train = next(
                    (
                        j
                        for j in scenario_jobs
                        if str(j.get("job_type") or "").lower() == "train"
                    ),
                    None,
                )
                if isinstance(latest_train, dict):
                    status["training_context"] = self._training_context_for_job(latest_train)
                scenarios.append(status)
            except Exception as exc:
                scenarios.append(
                    {
                        "name": name,
                        "display_name": item.get("display_name") or name,
                        "description": item.get("description") or "",
                        "status": "error",
                        "error": str(exc),
                    }
                )

        state_counts: dict[str, int] = {}
        type_counts: dict[str, int] = {}
        last_error = ""
        for job in jobs:
            state = str(job.get("state") or "unknown").lower()
            jtype = str(job.get("job_type") or "unknown").lower()
            state_counts[state] = state_counts.get(state, 0) + 1
            type_counts[jtype] = type_counts.get(jtype, 0) + 1
            if not last_error and str(job.get("error") or "").strip():
                last_error = f"{job.get('scenario') or '-'} / {job.get('job_type') or '-'}: {job.get('error')}"

        storage: dict[str, Any] = {
            "state_dir": str(CVOPS_STATE_DIR),
            "jobs_db": str(CVOPS_DB_PATH),
            "catalog_db": str(CVOPS_CATALOG_DB_PATH),
            "cytoscape_cached": (CVOPS_STATE_DIR / "cache" / "cytoscape-3.30.2.min.js").is_file(),
        }
        try:
            usage = shutil.disk_usage(CVOPS_STATE_DIR)
            storage.update(
                {
                    "disk_total": int(usage.total),
                    "disk_used": int(usage.used),
                    "disk_free": int(usage.free),
                }
            )
        except Exception:
            pass

        return {
            "generated_at": now,
            "health": health,
            "scenarios": scenarios,
            "scenario_error": scenario_error,
            "jobs": jobs[:40],
            "job_counts": state_counts,
            "job_type_counts": type_counts,
            "storage": storage,
            "last_error": last_error,
            "active_training_scenarios": sorted(active),
        }

    def _settings_path(self) -> Path:
        return CVOPS_STATE_DIR / "settings.json"

    def _load_web_settings(self) -> CvOpsSettings:
        return load_cvops_settings(self._settings_path())

    def _settings_payload(self) -> dict[str, Any]:
        settings = self._load_web_settings()
        path = self._settings_path()
        background_path = ""
        if settings.workspace_background_asset:
            background_path = str((CVOPS_STATE_DIR / settings.workspace_background_asset).resolve())
        return {
            "settings_path": str(path),
            "dashboard_url": f"http://127.0.0.1:{int(settings.dashboard_port)}",
            "settings": {
                "color_scheme": settings.color_scheme,
                "button_shape": settings.button_shape,
                "ui_scale_pct": int(settings.ui_scale_pct),
                "time_format": settings.time_format,
                "auto_start_dashboard": bool(settings.auto_start_dashboard),
                "dashboard_port": int(settings.dashboard_port),
                "health_poll_ms": int(settings.health_poll_ms),
                "gallery_poll_ms": int(settings.gallery_poll_ms),
                "dashboard_poll_ms": int(settings.dashboard_poll_ms),
                "show_event_pulse": bool(settings.show_event_pulse),
                "custom_workspace_background": bool(settings.custom_workspace_background),
                "workspace_background_asset": settings.workspace_background_asset,
                "workspace_background_path": background_path,
            },
        }

    def _apply_settings_patch(self, req: SettingsUpdateRequest) -> dict[str, Any]:
        settings = self._load_web_settings()
        for key, value in req.model_dump(exclude_none=True).items():
            if hasattr(settings, key):
                setattr(settings, key, value)
        save_cvops_settings(self._settings_path(), settings)
        return self._settings_payload()

    def _recent_errors_payload(self) -> dict[str, Any]:
        try:
            jobs = [job.to_dict() for job in self.store.list_jobs(limit=80)]
        except Exception:
            jobs = []
        items: list[dict[str, Any]] = []
        for job in jobs:
            state = str(job.get("state") or "").lower()
            error = str(job.get("error") or "").strip()
            if state not in {"error", "failed", "cancelled", "canceled"} and not error:
                continue
            ts = float(job.get("finished_at") or job.get("started_at") or job.get("created_at") or 0.0)
            items.append(
                {
                    "timestamp": ts,
                    "source": "job",
                    "job_id": str(job.get("job_id") or ""),
                    "scenario": str(job.get("scenario") or ""),
                    "state": state or "error",
                    "message": error or f"job entered {state or 'error'} state",
                }
            )
        items.sort(key=lambda item: float(item.get("timestamp") or 0.0), reverse=True)
        return {"errors": items[:40], "count": len(items)}

    def _training_context_for_job(self, job: dict[str, Any]) -> dict[str, Any]:
        job_id = str(job.get("job_id") or "")
        payload = job.get("payload") if isinstance(job.get("payload"), dict) else {}
        events: list[dict[str, Any]] = []
        with self._train_history_lock:
            raw_events = list(self._train_history.get(job_id, []))
        for event in raw_events:
            if isinstance(event, dict):
                events.append(dict(event))

        latest_event: dict[str, Any] = {}
        latest_batch: dict[str, Any] = {}
        log_lines: list[str] = []
        for event in reversed(events):
            etype = str(event.get("event") or "").lower()
            if etype == "log":
                line = str(event.get("line") or "").strip()
                if line and len(log_lines) < 4:
                    log_lines.append(line)
                continue
            if etype == "batch_metrics" and not latest_batch:
                latest_batch = event
                continue
            if not latest_event:
                latest_event = event
            if latest_event and latest_batch and len(log_lines) >= 4:
                break

        start_event = next(
            (
                event
                for event in events
                if str(event.get("event") or "").lower() == "start"
            ),
            {},
        )
        context = {
            "job_id": job_id,
            "state": str(job.get("state") or ""),
            "trigger": str(payload.get("trigger") or "manual"),
            "update_mode": bool(payload.get("update_mode")),
            "resume": bool(payload.get("resume")),
            "save_period": payload.get("save_period"),
            "final_model_name": str(payload.get("final_model_name") or ""),
            "dataset_snapshot_id": str(start_event.get("dataset_snapshot_id") or ""),
            "dataset_snapshot_path": str(start_event.get("dataset_snapshot_path") or ""),
            "device": str(start_event.get("device") or ""),
            "trainer": str(start_event.get("trainer") or ""),
            "effective_hyperparams": start_event.get("effective_hyperparams")
            if isinstance(start_event.get("effective_hyperparams"), dict)
            else {},
            "latest_event": latest_event,
            "latest_batch": latest_batch,
            "recent_logs": list(reversed(log_lines)),
            "error": str(job.get("error") or latest_event.get("error") or ""),
        }
        return context

    def _build_heartbeat_payload(self) -> dict[str, Any]:
        health = self._health_snapshot()
        return {
            "type": "heartbeat",
            "service": "cvops",
            "state": "live" if bool(health.get("worker_alive")) else "degraded",
            "emitted_at": time.time(),
            **health,
        }

    async def _heartbeat_loop(self) -> None:
        while not self._stop.is_set():
            try:
                await asyncio.sleep(self._heartbeat_interval_s)
            except asyncio.CancelledError:
                break
            try:
                await self._broadcast(self._build_heartbeat_payload())
            except Exception:
                continue

    def _scenarios_with_active_training(self) -> set[str]:
        try:
            jobs = self.store.list_jobs(limit=200)
        except Exception:
            return set()
        return {
            j.scenario
            for j in jobs
            if j.job_type == "train" and j.state in ("queued", "running")
        }

    def _emit_scenario_updated(self, scenario: str) -> None:
        self._mark_graph_cache_dirty("scenario")
        try:
            payload = mlops_registry.get_scenario_status(scenario)
        except Exception as exc:
            payload = {"name": scenario, "status": "error", "error": str(exc)}
        if scenario in self._scenarios_with_active_training():
            payload["status"] = "training"
        self._emit({"type": "scenario_updated", "scenario": scenario, "status_payload": payload})

    def _emit_job_status(self, job: JobRecord) -> None:
        self._mark_graph_cache_dirty("job")
        self._emit(
            {
                "type": "job_status",
                "job_id": job.job_id,
                "job_type": job.job_type,
                "scenario": job.scenario,
                "state": job.state,
                "source": job.source,
                "created_at": job.created_at,
                "started_at": job.started_at,
                "finished_at": job.finished_at,
                "error": job.error,
                "cancel_requested": bool(job.cancel_requested),
            }
        )
        self._record_integration_event(job, event_type="job_status")

    def _record_integration_event(
        self,
        job: JobRecord,
        *,
        event_type: str,
        result: Optional[dict[str, Any]] = None,
    ) -> None:
        payload = {
            "type": event_type,
            "job_id": job.job_id,
            "job_type": job.job_type,
            "scenario": job.scenario,
            "state": job.state,
            "source": job.source,
            "created_at": job.created_at,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            "result_ref": job.result_ref,
            "job_payload": job.payload,
            "emitted_at": time.time(),
        }
        if isinstance(result, dict):
            payload["result"] = {
                "summary": str(result.get("summary") or ""),
                "detections_count": len(result.get("detections") or []),
                "elapsed_ms": result.get("elapsed_ms"),
                "artifact_policy": result.get("artifact_policy"),
                "error": str(result.get("error") or ""),
                "result_path": str(result.get("result_path") or ""),
                "weights": str(result.get("weights") or ""),
                "map50": str(result.get("map50") or ""),
                "map50_95": str(result.get("map50_95") or ""),
                "quality_stop": result.get("quality_stop") if isinstance(result.get("quality_stop"), dict) else {},
            }
        try:
            append_integration_event(payload)
        except Exception:
            # Integration stream must never break runtime jobs.
            return

    def _record_catalog_event(self, event_type: str, payload: dict[str, Any]) -> None:
        self._mark_graph_cache_dirty("catalog")
        body = {
            "type": str(event_type or "catalog_event"),
            "source": "cvops_catalog",
            "emitted_at": time.time(),
            "payload": payload,
        }
        for key in ("name", "asset_id", "sector_id", "sector_path", "collection_id", "source_type"):
            value = payload.get(key)
            if value not in (None, ""):
                body[key] = value
        self._emit(body)
        try:
            append_integration_event(body)
        except Exception:
            return

    def _record_training_progress(self, job_id: str, payload: dict[str, Any]) -> None:
        with self._train_history_lock:
            items = self._train_history.setdefault(job_id, [])
            items.append(dict(payload))
            # Separate caps so a flood of log lines can't evict epoch events.
            is_log = str(payload.get("event") or "") == "log"
            if is_log:
                log_count = sum(1 for it in items if str(it.get("event") or "") == "log")
                if log_count > 400:
                    drop = log_count - 400
                    new_items: list[dict[str, Any]] = []
                    skipped = 0
                    for it in items:
                        if skipped < drop and str(it.get("event") or "") == "log":
                            skipped += 1
                            continue
                        new_items.append(it)
                    items[:] = new_items
            else:
                non_log = [it for it in items if str(it.get("event") or "") != "log"]
                if len(non_log) > 500:
                    drop_non_log = len(non_log) - 500
                    new_items = []
                    skipped = 0
                    for it in items:
                        if skipped < drop_non_log and str(it.get("event") or "") != "log":
                            skipped += 1
                            continue
                        new_items.append(it)
                    items[:] = new_items
        self._mark_startup_resync_dirty()

    def _emit_training_progress(self, job: JobRecord, payload: dict[str, Any]) -> None:
        body = {
            "type": "training_progress",
            "job_id": job.job_id,
            "scenario": job.scenario,
        }
        body.update(payload)
        self._record_training_progress(job.job_id, body)
        self._emit(body)

    async def _send_ws_json(self, ws: WebSocket, payload: dict[str, Any]) -> None:
        await asyncio.wait_for(ws.send_json(payload), timeout=self._ws_send_timeout_s)

    async def _broadcast(self, payload: dict[str, Any]) -> None:
        lock = self._ws_broadcast_lock
        if lock is None:
            lock = asyncio.Lock()
            self._ws_broadcast_lock = lock
        async with lock:
            with self._ws_lock:
                clients = list(self._ws_clients)
            if not clients:
                return
            stale: list[WebSocket] = []
            for ws in clients:
                try:
                    await self._send_ws_json(ws, payload)
                except Exception:
                    stale.append(ws)
            if stale:
                with self._ws_lock:
                    for ws in stale:
                        self._ws_clients.discard(ws)

    def _dispatcher_loop(self) -> None:
        while not self._stop.is_set():
            if not self._slot_sem.acquire(timeout=0.35):
                continue
            running_trains = self.store.list_running_train_scenarios()
            exclude = running_trains if running_trains else None
            job = self.store.claim_next_queued_job(exclude_scenarios_with_running=exclude)
            if job is None:
                self._slot_sem.release()
                self._stop.wait(0.12)
                continue
            self._emit_job_status(job)
            try:
                self._executor.submit(self._run_job_with_slot, job)
            except Exception as exc:
                self._slot_sem.release()
                try:
                    self.store.set_job_state(job.job_id, "error", error=f"executor submit failed: {exc}")
                except Exception:
                    pass

    def _run_job_with_slot(self, job: JobRecord) -> None:
        try:
            self._execute_job(job)
        finally:
            try:
                self._slot_sem.release()
            except Exception:
                pass

    @staticmethod
    def _run_version_from_result(result: dict[str, Any]) -> str:
        run_version = str(result.get("run_version") or "").strip()
        if run_version:
            return run_version
        run_dir = str(result.get("output") or result.get("result_path") or "").strip()
        if run_dir:
            return Path(run_dir).name
        weights = str(result.get("weights") or "").strip()
        if weights:
            return Path(weights).parent.name
        return ""

    def _apply_ci_cd_after_training(self, job: JobRecord, result: dict[str, Any]) -> dict[str, Any]:
        try:
            policy = mlops_registry.get_scenario_ci_cd_policy(job.scenario)
        except Exception as exc:
            result["ci_cd"] = {"enabled": False, "error": str(exc)}
            return result
        if not bool(policy.get("enabled")) or str(result.get("error") or ""):
            return result
        run_version = self._run_version_from_result(result)
        if not run_version:
            result["ci_cd"] = {
                "enabled": True,
                "gate_status": "failed",
                "error": "unable to resolve run version for CI/CD gate",
            }
            return result
        try:
            report = evaluate_run_gate(job.scenario, run_version, policy=policy, update_registry=True)
            result["ci_cd"] = report
            self._emit_training_progress(
                job,
                {
                    "event": "ci_cd_gate",
                    "run_version": run_version,
                    "gate_status": str(report.get("gate_status") or ""),
                    "passed": bool(report.get("passed")),
                    "report_path": str(report.get("report_path") or ""),
                    "timestamp": time.time(),
                },
            )
            if bool(report.get("passed")) and str(policy.get("promotion") or "manual") == "auto":
                # Automation stages the challenger; a human promotes staging ->
                # prod (which overwrites the live serving weights). This keeps
                # "auto" safe and makes prod promotion a deliberate act.
                promotion = promote_run(
                    job.scenario,
                    run_version,
                    target_alias="staging",
                    actor="cvops:auto",
                    reason="CI/CD auto promotion to staging",
                    override=False,
                )
                result["ci_cd_promotion"] = promotion
        except Exception as exc:
            result["ci_cd"] = {
                "enabled": True,
                "gate_status": "failed",
                "error": str(exc),
            }
        return result

    def _execute_job(self, job: JobRecord) -> None:
        try:
            if str(job.job_type or "").startswith("archive_"):
                result = self._run_archive_job(job)
            elif job.job_type == "infer":
                result = self._run_infer_job(job)
            else:
                result = self._run_train_job(job)
            # Attach backbone_type so the UI can render results appropriately.
            if isinstance(result, dict) and "backbone_type" not in result:
                try:
                    cfg = mlops_registry.get_scenario_config(job.scenario)
                    result["backbone_type"] = str(cfg.backbone_type or "yolo_detection")
                except Exception:
                    result["backbone_type"] = "yolo_detection"
            if (
                str(job.job_type or "").startswith("archive_")
                and not str(job.scenario or "").startswith("archive:")
                and not str(result.get("error") or "")
            ):
                try:
                    mlops_registry.patch_scenario_backbone_config(
                        job.scenario,
                        {
                            "corpus_id": str(result.get("corpus_id") or ""),
                            "dataset_version_id": str(result.get("dataset_version_id") or ""),
                            "latest_snapshot_id": str(result.get("snapshot_id") or ""),
                        },
                    )
                except Exception:
                    pass
            if job.job_type == "train" and not str(result.get("error") or ""):
                result = self._apply_ci_cd_after_training(job, result)
            self.store.write_result(job.job_id, result)
            state = "error" if result.get("error") else "done"
            # For training/update jobs, result_ref is the model version_id so
            # the ecosystem graph can draw a produces edge without reading results.
            result_ref_val = job.job_id
            if str(job.job_type or "").startswith("archive_") and not result.get("error"):
                result_ref_val = str(result.get("snapshot_id") or job.job_id)
            elif job.job_type in ("train", "update"):
                explicit_ref = str(
                    result.get("model_version_id") or result.get("model_version") or ""
                ).strip()
                if explicit_ref:
                    result_ref_val = explicit_ref
                elif not result.get("error"):
                    _scen = str(result.get("scenario") or job.scenario or "").strip()
                    _out  = str(result.get("output") or result.get("result_path") or "").strip()
                    if not _out:
                        _weights = str(result.get("weights") or "").strip()
                        _out = str(Path(_weights).parent) if _weights else ""
                    _rv   = Path(_out).name if _out else ""
                    if _scen and _rv:
                        result_ref_val = f"{_scen}:{_rv}"
            updated = self.store.set_job_state(
                job.job_id,
                state,
                error=str(result.get("error") or ""),
                result_ref=result_ref_val,
            )
            self._emit_job_status(updated)
            self._emit(
                {
                    "type": "job_result",
                    "job_id": job.job_id,
                    "job_type": job.job_type,
                    "scenario": job.scenario,
                    "state": updated.state,
                    "error": str(result.get("error") or ""),
                    "result": result,
                }
            )
            self._record_integration_event(updated, event_type="job_result", result=result)
            if job.job_type == "train":
                self._emit_scenario_updated(job.scenario)
            elif str(job.job_type or "").startswith("archive_") and not str(job.scenario or "").startswith("archive:"):
                self._emit_scenario_updated(job.scenario)
        except Exception as exc:
            updated = self.store.set_job_state(job.job_id, "error", error=str(exc), result_ref=job.job_id)
            self._emit_job_status(updated)
            if job.job_type == "train":
                self._emit_scenario_updated(job.scenario)
            elif str(job.job_type or "").startswith("archive_") and not str(job.scenario or "").startswith("archive:"):
                self._emit_scenario_updated(job.scenario)
            self._record_integration_event(
                updated,
                event_type="job_result",
                result={"summary": "", "detections": [], "elapsed_ms": 0, "artifact_policy": "", "error": str(exc)},
            )

    def _emit_cell_progress(self, job: JobRecord, payload: dict[str, Any]) -> None:
        self._emit(
            {
                "type": "cell_progress",
                "job_id": job.job_id,
                "scenario": job.scenario,
                "job_type": job.job_type,
                **payload,
            }
        )

    def _run_infer_job(self, job: JobRecord) -> dict[str, Any]:
        if self.store.was_cancel_requested(job.job_id):
            return {
                "job_id": job.job_id,
                "scenario": job.scenario,
                "summary": "",
                "detections": [],
                "error": "cancelled before inference",
                "artifact_policy": "inline_overlay_optional",
            }
        image = cv2.imread(job.image_path)
        if image is None:
            return {
                "job_id": job.job_id,
                "scenario": job.scenario,
                "summary": "",
                "detections": [],
                "error": f"unable to read image: {job.image_path}",
                "artifact_policy": "inline_overlay_optional",
            }
        payload = job.payload if isinstance(job.payload, dict) else {}
        version = str(payload.get("version") or "")

        def _cell_cb(cell_payload: dict[str, Any]) -> None:
            self._emit_cell_progress(job, cell_payload)

        # Dispatch to backbone if scenario isn't YOLO. This enables inference for
        # additional backbones like face_recognition.
        try:
            from mlops.pipeline.registry import get_scenario_config as _gsc
            _cfg = _gsc(job.scenario)
            _backbone_type = str(_cfg.backbone_type or "yolo_detection")
        except Exception:
            _cfg = None
            _backbone_type = "yolo_detection"

        if _backbone_type != "yolo_detection" and _cfg is not None:
            from mlops.pipeline.backbone import BackboneContext
            from mlops.pipeline.backbones import get_backbone
            from dataclasses import replace as _replace
            from pathlib import Path as _Path

            override = payload.get("backbone_config_override")
            if isinstance(override, dict) and override:
                merged = dict(getattr(_cfg, "backbone_config", {}) or {})
                merged.update(dict(override))
                _cfg = _replace(_cfg, backbone_config=merged)

            backbone = get_backbone(_backbone_type, _cfg)
            ctx = BackboneContext(
                scenario_config=_cfg,
                job_id=job.job_id,
                job_type="infer",
                image_bgr=image,
                payload=payload,
                cell_callback=_cell_cb,
            )
            result = backbone.run(ctx)
            out = dict(result) if isinstance(result, dict) else {}
            out["job_id"] = job.job_id
            out["scenario"] = job.scenario
            out.setdefault("artifact_policy", "inline_overlay_optional")
            out["summary"] = str(out.get("summary") or "inference completed")
            out["error"] = str(out.get("error") or "")
            weights = str(out.get("weights") or out.get("weights_path") or "").strip()
            out["weights"] = weights
            result_path = str(out.get("result_path") or "").strip()
            if not result_path and weights:
                p = _Path(weights)
                if not p.is_absolute():
                    p = (ROOT_DIR / p).resolve()
                result_path = str(p.parent)
            out["result_path"] = result_path
            out["raw"] = dict(result) if isinstance(result, dict) else {"result": result}
            return out

        from mlops.pipeline import infer as mlops_infer  # lazy: pulls torch
        result = mlops_infer.run_scenario(
            job.scenario, image, version=version,
            payload_extra={
                "infer_overrides": payload.get("infer_overrides") if isinstance(payload.get("infer_overrides"), dict) else {},
                "weights_path": str(payload.get("weights_path") or ""),
                "model_artifact": str(payload.get("model_artifact") or ""),
            },
            cell_callback=_cell_cb, job_id=job.job_id,
        )
        signal = result.get("signal") if isinstance(result.get("signal"), dict) else {}
        return {
            "job_id": job.job_id,
            "scenario": job.scenario,
            "model_version": str(result.get("model_version") or version),
            "weights": str(result.get("weights") or payload.get("weights_path") or ""),
            "summary": str(signal.get("summary") or ""),
            "detections": result.get("detections", []),
            "elapsed_ms": result.get("elapsed_ms", 0),
            "overlay_image": result.get("overlay_image", ""),
            "error": str(result.get("error") or ""),
            "artifact_policy": "inline_overlay_optional",
            "raw": result,
        }

    def _run_archive_job(self, job: JobRecord) -> dict[str, Any]:
        payload = job.payload if isinstance(job.payload, dict) else {}
        corpus_id = str(payload.get("corpus_id") or "").strip()
        dataset_version_id = str(payload.get("dataset_version_id") or "").strip()
        phase = str(payload.get("phase") or job.job_type or "").strip()
        parent_snapshot_id = str(payload.get("parent_snapshot_id") or "").strip()
        if not corpus_id or not dataset_version_id:
            return {
                "job_id": job.job_id,
                "scenario": job.scenario,
                "summary": "",
                "error": "archive jobs require corpus_id and dataset_version_id",
                "artifact_policy": "path_only",
                "backbone_type": "archival_ingestion",
            }

        def _cell_cb(cell_payload: dict[str, Any]) -> None:
            self._emit_cell_progress(job, cell_payload)

        run_dir: Optional[Path] = None
        if bool(payload.get("write_run_artifacts", True)):
            run_dir = (MLOPS_ROOT / "models" / "archive_jobs" / job.job_id).resolve()

        result = self._archive_engine().run_archive_job(
            self.archives,
            corpus_id=corpus_id,
            dataset_version_id=dataset_version_id,
            phase=phase,
            parent_snapshot_id=parent_snapshot_id,
            provider_config=dict(payload.get("provider_config") or {}) if isinstance(payload.get("provider_config"), dict) else None,
            job_id=job.job_id,
            cell_callback=_cell_cb,
            write_run_artifacts=run_dir is not None,
            artifact_root=run_dir,
        )
        result["job_id"] = job.job_id
        result["scenario"] = job.scenario
        return result

    def _run_yolo_train_subprocess(
        self,
        job: JobRecord,
        spec: dict[str, Any],
        on_progress: "Any",
    ) -> dict[str, Any]:
        """Run YOLO training in a subprocess so cancel can SIGKILL the group.

        `spec` is the JSON-serializable kwargs dict for `run_training`. Progress
        events tunnel back as `__CVOPS_EVT__{json}` lines on the worker's
        stdout; everything else is forwarded as a log event so ultralytics'
        own stdout still surfaces in the training console.
        """
        EVENT_MARKER = "__CVOPS_EVT__"
        tmpdir = Path(tempfile.mkdtemp(prefix=f"cvops_train_{job.job_id}_"))
        input_path = tmpdir / "input.json"
        output_path = tmpdir / "output.json"
        try:
            input_path.write_text(json.dumps(spec, default=str), encoding="utf-8")
        except Exception as exc:
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError(f"failed to stage training spec: {exc}") from exc

        env = os.environ.copy()
        pypath = env.get("PYTHONPATH", "")
        root_str = str(ROOT_DIR)
        if root_str not in pypath.split(os.pathsep):
            env["PYTHONPATH"] = root_str + (os.pathsep + pypath if pypath else "")
        env["PYTHONUNBUFFERED"] = "1"

        argv = [
            sys.executable,
            "-u",
            "-m",
            "mlops.pipeline.train_worker",
            "--input-json",
            str(input_path),
            "--output-json",
            str(output_path),
        ]
        # start_new_session=True puts the worker (and any children it spawns,
        # e.g. dataloader workers) into a fresh process group so killpg cleans
        # everything in one shot.
        popen = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROOT_DIR),
            env=env,
            start_new_session=True,
            bufsize=1,
            text=True,
        )
        with self._train_procs_lock:
            self._train_procs[job.job_id] = popen

        cancelled = threading.Event()
        watchdog_stop = threading.Event()

        def _watchdog() -> None:
            while not watchdog_stop.is_set():
                if popen.poll() is not None:
                    return
                try:
                    if self.store.was_cancel_requested(job.job_id):
                        if popen.poll() is not None:
                            return
                        cancelled.set()
                        try:
                            os.killpg(os.getpgid(popen.pid), signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        except Exception:
                            try:
                                popen.kill()
                            except Exception:
                                pass
                        return
                except Exception:
                    pass
                watchdog_stop.wait(0.2)

        wd = threading.Thread(
            target=_watchdog, name=f"TrainCancelWatchdog-{job.job_id}", daemon=True
        )
        wd.start()

        try:
            assert popen.stdout is not None
            for raw_line in popen.stdout:
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                if line.startswith(EVENT_MARKER):
                    try:
                        evt = json.loads(line[len(EVENT_MARKER):])
                    except Exception:
                        evt = {
                            "event": "log",
                            "line": line,
                            "stream": "stdout",
                            "timestamp": time.time(),
                        }
                    try:
                        on_progress(evt)
                    except Exception:
                        pass
                else:
                    try:
                        on_progress(
                            {
                                "event": "log",
                                "line": line,
                                "stream": "stdout",
                                "timestamp": time.time(),
                            }
                        )
                    except Exception:
                        pass
            return_code = popen.wait()
        finally:
            watchdog_stop.set()
            wd.join(timeout=0.5)
            with self._train_procs_lock:
                self._train_procs.pop(job.job_id, None)

        if cancelled.is_set():
            shutil.rmtree(tmpdir, ignore_errors=True)
            raise RuntimeError("training cancelled by operator")

        if output_path.exists():
            try:
                summary = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:
                shutil.rmtree(tmpdir, ignore_errors=True)
                raise RuntimeError(f"worker summary unreadable: {exc}") from exc
        else:
            summary = {}

        shutil.rmtree(tmpdir, ignore_errors=True)

        if isinstance(summary, dict) and summary.get("error"):
            raise RuntimeError(str(summary.get("error")))
        if return_code != 0:
            raise RuntimeError(
                f"training worker exited with code {return_code}"
            )
        if not isinstance(summary, dict):
            raise RuntimeError("training worker returned no summary")
        return summary

    def _training_events_for_job(self, job_id: str) -> list[dict[str, Any]]:
        with self._train_history_lock:
            return [
                dict(event)
                for event in self._train_history.get(job_id, [])
                if isinstance(event, dict)
            ]

    def _training_failure_status(self, job: JobRecord, reason: str) -> str:
        text = str(reason or "").lower()
        try:
            was_cancelled = bool(self.store.was_cancel_requested(job.job_id))
        except Exception:
            was_cancelled = False
        if was_cancelled or "cancelled" in text or "canceled" in text:
            return "canceled"
        return "interrupted"

    @staticmethod
    def _json_dict(path: Path) -> dict[str, Any]:
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @staticmethod
    def _event_metric(events: list[dict[str, Any]], key: str) -> Any:
        for event in reversed(events):
            value = event.get(key)
            if value not in (None, ""):
                return value
        return ""

    @staticmethod
    def _latest_training_point(events: list[dict[str, Any]]) -> dict[str, Any]:
        for event in reversed(events):
            if str(event.get("event") or "") not in {"log", "log_batch"}:
                return dict(event)
        return {}

    @staticmethod
    def _first_training_start(events: list[dict[str, Any]]) -> dict[str, Any]:
        for event in events:
            if str(event.get("event") or "") == "start":
                return dict(event)
        return {}

    @staticmethod
    def _parse_base_model_from_events(events: list[dict[str, Any]]) -> str:
        prefix = "[trainer] base model:"
        for event in events:
            if str(event.get("event") or "") != "log":
                continue
            line = str(event.get("line") or "").strip()
            if line.lower().startswith(prefix):
                return line.split(":", 1)[1].strip()
        return ""

    def _scenario_model_root(self, scenario: str) -> Path:
        try:
            cfg = mlops_registry.get_scenario_config(scenario)
            name = str(getattr(cfg, "name", "") or scenario).strip() or scenario
        except Exception:
            name = scenario
        return (MLOPS_ROOT / "models" / name).resolve()

    def _latest_unfinished_run_dir(self, job: JobRecord) -> Optional[Path]:
        root = self._scenario_model_root(job.scenario)
        try:
            runs = [p for p in root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
        except Exception:
            return None
        if not runs:
            return None
        runs.sort(key=lambda p: int(p.name[1:]), reverse=True)
        latest = runs[0]
        metrics = self._json_dict(latest / "metrics.json")
        marker = str(metrics.get("run_status") or metrics.get("status") or "").strip().lower()
        if not metrics:
            return latest
        existing_job_id = str(metrics.get("job_id") or "").strip()
        if (
            marker in {"partial", "canceled", "cancelled", "interrupted", "error", "failed"}
            and (not existing_job_id or existing_job_id == job.job_id)
        ):
            return latest
        return None

    def _next_partial_run_dir(self, scenario: str) -> Path:
        root = self._scenario_model_root(scenario)
        latest = 0
        try:
            runs = [p for p in root.glob("v*") if p.is_dir() and p.name[1:].isdigit()]
            if runs:
                latest = max(int(p.name[1:]) for p in runs)
        except Exception:
            latest = 0
        return root / f"v{latest + 1}"

    def _run_dir_from_training_events(self, job: JobRecord, events: list[dict[str, Any]]) -> Path:
        for event in reversed(events):
            value = str(event.get("run_dir") or "").strip()
            if value:
                path = Path(value)
                return path if path.is_absolute() else (MLOPS_ROOT / path).resolve()
        for event in reversed(events):
            line = str(event.get("line") or "").strip()
            if "run_dir=" not in line:
                continue
            value = line.split("run_dir=", 1)[1].strip()
            if value:
                path = Path(value)
                return path if path.is_absolute() else (MLOPS_ROOT / path).resolve()
        unfinished = self._latest_unfinished_run_dir(job)
        if unfinished is not None:
            return unfinished
        return self._next_partial_run_dir(job.scenario)

    @staticmethod
    def _first_existing_training_artifact(run_dir: Path) -> tuple[str, str]:
        candidates = (
            run_dir / "weights.pt",
            run_dir / "weights" / "best.pt",
            run_dir / "weights" / "last.pt",
            run_dir / "weights.pth",
            run_dir / "model.pkl",
        )
        for path in candidates:
            try:
                if path.is_file() and path.stat().st_size > 0:
                    return str(path), str(path)
            except Exception:
                continue
        return "", ""

    def _record_interrupted_yolo_train(
        self,
        job: JobRecord,
        *,
        reason: str,
        status: str,
    ) -> dict[str, Any]:
        events = self._training_events_for_job(job.job_id)
        run_dir = self._run_dir_from_training_events(job, events)
        run_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = run_dir / "metrics.json"
        existing = self._json_dict(metrics_path)
        now = datetime.now(timezone.utc).isoformat()
        start_event = self._first_training_start(events)
        latest_event = self._latest_training_point(events)
        start_ts = 0.0
        for value in (
            start_event.get("timestamp"),
            job.started_at,
            job.created_at,
        ):
            try:
                start_ts = float(value)
            except Exception:
                start_ts = 0.0
            if start_ts > 0:
                break
        stopped_ts = time.time()
        duration_seconds = max(0.0, stopped_ts - start_ts) if start_ts > 0 else 0.0
        start_iso = datetime.fromtimestamp(start_ts, timezone.utc).isoformat() if start_ts > 0 else ""
        base_model = str(existing.get("base_model") or "").strip()
        if not base_model:
            base_model = self._parse_base_model_from_events(events)
        map50 = existing.get("map50", self._event_metric(events, "map50"))
        map50_95 = existing.get("map50_95", self._event_metric(events, "map50_95"))
        weights_path, checkpoint_path = self._first_existing_training_artifact(run_dir)
        data_yaml_path = run_dir / "data.generated.yaml"

        payload = dict(existing)
        payload.update(
            {
                "scenario": job.scenario,
                "status": status,
                "run_status": status,
                "partial": True,
                "cancelled": status == "canceled",
                "interrupted": status != "canceled",
                "error": str(reason or ""),
                "job_id": job.job_id,
                "trained_at": str(existing.get("trained_at") or now),
                "stopped_at": now,
                "training_started_at": start_iso,
                "training_finished_at": now,
                "training_duration_seconds": duration_seconds,
                "run_dir": str(run_dir),
                "base_model": base_model,
                "final_model_name": str(
                    (job.payload if isinstance(job.payload, dict) else {}).get("final_model_name")
                    or existing.get("final_model_name")
                    or ""
                ),
                "data_yaml": str(data_yaml_path) if data_yaml_path.is_file() else "",
                "weights": weights_path,
                "checkpoint": checkpoint_path,
                "save_period": start_event.get("save_period", existing.get("save_period", "")),
                "dataset_snapshot_id": str(
                    start_event.get("dataset_snapshot_id")
                    or existing.get("dataset_snapshot_id")
                    or ""
                ),
                "dataset_snapshot_path": str(
                    start_event.get("dataset_snapshot_path")
                    or existing.get("dataset_snapshot_path")
                    or ""
                ),
                "progress": latest_event.get("progress", existing.get("progress", "")),
                "epoch": latest_event.get("epoch", existing.get("epoch", "")),
                "epochs": latest_event.get("epochs", existing.get("epochs", "")),
                "metrics": {
                    **(
                        existing.get("metrics")
                        if isinstance(existing.get("metrics"), dict)
                        else {}
                    ),
                    **{
                        key: value
                        for key, value in latest_event.items()
                        if key
                        in {
                            "event",
                            "epoch",
                            "epochs",
                            "progress",
                            "map50",
                            "map50_95",
                            "precision",
                            "recall",
                            "loss",
                            "box_loss",
                            "cls_loss",
                            "dfl_loss",
                            "verdict",
                            "verdict_reason",
                        }
                    },
                },
                "training_history_tail": events[-25:],
            }
        )
        if map50 not in (None, ""):
            payload["map50"] = map50
        if map50_95 not in (None, ""):
            payload["map50_95"] = map50_95
        try:
            metrics_path.write_text(
                json.dumps(payload, indent=2, ensure_ascii=True, default=str),
                encoding="utf-8",
            )
        except Exception:
            pass

        run_version = run_dir.name
        version_id = f"{job.scenario}:{run_version}" if run_version else ""
        try:
            entry = register_model_version(
                scenario=job.scenario,
                run_version=run_version,
                initial_status=status,
                set_candidate=False,
                artifacts={
                    "run_dir": str(run_dir),
                    "metrics_path": str(metrics_path),
                    "weights": weights_path,
                    "checkpoint": checkpoint_path,
                    "data_yaml": str(data_yaml_path) if data_yaml_path.is_file() else "",
                    "partial": True,
                },
                lineage={
                    "source": "cvops_training_interrupt",
                    "job_id": job.job_id,
                    "dataset_snapshot_id": str(payload.get("dataset_snapshot_id") or ""),
                    "dataset_snapshot_path": str(payload.get("dataset_snapshot_path") or ""),
                    "base_model": base_model,
                },
                metrics={
                    "status": status,
                    "error": str(reason or ""),
                    "map50": payload.get("map50", ""),
                    "map50_95": payload.get("map50_95", ""),
                    "training_duration_seconds": duration_seconds,
                    "partial": True,
                },
            )
            version_id = str(entry.get("version_id") or version_id)
        except Exception:
            pass

        return {
            "run_dir": str(run_dir),
            "run_version": run_version,
            "model_version_id": version_id,
            "metrics_path": str(metrics_path),
            "weights": weights_path,
            "data_yaml": str(data_yaml_path) if data_yaml_path.is_file() else "",
            "map50": payload.get("map50", ""),
            "map50_95": payload.get("map50_95", ""),
            "training_duration_seconds": payload.get("training_duration_seconds", ""),
            "status": status,
        }

    def _run_train_job(self, job: JobRecord) -> dict[str, Any]:
        try:
            with self._train_history_lock:
                self._train_history[job.job_id] = []

            payload = job.payload if isinstance(job.payload, dict) else {}

            # Dispatch to backbone if scenario is not YOLO detection; otherwise use
            # the existing ultralytics YOLO training path.
            try:
                from mlops.pipeline.registry import get_scenario_config as _gsc
                _cfg = _gsc(job.scenario)
                _backbone_type = str(_cfg.backbone_type or "yolo_detection")
            except Exception:
                _backbone_type = "yolo_detection"

            if _backbone_type != "yolo_detection":
                from mlops.pipeline.backbone import BackboneContext
                from mlops.pipeline.backbones import get_backbone
                from dataclasses import replace as _replace
                from pathlib import Path as _Path

                def _cell_cb(cell_payload: dict[str, Any]) -> None:
                    self._emit_cell_progress(job, cell_payload)

                override = payload.get("backbone_config_override")
                if isinstance(override, dict) and override:
                    merged = dict(getattr(_cfg, "backbone_config", {}) or {})
                    merged.update(dict(override))
                    _cfg = _replace(_cfg, backbone_config=merged)

                backbone = get_backbone(_backbone_type, _cfg)
                ctx = BackboneContext(
                    scenario_config=_cfg,
                    job_id=job.job_id,
                    job_type="train",
                    image_bgr=None,
                    payload=payload,
                    cell_callback=_cell_cb,
                )
                result = backbone.run(ctx)
                out = dict(result) if isinstance(result, dict) else {}
                out["job_id"] = job.job_id
                out["scenario"] = job.scenario
                out.setdefault("artifact_policy", "path_only")
                out["summary"] = str(out.get("summary") or "training completed")
                out["error"] = str(out.get("error") or "")
                weights = str(out.get("weights") or out.get("weights_path") or "").strip()
                out["weights"] = weights
                # Ensure result_path points at the run directory so /jobs/{id}/artifacts works.
                result_path = str(out.get("result_path") or "").strip()
                if not result_path and weights:
                    p = _Path(weights)
                    if not p.is_absolute():
                        p = (ROOT_DIR / p).resolve()
                    result_path = str(p.parent)
                out["result_path"] = result_path
                return out

            # YOLO detection — original training path.
            def _on_progress(progress_payload: dict[str, Any]) -> None:
                self._emit_training_progress(job, progress_payload)

            resume_training = bool(payload.get("resume", True))
            auto_fresh_on_completed_resume = bool(payload.get("auto_fresh_on_completed_resume", True))
            save_period_raw = payload.get("save_period", 1)
            try:
                save_period = int(save_period_raw)
            except Exception:
                save_period = 1
            if save_period <= 0:
                save_period = 1

            hpo: dict[str, Any] = {}
            for key in ("training_assets_root", "asset_save_root", "save_root"):
                v = str(payload.get(key) or "").strip()
                if v:
                    hpo["training_assets_root"] = v
                    break
            dev_override = str(payload.get("device") or "").strip()
            if dev_override:
                hpo["device"] = dev_override
            train_spec: dict[str, Any] = {
                "scenario": job.scenario,
                "checkpoint_period_override": save_period,
                "resume": resume_training,
                "auto_fresh_on_completed_resume": auto_fresh_on_completed_resume,
                "base_model_override": str(payload.get("base_model_override") or "") or None,
                "final_model_name": str(payload.get("final_model_name") or ""),
                "hyperparams_overrides": hpo if hpo else None,
            }
            summary = self._run_yolo_train_subprocess(job, train_spec, _on_progress)
            map50 = str(summary.get("map50") or "").strip()
            resumed_from = str(summary.get("resumed_from") or "").strip()
            if resumed_from:
                summary_text = "training run resumed and completed"
            else:
                summary_text = "training run completed"
            if map50:
                summary_text += f" (map50={map50})"
            quality_stop = summary.get("quality_stop") if isinstance(summary.get("quality_stop"), dict) else {}
            if bool(quality_stop.get("triggered")):
                mode = str(quality_stop.get("mode") or "")
                reason = str(quality_stop.get("reason") or "").strip()
                metric = str(quality_stop.get("metric") or "quality")
                value = quality_stop.get("value")
                if mode == "regression" and reason:
                    summary_text += f" [regression guard: {reason}]"
                elif mode == "time_budget" and reason:
                    summary_text += f" [time budget: {reason}]"
                elif isinstance(value, (int, float)):
                    summary_text += f" [{metric} target reached at {value:.4f}]"
                else:
                    summary_text += f" [{reason or f'{metric} target reached'}]"
            verdict = str(quality_stop.get("verdict") or "").strip()
            if verdict and bool(quality_stop.get("attempt_mode")):
                peak = quality_stop.get("verdict_peak_value")
                thr = quality_stop.get("verdict_threshold")
                vmetric = str(quality_stop.get("verdict_metric") or "quality")
                if isinstance(peak, (int, float)) and isinstance(thr, (int, float)):
                    summary_text += (
                        f" [attempt={verdict}: {vmetric} peak {peak:.4f} vs threshold {thr:.4f}]"
                    )
                else:
                    summary_text += f" [attempt={verdict}]"
            return {
                "job_id": job.job_id,
                "scenario": job.scenario,
                "run_version": summary.get("run_version", ""),
                "summary": summary_text,
                "result_path": summary.get("output", ""),
                "weights": summary.get("weights", ""),
                "data_yaml": summary.get("data_yaml", ""),
                "map50": summary.get("map50", ""),
                "map50_95": summary.get("map50_95", ""),
                "training_duration_seconds": summary.get("training_duration_seconds", ""),
                "quality_stop": quality_stop,
                "final_model_name": summary.get("final_model_name", ""),
                "final_model_file": summary.get("final_model_file", ""),
                "resumed_from": resumed_from,
                "save_period": summary.get("save_period", ""),
                "error": "",
                "artifact_policy": "path_only",
            }
        except Exception as exc:
            backbone_type = str(locals().get("_backbone_type") or "yolo_detection")
            status = (
                self._training_failure_status(job, str(exc))
                if backbone_type == "yolo_detection"
                else "error"
            )
            partial: dict[str, Any] = {}
            try:
                if backbone_type == "yolo_detection":
                    partial = self._record_interrupted_yolo_train(
                        job,
                        reason=str(exc),
                        status=status,
                    )
            except Exception:
                partial = {}
            event_name = (
                "canceled"
                if status == "canceled"
                else ("interrupted" if status == "interrupted" else "failed")
            )
            self._emit_training_progress(
                job,
                {
                    "event": event_name,
                    "epoch": -1,
                    "epochs": 0,
                    "progress": 0.0,
                    "error": str(exc),
                    "status": status,
                    "run_dir": str(partial.get("run_dir") or ""),
                    "timestamp": time.time(),
                },
            )
            if status == "canceled":
                summary = "training run canceled"
            elif status == "interrupted":
                summary = "training run interrupted"
            else:
                summary = ""
            return {
                "job_id": job.job_id,
                "scenario": job.scenario,
                "summary": summary,
                "status": status,
                "result_path": str(partial.get("run_dir") or ""),
                "weights": str(partial.get("weights") or ""),
                "data_yaml": str(partial.get("data_yaml") or ""),
                "map50": partial.get("map50", ""),
                "map50_95": partial.get("map50_95", ""),
                "training_duration_seconds": partial.get("training_duration_seconds", ""),
                "model_version_id": str(partial.get("model_version_id") or ""),
                "run_version": str(partial.get("run_version") or ""),
                "metrics_path": str(partial.get("metrics_path") or ""),
                "partial": bool(partial),
                "error": str(exc),
                "artifact_policy": "path_only",
            }


class CvOpsServerHandle:
    def __init__(self, host: str, port: int, db_path: Path = CVOPS_DB_PATH) -> None:
        self.host = host
        self.port = port
        self.service = CvOpsService(db_path=db_path)
        self.config = uvicorn.Config(
            self.service.app,
            host=host,
            port=port,
            log_level="warning",
            ws_ping_interval=None,
            ws_ping_timeout=None,
        )
        self.server = uvicorn.Server(self.config)
        self.thread = threading.Thread(target=self.server.run, daemon=True, name="CvOpsApi")

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def stop(self, timeout: float = 1.5) -> None:
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=timeout)


def run_service(host: str = "127.0.0.1", port: int = 8787) -> None:
    server = CvOpsServerHandle(host=host, port=port)
    server.start()
    try:
        while server.thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
