"""Draft custom-code cells under ``mlops/custom_cells/<scenario>/draft/``."""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml

from . import registry as _reg
from .registry import get_scenario_config, sanitize_scenario_name


def _sanitize_cell_id(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "").strip()).strip("_")
    return (s[:64] or "cell").lower()


def _sanitize_template_name(raw: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "").strip()).strip("_-")
    if not s:
        raise ValueError("template_name is required")
    return s[:80].lower()


def _safe_internal_asset_path(raw: str, fallback: str) -> Path:
    """Sanitize a cell-local asset path while preserving simple subfolders."""
    text = str(raw or "").strip().replace("\\", "/").lstrip("/")
    parts: list[str] = []
    for part in text.split("/"):
        piece = re.sub(r"[^A-Za-z0-9._ -]+", "_", part).strip(" .")
        if not piece or piece in {".", ".."}:
            continue
        parts.append(piece[:80])
    if not parts:
        parts = [fallback]
    return Path(*parts)


def draft_dir(scenario: str) -> Path:
    scen = sanitize_scenario_name(str(scenario or ""))
    return _reg.MLOPS_ROOT / "custom_cells" / scen / "draft"


def draft_manifest_path(scenario: str) -> Path:
    return draft_dir(scenario) / "manifest.json"


def cell_script_path(scenario: str, cell_id: str) -> Path:
    cid = _sanitize_cell_id(cell_id)
    return draft_dir(scenario) / f"cell_{cid}.py"


def pasted_data_dir(scenario: str, cell_id: str) -> Path:
    cid = _sanitize_cell_id(cell_id)
    return draft_dir(scenario) / "data" / cid


def read_draft(scenario: str) -> dict[str, Any]:
    scen = sanitize_scenario_name(str(scenario or ""))
    root = draft_dir(scen)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"scenario": scen, "cells": [], "scenario_datasets": []}
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return {"scenario": scen, "cells": [], "scenario_datasets": []}
    if not isinstance(data, dict):
        return {"scenario": scen, "cells": [], "scenario_datasets": []}
    cells_raw = data.get("cells")
    cells: list[dict[str, Any]] = []
    if isinstance(cells_raw, list):
        for item in cells_raw:
            if not isinstance(item, dict):
                continue
            cell = dict(item)
            path = str(cell.get("path") or "").strip()
            code = ""
            if path:
                try:
                    p = (_reg.REPO_ROOT / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
                    if p.is_file():
                        code = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    code = ""
            cell["code"] = code
            cell_id = _sanitize_cell_id(str(cell.get("id") or cell.get("name") or Path(path).stem or "cell"))
            data_root = pasted_data_dir(scen, cell_id).resolve()
            pasted_files: list[dict[str, Any]] = []
            for d in cell.get("datasets") or []:
                if not isinstance(d, dict):
                    continue
                if str(d.get("mode") or "") != "managed_copy":
                    continue
                data_path = str(d.get("path") or "").strip()
                if not data_path:
                    continue
                try:
                    asset_path = (
                        (_reg.REPO_ROOT / data_path).resolve()
                        if not Path(data_path).is_absolute()
                        else Path(data_path).resolve()
                    )
                except Exception:
                    continue
                if not asset_path.is_file():
                    continue
                try:
                    rel_name = asset_path.relative_to(data_root).as_posix()
                except Exception:
                    rel_name = asset_path.name
                fmt = str(d.get("format") or asset_path.suffix.lower().lstrip(".") or "text")
                try:
                    content = asset_path.read_text(encoding="utf-8")
                    pasted_files.append({"name": rel_name, "content": content, "format": fmt})
                except Exception:
                    import base64 as _b64
                    try:
                        content = _b64.b64encode(asset_path.read_bytes()).decode("ascii")
                    except Exception:
                        continue
                    pasted_files.append(
                        {"name": rel_name, "content": content, "encoding": "base64", "format": fmt or "binary"}
                    )
            if pasted_files:
                cell["pasted_files"] = pasted_files
            cells.append(cell)
    sd = data.get("scenario_datasets")
    scenario_datasets = list(sd) if isinstance(sd, list) else []
    return {"scenario": scen, "cells": cells, "scenario_datasets": scenario_datasets}


def write_draft(scenario: str, body: dict[str, Any]) -> dict[str, Any]:
    scen = sanitize_scenario_name(str(scenario or ""))
    root = draft_dir(scen)
    root.mkdir(parents=True, exist_ok=True)
    cells_in = body.get("cells")
    if not isinstance(cells_in, list):
        raise ValueError("body.cells must be a list")
    sd_in = body.get("scenario_datasets")
    scenario_datasets: list[Any] = list(sd_in) if isinstance(sd_in, list) else []

    manifest_cells: list[dict[str, Any]] = []
    for raw in cells_in:
        if not isinstance(raw, dict):
            continue
        cell_id = _sanitize_cell_id(str(raw.get("id") or raw.get("name") or "cell"))
        name = str(raw.get("name") or cell_id).strip() or cell_id
        entry = str(raw.get("entry") or "run").strip() or "run"
        code = str(raw.get("code") or "")
        rel_path = f"mlops/custom_cells/{scen}/draft/cell_{cell_id}.py"
        py_path = _reg.REPO_ROOT / rel_path
        py_path.parent.mkdir(parents=True, exist_ok=True)
        py_path.write_text(code, encoding="utf-8")

        ds_raw = raw.get("datasets")
        cell_datasets: list[Any] = []
        has_pasted_files = isinstance(raw.get("pasted_files"), list)
        if isinstance(ds_raw, list):
            for d in ds_raw:
                if isinstance(d, dict):
                    if has_pasted_files and str(d.get("mode") or "") == "managed_copy":
                        path = str(d.get("path") or "").strip().replace("\\", "/")
                        if f"/draft/data/{cell_id}/" in f"/{path}":
                            continue
                    cell_datasets.append(dict(d))

        pasted = raw.get("pasted_files")
        if isinstance(pasted, list):
            data_root = pasted_data_dir(scen, cell_id)
            data_root.mkdir(parents=True, exist_ok=True)
            for i, blob in enumerate(pasted):
                if not isinstance(blob, dict):
                    continue
                rel_name = _safe_internal_asset_path(
                    str(blob.get("name") or ""),
                    f"inline_{i}.txt",
                )
                content = blob.get("content")
                if content is None:
                    continue
                encoding = str(blob.get("encoding") or "").strip().lower()
                dest = data_root / rel_name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if encoding == "base64":
                    # Binary assets (images, weights, archives) arrive base64-encoded
                    # so they survive the JSON round-trip; decode to bytes on disk.
                    import base64 as _b64
                    try:
                        raw_bytes = _b64.b64decode(str(content), validate=False)
                    except Exception:
                        continue
                    dest.write_bytes(raw_bytes)
                    kind = "inline_asset"
                    default_fmt = "binary"
                else:
                    text = content if isinstance(content, str) else str(content)
                    dest.write_text(text, encoding="utf-8", errors="replace")
                    kind = "inline_text"
                    default_fmt = "text"
                rel_name_str = rel_name.as_posix()
                cell_datasets.append(
                    {
                        "name": Path(rel_name_str).stem,
                        "kind": kind,
                        "path": f"mlops/custom_cells/{scen}/draft/data/{cell_id}/{rel_name_str}",
                        "format": str(blob.get("format") or default_fmt),
                        "mode": "managed_copy",
                    }
                )

        manifest_cells.append(
            {
                "id": cell_id,
                "name": name,
                "path": rel_path,
                "entry": entry,
                "datasets": cell_datasets,
            }
        )

    manifest = {
        "version": 1,
        "scenario": scen,
        "cells": manifest_cells,
        "scenario_datasets": scenario_datasets,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    return read_draft(scen)


def promote_draft(
    scenario: str,
    template_name: str,
    *,
    cell_ids: list[str] | None = None,
) -> dict[str, Any]:
    scen = sanitize_scenario_name(str(scenario or ""))
    tpl = _sanitize_template_name(template_name)
    cfg = get_scenario_config(scen)
    if str(cfg.backbone_type or "") != "custom_code":
        raise ValueError("promotion is only supported for backbone_type: custom_code")

    draft = read_draft(scen)
    cells = list(draft.get("cells") or [])
    want: set[str] | None = None
    if cell_ids:
        want = {_sanitize_cell_id(x) for x in cell_ids if str(x).strip()}
    algos_dir = _reg.MLOPS_ROOT / "algos"
    algos_dir.mkdir(parents=True, exist_ok=True)

    new_specs: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        cid = _sanitize_cell_id(str(cell.get("id") or ""))
        if want is not None and cid not in want:
            continue
        src_rel = str(cell.get("path") or "").strip()
        if not src_rel:
            continue
        src = (_reg.REPO_ROOT / src_rel).resolve()
        if not src.is_file():
            continue
        dest_name = f"{tpl}__{cid}.py"
        dest_rel = f"mlops/algos/{dest_name}"
        dest = _reg.REPO_ROOT / dest_rel
        shutil.copy2(src, dest)
        new_specs.append(
            {
                "id": cid,
                "name": str(cell.get("name") or cid),
                "path": dest_rel,
                "entry": str(cell.get("entry") or "run"),
                "datasets": cell.get("datasets") if isinstance(cell.get("datasets"), list) else [],
            }
        )

    if not new_specs:
        raise ValueError("no cells were promoted (check draft manifest and cell_ids)")

    raw_path = Path(str(cfg.config_path))
    raw = yaml.safe_load(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("invalid scenario yaml")
    bcfg = raw.get("backbone_config")
    backbone_cfg = dict(bcfg) if isinstance(bcfg, dict) else {}
    backbone_cfg["cells"] = new_specs
    raw["backbone_config"] = backbone_cfg
    raw_path.write_text(yaml.safe_dump(raw, sort_keys=False, allow_unicode=False), encoding="utf-8")

    return {
        "scenario": scen,
        "template_name": tpl,
        "cells": new_specs,
        "algo_paths": [c["path"] for c in new_specs],
    }
