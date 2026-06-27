"""Local ComfyUI + ComfyUI-Trellis2 integration (HTTP API).

CV Ops drives a **user-managed** ComfyUI instance (e.g. Apple Silicon + MPS) by:

1. ``GET /object_info`` — resolve widget input order for each node type.
2. Convert bundled UI-format workflow JSON (from ``ComfyUI-Trellis2-main/example_workflows``)
   into the ``prompt`` dict ComfyUI expects.
3. ``POST /upload/image`` — place the source image in ComfyUI's input folder.
4. ``POST /prompt`` — queue the graph; poll ``GET /history`` until completion.
5. Copy the exported ``.glb`` into the trellis2 job directory.

Environment:

- ``CVOPS_COMFY_URL`` — default base URL if the UI field is empty (default ``http://127.0.0.1:8188``).

Install notes live in ``mlops/three_d/ComfyUI-Trellis2-main/README.md`` (custom_nodes copy, models, torch).
"""

from __future__ import annotations

import json
import logging
import mimetypes
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mlops.trellis2.client import SamplingParams, Trellis2Error

log = logging.getLogger(__name__)

_REPO_THREE_D = Path(__file__).resolve().parent
BUNDLED_TRELLIS2_WORKFLOWS = _REPO_THREE_D / "ComfyUI-Trellis2-main" / "example_workflows"

StatusCallback = Callable[[str, str, float], None]


def _noop(_stage: str, _message: str, _progress: float = -1.0) -> None:
    return None


def default_comfy_base_url() -> str:
    import os

    return (os.environ.get("CVOPS_COMFY_URL") or "http://127.0.0.1:8188").rstrip("/")


def list_bundled_workflows() -> list[Path]:
    root = BUNDLED_TRELLIS2_WORKFLOWS
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob("*.json") if p.is_file())


def fetch_object_info(base_url: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/object_info"
    try:
        with urlopen(Request(url, method="GET"), timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise Trellis2Error(f"ComfyUI object_info HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise Trellis2Error(f"ComfyUI not reachable at {base_url!r}: {exc.reason}") from exc


def comfy_ping(base_url: str) -> tuple[bool, str]:
    """Return (ok, message) for a quick health check."""
    try:
        fetch_object_info(base_url)
        return True, "ComfyUI responded to /object_info."
    except Trellis2Error as exc:
        return False, str(exc)


def _required_input_names(class_type: str, object_info: dict[str, Any]) -> list[str]:
    meta = object_info.get(class_type)
    if not meta:
        raise Trellis2Error(
            f"ComfyUI does not register node type {class_type!r}. "
            "Install ComfyUI-Trellis2 under custom_nodes and restart ComfyUI."
        )
    order = (meta.get("input_order") or {}).get("required")
    if isinstance(order, list) and order:
        return [str(x) for x in order]
    req = (meta.get("input") or {}).get("required") or {}
    if isinstance(req, dict):
        return list(req.keys())
    return []


def ui_workflow_to_prompt(wf: dict[str, Any], object_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Convert ComfyUI UI-exported workflow (``nodes`` + ``links``) to API ``prompt`` mapping."""
    nodes = wf.get("nodes")
    if not isinstance(nodes, list):
        raise Trellis2Error("Workflow JSON has no top-level 'nodes' list.")
    links_raw = wf.get("links") or []
    nodes_by_id: dict[int, dict[str, Any]] = {}
    for n in nodes:
        if isinstance(n, dict) and "id" in n:
            nodes_by_id[int(n["id"])] = n

    incoming: dict[tuple[int, int], tuple[int, int]] = {}
    for L in links_raw:
        if not isinstance(L, (list, tuple)) or len(L) < 5:
            continue
        _lid = int(L[0])
        src_node = int(L[1])
        src_slot = int(L[2])
        tgt_node = int(L[3])
        tgt_slot = int(L[4])
        incoming[(tgt_node, tgt_slot)] = (src_node, src_slot)

    prompt: dict[str, dict[str, Any]] = {}
    for nid, node in sorted(nodes_by_id.items(), key=lambda kv: kv[0]):
        class_type = str(node.get("type") or "")
        if not class_type:
            continue
        try:
            names = _required_input_names(class_type, object_info)
        except Trellis2Error:
            log.warning("Skipping workflow node %s: unknown class_type %r", nid, class_type)
            continue
        if not names:
            prompt[str(nid)] = {"class_type": class_type, "inputs": {}}
            continue
        inputs: dict[str, Any] = {}
        wv = list(node.get("widgets_values") or [])
        for slot_idx, iname in enumerate(names):
            lk = incoming.get((nid, slot_idx))
            if lk is not None:
                inputs[iname] = [str(lk[0]), lk[1]]
            else:
                if not wv:
                    raise Trellis2Error(
                        f"Workflow node {nid} ({class_type}) missing widget value for input {iname!r} "
                        f"(slot {slot_idx}). Re-save the workflow in ComfyUI or pick another preset."
                    )
                inputs[iname] = wv.pop(0)
        if wv:
            log.debug("Ignoring extra widgets_values on node %s (%s): %s", nid, class_type, wv)
        prompt[str(nid)] = {"class_type": class_type, "inputs": inputs}
    return prompt


def _find_node_ids(wf: dict[str, Any], class_type: str) -> list[int]:
    out: list[int] = []
    for n in wf.get("nodes") or []:
        if isinstance(n, dict) and str(n.get("type")) == class_type and "id" in n:
            out.append(int(n["id"]))
    return out


def _patch_load_image(
    prompt: dict[str, dict[str, Any]],
    wf: dict[str, Any],
    uploaded_basename: str,
) -> None:
    for nid in _find_node_ids(wf, "Trellis2LoadImageWithTransparency"):
        entry = prompt.get(str(nid))
        if entry and "inputs" in entry:
            entry["inputs"]["image"] = uploaded_basename
            return
    raise Trellis2Error("Workflow has no Trellis2LoadImageWithTransparency node.")


def _patch_export_prefix(
    prompt: dict[str, dict[str, Any]],
    wf: dict[str, Any],
    prefix: str,
    object_info: dict[str, Any],
) -> None:
    """Set PrimitiveString value that feeds Trellis2ExportMesh filename_prefix, if present."""
    export_ids = _find_node_ids(wf, "Trellis2ExportMesh")
    if not export_ids:
        return
    exp_id = export_ids[0]
    prim_keys = _required_input_names("PrimitiveString", object_info)
    prim_field = prim_keys[0] if prim_keys else "value"
    for L in wf.get("links") or []:
        if not isinstance(L, (list, tuple)) or len(L) < 5:
            continue
        src_node = int(L[1])
        src_slot = int(L[2])
        tgt_node = int(L[3])
        tgt_slot = int(L[4])
        if tgt_node == exp_id and tgt_slot == 1 and src_slot == 0:
            src = str(src_node)
            ent = prompt.get(src)
            if ent and ent.get("class_type") == "PrimitiveString":
                ent["inputs"][prim_field] = prefix
                return
    for nid in _find_node_ids(wf, "Trellis2ExportMesh"):
        ent = prompt.get(str(nid))
        if ent and isinstance(ent.get("inputs"), dict) and "filename_prefix" in ent["inputs"]:
            v = ent["inputs"]["filename_prefix"]
            if isinstance(v, str):
                ent["inputs"]["filename_prefix"] = prefix


def _patch_export_mesh_file_format(prompt: dict[str, dict[str, Any]], wf: dict[str, Any]) -> None:
    allowed = frozenset({"glb", "obj", "ply", "stl", "3mf", "dae"})
    for nid in _find_node_ids(wf, "Trellis2ExportMesh"):
        ent = prompt.get(str(nid))
        if not ent or not isinstance(ent.get("inputs"), dict):
            continue
        ff = ent["inputs"].get("file_format")
        if ff not in allowed:
            ent["inputs"]["file_format"] = "glb"


def _patch_sparse_generator(
    prompt: dict[str, dict[str, Any]],
    wf: dict[str, Any],
    params: SamplingParams,
) -> None:
    for nid in _find_node_ids(wf, "Trellis2SparseGenerator"):
        ent = prompt.get(str(nid))
        if not ent or not isinstance(ent.get("inputs"), dict):
            continue
        ins = ent["inputs"]
        seed = random.randint(0, 0x7FFFFFFF) if params.randomize_seed else int(params.seed)
        ins["seed"] = int(seed)
        ins["sparse_structure_steps"] = int(params.ss_sampling_steps)
        ins["sparse_structure_guidance_strength"] = float(params.ss_guidance_strength)
        ins["sparse_structure_guidance_rescale"] = float(params.ss_guidance_rescale)
        ins["sparse_structure_rescale_t"] = float(params.ss_rescale_t)
        return


def _multipart_upload_image(base_url: str, image_path: Path) -> str:
    boundary = f"----cvopsBoundary{uuid.uuid4().hex}"
    crlf = b"\r\n"
    parts: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        parts.append(f"--{boundary}".encode())
        parts.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        parts.append(crlf)
        parts.append(value.encode("utf-8"))
        parts.append(crlf)

    add_field("type", "input")
    add_field("subfolder", "")
    add_field("overwrite", "true")
    fname = image_path.name
    ctype = mimetypes.guess_type(fname)[0] or "application/octet-stream"
    parts.append(f"--{boundary}".encode())
    parts.append(
        f'Content-Disposition: form-data; name="image"; filename="{fname}"'.encode()
    )
    parts.append(f"Content-Type: {ctype}".encode())
    parts.append(crlf)
    parts.append(image_path.read_bytes())
    parts.append(crlf)
    parts.append(f"--{boundary}--".encode() + crlf)

    body = b"".join(parts)
    url = base_url.rstrip("/") + "/upload/image"
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    try:
        with urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise Trellis2Error(f"ComfyUI upload/image HTTP {exc.code}: {exc.reason}") from exc
    except URLError as exc:
        raise Trellis2Error(f"ComfyUI upload failed: {exc.reason}") from exc
    name = data.get("name") if isinstance(data, dict) else None
    if not name:
        raise Trellis2Error(f"Unexpected upload response: {data!r}")
    return str(name)


def _post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = base_url.rstrip("/") + path
    body = json.dumps(payload).encode("utf-8")
    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8", errors="replace")
        except Exception:
            detail = ""
        raise Trellis2Error(f"ComfyUI POST {path} HTTP {exc.code}: {detail or exc.reason}") from exc


def _fetch_history(base_url: str) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/history"
    with urlopen(Request(url, method="GET"), timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _collect_glb_paths(obj: Any, out: list[str]) -> None:
    if isinstance(obj, str) and obj.lower().endswith(".glb") and ("/" in obj or "\\" in obj or obj.startswith("/")):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            _collect_glb_paths(x, out)
    elif isinstance(obj, dict):
        for v in obj.values():
            _collect_glb_paths(v, out)


def _wait_for_prompt(
    base_url: str,
    prompt_id: str,
    *,
    timeout_s: float = 7200.0,
    poll_s: float = 1.0,
    on_status: StatusCallback = _noop,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_prog = 0.05
    while time.monotonic() < deadline:
        try:
            hist = _fetch_history(base_url)
        except (URLError, HTTPError, TimeoutError, OSError) as exc:
            on_status("comfy", f"Waiting for ComfyUI history… ({exc})", last_prog)
            time.sleep(poll_s)
            continue
        entry = hist.get(prompt_id) if isinstance(hist, dict) else None
        if isinstance(entry, dict):
            status = entry.get("status", {})
            if isinstance(status, dict) and status.get("status_str") == "error":
                msgs = status.get("messages") or []
                raise Trellis2Error(f"ComfyUI run failed: {msgs}")
            outputs = entry.get("outputs")
            if outputs:
                return dict(entry)
        last_prog = min(0.95, last_prog + 0.01)
        on_status("comfy", f"ComfyUI running (prompt {prompt_id[:8]}…)", last_prog)
        time.sleep(poll_s)
    raise Trellis2Error(f"Timed out after {timeout_s:.0f}s waiting for ComfyUI prompt {prompt_id}.")


def run_comfy_trellis_job(
    *,
    comfy_base_url: str,
    workflow_path: Path,
    image_path: Path,
    out_dir: Path,
    params: SamplingParams,
    on_status: StatusCallback = _noop,
    job_prefix: str = "cvops_mesh",
) -> dict[str, str]:
    """Queue a bundled Trellis2 workflow on local ComfyUI; copy GLB into ``out_dir``."""
    base = comfy_base_url.rstrip("/")
    if not workflow_path.is_file():
        raise Trellis2Error(f"Workflow file not found: {workflow_path}")

    on_status("comfy", "Loading ComfyUI object_info…", 0.02)
    object_info = fetch_object_info(base)

    wf = json.loads(workflow_path.read_text(encoding="utf-8"))
    on_status("comfy", "Converting workflow to API prompt…", 0.05)
    prompt = ui_workflow_to_prompt(wf, object_info)

    on_status("comfy", f"Uploading {image_path.name}…", 0.08)
    uploaded = _multipart_upload_image(base, image_path)
    _patch_load_image(prompt, wf, uploaded)
    _patch_sparse_generator(prompt, wf, params)
    _patch_export_prefix(prompt, wf, job_prefix, object_info)
    _patch_export_mesh_file_format(prompt, wf)

    client_id = str(uuid.uuid4())
    on_status("comfy", "Queueing prompt on ComfyUI…", 0.1)
    qresp = _post_json(base, "/prompt", {"prompt": prompt, "client_id": client_id})
    if not isinstance(qresp, dict):
        raise Trellis2Error(f"Unexpected /prompt response: {qresp!r}")
    if qresp.get("error"):
        raise Trellis2Error(f"ComfyUI /prompt error: {qresp.get('error')}")
    node_errors = qresp.get("node_errors")
    if node_errors:
        raise Trellis2Error(f"ComfyUI node_errors: {node_errors}")
    prompt_id = qresp.get("prompt_id")
    if not prompt_id:
        raise Trellis2Error(f"ComfyUI /prompt missing prompt_id: {qresp!r}")

    on_status("comfy", "Running graph on ComfyUI (this can take many minutes)…", 0.12)
    entry = _wait_for_prompt(base, str(prompt_id), on_status=on_status)
    outputs = entry.get("outputs") or {}
    glb_candidates: list[str] = []
    _collect_glb_paths(outputs, glb_candidates)
    if not glb_candidates:
        raise Trellis2Error(
            "ComfyUI finished but no .glb path was found in history outputs. "
            "Check Trellis2ExportMesh and ComfyUI output folder permissions."
        )
    src_glb = Path(glb_candidates[-1]).expanduser()
    if not src_glb.is_file():
        raise Trellis2Error(f"Exported GLB path missing on disk: {src_glb}")

    out_dir.mkdir(parents=True, exist_ok=True)
    dest_glb = out_dir / "output.glb"
    shutil.copy2(src_glb, dest_glb)

    on_status("done", "Local ComfyUI mesh export complete.", 1.0)
    sgen = _find_node_ids(wf, "Trellis2SparseGenerator")
    seed_str = ""
    if sgen:
        ins = prompt.get(str(sgen[0]), {}).get("inputs") or {}
        seed_str = str(ins.get("seed", ""))
    return {
        "glb_path": str(dest_glb),
        "preview_path": "",
        "preview_html_path": "",
        "seed": seed_str,
    }
