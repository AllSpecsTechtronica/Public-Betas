"""Native Apple Silicon TRELLIS.2 integration.

This drives the local FastAPI server in ``mlops/three_d/trellis2-apple-main``:

    python api_server.py --host 127.0.0.1 --port 8082 --weights weights/TRELLIS.2-4B

CV Ops treats that process as user-managed for now. The request is synchronous
and writes the output GLB into the normal ``~/.trellis2/jobs/<id>`` artifact dir.
"""

from __future__ import annotations

import base64
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from mlops.trellis2.client import SamplingParams, Trellis2Error

StatusCallback = Callable[[str, str, float], None]

_REPO_ROOT = Path(__file__).resolve().parent / "trellis2-apple-main"


def repo_root() -> Path:
    return _REPO_ROOT


def default_api_base_url() -> str:
    return (os.environ.get("CVOPS_TRELLIS_APPLE_URL") or "http://127.0.0.1:8082").rstrip("/")


def available() -> tuple[bool, str]:
    root = repo_root()
    if not (root / "api_server.py").is_file():
        return False, f"trellis2-apple-main/api_server.py not found at {root}"
    if not (root / "mlx_backend" / "pipeline.py").is_file():
        return False, f"MLX backend not found under {root}"
    return True, str(root)


def ping(base_url: str | None = None) -> tuple[bool, str]:
    base = (base_url or default_api_base_url()).rstrip("/")
    try:
        with urlopen(Request(base + "/health", method="GET"), timeout=10) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        return False, f"Trellis2 Apple API HTTP {exc.code}: {exc.reason}"
    except URLError as exc:
        return False, f"Trellis2 Apple API not reachable at {base!r}: {exc.reason}"
    except Exception as exc:
        return False, f"Trellis2 Apple API health check failed: {exc}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return False, f"Trellis2 Apple API returned non-JSON health payload: {raw[:120]}"
    status = str(data.get("status") or "")
    if status == "ok":
        return True, "Trellis2 Apple API ready."
    return False, f"Trellis2 Apple API status: {status or 'unknown'}"


def run_trellis2_apple_job(
    *,
    base_url: str | None,
    image_path: Path,
    out_dir: Path,
    params: SamplingParams,
    on_status: StatusCallback,
) -> dict[str, str]:
    ok, msg = available()
    if not ok:
        raise Trellis2Error(msg)

    base = (base_url or default_api_base_url()).rstrip("/")
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path = out_dir / "output.glb"

    on_status("connect", f"checking Trellis2 Apple API at {base}", 0.02)
    ok, health_msg = ping(base)
    if not ok:
        raise Trellis2Error(
            health_msg
            + " Start it from mlops/three_d/trellis2-apple-main with: "
            "python api_server.py --host 127.0.0.1 --port 8082 --weights weights/TRELLIS.2-4B"
        )

    seed = random.randint(0, 2**31 - 1) if params.randomize_seed else int(params.seed)
    pipeline_type = {
        "512": "512",
        "1024": "1024_cascade",
        "1536": "1536_cascade",
    }.get(str(params.resolution), "1024_cascade")

    try:
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    except OSError as exc:
        raise Trellis2Error(f"Could not read image for Trellis2 Apple API: {exc}") from exc

    payload: dict[str, Any] = {
        "image": encoded,
        "seed": seed,
        "pipeline_type": pipeline_type,
        "output_path": str(glb_path),
        "decimation_target": int(params.decimation_target),
        "texture_size": int(params.texture_size),
        "steps": int(
            max(params.ss_sampling_steps, params.shape_slat_sampling_steps, params.tex_slat_sampling_steps)
        ),
        "guidance_strength": float(
            max(params.ss_guidance_strength, params.shape_slat_guidance_strength)
        ),
        "texture_guidance": float(params.tex_slat_guidance_strength),
    }

    on_status("generate", "running native Apple MLX pipeline", 0.08)
    req = Request(
        base + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    try:
        with urlopen(req, timeout=60 * 60) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.reason
        try:
            body = exc.read().decode("utf-8")
            if body:
                detail = body
        except Exception:
            pass
        raise Trellis2Error(f"Trellis2 Apple generation HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise Trellis2Error(f"Trellis2 Apple API failed during generation: {exc.reason}") from exc

    on_status("extract", "saving GLB artifact", 0.96)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise Trellis2Error("Trellis2 Apple API returned invalid JSON from /generate") from exc

    if not glb_path.is_file():
        glb_raw = data.get("glb")
        if not isinstance(glb_raw, str) or not glb_raw:
            raise Trellis2Error("Trellis2 Apple API response did not include a GLB artifact.")
        try:
            glb_path.write_bytes(base64.b64decode(glb_raw))
        except Exception as exc:
            raise Trellis2Error(f"Could not decode Trellis2 Apple GLB response: {exc}") from exc

    verts = data.get("vertices", "?")
    faces = data.get("faces", "?")
    dt = data.get("generation_time")
    msg = f"complete ({verts} vertices, {faces} faces"
    if dt is not None:
        msg += f", {dt}s server time"
    msg += f", {time.time() - t0:.1f}s wall)"
    on_status("done", msg, 1.0)

    return {
        "preview_path": "",
        "preview_html_path": "",
        "glb_path": str(glb_path),
        "seed": str(seed),
    }
