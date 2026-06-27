"""Thin wrapper around the TRELLIS.2 Hugging Face Space gradio API.

Uses the verified endpoint schema at:
    https://microsoft-trellis-2.hf.space/gradio_api/info

Pipeline (cross-checked against the Space's app.py):
    /preprocess_image(input)        -> local path of preprocessed RGBA image
    /get_seed(randomize_seed, seed) -> resolved seed int
    /image_to_3d(image, seed, ...)  -> HTML preview string (state is hidden gr.State)
    /extract_glb(decimation_target, texture_size) -> (glb_path, glb_path)

Notes:
- The Space hides a ``gr.State()`` (output_buf) from the API. extract_glb reads
  state via session_hash, so we must reuse a single Client instance for the
  whole job (which we do).
- ``/preprocess_image`` returns a server-side image; gradio_client downloads it
  to a local temp path. We re-wrap that path with ``handle_file()`` for the
  /image_to_3d call so the receiving ``gr.Image(type="pil")`` component can
  load it correctly.
- ``/image_to_3d`` returns embedded base64 preview images inside HTML. We
  extract the first one as ``preview.png`` for the dashboard to render.
"""

from __future__ import annotations

import base64
import re
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

DEFAULT_HOST = "https://microsoft-trellis-2.hf.space"

# Defaults pulled directly from the Space's /gradio_api/info schema.
@dataclass
class SamplingParams:
    seed: int = 0
    randomize_seed: bool = True
    resolution: str = "1024"
    ss_guidance_strength: float = 7.5
    ss_guidance_rescale: float = 0.7
    ss_sampling_steps: int = 12
    ss_rescale_t: float = 5.0
    shape_slat_guidance_strength: float = 7.5
    shape_slat_guidance_rescale: float = 0.5
    shape_slat_sampling_steps: int = 12
    shape_slat_rescale_t: float = 3.0
    tex_slat_guidance_strength: float = 1.0
    tex_slat_guidance_rescale: float = 0.0
    tex_slat_sampling_steps: int = 12
    tex_slat_rescale_t: float = 3.0
    decimation_target: int = 300_000
    texture_size: int = 2048

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_PARAMS = SamplingParams()


class Trellis2Error(RuntimeError):
    pass


StatusCallback = Callable[[str, str, float], None]
"""(stage, message, progress) -> None.

Stages: 'connect', 'preprocess', 'generate', 'extract', 'done', 'error'.
Progress is in [0.0, 1.0]. Pass -1.0 to leave the bar unchanged.
"""


def _noop(_stage: str, _message: str, _progress: float = -1.0) -> None:
    return None


class Trellis2Client:
    """Calls the hosted TRELLIS.2 Space via gradio_client.

    gradio_client is imported lazily so the dashboard remains importable on
    machines that have not yet installed it.
    """

    def __init__(self, host: str = DEFAULT_HOST, hf_token: Optional[str] = None) -> None:
        self.host = host.rstrip("/")
        self.hf_token = hf_token
        self._client: Any = None

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from gradio_client import Client  # type: ignore[import-not-found]
        except Exception as exc:
            raise Trellis2Error(
                "gradio_client is not installed. Run: pip install -r mlops/dashboard/requirements.txt"
            ) from exc
        kwargs: dict[str, Any] = {}
        if self.hf_token:
            kwargs["hf_token"] = self.hf_token
        self._client = Client(self.host, **kwargs)
        return self._client

    def generate(
        self,
        image_path: Path,
        params: SamplingParams,
        out_dir: Path,
        on_status: StatusCallback = _noop,
    ) -> dict[str, str]:
        """Run the full image -> GLB pipeline. Returns paths to artifacts in out_dir."""
        try:
            from gradio_client import handle_file  # type: ignore[import-not-found]
        except Exception as exc:
            raise Trellis2Error("gradio_client not installed") from exc

        out_dir.mkdir(parents=True, exist_ok=True)

        on_status("connect", f"connecting to {self.host}", 0.0)
        client = self._ensure_client()

        on_status("preprocess", "preprocessing image (background removal + crop)", 0.05)
        preprocessed = client.predict(
            input=handle_file(str(image_path)),
            api_name="/preprocess_image",
        )
        preprocessed_path = _first_path(preprocessed)
        if not preprocessed_path:
            raise Trellis2Error(f"preprocess_image returned unexpected payload: {preprocessed!r}")

        on_status("preprocess", "resolving seed", 0.1)
        seed_value = client.predict(
            randomize_seed=params.randomize_seed,
            seed=params.seed,
            api_name="/get_seed",
        )

        on_status("generate", "queued on ZeroGPU", 0.12)
        i23_job = client.submit(
            handle_file(preprocessed_path),
            int(seed_value),
            params.resolution,
            params.ss_guidance_strength,
            params.ss_guidance_rescale,
            params.ss_sampling_steps,
            params.ss_rescale_t,
            params.shape_slat_guidance_strength,
            params.shape_slat_guidance_rescale,
            params.shape_slat_sampling_steps,
            params.shape_slat_rescale_t,
            params.tex_slat_guidance_strength,
            params.tex_slat_guidance_rescale,
            params.tex_slat_sampling_steps,
            params.tex_slat_rescale_t,
            api_name="/image_to_3d",
        )
        i23_result = _await_with_progress(
            i23_job, on_status, stage="generate", base=0.12, span=0.73
        )
        preview_html = i23_result if isinstance(i23_result, str) else (
            i23_result[-1] if isinstance(i23_result, (list, tuple)) and i23_result else ""
        )
        preview_html_path = out_dir / "preview.html"
        if preview_html:
            preview_html_path.write_text(preview_html, encoding="utf-8")
        preview_png_path = out_dir / "preview.png"
        if preview_html and _save_first_embedded_image(preview_html, preview_png_path):
            preview_artifact = str(preview_png_path)
        else:
            preview_artifact = ""

        on_status("extract", "extracting GLB (decimation + texture bake)", 0.86)
        glb_job = client.submit(
            params.decimation_target,
            params.texture_size,
            api_name="/extract_glb",
        )
        glb_result = _await_with_progress(
            glb_job, on_status, stage="extract", base=0.86, span=0.13
        )
        glb_src = _first_path(glb_result)
        if glb_src is None:
            raise Trellis2Error(f"extract_glb returned unexpected payload: {glb_result!r}")
        glb_dest = out_dir / "output.glb"
        _copy_artifact(glb_src, glb_dest)

        on_status("done", "complete", 1.0)
        return {
            "preview_path": preview_artifact,
            "preview_html_path": str(preview_html_path) if preview_html else "",
            "glb_path": str(glb_dest),
            "seed": str(seed_value),
        }


def _await_with_progress(
    job: Any,
    on_status: StatusCallback,
    *,
    stage: str,
    base: float,
    span: float,
    poll_interval: float = 0.5,
) -> Any:
    """Block until ``job`` completes, mapping Gradio progress_data to on_status.

    ``base`` is the overall fraction reached when this stage starts; ``span`` is
    how much of the overall bar this stage covers. Inside the stage we call
    ``on_status`` with ``base + span * fraction_of_stage``.
    """
    last_msg = ""
    while True:
        if job.done():
            break
        try:
            update = job.status()
        except Exception:
            update = None
        if update is not None:
            frac = _stage_fraction_from_update(update)
            msg = _stage_message_from_update(update)
            if frac is None and msg == "":
                # No progress info yet; emit a heartbeat so the UI shows we're alive.
                on_status(stage, last_msg or "running...", base + span * 0.0)
            else:
                if msg:
                    last_msg = msg
                progress = base + span * (frac if frac is not None else 0.0)
                on_status(stage, last_msg or "running...", min(max(progress, base), base + span))
        time.sleep(poll_interval)

    # Surface app-side exceptions cleanly.
    if hasattr(job, "exception"):
        try:
            exc = job.exception()
        except Exception:
            exc = None
        if exc is not None:
            raise Trellis2Error(f"upstream Space raised: {exc}") from exc

    return job.result()


def _stage_fraction_from_update(update: Any) -> Optional[float]:
    progress_data = getattr(update, "progress_data", None)
    if not progress_data:
        return None
    # progress_data is a list[ProgressUnit]. Use the most-advanced unit.
    best: Optional[float] = None
    for unit in progress_data:
        idx = getattr(unit, "index", None)
        length = getattr(unit, "length", None)
        prog = getattr(unit, "progress", None)
        frac: Optional[float] = None
        if isinstance(prog, (int, float)) and prog > 0:
            frac = float(prog)
        elif isinstance(idx, (int, float)) and isinstance(length, (int, float)) and length:
            frac = float(idx) / float(length)
        if frac is None:
            continue
        if best is None or frac > best:
            best = frac
    if best is None:
        return None
    return max(0.0, min(1.0, best))


def _stage_message_from_update(update: Any) -> str:
    progress_data = getattr(update, "progress_data", None)
    if progress_data:
        for unit in progress_data:
            desc = getattr(unit, "desc", None)
            idx = getattr(unit, "index", None)
            length = getattr(unit, "length", None)
            if isinstance(idx, (int, float)) and isinstance(length, (int, float)) and length:
                if desc:
                    return f"{desc} {int(idx)}/{int(length)}"
                return f"step {int(idx)}/{int(length)}"
            if desc:
                return str(desc)
    code = getattr(update, "code", None)
    if code is not None:
        return f"status: {code!s}"
    return ""


def _first_path(payload: Any) -> Optional[str]:
    """Gradio sometimes returns a tuple (file, file) or a dict {'value': path}."""
    if payload is None:
        return None
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (list, tuple)) and payload:
        return _first_path(payload[0])
    if isinstance(payload, dict):
        for key in ("path", "name", "value"):
            v = payload.get(key)
            if isinstance(v, str):
                return v
    return None


_BASE64_IMG_RE = re.compile(
    r'src="data:image/(?P<ext>png|jpeg|jpg|webp);base64,(?P<data>[A-Za-z0-9+/=]+)"',
    re.IGNORECASE,
)


def _save_first_embedded_image(html: str, dest: Path) -> bool:
    m = _BASE64_IMG_RE.search(html or "")
    if not m:
        return False
    try:
        raw = base64.b64decode(m.group("data"), validate=False)
    except Exception:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(raw)
    return True


def _copy_artifact(src: Any, dest: Path) -> None:
    src_path = _first_path(src) if not isinstance(src, str) else src
    if not src_path:
        raise Trellis2Error(f"missing artifact source: {src!r}")
    p = Path(src_path)
    if not p.exists():
        raise Trellis2Error(f"artifact not found at {p}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(p, dest)
