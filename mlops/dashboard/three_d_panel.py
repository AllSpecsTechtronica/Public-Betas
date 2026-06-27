"""3D Generation panel for the CV MLOps dashboard.

Wraps the hosted TRELLIS.2 Hugging Face Space so users can upload an image
and download a .glb without leaving the dashboard. On macOS the local engine
is unavailable, so this panel always uses the cloud backend.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from . import image_to_3d_panel
except ImportError:
    from mlops.dashboard import image_to_3d_panel

from mlops.trellis2 import (
    DEFAULT_PARAMS,
    Capabilities,
    Job,
    JobStatus,
    JobStore,
    SamplingParams,
    Trellis2Client,
    Trellis2Error,
    detect,
)

log = logging.getLogger(__name__)

PANEL_TITLE = "3D Generation"
PANEL_CAPTION = (
    "Image-to-3D via the hosted TRELLIS.2 Hugging Face Space. "
    "[CLOUD] backend on macOS; [LOCAL] backend reserved for Linux/Windows + NVIDIA CUDA hosts. "
    "The owned base pipeline with your local Depth Anything model is available in this page as well."
)

OS_LABEL = {"darwin": "macOS", "windows": "Windows", "linux": "Linux"}


def _capability_banner(caps: Capabilities) -> None:
    os_label = OS_LABEL.get(caps.os, caps.os)
    if caps.os == "darwin":
        st.warning(
            f"**[MAC DETECTED]** {os_label} cannot run TRELLIS.2 locally "
            "(requires Linux or Windows with an NVIDIA CUDA GPU). "
            "Using the hosted Hugging Face Space."
        )
    elif not caps.local.available:
        st.warning(f"**[LOCAL UNAVAILABLE]** {caps.local.reason} Falling back to cloud backend.")
    else:
        st.success(f"**[LOCAL READY]** {caps.gpu.name} detected. You can run on-host.")


def _capability_kv(caps: Capabilities) -> None:
    st.markdown("**Host capabilities**")
    st.write(
        {
            "operating_system": OS_LABEL.get(caps.os, caps.os),
            "gpu": caps.gpu.name or "not detected",
            "cuda": caps.gpu.cuda,
            "default_backend": caps.default_backend,
            "local_available": caps.local.available,
            "local_reason": caps.local.reason or "",
        }
    )


def _params_form(defaults: SamplingParams) -> SamplingParams:
    """Render the advanced sampling controls. Returns a fresh SamplingParams."""
    with st.expander("Advanced sampling parameters", expanded=False):
        st.caption(
            "Defaults match the TRELLIS.2 Space. Tune only if you know what these do; "
            "out-of-range values will degrade quality."
        )
        general_tab, sparse_tab, shape_tab, texture_tab = st.tabs(
            ["General", "Sparse structure", "Shape SLAT", "Texture SLAT"]
        )
        with general_tab:
            randomize_seed = st.checkbox("randomize_seed", value=defaults.randomize_seed)
            seed = st.number_input("seed", value=int(defaults.seed), step=1)
            resolution = st.selectbox(
                "resolution", options=["512", "768", "1024"],
                index=["512", "768", "1024"].index(defaults.resolution),
            )
            decimation_target = st.number_input(
                "decimation_target", min_value=1_000, max_value=2_000_000,
                value=int(defaults.decimation_target), step=10_000,
            )
            texture_size = st.selectbox(
                "texture_size", options=[1024, 2048, 4096],
                index=[1024, 2048, 4096].index(int(defaults.texture_size)),
            )
        with sparse_tab:
            ss_g_strength = st.slider("ss_guidance_strength", 0.0, 15.0, float(defaults.ss_guidance_strength), 0.1)
            ss_g_rescale = st.slider("ss_guidance_rescale", 0.0, 1.0, float(defaults.ss_guidance_rescale), 0.05)
            ss_steps = st.slider("ss_sampling_steps", 1, 50, int(defaults.ss_sampling_steps), 1)
            ss_rescale_t = st.slider("ss_rescale_t", 0.0, 10.0, float(defaults.ss_rescale_t), 0.1)
        with shape_tab:
            shape_g_strength = st.slider("shape_slat_guidance_strength", 0.0, 15.0, float(defaults.shape_slat_guidance_strength), 0.1)
            shape_g_rescale = st.slider("shape_slat_guidance_rescale", 0.0, 1.0, float(defaults.shape_slat_guidance_rescale), 0.05)
            shape_steps = st.slider("shape_slat_sampling_steps", 1, 50, int(defaults.shape_slat_sampling_steps), 1)
            shape_rescale_t = st.slider("shape_slat_rescale_t", 0.0, 10.0, float(defaults.shape_slat_rescale_t), 0.1)
        with texture_tab:
            tex_g_strength = st.slider("tex_slat_guidance_strength", 0.0, 15.0, float(defaults.tex_slat_guidance_strength), 0.1)
            tex_g_rescale = st.slider("tex_slat_guidance_rescale", 0.0, 1.0, float(defaults.tex_slat_guidance_rescale), 0.05)
            tex_steps = st.slider("tex_slat_sampling_steps", 1, 50, int(defaults.tex_slat_sampling_steps), 1)
            tex_rescale_t = st.slider("tex_slat_rescale_t", 0.0, 10.0, float(defaults.tex_slat_rescale_t), 0.1)

    return SamplingParams(
        seed=int(seed),
        randomize_seed=bool(randomize_seed),
        resolution=str(resolution),
        ss_guidance_strength=float(ss_g_strength),
        ss_guidance_rescale=float(ss_g_rescale),
        ss_sampling_steps=int(ss_steps),
        ss_rescale_t=float(ss_rescale_t),
        shape_slat_guidance_strength=float(shape_g_strength),
        shape_slat_guidance_rescale=float(shape_g_rescale),
        shape_slat_sampling_steps=int(shape_steps),
        shape_slat_rescale_t=float(shape_rescale_t),
        tex_slat_guidance_strength=float(tex_g_strength),
        tex_slat_guidance_rescale=float(tex_g_rescale),
        tex_slat_sampling_steps=int(tex_steps),
        tex_slat_rescale_t=float(tex_rescale_t),
        decimation_target=int(decimation_target),
        texture_size=int(texture_size),
    )


def _save_upload(uploaded: Any, job_dir: Path) -> Path:
    suffix = Path(uploaded.name).suffix.lower() or ".png"
    dest = job_dir / f"input{suffix}"
    dest.write_bytes(uploaded.getvalue())
    return dest


def _start_job_thread(job: Job, store: JobStore, image_path: Path, params: SamplingParams) -> None:
    """Run the cloud pipeline in a background thread, persisting status to disk."""

    def run() -> None:
        def on_status(stage: str, message: str, progress: float = -1.0) -> None:
            j = store.load(job.job_id)
            if j is None:
                return
            j.stage = stage
            j.message = message
            if progress >= 0.0:
                j.progress = max(0.0, min(1.0, progress))
            j.status = JobStatus.RUNNING
            store.save(j)

        try:
            j = store.load(job.job_id)
            if j is None:
                return
            j.status = JobStatus.RUNNING
            j.input_path = str(image_path)
            store.save(j)

            client = Trellis2Client()
            result = client.generate(
                image_path=image_path,
                params=params,
                out_dir=store.dir(job.job_id),
                on_status=on_status,
            )

            j = store.load(job.job_id) or j
            j.status = JobStatus.COMPLETED
            j.stage = "done"
            j.message = "complete"
            j.preview_path = result.get("preview_path", "")
            j.preview_html_path = result.get("preview_html_path", "")
            j.glb_path = result.get("glb_path", "")
            j.seed = result.get("seed", "")
            store.save(j)
        except Trellis2Error as exc:
            log.exception("trellis2 cloud job failed")
            j = store.load(job.job_id)
            if j is not None:
                j.status = JobStatus.FAILED
                j.stage = "error"
                j.error = str(exc)
                store.save(j)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("unexpected trellis2 failure")
            j = store.load(job.job_id)
            if j is not None:
                j.status = JobStatus.FAILED
                j.stage = "error"
                j.error = f"unexpected: {exc!r}"
                store.save(j)

    threading.Thread(target=run, name=f"trellis2-{job.job_id}", daemon=True).start()


@st.fragment(run_every=1.0)
def _render_active_job(job_id: str, store_root: str) -> None:
    """Auto-refreshing status panel.

    Polls ``status.json`` once per second; when the job reaches a terminal
    state we render the final outputs and stop emitting heartbeats by checking
    status before each redraw.
    """
    store = JobStore(Path(store_root)) if store_root else JobStore()
    fresh = store.load(job_id)
    if fresh is None:
        st.caption(f"job `{job_id}` not found")
        return

    st.markdown(f"**Active job:** `{fresh.job_id}` (backend: {fresh.backend})")

    if fresh.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        stage = fresh.stage.upper() or "QUEUED"
        st.info(f"[{stage}] {fresh.message or 'starting...'}")
        st.progress(min(max(fresh.progress, 0.0), 1.0))
        st.caption(f"updated {time.strftime('%H:%M:%S', time.localtime(fresh.updated_at))}")
    elif fresh.status == JobStatus.COMPLETED:
        st.success("[COMPLETED] artifacts ready")
        st.progress(1.0)
        _render_outputs(fresh)
    elif fresh.status == JobStatus.FAILED:
        st.error(f"[FAILED] {fresh.error}")
        if st.button("Dismiss", key=f"dismiss_{fresh.job_id}"):
            st.session_state.pop("three_d_active_job", None)
            st.rerun()


def _render_outputs(job: Job) -> None:
    preview_tab, interactive_tab, asset_tab = st.tabs(["Preview", "Interactive Preview", "GLB Asset"])
    with preview_tab:
        st.markdown("**Preview**")
        if job.preview_path and Path(job.preview_path).exists():
            st.image(job.preview_path, caption="render snapshot from TRELLIS.2")
        else:
            st.caption("no static preview extracted")
    with interactive_tab:
        if job.preview_html_path and Path(job.preview_html_path).exists():
            st.caption(
                "The Space embeds 48 base64 renders in HTML for its interactive viewer. "
                "Static rendering here may look incomplete; download the GLB and view in a 3D tool."
            )
            html = Path(job.preview_html_path).read_text(encoding="utf-8", errors="replace")
            st.components.v1.html(html, height=920, scrolling=True)
        else:
            st.caption("no interactive HTML preview extracted")
    with asset_tab:
        st.markdown("**GLB asset**")
        if job.glb_path and Path(job.glb_path).exists():
            st.write({"path": job.glb_path, "size_bytes": Path(job.glb_path).stat().st_size, "seed": job.seed})
            with open(job.glb_path, "rb") as f:
                st.download_button(
                    "Download .glb",
                    data=f.read(),
                    file_name=Path(job.glb_path).name,
                    mime="model/gltf-binary",
                    key=f"dl_{job.job_id}",
                )
        else:
            st.caption("glb missing")


def _render_history(store: JobStore) -> None:
    jobs = store.list_recent(limit=8)
    if not jobs:
        return
    with st.expander("Recent jobs", expanded=False):
        for j in jobs:
            cols = st.columns([3, 2, 3, 2])
            cols[0].write(j.job_id)
            cols[1].write(j.status.value)
            cols[2].write(j.message or j.error or j.stage)
            if j.status == JobStatus.COMPLETED and cols[3].button("Open", key=f"open_{j.job_id}"):
                st.session_state["three_d_active_job"] = j.job_id
                st.rerun()


def render() -> None:
    st.subheader(PANEL_TITLE)
    st.caption(PANEL_CAPTION)

    base_tab, trellis_tab = st.tabs(["Base Pipeline", "Hosted TRELLIS.2"])
    with base_tab:
        image_to_3d_panel.render(show_header=False)

    with trellis_tab:
        _render_trellis()


def _render_trellis() -> None:

    caps = detect()
    _capability_banner(caps)
    _capability_kv(caps)

    st.divider()

    backend_options: list[tuple[str, str]] = []
    backend_options.append(("cloud", "Cloud (Hugging Face)"))
    if caps.local.available:
        backend_options.append(("local", "Local (advanced)"))
    backend = st.radio(
        "Backend",
        options=[opt[0] for opt in backend_options],
        format_func=lambda v: dict(backend_options)[v],
        index=0,
        horizontal=True,
    )
    if backend == "local":
        st.info("Local engine path is reserved for the upcoming Linux/Windows + CUDA build. Routing to cloud for now.")
        backend = "cloud"

    uploaded = st.file_uploader(
        "Upload an image (PNG/JPG; alpha-masked foreground recommended)",
        type=["png", "jpg", "jpeg", "webp"],
    )

    params = _params_form(DEFAULT_PARAMS)

    store = JobStore()

    run_col, _ = st.columns([1, 4])
    run_disabled = uploaded is None
    if run_col.button("Generate 3D", type="primary", disabled=run_disabled):
        job = store.create(backend=backend, params=params.as_dict())
        image_path = _save_upload(uploaded, store.dir(job.job_id))
        job = store.load(job.job_id) or job
        job.input_path = str(image_path)
        store.save(job)
        st.session_state["three_d_active_job"] = job.job_id
        _start_job_thread(job, store, image_path, params)
        st.rerun()

    active_id = st.session_state.get("three_d_active_job")
    if active_id:
        if store.load(active_id) is not None:
            st.divider()
            _render_active_job(active_id, str(store.root))

    _render_history(store)
