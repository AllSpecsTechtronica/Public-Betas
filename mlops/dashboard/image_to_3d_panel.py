"""Owned Image-to-3D base pipeline panel."""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

from mlops.image_to_3d import Job, JobStatus, JobStore, PipelineConfig, detect, run_pipeline
from mlops.image_to_3d.capability import coreml_depth_model_path

log = logging.getLogger(__name__)

PANEL_TITLE = "Image-to-3D (base)"
PANEL_CAPTION = (
    "Owned single-image pipeline: intrinsics, monocular depth, RGBD point cloud, "
    "mesh, walkability stub, provenance, and optional TRELLIS object enhancement."
)
PRESET_ROOT = Path(__file__).resolve().parents[1] / "image_to_3d" / "presets"


def _load_presets() -> dict[str, dict[str, Any]]:
    try:
        import yaml
    except Exception:
        return {}
    presets: dict[str, dict[str, Any]] = {}
    for path in sorted(PRESET_ROOT.glob("*.yaml")):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        if isinstance(raw, dict):
            name = str(raw.get("name") or path.stem)
            presets[name] = raw
    return presets


def _capability_banner() -> None:
    caps = detect(check_trellis=False)
    if caps.coreml_depth.available:
        st.success("**[COREML READY]** Local Apple Depth Anything package is available.")
    elif caps.coreml_depth.reason.startswith("missing:"):
        st.warning(f"**[COREML MISSING]** {caps.coreml_depth.reason}")
    else:
        st.warning(f"**[COREML UNAVAILABLE]** {caps.coreml_depth.reason}")
    if caps.mps.available:
        st.info("**[MPS READY]** Apple GPU acceleration is available for the Transformers fallback.")
    elif caps.torch.available:
        st.warning(f"**[CPU FALLBACK]** {caps.mps.reason}")
    else:
        st.error("**[TORCH MISSING]** Install dashboard requirements before running the pipeline.")
    st.write(
        {
            "os": caps.os,
            "torch": caps.torch.available,
            "mps": caps.mps.available,
            "depth_model": caps.depth_model.reason or "ready",
            "coreml_depth": caps.coreml_depth.reason or "ready",
            "transformers_depth": caps.transformers_depth.reason or "ready",
            "default_device": caps.default_device,
            "trellis_cloud": caps.trellis_cloud.reason or "optional",
        }
    )


def _config_form(presets: dict[str, dict[str, Any]]) -> PipelineConfig:
    names = list(presets) or ["fast_mono"]
    preset_name = st.selectbox(
        "Preset",
        options=names,
        format_func=lambda n: str(presets.get(n, {}).get("display_name") or n),
    )
    raw = dict(presets.get(preset_name, {}))
    config = PipelineConfig.from_dict(raw)

    st.caption(str(raw.get("description") or "Single-image base reconstruction."))
    enhance = st.checkbox("Enhance foreground object with TRELLIS", value=bool(config.enhance_trellis))
    config.enhance_trellis = enhance

    st.checkbox("World/map anchoring (v2)", value=False, disabled=True)
    st.checkbox("Multi-image SfM (v2)", value=False, disabled=True)
    st.checkbox("Gaussian splats (v2)", value=False, disabled=True)

    with st.expander("Advanced params", expanded=False):
        c1, c2, c3 = st.columns(3)
        config.max_image_size = int(c1.number_input("max_image_size", 256, 1536, int(config.max_image_size), 64))
        config.fov_degrees = float(c1.slider("fov_degrees", 30.0, 90.0, float(config.fov_degrees), 1.0))
        config.mesh_stride = int(c2.slider("mesh_stride", 1, 8, int(config.mesh_stride), 1))
        config.max_depth = float(c2.slider("max_depth", 1.0, 10.0, float(config.max_depth), 0.5))
        config.floor_percentile = float(c3.slider("floor_percentile", 50.0, 98.0, float(config.floor_percentile), 1.0))
        backend_options = ["auto", "coreml", "transformers"]
        backend_value = str(getattr(config, "depth_backend", "auto") or "auto").lower()
        picked_backend = c1.selectbox(
            "depth_backend",
            options=backend_options,
            index=backend_options.index(backend_value) if backend_value in backend_options else 0,
            help="auto prefers the local CoreML .mlpackage when it is present.",
        )
        config.depth_backend = picked_backend
        default_coreml_path = str(coreml_depth_model_path())
        config.depth_model_path = c2.text_input(
            "coreml_model_path",
            value=str(getattr(config, "depth_model_path", "") or default_coreml_path),
            help="Used by the coreml backend. Leave as the repo-local models path unless you moved the package.",
        ).strip()
        device_options = ["auto", "mps", "cpu"]
        device_value = config.device or "auto"
        picked_device = c3.selectbox("device", options=device_options, index=device_options.index(device_value) if device_value in device_options else 0)
        config.device = None if picked_device == "auto" else picked_device
    return config


def _save_upload(uploaded: Any, job_dir: Path) -> Path:
    from PIL import Image

    dest = job_dir / "input.png"
    image = Image.open(uploaded).convert("RGBA")
    image.save(dest)
    return dest


def _start_job_thread(job: Job, store: JobStore, image_path: Path, config: PipelineConfig) -> None:
    def run() -> None:
        fresh = store.load(job.job_id) or job
        fresh.input_path = str(image_path)
        fresh.params = config.as_dict()
        store.save(fresh)
        try:
            run_pipeline(job=fresh, store=store, input_path=image_path, config=config)
        except Exception as exc:
            log.exception("image-to-3d job failed")
            failed = store.load(job.job_id)
            if failed is not None:
                failed.status = JobStatus.FAILED
                failed.stage = "error"
                failed.error = str(exc)
                store.save(failed)

    threading.Thread(target=run, name=f"image-to-3d-{job.job_id}", daemon=True).start()


@st.fragment(run_every=1.0)
def _render_active_job(job_id: str, store_root: str) -> None:
    store = JobStore(Path(store_root)) if store_root else JobStore()
    job = store.load(job_id)
    if job is None:
        st.caption(f"job `{job_id}` not found")
        return

    st.markdown(f"**Active job:** `{job.job_id}`")
    if job.status in (JobStatus.QUEUED, JobStatus.RUNNING):
        st.info(f"[{(job.stage or 'queued').upper()}] {job.message or 'starting...'}")
        st.progress(min(max(job.progress, 0.0), 1.0))
        st.caption(f"updated {time.strftime('%H:%M:%S', time.localtime(job.updated_at))}")
        if job.stage_status:
            st.write(job.stage_status)
        return

    if job.status == JobStatus.FAILED:
        st.error(f"[FAILED] {job.error}")
        _render_provenance(job)
        return

    st.success("[COMPLETED] artifacts ready")
    st.progress(1.0)
    _render_outputs(job)


def _render_outputs(job: Job) -> None:
    depth_tab, points_tab, mesh_tab, prov_tab = st.tabs(["Depth", "Point cloud", "Mesh", "Provenance"])
    with depth_tab:
        if job.depth_vis_path and Path(job.depth_vis_path).exists():
            st.image(job.depth_vis_path)
        else:
            st.caption("depth visualization missing")
    with points_tab:
        _render_point_cloud(job)
    with mesh_tab:
        _render_mesh(job)
    with prov_tab:
        _render_provenance(job)


def _render_point_cloud(job: Job) -> None:
    path = Path(job.points_path) if job.points_path else Path("")
    if not path.exists():
        st.caption("point cloud missing")
        return
    try:
        import numpy as np
        import plotly.graph_objects as go
    except Exception as exc:
        st.error("Plotly/Numpy dependencies are missing.")
        st.caption(str(exc))
        return
    try:
        points, colors = _load_points(path, limit=30_000)
    except Exception as exc:
        st.error(f"Could not read point cloud: {exc}")
        return
    if len(points) == 0:
        st.caption("point cloud is empty")
        return
    color_values = [f"rgb({int(c[0])},{int(c[1])},{int(c[2])})" for c in colors]
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker={"size": 1.5, "color": color_values},
            )
        ]
    )
    fig.update_layout(height=760, margin={"l": 0, "r": 0, "t": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)


def _render_mesh(job: Job) -> None:
    path = Path(job.scene_path or job.mesh_path)
    if not path.exists():
        st.caption("mesh missing")
        return
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    html = f"""
<script type="module" src="https://unpkg.com/@google/model-viewer/dist/model-viewer.min.js"></script>
<model-viewer
  src="data:model/gltf-binary;base64,{encoded}"
  camera-controls
  interaction-prompt="none"
  style="width:100%;height:760px;background:#0b0f0d;"
></model-viewer>
"""
    st.components.v1.html(html, height=790, scrolling=False)
    st.download_button(
        "Download scene.glb",
        data=data,
        file_name=path.name,
        mime="model/gltf-binary",
        key=f"image_to_3d_dl_{job.job_id}",
    )


def _render_provenance(job: Job) -> None:
    path = Path(job.provenance_path) if job.provenance_path else Path("")
    if not path.exists():
        path = Path(job.scene_path).with_name("provenance.json") if job.scene_path else Path("")
    if path.exists():
        st.json(json.loads(path.read_text(encoding="utf-8")))
    else:
        st.caption("provenance missing")


def _load_points(path: Path, limit: int):
    import numpy as np

    try:
        import open3d as o3d  # type: ignore[import-not-found]

        pc = o3d.io.read_point_cloud(str(path))
        points = np.asarray(pc.points)
        colors = (np.asarray(pc.colors) * 255.0).clip(0, 255).astype("uint8")
    except Exception:
        points, colors = _load_ascii_ply(path)
    if len(points) > limit:
        idx = np.linspace(0, len(points) - 1, limit).astype("int64")
        points = points[idx]
        colors = colors[idx]
    return points, colors


def _load_ascii_ply(path: Path):
    import numpy as np

    lines = path.read_text(encoding="ascii", errors="ignore").splitlines()
    try:
        start = lines.index("end_header") + 1
    except ValueError:
        return np.empty((0, 3)), np.empty((0, 3), dtype="uint8")
    pts = []
    cols = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) < 6:
            continue
        pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
        cols.append([int(parts[3]), int(parts[4]), int(parts[5])])
    return np.asarray(pts, dtype="float32"), np.asarray(cols, dtype="uint8")


def _render_history(store: JobStore) -> None:
    jobs = store.list_recent(limit=8)
    if not jobs:
        return
    with st.expander("Recent jobs", expanded=False):
        for job in jobs:
            cols = st.columns([3, 2, 3, 2])
            cols[0].write(job.job_id)
            cols[1].write(job.status.value)
            cols[2].write(job.message or job.error or job.stage)
            if cols[3].button("Open", key=f"image_to_3d_open_{job.job_id}"):
                st.session_state["image_to_3d_active_job"] = job.job_id
                st.rerun()


def render(*, show_header: bool = True) -> None:
    if show_header:
        st.subheader(PANEL_TITLE)
        st.caption(PANEL_CAPTION)
    _capability_banner()
    st.divider()

    presets = _load_presets()
    config = _config_form(presets)
    uploaded = st.file_uploader("Upload one image", type=["png", "jpg", "jpeg", "webp"])

    store = JobStore()
    run_disabled = uploaded is None
    if st.button("Run base pipeline", type="primary", disabled=run_disabled):
        job = store.create(params=config.as_dict())
        image_path = _save_upload(uploaded, store.dir(job.job_id))
        st.session_state["image_to_3d_active_job"] = job.job_id
        _start_job_thread(job, store, image_path, config)
        st.rerun()

    active_id = st.session_state.get("image_to_3d_active_job")
    if active_id and store.load(active_id) is not None:
        st.divider()
        _render_active_job(active_id, str(store.root))

    _render_history(store)
