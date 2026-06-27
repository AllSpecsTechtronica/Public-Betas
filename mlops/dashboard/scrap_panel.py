from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import streamlit as st

from mlops.pipeline import registry as reg
from mlops.scrap import filter as scrap_filter
from mlops.scrap.emit import LabeledItem, emit_yolo_dataset
from mlops.scrap.jobs import JobState, JobStore

log = logging.getLogger(__name__)

PANEL_TITLE = "Scrap"
PANEL_CAPTION = (
    "Build custom datasets by scraping topic images from the web, "
    "labeling boxes by hand, and emitting a YOLO dataset + scenario "
    "the trainer can pick up directly. [PERSONAL/RESEARCH USE]"
)


def _list_scrap_slugs() -> list[str]:
    try:
        names = reg.list_library_dataset_names()
    except Exception:
        return []
    return [n for n in names if JobStore.exists(n)]


def _job_dirs(slug: str) -> tuple[Path, Path, Path]:
    base = reg.resolve_library_dataset_path(slug)
    return base, base / "raw", base / "staged"


def _start_scrape_thread(slug: str, query: str, count: int) -> None:
    """Selenium import is deferred until the user actually clicks 'Start scrape'
    so the rest of the dashboard works on machines without Chrome installed."""
    def _run() -> None:
        from mlops.scrap.selenium_search import search_google_images
        try:
            JobStore.update(slug, state="scraping", message=f"searching '{query}'")
            _, raw_dir, staged_dir = _job_dirs(slug)
            raw_dir.mkdir(parents=True, exist_ok=True)
            result = search_google_images(query, count, raw_dir)
            JobStore.update(
                slug,
                raw_count=len(result.saved),
                message=f"downloaded {len(result.saved)} (attempted {result.attempted}, skipped {result.skipped}); staging",
            )
            stage = scrap_filter.dedupe_and_stage(raw_dir, staged_dir, min_size=0)
            JobStore.update(
                slug,
                state="staged",
                staged_count=len(stage.staged),
                message=(
                    f"staged {len(stage.staged)}; "
                    f"kept readable small images; dup={stage.skipped_dup} "
                    f"unreadable={stage.skipped_unreadable}"
                ),
            )
        except Exception as exc:
            log.exception("scrape thread failed for %s", slug)
            JobStore.update(slug, state="error", message=f"scrape failed: {exc}")

    t = threading.Thread(target=_run, name=f"scrap-{slug}", daemon=True)
    t.start()


def _new_job_view() -> None:
    st.markdown("### New job")
    st.write(
        "Pick a topic, a search query, and a target count. Scraping runs in a background "
        "thread; come back to the **Label** tab once the state flips to `staged`."
    )

    with st.form("scrap_new_job", clear_on_submit=False):
        topic = st.text_input("Topic", placeholder="e.g. african elephant")
        query = st.text_input("Search query", placeholder="african elephant savanna")
        count = st.number_input("Target image count", min_value=10, max_value=500, value=50, step=10)
        submitted = st.form_submit_button("Start scrape")

    if not submitted:
        return
    topic_clean = (topic or "").strip()
    if not topic_clean:
        st.error("Topic is required.")
        return
    raw_slug = topic_clean.lower().replace(" ", "_").replace("-", "_")
    raw_slug = "".join(ch for ch in raw_slug if ch.isalnum() or ch == "_") or "topic"
    try:
        slug = reg.pick_unique_library_dataset_slug(f"scrap_{raw_slug}")
        reg.create_library_dataset_root(slug)
    except Exception as exc:
        st.error(f"Could not create dataset folder: {exc}")
        return
    JobStore.save(JobState(
        slug=slug,
        topic=topic_clean,
        target_count=int(count),
        state="pending",
        message="job created",
    ))
    _start_scrape_thread(slug, (query or topic_clean).strip(), int(count))
    st.success(f"Started scrape for `{slug}`. Switch to the **Label** view to monitor progress.")


def _staged_images(slug: str) -> list[Path]:
    _, _, staged_dir = _job_dirs(slug)
    if not staged_dir.exists():
        return []
    return sorted(p for p in staged_dir.iterdir() if p.is_file())


def _label_view(job: JobState) -> None:
    st.markdown(f"### Label — `{job.slug}`")
    st.write(f"State: `{job.state}` — {job.message or '...'}")
    if job.state in {"pending", "scraping"}:
        st.info("Scrape in progress. Refresh in a moment.")
        if st.button("Refresh", key=f"refresh_{job.slug}"):
            st.rerun()
        return
    if job.state == "error":
        st.error(job.message or "Job failed")
        return

    images = _staged_images(job.slug)
    if not images:
        st.warning("No staged images found. The scrape may have returned zero usable results.")
        return

    try:
        from streamlit_drawable_canvas import st_canvas
        from PIL import Image
    except Exception as exc:
        st.error(
            "Labeler dependencies are missing. Install them with: "
            "`pip install streamlit-drawable-canvas pillow imagehash`."
        )
        st.caption(f"Import error: {exc}")
        return

    st.markdown("**Classes**")
    classes = list(job.classes)
    cols = st.columns([3, 1])
    new_class = cols[0].text_input("Add class", key=f"newcls_{job.slug}", label_visibility="collapsed", placeholder="class name")
    if cols[1].button("[ADD]", key=f"addcls_{job.slug}"):
        nc = (new_class or "").strip()
        if nc and nc not in classes:
            classes.append(nc)
            JobStore.update(job.slug, classes=classes)
            st.rerun()
    if not classes:
        st.warning("Add at least one class before drawing boxes.")
        return
    st.write("Existing classes: " + ", ".join(f"`{c}`" for c in classes))

    state_key = f"label_idx_{job.slug}"
    if state_key not in st.session_state:
        unlabeled = next((i for i, p in enumerate(images) if p.name not in job.labels), 0)
        st.session_state[state_key] = unlabeled

    idx = int(st.session_state[state_key]) % len(images)
    current = images[idx]
    progress = sum(1 for p in images if p.name in job.labels)
    st.progress(progress / max(1, len(images)), text=f"Labeled {progress} / {len(images)}")

    nav = st.columns(4)
    if nav[0].button("[PREV]", key=f"prev_{job.slug}"):
        st.session_state[state_key] = (idx - 1) % len(images)
        st.rerun()
    if nav[1].button("[NEXT]", key=f"next_{job.slug}"):
        st.session_state[state_key] = (idx + 1) % len(images)
        st.rerun()
    if nav[2].button("[NEXT UNLABELED]", key=f"nextu_{job.slug}"):
        for off in range(1, len(images) + 1):
            j = (idx + off) % len(images)
            if images[j].name not in job.labels:
                st.session_state[state_key] = j
                break
        st.rerun()
    if nav[3].button("[CLEAR THIS]", key=f"clear_{job.slug}"):
        if current.name in job.labels:
            new_labels = dict(job.labels)
            new_labels.pop(current.name, None)
            JobStore.update(job.slug, labels=new_labels)
            st.rerun()

    cls_pick = st.selectbox("Class for new boxes", options=classes, key=f"clspick_{job.slug}")
    cls_idx = classes.index(cls_pick)

    with Image.open(current) as im:
        im = im.convert("RGB")
        img_w, img_h = im.size
    canvas_w = min(900, img_w)
    scale = canvas_w / img_w
    canvas_h = int(img_h * scale)

    canvas = st_canvas(
        fill_color="rgba(255, 0, 0, 0.15)",
        stroke_width=2,
        stroke_color="#ff2a2a",
        background_image=Image.open(current),
        update_streamlit=True,
        height=canvas_h,
        width=canvas_w,
        drawing_mode="rect",
        key=f"canvas_{job.slug}_{idx}",
    )

    if canvas is not None and canvas.json_data is not None:
        objects = canvas.json_data.get("objects") or []
        boxes: list[list[float]] = []
        for obj in objects:
            if obj.get("type") != "rect":
                continue
            left = float(obj.get("left", 0.0)) / scale
            top = float(obj.get("top", 0.0)) / scale
            w = float(obj.get("width", 0.0)) * float(obj.get("scaleX", 1.0)) / scale
            h = float(obj.get("height", 0.0)) * float(obj.get("scaleY", 1.0)) / scale
            if w <= 1 or h <= 1:
                continue
            cx = (left + w / 2.0) / img_w
            cy = (top + h / 2.0) / img_h
            nw = w / img_w
            nh = h / img_h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))
            boxes.append([float(cls_idx), cx, cy, nw, nh])

        save_cols = st.columns(2)
        if save_cols[0].button("[SAVE BOXES]", key=f"save_{job.slug}_{idx}"):
            new_labels = dict(job.labels)
            new_labels[current.name] = boxes
            JobStore.update(job.slug, labels=new_labels)
            st.success(f"Saved {len(boxes)} box(es) for {current.name}")
        save_cols[1].caption(f"Drawn (unsaved): {len(boxes)} box(es). Image: {img_w}x{img_h}.")


def _emit_view(job: JobState) -> None:
    st.markdown(f"### Emit — `{job.slug}`")
    if job.state in {"pending", "scraping"}:
        st.info("Scrape still in progress.")
        return
    if not job.classes:
        st.warning("Add at least one class on the Label tab first.")
        return
    images = _staged_images(job.slug)
    labeled = [p for p in images if p.name in job.labels and job.labels[p.name]]
    st.write(f"Staged: {len(images)} | Labeled: {len(labeled)} | Classes: {len(job.classes)}")
    class_counts: dict[int, int] = {}
    for boxes in job.labels.values():
        for b in boxes:
            class_counts[int(b[0])] = class_counts.get(int(b[0]), 0) + 1
    st.write({job.classes[k]: v for k, v in sorted(class_counts.items()) if 0 <= k < len(job.classes)})

    val_frac = st.slider("Validation fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)
    epochs = st.number_input("Default training epochs", min_value=1, max_value=300, value=20)
    base_model = st.text_input("Base model", value="assets/models/yolov10n.pt")

    if st.button("[EMIT DATASET + SCENARIO]", key=f"emit_{job.slug}", type="primary"):
        if len(labeled) < 2:
            st.error("Need at least 2 labeled images to emit a train/val split.")
            return
        items = [
            LabeledItem(
                image_path=p,
                boxes=tuple((int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])) for b in job.labels[p.name]),
            )
            for p in labeled
        ]
        try:
            ds_root = emit_yolo_dataset(
                slug=job.slug,
                classes=list(job.classes),
                items=items,
                val_frac=float(val_frac),
            )
        except Exception as exc:
            st.error(f"Failed to emit dataset: {exc}")
            return
        try:
            info = reg.inspect_library_dataset_at(ds_root)
            if info.get("format") != reg.LIBRARY_DATASET_FORMAT_YOLO:
                st.error(f"Emitted dataset has format `{info.get('format')}`, expected YOLO.")
                return
        except Exception as exc:
            st.error(f"Dataset inspection failed: {exc}")
            return
        try:
            reg.create_scenario_profile(
                name=job.slug,
                display_name=job.topic.title(),
                description=f"Scrap-built scenario for topic '{job.topic}'.",
                base_model=base_model,
                dataset=job.slug,
                classes=list(job.classes),
                hyperparams={"epochs": int(epochs), "imgsz": 640},
                guard_profile="balanced",
                backbone_type="yolo_detection",
            )
        except Exception as exc:
            st.error(f"Dataset emitted but scenario creation failed: {exc}")
            return
        JobStore.update(job.slug, state="emitted", message="dataset + scenario emitted")
        st.success(
            f"Emitted dataset to `{ds_root}` and scenario `{job.slug}.yaml`. "
            f"Train with: `python -m mlops.pipeline.train --scenario {job.slug}`."
        )


def render() -> None:
    st.subheader(PANEL_TITLE)
    st.caption(PANEL_CAPTION)

    slugs = _list_scrap_slugs()
    options = ["[NEW JOB]"] + slugs
    pick = st.sidebar.selectbox("Scrap job", options=options, key="scrap_job_pick")

    if pick == "[NEW JOB]":
        _new_job_view()
        return

    job = JobStore.load(pick)
    if job is None:
        st.error(f"Job state for `{pick}` not found.")
        return

    view = st.radio(
        "View",
        options=["Label", "Emit", "Status"],
        horizontal=True,
        key=f"scrap_view_{job.slug}",
    )
    if view == "Label":
        _label_view(job)
    elif view == "Emit":
        _emit_view(job)
    else:
        st.json({
            "slug": job.slug,
            "topic": job.topic,
            "state": job.state,
            "message": job.message,
            "raw_count": job.raw_count,
            "staged_count": job.staged_count,
            "classes": job.classes,
            "labeled_images": sum(1 for v in job.labels.values() if v),
        })
        if st.button("Refresh", key=f"refresh_status_{job.slug}"):
            st.rerun()
