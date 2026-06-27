from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from mlops.pipeline import registry as reg

try:
    from .tabular_inspect import load_csv, quality_checks, summarize_frame
    from .utils import human_bytes, stable_jsonable, stat_path
    from .yolo_inspect import check_yolo_pairs, summarize_yolo_labels
    from . import custom_cells_panel
    from . import image_to_3d_panel
    from . import scrap_panel
    from . import three_d_panel
except ImportError:
    # Streamlit "run path/to/app.py" executes this as a script, not as a package module.
    from mlops.dashboard.tabular_inspect import load_csv, quality_checks, summarize_frame
    from mlops.dashboard.utils import human_bytes, stable_jsonable, stat_path
    from mlops.dashboard.yolo_inspect import check_yolo_pairs, summarize_yolo_labels
    from mlops.dashboard import custom_cells_panel
    from mlops.dashboard import image_to_3d_panel
    from mlops.dashboard import scrap_panel
    from mlops.dashboard import three_d_panel


APP_TITLE = "CV MLOps Dashboard"


def _scenario_state(row: dict[str, Any]) -> str:
    status = str(row.get("status") or "").strip().lower()
    if status in {"ready", "trained"}:
        return "ready"
    if status == "error" or row.get("error"):
        return "error"
    if bool(row.get("weights_ready")) and bool(row.get("verified", True)):
        return "ready"
    return "partial"


def _inject_embedded_theme() -> None:
    st.markdown(
        """
<style>
:root {
  --insight-bg: #000000;
  --insight-panel: #0c100e;
  --insight-panel-soft: #111813;
  --insight-accent: #c5ff46;
  --insight-warm: #ff9a3c;
  --insight-warm-soft: rgba(255, 154, 60, 0.14);
  --insight-warm-strong: rgba(255, 154, 60, 0.24);
  --insight-warm-border: rgba(255, 154, 60, 0.34);
  --insight-text: #f5f7f2;
  --insight-muted: rgba(197, 255, 70, 0.68);
  --insight-border: rgba(197, 255, 70, 0.20);
}
.stApp {
  background: var(--insight-bg);
  color: var(--insight-text);
}
.block-container {
  max-width: 1720px;
  padding-top: 1.5rem;
  padding-bottom: 2rem;
}
[data-testid="stSidebar"] {
  background: var(--insight-panel);
  border-right: 1px solid var(--insight-border);
}
[data-testid="stHeader"] {
  background: rgba(0, 0, 0, 0.82);
}
[data-testid="stMetric"],
[data-testid="stDataFrame"],
div[data-testid="stExpander"] {
  background: var(--insight-panel-soft);
  border: 1px solid var(--insight-border);
  border-radius: 8px;
}
[data-testid="stMetric"] {
  padding: 10px 12px;
}
div[data-baseweb="tab-list"] {
  gap: 0.5rem;
  overflow-x: auto;
  overflow-y: hidden;
  flex-wrap: nowrap;
  padding-bottom: 0.4rem;
  scrollbar-width: thin;
  scrollbar-color: var(--insight-accent) rgba(197, 255, 70, 0.12);
}
div[data-baseweb="tab-list"]::-webkit-scrollbar {
  height: 10px;
}
div[data-baseweb="tab-list"]::-webkit-scrollbar-track {
  background: rgba(197, 255, 70, 0.08);
  border-radius: 999px;
}
div[data-baseweb="tab-list"]::-webkit-scrollbar-thumb {
  background: rgba(197, 255, 70, 0.55);
  border-radius: 999px;
}
button[data-baseweb="tab"] {
  flex: 0 0 auto;
  min-width: 13rem;
  padding-inline: 1rem;
  border-radius: 10px 10px 0 0;
  white-space: nowrap;
  background: linear-gradient(180deg, rgba(255, 154, 60, 0.18), rgba(255, 154, 60, 0.10)) !important;
  border: 1px solid var(--insight-warm-border) !important;
  box-shadow: inset 0 0 0 1px rgba(255, 154, 60, 0.08);
  color: var(--insight-text) !important;
}
button[data-baseweb="tab"] p {
  white-space: nowrap;
}
button[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(180deg, rgba(255, 154, 60, 0.34), rgba(255, 154, 60, 0.18)) !important;
  border-color: rgba(255, 154, 60, 0.54) !important;
}
button[data-baseweb="tab"][aria-selected="false"]:hover {
  background: linear-gradient(180deg, rgba(255, 154, 60, 0.26), rgba(255, 154, 60, 0.14)) !important;
}
div[role="radiogroup"] label,
div[data-testid="stSegmentedControl"] button,
div[data-baseweb="select"] > div,
div[data-baseweb="base-input"] > div {
  background: var(--insight-warm-soft) !important;
  border-color: var(--insight-warm-border) !important;
}
div[role="radiogroup"] label:hover,
div[data-testid="stSegmentedControl"] button:hover,
div[data-baseweb="select"] > div:hover,
div[data-baseweb="base-input"] > div:hover {
  background: var(--insight-warm-strong) !important;
}
div[role="tabpanel"] {
  padding-top: 1rem;
}
h1, h2, h3 {
  color: var(--insight-text);
}
a, .st-emotion-cache-10trblm, .st-emotion-cache-16idsys p {
  color: var(--insight-accent);
}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False, ttl=20)
def _load_registry_payload() -> dict[str, Any]:
    return reg.load_registry()


@st.cache_data(show_spinner=False, ttl=20)
def _scenario_statuses(names: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name in names:
        status = reg.get_scenario_status(name)
        try:
            cfg = reg.get_scenario_config(name)
            status["config_path"] = str(cfg.config_path)
            status["weights"] = str(cfg.weights or "")
        except Exception:
            status["config_path"] = ""
            status["weights"] = ""
        out.append(status)
    return out


@st.cache_data(show_spinner=False, ttl=60)
def _inspect_dataset(dataset_slug: str) -> dict[str, Any]:
    base = reg.resolve_library_dataset_path(dataset_slug)
    info = reg.inspect_library_dataset_at(base)
    info["path"] = str(base)
    info["slug"] = dataset_slug
    info["format"] = str(info.get("format") or "")
    info["category"] = reg.dataset_category(info["format"])
    return stable_jsonable(info)


def _page_header() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_embedded_theme()
    st.title(APP_TITLE)
    st.caption("Embedded service overview, scenario readiness, datasets, and artifacts.")


def _render_registry_overview(statuses: list[dict[str, Any]]) -> None:
    st.subheader("Scenarios")
    df = pd.DataFrame(statuses)
    cols = [
        "name",
        "display_name",
        "status",
        "dataset",
        "dataset_count",
        "weights_ready",
        "verified",
        "base_model_exists",
        "weights",
        "history_count",
        "backbone_type",
    ]
    view = df[[c for c in cols if c in df.columns]].copy()

    states = [_scenario_state(row) for row in statuses]
    total = len(states)
    ready = states.count("ready")
    partial = states.count("partial")
    failed = states.count("error")
    weights_ready = int(view.get("weights_ready", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not view.empty else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Scenarios", total)
    m2.metric("Ready", ready)
    m3.metric("Partial", partial)
    m4.metric("Issues", failed)

    readiness = float(ready / total) if total else 0.0
    st.progress(readiness)
    st.caption(f"Readiness {ready}/{total} - weights available for {weights_ready} scenario(s).")

    chart_left, chart_right = st.columns([1, 2])
    with chart_left:
        state_df = pd.DataFrame(
            [
                {"state": "ready", "count": ready},
                {"state": "partial", "count": partial},
                {"state": "error", "count": failed},
            ]
        ).set_index("state")
        st.bar_chart(state_df, use_container_width=True)
    with chart_right:
        if not view.empty and "dataset_count" in view.columns and "name" in view.columns:
            dataset_counts = (
                view[["name", "dataset_count"]]
                .assign(dataset_count=lambda d: pd.to_numeric(d["dataset_count"], errors="coerce").fillna(0))
                .sort_values("dataset_count", ascending=False)
                .head(12)
                .set_index("name")
            )
            st.bar_chart(dataset_counts, use_container_width=True)

    st.dataframe(view, use_container_width=True, hide_index=True)

    with st.expander("Raw status payloads (JSON)", expanded=False):
        st.json(statuses)


def _render_dataset_lineage(info: dict[str, Any]) -> None:
    lineage = info.get("lineage") if isinstance(info.get("lineage"), dict) else {}
    nodes = lineage.get("nodes") if isinstance(lineage.get("nodes"), list) else []
    edges = lineage.get("edges") if isinstance(lineage.get("edges"), list) else []

    st.markdown("**Lineage**")
    st.write({"dataset_root": lineage.get("dataset_root") or info.get("path") or ""})
    if lineage.get("error"):
        st.warning(str(lineage.get("error")))

    if nodes:
        node_rows = [
            {
                "id": n.get("id"),
                "label": n.get("label"),
                "evidence": n.get("evidence"),
            }
            for n in nodes
            if isinstance(n, dict)
        ]
        if node_rows:
            st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)

    if edges:
        edge_rows = [
            {
                "from": e.get("from"),
                "to": e.get("to"),
                "evidence": e.get("evidence"),
            }
            for e in edges
            if isinstance(e, dict)
        ]
        if edge_rows:
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)
    elif not nodes:
        st.info("No lineage evidence found for this dataset folder.")


def _render_scenario_detail(scenario: str) -> None:
    st.subheader(f"Scenario: `{scenario}`")
    status = reg.get_scenario_status(scenario)
    st.write("Derived status (filesystem-only):")
    st.json(stable_jsonable(status))

    try:
        cfg = reg.get_scenario_config(scenario)
    except Exception as exc:
        st.error(str(exc))
        return

    left, right = st.columns(2)
    with left:
        st.markdown("**Config**")
        st.write(
            {
                "display_name": cfg.display_name,
                "dataset": cfg.dataset,
                "backbone_type": cfg.backbone_type,
                "weights": cfg.weights,
                "base_model": cfg.base_model,
                "postproc": cfg.postproc,
            }
        )
        st.markdown("**Artifacts**")
        weights_stat = stat_path(cfg.weights_path) if str(cfg.weights_path) else None
        if weights_stat is not None:
            st.write(
                {
                    "weights_path": weights_stat.path,
                    "exists": weights_stat.exists,
                    "size": human_bytes(weights_stat.size_bytes),
                }
            )
    with right:
        st.markdown("**Training guard**")
        st.json(stable_jsonable(status.get("training_guard") or {}))

    with st.expander("Scenario YAML (raw)", expanded=False):
        st.code(cfg.config_path.read_text(encoding="utf-8", errors="replace"), language="yaml")

    st.divider()
    st.subheader("Dataset")
    if not cfg.dataset:
        st.warning("Scenario has no dataset configured.")
        return

    try:
        info = _inspect_dataset(cfg.dataset)
    except Exception as exc:
        st.error(f"Failed to inspect dataset: {exc}")
        return

    st.write(
        {
            "slug": info.get("slug"),
            "path": info.get("path"),
            "format": info.get("format"),
            "category": info.get("category"),
            "split_counts": info.get("split_counts", {}),
            "classes": info.get("classes", []),
        }
    )
    _render_dataset_lineage(info)

    if info.get("format") == reg.LIBRARY_DATASET_FORMAT_YOLO:
        base = Path(str(info["path"]))
        pair = check_yolo_pairs(base)
        st.markdown("**Pairing checks (YOLO)**")
        st.write(stable_jsonable(pair.__dict__))

        labels_root = base / "labels"
        label_summary = summarize_yolo_labels(labels_root)
        st.markdown("**Label summary (YOLO)**")
        st.write(
            {
                "label_files_scanned": label_summary.label_files_scanned,
                "objects_total": label_summary.objects_total,
                "class_counts": label_summary.class_counts,
                "parse_errors": label_summary.parse_errors[:10],
            }
        )
        if label_summary.class_counts:
            cc = (
                pd.DataFrame(
                    [{"class_id": k, "objects": v} for k, v in sorted(label_summary.class_counts.items())]
                )
                .sort_values("objects", ascending=False)
                .reset_index(drop=True)
            )
            st.dataframe(cc, use_container_width=True, hide_index=True)
    elif info.get("format") == reg.LIBRARY_DATASET_FORMAT_CSV:
        st.info("This dataset folder contains CSV files. Use the Tabular panel for deep inspection.")
    else:
        st.info("Basic inspection available; dataset-specific deep checks are limited for this format.")

    with st.expander("Raw dataset inspection (JSON)", expanded=False):
        st.json(info)


def _render_tabular_panel() -> None:
    st.subheader("Tabular datasets")
    entries = reg.list_tabular_dataset_entries()
    if not entries:
        st.info("No tabular datasets found under `mlops/datasets/`.")
        return

    df = pd.DataFrame(entries)
    st.dataframe(df, use_container_width=True, hide_index=True)

    pick = st.selectbox(
        "Dataset file",
        options=[e["path"] for e in entries],
        format_func=lambda p: Path(str(p)).name,
    )
    path = (reg.REPO_ROOT / pick).resolve()
    if not path.exists():
        st.error(f"Missing file: {path}")
        return

    nrows = st.number_input("Rows to load (0 = full file)", min_value=0, value=50_000, step=10_000)
    nrows_opt = None if int(nrows) == 0 else int(nrows)

    with st.spinner("Loading CSV..."):
        frame = load_csv(path, nrows=nrows_opt)

    summary = summarize_frame(frame)
    st.markdown("**Summary**")
    st.write(stable_jsonable(summary.__dict__))

    checks = quality_checks(frame)
    st.markdown("**Quality checks**")
    st.write(stable_jsonable(checks.__dict__))

    st.markdown("**Preview**")
    st.dataframe(frame.head(200), use_container_width=True)

    with st.expander("Describe (include='all')", expanded=False):
        try:
            st.dataframe(frame.describe(include="all").transpose(), use_container_width=True)
        except Exception as exc:
            st.error(str(exc))

    numeric_cols = [c for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])]
    if numeric_cols:
        st.markdown("**Numeric distribution**")
        col = st.selectbox("Column", options=numeric_cols)
        st.bar_chart(frame[col].dropna().value_counts(bins=30).sort_index())

        with st.expander("Correlation (numeric-only)", expanded=False):
            try:
                corr = frame[numeric_cols].corr(numeric_only=True)
                st.dataframe(corr, use_container_width=True)
            except Exception as exc:
                st.error(str(exc))


def _render_datasets_panel() -> None:
    st.subheader("Dataset library")
    names = reg.list_library_dataset_names()
    if not names:
        st.info("No dataset folders found under `database/` or `mlops/datasets/`.")
        return

    dataset = st.selectbox("Dataset", options=names)
    try:
        info = _inspect_dataset(dataset)
    except Exception as exc:
        st.error(str(exc))
        return

    st.write(
        {
            "slug": dataset,
            "path": info.get("path"),
            "format": info.get("format"),
            "category": info.get("category"),
            "split_counts": info.get("split_counts", {}),
            "classes": info.get("classes", []),
        }
    )
    _render_dataset_lineage(info)
    if info.get("format") == reg.LIBRARY_DATASET_FORMAT_YOLO:
        base = Path(str(info["path"]))
        pair = check_yolo_pairs(base)
        st.markdown("**Pairing checks (YOLO)**")
        st.write(stable_jsonable(pair.__dict__))

    with st.expander("Raw dataset inspection (JSON)", expanded=False):
        st.json(info)


def _render_health_panel(statuses: list[dict[str, Any]]) -> None:
    st.subheader("Health checks")
    st.caption("Static checks derived from registry/config/filesystem; no services required.")

    problems: list[dict[str, Any]] = []
    overflow_rows: list[dict[str, Any]] = []
    for s in statuses:
        if s.get("status") == "error":
            problems.append({"scenario": s.get("name"), "type": "scenario_error", "detail": s.get("error")})
        if not s.get("base_model_exists", True) and s.get("base_model"):
            problems.append({"scenario": s.get("name"), "type": "missing_base_model", "detail": s.get("base_model")})
        if s.get("dataset") and int(s.get("dataset_count") or 0) == 0:
            problems.append({"scenario": s.get("name"), "type": "empty_dataset", "detail": s.get("dataset")})
        if s.get("weights") and s.get("weights_ready") is False:
            problems.append({"scenario": s.get("name"), "type": "missing_or_small_weights", "detail": s.get("weights")})
        guard = s.get("training_guard") if isinstance(s.get("training_guard"), dict) else {}
        overflow = guard.get("overflow_protocol") if isinstance(guard.get("overflow_protocol"), dict) else {}
        if overflow:
            if overflow.get("status") in {"overflow", "no_space"}:
                problems.append({
                    "scenario": s.get("name"),
                    "type": f"overflow_protocol_{overflow.get('status')}",
                    "detail": overflow.get("message"),
                })
            for drive in overflow.get("drives") or []:
                if not isinstance(drive, dict):
                    continue
                overflow_rows.append({
                    "scenario": s.get("name"),
                    "active": bool(drive.get("active")),
                    "label": drive.get("label"),
                    "free_gb": drive.get("free_gb"),
                    "total_gb": drive.get("total_gb"),
                    "used_pct": drive.get("used_pct"),
                    "asset_root": drive.get("asset_root"),
                })

    if not problems:
        st.success("No issues detected.")
    else:
        st.warning(f"{len(problems)} issue(s) detected.")
        st.dataframe(pd.DataFrame(problems), use_container_width=True, hide_index=True)

    if overflow_rows:
        st.markdown("**Overflow protocol storage state**")
        st.dataframe(pd.DataFrame(overflow_rows), use_container_width=True, hide_index=True)


def main() -> None:
    _page_header()

    payload = _load_registry_payload()
    scenario_entries = payload.get("scenarios") if isinstance(payload, dict) else []
    scenario_names = [
        str(s.get("name"))
        for s in (scenario_entries or [])
        if isinstance(s, dict) and s.get("enabled", True) and s.get("name")
    ]
    scenario_names = sorted(set(scenario_names), key=lambda x: x.lower())
    statuses = _scenario_statuses(scenario_names)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Page",
        options=[
            "Overview",
            "Scenario",
            "Datasets",
            "Tabular",
            "Custom Cells",
            "Scrap",
            "Image-to-3D (base)",
            "3D Generation",
            "Health",
        ],
    )

    if page == "Overview":
        _render_registry_overview(statuses)
    elif page == "Scenario":
        scenario = st.sidebar.selectbox("Scenario", options=scenario_names)
        _render_scenario_detail(scenario)
    elif page == "Datasets":
        _render_datasets_panel()
    elif page == "Tabular":
        _render_tabular_panel()
    elif page == "Custom Cells":
        custom_cells_panel.render(scenario_names)
    elif page == "Scrap":
        scrap_panel.render()
    elif page == "Image-to-3D (base)":
        image_to_3d_panel.render()
    elif page == "3D Generation":
        three_d_panel.render()
    else:
        _render_health_panel(statuses)

    with st.sidebar.expander("Registry (raw)", expanded=False):
        st.json(stable_jsonable(payload))


if __name__ == "__main__":
    main()
