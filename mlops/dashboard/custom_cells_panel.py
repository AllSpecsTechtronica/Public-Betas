"""Colab-style editor/runner for mlops/custom_cells/<scenario>/draft/ cells."""
from __future__ import annotations

import contextlib
import io
import time
import traceback
import uuid
from typing import Any

import streamlit as st

from mlops.pipeline import custom_cells_store as ccs
from mlops.pipeline import registry as reg
from mlops.pipeline.backbone import BackboneContext, CellResult
from mlops.pipeline.python_cells import PythonFileCell

try:
    from streamlit_ace import st_ace  # type: ignore
    _ACE_AVAILABLE = True
except Exception:
    _ACE_AVAILABLE = False


_DEFAULT_CELL_CODE = (
    "# [CUSTOM CELL] Colab-style: top-level statements run as the cell.\n"
    "print('[INFO] hello from cell')\n"
    "\n"
    "# Optional: define `run(ctx, prev)` to receive context and forward data\n"
    "# to later cells. If defined, it runs after the top-level statements.\n"
    "#\n"
    "# def run(ctx, prev):\n"
    "#     return {'data': {'key': 'value'}}\n"
)


def _available_scenarios() -> list[str]:
    names: set[str] = set()
    root = reg.MLOPS_ROOT / "custom_cells"
    if root.is_dir():
        for child in root.iterdir():
            if child.is_dir() and (child / "draft").is_dir():
                names.add(child.name)
    try:
        payload = reg.load_registry()
        if isinstance(payload, dict):
            for s in payload.get("scenarios") or []:
                if isinstance(s, dict) and s.get("name"):
                    names.add(str(s["name"]))
    except Exception:
        pass
    return sorted(names, key=lambda s: s.lower())


def _state_key(scenario: str, cell_id: str, suffix: str) -> str:
    return f"ccpanel::{scenario}::{cell_id}::{suffix}"


def _editor(initial_value: str, key: str) -> str:
    if _ACE_AVAILABLE:
        return st_ace(
            value=initial_value,
            language="python",
            theme="monokai",
            key=key,
            height=320,
            show_gutter=True,
            show_print_margin=False,
            wrap=False,
            auto_update=True,
            font_size=13,
            tab_size=4,
        )
    if key not in st.session_state:
        st.session_state[key] = initial_value
    return st.text_area(
        "code",
        key=key,
        height=320,
        label_visibility="collapsed",
    )


def _minimal_context(scenario: str, cell_spec: dict[str, Any], script_path: str) -> BackboneContext:
    try:
        scen_cfg = reg.get_scenario_config(scenario)
    except Exception:
        scen_cfg = None
    cell_id = str(cell_spec.get("id") or "cell")
    return BackboneContext(
        scenario_config=scen_cfg,
        job_id=f"ccpanel-{uuid.uuid4().hex[:8]}",
        job_type="infer",
        image_bgr=None,
        payload={},
        cell_callback=lambda _event: None,
        datasets=[],
        active_cell={
            "id": cell_id,
            "name": str(cell_spec.get("name") or cell_id),
            "path": script_path,
            "entry": str(cell_spec.get("entry") or "run") or "run",
            "datasets": cell_spec.get("datasets") or [],
            "pasted_data_dir": str(ccs.pasted_data_dir(scenario, cell_id)),
        },
    )


def _run_cell_in_isolation(
    scenario: str,
    cell_spec: dict[str, Any],
    code_text: str,
) -> CellResult:
    cell_id = str(cell_spec.get("id") or "cell")
    cell_name = str(cell_spec.get("name") or cell_id)
    entry = str(cell_spec.get("entry") or "run") or "run"

    script_path = ccs.cell_script_path(scenario, cell_id)
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(code_text, encoding="utf-8")

    py_cell = PythonFileCell({
        "path": str(script_path),
        "entry": entry,
        "name": cell_name,
    })
    ctx = _minimal_context(scenario, cell_spec, str(script_path))

    buf = io.StringIO()
    started = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            result = py_cell.run(ctx, [])
    except AttributeError as exc:
        # Colab-style: cell has no `run(ctx, prev)` entrypoint, only top-level
        # statements. The module body already executed during load_module_from_file
        # inside py_cell.run; the stdout is in `buf`.
        if "does not export entrypoint" not in str(exc):
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            captured = buf.getvalue().rstrip()
            tb = traceback.format_exc(limit=8)
            err_output = (captured + "\n" + tb).strip() if captured else tb
            return CellResult(
                cell_name=cell_name,
                status="error",
                output=f"{err_output}\n[ERR] {type(exc).__name__}: {exc}".strip(),
                elapsed_ms=elapsed,
                data={},
            )
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        return CellResult(
            cell_name=cell_name,
            status="done",
            output=buf.getvalue().rstrip(),
            elapsed_ms=elapsed,
            data={},
        )
    except Exception as exc:
        elapsed = round((time.perf_counter() - started) * 1000, 2)
        captured = buf.getvalue().rstrip()
        tb = traceback.format_exc(limit=8)
        err_output = (captured + "\n" + tb).strip() if captured else tb
        return CellResult(
            cell_name=cell_name,
            status="error",
            output=f"{err_output}\n[ERR] {type(exc).__name__}: {exc}".strip(),
            elapsed_ms=elapsed,
            data={},
        )
    elapsed = round((time.perf_counter() - started) * 1000, 2)
    captured = buf.getvalue().rstrip()
    merged_output = captured
    if result.output:
        if captured and result.output not in captured:
            merged_output = (captured + "\n" + result.output).strip()
        elif not captured:
            merged_output = result.output
    return CellResult(
        cell_name=result.cell_name,
        status=result.status,
        output=merged_output,
        elapsed_ms=elapsed,
        data=dict(result.data or {}),
    )


def _gather_draft_payload(scenario: str, draft: dict[str, Any]) -> dict[str, Any]:
    """Build a write_draft payload from the current in-memory editor state."""
    cells_out: list[dict[str, Any]] = []
    for cell in draft.get("cells") or []:
        cid = str(cell.get("id") or "cell")
        code_key = _state_key(scenario, cid, "code")
        code = st.session_state.get(code_key)
        if code is None:
            code = cell.get("code") or ""
        cells_out.append({
            "id": cid,
            "name": str(cell.get("name") or cid),
            "entry": str(cell.get("entry") or "run") or "run",
            "code": code,
            "datasets": cell.get("datasets") or [],
        })
    return {
        "cells": cells_out,
        "scenario_datasets": draft.get("scenario_datasets") or [],
    }


def _render_cell(scenario: str, index: int, cell: dict[str, Any]) -> None:
    cell_id = str(cell.get("id") or f"cell{index}")
    name = str(cell.get("name") or cell_id)
    result_key = _state_key(scenario, cell_id, "last_result")
    code_key = _state_key(scenario, cell_id, "code")

    with st.container(border=True):
        header_left, header_right = st.columns([3, 1])
        with header_left:
            st.markdown(f"### Cell `{cell_id}` — {name}")
            st.caption(
                f"entry={cell.get('entry') or 'run'} | "
                f"path={cell.get('path') or '(unsaved)'}"
            )
        with header_right:
            run_clicked = st.button(
                "Run cell",
                key=_state_key(scenario, cell_id, "run_btn"),
                type="primary",
                use_container_width=True,
            )

        initial = cell.get("code") or _DEFAULT_CELL_CODE
        code_value = _editor(initial, key=code_key)

        datasets = cell.get("datasets") or []
        if datasets:
            with st.expander(f"Managed datasets ({len(datasets)})", expanded=False):
                st.json(datasets)

        if run_clicked:
            with st.spinner(f"Running {cell_id}..."):
                result = _run_cell_in_isolation(scenario, cell, code_value or "")
            st.session_state[result_key] = {
                "status": result.status,
                "output": result.output,
                "data": result.data,
                "elapsed_ms": result.elapsed_ms,
            }

        last = st.session_state.get(result_key)
        if last:
            status = str(last.get("status") or "done")
            tag = {"done": "[OK]", "error": "[ERR]", "skipped": "[SKIP]"}.get(
                status, f"[{status.upper()}]"
            )
            st.markdown(f"**{tag} status={status}** — {last.get('elapsed_ms')} ms")
            output = str(last.get("output") or "")
            if output:
                st.code(output, language="text")
            data = last.get("data") or {}
            if data:
                st.markdown("**Returned data**")
                st.json(data)
            elif status == "done":
                st.caption("(cell returned no data)")


def render(registered_scenarios: list[str] | None = None) -> None:
    st.subheader("Custom Cells")
    st.caption(
        "Colab-style Python drop-in. Each cell runs in isolation with prev=[] and a minimal "
        "BackboneContext. Save Draft persists editor state to the scenario draft directory."
    )
    if not _ACE_AVAILABLE:
        st.warning(
            "streamlit-ace not installed — falling back to plain text editor. "
            "Install with: pip install streamlit-ace"
        )

    found = _available_scenarios()
    scenarios = sorted(set(found) | set(registered_scenarios or []), key=lambda s: s.lower())

    pick_col, new_col = st.columns([3, 2])
    with pick_col:
        pick = st.selectbox(
            "Scenario",
            options=scenarios or ["(none)"],
            index=0 if scenarios else 0,
            help="Drafts live at mlops/custom_cells/<scenario>/draft/.",
        )
    with new_col:
        new_scenario = st.text_input(
            "or create new",
            value="",
            placeholder="e.g. my_experiment",
        )

    scenario = (new_scenario.strip() or (pick or "")).strip()
    if not scenario or scenario == "(none)":
        st.info("Pick or name a scenario to start editing cells.")
        return

    draft = ccs.read_draft(scenario)
    cells = draft.get("cells") or []

    if cells:
        for i, cell in enumerate(cells):
            _render_cell(scenario, i, cell)
    else:
        st.info("No cells yet. Click **Add cell** below to scaffold a blank one.")

    st.divider()
    act_add, act_save, act_del, _spacer = st.columns([1, 1, 1, 2])
    with act_add:
        if st.button("Add cell", use_container_width=True, key=f"ccpanel::{scenario}::add"):
            payload = _gather_draft_payload(scenario, draft)
            new_id = f"c{len(payload['cells']) + 1}"
            payload["cells"].append({
                "id": new_id,
                "name": new_id,
                "entry": "run",
                "code": _DEFAULT_CELL_CODE,
                "datasets": [],
            })
            ccs.write_draft(scenario, payload)
            st.rerun()
    with act_save:
        if st.button(
            "Save draft",
            use_container_width=True,
            key=f"ccpanel::{scenario}::save",
            disabled=not cells,
        ):
            payload = _gather_draft_payload(scenario, draft)
            ccs.write_draft(scenario, payload)
            st.success(f"Saved {len(payload['cells'])} cell(s) to draft.")
    with act_del:
        if st.button(
            "Delete last",
            use_container_width=True,
            key=f"ccpanel::{scenario}::del",
            disabled=not cells,
        ):
            payload = _gather_draft_payload(scenario, draft)
            payload["cells"] = payload["cells"][:-1]
            ccs.write_draft(scenario, payload)
            st.rerun()

    with st.expander("Manifest (raw, code stripped)", expanded=False):
        st.json({
            "scenario": draft.get("scenario"),
            "cells": [{k: v for k, v in c.items() if k != "code"} for c in cells],
            "scenario_datasets": draft.get("scenario_datasets") or [],
        })
