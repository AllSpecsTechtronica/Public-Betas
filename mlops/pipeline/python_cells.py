"""Shared helpers for file-backed Python execution cells (torch_tabular, custom_code).

Each cell references a ``.py`` file that exports a callable (default name ``run``).
Return values map to ``CellResult`` the same way across backbones.
"""
from __future__ import annotations

import hashlib
import importlib.util
import inspect
from pathlib import Path
from types import ModuleType
from typing import Any

from . import registry as _reg
from .backbone import BackboneCell, BackboneContext, CellResult


def resolve_repo_path(path_value: str) -> Path:
    value = str(path_value or "").strip()
    if not value:
        return Path("")
    p = Path(value)
    if not p.is_absolute():
        p = _reg.REPO_ROOT / p
    return p


def load_module_from_file(path: Path) -> ModuleType:
    resolved = path.resolve()
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(f"Cell file not found: {resolved}")
    try:
        mtime_ns = int(resolved.stat().st_mtime_ns)
    except Exception:
        mtime_ns = 0
    digest = hashlib.sha1(str(resolved).encode("utf-8", errors="replace")).hexdigest()[:10]
    mod_name = f"_cvops_cell_{digest}_{mtime_ns}"
    spec = importlib.util.spec_from_file_location(mod_name, str(resolved))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from file: {resolved}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def parse_cell_specs(raw: Any) -> list[dict[str, Any]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        return []
    specs: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            path = item.strip()
            if path:
                specs.append({"path": path})
            continue
        if isinstance(item, dict):
            path = str(item.get("path") or "").strip()
            if not path:
                continue
            spec = {k: v for k, v in item.items()}
            spec["path"] = path
            entry = str(spec.get("entry") or "").strip()
            if entry:
                spec["entry"] = entry
            else:
                spec["entry"] = "run"
            specs.append(spec)
            continue
    return specs


def call_user_cell_fn(fn: Any, ctx: BackboneContext, prev: list[CellResult]) -> Any:
    if not callable(fn):
        raise TypeError("Cell entrypoint is not callable")
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
    except Exception:
        params = []

    try:
        if len(params) >= 2:
            return fn(ctx, prev)
        if len(params) == 1:
            return fn(ctx)
        return fn()
    except TypeError:
        try:
            return fn(ctx, prev)
        except TypeError:
            try:
                return fn(ctx)
            except TypeError:
                return fn()


def user_result_to_cell_result(cell_name: str, result: Any) -> CellResult:
    if isinstance(result, CellResult):
        if result.cell_name != cell_name:
            return CellResult(
                cell_name=cell_name,
                status=result.status,
                output=result.output,
                elapsed_ms=result.elapsed_ms,
                data=result.data,
            )
        return result

    if isinstance(result, dict):
        payload = dict(result)
        data_raw = payload.get("data")
        if isinstance(data_raw, dict):
            data = dict(data_raw)
        else:
            data = {k: v for k, v in payload.items() if k not in {"status", "output", "data"}}
        return CellResult(
            cell_name=cell_name,
            status=str(payload.get("status") or "done"),
            output=str(payload.get("output") or ""),
            elapsed_ms=0,
            data=data,
        )

    if isinstance(result, str):
        return CellResult(
            cell_name=cell_name,
            status="done",
            output=result,
            elapsed_ms=0,
            data={},
        )

    return CellResult(
        cell_name=cell_name,
        status="done",
        output="",
        elapsed_ms=0,
        data={},
    )


class PythonFileCell(BackboneCell):
    """Load ``path`` on each run and invoke ``entry`` (default ``run``)."""

    def __init__(self, spec: dict[str, Any]) -> None:
        self._spec = dict(spec)
        path = str(spec.get("path") or "").strip()
        resolved = resolve_repo_path(path)
        self._resolved = resolved
        self._entry = str(spec.get("entry") or "run").strip() or "run"
        default_name = resolved.stem if str(resolved) not in ("", ".") else "Python Cell"
        self.name = str(spec.get("name") or default_name)
        self.description = str(spec.get("description") or "")

    def run(self, ctx: BackboneContext, prev: list[CellResult]) -> CellResult:
        mod = load_module_from_file(self._resolved)
        fn = getattr(mod, self._entry, None)
        if fn is None:
            raise AttributeError(
                f"Cell file '{self._resolved}' does not export entrypoint '{self._entry}'"
            )
        raw = call_user_cell_fn(fn, ctx, prev)
        return user_result_to_cell_result(self.name, raw)
