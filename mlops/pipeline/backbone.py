"""backbone.py — Abstract backbone interface for cvops scenario execution.

A backbone encapsulates a complete train/infer pipeline as a sequence of named
cells, mirroring the Google Colab cell model: each cell streams stdout in real-
time, runs sequentially, and emits WebSocket progress events so the UI can render
cell-by-cell output.

Concrete backbones include:
  - yolo_detection  (mlops.pipeline.backbones.yolo_detection)
  - torch_tabular   (mlops.pipeline.backbones.torch_tabular)
  - custom_code     (mlops.pipeline.backbones.custom_code)
  - llm_fine_tuning (mlops.pipeline.backbones.llm_fine_tuning)
"""
from __future__ import annotations

import contextlib
import io
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, ClassVar

if TYPE_CHECKING:
    from .registry import ScenarioConfig


class _LiveStdoutBuffer(io.StringIO):
    def __init__(self, on_emit: Callable[[str], None], emit_interval_s: float = 0.75) -> None:
        super().__init__()
        self._on_emit = on_emit
        self._emit_interval_s = max(0.1, float(emit_interval_s))
        self._last_emit_ts = 0.0
        self._last_emitted_len = 0

    def write(self, s: str) -> int:
        n = super().write(s)
        now = time.perf_counter()
        if "\n" in s or "\r" in s or (now - self._last_emit_ts) >= self._emit_interval_s:
            self._emit(now)
        return n

    def flush(self) -> None:
        self._emit(time.perf_counter())

    def _emit(self, now: float) -> None:
        raw = self.getvalue()
        if len(raw) <= self._last_emitted_len:
            return
        out = raw.rstrip()
        if not out:
            return
        self._on_emit(out)
        self._last_emit_ts = now
        self._last_emitted_len = len(raw)


@dataclass
class CellResult:
    """Result produced by a single BackboneCell run."""

    cell_name: str
    status: str          # "running" | "done" | "error" | "skipped"
    output: str          # captured stdout from print() calls inside the cell
    elapsed_ms: float
    data: dict[str, Any] = field(default_factory=dict)  # forwarded to subsequent cells


@dataclass
class BackboneContext:
    """Runtime context passed to every cell."""

    scenario_config: Any           # ScenarioConfig (avoid circular import at module level)
    job_id: str
    job_type: str                  # "train" | "infer"
    image_bgr: Any                 # np.ndarray | None — CV jobs only
    payload: dict[str, Any]        # full job payload dict
    cell_callback: Callable[[dict[str, Any]], None]  # fires cell_progress WS event
    # Optional: custom_code backbone sets scenario- and cell-level dataset views.
    datasets: list[dict[str, Any]] | None = None
    active_cell: dict[str, Any] | None = None


class BackboneCell(ABC):
    """One executable step in a backbone pipeline."""

    name: ClassVar[str] = "cell"
    description: ClassVar[str] = ""

    @abstractmethod
    def run(
        self,
        ctx: BackboneContext,
        prev: list[CellResult],
    ) -> CellResult:
        """Execute the cell.  All print() output is captured by BackboneBase.run()."""
        ...


class BackboneBase(ABC):
    """Base class for backbone implementations.

    Subclasses must implement ``cells`` (property returning the ordered cell list
    for the current job type) and ``_build_result`` (final result assembly).

    The ``run()`` method is fully implemented here: it iterates cells, captures
    stdout, fires WS events, and aborts on first error.
    """

    backbone_type: ClassVar[str] = "base"

    @property
    @abstractmethod
    def cells(self) -> list[BackboneCell]:
        """Return the ordered list of cells for the current execution context."""
        ...

    @abstractmethod
    def _build_result(
        self,
        ctx: BackboneContext,
        cell_results: list[CellResult],
    ) -> dict[str, Any]:
        """Assemble the final job result dict from accumulated cell results."""
        ...

    def run(self, ctx: BackboneContext) -> dict[str, Any]:
        """Drive all cells sequentially, emitting progress events before and after each."""
        cell_list = self.cells
        results: list[CellResult] = []

        for index, cell in enumerate(cell_list):
            # Announce cell start.
            ctx.cell_callback({
                "cell_index": index,
                "cell_name": cell.name,
                "cell_status": "running",
                "output": "",
                "elapsed_ms": 0,
            })

            started = time.perf_counter()
            def _emit_running_output(text: str) -> None:
                elapsed_running = round((time.perf_counter() - started) * 1000, 2)
                ctx.cell_callback({
                    "cell_index": index,
                    "cell_name": cell.name,
                    "cell_status": "running",
                    "output": text,
                    "elapsed_ms": elapsed_running,
                })

            buf = _LiveStdoutBuffer(on_emit=_emit_running_output)
            cell_result: CellResult
            try:
                with contextlib.redirect_stdout(buf):
                    cell_result = cell.run(ctx, results)
            except Exception as exc:
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                buf.flush()
                captured = buf.getvalue()
                err_output = f"{captured}\n[ERR] {exc}".strip()
                cell_result = CellResult(
                    cell_name=cell.name,
                    status="error",
                    output=err_output,
                    elapsed_ms=elapsed,
                )
                ctx.cell_callback({
                    "cell_index": index,
                    "cell_name": cell.name,
                    "cell_status": "error",
                    "output": err_output,
                    "elapsed_ms": elapsed,
                })
                results.append(cell_result)
                # Remaining cells are skipped.
                for skip_i, skip_cell in enumerate(cell_list[index + 1:], start=index + 1):
                    ctx.cell_callback({
                        "cell_index": skip_i,
                        "cell_name": skip_cell.name,
                        "cell_status": "skipped",
                        "output": "",
                        "elapsed_ms": 0,
                    })
                break
            else:
                elapsed = round((time.perf_counter() - started) * 1000, 2)
                # Merge any captured stdout into the result's output.
                buf.flush()
                captured = buf.getvalue()
                if captured and not cell_result.output:
                    # Replace blank output with captured stdout.
                    cell_result = CellResult(
                        cell_name=cell_result.cell_name,
                        status=cell_result.status,
                        output=captured.rstrip(),
                        elapsed_ms=elapsed,
                        data=cell_result.data,
                    )
                elif captured:
                    cell_result = CellResult(
                        cell_name=cell_result.cell_name,
                        status=cell_result.status,
                        output=(captured.rstrip() + "\n" + cell_result.output).strip(),
                        elapsed_ms=elapsed,
                        data=cell_result.data,
                    )
                else:
                    cell_result = CellResult(
                        cell_name=cell_result.cell_name,
                        status=cell_result.status,
                        output=cell_result.output,
                        elapsed_ms=elapsed,
                        data=cell_result.data,
                    )
                ctx.cell_callback({
                    "cell_index": index,
                    "cell_name": cell.name,
                    "cell_status": cell_result.status,
                    "output": cell_result.output,
                    "elapsed_ms": elapsed,
                })
                results.append(cell_result)
                if cell_result.status == "error":
                    # Cell signalled error without raising — skip remaining.
                    for skip_i, skip_cell in enumerate(cell_list[index + 1:], start=index + 1):
                        ctx.cell_callback({
                            "cell_index": skip_i,
                            "cell_name": skip_cell.name,
                            "cell_status": "skipped",
                            "output": "",
                            "elapsed_ms": 0,
                        })
                    break

        return self._build_result(ctx, results)
