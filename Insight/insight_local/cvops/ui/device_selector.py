from __future__ import annotations

"""Reusable accelerator / machine selector.

A single source of truth for "which GPU / CPU runs this work". The same widget
backs the System Guard card's training-device picker and the Cell Space machine
selector, so both stay in sync on how devices are detected and labelled.

The combo stores a device token in each item's userData:
    ""      -> Auto (system default)
    "0"     -> CUDA device index 0 (etc.)
    "mps"   -> Apple Metal
    "cpu"   -> force CPU

`detect_gpu_entries()` is exposed separately so callers that already have a
detected inventory (e.g. from a service spec payload) can feed it in via
`set_entries()` instead of re-probing torch.
"""

from typing import Any, Callable, Optional

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QComboBox


def detect_gpu_entries() -> list[dict[str, Any]]:
    """Probe torch for available accelerators.

    Returns a list of dicts shaped like:
        {"index": int, "name": str, "memory_gb": float | None, "backend": str}
    where backend is one of "cuda" / "mps". Empty list means CPU-only.
    """
    entries: list[dict[str, Any]] = []
    try:
        import torch

        if torch.cuda.is_available():
            for idx in range(int(torch.cuda.device_count() or 0)):
                try:
                    name = str(torch.cuda.get_device_name(idx) or f"CUDA GPU {idx}")
                except Exception:
                    name = f"CUDA GPU {idx}"
                try:
                    mem_gb = round(
                        float(torch.cuda.get_device_properties(idx).total_memory) / (1024 ** 3),
                        1,
                    )
                except Exception:
                    mem_gb = None
                entries.append({"index": idx, "name": name, "memory_gb": mem_gb, "backend": "cuda"})
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            entries.append({"index": 0, "name": "Apple GPU", "memory_gb": None, "backend": "mps"})
    except Exception:
        pass
    return entries


class DeviceSelector(QComboBox):
    """Combo of detected accelerators with an Auto + CPU fallback.

    Emits ``deviceChanged(token)`` whenever the user picks a different device.
    """

    deviceChanged = pyqtSignal(str)

    def __init__(
        self,
        *,
        auto_label: str = "Auto (system default)",
        cpu_label: str = "CPU (skip GPU)",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._auto_label = auto_label
        self._cpu_label = cpu_label
        self._user_device = ""
        self._suspend = False
        self.setMinimumWidth(220)
        self.currentIndexChanged.connect(self._on_index_changed)
        self.set_entries(detect_gpu_entries())

    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Re-probe torch and rebuild the list (preserving the selection)."""
        self.set_entries(detect_gpu_entries())

    def set_entries(self, gpu_entries: list[dict[str, Any]]) -> None:
        """Rebuild from a detected inventory, preserving the prior selection."""
        self._suspend = True
        try:
            self.clear()
            self.addItem(self._auto_label, "")
            for entry in gpu_entries or []:
                if not isinstance(entry, dict):
                    continue
                try:
                    idx_int = int(entry.get("index"))
                except Exception:
                    idx_int = 0
                name = str(entry.get("name") or f"GPU {idx_int}")
                backend = str(entry.get("backend") or "").lower()
                pieces = [f"GPU {idx_int}: {name}"]
                mem = entry.get("memory_gb")
                if mem:
                    pieces.append(f"{mem} GB")
                if backend:
                    pieces.append(f"[{backend.upper()}]")
                label = "  ".join(pieces)
                if backend == "mps":
                    self.addItem(label, "mps")
                elif backend == "cuda":
                    self.addItem(label, str(idx_int))
                # Generic / unknown backends fall through to the CPU item.
            self.addItem(self._cpu_label, "cpu")

            target = self._user_device or ""
            found = self.findData(target) if target else 0
            if found < 0:
                found = 0
            self.setCurrentIndex(found)
            self.setEnabled(self.count() > 1)
        finally:
            self._suspend = False

    def device(self) -> str:
        """Return the selected device token ("" / "0" / "mps" / "cpu")."""
        return str(self.currentData() or "")

    def set_device(self, token: str) -> None:
        """Select a device token if present; remembered for future rebuilds."""
        self._user_device = str(token or "")
        idx = self.findData(self._user_device) if self._user_device else 0
        if idx < 0:
            idx = 0
        self.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    def _on_index_changed(self, _index: int) -> None:
        if self._suspend:
            return
        self._user_device = self.device()
        self.deviceChanged.emit(self._user_device)
