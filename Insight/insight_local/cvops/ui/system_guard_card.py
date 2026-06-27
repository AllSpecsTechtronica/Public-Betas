from __future__ import annotations

import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .device_selector import DeviceSelector
from .status_pill import StatusPill

try:
    import psutil as _psutil
except Exception:
    _psutil = None

try:
    import pynvml as _pynvml
    _pynvml.nvmlInit()
    _NVML_OK = True
except Exception:
    _pynvml = None
    _NVML_OK = False


_STATUS_PILL_MAP = {
    "ok": "ready",
    "adjusted": "trained",
    "blocked": "error",
}


# nvmlClocksThrottleReasons bit flags (from NVML headers). Documented here so
# we don't need to pull in the enum names, which move between pynvml versions.
_THROTTLE_REASONS: tuple[tuple[int, str], ...] = (
    (0x0000000000000001, "GPU_IDLE"),
    (0x0000000000000002, "APP_CLK_SETTING"),
    (0x0000000000000004, "SW_POWER_CAP"),
    (0x0000000000000008, "HW_SLOWDOWN"),
    (0x0000000000000010, "SYNC_BOOST"),
    (0x0000000000000020, "SW_THERMAL"),
    (0x0000000000000040, "HW_THERMAL"),
    (0x0000000000000080, "HW_POWER_BRAKE"),
    (0x0000000000000100, "DISPLAY_CLK_SETTING"),
)


def _bar(pct: float, width: int = 10) -> str:
    pct = max(0.0, min(100.0, float(pct)))
    filled = int(round((pct / 100.0) * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


_BAR_QSS_LOW = (
    "QProgressBar{border:none;border-top:1px solid rgba(90,104,98,0.55);"
    "border-bottom:1px solid rgba(90,104,98,0.55);background:#1A211E;height:8px;}"
    "QProgressBar::chunk{background:#5aaaaa;}"
)
_BAR_QSS_WARN = (
    "QProgressBar{border:none;border-top:1px solid rgba(90,104,98,0.55);"
    "border-bottom:1px solid rgba(90,104,98,0.55);background:#1A211E;height:8px;}"
    "QProgressBar::chunk{background:#c57a2e;}"
)
_BAR_QSS_HIGH = (
    "QProgressBar{border:none;border-top:1px solid rgba(90,104,98,0.55);"
    "border-bottom:1px solid rgba(90,104,98,0.55);background:#1A211E;height:8px;}"
    "QProgressBar::chunk{background:#aa5555;}"
)


def _make_bar(width: int = 110) -> QProgressBar:
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(0)
    bar.setTextVisible(False)
    bar.setFixedWidth(width)
    bar.setFixedHeight(8)
    bar.setStyleSheet(_BAR_QSS_LOW)
    return bar


def _set_bar(bar: QProgressBar, pct: float) -> None:
    v = max(0, min(100, int(round(pct))))
    bar.setValue(v)
    if pct >= 90:
        bar.setStyleSheet(_BAR_QSS_HIGH)
    elif pct >= 70:
        bar.setStyleSheet(_BAR_QSS_WARN)
    else:
        bar.setStyleSheet(_BAR_QSS_LOW)


def _nvml_clock(handle: Any, clock_type: int) -> Optional[int]:
    try:
        return int(_pynvml.nvmlDeviceGetClockInfo(handle, clock_type))
    except Exception:
        return None


def _nvml_max_clock(handle: Any, clock_type: int) -> Optional[int]:
    try:
        return int(_pynvml.nvmlDeviceGetMaxClockInfo(handle, clock_type))
    except Exception:
        return None


def _cuda_telemetry(idx: int) -> dict[str, Any]:
    """Return a data-rich telemetry bundle for CUDA device idx.

    Fields default to None when unsupported so the renderer can skip them
    gracefully. Only the NVML-backed path populates throttle/pcie/clocks.
    """
    out: dict[str, Any] = {
        "util_pct": None,
        "mem_used_gb": None,
        "mem_total_gb": None,
        "temp_c": None,
        "power_w": None,
        "power_cap_w": None,
        "fan_pct": None,
        "sm_clock_mhz": None,
        "sm_clock_max_mhz": None,
        "mem_clock_mhz": None,
        "mem_clock_max_mhz": None,
        "pcie_gen": None,
        "pcie_gen_max": None,
        "pcie_width": None,
        "pcie_width_max": None,
        "pcie_tx_mb_s": None,
        "pcie_rx_mb_s": None,
        "throttle_reasons": [],
    }
    gib = 1024 ** 3
    if _NVML_OK and _pynvml is not None:
        try:
            handle = _pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception:
            handle = None
        if handle is not None:
            try:
                util = _pynvml.nvmlDeviceGetUtilizationRates(handle)
                out["util_pct"] = float(util.gpu)
            except Exception:
                pass
            try:
                mem = _pynvml.nvmlDeviceGetMemoryInfo(handle)
                out["mem_used_gb"] = mem.used / gib
                out["mem_total_gb"] = mem.total / gib
            except Exception:
                pass
            try:
                out["temp_c"] = float(
                    _pynvml.nvmlDeviceGetTemperature(handle, _pynvml.NVML_TEMPERATURE_GPU)
                )
            except Exception:
                pass
            try:
                out["power_w"] = _pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                pass
            try:
                out["power_cap_w"] = _pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0
            except Exception:
                pass
            try:
                out["fan_pct"] = float(_pynvml.nvmlDeviceGetFanSpeed(handle))
            except Exception:
                pass
            # pynvml exposes named constants but they shift between versions,
            # so we lazily look them up and fall back to int values.
            sm_const = getattr(_pynvml, "NVML_CLOCK_SM", 1)
            mem_const = getattr(_pynvml, "NVML_CLOCK_MEM", 2)
            out["sm_clock_mhz"] = _nvml_clock(handle, sm_const)
            out["sm_clock_max_mhz"] = _nvml_max_clock(handle, sm_const)
            out["mem_clock_mhz"] = _nvml_clock(handle, mem_const)
            out["mem_clock_max_mhz"] = _nvml_max_clock(handle, mem_const)
            try:
                out["pcie_gen"] = int(_pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))
            except Exception:
                pass
            try:
                out["pcie_gen_max"] = int(_pynvml.nvmlDeviceGetMaxPcieLinkGeneration(handle))
            except Exception:
                pass
            try:
                out["pcie_width"] = int(_pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))
            except Exception:
                pass
            try:
                out["pcie_width_max"] = int(_pynvml.nvmlDeviceGetMaxPcieLinkWidth(handle))
            except Exception:
                pass
            # nvmlDeviceGetPcieThroughput returns KB/s per direction (0=tx, 1=rx).
            try:
                tx_const = getattr(_pynvml, "NVML_PCIE_UTIL_TX_BYTES", 0)
                rx_const = getattr(_pynvml, "NVML_PCIE_UTIL_RX_BYTES", 1)
                tx_kbs = _pynvml.nvmlDeviceGetPcieThroughput(handle, tx_const)
                rx_kbs = _pynvml.nvmlDeviceGetPcieThroughput(handle, rx_const)
                out["pcie_tx_mb_s"] = float(tx_kbs) / 1024.0
                out["pcie_rx_mb_s"] = float(rx_kbs) / 1024.0
            except Exception:
                pass
            try:
                mask = int(_pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))
                # Mask off the benign "GPU_IDLE" flag — it's always set when the
                # card is sitting at idle and would clutter the readout.
                reasons = [name for bit, name in _THROTTLE_REASONS if (mask & bit) and name != "GPU_IDLE"]
                out["throttle_reasons"] = reasons
            except Exception:
                pass
    if out["mem_used_gb"] is None or out["mem_total_gb"] is None:
        try:
            import torch
            if torch.cuda.is_available():
                out["mem_used_gb"] = torch.cuda.memory_reserved(idx) / gib
                out["mem_total_gb"] = torch.cuda.get_device_properties(idx).total_memory / gib
        except Exception:
            pass
    return out


class SystemGuardCard(QFrame):
    """Displays the mlops training guard: host specs, requested vs effective,
    adjustments and a status pill."""

    guardProfileChanged = pyqtSignal(str)
    deviceChanged = pyqtSignal(str)
    storageRootChanged = pyqtSignal(str)
    refreshRequested = pyqtSignal()

    def __init__(self, parent: Optional[QWidget] = None, *, show_title: bool = True) -> None:
        super().__init__(parent)
        self.setObjectName("systemGuardCard")
        self.setFrameShape(QFrame.Shape.StyledPanel)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(5)

        self._pill = StatusPill("empty")
        if show_title:
            head = QHBoxLayout()
            title = QLabel("System & Guard")
            title.setProperty("isTitle", True)
            title.setStyleSheet("font-size: 10px; font-weight: 600; border: none;")
            head.addWidget(title, stretch=0)
            head.addStretch(1)
            head.addWidget(self._pill)
            outer.addLayout(head)
        else:
            head = QHBoxLayout()
            head.addStretch(1)
            head.addWidget(self._pill)
            outer.addLayout(head)

        self._host_label = QLabel("Host: [UNKNOWN]")
        self._host_label.setWordWrap(True)
        outer.addWidget(self._host_label)

        # CPU spec + live bars
        self._cpu_label = QLabel("")
        self._cpu_label.setWordWrap(True)
        self._cpu_label.setVisible(False)
        outer.addWidget(self._cpu_label)

        self._cpu_bars_widget = QWidget()
        self._cpu_bars_widget.setVisible(False)
        cpu_bars_row = QHBoxLayout(self._cpu_bars_widget)
        cpu_bars_row.setContentsMargins(0, 0, 0, 0)
        cpu_bars_row.setSpacing(6)
        _lbl_style = "border: none; font-size: 9px; color: rgba(168,180,172,0.65);"
        cpu_lbl = QLabel("CPU")
        cpu_lbl.setStyleSheet(_lbl_style)
        cpu_lbl.setFixedWidth(30)
        cpu_bars_row.addWidget(cpu_lbl)
        self._cpu_bar = _make_bar()
        cpu_bars_row.addWidget(self._cpu_bar)
        self._cpu_pct_lbl = QLabel("")
        self._cpu_pct_lbl.setStyleSheet("border: none; font-size: 9px; font-family: monospace; min-width: 36px;")
        cpu_bars_row.addWidget(self._cpu_pct_lbl)
        ram_lbl = QLabel("RAM")
        ram_lbl.setStyleSheet(_lbl_style)
        ram_lbl.setFixedWidth(30)
        cpu_bars_row.addWidget(ram_lbl)
        self._ram_bar = _make_bar()
        cpu_bars_row.addWidget(self._ram_bar)
        self._ram_lbl = QLabel("")
        self._ram_lbl.setStyleSheet("border: none; font-size: 9px; font-family: monospace;")
        cpu_bars_row.addWidget(self._ram_lbl)
        cpu_bars_row.addStretch(1)
        outer.addWidget(self._cpu_bars_widget)

        # GPU spec + live bars (one row per device)
        self._gpu_label = QLabel("")
        self._gpu_label.setWordWrap(True)
        self._gpu_label.setTextFormat(Qt.TextFormat.PlainText)
        self._gpu_label.setVisible(False)
        outer.addWidget(self._gpu_label)

        self._gpu_bars_container = QWidget()
        self._gpu_bars_container.setVisible(False)
        self._gpu_bars_layout = QVBoxLayout(self._gpu_bars_container)
        self._gpu_bars_layout.setContentsMargins(0, 0, 0, 0)
        self._gpu_bars_layout.setSpacing(2)
        outer.addWidget(self._gpu_bars_container)
        self._gpu_bar_rows: list[tuple[QLabel, QProgressBar, QLabel, QProgressBar, QLabel]] = []

        # Guard control echo (always visible) so train overrides remain readable
        # even if platform/theme styling causes combo rows to render poorly.
        self._control_echo_label = QLabel("")
        self._control_echo_label.setWordWrap(True)
        self._control_echo_label.setTextFormat(Qt.TextFormat.PlainText)
        self._control_echo_label.setStyleSheet(
            "border: none; font-size: 10px; font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace;"
        )
        self._control_echo_label.setVisible(False)
        outer.addWidget(self._control_echo_label)

        profile_row = QHBoxLayout()
        profile_lbl = QLabel("Guard profile:")
        profile_row.addWidget(profile_lbl)
        self._profile_combo = QComboBox()
        self._profile_combo.addItem("Balanced", "balanced")
        self._profile_combo.addItem("Stable", "stable")
        self._profile_combo.addItem("Fast", "fast")
        self._profile_combo.setToolTip("Controls auto batch/workers/imgsz limits for local training stability.")
        self._profile_combo.currentIndexChanged.connect(self._on_profile_combo_changed)
        profile_row.addWidget(self._profile_combo)
        profile_row.addStretch(1)
        outer.addLayout(profile_row)

        # Device selector: where the training run executes (which GPU / CPU).
        device_row = QHBoxLayout()
        device_lbl = QLabel("Training device:")
        device_lbl.setStyleSheet("border: none;")
        device_row.addWidget(device_lbl)
        self._device_combo = DeviceSelector()
        self._device_combo.setToolTip(
            "Choose which accelerator runs this training. Auto picks the detected "
            "default; pick a specific GPU index to pin to that device, or CPU to "
            "skip GPU entirely."
        )
        self._device_combo.currentIndexChanged.connect(self._on_device_combo_changed)
        device_row.addWidget(self._device_combo)
        device_row.addStretch(1)
        outer.addLayout(device_row)

        # Storage destination: which drive receives weights, checkpoints, caches.
        storage_row = QHBoxLayout()
        storage_lbl = QLabel("Training save root:")
        storage_lbl.setStyleSheet("border: none;")
        storage_row.addWidget(storage_lbl)
        self._storage_combo = QComboBox()
        self._storage_combo.setToolTip(
            "Where the entire training run (weights, checkpoints, caches, logs) is "
            "stored. Pick a roomy external drive to keep the system disk free."
        )
        self._storage_combo.setMinimumWidth(360)
        self._storage_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._storage_combo.addItem("Auto (overflow protocol)", "")
        self._storage_combo.currentIndexChanged.connect(self._on_storage_combo_changed)
        storage_row.addWidget(self._storage_combo, stretch=1)
        self._storage_browse_btn = QPushButton("Browse...")
        self._storage_browse_btn.setToolTip("Pick a custom directory to store this training run.")
        self._storage_browse_btn.clicked.connect(self._on_storage_browse)
        storage_row.addWidget(self._storage_browse_btn)
        outer.addLayout(storage_row)

        actions_row = QHBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        self._refresh_btn = QPushButton("Refresh Guard")
        self._refresh_btn.setProperty("variant", "ghost")
        self._refresh_btn.clicked.connect(self._on_refresh_guard)
        actions_row.addWidget(self._refresh_btn)
        self._reset_overrides_btn = QPushButton("Reset Overrides")
        self._reset_overrides_btn.setProperty("variant", "ghost")
        self._reset_overrides_btn.clicked.connect(self._on_reset_overrides)
        actions_row.addWidget(self._reset_overrides_btn)
        actions_row.addStretch(1)
        outer.addLayout(actions_row)

        self._overflow_toggle = QToolButton()
        self._overflow_toggle.setText("Overflow Destinations")
        self._overflow_toggle.setCheckable(True)
        self._overflow_toggle.setChecked(True)
        self._overflow_toggle.toggled.connect(self._on_overflow_toggled)
        outer.addWidget(self._overflow_toggle)

        # Active overflow / disk state — structured per-drive rows with color coding.
        self._overflow_container = QWidget()
        self._overflow_container.setVisible(False)
        self._overflow_layout = QVBoxLayout(self._overflow_container)
        self._overflow_layout.setContentsMargins(0, 0, 0, 0)
        self._overflow_layout.setSpacing(1)
        outer.addWidget(self._overflow_container)

        self._cache_status_label = QLabel("")
        self._cache_status_label.setWordWrap(True)
        self._cache_status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._cache_status_label.setStyleSheet(
            "border: none; font-size: 10px; font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace;"
            "color: rgba(170,170,170,0.95);"
        )
        outer.addWidget(self._cache_status_label)

        self._storage_pressure_label = QLabel("")
        self._storage_pressure_label.setWordWrap(True)
        self._storage_pressure_label.setTextFormat(Qt.TextFormat.PlainText)
        self._storage_pressure_label.setStyleSheet(
            "border: none; font-size: 10px; font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace;"
            "color: rgba(220, 50, 47, 0.95);"
        )
        self._storage_pressure_label.setVisible(False)
        outer.addWidget(self._storage_pressure_label)

        self._user_device: str = ""
        self._user_storage: str = ""
        self._suspend_signals: bool = False

        grid = QGridLayout()
        grid.setContentsMargins(0, 2, 0, 2)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(3)
        header_style = "border: none; font-weight: 600; font-size: 9px;"
        h0 = QLabel("")
        h0.setStyleSheet(header_style)
        h1 = QLabel("Requested")
        h1.setStyleSheet(header_style)
        h2 = QLabel("Effective")
        h2.setStyleSheet(header_style)
        h3 = QLabel("Limit")
        h3.setStyleSheet(header_style)
        grid.addWidget(h0, 0, 0)
        grid.addWidget(h1, 0, 1)
        grid.addWidget(h2, 0, 2)
        grid.addWidget(h3, 0, 3)
        self._cells: dict[str, tuple[QLabel, QLabel, QLabel]] = {}
        for row, key in enumerate(["imgsz", "batch", "workers", "epochs"], start=1):
            name = QLabel(key)
            name.setStyleSheet("border: none; font-size: 10px;")
            req = QLabel("-")
            req.setStyleSheet("border: none; font-size: 10px;")
            eff = QLabel("-")
            eff.setStyleSheet("border: none; font-size: 10px;")
            lim = QLabel("-")
            lim.setStyleSheet("border: none; font-size: 10px; color: rgba(160,160,160,0.9);")
            grid.addWidget(name, row, 0)
            grid.addWidget(req, row, 1)
            grid.addWidget(eff, row, 2)
            grid.addWidget(lim, row, 3)
            self._cells[key] = (req, eff, lim)
        grid.setColumnStretch(4, 1)
        outer.addLayout(grid)

        # Projected training footprint (VRAM/RAM) based on effective settings.
        self._projection_label = QLabel("")
        self._projection_label.setWordWrap(True)
        self._projection_label.setVisible(False)
        outer.addWidget(self._projection_label)

        # Derivation trail: shows how base -> profile -> scale -> clamp produced
        # the effective params. Monospace so the columns align.
        self._derivation_label = QLabel("")
        self._derivation_label.setWordWrap(False)
        self._derivation_label.setTextFormat(Qt.TextFormat.PlainText)
        self._derivation_label.setStyleSheet(
            "border: none; font-size: 9px; font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace;"
            "color: rgba(170,170,170,0.9);"
        )
        self._derivation_label.setVisible(False)
        self._derivation_toggle = QToolButton()
        self._derivation_toggle.setText("Derivation Trail")
        self._derivation_toggle.setCheckable(True)
        self._derivation_toggle.setChecked(False)
        self._derivation_toggle.setVisible(False)
        self._derivation_toggle.toggled.connect(self._on_derivation_toggled)
        outer.addWidget(self._derivation_toggle)
        outer.addWidget(self._derivation_label)

        self._adjust_label = QLabel("")
        self._adjust_label.setWordWrap(True)
        outer.addWidget(self._adjust_label)

        self._empty_hint = QLabel("No guard data available yet.")
        self._empty_hint.setStyleSheet("border: none; font-size: 10px; font-style: italic;")
        self._empty_hint.setVisible(True)
        outer.addWidget(self._empty_hint)

        self._cpu_spec_text: str = ""
        self._gpu_spec_entries: list[dict[str, Any]] = []
        self._fallback_host_label: str = "Host: [UNKNOWN]"
        self._fallback_cpu_spec_text: str = ""
        self._fallback_gpu_entries: list[dict[str, Any]] = []
        self._cache_probe_ts = 0.0
        self._cache_probe_text = ""
        self._load_runtime_host_fallback()
        if _psutil is not None:
            try:
                _psutil.cpu_percent(interval=None)
            except Exception:
                pass
        self._capacity_timer = QTimer(self)
        self._capacity_timer.setInterval(1500)
        self._capacity_timer.timeout.connect(self._refresh_capacity)
        self._capacity_timer.start()
        self._refresh_cache_status(force=True)
        self._on_overflow_toggled(self._overflow_toggle.isChecked())
        self._on_derivation_toggled(self._derivation_toggle.isChecked())
        self._refresh_control_echo()

    def _load_runtime_host_fallback(self) -> None:
        """Populate host/cpu/gpu lines without needing scenario guard data."""
        host_parts: list[str] = []
        sys_name = str(platform.system() or "")
        machine = str(platform.machine() or "")
        if sys_name or machine:
            host_parts.append(f"{sys_name} {machine}".strip())

        logical = None
        physical = None
        if _psutil is not None:
            try:
                vm = _psutil.virtual_memory()
                host_parts.append(f"{vm.total / (1024 ** 3):.1f} GB RAM")
            except Exception:
                pass
            try:
                logical = int(_psutil.cpu_count(logical=True) or 0) or None
            except Exception:
                logical = None
            try:
                physical = int(_psutil.cpu_count(logical=False) or 0) or None
            except Exception:
                physical = None

        gpu_entries: list[dict[str, Any]] = []
        accelerator = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                accelerator = "cuda"
                for idx in range(int(torch.cuda.device_count() or 0)):
                    try:
                        name = str(torch.cuda.get_device_name(idx) or f"CUDA GPU {idx}")
                    except Exception:
                        name = f"CUDA GPU {idx}"
                    try:
                        mem_gb = round(float(torch.cuda.get_device_properties(idx).total_memory) / (1024 ** 3), 1)
                    except Exception:
                        mem_gb = None
                    gpu_entries.append(
                        {
                            "index": idx,
                            "name": name,
                            "memory_gb": mem_gb,
                            "backend": "cuda",
                        }
                    )
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                accelerator = "mps"
                gpu_entries.append(
                    {
                        "index": 0,
                        "name": "Apple GPU",
                        "memory_gb": None,
                        "backend": "mps",
                    }
                )
        except Exception:
            pass
        host_parts.append(f"runtime={accelerator.upper()}")

        brand = str(platform.processor() or "").strip()
        cpu_bits: list[str] = []
        if brand:
            cpu_bits.append(brand)
        if logical:
            core_txt = f"{logical} logical"
            if physical and physical != logical:
                core_txt += f" / {physical} physical"
            cpu_bits.append(core_txt)

        self._fallback_host_label = "Host: " + " | ".join(host_parts) if host_parts else "Host: [UNKNOWN]"
        self._fallback_cpu_spec_text = "CPU 0: " + " - ".join(cpu_bits) if cpu_bits else ""
        self._fallback_gpu_entries = gpu_entries
        self._host_label.setText(self._fallback_host_label)
        self._cpu_spec_text = self._fallback_cpu_spec_text
        self._gpu_spec_entries = list(self._fallback_gpu_entries)
        self._refresh_capacity()
        self._populate_device_combo(self._gpu_spec_entries)

    def _refresh_capacity(self) -> None:
        # --- CPU + RAM bars ---
        if self._cpu_spec_text:
            self._cpu_label.setText(self._cpu_spec_text)
            self._cpu_label.setVisible(True)
            if _psutil is not None:
                try:
                    cpu_pct = float(_psutil.cpu_percent(interval=None))
                    _set_bar(self._cpu_bar, cpu_pct)
                    self._cpu_pct_lbl.setText(f"{cpu_pct:4.1f}%")
                except Exception:
                    pass
                try:
                    vm = _psutil.virtual_memory()
                    used_gb = (vm.total - vm.available) / (1024 ** 3)
                    total_gb = vm.total / (1024 ** 3)
                    _set_bar(self._ram_bar, float(vm.percent))
                    self._ram_lbl.setText(f"{used_gb:.1f}/{total_gb:.1f} GB")
                except Exception:
                    pass
            self._cpu_bars_widget.setVisible(True)
        else:
            self._cpu_label.setVisible(False)
            self._cpu_bars_widget.setVisible(False)

        # --- GPU spec line + per-device bars ---
        if self._gpu_spec_entries:
            gpu_spec_lines: list[str] = []
            # Rebuild bar rows to match current entry count
            while len(self._gpu_bar_rows) < len(self._gpu_spec_entries):
                row_widget = QWidget()
                row_layout = QHBoxLayout(row_widget)
                row_layout.setContentsMargins(0, 0, 0, 0)
                row_layout.setSpacing(6)
                _ls = "border: none; font-size: 9px; color: rgba(168,180,172,0.65);"
                util_lbl = QLabel("UTIL")
                util_lbl.setStyleSheet(_ls)
                util_lbl.setFixedWidth(30)
                row_layout.addWidget(util_lbl)
                util_bar = _make_bar()
                row_layout.addWidget(util_bar)
                util_pct_lbl = QLabel("")
                util_pct_lbl.setStyleSheet("border: none; font-size: 9px; font-family: monospace; min-width: 36px;")
                row_layout.addWidget(util_pct_lbl)
                vram_lbl = QLabel("VRAM")
                vram_lbl.setStyleSheet(_ls)
                vram_lbl.setFixedWidth(36)
                row_layout.addWidget(vram_lbl)
                vram_bar = _make_bar()
                row_layout.addWidget(vram_bar)
                vram_text_lbl = QLabel("")
                vram_text_lbl.setStyleSheet("border: none; font-size: 9px; font-family: monospace;")
                row_layout.addWidget(vram_text_lbl)
                row_layout.addStretch(1)
                self._gpu_bars_layout.addWidget(row_widget)
                self._gpu_bar_rows.append((util_lbl, util_bar, util_pct_lbl, vram_bar, vram_text_lbl))

            for i, entry in enumerate(self._gpu_spec_entries):
                idx = int(entry.get("index", 0) or 0)
                name = str(entry.get("name") or "Unknown")
                mem = entry.get("memory_gb")
                backend = str(entry.get("backend") or "").lower()
                spec_header = f"GPU {idx}: {name}"
                if mem:
                    spec_header += f" ({mem} GB)"
                spec_header += f" [{backend.upper()}]"

                if i < len(self._gpu_bar_rows):
                    _, util_bar, util_pct_lbl, vram_bar, vram_text_lbl = self._gpu_bar_rows[i]
                    if backend == "cuda":
                        tel = _cuda_telemetry(idx)
                        util = tel.get("util_pct")
                        if util is not None:
                            _set_bar(util_bar, float(util))
                            util_pct_lbl.setText(f"{util:4.1f}%")
                        used_gb = tel.get("mem_used_gb")
                        total_gb_v = tel.get("mem_total_gb")
                        if used_gb is not None and total_gb_v:
                            vram_pct = (used_gb / total_gb_v) * 100.0
                            _set_bar(vram_bar, vram_pct)
                            vram_text_lbl.setText(f"{used_gb:.1f}/{total_gb_v:.1f} GB")
                        # append thermal detail to spec text
                        tp_bits: list[str] = []
                        temp = tel.get("temp_c")
                        if temp is not None:
                            tp_bits.append(f"temp {temp:.0f}°C")
                        power = tel.get("power_w")
                        cap = tel.get("power_cap_w")
                        if power is not None and cap:
                            tp_bits.append(f"power {power:.0f}/{cap:.0f} W")
                        elif power is not None:
                            tp_bits.append(f"power {power:.0f} W")
                        fan = tel.get("fan_pct")
                        if fan is not None:
                            tp_bits.append(f"fan {fan:.0f}%")
                        if tp_bits:
                            spec_header += "  " + "  ".join(tp_bits)
                        reasons = tel.get("throttle_reasons") or []
                        if reasons:
                            spec_header += "  [THROTTLE] " + ", ".join(reasons)
                        self._gpu_bars_container.setVisible(True)
                    elif backend == "mps":
                        spec_header += "  [util n/a - macOS]"
                        util_pct_lbl.setText("n/a")
                        vram_text_lbl.setText("")
                        self._gpu_bars_container.setVisible(False)
                    else:
                        self._gpu_bars_container.setVisible(False)
                gpu_spec_lines.append(spec_header)

            self._gpu_label.setText("\n".join(gpu_spec_lines))
            self._gpu_label.setVisible(True)
        else:
            self._gpu_label.setVisible(False)
            self._gpu_bars_container.setVisible(False)
        self._refresh_cache_status()

    def _refresh_cache_status(self, *, force: bool = False) -> None:
        now = time.monotonic()
        if not force and self._cache_probe_text and (now - self._cache_probe_ts) < 30.0:
            return
        self._cache_probe_ts = now

        home = Path.home()
        cache_path = home / ".cache"
        cache_text = "~/.cache: not found"
        try:
            if cache_path.exists():
                proc = subprocess.run(
                    ["du", "-sk", str(cache_path)],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc.returncode == 0:
                    parts = (proc.stdout or "").strip().split(maxsplit=1)
                    if parts:
                        cache_gib = int(parts[0]) / (1024 ** 2)
                        cache_text = f"~/.cache {cache_gib:.2f} GiB"
        except Exception:
            cache_text = "~/.cache: unavailable"

        disk_text = "system disk: unavailable"
        try:
            usage = shutil.disk_usage(home)
            gib = 1024 ** 3
            used = usage.total - usage.free
            used_pct = (used / usage.total) * 100.0 if usage.total else 0.0
            disk_text = (
                f"system free {usage.free / gib:.1f}/{usage.total / gib:.0f} GiB "
                f"({used_pct:.0f}% used)"
            )
        except Exception:
            pass

        self._cache_probe_text = f"[CACHE] {cache_text}  |  {disk_text}"
        self._cache_status_label.setText(self._cache_probe_text)

    def _on_profile_combo_changed(self, _index: int) -> None:
        try:
            value = str(self._profile_combo.currentData() or "balanced")
        except Exception:
            value = "balanced"
        self._refresh_control_echo()
        self.guardProfileChanged.emit(value)

    def _on_device_combo_changed(self, _index: int) -> None:
        if self._suspend_signals:
            return
        try:
            value = str(self._device_combo.currentData() or "")
        except Exception:
            value = ""
        self._user_device = value
        self._refresh_control_echo()
        self.deviceChanged.emit(value)

    def _on_storage_combo_changed(self, _index: int) -> None:
        if self._suspend_signals:
            return
        try:
            value = str(self._storage_combo.currentData() or "")
        except Exception:
            value = ""
        # The "Browse..." sentinel triggers the file dialog without persisting
        # an empty selection.
        if value == "__browse__":
            self._suspend_signals = True
            try:
                self._storage_combo.setCurrentIndex(0)
            finally:
                self._suspend_signals = False
            self._on_storage_browse()
            return
        self._user_storage = value
        self._refresh_control_echo()
        self.storageRootChanged.emit(value)

    def _on_storage_browse(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Pick training save root",
            self._user_storage or "",
        )
        if not path:
            return
        self._user_storage = path
        # Insert (or reuse) a custom entry so the user can see the path they picked.
        self._suspend_signals = True
        try:
            existing = self._storage_combo.findData(path)
            if existing < 0:
                self._storage_combo.addItem(f"[CUSTOM] {path}", path)
                existing = self._storage_combo.count() - 1
            self._storage_combo.setCurrentIndex(existing)
        finally:
            self._suspend_signals = False
        self._refresh_control_echo()
        self.storageRootChanged.emit(path)

    def _refresh_control_echo(self) -> None:
        profile = str(self._profile_combo.currentData() or "balanced").strip() or "balanced"
        device = self._user_device or "auto"
        storage = self._user_storage or "auto (overflow protocol)"
        self._control_echo_label.setText(
            f"Guard controls: profile={profile}   device={device}   save_root={storage}"
        )
        self._control_echo_label.setVisible(True)

    def _on_refresh_guard(self) -> None:
        self.refreshRequested.emit()

    def _on_reset_overrides(self) -> None:
        self._suspend_signals = True
        try:
            i_device = self._device_combo.findData("")
            self._device_combo.setCurrentIndex(i_device if i_device >= 0 else 0)
            i_storage = self._storage_combo.findData("")
            self._storage_combo.setCurrentIndex(i_storage if i_storage >= 0 else 0)
        finally:
            self._suspend_signals = False
        self._user_device = ""
        self._user_storage = ""
        self._refresh_control_echo()
        self.deviceChanged.emit("")
        self.storageRootChanged.emit("")

    def _on_overflow_toggled(self, checked: bool) -> None:
        has_rows = self._overflow_layout.count() > 0
        self._overflow_container.setVisible(bool(checked) and has_rows)

    def _on_derivation_toggled(self, checked: bool) -> None:
        has_text = bool(str(self._derivation_label.text() or "").strip())
        self._derivation_label.setVisible(bool(checked) and has_text)

    def selected_device(self) -> str:
        return self._user_device

    def selected_storage_root(self) -> str:
        return self._user_storage

    def apply_guard(self, guard: Optional[dict[str, Any]]) -> None:
        if not isinstance(guard, dict) or not guard:
            self._empty_hint.setVisible(True)
            self._host_label.setText(self._fallback_host_label)
            self._cpu_spec_text = self._fallback_cpu_spec_text
            self._gpu_spec_entries = list(self._fallback_gpu_entries)
            self._refresh_capacity()
            self._adjust_label.setText("")
            self._projection_label.setText("")
            self._projection_label.setVisible(False)
            self._derivation_label.setText("")
            self._derivation_label.setVisible(False)
            self._derivation_toggle.setVisible(False)
            self._render_overflow_status({})
            self._overflow_toggle.setVisible(False)
            self.clear_storage_diagnosis()
            self._pill.set_status("empty")
            self._refresh_control_echo()
            for req, eff, lim in self._cells.values():
                req.setText("-")
                eff.setText("-")
                lim.setText("-")
            try:
                self._profile_combo.blockSignals(True)
                self._profile_combo.setCurrentIndex(0)
                self._profile_combo.setEnabled(False)
            finally:
                self._profile_combo.blockSignals(False)
            self._populate_device_combo(self._gpu_spec_entries)
            self._populate_storage_combo({})
            return

        self._empty_hint.setVisible(False)
        status = str(guard.get("status") or "ok").lower()
        self._pill.set_status(_STATUS_PILL_MAP.get(status, "empty"))
        profile = str(guard.get("profile") or "balanced").strip().lower() or "balanced"
        idx = self._profile_combo.findData(profile)
        if idx < 0:
            idx = 0
        try:
            self._profile_combo.blockSignals(True)
            self._profile_combo.setCurrentIndex(idx)
            self._profile_combo.setEnabled(True)
        finally:
            self._profile_combo.blockSignals(False)

        specs = guard.get("system_specs") or {}
        if isinstance(specs, dict):
            parts = []
            sys_name = str(specs.get("system") or "")
            machine = str(specs.get("machine") or "")
            if sys_name or machine:
                parts.append(f"{sys_name} {machine}".strip())
            mem_gb = specs.get("total_memory_gb")
            if mem_gb:
                parts.append(f"{mem_gb} GB RAM")
            accel = str(specs.get("accelerator") or "").upper()
            if accel:
                parts.append(f"runtime={accel}")
            disk_gb = specs.get("free_disk_gb")
            if disk_gb:
                parts.append(f"{disk_gb} GB free disk")
            self._host_label.setText("Host: " + " | ".join(parts) if parts else "Host: [UNKNOWN]")

            brand = str(specs.get("cpu_brand") or "").strip()
            logical = specs.get("cpu_logical_cores")
            physical = specs.get("cpu_physical_cores")
            cpu_bits: list[str] = []
            if brand:
                cpu_bits.append(brand)
            if logical:
                core_txt = f"{logical} logical"
                if physical and physical != logical:
                    core_txt += f" / {physical} physical"
                cpu_bits.append(core_txt)
            self._cpu_spec_text = "CPU 0: " + " - ".join(cpu_bits) if cpu_bits else ""

            gpus = specs.get("gpus")
            entries: list[dict[str, Any]] = []
            if isinstance(gpus, list) and gpus:
                for entry in gpus:
                    if isinstance(entry, dict):
                        entries.append(dict(entry))
            else:
                gpu_name = str(specs.get("gpu_name") or "")
                if gpu_name:
                    entries.append({
                        "index": 0,
                        "name": gpu_name,
                        "memory_gb": specs.get("gpu_memory_gb"),
                        "backend": str(specs.get("accelerator") or "").lower(),
                    })
            self._gpu_spec_entries = entries
            self._refresh_capacity()
            self._populate_device_combo(entries)
        else:
            self._host_label.setText(self._fallback_host_label)
            self._cpu_spec_text = self._fallback_cpu_spec_text
            self._gpu_spec_entries = list(self._fallback_gpu_entries)
            self._refresh_capacity()
            self._populate_device_combo(self._gpu_spec_entries)

        overflow = guard.get("overflow_protocol") if isinstance(guard.get("overflow_protocol"), dict) else {}
        self._populate_storage_combo(overflow)
        self._render_overflow_status(overflow)

        requested = guard.get("requested_hyperparams") or {}
        effective = guard.get("effective_hyperparams") or {}
        limits = guard.get("limits") or {}
        # Map cell key -> limits key (epochs has no hard limit in the guard).
        limit_key_map = {
            "imgsz": "max_imgsz",
            "batch": "max_batch",
            "workers": "max_workers",
            "epochs": None,
        }
        for key, (req_lbl, eff_lbl, lim_lbl) in self._cells.items():
            rv = requested.get(key) if isinstance(requested, dict) else None
            ev = effective.get(key) if isinstance(effective, dict) else None
            req_lbl.setText("-" if rv in (None, "") else str(rv))
            eff_lbl.setText("-" if ev in (None, "") else str(ev))
            lim_k = limit_key_map.get(key)
            lv = limits.get(lim_k) if (lim_k and isinstance(limits, dict)) else None
            lim_lbl.setText("-" if lv in (None, "") else str(lv))

        projection = guard.get("projections") or {}
        vram = projection.get("vram") if isinstance(projection, dict) else None
        if isinstance(vram, dict) and vram.get("peak_gb") is not None:
            peak = vram.get("peak_gb")
            budget = vram.get("budget_gb")
            headroom = vram.get("headroom_pct")
            target = str(vram.get("target") or "memory")
            risk = str(vram.get("risk") or "ok").lower()
            risk_tag = {
                "over": "[OVER BUDGET]",
                "tight": "[TIGHT]",
                "ok": "[OK]",
                "unknown": "[UNKNOWN]",
            }.get(risk, "[OK]")
            risk_color = {
                "over": "rgba(220, 50, 47, 0.95)",
                "tight": "rgba(203, 130, 28, 0.95)",
                "ok": "rgba(133, 153, 0, 0.90)",
                "unknown": "rgba(160, 160, 160, 0.90)",
            }.get(risk, "rgba(160, 160, 160, 0.90)")
            head_bits = [f"Projected peak: {peak:.1f} GB {target}"]
            if budget:
                head_bits.append(f"budget {budget:.1f} GB")
            if headroom is not None:
                head_bits.append(f"{headroom:.0f}% {risk_tag}")
            detail = (
                f"  weights {vram.get('weights_gb', 0):.2f} GB x4 "
                f"+ activations {vram.get('activations_gb', 0):.2f} GB "
                f"+ workspace {vram.get('workspace_gb', 0):.2f} GB "
                f"(params {vram.get('params_m', 0):.1f}M)"
            )
            self._projection_label.setText("  ".join(head_bits) + "\n" + detail)
            self._projection_label.setStyleSheet(
                f"border: none; font-size: 10px; color: {risk_color};"
            )
            self._projection_label.setVisible(True)
        else:
            self._projection_label.setText("")
            self._projection_label.setVisible(False)

        derivation = guard.get("derivation")
        if isinstance(derivation, list) and derivation:
            trail_lines: list[str] = ["Derivation (imgsz/batch/workers):"]
            for step in derivation:
                if not isinstance(step, dict):
                    continue
                label = str(step.get("step") or "?")
                note = str(step.get("note") or "")
                triple = f"{step.get('imgsz', '-')}/{step.get('batch', '-')}/{step.get('workers', '-')}"
                trail_lines.append(f"  {label:<18} {triple:<14}  {note}")
            self._derivation_label.setText("\n".join(trail_lines))
            self._derivation_toggle.setVisible(True)
            self._on_derivation_toggled(self._derivation_toggle.isChecked())
        else:
            self._derivation_label.setText("")
            self._derivation_label.setVisible(False)
            self._derivation_toggle.setVisible(False)

        adjustments = guard.get("adjustments")
        blocking = guard.get("blocking_reasons")
        lines: list[str] = []
        if isinstance(blocking, list) and blocking:
            for item in blocking:
                lines.append(f"[BLOCKED] {item}")
        if isinstance(adjustments, list):
            for item in adjustments:
                text = str(item or "").strip()
                if not text:
                    continue
                if text.startswith("[BLOCKED]"):
                    if text not in lines:
                        lines.append(text)
                    continue
                lines.append(f"- {text}")
        if lines:
            self._adjust_label.setText("Adjustments:\n" + "\n".join(lines))
        else:
            self._adjust_label.setText("Adjustments: none")
        self._refresh_control_echo()

    def clear(self) -> None:
        self.apply_guard(None)

    def set_storage_diagnosis(self, text: str) -> None:
        clean = str(text or "").strip()
        if not clean:
            self.clear_storage_diagnosis()
            return
        self._storage_pressure_label.setText("[STORAGE PRESSURE]\n" + clean)
        self._storage_pressure_label.setVisible(True)

    def clear_storage_diagnosis(self) -> None:
        self._storage_pressure_label.setText("")
        self._storage_pressure_label.setVisible(False)

    def _populate_device_combo(self, gpu_entries: list[dict[str, Any]]) -> None:
        """Rebuild the device combo from detected GPUs while preserving the
        user's prior selection if it still exists. Delegates list-building to the
        shared DeviceSelector so labels stay consistent with Cell Space.
        """
        self._suspend_signals = True
        try:
            # Keep the shared widget's remembered selection aligned with ours.
            self._device_combo.set_device(self._user_device or "")
            self._device_combo.set_entries(gpu_entries or [])
        finally:
            self._suspend_signals = False

    def _populate_storage_combo(self, overflow: dict[str, Any]) -> None:
        """Rebuild the storage combo from the overflow protocol's drive list.

        Each entry shows the drive's purpose (preferred/primary/overflow), the
        target path, and free/total disk so the user can pick the roomiest
        volume at a glance. A "Browse..." sentinel item opens a directory
        picker for a fully custom location.
        """
        self._suspend_signals = True
        try:
            self._storage_combo.clear()
            self._storage_combo.addItem("Auto (overflow protocol)", "")
            drives = overflow.get("drives") if isinstance(overflow, dict) else None
            if isinstance(drives, list):
                for drive in drives:
                    if not isinstance(drive, dict):
                        continue
                    path = str(drive.get("asset_root") or "").strip()
                    if not path:
                        continue
                    kind = str(drive.get("kind") or "").lower()
                    free_gb = drive.get("free_gb")
                    total_gb = drive.get("total_gb")
                    avail = bool(drive.get("available"))
                    active = bool(drive.get("active"))
                    tag_bits: list[str] = []
                    if active:
                        tag_bits.append("[ACTIVE]")
                    if kind:
                        tag_bits.append(f"[{kind.upper()}]")
                    if not avail:
                        tag_bits.append("[LOW SPACE]")
                    head = " ".join(tag_bits)
                    space_bits: list[str] = []
                    if free_gb is not None:
                        space_bits.append(f"{free_gb:.0f} GB free")
                    if total_gb:
                        space_bits.append(f"of {total_gb:.0f} GB")
                    space = " ".join(space_bits)
                    label = f"{head}  {path}  ({space})".strip()
                    self._storage_combo.addItem(label, path)
            self._storage_combo.insertSeparator(self._storage_combo.count())
            self._storage_combo.addItem("Browse for custom folder...", "__browse__")

            target = self._user_storage or ""
            found = self._storage_combo.findData(target) if target else 0
            if found < 0:
                # Selection points at a path that isn't in the drive list (a custom
                # path picked previously). Re-add it so the user sees their choice.
                self._storage_combo.insertItem(1, f"[CUSTOM] {target}", target)
                found = 1
            self._storage_combo.setCurrentIndex(found)
        finally:
            self._suspend_signals = False

    def _render_overflow_status(self, overflow: dict[str, Any]) -> None:
        # Clear any previously built drive rows.
        while self._overflow_layout.count():
            item = self._overflow_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        if not isinstance(overflow, dict) or not overflow:
            self._overflow_container.setVisible(False)
            self._overflow_toggle.setVisible(False)
            return

        status = str(overflow.get("status") or "").lower()
        active_root = str(overflow.get("active_asset_root") or "").strip()

        status_color = {
            "ok": "#7AE860",
            "overflow": "#c57a2e",
            "no_space": "#aa5555",
        }.get(status, "rgba(160,160,160,0.95)")

        tag = {
            "ok": "[OK]",
            "overflow": "[OVERFLOW ACTIVE]",
            "no_space": "[NO SPACE]",
        }.get(status, "[INFO]")

        # Header row: colored status tag + active destination path.
        header = QLabel(f"{tag}  dest: {active_root or '(none)'}")
        header.setWordWrap(True)
        header.setStyleSheet(
            "border: none; font-size: 10px;"
            "font-family: 'JetBrains Mono', 'IBM Plex Mono', monospace;"
            f"color: {status_color}; font-weight: 600;"
        )
        self._overflow_layout.addWidget(header)

        msg = str(overflow.get("message") or "").strip()
        if msg:
            msg_lbl = QLabel(msg)
            msg_lbl.setWordWrap(True)
            msg_lbl.setStyleSheet(
                f"border: none; font-size: 9px; color: {status_color}; font-style: italic;"
            )
            self._overflow_layout.addWidget(msg_lbl)

        # Per-drive rows — one QFrame per drive, color-coded by state.
        drives = overflow.get("drives") if isinstance(overflow.get("drives"), list) else []
        shown = 0
        valid_drives = [d for d in drives if isinstance(d, dict) and d.get("asset_root")]
        for drive in valid_drives:
            if shown >= 8:
                omit_lbl = QLabel(f"  ... +{len(valid_drives) - shown} more destinations")
                omit_lbl.setStyleSheet(
                    "border: none; font-size: 9px; color: rgba(160,160,160,0.7);"
                )
                self._overflow_layout.addWidget(omit_lbl)
                break

            path = str(drive.get("asset_root") or "")
            kind = str(drive.get("kind") or "")
            free_gb = float(drive.get("free_gb") or 0)
            total_gb = float(drive.get("total_gb") or 0)
            is_active = bool(drive.get("active"))
            is_avail = bool(drive.get("available", True))

            row_frame = QFrame()
            row_frame.setFrameShape(QFrame.Shape.NoFrame)
            # Only the active (and unavailable) rows carry an accent marker; the
            # rest stay borderless free-floating text so the list does not read
            # as a stack of boxes. State for normal rows is shown via text color.
            if is_active:
                row_frame.setStyleSheet(
                    f"QFrame {{ border: none; border-left: 1px solid {status_color};"
                    " background: rgba(90,170,170,0.06); }"
                )
            elif not is_avail:
                row_frame.setStyleSheet(
                    "QFrame { border: none; border-left: 1px solid #aa5555;"
                    " background: rgba(170,85,85,0.04); }"
                )
            else:
                row_frame.setStyleSheet("QFrame { border: none; background: transparent; }")

            row_layout = QHBoxLayout(row_frame)
            row_layout.setContentsMargins(4, 1, 4, 1)
            row_layout.setSpacing(6)

            active_lbl = QLabel("*" if is_active else " ")
            active_lbl.setFixedWidth(10)
            active_lbl.setStyleSheet(
                f"border: none; font-size: 9px; color: {status_color}; font-weight: 700;"
            )
            row_layout.addWidget(active_lbl)

            kind_lbl = QLabel(f"[{kind}]" if kind else "")
            kind_lbl.setFixedWidth(72)
            kind_lbl.setStyleSheet(
                "border: none; font-size: 9px; color: rgba(168,180,172,0.7);"
            )
            row_layout.addWidget(kind_lbl)

            path_lbl = QLabel(path)
            path_lbl.setStyleSheet(
                "border: none; font-size: 9px; color: rgba(208,220,212,0.9);"
            )
            row_layout.addWidget(path_lbl, stretch=1)

            space_lbl = QLabel(f"{free_gb:.1f}/{total_gb:.1f} GB")
            space_lbl.setFixedWidth(90)
            space_lbl.setStyleSheet(
                "border: none; font-size: 9px; font-family: monospace;"
                " color: rgba(168,180,172,0.8);"
            )
            row_layout.addWidget(space_lbl)

            if not is_avail:
                low_lbl = QLabel("[LOW]")
                low_lbl.setStyleSheet(
                    "border: none; font-size: 9px; color: #aa5555; font-weight: 600;"
                )
                row_layout.addWidget(low_lbl)

            self._overflow_layout.addWidget(row_frame)
            shown += 1

        self._overflow_toggle.setVisible(True)
        self._on_overflow_toggled(self._overflow_toggle.isChecked())
