from __future__ import annotations

import urllib.parse
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent

_BACKBONE_OPTIONS: list[tuple[str, str]] = [
    ("yolo_detection", "CV Detection (YOLO)  —  image datasets, requires training"),
    ("torch_tabular",  "ML / Tabular (PyTorch)  —  custom model, CSV / signal data"),
    ("custom_code", "Custom Code (Python Cells)  —  draft cells, any dataset category"),
    ("archival_ingestion", "Archival Ingestion  —  managed folders/files -> timeline graph"),
    ("face_recognition", "Face Recognition (Gallery)  —  ImageFolder identities, builds gallery.db"),
    ("audio_recognition", "Audio Recognition  —  AudioFolder classes, builds model.json"),
    ("llm_fine_tuning", "LLM Fine Tuning  —  JSONL instruction data, builds LoRA adapter + Modelfile"),
]

import re

from .cvops_theme import repolish
from ...config import ROOT_DIR
from .algo_catalog import list_algo_files, reveal_in_finder
from ..ollama_model_discovery import list_finetune_base_candidates


_SCENARIO_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _suggest_scenario_name(label: str) -> str:
    base = Path(str(label or "").strip()).stem or str(label or "").strip()
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", base).strip("_-")
    if not text:
        return ""
    if not re.match(r"^[A-Za-z0-9]", text):
        text = f"scenario_{text}"
    return text[:64]


def _suggest_display_name(label: str) -> str:
    base = Path(str(label or "").strip()).stem or str(label or "").strip()
    text = re.sub(r"[_-]+", " ", base).strip()
    return text.title() if text else ""


def _parse_classes(text: str) -> list[str]:
    raw = str(text or "")
    parts: list[str] = []
    for line in raw.splitlines():
        for seg in line.replace(";", ",").split(","):
            name = str(seg or "").strip()
            if name:
                parts.append(name)
    out: list[str] = []
    seen: set[str] = set()
    for p in parts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


class _DatasetDropBox(QLabel):
    def __init__(self, on_drop: Callable[[str], None], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_drop = on_drop
        self.setAcceptDrops(True)
        self.setText("Drop dataset folder or CSV here")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(74)
        self._apply_style(active=False)

    def _apply_style(self, *, active: bool) -> None:
        if active:
            border = "rgba(133,153,0,0.85)"
            bg = "rgba(133,153,0,0.10)"
        else:
            border = "rgba(120,120,120,0.55)"
            bg = "rgba(0,0,0,0.02)"
        self.setStyleSheet(
            "QLabel {"
            f" border: 1px dashed {border};"
            f" background: {bg};"
            " border-radius: 0px;"
            " padding: 8px;"
            " font-size: 10px;"
            " color: rgba(60,60,60,0.95);"
            "}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is not None and md.hasUrls():
            event.acceptProposedAction()
            self._apply_style(active=True)
            return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._apply_style(active=False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self._apply_style(active=False)
        md = event.mimeData()
        if md is None or not md.hasUrls():
            return
        for url in md.urls():
            if not url.isLocalFile():
                continue
            try:
                self._on_drop(url.toLocalFile())
            except Exception:
                continue
            event.acceptProposedAction()
            return


# Local model weight files that can be used directly as a base model. .mlpackage
# is a directory bundle, handled separately.
_MODEL_FILE_SUFFIXES = frozenset(
    {".pt", ".pth", ".torchscript", ".onnx", ".engine", ".mlmodel", ".tflite", ".weights"}
)


class _ModelDropBox(QLabel):
    """Compact drop strip for adding a local model file as the base model."""

    def __init__(self, on_drop: Callable[[str], None], parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_drop = on_drop
        self.setAcceptDrops(True)
        self.setText("Drop a model file (.pt / .onnx / .torchscript …)")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(30)
        self._apply_style(active=False)

    def _apply_style(self, *, active: bool) -> None:
        if active:
            border = "rgba(133,153,0,0.85)"
            bg = "rgba(133,153,0,0.10)"
        else:
            border = "rgba(120,120,120,0.55)"
            bg = "rgba(0,0,0,0.02)"
        self.setStyleSheet(
            "QLabel {"
            f" border: 1px dashed {border};"
            f" background: {bg};"
            " border-radius: 0px;"
            " padding: 5px;"
            " font-size: 10px;"
            " color: rgba(60,60,60,0.95);"
            "}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        md = event.mimeData()
        if md is not None and md.hasUrls():
            event.acceptProposedAction()
            self._apply_style(active=True)
            return
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self._apply_style(active=False)
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        self._apply_style(active=False)
        md = event.mimeData()
        if md is None or not md.hasUrls():
            return
        for url in md.urls():
            if not url.isLocalFile():
                continue
            try:
                self._on_drop(url.toLocalFile())
            except Exception:
                continue
            event.acceptProposedAction()
            return


class NewScenarioDialog(QDialog):
    def __init__(
        self,
        *,
        http_get: Callable[[str], dict[str, Any]],
        http_post: Callable[[str, Optional[dict[str, Any]]], dict[str, Any]],
        models: list[dict[str, Any]],
        datasets_payload: Optional[dict[str, Any]] = None,
        dataset_info_cache: Optional[dict[str, dict[str, Any]]] = None,
        initial_dataset: str = "",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._http_get = http_get
        self._http_post = http_post
        self._models = [m for m in (models or []) if isinstance(m, dict)]
        self._datasets_payload = dict(datasets_payload or {}) if isinstance(datasets_payload, dict) else None
        self._dataset_info_cache = {
            str(k): dict(v)
            for k, v in (dataset_info_cache or {}).items()
            if str(k).strip() and isinstance(v, dict)
        }
        self._pending_dataset_select = str(initial_dataset or "").strip()
        self._created = ""
        self._dataset_fmt = ""
        self._dataset_classes: list[str] = []
        # Full unfiltered dataset list: list of (display_label, slug_or_path, category)
        self._all_dataset_items: list[tuple[str, str, str]] = []

        self.setWindowTitle("New Scenario / Profile")
        self.resize(640, 580)
        self.setMinimumSize(500, 420)

        # Tracks which row-widgets belong to each backbone mode.
        self._cv_rows: list[tuple[QWidget | None, QWidget]] = []
        self._tabular_rows: list[tuple[QWidget | None, QWidget]] = []
        self._llm_rows: list[tuple[QWidget | None, QWidget]] = []
        self._config_rows: list[tuple[QWidget | None, QWidget]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll, stretch=1)

        content = QWidget()
        scroll.setWidget(content)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        title = QLabel("Create Training Scenario")
        title.setProperty("isTitle", True)
        title.setStyleSheet("font-size: 12px; font-weight: 700;")
        content_layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(5)

        r = 0

        self._add_section_header(grid, r, "Task Type")
        r += 1

        # -- Backbone type selector (always visible, at top) --
        backbone_lbl = QLabel("Backbone:")
        self._backbone = QComboBox()
        for btype, blabel in _BACKBONE_OPTIONS:
            self._backbone.addItem(blabel, btype)
        self._backbone.currentIndexChanged.connect(self._on_backbone_changed)
        grid.addWidget(backbone_lbl, r, 0)
        grid.addWidget(self._backbone, r, 1)
        r += 1

        self._add_section_header(grid, r, "Scenario")
        r += 1

        # -- Common fields --
        grid.addWidget(QLabel("Name:"), r, 0)
        self._name = QLineEdit()
        self._name.setPlaceholderText("example: donut_detector")
        grid.addWidget(self._name, r, 1)
        r += 1

        grid.addWidget(QLabel("Display name:"), r, 0)
        self._display = QLineEdit()
        self._display.setPlaceholderText("Example: Donut Detector")
        grid.addWidget(self._display, r, 1)
        r += 1

        grid.addWidget(QLabel("Description:"), r, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self._desc = QTextEdit()
        self._desc.setMinimumHeight(52)
        self._desc.setMaximumHeight(120)
        self._desc.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._desc.setPlaceholderText("What this scenario does, and what dataset it uses.")
        grid.addWidget(self._desc, r, 1)
        r += 1

        self._add_section_header(grid, r, "Dataset")
        r += 1

        # -- Dataset row (shared; filters by category based on backbone) --
        ds_lbl = QLabel("Dataset:")
        ds_wrap_layout = QVBoxLayout()
        ds_wrap_layout.setContentsMargins(0, 0, 0, 0)
        ds_wrap_layout.setSpacing(3)
        self._dataset_search = QLineEdit()
        self._dataset_search.setPlaceholderText("Search datasets...")
        self._dataset_search.textChanged.connect(self._filter_dataset_list)
        ds_wrap_layout.addWidget(self._dataset_search)
        self._dataset_drop = _DatasetDropBox(self._handle_dataset_drop)
        ds_wrap_layout.addWidget(self._dataset_drop)
        self._dataset = QComboBox()
        self._dataset.currentIndexChanged.connect(self._on_dataset_changed)
        ds_wrap_layout.addWidget(self._dataset)
        ds_box = QWidget()
        ds_box.setLayout(ds_wrap_layout)
        grid.addWidget(ds_lbl, r, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(ds_box, r, 1)
        r += 1
        self._dataset_meta = QLabel("")
        self._dataset_meta.setObjectName("datasetMeta")
        self._dataset_meta.setWordWrap(True)
        self._dataset_meta.setStyleSheet("font-size: 10px;")
        self._dataset_meta.setProperty("state", "idle")
        grid.addWidget(self._dataset_meta, r, 1)
        r += 1

        self._add_section_header(grid, r, "Training Defaults")
        r += 1

        # -- CV-only fields --
        _cv_model_lbl = QLabel("Base model:")
        model_box = QWidget()
        model_box_l = QVBoxLayout(model_box)
        model_box_l.setContentsMargins(0, 0, 0, 0)
        model_box_l.setSpacing(3)
        self._model = QComboBox()
        model_box_l.addWidget(self._model)
        # Drop-in row: add a local model file as the base model without having to
        # upload it through Range first. The file's path is used directly.
        model_add_row = QHBoxLayout()
        model_add_row.setContentsMargins(0, 0, 0, 0)
        model_add_row.setSpacing(4)
        self._model_drop = _ModelDropBox(self._handle_model_drop)
        model_add_row.addWidget(self._model_drop, stretch=1)
        self._model_browse_btn = QPushButton("Choose…")
        self._model_browse_btn.setToolTip(
            "Add a local model file (.pt / .onnx / .torchscript / .engine / …) as "
            "the base model for this scenario."
        )
        self._model_browse_btn.clicked.connect(self._browse_model_file)
        model_add_row.addWidget(self._model_browse_btn)
        model_box_l.addLayout(model_add_row)
        grid.addWidget(_cv_model_lbl, r, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(model_box, r, 1)
        self._cv_rows.append((_cv_model_lbl, model_box))
        r += 1

        hp_row = QHBoxLayout()
        hp_row.addWidget(QLabel("epochs"))
        self._epochs = QSpinBox()
        self._epochs.setRange(1, 9999)
        self._epochs.setValue(20)
        hp_row.addWidget(self._epochs)
        hp_row.addSpacing(10)
        hp_row.addWidget(QLabel("imgsz"))
        self._imgsz = QSpinBox()
        self._imgsz.setRange(32, 4096)
        self._imgsz.setSingleStep(32)
        self._imgsz.setValue(640)
        hp_row.addWidget(self._imgsz)
        hp_row.addSpacing(10)
        hp_row.addWidget(QLabel("guard"))
        self._guard = QComboBox()
        self._guard.addItem("Balanced", "balanced")
        self._guard.addItem("Stable", "stable")
        self._guard.addItem("Fast", "fast")
        hp_row.addWidget(self._guard)
        hp_row.addStretch(1)
        _training_lbl = QLabel("Training:")
        hp_wrap = QWidget()
        hp_wrap.setLayout(hp_row)
        grid.addWidget(_training_lbl, r, 0)
        grid.addWidget(hp_wrap, r, 1)
        self._cv_rows.append((_training_lbl, hp_wrap))
        r += 1

        _classes_lbl = QLabel("Classes:")
        cls_wrap_layout = QVBoxLayout()
        self._classes = QTextEdit()
        self._classes.setPlaceholderText("One class per line (or comma-separated).")
        self._classes.setMinimumHeight(80)
        self._classes.setMaximumHeight(180)
        self._classes.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cls_wrap_layout.addWidget(self._classes)
        cls_btn_row = QHBoxLayout()
        self._load_classes_btn = QPushButton("Load From Dataset")
        self._load_classes_btn.clicked.connect(self._load_classes_from_dataset)
        cls_btn_row.addWidget(self._load_classes_btn)
        cls_btn_row.addStretch(1)
        cls_wrap_layout.addLayout(cls_btn_row)
        cls_box = QWidget()
        cls_box.setLayout(cls_wrap_layout)
        grid.addWidget(_classes_lbl, r, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(cls_box, r, 1)
        self._cv_rows.append((_classes_lbl, cls_box))
        r += 1

        # -- ML-only fields --
        _cells_lbl = QLabel("Algo cells (.py):")
        cells_wrap = QWidget()
        cells_v = QVBoxLayout(cells_wrap)
        cells_v.setContentsMargins(0, 0, 0, 0)
        cells_v.setSpacing(4)

        # Catalog of known algo scripts (mlops/algos + templates).
        cat_title = QLabel("Algo Catalog")
        cat_title.setStyleSheet("font-weight: 600;")
        cells_v.addWidget(cat_title)
        cat_row = QHBoxLayout()
        self._algo_catalog = QListWidget()
        self._algo_catalog.setMinimumHeight(70)
        self._algo_catalog.setMaximumHeight(150)
        cat_row.addWidget(self._algo_catalog, stretch=1)
        cat_btns = QVBoxLayout()
        self._refresh_algo_catalog_btn = QPushButton("Refresh")
        self._refresh_algo_catalog_btn.clicked.connect(self._populate_algo_catalog)
        self._add_from_catalog_btn = QPushButton("Add →")
        self._add_from_catalog_btn.clicked.connect(self._add_selected_catalog_algo)
        self._reveal_catalog_btn = QPushButton("Reveal")
        self._reveal_catalog_btn.clicked.connect(self._reveal_selected_catalog_algo)
        cat_btns.addWidget(self._refresh_algo_catalog_btn)
        cat_btns.addWidget(self._add_from_catalog_btn)
        cat_btns.addWidget(self._reveal_catalog_btn)
        cat_btns.addStretch(1)
        cat_row.addLayout(cat_btns)
        cells_v.addLayout(cat_row)

        sel_title = QLabel("Selected Cells (execution order)")
        sel_title.setStyleSheet("font-weight: 600;")
        cells_v.addWidget(sel_title)
        self._algo_cells = QListWidget()
        self._algo_cells.setMinimumHeight(70)
        self._algo_cells.setMaximumHeight(140)
        cells_v.addWidget(self._algo_cells)

        cell_btn_row = QHBoxLayout()
        self._reveal_algo_cell_btn = QPushButton("Reveal Selected")
        self._reveal_algo_cell_btn.clicked.connect(self._reveal_selected_algo_cell)
        self._remove_algo_cell_btn = QPushButton("Remove Selected")
        self._remove_algo_cell_btn.clicked.connect(self._remove_selected_algo_cell)
        self._clear_algo_cells_btn = QPushButton("Clear")
        self._clear_algo_cells_btn.clicked.connect(lambda: self._algo_cells.clear())
        cell_btn_row.addWidget(self._reveal_algo_cell_btn)
        cell_btn_row.addWidget(self._remove_algo_cell_btn)
        cell_btn_row.addWidget(self._clear_algo_cells_btn)
        cell_btn_row.addStretch(1)
        cells_v.addLayout(cell_btn_row)

        hint = QLabel(
            "One file per cell. Each file can own internal files/data and should export `run(ctx, prev)`."
        )
        hint.setStyleSheet("font-size: 10px; color: rgba(120,120,120,0.8);")
        hint.setWordWrap(True)
        cells_v.addWidget(hint)

        grid.addWidget(_cells_lbl, r, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(cells_wrap, r, 1)
        self._tabular_rows.append((_cells_lbl, cells_wrap))
        r += 1

        _csv_lbl = QLabel("Dataset CSV:")
        self._dataset_csv = QLineEdit()
        self._dataset_csv.setPlaceholderText("e.g. mlops/datasets/my_data.csv  (optional)")
        grid.addWidget(_csv_lbl, r, 0)
        grid.addWidget(self._dataset_csv, r, 1)
        self._tabular_rows.append((_csv_lbl, self._dataset_csv))
        r += 1

        # -- LLM-only fields --
        _llm_model_lbl = QLabel("HF base model:")
        self._llm_base_model = QLineEdit()
        self._llm_base_model.setPlaceholderText("e.g. meta-llama/Llama-3.2-1B or /models/local-llm")
        grid.addWidget(_llm_model_lbl, r, 0)
        grid.addWidget(self._llm_base_model, r, 1)
        self._llm_rows.append((_llm_model_lbl, self._llm_base_model))
        r += 1

        _ollama_lbl = QLabel("Ollama base:")
        ollama_row = QHBoxLayout()
        ollama_row.setContentsMargins(0, 0, 0, 0)
        ollama_row.setSpacing(6)
        self._ollama_base_model = QComboBox()
        self._ollama_base_model.setEditable(True)
        self._ollama_base_model.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        le_ollama = self._ollama_base_model.lineEdit()
        if le_ollama is not None:
            le_ollama.setPlaceholderText("Ollama tag or path to .gguf (FROM line)")
        self._refresh_ollama_models_btn = QPushButton("Refresh")
        self._refresh_ollama_models_btn.setToolTip("Reload models from Ollama (api/tags) and scan for local .gguf files")
        self._refresh_ollama_models_btn.clicked.connect(lambda: self._refresh_ollama_base_models(silent=False))
        ollama_row.addWidget(self._ollama_base_model, stretch=1)
        ollama_row.addWidget(self._refresh_ollama_models_btn)
        _ollama_wrap = QWidget()
        _ollama_wrap.setLayout(ollama_row)
        grid.addWidget(_ollama_lbl, r, 0)
        grid.addWidget(_ollama_wrap, r, 1)
        self._llm_rows.append((_ollama_lbl, _ollama_wrap))
        r += 1

        _llm_training_lbl = QLabel("LLM training:")
        llm_hp_row = QHBoxLayout()
        llm_hp_row.addWidget(QLabel("epochs"))
        self._llm_epochs = QSpinBox()
        self._llm_epochs.setRange(1, 1000)
        self._llm_epochs.setValue(1)
        llm_hp_row.addWidget(self._llm_epochs)
        llm_hp_row.addWidget(QLabel("seq"))
        self._llm_max_seq = QSpinBox()
        self._llm_max_seq.setRange(128, 32768)
        self._llm_max_seq.setSingleStep(128)
        self._llm_max_seq.setValue(1024)
        llm_hp_row.addWidget(self._llm_max_seq)
        llm_hp_row.addWidget(QLabel("batch"))
        self._llm_batch = QSpinBox()
        self._llm_batch.setRange(1, 256)
        self._llm_batch.setValue(1)
        llm_hp_row.addWidget(self._llm_batch)
        llm_hp_row.addWidget(QLabel("lr"))
        self._llm_lr = QLineEdit("0.0002")
        self._llm_lr.setMaximumWidth(90)
        llm_hp_row.addWidget(self._llm_lr)
        llm_hp_row.addStretch(1)
        llm_hp_wrap = QWidget()
        llm_hp_wrap.setLayout(llm_hp_row)
        grid.addWidget(_llm_training_lbl, r, 0)
        grid.addWidget(llm_hp_wrap, r, 1)
        self._llm_rows.append((_llm_training_lbl, llm_hp_wrap))
        r += 1

        _lora_lbl = QLabel("LoRA:")
        lora_row = QHBoxLayout()
        lora_row.addWidget(QLabel("r"))
        self._llm_lora_r = QSpinBox()
        self._llm_lora_r.setRange(1, 1024)
        self._llm_lora_r.setValue(8)
        lora_row.addWidget(self._llm_lora_r)
        lora_row.addWidget(QLabel("alpha"))
        self._llm_lora_alpha = QSpinBox()
        self._llm_lora_alpha.setRange(1, 4096)
        self._llm_lora_alpha.setValue(16)
        lora_row.addWidget(self._llm_lora_alpha)
        lora_row.addWidget(QLabel("targets"))
        self._llm_targets = QLineEdit("q_proj,v_proj")
        self._llm_targets.setPlaceholderText("q_proj,v_proj")
        lora_row.addWidget(self._llm_targets, stretch=1)
        lora_wrap = QWidget()
        lora_wrap.setLayout(lora_row)
        grid.addWidget(_lora_lbl, r, 0)
        grid.addWidget(lora_wrap, r, 1)
        self._llm_rows.append((_lora_lbl, lora_wrap))
        r += 1

        self._add_section_header(grid, r, "Advanced Config")
        r += 1

        _bcfg_lbl = QLabel("backbone_config\n(JSON):")
        _bcfg_lbl.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._backbone_config_edit = QTextEdit()
        self._backbone_config_edit.setPlaceholderText('{"num_classes": 5, "epochs": 100, ...}')
        self._backbone_config_edit.setMinimumHeight(80)
        self._backbone_config_edit.setMaximumHeight(150)
        grid.addWidget(_bcfg_lbl, r, 0, alignment=Qt.AlignmentFlag.AlignTop)
        grid.addWidget(self._backbone_config_edit, r, 1)
        self._config_rows.append((_bcfg_lbl, self._backbone_config_edit))
        r += 1

        content_layout.addLayout(grid)
        content_layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self._create_btn = buttons.addButton("Create", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.rejected.connect(self.reject)
        self._create_btn.clicked.connect(self._create)
        outer.addWidget(buttons)

        # Apply initial visibility (after _create_btn exists). If the Collect
        # handoff preloaded the target dataset, start in its matching backbone
        # so the first dataset probe hits the intended cached dataset instead
        # of an unrelated first image dataset.
        initial_payload = self._dataset_info_cache.get(self._pending_dataset_select, {})
        initial_fmt = str(initial_payload.get("format") or "").strip().lower()
        initial_backbone = self._backbone_for_dataset_format(initial_fmt)
        if initial_backbone:
            self._set_backbone_type(initial_backbone)
        self._on_backbone_changed()

        self._populate_models()
        self._populate_datasets()
        self._populate_algo_catalog()

    @staticmethod
    def _add_section_header(grid: QGridLayout, row: int, title: str) -> QLabel:
        label = QLabel(title)
        label.setObjectName("dialogSectionHeader")
        label.setStyleSheet("font-size: 11px; font-weight: 700; padding-top: 8px;")
        grid.addWidget(label, row, 0, 1, 2)
        return label

    def _populate_algo_catalog(self) -> None:
        selected_item = self._algo_catalog.currentItem()
        selected = str(selected_item.data(Qt.ItemDataRole.UserRole) or "") if selected_item is not None else ""
        entries = list_algo_files()
        self._algo_catalog.blockSignals(True)
        self._algo_catalog.clear()
        for e in entries:
            name = str(e.get("name") or "")
            path = str(e.get("path") or "")
            doc = str(e.get("doc") or "").strip()
            if not path:
                continue
            label = f"{name}  —  {doc}" if doc else name
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, path)
            item.setToolTip(path)
            self._algo_catalog.addItem(item)
        self._algo_catalog.blockSignals(False)
        if selected:
            for i in range(self._algo_catalog.count()):
                it = self._algo_catalog.item(i)
                if it is not None and str(it.data(Qt.ItemDataRole.UserRole) or "") == selected:
                    self._algo_catalog.setCurrentRow(i)
                    break

    def _add_selected_catalog_algo(self) -> None:
        it = self._algo_catalog.currentItem()
        if it is None:
            return
        path = str(it.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not path:
            return
        self._add_algo_cell_path(path)

    def _reveal_selected_catalog_algo(self) -> None:
        it = self._algo_catalog.currentItem()
        if it is None:
            return
        path = str(it.data(Qt.ItemDataRole.UserRole) or "").strip()
        if path:
            reveal_in_finder(path)

    def _add_algo_cell_path(self, path: str) -> None:
        text = str(path or "").strip()
        if not text:
            return
        # Avoid duplicates.
        for i in range(self._algo_cells.count()):
            it = self._algo_cells.item(i)
            if it is not None and str(it.data(Qt.ItemDataRole.UserRole) or it.text()) == text:
                return
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, text)
        self._algo_cells.addItem(item)

    def _remove_selected_algo_cell(self) -> None:
        row = self._algo_cells.currentRow()
        if row >= 0:
            self._algo_cells.takeItem(row)

    def _reveal_selected_algo_cell(self) -> None:
        it = self._algo_cells.currentItem()
        if it is None:
            return
        path = str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "").strip()
        if path:
            reveal_in_finder(path)

    def created_scenario_name(self) -> str:
        return str(self._created or "").strip()

    def _current_backbone(self) -> str:
        return str(self._backbone.currentData() or "yolo_detection")

    # ---------- Backbone toggle ----------

    def _on_backbone_changed(self, _index: int = 0) -> None:
        btype = self._current_backbone()
        is_cv = btype == "yolo_detection"
        is_tabular = btype == "torch_tabular"
        is_custom_code = btype == "custom_code"
        is_archival = btype == "archival_ingestion"
        is_face = btype == "face_recognition"
        is_audio = btype == "audio_recognition"
        is_llm = btype == "llm_fine_tuning"
        for lbl, widget in self._cv_rows:
            widget.setVisible(is_cv)
            if lbl is not None:
                lbl.setVisible(is_cv)
        for lbl, widget in self._tabular_rows:
            widget.setVisible(is_tabular)
            if lbl is not None:
                lbl.setVisible(is_tabular)
        for lbl, widget in self._llm_rows:
            widget.setVisible(is_llm)
            if lbl is not None:
                lbl.setVisible(is_llm)
        for lbl, widget in self._config_rows:
            widget.setVisible(is_tabular or is_face or is_audio or is_custom_code or is_llm or is_archival)
            if lbl is not None:
                lbl.setVisible(is_tabular or is_face or is_audio or is_custom_code or is_llm or is_archival)
        # Clear search and re-filter by category (uses already-fetched list).
        self._dataset_search.blockSignals(True)
        self._dataset_search.clear()
        self._dataset_search.blockSignals(False)
        self._apply_dataset_filter()
        if is_tabular:
            self._ensure_default_tabular_cells()
        # Do not silently scan local Ollama / GGUF roots on mode changes. On
        # large workstations that filesystem walk can block the modal dialog
        # long enough for the whole app to look wedged; the explicit Refresh
        # button still performs discovery when the user asks for it.

    def _ollama_base_text(self) -> str:
        return str(self._ollama_base_model.currentText() or "").strip()

    def _refresh_ollama_base_models(self, *, silent: bool = False) -> None:
        prev = self._ollama_base_text()
        try:
            ollama_names, ggufs = list_finetune_base_candidates(repo_root=Path(ROOT_DIR))
        except Exception as exc:
            if not silent:
                QMessageBox.warning(self, "Model discovery failed", str(exc))
            return
        self._ollama_base_model.blockSignals(True)
        self._ollama_base_model.clear()
        for name in ollama_names:
            self._ollama_base_model.addItem(name)
        for path in ggufs:
            self._ollama_base_model.addItem(path)
        self._ollama_base_model.blockSignals(False)
        if prev:
            idx = self._ollama_base_model.findText(prev, Qt.MatchFlag.MatchExactly)
            if idx >= 0:
                self._ollama_base_model.setCurrentIndex(idx)
            else:
                self._ollama_base_model.setCurrentText(prev)
        if not silent and not ollama_names and not ggufs:
            QMessageBox.information(
                self,
                "No models found",
                "Could not list Ollama models (is the daemon running?) and no .gguf files were found "
                "under the repo, ~/models, ~/Downloads, OLLAMA_MODELS, or INSIGHT_GGUF_SCAN_ROOTS "
                "(set INSIGHT_GGUF_SCAN_DOT_OLLAMA=1 to include ~/.ollama). "
                "You can still type an Ollama tag or a full path to a .gguf file.",
            )

    def _ensure_default_tabular_cells(self) -> None:
        """If no tabular algo cells were added yet, seed with the editable template."""
        try:
            count = int(self._algo_cells.count())
        except Exception:
            count = 0
        if count > 0:
            return
        default_path = str(Path("basic_cnn_for_editing.py").as_posix())
        self._add_algo_cell_path(default_path)

    # ---------- Populate ----------

    def _browse_model_file(self) -> None:
        exts = "*.pt *.pth *.torchscript *.onnx *.engine *.mlmodel *.tflite"
        path_str, _ = QFileDialog.getOpenFileName(
            self,
            "Select Base Model Weights",
            "",
            f"Model Weights ({exts} *.mlpackage);;All Files (*.*)",
        )
        if not path_str:
            # .mlpackage is a directory bundle; offer a folder picker as fallback.
            path_str = QFileDialog.getExistingDirectory(
                self, "Select .mlpackage Bundle (directory)", ""
            )
        if path_str:
            self._handle_model_drop(path_str)

    def _handle_model_drop(self, path_str: str) -> None:
        """Add a dropped/chosen local model file to the base-model list and select
        it. The resolved path is stored as the scenario's base_model, so the model
        does not need to be uploaded through Range first."""
        p = Path(str(path_str or "")).expanduser()
        if not p.exists():
            QMessageBox.warning(self, "Model", f"Path does not exist:\n{p}")
            return
        suffix = p.suffix.lower()
        is_mlpackage = p.is_dir() and suffix == ".mlpackage"
        if not is_mlpackage and (p.is_dir() or suffix not in _MODEL_FILE_SUFFIXES):
            QMessageBox.warning(
                self,
                "Unsupported Model",
                f"'{p.name}' is not a supported model file.\n"
                "Use .pt, .pth, .torchscript, .onnx, .engine, .mlmodel, .tflite, "
                "or an .mlpackage bundle.",
            )
            return
        key = str(p.resolve())
        idx = self._model.findData(key)
        if idx >= 0:
            self._model.setCurrentIndex(idx)
            return
        self._model.addItem(p.name, key)
        self._model.setCurrentIndex(self._model.count() - 1)

    def _populate_models(self) -> None:
        self._model.clear()
        for m in self._models:
            value = str(m.get("value") or m.get("path") or "").strip()
            if not value:
                continue
            label = str(m.get("name") or Path(value).name)
            self._model.addItem(label, value)

    def _populate_datasets(self) -> None:
        """Fetch all datasets and rebuild the full dataset item list, then filter."""
        try:
            payload = self._datasets_payload if self._datasets_payload is not None else self._http_get("/database")
            names = list(payload.get("datasets") or [])
            categories: dict[str, str] = dict(payload.get("categories") or {})
            tabular_entries = list(payload.get("tabular_datasets") or [])
            text_entries = list(payload.get("text_datasets") or [])
            for cached_name, cached_payload in self._dataset_info_cache.items():
                if cached_name in {str(n) for n in names}:
                    continue
                names.append(cached_name)
                categories[cached_name] = self._category_for_dataset_format(
                    str(cached_payload.get("format") or "")
                )
        except Exception as exc:
            QMessageBox.warning(self, "Dataset Load Failed", str(exc))
            self._all_dataset_items = []
            self._apply_dataset_filter()
            return

        items: list[tuple[str, str, str]] = []
        # Image datasets from database/.
        for n in names:
            cat = categories.get(str(n), "image")
            items.append((str(n), str(n), cat))
        # Tabular CSV datasets from mlops/datasets/.
        for entry in tabular_entries:
            label = str(entry.get("filename") or entry.get("name") or "")
            path = str(entry.get("path") or "")
            if label and path:
                items.append((label, path, "tabular"))
        for entry in text_entries:
            label = str(entry.get("filename") or entry.get("name") or "")
            path = str(entry.get("path") or "")
            if label and path:
                items.append((label, path, "text"))

        self._all_dataset_items = items
        self._apply_dataset_filter()

    def _set_backbone_type(self, btype: str) -> None:
        want = str(btype or "").strip().lower()
        for i in range(self._backbone.count()):
            if str(self._backbone.itemData(i) or "").strip().lower() == want:
                self._backbone.setCurrentIndex(i)
                return

    def _handle_dataset_drop(self, path_str: str) -> None:
        raw = str(path_str or "").strip()
        if not raw:
            return
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (Path(ROOT_DIR) / p).resolve()
        if not p.exists():
            QMessageBox.warning(self, "Invalid Drop", f"Path does not exist:\n{p}")
            return

        # CSV/JSONL drop: prefer matching data-oriented backbone.
        if p.is_file() and p.suffix.lower() in {".csv", ".jsonl"}:
            cur = self._current_backbone()
            wanted = "llm_fine_tuning" if p.suffix.lower() == ".jsonl" else "torch_tabular"
            wanted_label = "LLM Fine Tuning" if wanted == "llm_fine_tuning" else "ML / Tabular (PyTorch)"
            if cur not in (wanted, "custom_code"):
                resp = QMessageBox.question(
                    self,
                    "Switch Backbone?",
                    f"You dropped a {p.suffix.lower()} file. Switch backbone to '{wanted_label}'?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.Yes,
                )
                if resp != QMessageBox.StandardButton.Yes:
                    return
                self._set_backbone_type(wanted)
            if p.suffix.lower() == ".csv":
                self._dataset_csv.setText(str(p))
                cat = "tabular"
            else:
                cat = "text"
            rel_value = str(p)
            try:
                rel_value = p.resolve().relative_to(Path(ROOT_DIR).resolve()).as_posix()
            except Exception:
                rel_value = str(p)
            if not any(value == rel_value for _label, value, _cat in self._all_dataset_items):
                self._all_dataset_items.append((p.name, rel_value, cat))
                self._apply_dataset_filter()
            idx = self._dataset.findData(rel_value)
            if idx >= 0:
                self._dataset.setCurrentIndex(idx)
            return

        # Folder drop: must map to database/<slug> or mlops/datasets/<slug>.
        if not p.is_dir():
            QMessageBox.warning(self, "Unsupported Drop", "Drop a dataset folder (from database/) or a .csv file.")
            return

        db_root = (Path(ROOT_DIR) / "database").resolve()
        ds_root = (Path(ROOT_DIR) / "mlops" / "datasets").resolve()
        slug = ""
        try:
            rel = p.resolve().relative_to(db_root)
            slug = str(rel.parts[0] if rel.parts else "").strip()
        except Exception:
            try:
                rel = p.resolve().relative_to(ds_root)
                slug = str(rel.parts[0] if rel.parts else "").strip()
            except Exception:
                QMessageBox.warning(
                    self,
                    "Unsupported Folder",
                    f"Dropped folder must live under:\n{db_root}\n(or)\n{ds_root}\n\nGot:\n{p.resolve()}",
                )
                return
        if not slug:
            QMessageBox.warning(self, "Invalid Folder", "Could not infer dataset slug from dropped path.")
            return

        # Use server dataset format to pick a recommended backbone.
        preferred = "yolo_detection"
        fmt = ""
        try:
            payload = self._dataset_payload(slug)
            fmt = str(payload.get("format") or "")
        except Exception:
            fmt = ""
        if fmt == "imagefolder_classification" or fmt == "face_csv":
            preferred = "face_recognition"
        elif fmt == "audiofolder_classification":
            preferred = "audio_recognition"
        elif fmt == "llm_instruction_jsonl":
            preferred = "llm_fine_tuning"
        elif fmt == "yolo_detection":
            preferred = "yolo_detection"

        cur = self._current_backbone()
        if cur == "torch_tabular" or cur != preferred:
            resp = QMessageBox.question(
                self,
                "Switch Backbone?",
                f"You dropped dataset '{slug}' (format: {fmt or 'unknown'}).\n\nSwitch backbone to '{preferred}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if resp == QMessageBox.StandardButton.Yes:
                self._set_backbone_type(preferred)

        # Ensure datasets are populated, then select the slug.
        idx = self._dataset.findData(slug)
        if idx < 0:
            self._populate_datasets()
            idx = self._dataset.findData(slug)
        if idx >= 0:
            self._dataset.setCurrentIndex(idx)
        else:
            QMessageBox.warning(self, "Dataset Not Found", f"Dataset '{slug}' was not found in /database list.")

    def _apply_dataset_filter(self) -> None:
        """Filter _all_dataset_items by backbone category + search text, repopulate combo."""
        btype = self._current_backbone()
        if btype == "torch_tabular":
            want_category = "tabular"
        elif btype == "archival_ingestion":
            want_category = None
        elif btype == "custom_code":
            want_category = None
        elif btype == "audio_recognition":
            want_category = "audio"
        elif btype == "llm_fine_tuning":
            want_category = "text"
        else:
            want_category = "image"
        query = str(self._dataset_search.text() or "").strip().lower()
        current_value = str(self._dataset.currentData() or "").strip()
        preferred_value = str(self._pending_dataset_select or current_value or "").strip()

        self._dataset.blockSignals(True)
        self._dataset.clear()
        for label, value, cat in self._all_dataset_items:
            if want_category is not None and cat != want_category:
                continue
            if query and query not in label.lower():
                continue
            self._dataset.addItem(label, value)
        if preferred_value:
            idx = self._dataset.findData(preferred_value)
            if idx >= 0:
                self._dataset.setCurrentIndex(idx)
                if preferred_value == self._pending_dataset_select:
                    self._pending_dataset_select = ""
        self._dataset.blockSignals(False)
        # Trigger validation for the newly visible selection.
        self._on_dataset_changed()

    def _filter_dataset_list(self, _text: str = "") -> None:
        self._apply_dataset_filter()

    def _on_dataset_changed(self) -> None:
        btype = self._current_backbone()
        is_tabular = btype == "torch_tabular"
        is_custom_code = btype == "custom_code"
        is_archival = btype == "archival_ingestion"
        is_llm = btype == "llm_fine_tuning"
        self._maybe_autofill_names_from_dataset()

        if is_custom_code:
            self._dataset_meta.setProperty("state", "ok")
            repolish(self._dataset_meta)
            self._dataset_meta.setText(
                "Any dataset category is allowed. Optional: pick a library dataset for scenario-level links."
            )
            self._create_btn.setEnabled(bool(str(self._name.text() or "").strip()))
            return

        if is_archival:
            self._dataset_meta.setProperty("state", "ok")
            repolish(self._dataset_meta)
            self._dataset_meta.setText(
                "Archival scenarios attach to imported corpora later. Optional: keep backbone_config JSON for domain/profile defaults."
            )
            self._create_btn.setEnabled(bool(str(self._name.text() or "").strip()))
            return

        if is_tabular:
            # Auto-fill the Dataset CSV field when a CSV entry is selected.
            csv_path = str(self._dataset.currentData() or "").strip()
            if csv_path and not str(self._dataset_csv.text() or "").strip():
                self._dataset_csv.setText(csv_path)
            self._create_btn.setEnabled(True)
            return

        if is_llm:
            dataset_ref = str(self._dataset.currentData() or "").strip()
            if not dataset_ref:
                self._dataset_meta.setText("")
                self._create_btn.setEnabled(False)
                self._dataset_fmt = ""
                self._dataset_classes = []
                return
            payload: dict[str, Any] = {}
            fmt = "llm_instruction_jsonl"
            count = 0
            if "/" not in dataset_ref and "\\" not in dataset_ref and not dataset_ref.lower().endswith(".jsonl"):
                try:
                    payload = self._dataset_payload(dataset_ref)
                    fmt = str(payload.get("format") or "")
                    count = int(payload.get("count") or 0)
                except Exception:
                    payload = {}
            if fmt == "llm_instruction_jsonl" or dataset_ref.lower().endswith(".jsonl"):
                self._dataset_meta.setProperty("state", "ok")
                repolish(self._dataset_meta)
                note = "format: JSONL instruction data"
                if count:
                    note += f"  |  examples: {count}"
                self._dataset_meta.setText(note)
                self._create_btn.setEnabled(True)
            else:
                self._dataset_meta.setProperty("state", "error")
                repolish(self._dataset_meta)
                self._dataset_meta.setText("format: unsupported. LLM Fine Tuning requires JSONL instruction data.")
                self._create_btn.setEnabled(False)
            return

        slug = str(self._dataset.currentData() or "").strip()
        if not slug:
            self._dataset_meta.setText("")
            self._create_btn.setEnabled(False)
            self._dataset_fmt = ""
            self._dataset_classes = []
            return
        try:
            payload = self._dataset_payload(slug)
        except Exception as exc:
            self._dataset_meta.setText(f"Dataset info failed: {exc}")
            self._create_btn.setEnabled(False)
            self._dataset_fmt = ""
            self._dataset_classes = []
            return
        fmt = str(payload.get("format") or "")
        self._dataset_fmt = fmt
        try:
            count = int(payload.get("count") or 0)
        except Exception:
            count = 0
        classes = payload.get("classes") if isinstance(payload, dict) else None
        self._dataset_classes = [str(c) for c in classes if str(c).strip()] if isinstance(classes, list) else []
        class_count = len(self._dataset_classes)

        if btype == "yolo_detection" and fmt == "yolo_detection":
            self._dataset_meta.setProperty("state", "ok")
            repolish(self._dataset_meta)
            note = f"format: YOLO detection  |  images: {count}"
            if class_count:
                note += f"  |  classes: {class_count}"
            else:
                note += "  |  classes: (missing classes.txt)"
            self._dataset_meta.setText(note)
            self._create_btn.setEnabled(True)
            # If the dataset provides classes and the editor is empty, prefill.
            if class_count and not str(self._classes.toPlainText() or "").strip():
                lines = "\n".join(self._dataset_classes)
                if lines.strip():
                    self._classes.setPlainText(lines + "\n")
        elif btype == "face_recognition" and fmt in {"imagefolder_classification", "face_csv"}:
            self._dataset_meta.setProperty("state", "ok")
            repolish(self._dataset_meta)
            if fmt == "face_csv":
                note = f"format: face_csv (CSV labels + images)  |  images: {count}"
            else:
                note = f"format: ImageFolder identities  |  images: {count}"
            if class_count:
                note += f"  |  classes: {class_count}"
            self._dataset_meta.setText(note)
            self._create_btn.setEnabled(True)
        elif btype == "audio_recognition" and fmt == "audiofolder_classification":
            self._dataset_meta.setProperty("state", "ok")
            repolish(self._dataset_meta)
            note = f"format: AudioFolder classes  |  clips: {count}"
            if class_count:
                note += f"  |  classes: {class_count}"
            self._dataset_meta.setText(note)
            self._create_btn.setEnabled(True)
        elif fmt == "imagefolder_classification":
            self._dataset_meta.setProperty("state", "error")
            repolish(self._dataset_meta)
            self._dataset_meta.setText(
                "format: ImageFolder classification. Use Face Recognition (Gallery) backbone, or convert this dataset to YOLO (Datasets panel)."
            )
            self._create_btn.setEnabled(False)
        else:
            self._dataset_meta.setProperty("state", "error")
            repolish(self._dataset_meta)
            self._dataset_meta.setText(
                f"format: {fmt or 'unknown'}. Scenario creation supports YOLO, ImageFolder, face CSV, AudioFolder, JSONL, or CSV/tabular datasets."
            )
            self._create_btn.setEnabled(False)

    def _maybe_autofill_names_from_dataset(self) -> None:
        label = str(self._dataset.currentText() or self._dataset.currentData() or "").strip()
        if not label:
            return
        if not str(self._name.text() or "").strip():
            suggestion = _suggest_scenario_name(label)
            if suggestion:
                self._name.setText(suggestion)
        if not str(self._display.text() or "").strip():
            suggestion = _suggest_display_name(label)
            if suggestion:
                self._display.setText(suggestion)

    def _load_classes_from_dataset(self) -> None:
        slug = str(self._dataset.currentData() or "").strip()
        if not slug:
            return
        try:
            payload = self._dataset_payload(slug)
        except Exception as exc:
            QMessageBox.warning(self, "Class Load Failed", str(exc))
            return
        classes = payload.get("classes") if isinstance(payload, dict) else None
        if not isinstance(classes, list) or not classes:
            QMessageBox.information(self, "No Classes Found", "Dataset did not report any classes.")
            return
        lines = "\n".join(str(c) for c in classes if str(c).strip())
        if lines.strip():
            self._classes.setPlainText(lines + "\n")

    def preselect_dataset(self, slug: str) -> None:
        """Pre-fill the dialog for an existing library dataset, silently.

        Picks a recommended backbone from the dataset's reported format, selects
        the dataset in the combo, and pre-loads its classes — so creating a
        scenario from an already-made dataset is just a confirm click. Unlike the
        drag-drop path this raises no prompts (it's invoked programmatically)."""
        name = str(slug or "").strip()
        if not name:
            return
        payload: dict[str, Any] = {}
        try:
            payload = self._dataset_payload(name) or {}
        except Exception:
            payload = {}
        fmt = str(payload.get("format") or "").strip().lower()
        backbone = self._backbone_for_dataset_format(fmt) or "yolo_detection"
        self._set_backbone_type(backbone)
        idx = self._dataset.findData(name)
        if idx < 0:
            self._populate_datasets()
            idx = self._dataset.findData(name)
        if idx >= 0:
            self._dataset.setCurrentIndex(idx)
        raw_classes = payload.get("classes")
        if isinstance(raw_classes, list):
            classes = [str(c) for c in raw_classes if str(c).strip()]
            if classes and not str(self._classes.toPlainText() or "").strip():
                self._classes.setPlainText("\n".join(classes) + "\n")
        self._maybe_autofill_names_from_dataset()

    @staticmethod
    def _backbone_for_dataset_format(fmt: str) -> str:
        return {
            "yolo_detection": "yolo_detection",
            "imagefolder_classification": "face_recognition",
            "face_csv": "face_recognition",
            "audiofolder_classification": "audio_recognition",
            "llm_instruction_jsonl": "llm_fine_tuning",
            "csv_tabular": "torch_tabular",
        }.get(str(fmt or "").strip().lower(), "")

    @staticmethod
    def _category_for_dataset_format(fmt: str) -> str:
        value = str(fmt or "").strip().lower()
        if value == "audiofolder_classification":
            return "audio"
        if value == "llm_instruction_jsonl":
            return "text"
        if value == "csv_tabular":
            return "tabular"
        return "image"

    def _dataset_payload(self, slug: str) -> dict[str, Any]:
        name = str(slug or "").strip()
        if name in self._dataset_info_cache:
            return dict(self._dataset_info_cache[name])
        enc = urllib.parse.quote(name, safe="")
        payload = self._http_get(f"/database/{enc}")
        if isinstance(payload, dict):
            self._dataset_info_cache[name] = dict(payload)
            return payload
        return {}

    # ---------- Create ----------

    def _create(self) -> None:
        import json as _json
        name = str(self._name.text() or "").strip()
        display = str(self._display.text() or "").strip()
        desc = str(self._desc.toPlainText() or "").strip()
        dataset = str(self._dataset.currentData() or "").strip()
        btype = self._current_backbone()
        is_tabular = btype == "torch_tabular"
        is_custom_code = btype == "custom_code"
        is_archival = btype == "archival_ingestion"
        is_face = btype == "face_recognition"
        is_audio = btype == "audio_recognition"
        is_llm = btype == "llm_fine_tuning"

        if not name:
            QMessageBox.warning(self, "Missing Name", "Scenario name is required.")
            return
        if not _SCENARIO_NAME_RE.match(name):
            QMessageBox.warning(
                self,
                "Invalid Name",
                "Use letters/numbers plus '_' or '-' (no spaces).",
            )
            return

        payload: dict = {
            "name": name,
            "display_name": display,
            "description": desc,
            "dataset": dataset,
            "backbone_type": btype,
        }

        if is_custom_code:
            from mlops.pipeline.registry import sanitize_library_dataset_slug as _slug

            dataset_raw = str(self._dataset.currentData() or "").strip()
            dataset_slug = ""
            if dataset_raw:
                try:
                    dataset_slug = _slug(dataset_raw)
                except Exception:
                    dataset_slug = ""
            payload["dataset"] = dataset_slug
            bcfg_raw = str(self._backbone_config_edit.toPlainText() or "").strip()
            backbone_cfg: dict = {"cells": [], "datasets": []}
            if bcfg_raw:
                try:
                    extra = _json.loads(bcfg_raw)
                    if isinstance(extra, dict):
                        backbone_cfg.update(extra)
                except Exception:
                    QMessageBox.warning(
                        self,
                        "Invalid backbone_config",
                        "backbone_config must be valid JSON.",
                    )
                    return
            payload["backbone_config"] = backbone_cfg
        elif is_archival:
            payload["dataset"] = ""
            bcfg_raw = str(self._backbone_config_edit.toPlainText() or "").strip()
            backbone_cfg: dict[str, Any] = {
                "domain_profile": {},
                "assembly_rules": {},
                "providers": {},
                "phase_defaults": {},
                "archive_storage_root": "state/insight_local/cvops/archive_corpora",
            }
            if bcfg_raw:
                try:
                    extra = _json.loads(bcfg_raw)
                    if isinstance(extra, dict):
                        backbone_cfg.update(extra)
                except Exception:
                    QMessageBox.warning(
                        self,
                        "Invalid backbone_config",
                        "backbone_config must be valid JSON.",
                    )
                    return
            payload["backbone_config"] = backbone_cfg
        elif is_tabular:
            # For tabular scenarios, the dataset dropdown is used as a CSV picker
            # (values look like "mlops/datasets/xyz.csv"). The registry expects
            # "dataset" to be a library slug under database/ (optional for tabular),
            # so we avoid passing CSV paths here to prevent "Invalid library dataset name."
            payload["dataset"] = ""
            cells: list[dict[str, str]] = []
            for i in range(self._algo_cells.count()):
                it = self._algo_cells.item(i)
                if it is None:
                    continue
                path = str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "").strip()
                if path:
                    cells.append({"path": path})
            if not cells:
                QMessageBox.warning(
                    self,
                    "Missing Algo Cells",
                    "Add at least one Python file to run as an execution cell.",
                )
                return
            backbone_cfg: dict = {}
            dataset_csv = str(self._dataset_csv.text() or "").strip()
            if dataset_csv:
                backbone_cfg["dataset_csv"] = dataset_csv
            bcfg_raw = str(self._backbone_config_edit.toPlainText() or "").strip()
            if bcfg_raw:
                try:
                    extra = _json.loads(bcfg_raw)
                    if isinstance(extra, dict):
                        backbone_cfg.update(extra)
                except Exception:
                    QMessageBox.warning(
                        self,
                        "Invalid backbone_config",
                        "backbone_config must be valid JSON (e.g. {\"num_classes\": 5}).",
                    )
                    return
            backbone_cfg["cells"] = cells
            payload["backbone_config"] = backbone_cfg
        elif is_llm:
            if not dataset:
                QMessageBox.warning(self, "Missing Dataset", "Select a JSONL instruction dataset.")
                return
            base_model = str(self._llm_base_model.text() or "").strip()
            if not base_model:
                QMessageBox.warning(self, "Missing Base Model", "Enter a Hugging Face model id or local model path.")
                return
            ollama_base = self._ollama_base_text() or base_model
            try:
                learning_rate = float(str(self._llm_lr.text() or "0.0002").strip())
            except Exception:
                QMessageBox.warning(self, "Invalid Learning Rate", "LLM learning rate must be a number.")
                return
            targets = [
                part.strip()
                for part in str(self._llm_targets.text() or "").replace(";", ",").split(",")
                if part.strip()
            ]
            backbone_cfg: dict = {
                "base_model": base_model,
                "ollama_base_model": ollama_base,
                "sources": ["jsonl", "feedback"],
                "epochs": int(self._llm_epochs.value()),
                "max_seq_length": int(self._llm_max_seq.value()),
                "batch_size": int(self._llm_batch.value()),
                "learning_rate": learning_rate,
                "gradient_accumulation_steps": 4,
                "lora_r": int(self._llm_lora_r.value()),
                "lora_alpha": int(self._llm_lora_alpha.value()),
                "lora_dropout": 0.05,
                "target_modules": targets or ["q_proj", "v_proj"],
            }
            bcfg_raw = str(self._backbone_config_edit.toPlainText() or "").strip()
            if bcfg_raw:
                try:
                    extra = _json.loads(bcfg_raw)
                    if isinstance(extra, dict):
                        backbone_cfg.update(extra)
                except Exception:
                    QMessageBox.warning(
                        self,
                        "Invalid backbone_config",
                        "backbone_config must be valid JSON.",
                    )
                    return
            payload["base_model"] = base_model
            payload["classes"] = None
            payload["epochs"] = 1
            payload["imgsz"] = 0
            payload["guard_profile"] = "balanced"
            payload["backbone_config"] = backbone_cfg
        elif is_face or is_audio:
            if not dataset:
                QMessageBox.warning(self, "Missing Dataset", "Select a dataset.")
                return
            payload["base_model"] = ""
            payload["classes"] = None
            payload["epochs"] = 1
            payload["imgsz"] = 0
            payload["guard_profile"] = "balanced"
            bcfg_raw = str(self._backbone_config_edit.toPlainText() or "").strip()
            if bcfg_raw:
                try:
                    extra = _json.loads(bcfg_raw)
                    if isinstance(extra, dict):
                        payload["backbone_config"] = dict(extra)
                except Exception:
                    QMessageBox.warning(
                        self,
                        "Invalid backbone_config",
                        "backbone_config must be valid JSON (e.g. {\"max_classes\": 100}).",
                    )
                    return
        else:
            base_model = str(self._model.currentData() or "").strip()
            if not dataset:
                QMessageBox.warning(self, "Missing Dataset", "Select a dataset.")
                return
            if not base_model:
                QMessageBox.warning(self, "Missing Model", "Select a base model.")
                return
            classes = _parse_classes(self._classes.toPlainText())
            if not classes and not self._dataset_classes:
                QMessageBox.warning(
                    self,
                    "Missing Classes",
                    "Enter at least one class name (or Load From Dataset if classes.txt exists).",
                )
                return
            payload["base_model"] = base_model
            payload["classes"] = classes or None
            payload["epochs"] = int(self._epochs.value())
            payload["imgsz"] = int(self._imgsz.value())
            payload["guard_profile"] = str(self._guard.currentData() or "balanced")

        try:
            res = self._http_post("/scenarios", payload)
        except Exception as exc:
            detail = getattr(exc, "response_body", "") or str(exc)
            QMessageBox.critical(self, "Create Failed", str(detail))
            return
        created = str(res.get("name") or name).strip()
        if not created:
            QMessageBox.critical(self, "Create Failed", "Server returned an invalid response.")
            return
        self._created = created
        self.accept()
