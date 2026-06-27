from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import quote

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

# Notebook (custom_code scenario) names: letters/numbers plus '_' or '-', no spaces.
_NOTEBOOK_NAME_RE = re.compile(r"^[A-Za-z0-9_-]+$")

from .custom_cells_editor import CustomCellsEditor
from .dropdown_pane_stack import DropdownPaneStack
from .cvops_theme import cvops_color, set_cvops_stylesheet


_TYPE_PAGES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("vision", "Vision", ("yolo_detection",)),
    ("tabular", "Tabular", ("torch_tabular",)),
    ("face", "Face", ("face_recognition",)),
    ("audio", "Audio", ("audio_recognition",)),
    ("custom", "Custom", ("custom_code",)),
    ("llm", "LLM", ("llm_fine_tuning",)),
)

_BACKBONE_LABELS: dict[str, str] = {
    "yolo_detection": "Vision",
    "torch_tabular": "Tabular",
    "face_recognition": "Face",
    "audio_recognition": "Audio",
    "custom_code": "Custom",
    "llm_fine_tuning": "LLM",
}

_BUILTIN_RUNTIME: dict[str, dict[str, Any]] = {
    "yolo_detection": {
        "train": ["Native YOLO training pipeline"],
        "infer": ["Resolve Weights", "Run Prediction", "Postprocess"],
        "note": "Inference is cell-driven. Training stays on the native detector trainer.",
    },
    "torch_tabular": {
        "train": [],
        "infer": [],
        "note": "Cells come from backbone_config.cells, train_cells, infer_cells, or cells_module.",
    },
    "face_recognition": {
        "train": ["Build Gallery", "Incremental Update (when backbone_config.incremental=true)"],
        "infer": ["Infer Not Supported"],
        "note": "This backbone trains by building or incrementally updating a gallery database.",
    },
    "audio_recognition": {
        "train": ["Build Audio Model"],
        "infer": ["Recognize Audio"],
        "note": "Audio scenarios keep a compact train/infer cell chain.",
    },
    "custom_code": {
        "train": [],
        "infer": [],
        "note": "Custom scenarios can run draft or promoted Python cells with attached datasets.",
    },
    "llm_fine_tuning": {
        "train": ["Prepare Instruction Data", "Train LoRA Adapter", "Package Ollama Modelfile"],
        "infer": [],
        "note": "LLM fine-tuning scenarios are train-only in this CV Ops version.",
    },
}


class _StatusPill(QLabel):
    def __init__(self, status: str = "empty") -> None:
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setMinimumWidth(96)
        self.setObjectName("statusPill")
        self.set_status(status)

    def set_status(self, status: str) -> None:
        key = str(status or "empty").lower()
        self.setProperty("status", key)
        self.setText(key.replace("_", " ").upper())


def _cell_specs_from(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    specs: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, str):
            path = item.strip()
            if path:
                specs.append({"path": path, "entry": "run"})
            continue
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if not path:
            continue
        spec = dict(item)
        spec["path"] = path
        spec["entry"] = str(spec.get("entry") or "run").strip() or "run"
        specs.append(spec)
    return specs


def _first_meaningful_line(text: str) -> str:
    for raw in str(text or "").splitlines():
        line = raw.strip()
        if line:
            return line[:96]
    return ""


def _label_text(value: Any, fallback: str = "—") -> str:
    text = str(value or "").strip()
    return text or fallback


def _cell_summary(spec: dict[str, Any]) -> str:
    path = str(spec.get("path") or "").strip()
    name = str(spec.get("name") or Path(path).stem or "cell").strip() or "cell"
    entry = str(spec.get("entry") or "run").strip() or "run"
    parts = [name]
    if path:
        parts.append(path)
    if entry != "run":
        parts.append(f"entry:{entry}")
    datasets = spec.get("datasets")
    if isinstance(datasets, list) and datasets:
        parts.append(f"{len(datasets)} dataset(s)")
    return " | ".join(parts)


def _draft_cell_summary(spec: dict[str, Any]) -> str:
    base = _cell_summary(spec)
    preview = _first_meaningful_line(spec.get("code") or "")
    if preview:
        return f"{base} | {preview}"
    return base


def _dataset_summary(dataset: dict[str, Any]) -> str:
    name = _label_text(dataset.get("name"))
    kind = _label_text(dataset.get("kind"), "")
    path = _label_text(dataset.get("path"))
    pieces = [name]
    if kind:
        pieces.append(kind)
    pieces.append(path)
    return " | ".join(pieces)


def _html_list(items: list[str], empty_copy: str) -> str:
    if not items:
        return html.escape(empty_copy)
    escaped = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<ul style='margin: 4px 0 0 16px;'>{escaped}</ul>"


def _clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        child_layout = item.layout()
        if child_layout is not None and isinstance(child_layout, QVBoxLayout):
            _clear_layout(child_layout)
        if widget is not None:
            widget.deleteLater()


class _ScenarioTypePage(QWidget):
    errorRaised = pyqtSignal(str)
    openWorkflowTrainRequested = pyqtSignal(str)
    scenarioSelected = pyqtSignal(dict)

    def __init__(
        self,
        *,
        title: str,
        backbone_types: tuple[str, ...],
        http_get: Optional[Callable[[str], dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._backbone_types = backbone_types
        self._http_get = http_get
        self._entries: list[dict[str, Any]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        self._summary.setObjectName("stageInfo")
        outer.addWidget(self._summary)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer.addWidget(scroll, stretch=1)

        self._cards_host = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_host)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(8)
        scroll.setWidget(self._cards_host)

    def set_entries(self, entries: list[dict[str, Any]]) -> None:
        self._entries = list(entries)
        _clear_layout(self._cards_layout)

        if not self._entries:
            self._summary.setText(f"No {self._title.lower()} scenarios yet.")
            empty = QLabel(f"No {self._title.lower()} scenarios are configured.")
            empty.setWordWrap(True)
            empty.setProperty("muted", True)
            self._cards_layout.addWidget(empty)
            self._cards_layout.addStretch(1)
            return

        configured_total = sum(self._configured_cell_total(entry) for entry in self._entries)
        self._summary.setText(
            f"{len(self._entries)} scenario(s) in {self._title.lower()} with "
            f"{configured_total} configured project-backed cell file(s)."
        )

        ordered = sorted(
            self._entries,
            key=lambda entry: str(entry.get("display_name") or entry.get("name") or "").lower(),
        )
        for entry in ordered:
            self._cards_layout.addWidget(self._build_card(entry))
        self._cards_layout.addStretch(1)

    @staticmethod
    def _configured_cell_total(entry: dict[str, Any]) -> int:
        cfg = entry.get("backbone_config")
        backbone_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        total = 0
        for key in ("cells", "train_cells", "infer_cells"):
            total += len(_cell_specs_from(backbone_cfg.get(key)))
        return total

    def _build_card(self, entry: dict[str, Any]) -> QWidget:
        frame = QFrame()
        frame.setObjectName("opsCell")
        frame.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

        outer = QVBoxLayout(frame)
        outer.setContentsMargins(10, 9, 10, 10)
        outer.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(8)
        title = QLabel(_label_text(entry.get("display_name") or entry.get("name")))
        title.setProperty("isTitle", True)
        title_row.addWidget(title, stretch=0)
        title_row.addStretch(1)
        pill = _StatusPill(str(entry.get("status") or "empty"))
        title_row.addWidget(pill)
        outer.addLayout(title_row)

        meta_parts = [
            f"scenario: {_label_text(entry.get('name'))}",
            f"dataset: {_label_text(entry.get('dataset'))}",
            f"type: {_label_text(_BACKBONE_LABELS.get(str(entry.get('backbone_type') or '').strip().lower()), 'Unknown')}",
            f"items: {int(entry.get('dataset_count') or 0)}",
        ]
        if entry.get("history_count") is not None:
            meta_parts.append(f"runs: {int(entry.get('history_count') or 0)}")
        meta = QLabel(" | ".join(meta_parts))
        meta.setWordWrap(True)
        meta.setProperty("muted", True)
        outer.addWidget(meta)

        backbone_type = str(entry.get("backbone_type") or "").strip().lower()
        runtime = _BUILTIN_RUNTIME.get(backbone_type, {})
        train_cells = list(runtime.get("train") or [])
        infer_cells = list(runtime.get("infer") or [])
        note = str(runtime.get("note") or "").strip()

        self._add_section(
            outer,
            "Built-In Train Cells",
            train_cells,
            "This scenario type does not register built-in train cells.",
        )
        self._add_section(
            outer,
            "Built-In Infer Cells",
            infer_cells,
            "This scenario type does not register built-in infer cells.",
        )
        if note:
            note_label = QLabel(note)
            note_label.setWordWrap(True)
            note_label.setProperty("muted", True)
            outer.addWidget(note_label)

        cfg = entry.get("backbone_config")
        backbone_cfg = dict(cfg) if isinstance(cfg, dict) else {}
        self._add_section(
            outer,
            "Configured Default Cells",
            [_cell_summary(spec) for spec in _cell_specs_from(backbone_cfg.get("cells"))],
            "No backbone_config.cells entries.",
        )
        self._add_section(
            outer,
            "Configured Train Overrides",
            [_cell_summary(spec) for spec in _cell_specs_from(backbone_cfg.get("train_cells"))],
            "No backbone_config.train_cells overrides.",
        )
        self._add_section(
            outer,
            "Configured Infer Overrides",
            [_cell_summary(spec) for spec in _cell_specs_from(backbone_cfg.get("infer_cells"))],
            "No backbone_config.infer_cells overrides.",
        )
        cells_module = str(backbone_cfg.get("cells_module") or "").strip()
        if cells_module:
            module_label = QLabel(f"Legacy cells_module: {cells_module}")
            module_label.setWordWrap(True)
            outer.addWidget(module_label)

        scenario_key = str(entry.get("name") or "").strip()
        actions = QHBoxLayout()
        actions.setContentsMargins(0, 4, 0, 0)
        actions.setSpacing(8)
        open_btn = QPushButton("Open in Workbench (Train)")
        open_btn.setToolTip(
            "Switch to the Workbench tab, select this scenario, and open the Train step. "
            "There you can configure backbone cells, attach algo overrides, and for custom_code "
            "scenarios use Save Draft / Run Draft in the Custom Cells section."
        )
        open_btn.clicked.connect(
            lambda _checked=False, sk=scenario_key: self.openWorkflowTrainRequested.emit(sk)
        )
        open_btn.setEnabled(bool(scenario_key))
        actions.addWidget(open_btn)
        select_btn = QPushButton("Edit in Cells")
        select_btn.setToolTip("Select this scenario in the Cells workspace.")
        select_btn.clicked.connect(
            lambda _checked=False, selected=dict(entry): self.scenarioSelected.emit(selected)
        )
        select_btn.setEnabled(bool(scenario_key))
        actions.addWidget(select_btn)
        actions.addStretch(1)
        outer.addLayout(actions)

        if backbone_type == "custom_code":
            draft = self._load_custom_draft(str(entry.get("name") or ""))
            self._add_section(
                outer,
                "Draft Cells",
                [_draft_cell_summary(spec) for spec in draft.get("cells") or [] if isinstance(spec, dict)],
                "No draft custom cells saved yet.",
            )
            self._add_section(
                outer,
                "Draft Scenario Datasets",
                [_dataset_summary(dataset) for dataset in draft.get("scenario_datasets") or [] if isinstance(dataset, dict)],
                "No draft scenario datasets attached.",
            )

        return frame

    def _load_custom_draft(self, scenario: str) -> dict[str, Any]:
        if self._http_get is None or not scenario:
            return {"scenario": scenario, "cells": [], "scenario_datasets": []}
        try:
            encoded = quote(str(scenario or "").strip(), safe="")
            payload = self._http_get(f"/scenarios/{encoded}/custom_cells")
            if isinstance(payload, dict):
                return payload
        except Exception as exc:
            self.errorRaised.emit(f"Unable to load custom cells for {scenario}: {exc}")
        return {"scenario": scenario, "cells": [], "scenario_datasets": []}

    @staticmethod
    def _add_section(layout: QVBoxLayout, title: str, items: list[str], empty_copy: str) -> None:
        label = QLabel(f"<b>{html.escape(title)}</b>{_html_list(items, empty_copy)}")
        label.setTextFormat(Qt.TextFormat.RichText)
        label.setWordWrap(True)
        layout.addWidget(label)


class CellsPanel(QWidget):
    errorRaised = pyqtSignal(str)
    openWorkflowTrainRequested = pyqtSignal(str)
    scenarioMutated = pyqtSignal(str)
    trainKicked = pyqtSignal(str, str)

    def __init__(
        self,
        *,
        http_get: Callable[[str], dict[str, Any]],
        http_put: Optional[Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]] = None,
        http_post: Optional[Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cellsWorkspaceMode")
        self._http_get = http_get
        self._http_put = http_put
        self._http_post = http_post
        self._scenarios: list[dict[str, Any]] = []
        self._entries_by_name: dict[str, dict[str, Any]] = {}
        self._custom_entries: list[dict[str, Any]] = []
        self._selected_scenario = ""
        self._pending_select = ""
        set_cvops_stylesheet(self, self._cells_qss)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(10)

        # Always-visible header so notebooks can be browsed and created from here
        # directly — a small Colab/Paperspace-style workspace — without having to
        # detour through the Workbench New Scenario flow first.
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)
        header_title = QLabel("Notebooks")
        header_title.setObjectName("cellsModeHeaderTitle")
        header.addWidget(header_title)
        self._notebook_count = QLabel("")
        self._notebook_count.setObjectName("cellsModeFieldLabel")
        header.addWidget(self._notebook_count)
        header.addStretch(1)
        self._new_btn = QPushButton("New notebook")
        self._new_btn.setObjectName("cellsModePrimaryButton")
        self._new_btn.setToolTip("Create a new Custom Code notebook (scenario) and open it here.")
        self._new_btn.clicked.connect(self._create_notebook)
        self._new_btn.setEnabled(self._http_post is not None)
        header.addWidget(self._new_btn)
        outer.addLayout(header)

        scenario_label = QLabel("Scenario")
        scenario_label.setObjectName("cellsModeFieldLabel")
        self._scenario_combo = QComboBox()
        self._scenario_combo.setObjectName("cellsModeScenarioCombo")
        self._scenario_combo.setMinimumWidth(260)
        self._scenario_combo.currentIndexChanged.connect(self._on_scenario_combo_changed)

        self._reload_btn = QPushButton("Reload")
        self._reload_btn.setObjectName("cellsModeSecondaryButton")
        self._reload_btn.setToolTip("Reload the selected draft from the server.")
        self._reload_btn.clicked.connect(self._reload_selected_scenario)

        self._open_train_btn = QPushButton("Open Train")
        self._open_train_btn.setObjectName("cellsModePrimaryButton")
        self._open_train_btn.setToolTip(
            "Jump to the Workbench Train preset for the selected scenario."
        )
        self._open_train_btn.clicked.connect(self._emit_open_train)

        self._locked_shell = QFrame()
        self._locked_shell.setObjectName("cellsModeEmpty")
        self._locked_shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        locked_layout = QVBoxLayout(self._locked_shell)
        locked_layout.setContentsMargins(20, 20, 20, 20)
        locked_layout.setSpacing(8)
        locked_title = QLabel("No custom notebooks yet")
        locked_title.setObjectName("cellsModeEmptyTitle")
        locked_layout.addWidget(locked_title)
        self._locked = QLabel(
            "Notebooks are Custom Code scenarios — draft Python files you edit and run cell-by-cell with live output. "
            "Create one here to start, or open an existing Custom Code scenario from the Scenario picker above."
        )
        self._locked.setWordWrap(True)
        self._locked.setObjectName("cellsModeEmptyBody")
        locked_layout.addWidget(self._locked)
        locked_actions = QHBoxLayout()
        locked_actions.setContentsMargins(0, 4, 0, 0)
        locked_actions.setSpacing(8)
        self._locked_new_btn = QPushButton("Create your first notebook")
        self._locked_new_btn.setObjectName("cellsModePrimaryButton")
        self._locked_new_btn.clicked.connect(self._create_notebook)
        self._locked_new_btn.setEnabled(self._http_post is not None)
        locked_actions.addWidget(self._locked_new_btn)
        locked_actions.addStretch(1)
        locked_layout.addLayout(locked_actions)
        locked_layout.addStretch(1)
        outer.addWidget(self._locked_shell)

        self._editor = CustomCellsEditor(
            http_get=self._http_get,
            http_put=self._http_put,
            http_post=self._http_post,
        )
        self._editor.errorRaised.connect(self.errorRaised)
        self._editor.draftSaved.connect(self.scenarioMutated.emit)
        self._editor.scenarioMutated.connect(self.scenarioMutated.emit)
        self._editor.runDraftRequested.connect(self._run_draft)
        self._editor.install_context_controls(
            scenario_label,
            self._scenario_combo,
            self._reload_btn,
            self._open_train_btn,
        )
        self._editor.setVisible(False)
        outer.addWidget(self._editor, stretch=1)
        self._reload_btn.setEnabled(False)
        self._open_train_btn.setEnabled(False)

    @staticmethod
    def _cells_qss() -> str:
        panel = cvops_color("bg_panel")
        field = cvops_color("bg_void")
        text = cvops_color("text_signal")
        muted = cvops_color("text_iron")
        border = cvops_color("line_light")
        accent = cvops_color("line_bright")
        return f"""
QWidget#cellsWorkspaceMode {{
    background: transparent;
}}
QLabel#cellsModeHeaderTitle {{
    color: {text};
    font-size: 14px;
    font-weight: 700;
    background: transparent;
    border: none;
}}
QFrame#cellsModeEmpty {{
    background: {panel};
    border: 1px solid {border};
    border-radius: 0px;
}}
QLabel#cellsModeEmptyTitle {{
    color: {text};
    font-size: 14px;
    font-weight: 700;
    background: transparent;
    border: none;
}}
QLabel#cellsModeEmptyBody,
QLabel#cellsModeFieldLabel {{
    color: {muted};
    font-size: 10px;
    background: transparent;
    border: none;
}}
QComboBox#cellsModeScenarioCombo {{
    min-height: 30px;
    padding: 4px 10px;
    border: 1px solid {border};
    border-radius: 0px;
    background: {field};
    color: {text};
}}
QPushButton#cellsModePrimaryButton,
QPushButton#cellsModeSecondaryButton {{
    min-height: 30px;
    padding: 4px 12px;
}}
QPushButton#cellsModePrimaryButton {{
    background: {accent};
    color: {field};
    border: 1px solid {accent};
}}
QPushButton#cellsModePrimaryButton:hover {{
    color: {text};
}}
QPushButton#cellsModeSecondaryButton {{
    background: transparent;
    color: {text};
    border: 1px solid {border};
}}
QPushButton#cellsModeSecondaryButton:hover {{
    border-color: {accent};
    color: {accent};
}}
"""

    def set_scenarios(self, scenarios: list[dict[str, Any]]) -> None:
        self._scenarios = list(scenarios)
        self._entries_by_name = {
            str(entry.get("name") or ""): entry
            for entry in self._scenarios
            if str(entry.get("name") or "").strip()
        }
        self._custom_entries = sorted(
            [
                entry
                for entry in self._scenarios
                if str(entry.get("backbone_type") or "").strip().lower() == "custom_code"
            ],
            key=lambda entry: str(entry.get("display_name") or entry.get("name") or "").lower(),
        )

        count = len(self._custom_entries)
        self._notebook_count.setText(f"{count} notebook(s)" if count else "")

        current = self._selected_scenario
        self._scenario_combo.blockSignals(True)
        self._scenario_combo.clear()
        for entry in self._custom_entries:
            name = str(entry.get("name") or "").strip()
            if not name:
                continue
            label = _label_text(entry.get("display_name") or name)
            self._scenario_combo.addItem(label, userData=name)
        self._scenario_combo.blockSignals(False)

        if not self._custom_entries:
            self._selected_scenario = ""
            self._pending_select = ""
            self._editor.clear_scenario("No Custom Code scenarios configured.")
            self._editor.setVisible(False)
            self._locked_shell.setVisible(True)
            self._reload_btn.setEnabled(False)
            self._open_train_btn.setEnabled(False)
            return

        custom_names = {
            str(entry.get("name") or "").strip()
            for entry in self._custom_entries
            if str(entry.get("name") or "").strip()
        }
        # Prefer a notebook just created from this panel, then the prior selection,
        # then the first available notebook.
        target = ""
        if self._pending_select and self._pending_select in custom_names:
            target = self._pending_select
        elif current in custom_names:
            target = current
        if not target:
            target = str(self._custom_entries[0].get("name") or "").strip()
        self._pending_select = ""
        self.select_scenario(target)

    def apply_cell_progress(self, payload: dict[str, Any]) -> None:
        """Forward a cell_progress websocket event to the inline editor terminal."""
        self._editor.apply_cell_progress(payload)

    def _create_notebook(self) -> None:
        if self._http_post is None:
            self.errorRaised.emit("Creating a notebook requires HTTP POST from the app shell.")
            return
        name, ok = QInputDialog.getText(
            self,
            "New Notebook",
            "Notebook name (letters/numbers plus '_' or '-', no spaces):",
        )
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            return
        if not _NOTEBOOK_NAME_RE.match(name):
            self.errorRaised.emit("Invalid notebook name — use letters/numbers plus '_' or '-' (no spaces).")
            return
        if name in self._entries_by_name:
            self.errorRaised.emit(f"A scenario named '{name}' already exists.")
            return
        payload = {
            "name": name,
            "display_name": name,
            "description": "",
            "dataset": "",
            "backbone_type": "custom_code",
            "backbone_config": {"cells": [], "datasets": []},
        }
        try:
            res = self._http_post("/scenarios", payload)
        except Exception as exc:
            body = getattr(exc, "response_body", "") or str(exc)
            self.errorRaised.emit(f"Create notebook failed: {body}")
            return
        created = str(res.get("name") or name).strip() if isinstance(res, dict) else name
        # Auto-select the new notebook once the refreshed scenario list arrives.
        self._pending_select = created
        # scenarioMutated is wired to the app shell's scenario refresh, which calls
        # set_scenarios() back on this panel with the new custom_code entry.
        self.scenarioMutated.emit(created)

    def select_scenario(self, scenario: str) -> None:
        name = str(scenario or "").strip()
        if not name:
            return
        idx = self._scenario_combo.findData(name)
        if idx < 0:
            return
        self._scenario_combo.blockSignals(True)
        self._scenario_combo.setCurrentIndex(idx)
        self._scenario_combo.blockSignals(False)
        entry = self._entries_by_name.get(name)
        if isinstance(entry, dict):
            self._select_entry(entry)

    def _on_scenario_combo_changed(self, _index: int) -> None:
        name = str(self._scenario_combo.currentData() or "").strip()
        if not name:
            return
        entry = self._entries_by_name.get(name)
        if isinstance(entry, dict):
            self._select_entry(entry)

    def _reload_selected_scenario(self) -> None:
        name = self._selected_scenario
        if not name:
            return
        self._editor.load_from_server(name)

    def _emit_open_train(self) -> None:
        if self._selected_scenario:
            self.openWorkflowTrainRequested.emit(self._selected_scenario)

    def _select_entry(self, entry: dict[str, Any]) -> None:
        name = str(entry.get("name") or "").strip()
        btype = str(entry.get("backbone_type") or "").strip().lower()
        self._selected_scenario = name
        title = _label_text(entry.get("display_name") or name, "Cell Space")
        if not name:
            self._editor.setVisible(False)
            self._locked_shell.setVisible(True)
            self._locked.setText("Select a Custom Code scenario to create and run draft notebook files.")
            self._reload_btn.setEnabled(False)
            self._open_train_btn.setEnabled(False)
            return
        if btype != "custom_code":
            self._editor.clear_scenario("")
            self._editor.setVisible(False)
            self._locked_shell.setVisible(True)
            pretty = _BACKBONE_LABELS.get(btype, btype.replace("_", " ").title())
            self._locked.setText(
                f"{title} uses the {pretty} pipeline. Draft notebook editing is only available for Custom Code scenarios."
            )
            self._reload_btn.setEnabled(False)
            self._open_train_btn.setEnabled(bool(name))
            return
        self._locked_shell.setVisible(False)
        self._editor.setVisible(True)
        self._editor.set_scenario(name, entry)
        self._reload_btn.setEnabled(True)
        self._open_train_btn.setEnabled(True)

    def _run_draft(self, scenario: str, override: dict[str, Any]) -> None:
        name = str(scenario or "").strip()
        if not name:
            return
        if self._http_post is None:
            self._editor.set_status("Run draft requires HTTP POST from the app shell.")
            return
        payload: dict[str, Any] = {}
        if override:
            payload["backbone_config_override"] = override
        try:
            result = self._http_post(f"/scenarios/{name}/train", payload or None)
        except Exception as exc:
            body = getattr(exc, "response_body", "") or str(exc)
            msg = f"Run draft failed: {body}"
            self._editor.set_status(msg)
            self.errorRaised.emit(msg)
            return
        job_id = str(result.get("job_id") or "") if isinstance(result, dict) else ""
        self._editor.append_console_line(f"[machine] notebook run queued: {job_id}", "stdout")
        self._editor.set_status(f"Notebook run queued: {job_id}")
        self.trainKicked.emit(name, job_id)
        self.scenarioMutated.emit(name)
