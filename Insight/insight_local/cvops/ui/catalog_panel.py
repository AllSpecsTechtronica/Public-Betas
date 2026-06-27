from __future__ import annotations

import importlib
import json
import re
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


def _ghost(button: QAbstractButton, *, min_width: int = 0) -> QAbstractButton:
    """Mark a button as the secondary 'ghost' variant for the global QSS rule."""
    button.setProperty("variant", "ghost")
    if min_width:
        button.setMinimumWidth(min_width)
    button.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
    style = button.style()
    style.unpolish(button)
    style.polish(button)
    return button

from .collapsible_section import CollapsibleSection
from .cvops_theme import (
    WB_TEXT_IRON,
    WB_LINE_LIGHT,
    WB_FONT_MONO,
    repolish,
)
from .selectable_panel import SelectablePanel
from .status_pill import StatusPill
from .time_format import format_datetime_text, format_duration_seconds
from ...config import ROOT_DIR


_OP_SLOTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("all", "All", ()),
    ("vision", "Vision", ("yolo_detection",)),
    ("tabular", "Tabular", ("torch_tabular", "custom_code")),
    ("archive", "Archive", ("archival_ingestion",)),
    ("audio", "Audio", ("audio_recognition",)),
    ("face", "Face", ("face_recognition",)),
    ("llm", "LLM", ("llm_fine_tuning",)),
)

_BACKBONE_LABELS: dict[str, str] = {
    "yolo_detection": "Vision (YOLO)",
    "torch_tabular": "Tabular",
    "custom_code": "Custom code",
    "archival_ingestion": "Archive",
    "face_recognition": "Face",
    "audio_recognition": "Audio",
    "llm_fine_tuning": "LLM",
}


def _backbone_table_label(backbone: str) -> str:
    b = str(backbone or "").strip().lower() or "yolo_detection"
    return _BACKBONE_LABELS.get(b, b.replace("_", " ").title())


def _lazy_symbol(module_path: str, symbol: str) -> object:
    module = importlib.import_module(module_path, package=__package__)
    return getattr(module, symbol)


class _GuardFetcher(QThread):
    """Background thread: fetches /scenarios/{scenario}/guard without blocking the UI."""

    finished: pyqtSignal = pyqtSignal(str, dict)

    def __init__(self, base_url: str, scenario: str, parent: Optional[Any] = None) -> None:
        super().__init__(parent)
        self._base_url = str(base_url).rstrip("/")
        self._scenario = scenario

    def run(self) -> None:
        try:
            slug = urllib.parse.quote(self._scenario, safe="")
            url = f"{self._base_url}/scenarios/{slug}/guard"
            with urllib.request.urlopen(url, timeout=12) as resp:  # noqa: S310
                data = json.loads(resp.read())
            if isinstance(data, dict):
                self.finished.emit(self._scenario, data)
        except Exception:
            pass


class CatalogPanel(SelectablePanel, QWidget):
    """Scenario catalog: left table + right detail (status / dataset / train / verify)."""

    panel_entity_type = "scenario"

    scenarioSelected = pyqtSignal(str)
    trainKicked = pyqtSignal(str, str)   # scenario, job_id
    scenarioMutated = pyqtSignal(str)    # scenario name (dataset, train, verify)
    errorRaised = pyqtSignal(str)
    artifactsJobRequested = pyqtSignal(str)   # job_id to load in artifacts panel

    def __init__(
        self,
        base_url: str,
        http_get: Callable[[str], dict[str, Any]],
        http_post: Callable[[str, Optional[dict[str, Any]]], dict[str, Any]],
        http_delete: Callable[[str], dict[str, Any]],
        http_get_text: Optional[Callable[[str], str]] = None,
        http_put: Optional[Callable[[str, Optional[dict[str, Any]]], dict[str, Any]]] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._http_get = http_get
        self._http_post = http_post
        self._http_put = http_put
        self._http_delete = http_delete
        self._http_get_text = http_get_text or (lambda _p: "")
        self._latest_train_jobs: dict[str, str] = {}
        self._console_buffers: dict[str, list[tuple[str, str]]] = {}
        self._entries: dict[str, dict[str, Any]] = {}
        self._cc_cells_data: list[dict[str, Any]] = []
        self._cc_prev_list_row: int = -1
        self._cc_loaded_scenario: str = ""
        self._models: list[dict[str, Any]] = []
        self._training_points: dict[str, list[dict[str, Any]]] = {}
        self._pending_select_scenario = ""
        self._guard_fetcher: Optional[QThread] = None
        self._guard_fetch_scenario: str = ""
        self._op_slot = "all"
        self._op_buttons: dict[str, QPushButton] = {}
        self._guard_card: Any = None
        self._dataset_panel: Any = None
        self._audio_studio_panel: Any = None
        self._dataset_stack: Optional[QStackedWidget] = None
        self._pending_dataset_context: tuple[str, str, str, dict[str, Any]] = ("", "", "", {})
        self._hp_panel: Any = None
        self._train_console: Any = None
        self._ci_cd_bar: Any = None
        self._artifacts_panel: Any = None
        self._history_panel: Any = None
        self._run_compare_panel: Any = None
        self._data_viz_panel: Any = None
        self._custom_cells_editor: Any = None
        self._pending_guard_data: Optional[dict[str, Any]] = None
        self._algo_catalog_loaded = False
        self._responsive_timer = QTimer(self)
        self._responsive_timer.setSingleShot(True)
        self._responsive_timer.setInterval(48)
        self._responsive_timer.timeout.connect(self._run_responsive_refresh)
        self._guard_poll_timer = QTimer(self)
        self._guard_poll_timer.setInterval(5000)
        self._guard_poll_timer.timeout.connect(self._poll_guard_for_current)
        self._guard_poll_timer.start()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(3)

        # Left: scenario list
        left = QWidget()
        left.setMinimumWidth(28)
        left.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        self._list_widget = left
        left_root = QVBoxLayout(left)
        left_root.setContentsMargins(0, 0, 0, 0)
        left_root.setSpacing(0)
        self._list_stack = QStackedWidget(left)
        left_root.addWidget(self._list_stack, stretch=1)

        list_page = QWidget()
        self._list_page = list_page
        ll = QVBoxLayout(list_page)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        list_head = QHBoxLayout()
        list_title = QLabel("Scenarios")
        list_title.setProperty("isTitle", True)
        list_head.addWidget(list_title, stretch=0)
        list_head.addStretch(1)

        self._new_scenario_btn = QPushButton("New Scenario")
        self._new_scenario_btn.clicked.connect(self._open_new_scenario_dialog)
        self._new_scenario_btn.setMinimumWidth(118)
        self._new_scenario_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        list_head.addWidget(self._new_scenario_btn)
        self._scenario_header_refresh_btn = QPushButton("↻")
        self._scenario_header_refresh_btn.setToolTip("Refresh the scenario catalog")
        self._scenario_header_refresh_btn.clicked.connect(self._request_scenarios_refresh)
        self._scenario_header_refresh_btn.setFixedWidth(28)
        self._scenario_header_refresh_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        _ghost(self._scenario_header_refresh_btn)
        list_head.addWidget(self._scenario_header_refresh_btn)
        ll.addLayout(list_head)

        self._refresh_scenarios_btn = QPushButton("Refresh")
        self._refresh_scenarios_btn.clicked.connect(self._request_scenarios_refresh)
        _ghost(self._refresh_scenarios_btn)
        self._refresh_scenarios_btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        list_block = QWidget()
        lb = QHBoxLayout(list_block)
        lb.setContentsMargins(0, 0, 0, 0)
        lb.setSpacing(10)

        slot_rail = QFrame()
        slot_rail.setObjectName("scenarioSlotRail")
        sr = QVBoxLayout(slot_rail)
        sr.setContentsMargins(0, 2, 0, 0)
        sr.setSpacing(4)
        rail_btns = QVBoxLayout()
        rail_btns.setSpacing(4)
        rail_btns.addWidget(self._refresh_scenarios_btn)
        sr.addLayout(rail_btns)
        rail_title = QLabel("Type")
        rail_title.setProperty("muted", True)
        repolish(rail_title)
        sr.addWidget(rail_title)
        self._op_group = QButtonGroup(self)
        self._op_group.setExclusive(True)
        for slot, label, _types in _OP_SLOTS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("slotFilter", True)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.setMinimumWidth(112)
            btn.clicked.connect(lambda _checked=False, s=slot: self._set_op_slot(s))
            self._op_group.addButton(btn)
            self._op_buttons[slot] = btn
            sr.addWidget(btn)
        self._op_buttons["all"].setChecked(True)
        sr.addStretch(1)
        lb.addWidget(slot_rail)

        table_wrap = QWidget()
        table_col = QVBoxLayout(table_wrap)
        table_col.setContentsMargins(0, 0, 0, 0)
        table_col.setSpacing(4)

        self._filter_edit = QLineEdit()
        self._filter_edit.setObjectName("scenarioFilter")
        self._filter_edit.setPlaceholderText("Filter current slot… name, status:trained, type:yolo")
        self._filter_edit.setClearButtonEnabled(True)
        self._filter_edit.textChanged.connect(self._apply_scenario_filter)
        table_col.addWidget(self._filter_edit)

        self._scenario_table = QTableWidget(0, 3)
        self._scenario_table.setObjectName("scenarioCatalogTable")
        self._scenario_table.setHorizontalHeaderLabels(["Name", "Type", "Status"])
        self._scenario_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._scenario_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._scenario_table.setShowGrid(True)
        self._scenario_table.setAlternatingRowColors(True)
        self._scenario_table.verticalHeader().setVisible(False)
        self._scenario_table.setTabKeyNavigation(False)
        hdr = self._scenario_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._scenario_table.setMinimumHeight(120)
        self._scenario_table.itemSelectionChanged.connect(self._on_selection)

        # Swap the table for a centered status card when scenarios are connecting,
        # timed out, or failed to load — with a Refresh button to retry.
        self._scenario_stack = QStackedWidget()
        self._scenario_stack.addWidget(self._scenario_table)          # index 0
        self._scenario_status_page = self._build_scenario_status_page()  # index 1
        self._scenario_stack.addWidget(self._scenario_status_page)
        table_col.addWidget(self._scenario_stack, stretch=1)

        lb.addWidget(table_wrap, stretch=1)
        ll.addWidget(list_block, stretch=1)
        self._filter_status = QLabel("")
        self._filter_status.setProperty("muted", True)
        self._filter_status.setStyleSheet("font-size: 10px; padding: 2px 4px;")
        self._filter_status.setVisible(False)
        ll.addWidget(self._filter_status)
        self._list_stack.addWidget(list_page)
        self._list_status_page = self._build_left_catalog_status_page()
        self._list_stack.addWidget(self._list_status_page)
        splitter.addWidget(left)

        # Right: header + vertical splitter. Wrap in a scroll area so small windows scroll instead
        # of forcing splitters/cards to squeeze into unreadable clipped layouts.
        right_scroll = QScrollArea()
        self._detail_widget = right_scroll
        right_scroll.setMinimumWidth(28)
        right_scroll.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self._right_stack = QStackedWidget()
        self._right_stack.setMinimumWidth(28)
        self._right_stack.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        self._empty_view = self._build_empty_view()
        self._right_stack.addWidget(self._empty_view)
        self._catalog_status_view = self._build_catalog_status_view()
        self._right_stack.addWidget(self._catalog_status_view)

        right = QWidget()
        right.setMinimumWidth(28)
        right.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(6)
        rl.setSizeConstraint(QLayout.SizeConstraint.SetDefaultConstraint)

        head_row = QHBoxLayout()
        self._title = QLabel("Select a scenario")
        self._title.setProperty("isTitle", True)
        self._title.setStyleSheet("font-size: 12px; font-weight: 700;")
        head_row.addWidget(self._title, stretch=0)
        head_row.addStretch(1)
        self._detail_pill = StatusPill("empty")
        head_row.addWidget(self._detail_pill)
        rl.addLayout(head_row)

        self._meta = QLabel("")
        self._meta.setWordWrap(True)
        rl.addWidget(self._meta)

        self._readiness_strip = QFrame()
        self._readiness_strip.setObjectName("readinessStrip")
        self._readiness_strip.setMinimumWidth(28)
        self._readiness_strip.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        rs = QHBoxLayout(self._readiness_strip)
        rs.setContentsMargins(8, 7, 8, 7)
        rs.setSpacing(6)
        self._ready_dataset = self._readiness_item("Dataset", "0")
        self._ready_model = self._readiness_item("Model", "Unset")
        self._ready_train = self._readiness_item("Train", "Empty")
        self._ready_latest = self._readiness_item("Latest", "None")
        self._ready_verified = self._readiness_item("Verified", "No")
        for widget in (
            self._ready_dataset,
            self._ready_model,
            self._ready_train,
            self._ready_latest,
            self._ready_verified,
        ):
            rs.addWidget(widget, stretch=1)
        rl.addWidget(self._readiness_strip)

        guard_card = self._card("System & Guard")
        self._train_device_override: str = ""
        self._train_storage_override: str = ""
        self._guard_host = self._lazy_panel_host(
            "System guard controls",
            self._ensure_guard_card,
        )
        guard_card.body_layout().addWidget(self._guard_host)
        rl.addWidget(guard_card)

        self._detail_main_split = QSplitter(Qt.Orientation.Horizontal)
        self._detail_main_split.setObjectName("catalogDetailMainSplit")
        self._detail_main_split.setChildrenCollapsible(False)
        self._detail_main_split.setHandleWidth(3)

        workflow_tab = QWidget()
        workflow_tab.setMinimumWidth(28)
        workflow_tab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        workflow_layout = QVBoxLayout(workflow_tab)
        workflow_layout.setContentsMargins(0, 0, 0, 0)
        workflow_layout.setSpacing(6)
        workflow_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self._detail_splitter = QSplitter(Qt.Orientation.Vertical)
        self._detail_splitter.setChildrenCollapsible(False)
        self._detail_splitter.setHandleWidth(4)

        cards_card = self._card_collapsed("Model & Dataset Cards")
        self._cards_browser = QTextBrowser()
        self._cards_browser.setObjectName("cardsBrowser")
        self._cards_browser.setReadOnly(True)
        self._cards_browser.setOpenExternalLinks(False)
        self._cards_browser.setPlaceholderText("Markdown cards load when a scenario is selected.")
        self._cards_browser.setMinimumHeight(64)
        self._cards_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cards_card.body_layout().addWidget(self._cards_browser, stretch=1)

        # Dataset card — stacks the generic DatasetPanel with the audio-only
        # AudioStudioPanel; the active one is chosen per scenario backbone.
        self._dataset_card = self._card("Dataset Readiness")
        self._dataset_host = self._lazy_panel_host(
            "Dataset tools",
            self._ensure_dataset_panels,
        )
        self._dataset_card.body_layout().addWidget(self._dataset_host, stretch=1)

        # Train / verify card
        train_card = self._card("Training")
        train_body = train_card.body_layout()
        self._model_controls_wrap = QWidget()
        model_row = QHBoxLayout(self._model_controls_wrap)
        model_row.setContentsMargins(0, 0, 0, 0)
        model_row.addWidget(QLabel("Base Model:"))
        self._model_combo = QComboBox()
        model_row.addWidget(self._model_combo, stretch=1)
        self._refresh_models_btn = QPushButton("Reload")
        self._refresh_models_btn.clicked.connect(lambda: self._load_models(force=True))
        self._set_model_btn = QPushButton("Apply Model")
        self._set_model_btn.clicked.connect(self._set_model)
        _ghost(self._refresh_models_btn)
        model_row.addWidget(self._refresh_models_btn)
        model_row.addWidget(self._set_model_btn)
        train_body.addWidget(self._model_controls_wrap)

        # Tabular-only: quick algo override cells picker (does not modify scenario YAML).
        self._algo_override_wrap = QWidget()
        aov = QVBoxLayout(self._algo_override_wrap)
        aov.setContentsMargins(0, 0, 0, 0)
        aov.setSpacing(4)
        title = QLabel("Algo cells override (.py):")
        title.setStyleSheet("font-weight: 600;")
        aov.addWidget(title)

        cat_row = QHBoxLayout()
        cat_row.setSpacing(6)
        self._algo_catalog = QListWidget()
        self._algo_catalog.setMinimumHeight(60)
        self._algo_catalog.setMaximumHeight(140)
        cat_row.addWidget(self._algo_catalog, stretch=1)
        cat_btns = QVBoxLayout()
        cat_btns.setSpacing(4)
        self._algo_catalog_refresh = QPushButton("Refresh")
        self._algo_catalog_refresh.clicked.connect(self._populate_algo_catalog)
        self._algo_catalog_add = QPushButton("Add")
        self._algo_catalog_add.clicked.connect(self._add_selected_catalog_algo)
        self._algo_catalog_reveal = QPushButton("Reveal")
        self._algo_catalog_reveal.clicked.connect(self._reveal_selected_catalog_algo)
        # Add is the meaningful action in this column → primary, others ghost.
        _ghost(self._algo_catalog_refresh, min_width=92)
        _ghost(self._algo_catalog_reveal, min_width=92)
        self._algo_catalog_add.setMinimumWidth(92)
        self._algo_catalog_add.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        cat_btns.addWidget(self._algo_catalog_refresh)
        cat_btns.addWidget(self._algo_catalog_add)
        cat_btns.addWidget(self._algo_catalog_reveal)
        cat_btns.addStretch(1)
        cat_row.addLayout(cat_btns)
        aov.addLayout(cat_row)

        aov.addWidget(QLabel("Selected cells (next run only):"))
        self._algo_override_cells = QListWidget()
        self._algo_override_cells.setMinimumHeight(60)
        self._algo_override_cells.setMaximumHeight(120)
        aov.addWidget(self._algo_override_cells)
        a_btns = QHBoxLayout()
        a_btns.setSpacing(6)
        self._algo_reveal_btn = QPushButton("Reveal")
        self._algo_reveal_btn.clicked.connect(self._reveal_selected_algo_override_cell)
        self._algo_remove_btn = QPushButton("Remove")
        self._algo_remove_btn.clicked.connect(self._remove_selected_algo_override_cell)
        self._algo_clear_btn = QPushButton("Clear")
        self._algo_clear_btn.clicked.connect(lambda: self._algo_override_cells.clear())
        for btn in (self._algo_reveal_btn, self._algo_remove_btn, self._algo_clear_btn):
            _ghost(btn)
            a_btns.addWidget(btn)
        a_btns.addStretch(1)
        aov.addLayout(a_btns)
        hint = QLabel("Used for the next training run only (sent as backbone_config_override).")
        hint.setStyleSheet(f"font-size: 10px; color: {WB_TEXT_IRON};")
        hint.setWordWrap(True)
        aov.addWidget(hint)
        self._algo_override_wrap.setVisible(False)
        train_body.addWidget(self._algo_override_wrap)

        self._custom_cells_wrap = self._lazy_panel_host(
            "Custom cells editor",
            self._ensure_custom_cells_editor,
        )
        self._custom_cells_wrap.setVisible(False)
        train_body.addWidget(self._custom_cells_wrap)

        # Hyperparameter suite editor — Schedule / Optimizer / Regularization /
        # Augmentation / Reproducibility. Only YOLO detection scenarios expose
        # these; other backbones hide the panel.
        self._hp_panel_host = self._lazy_panel_host(
            "Hyperparameter suite",
            self._ensure_hp_panel,
        )
        self._hp_panel_host.setVisible(False)
        train_body.addWidget(self._hp_panel_host)
        self._hp_schema_cache: dict[str, Any] = {}

        name_row = QHBoxLayout()
        name_row.setSpacing(8)
        name_row.addWidget(QLabel("Final model name:"))
        self._final_model_name = QLineEdit()
        self._final_model_name.setPlaceholderText("Optional; used for saved/exported model files")
        name_row.addWidget(self._final_model_name, stretch=1)
        train_body.addLayout(name_row)

        self._auto_fresh_resume = QCheckBox("Start fresh if resume checkpoint is complete")
        self._auto_fresh_resume.setChecked(True)
        self._auto_fresh_resume.setToolTip(
            "If a completed YOLO checkpoint refuses to resume, automatically clear resume state and start a new run."
        )
        train_body.addWidget(self._auto_fresh_resume)

        tr_row = QHBoxLayout()
        tr_row.setSpacing(8)
        self._kick_btn = QPushButton("Start Training")
        self._kick_btn.clicked.connect(self._kick_training)
        self._kick_btn.setMinimumWidth(140)
        self._kick_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._update_btn = QPushButton("Update Model")
        self._update_btn.clicked.connect(self._kick_update_training)
        self._update_btn.setMinimumWidth(140)
        self._update_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._stop_btn = QPushButton("Stop Training")
        self._stop_btn.clicked.connect(self._stop_training)
        self._stop_btn.setMinimumWidth(140)
        self._stop_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._verify_btn = QPushButton("Mark Verified")
        self._verify_btn.clicked.connect(self._mark_verified)
        self._unverify_btn = QPushButton("Clear Verified")
        self._unverify_btn.clicked.connect(self._clear_verified)
        _ghost(self._update_btn)
        _ghost(self._stop_btn)
        _ghost(self._verify_btn)
        _ghost(self._unverify_btn)
        tr_row.addWidget(self._kick_btn)
        tr_row.addWidget(self._update_btn)
        tr_row.addWidget(self._stop_btn)
        tr_row.addStretch(1)
        tr_row.addWidget(self._verify_btn)
        tr_row.addWidget(self._unverify_btn)
        train_body.addLayout(tr_row)
        self._train_meta = QLabel("")
        self._train_meta.setWordWrap(True)
        train_body.addWidget(self._train_meta)

        self._train_console_host = self._lazy_panel_host(
            "Training console",
            self._ensure_train_console,
        )
        train_body.addWidget(self._train_console_host, stretch=1)

        for sec in (self._dataset_card, train_card):
            sec.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._detail_splitter.addWidget(sec)
            sec.expandedChanged.connect(lambda _on: self.refresh_responsive_layout())
        self._detail_cards: tuple[CollapsibleSection, ...] = (self._dataset_card, train_card)
        self._detail_splitter.setStretchFactor(0, 2)
        self._detail_splitter.setStretchFactor(1, 5)
        self._detail_splitter.splitterMoved.connect(lambda _pos, _idx: self.refresh_responsive_layout())
        workflow_layout.addWidget(self._detail_splitter, stretch=1)

        results_card = self._card("Results")
        self._ci_cd_host = self._lazy_panel_host(
            "CI/CD lifecycle",
            self._ensure_ci_cd_bar,
        )
        results_card.body_layout().addWidget(self._ci_cd_host)
        self._artifacts_host = self._lazy_panel_host(
            "Run artifacts",
            self._ensure_artifacts_panel,
        )
        results_card.body_layout().addWidget(self._artifacts_host, stretch=1)

        advanced_tab = QWidget()
        advanced_tab.setMinimumWidth(28)
        advanced_tab.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        advanced_layout = QVBoxLayout(advanced_tab)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(6)
        advanced_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._advanced_splitter = QSplitter(Qt.Orientation.Vertical)
        self._advanced_splitter.setChildrenCollapsible(False)
        self._advanced_splitter.setHandleWidth(4)

        history_card = self._card("Model Gallery")
        self._history_host = self._lazy_panel_host(
            "Model gallery",
            self._ensure_history_panel,
        )
        history_card.body_layout().addWidget(self._history_host, stretch=1)

        compare_card = self._card("Run Comparison")
        self._run_compare_host = self._lazy_panel_host(
            "Run comparison",
            self._ensure_run_compare_panel,
        )
        compare_card.body_layout().addWidget(self._run_compare_host, stretch=1)

        viz_card = self._card("Data Visualization")
        self._data_viz_host = self._lazy_panel_host(
            "Data visualization",
            self._ensure_data_viz_panel,
        )
        viz_card.body_layout().addWidget(self._data_viz_host, stretch=1)
        results_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        results_card.expandedChanged.connect(lambda _on: self.refresh_responsive_layout())

        for sec in (
            cards_card,
            history_card,
            compare_card,
            viz_card,
            results_card,
        ):
            sec.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self._advanced_splitter.addWidget(sec)
            sec.expandedChanged.connect(lambda _on: self.refresh_responsive_layout())
        self._advanced_cards: tuple[CollapsibleSection, ...] = (
            cards_card,
            history_card,
            compare_card,
            viz_card,
            results_card,
        )
        stretch_weights = (2, 2, 2, 2, 2)
        for idx, w in enumerate(stretch_weights):
            self._advanced_splitter.setStretchFactor(idx, w)
        self._advanced_splitter.splitterMoved.connect(lambda _pos, _idx: self.refresh_responsive_layout())
        advanced_layout.addWidget(self._advanced_splitter, stretch=1)

        self._detail_main_split.addWidget(workflow_tab)
        self._detail_main_split.addWidget(advanced_tab)
        self._detail_main_split.setStretchFactor(0, 7)
        self._detail_main_split.setStretchFactor(1, 5)
        self._detail_main_split.splitterMoved.connect(lambda _pos, _idx: self.refresh_responsive_layout())
        rl.addWidget(self._detail_main_split, stretch=1)

        # Scenario flow is rendered natively inside the Data Viz hub's "Flow"
        # tab (see DataVizHub / ScenarioFlowView); this just caches the last
        # entry so live updates can re-render without a fresh selection.
        self._flow_entry: dict[str, Any] = {}

        self._right_stack.addWidget(right)
        self._right_detail_index = self._right_stack.indexOf(right)
        self._right_empty_index = self._right_stack.indexOf(self._empty_view)
        self._right_status_index = self._right_stack.indexOf(self._catalog_status_view)
        self._right_stack.setCurrentIndex(self._right_empty_index)
        right_scroll.setWidget(self._right_stack)
        splitter.addWidget(right_scroll)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, stretch=1)

        self._status = QLabel("")
        layout.addWidget(self._status)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self.refresh_responsive_layout()

    @staticmethod
    def _card(title: str) -> CollapsibleSection:
        return CollapsibleSection(title, expanded=True)

    @staticmethod
    def _card_collapsed(title: str) -> CollapsibleSection:
        return CollapsibleSection(title, expanded=False)

    @staticmethod
    def _readiness_item(label: str, value: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("readinessItem")
        frame.setMinimumWidth(24)
        frame.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.setSpacing(1)
        title = QLabel(label)
        title.setObjectName("readinessLabel")
        val = QLabel(value)
        val.setObjectName("readinessValue")
        val.setWordWrap(True)
        frame._value_label = val  # type: ignore[attr-defined]
        layout.addWidget(title)
        layout.addWidget(val)
        return frame

    @staticmethod
    def _set_readiness_value(widget: QFrame, value: str, state: str = "idle") -> None:
        label = getattr(widget, "_value_label", None)
        if isinstance(label, QLabel):
            label.setText(str(value or "—"))
        widget.setProperty("state", state)
        repolish(widget)

    def _lazy_panel_host(self, title: str, builder: Callable[[], object]) -> QFrame:
        host = QFrame()
        host.setObjectName("lazyCatalogPanelHost")
        host.setMinimumHeight(56)
        host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(host)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        label = QLabel(f"{title} not loaded")
        label.setProperty("muted", True)
        label.setWordWrap(True)
        layout.addWidget(label)
        btn = QPushButton(f"Load {title}")
        _ghost(btn)
        btn.clicked.connect(lambda _checked=False: builder())
        layout.addWidget(btn, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        return host

    @staticmethod
    def _replace_lazy_host(host: QFrame, widget: QWidget) -> None:
        layout = host.layout()
        if not isinstance(layout, QVBoxLayout):
            return
        while layout.count():
            item = layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        widget.setParent(host)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(widget, stretch=1)

    def _ensure_guard_card(self) -> object:
        if self._guard_card is not None:
            return self._guard_card
        SystemGuardCard = _lazy_symbol(".system_guard_card", "SystemGuardCard")
        card = SystemGuardCard(show_title=False)
        card.guardProfileChanged.connect(self._set_guard_profile)
        card.deviceChanged.connect(self._on_guard_device_changed)
        card.storageRootChanged.connect(self._on_guard_storage_changed)
        card.refreshRequested.connect(self._poll_guard_for_current)
        self._guard_card = card
        self._replace_lazy_host(self._guard_host, card)
        if self._pending_guard_data is not None:
            try:
                card.apply_guard(self._pending_guard_data)
            except Exception:
                pass
        elif self.current_scenario():
            self._poll_guard_for_current()
        return card

    def _ensure_dataset_panels(self) -> object:
        if self._dataset_stack is not None:
            return self._dataset_stack
        DatasetPanel = _lazy_symbol(".dataset_panel", "DatasetPanel")
        AudioStudioPanel = _lazy_symbol(".audio_studio_panel", "AudioStudioPanel")
        dataset_panel = DatasetPanel(
            base_url=self._base_url,
            http_get=self._http_get,
            http_post=self._http_post,
            http_delete=self._http_delete,
        )
        dataset_panel.datasetChanged.connect(lambda name: self.scenarioMutated.emit(name))
        dataset_panel.errorRaised.connect(self.errorRaised.emit)
        audio_panel = AudioStudioPanel(
            base_url=self._base_url,
            http_get=self._http_get,
            http_post=self._http_post,
            http_delete=self._http_delete,
        )
        audio_panel.datasetChanged.connect(lambda name: self.scenarioMutated.emit(name))
        audio_panel.errorRaised.connect(self.errorRaised.emit)
        stack = QStackedWidget()
        stack.addWidget(dataset_panel)
        stack.addWidget(audio_panel)
        self._dataset_panel = dataset_panel
        self._audio_studio_panel = audio_panel
        self._dataset_stack = stack
        self._replace_lazy_host(self._dataset_host, stack)
        scenario, dataset_folder, backbone_type, backbone_config = self._pending_dataset_context
        self._route_dataset_panel(scenario, dataset_folder, backbone_type, backbone_config)
        return stack

    def _ensure_hp_panel(self) -> object:
        if self._hp_panel is not None:
            return self._hp_panel
        HyperparamSuitePanel = _lazy_symbol(".hyperparam_suite_panel", "HyperparamSuitePanel")
        panel = HyperparamSuitePanel()
        panel.savePressed.connect(self._on_hp_save_pressed)
        panel.resetPressed.connect(self._on_hp_reset_pressed)
        self._hp_panel = panel
        self._replace_lazy_host(self._hp_panel_host, panel)
        current = self.current_scenario()
        if current:
            self._load_hyperparams_for(current)
        return panel

    def _ensure_train_console(self) -> object:
        if self._train_console is not None:
            return self._train_console
        TrainingConsoleWidget = _lazy_symbol(".training_console", "TrainingConsoleWidget")
        console = TrainingConsoleWidget()
        console.stop_requested.connect(self._stop_training)
        self._train_console = console
        self._replace_lazy_host(self._train_console_host, console)
        current = self.current_scenario()
        if current:
            console.set_lines(self._console_buffers.get(current, []))
            entry = self._entries.get(current, {}) or {}
            console.set_training_active(str(entry.get("status") or "") == "training")
            self._render_training_live(current)
        return console

    def _ensure_ci_cd_bar(self) -> object:
        if self._ci_cd_bar is not None:
            return self._ci_cd_bar
        CiCdLifecycleBar = _lazy_symbol(".ci_cd_lifecycle_bar", "CiCdLifecycleBar")
        bar = CiCdLifecycleBar(http_get=self._http_get, http_post=self._http_post)
        bar.errorRaised.connect(self.errorRaised.emit)
        bar.changed.connect(lambda: self.scenarioMutated.emit(self.current_scenario()))
        self._ci_cd_bar = bar
        self._replace_lazy_host(self._ci_cd_host, bar)
        bar.set_scenario(self.current_scenario())
        return bar

    def _ensure_artifacts_panel(self) -> object:
        if self._artifacts_panel is not None:
            return self._artifacts_panel
        RunArtifactsPanel = _lazy_symbol(".run_artifacts_panel", "RunArtifactsPanel")
        panel = RunArtifactsPanel(
            base_url=self._base_url,
            http_get=self._http_get,
            http_get_text=self._http_get_text,
        )
        self._artifacts_panel = panel
        self._replace_lazy_host(self._artifacts_host, panel)
        current = self.current_scenario()
        job_id = str(self._latest_train_jobs.get(current, "") or "")
        if job_id:
            panel.load_job(job_id)
        return panel

    def _ensure_history_panel(self) -> object:
        if self._history_panel is not None:
            return self._history_panel
        ScenarioHistoryPanel = _lazy_symbol(".scenario_history_panel", "ScenarioHistoryPanel")
        panel = ScenarioHistoryPanel(http_get=self._http_get)
        panel.runSelected.connect(self._on_history_selected)
        self._history_panel = panel
        self._replace_lazy_host(self._history_host, panel)
        current = self.current_scenario()
        if current:
            panel.load_scenario(current)
        return panel

    def _ensure_run_compare_panel(self) -> object:
        if self._run_compare_panel is not None:
            return self._run_compare_panel
        RunComparePanel = _lazy_symbol(".run_compare_panel", "RunComparePanel")
        panel = RunComparePanel(http_get=self._http_get)
        self._run_compare_panel = panel
        self._replace_lazy_host(self._run_compare_host, panel)
        panel.set_scenario(self.current_scenario())
        return panel

    def _ensure_data_viz_panel(self) -> object:
        if self._data_viz_panel is not None:
            return self._data_viz_panel
        DataVizHub = _lazy_symbol(".data_viz_hub", "DataVizHub")
        panel = DataVizHub(http_get=self._http_get, http_post=self._http_post)
        panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._data_viz_panel = panel
        self._replace_lazy_host(self._data_viz_host, panel)
        entry = self._entries.get(self.current_scenario(), {}) or {}
        if entry:
            self._sync_data_viz_panel(entry)
            self._render_scenario_flow(entry)
        return panel

    def _ensure_custom_cells_editor(self) -> object:
        if self._custom_cells_editor is not None:
            return self._custom_cells_editor
        CustomCellsEditor = _lazy_symbol(".custom_cells_editor", "CustomCellsEditor")
        editor = CustomCellsEditor(
            http_get=self._http_get,
            http_put=self._http_put,
            http_post=self._http_post,
        )
        editor.errorRaised.connect(self.errorRaised.emit)
        editor.statusChanged.connect(lambda msg: self._status.setText(msg))
        editor.draftSaved.connect(self.scenarioMutated.emit)
        editor.scenarioMutated.connect(self.scenarioMutated.emit)
        editor.runDraftRequested.connect(lambda _scenario, _override: self._kick_training())
        self._custom_cells_editor = editor
        self._replace_lazy_host(self._custom_cells_wrap, editor)
        current = self.current_scenario()
        if current:
            editor.set_scenario(current, self._entries.get(current, {}) or {})
        return editor

    def _render_scenario_flow(self, entry: dict[str, Any]) -> None:
        self._flow_entry = dict(entry or {})
        if not entry:
            if self._data_viz_panel is not None:
                self._data_viz_panel.set_flow("", "", [])
            return

        name = str(entry.get("name") or self.current_scenario() or "").strip()
        btype = str(entry.get("backbone_type") or "yolo_detection").strip() or "yolo_detection"
        bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
        status = str(entry.get("status") or "empty").strip() or "empty"
        payload = self._training_payload(btype) or {}
        dataset_name = (
            str(entry.get("dataset") or "").strip()
            or str((bcfg or {}).get("dataset") or "").strip()
            or str((bcfg or {}).get("dataset_csv") or "").strip()
            or str((bcfg or {}).get("dataset_version_id") or "").strip()
            or "Unlinked"
        )
        try:
            dataset_count = int(entry.get("dataset_count") or 0)
        except Exception:
            dataset_count = 0
        dataset_label = f"{dataset_count} item" if dataset_count == 1 else f"{dataset_count} items"
        latest_run = entry.get("latest_run") if isinstance(entry.get("latest_run"), dict) else None
        guard = entry.get("training_guard") if isinstance(entry.get("training_guard"), dict) else {}
        points = self._training_points.get(name, [])
        latest_point = points[-1] if points else {}
        job_id = str(self._latest_train_jobs.get(name, "") or latest_point.get("job_id") or "").strip()

        model_state, model_lines = self._flow_model_lines(entry, btype, bcfg)
        guard_state = str((guard or {}).get("status") or "pending").strip() or "pending"
        guard_lines = [
            f"Profile: {str((guard or {}).get('profile') or 'balanced')}",
            f"Device: {str(payload.get('device') or getattr(self, '_train_device_override', '') or 'auto')}",
            f"Storage: {str(payload.get('training_assets_root') or getattr(self, '_train_storage_override', '') or 'overflow protocol')}",
        ]
        adjustments = (guard or {}).get("adjustments")
        if isinstance(adjustments, list):
            guard_lines.append(f"Adjustments: {len([a for a in adjustments if str(a or '').strip()])}")
        blocking = (guard or {}).get("blocking_reasons")
        if isinstance(blocking, list) and blocking:
            guard_lines.append(f"Blocking: {len(blocking)}")

        config_lines = [
            f"Final model: {str(payload.get('final_model_name') or self._final_model_name.text() or self._final_model_name.placeholderText()).replace('Optional; default ', '')}",
            f"Fresh completed resume: {'Yes' if payload.get('auto_fresh_on_completed_resume', True) else 'No'}",
        ]
        if payload.get("base_model_override"):
            config_lines.append(f"Base override: {Path(str(payload.get('base_model_override'))).name}")
        if payload.get("backbone_config_override"):
            config_lines.append("Backbone override: selected")

        live_state = "active" if status == "training" or job_id else "waiting"
        live_lines = [f"Job: {job_id or 'none'}"]
        if latest_point:
            live_lines.append(f"Event: {latest_point.get('event') or 'update'}")
            for key in ("epoch", "epochs", "progress", "map50", "accuracy", "loss"):
                value = latest_point.get(key)
                if value not in (None, ""):
                    live_lines.append(f"{key}: {value}")
        else:
            live_lines.append("Progress: no live events")

        result_lines = [
            f"History: {entry.get('history_count', 0)}",
            f"Verified: {'Yes' if entry.get('verified') else 'No'}",
            "Review panes: cards, gallery, compare, data viz, artifacts",
        ]
        if latest_run:
            result_lines.insert(0, f"Latest: {latest_run.get('final_model_name') or latest_run.get('version') or 'run'}")
            if latest_run.get("map50") not in (None, ""):
                result_lines.insert(1, f"mAP50: {latest_run.get('map50')}")
        ci_cd = entry.get("ci_cd") if isinstance(entry.get("ci_cd"), dict) else {}
        ci_enabled = bool(ci_cd.get("enabled"))
        ci_metric = str(ci_cd.get("metric") or "map50_95")
        ci_threshold = ci_cd.get("threshold", "")
        ci_promotion = str(ci_cd.get("promotion") or "manual")
        gate_state = "pending"
        gate_lines = [
            f"Policy: {'enabled' if ci_enabled else 'legacy'}",
            f"Metric: {ci_metric}",
            f"Threshold: {ci_threshold}",
        ]
        if latest_run:
            gate_lines.append("Report: generated after candidate training")
            gate_state = "waiting"
        else:
            gate_lines.append("Report: none yet")
        promote_state = "manual" if ci_promotion != "auto" else "auto"
        promote_lines = [
            f"Mode: {ci_promotion}",
            "Requires: passed gate",
            "Writes: model registry prod + active weights",
        ]
        production_lines = [
            f"Active weights: {'ready' if entry.get('weights_ready') else 'not ready'}",
            f"Verified: {'Yes' if entry.get('verified') else 'No'}",
        ]
        if latest_run:
            production_lines.insert(0, f"Latest run: {latest_run.get('version') or 'run'}")

        steps = [
            ("Scenario", status, [
                f"Name: {name or 'Unnamed'}",
                f"Type: {_backbone_table_label(btype)}",
                f"Display: {entry.get('display_name') or name or 'N/A'}",
            ], self._flow_tone_for_status(status)),
            ("Dataset Preflight", "ready" if dataset_count > 0 else "needs data", [
                f"Source: {dataset_name}",
                f"Count: {dataset_label}",
                "Contract: checked during candidate training",
            ], "ok" if dataset_count > 0 else "warning"),
            ("Model", model_state, model_lines, "ok" if model_state == "ready" else "warning"),
            ("System & Guard", guard_state, guard_lines, self._flow_tone_for_status(guard_state)),
            ("Train Candidate", live_state, config_lines + live_lines, "active" if live_state == "active" else "idle"),
            ("CI Gate", gate_state, gate_lines, "ok" if gate_state == "passed" else "idle"),
            ("Promote Candidate", promote_state, promote_lines, "active" if ci_enabled else "idle"),
            ("Production", "active" if entry.get("weights_ready") else "pending", production_lines + result_lines, "ok" if entry.get("weights_ready") else "idle"),
        ]
        if self._data_viz_panel is not None:
            self._data_viz_panel.set_flow(name, dataset_name, steps)

    def _flow_model_lines(
        self,
        entry: dict[str, Any],
        btype: str,
        bcfg: dict[str, Any],
    ) -> tuple[str, list[str]]:
        if btype == "yolo_detection":
            base = str(entry.get("base_model") or "").strip()
            resolved = str(entry.get("base_model_resolved") or "").strip()
            ready = bool(entry.get("base_model_exists")) or bool(resolved)
            lines = [f"Base: {Path(base).name or 'Unset'}"]
            if resolved:
                lines.append(f"Resolved: {Path(resolved).name}")
            lines.append(f"Weights ready: {'Yes' if entry.get('weights_ready') else 'No'}")
            return ("ready" if ready else "missing"), lines
        if btype in {"torch_tabular", "custom_code"}:
            cells = (bcfg or {}).get("train_cells") or (bcfg or {}).get("cells") or []
            cell_count = len(cells) if isinstance(cells, list) else 0
            return (
                "ready" if cell_count else "needs cells",
                [f"Cells: {cell_count}", f"Backbone: {_backbone_table_label(btype)}"],
            )
        if btype == "llm_fine_tuning":
            base = str((bcfg or {}).get("base_model") or entry.get("base_model") or "").strip()
            return ("ready" if base else "missing", [f"Base: {base or 'Unset'}", "Adapter flow: fine tune"])
        return ("ready", [f"Backbone: {_backbone_table_label(btype)}"])

    @staticmethod
    def _flow_tone_for_status(status: str) -> str:
        value = str(status or "").lower()
        if value in {"trained", "ready", "ok", "available", "prepared"}:
            return "ok"
        if value in {"training", "active", "adjusted"}:
            return "active"
        if value in {"error", "blocked"}:
            return "error"
        if value in {"dataset", "pending", "missing", "needs data", "needs cells"}:
            return "warning"
        return "idle"

    def _build_custom_cells_section(self) -> QWidget:
        self._custom_cells_editor = CustomCellsEditor(
            http_get=self._http_get,
            http_put=self._http_put,
            http_post=self._http_post,
        )
        self._custom_cells_editor.errorRaised.connect(self.errorRaised)
        self._custom_cells_editor.statusChanged.connect(
            lambda msg: self._status.setText(msg) if hasattr(self, "_status") else None
        )
        self._custom_cells_editor.draftSaved.connect(self.scenarioMutated.emit)
        self._custom_cells_editor.scenarioMutated.connect(self.scenarioMutated.emit)
        self._custom_cells_editor.runDraftRequested.connect(
            lambda _scenario, _override: self._kick_training()
        )
        self._custom_cells_editor.setVisible(False)
        return self._custom_cells_editor

        wrap = QWidget()
        outer = QVBoxLayout(wrap)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        header = QVBoxLayout()
        header.setSpacing(2)
        title = QLabel("Custom Cells")
        title.setStyleSheet("font-weight: 600; font-size: 12px;")
        header.addWidget(title)
        subtitle = QLabel(
            "Author backbone cells inline. Save Draft persists to "
            "mlops/custom_cells/; Run Draft saves and trains with the override."
        )
        subtitle.setProperty("muted", True)
        subtitle.setStyleSheet(f"font-size: 10px; color: {WB_TEXT_IRON};")
        subtitle.setWordWrap(True)
        header.addWidget(subtitle)
        outer.addLayout(header)

        editor_split = QSplitter(Qt.Orientation.Horizontal)
        editor_split.setChildrenCollapsible(False)
        editor_split.setHandleWidth(3)

        list_pane = QWidget()
        list_lay = QVBoxLayout(list_pane)
        list_lay.setContentsMargins(0, 0, 0, 0)
        list_lay.setSpacing(4)
        list_label = QLabel("Cells")
        list_label.setStyleSheet("font-weight: 600; font-size: 11px;")
        list_lay.addWidget(list_label)
        self._cc_list = QListWidget()
        self._cc_list.setMinimumWidth(140)
        self._cc_list.setMaximumWidth(220)
        self._cc_list.setMinimumHeight(160)
        self._cc_list.currentRowChanged.connect(self._cc_on_list_row_changed)
        list_lay.addWidget(self._cc_list, stretch=1)

        list_btn_row = QHBoxLayout()
        list_btn_row.setSpacing(4)
        self._cc_add_cell_btn = QPushButton("[+] Add")
        self._cc_add_cell_btn.clicked.connect(self._cc_add_cell)
        self._cc_rm_cell_btn = QPushButton("[-] Remove")
        self._cc_rm_cell_btn.clicked.connect(self._cc_remove_cell)
        for b in (self._cc_add_cell_btn, self._cc_rm_cell_btn):
            _ghost(b)
            list_btn_row.addWidget(b)
        list_btn_row.addStretch(1)
        list_lay.addLayout(list_btn_row)
        editor_split.addWidget(list_pane)

        editor_pane = QWidget()
        editor_lay = QVBoxLayout(editor_pane)
        editor_lay.setContentsMargins(0, 0, 0, 0)
        editor_lay.setSpacing(4)
        name_label = QLabel("Cell name")
        name_label.setStyleSheet("font-weight: 600; font-size: 11px;")
        editor_lay.addWidget(name_label)
        self._cc_cell_name = QLineEdit()
        self._cc_cell_name.setPlaceholderText("e.g. preprocess_v1")
        editor_lay.addWidget(self._cc_cell_name)
        code_label = QLabel("Cell code")
        code_label.setStyleSheet("font-weight: 600; font-size: 11px;")
        editor_lay.addWidget(code_label)
        self._cc_code = QPlainTextEdit()
        self._cc_code.setPlaceholderText("def run(ctx, prev):\n    ...")
        self._cc_code.setMinimumHeight(140)
        self._cc_code.setStyleSheet(
            f"QPlainTextEdit {{ font-family: {WB_FONT_MONO}; }}"
        )
        editor_lay.addWidget(self._cc_code, stretch=1)
        editor_split.addWidget(editor_pane)

        editor_split.setStretchFactor(0, 0)
        editor_split.setStretchFactor(1, 1)
        outer.addWidget(editor_split, stretch=1)

        sep1 = QFrame()
        sep1.setFrameShape(QFrame.Shape.HLine)
        sep1.setFrameShadow(QFrame.Shadow.Plain)
        sep1.setStyleSheet(f"color: {WB_LINE_LIGHT};")
        outer.addWidget(sep1)

        ds_label = QLabel("Scenario datasets")
        ds_label.setStyleSheet("font-weight: 600; font-size: 11px;")
        outer.addWidget(ds_label)
        ds_hint = QLabel("JSON array of dataset refs. Resolved at draft save.")
        ds_hint.setStyleSheet(f"font-size: 10px; color: {WB_TEXT_IRON};")
        ds_hint.setWordWrap(True)
        outer.addWidget(ds_hint)
        self._cc_scenario_ds = QTextEdit()
        self._cc_scenario_ds.setMaximumHeight(72)
        self._cc_scenario_ds.setPlaceholderText(
            '[{"name": "primary", "kind": "folder", "path": "database/foo"}]'
        )
        self._cc_scenario_ds.setStyleSheet(
            f"QTextEdit {{ font-family: {WB_FONT_MONO}; }}"
        )
        outer.addWidget(self._cc_scenario_ds)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Plain)
        sep2.setStyleSheet(f"color: {WB_LINE_LIGHT};")
        outer.addWidget(sep2)

        paste_label = QLabel("Paste as file")
        paste_label.setStyleSheet("font-weight: 600; font-size: 11px;")
        outer.addWidget(paste_label)
        paste_hint = QLabel("Attach inline CSV / JSON / text to the selected cell.")
        paste_hint.setStyleSheet(f"font-size: 10px; color: {WB_TEXT_IRON};")
        paste_hint.setWordWrap(True)
        outer.addWidget(paste_hint)

        paste_row = QHBoxLayout()
        paste_row.setSpacing(6)
        paste_row.addWidget(QLabel("Filename:"))
        self._cc_paste_name = QLineEdit()
        self._cc_paste_name.setPlaceholderText("e.g. notes.csv")
        paste_row.addWidget(self._cc_paste_name, stretch=1)
        self._cc_add_paste_btn = QPushButton("Attach to cell")
        self._cc_add_paste_btn.clicked.connect(self._cc_add_pasted_file_to_cell)
        _ghost(self._cc_add_paste_btn, min_width=120)
        paste_row.addWidget(self._cc_add_paste_btn)
        outer.addLayout(paste_row)

        self._cc_paste_body = QTextEdit()
        self._cc_paste_body.setMaximumHeight(64)
        self._cc_paste_body.setPlaceholderText(
            "Paste file contents here, then press Attach to cell."
        )
        self._cc_paste_body.setStyleSheet(
            f"QTextEdit {{ font-family: {WB_FONT_MONO}; }}"
        )
        outer.addWidget(self._cc_paste_body)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setFrameShadow(QFrame.Shadow.Plain)
        sep3.setStyleSheet(f"color: {WB_LINE_LIGHT};")
        outer.addWidget(sep3)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._cc_save_btn = QPushButton("Save Draft")
        self._cc_save_btn.clicked.connect(self._cc_save_draft)
        self._cc_save_btn.setMinimumWidth(110)
        self._cc_save_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self._cc_run_btn = QPushButton("Run Draft")
        self._cc_run_btn.clicked.connect(self._cc_run_draft)
        self._cc_run_btn.setMinimumWidth(110)
        self._cc_run_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        actions.addWidget(self._cc_save_btn)
        actions.addWidget(self._cc_run_btn)
        actions.addStretch(1)
        self._cc_promote_btn = QPushButton("Promote to Template")
        self._cc_promote_btn.clicked.connect(self._cc_promote_template)
        self._cc_apply_btn = QPushButton("Apply Template")
        self._cc_apply_btn.clicked.connect(self._cc_apply_template)
        for b in (self._cc_promote_btn, self._cc_apply_btn):
            _ghost(b, min_width=140)
            actions.addWidget(b)
        outer.addLayout(actions)

        wrap.setVisible(False)
        return wrap

    def _build_empty_view(self) -> QWidget:
        """Quickstart card shown when no scenario is selected."""
        container = QWidget()
        container.setObjectName("emptyStateContainer")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(40, 80, 40, 40)
        outer.setSpacing(0)
        outer.addStretch(1)

        card = QFrame()
        card.setObjectName("emptyStateCard")
        card.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        card.setMinimumWidth(380)
        card.setMaximumWidth(520)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 24, 28, 24)
        cl.setSpacing(10)

        title = QLabel("No scenario selected")
        title.setStyleSheet("font-size: 12px; font-weight: 600;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(title)

        msg = QLabel(
            "Pick a scenario from the list, or create a new one to start training.\n"
            "Use the filter above the list to narrow by name, status, or backbone."
        )
        msg.setProperty("muted", True)
        msg.setWordWrap(True)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(msg)

        cl.addSpacing(6)

        cta_row = QHBoxLayout()
        cta_row.setSpacing(8)
        cta_row.addStretch(1)
        primary = QPushButton("New Scenario")
        primary.clicked.connect(self._open_new_scenario_dialog)
        primary.setMinimumWidth(140)
        primary.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        cta_row.addWidget(primary)
        secondary = QPushButton("Refresh List")
        secondary.clicked.connect(lambda: self.scenarioMutated.emit(""))
        _ghost(secondary, min_width=120)
        cta_row.addWidget(secondary)
        cta_row.addStretch(1)
        cl.addLayout(cta_row)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addWidget(card)
        wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(2)
        return container

    def _build_catalog_status_view(self) -> QWidget:
        """Centered catalog status shown in the main body while startup loads or times out."""
        container = QWidget()
        container.setObjectName("emptyStateContainer")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(40, 80, 40, 40)
        outer.setSpacing(0)
        outer.addStretch(1)

        card = QFrame()
        card.setObjectName("emptyStateCard")
        card.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        card.setMinimumWidth(380)
        card.setMaximumWidth(540)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(28, 24, 28, 24)
        cl.setSpacing(10)

        self._catalog_status_title = QLabel("Catalog unavailable")
        self._catalog_status_title.setStyleSheet("font-size: 12px; font-weight: 600;")
        self._catalog_status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self._catalog_status_title)

        self._catalog_status_body = QLabel(
            "The scenario catalog did not load yet. Refresh to retry once the service is ready."
        )
        self._catalog_status_body.setProperty("muted", True)
        self._catalog_status_body.setWordWrap(True)
        self._catalog_status_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self._catalog_status_body)

        cl.addSpacing(6)

        cta_row = QHBoxLayout()
        cta_row.setSpacing(8)
        cta_row.addStretch(1)
        self._catalog_status_refresh_btn = QPushButton("Refresh Catalog")
        self._catalog_status_refresh_btn.clicked.connect(self._on_scenario_status_refresh)
        self._catalog_status_refresh_btn.setMinimumWidth(160)
        self._catalog_status_refresh_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        cta_row.addWidget(self._catalog_status_refresh_btn)
        cta_row.addStretch(1)
        cl.addLayout(cta_row)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addWidget(card)
        wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(2)
        return container

    def _build_left_catalog_status_page(self) -> QWidget:
        """Centered side-panel status shown over the full scenario catalog pane."""
        container = QWidget()
        container.setObjectName("scenarioStatusContainer")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(0)
        outer.addStretch(1)

        card = QFrame()
        card.setObjectName("emptyStateCard")
        card.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        card.setMinimumWidth(260)
        card.setMaximumWidth(420)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(24, 22, 24, 22)
        cl.setSpacing(8)

        self._left_catalog_status_title = QLabel("Scenarios unavailable")
        self._left_catalog_status_title.setStyleSheet("font-size: 12px; font-weight: 600;")
        self._left_catalog_status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self._left_catalog_status_title)

        self._left_catalog_status_body = QLabel(
            "Could not load scenarios. Refresh once the service finishes starting."
        )
        self._left_catalog_status_body.setProperty("muted", True)
        self._left_catalog_status_body.setWordWrap(True)
        self._left_catalog_status_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self._left_catalog_status_body)

        cl.addSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)
        self._left_catalog_status_refresh_btn = QPushButton("Refresh Catalog")
        self._left_catalog_status_refresh_btn.clicked.connect(self._on_scenario_status_refresh)
        self._left_catalog_status_refresh_btn.setMinimumWidth(150)
        self._left_catalog_status_refresh_btn.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        btn_row.addWidget(self._left_catalog_status_refresh_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addWidget(card)
        wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(1)
        return container

    def _build_scenario_status_page(self) -> QWidget:
        """Centered message + Refresh shown over the scenario list when scenarios
        are connecting, timed out, or failed to load."""
        container = QWidget()
        container.setObjectName("scenarioStatusContainer")
        outer = QVBoxLayout(container)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(0)
        outer.addStretch(1)

        card = QFrame()
        card.setObjectName("emptyStateCard")
        card.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        card.setMinimumWidth(260)
        card.setMaximumWidth(420)
        cl = QVBoxLayout(card)
        cl.setContentsMargins(24, 22, 24, 22)
        cl.setSpacing(8)

        self._scenario_status_title = QLabel("Scenarios unavailable")
        self._scenario_status_title.setStyleSheet("font-size: 12px; font-weight: 600;")
        self._scenario_status_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self._scenario_status_title)

        self._scenario_status_body = QLabel(
            "Could not load scenarios. The service may be offline or the request timed out."
        )
        self._scenario_status_body.setProperty("muted", True)
        self._scenario_status_body.setWordWrap(True)
        self._scenario_status_body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cl.addWidget(self._scenario_status_body)

        cl.addSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch(1)
        self._scenario_status_refresh_btn = QPushButton("Refresh")
        self._scenario_status_refresh_btn.clicked.connect(self._on_scenario_status_refresh)
        self._scenario_status_refresh_btn.setMinimumWidth(140)
        self._scenario_status_refresh_btn.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        btn_row.addWidget(self._scenario_status_refresh_btn)
        btn_row.addStretch(1)
        cl.addLayout(btn_row)

        wrap = QHBoxLayout()
        wrap.addStretch(1)
        wrap.addWidget(card)
        wrap.addStretch(1)
        outer.addLayout(wrap)
        outer.addStretch(1)
        return container

    def _request_scenarios_refresh(self) -> None:
        self.show_scenarios_connecting()
        self.scenarioMutated.emit("")

    def _on_scenario_status_refresh(self) -> None:
        # Show the connecting state immediately for feedback, then kick the refresh.
        self._request_scenarios_refresh()

    def _set_header_refresh_busy(self, busy: bool) -> None:
        btn = getattr(self, "_scenario_header_refresh_btn", None)
        if btn is not None:
            btn.setEnabled(not busy)

    def _show_left_catalog_status(self, title: str, body: str, *, refreshing: bool) -> None:
        self._left_catalog_status_title.setText(title)
        self._left_catalog_status_body.setText(body)
        self._left_catalog_status_refresh_btn.setEnabled(not refreshing)
        self._left_catalog_status_refresh_btn.setText(
            "Refreshing…" if refreshing else "Refresh Catalog"
        )
        if hasattr(self, "_list_stack") and not self._entries:
            self._list_stack.setCurrentWidget(self._list_status_page)

    def _show_catalog_status(self, title: str, body: str, *, refreshing: bool) -> None:
        self._catalog_status_title.setText(title)
        self._catalog_status_body.setText(body)
        self._catalog_status_refresh_btn.setEnabled(not refreshing)
        self._catalog_status_refresh_btn.setText("Refreshing…" if refreshing else "Refresh Catalog")
        if hasattr(self, "_right_stack") and not self._entries:
            self._right_stack.setCurrentIndex(self._right_status_index)

    def show_scenarios_connecting(self) -> None:
        """Show a centered 'connecting and refreshing' message over the list."""
        self._set_header_refresh_busy(True)
        self._scenario_status_title.setText("Connecting…")
        self._scenario_status_body.setText("Connecting to the service and refreshing scenarios…")
        self._scenario_status_refresh_btn.setEnabled(False)
        self._scenario_status_refresh_btn.setText("Refreshing…")
        if hasattr(self, "_scenario_stack"):
            self._scenario_stack.setCurrentWidget(self._scenario_status_page)
        self._show_left_catalog_status(
            "Refreshing catalog…",
            "Connecting to the service and refreshing the scenario catalog…",
            refreshing=True,
        )
        self._show_catalog_status(
            "Refreshing catalog…",
            "Connecting to the service and refreshing the scenario catalog in the background…",
            refreshing=True,
        )

    def show_scenarios_error(self, message: str = "") -> None:
        """Show a centered failure/timeout message with a Refresh button."""
        resolved = str(message or "").strip() or (
            "Could not load scenarios. The service may be offline or the request timed out."
        )
        self._scenario_status_title.setText("Scenarios unavailable")
        self._scenario_status_body.setText(resolved)
        self._scenario_status_refresh_btn.setEnabled(True)
        self._scenario_status_refresh_btn.setText("Refresh")
        self._set_header_refresh_busy(False)
        if hasattr(self, "_scenario_stack"):
            self._scenario_stack.setCurrentWidget(self._scenario_status_page)
        self._show_left_catalog_status("Scenario catalog timed out", resolved, refreshing=False)
        self._show_catalog_status("Scenario catalog timed out", resolved, refreshing=False)

    def _show_scenario_table(self) -> None:
        """Restore the scenario table view (used once a load succeeds)."""
        self._scenario_status_refresh_btn.setEnabled(True)
        self._scenario_status_refresh_btn.setText("Refresh")
        self._set_header_refresh_busy(False)
        if hasattr(self, "_scenario_stack"):
            self._scenario_stack.setCurrentWidget(self._scenario_table)
        if hasattr(self, "_list_stack"):
            self._list_stack.setCurrentWidget(self._list_page)

    def _show_empty_state(self) -> None:
        if hasattr(self, "_right_stack"):
            self._right_stack.setCurrentIndex(self._right_empty_index)

    def _show_detail_view(self) -> None:
        if hasattr(self, "_right_stack"):
            self._right_stack.setCurrentIndex(self._right_detail_index)

    def refresh_responsive_layout(self) -> None:
        """Debounced: splitter moves and collapsible toggles can fire nested layout passes."""
        self._responsive_timer.start()

    def _run_responsive_refresh(self) -> None:
        self._compact_splitter_when_all_collapsed(self._detail_splitter, self._detail_cards)
        self._compact_splitter_when_all_collapsed(self._advanced_splitter, self._advanced_cards)
        if self._dataset_panel is not None:
            self._dataset_panel.refresh_responsive_layout()
        if self._audio_studio_panel is not None:
            self._audio_studio_panel.refresh_responsive_layout()

    def _compact_splitter_when_all_collapsed(
        self,
        splitter: QSplitter,
        sections: tuple[CollapsibleSection, ...],
    ) -> None:
        """When all cards are collapsed, shrink the stack so headers hug the top."""
        if not sections:
            return
        all_collapsed = all(not sec.is_expanded() for sec in sections)
        if all_collapsed:
            header_heights = [max(1, int(sec.minimumSizeHint().height())) for sec in sections]
            handles_h = max(0, int(splitter.handleWidth()) * max(0, splitter.count() - 1))
            h = int(max(1, sum(header_heights) + handles_h + 2))
            splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            splitter.setMinimumHeight(h)
            splitter.setMaximumHeight(h)
            # Ensure deterministic top-stack ordering when compressed.
            sizes = list(header_heights)
            if len(sizes) == splitter.count():
                splitter.setSizes(sizes)
            return
        splitter.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        splitter.setMinimumHeight(0)
        splitter.setMaximumHeight(16_777_215)
        if self._artifacts_panel is not None:
            self._artifacts_panel.refresh_responsive_layout()

    # ---------- external API ----------

    def current_scenario(self) -> str:
        r = self._scenario_table.currentRow()
        if r < 0:
            return ""
        it = self._scenario_table.item(r, 0)
        if it is None:
            return ""
        return str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "").strip()

    def _scenario_table_find_row(self, name: str) -> int:
        want = str(name or "").strip()
        if not want:
            return -1
        for r in range(self._scenario_table.rowCount()):
            it = self._scenario_table.item(r, 0)
            if it is not None and str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "") == want:
                return r
        return -1

    def _scenario_table_fill_row(self, row: int, entry: dict[str, Any]) -> None:
        name = str(entry.get("name") or "")
        bt = str(entry.get("backbone_type") or "yolo_detection")
        status = str(entry.get("status") or "empty").upper()
        name_item = QTableWidgetItem(name)
        name_item.setData(Qt.ItemDataRole.UserRole, name)
        non_edit = Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
        name_item.setFlags(non_edit)
        type_item = QTableWidgetItem(_backbone_table_label(bt))
        type_item.setFlags(non_edit)
        stat_item = QTableWidgetItem(status)
        stat_item.setFlags(non_edit)
        self._scenario_table.setItem(row, 0, name_item)
        self._scenario_table.setItem(row, 1, type_item)
        self._scenario_table.setItem(row, 2, stat_item)

    def select_scenario(self, name: str) -> bool:
        """Programmatically select a scenario by name. Returns True if found."""
        target = str(name or "").strip()
        if not target:
            return False
        if target in self._entries:
            self._select_by_name(target)
            return True
        # If scenarios haven't loaded yet (or this one is not in the cache), defer
        # the selection until the next apply_scenarios call.
        self._pending_select_scenario = target
        return False

    def apply_scenarios(self, scenarios: list[dict[str, Any]]) -> None:
        # A successful load always restores the table (even if the list is empty,
        # which is a distinct, non-error "no scenarios configured" state).
        self._show_scenario_table()
        prev = self.current_scenario()
        self._entries = {str(s.get("name") or ""): s for s in scenarios if s.get("name")}
        new_names = set(self._entries.keys())
        desired_order: list[str] = [str(s.get("name") or "") for s in scenarios if s.get("name")]

        self._scenario_table.blockSignals(True)
        self._scenario_table.setRowCount(0)
        for name in desired_order:
            entry = self._entries[name]
            r = self._scenario_table.rowCount()
            self._scenario_table.insertRow(r)
            self._scenario_table_fill_row(r, entry)
        self._scenario_table.blockSignals(False)

        desired = ""
        if self._pending_select_scenario and self._pending_select_scenario in new_names:
            desired = self._pending_select_scenario
            self._pending_select_scenario = ""
        elif prev and prev in new_names:
            desired = prev
        elif desired_order:
            desired = desired_order[0]
        else:
            self._title.setText("No scenarios")
            self._detail_pill.set_status("empty")
            self._meta.setText("Configure scenarios in mlops/registry.json.")
            self._route_dataset_panel("", "", "", {})
            if self._history_panel is not None:
                self._history_panel.clear()
            if self._artifacts_panel is not None:
                self._artifacts_panel.clear()
            if self._run_compare_panel is not None:
                self._run_compare_panel.set_scenario("")
            self._cards_browser.clear()
            self._pending_guard_data = None
            if self._guard_card is not None:
                self._guard_card.clear()
            self._update_readiness_strip({})
            self._render_scenario_flow({})
            if self._ci_cd_bar is not None:
                self._ci_cd_bar.set_scenario("")
            self._show_empty_state()
            self._apply_scenario_filter(self._filter_edit.text())
            return
        if desired:
            self._select_by_name(desired)
            self._render_detail(self._entries.get(desired, {}) or {})
        self._apply_scenario_filter(self._filter_edit.text())

    def apply_scenario_update(self, scenario: str, status_payload: dict[str, Any]) -> None:
        if not scenario:
            return
        existing = self._entries.get(scenario, {}) or {}
        existing.update(status_payload)
        self._entries[scenario] = existing
        row = self._scenario_table_find_row(scenario)
        if row >= 0:
            self._scenario_table_fill_row(row, existing)
        if self.current_scenario() == scenario:
            self._render_detail(existing)

    def _open_new_scenario_dialog(self) -> None:
        # Ensure model list is available for the dialog.
        self._load_models(force=False)
        NewScenarioDialog = _lazy_symbol(".new_scenario_dialog", "NewScenarioDialog")
        dlg = NewScenarioDialog(
            http_get=self._http_get,
            http_post=self._http_post,
            models=self._models,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        created = dlg.created_scenario_name()
        if not created:
            return
        self._pending_select_scenario = created
        self._status.setText(f"Scenario created: {created}")
        self.scenarioMutated.emit(created)

    def _on_guard_device_changed(self, device: str) -> None:
        self._train_device_override = str(device or "").strip()
        if self._train_device_override:
            self._status.setText(f"Training device pinned to: {self._train_device_override}")
        else:
            self._status.setText("Training device set to auto (system default)")
        self._render_scenario_flow(self._entries.get(self.current_scenario(), {}) or self._flow_entry)

    def _on_guard_storage_changed(self, path: str) -> None:
        self._train_storage_override = str(path or "").strip()
        if self._train_storage_override:
            self._status.setText(f"Training save root pinned to: {self._train_storage_override}")
        else:
            self._status.setText("Training save root set to auto (overflow protocol)")
        self._render_scenario_flow(self._entries.get(self.current_scenario(), {}) or self._flow_entry)

    def _poll_guard_for_current(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        self._fetch_guard_bg(name)

    def _fetch_guard_bg(self, scenario: str) -> None:
        scenario = str(scenario or "").strip()
        if not scenario:
            return
        if self._guard_fetcher is not None:
            try:
                if self._guard_fetcher.isRunning() and self._guard_fetch_scenario == scenario:
                    return
            except Exception:
                pass
            try:
                if not self._guard_fetcher.isRunning():
                    self._guard_fetcher.finished.disconnect(self._on_guard_fetched)  # type: ignore[attr-defined]
            except Exception:
                pass
        self._guard_fetch_scenario = scenario
        fetcher = _GuardFetcher(self._base_url, scenario, parent=self)
        fetcher.finished.connect(self._on_guard_fetched)  # type: ignore[attr-defined]
        self._guard_fetcher = fetcher
        fetcher.start()

    def _on_guard_fetched(self, scenario: str, guard_data: dict[str, Any]) -> None:
        if scenario != self.current_scenario():
            return
        self._pending_guard_data = dict(guard_data or {})
        if self._guard_card is not None:
            self._guard_card.apply_guard(guard_data)
        entry = self._entries.get(scenario, {}) or {}
        entry["training_guard"] = guard_data
        self._entries[scenario] = entry
        self._render_scenario_flow(entry)

    def _set_guard_profile(self, profile: str) -> None:
        name = self.current_scenario()
        if not name:
            return
        try:
            status = self._http_post(
                f"/scenarios/{name}/guard_profile",
                {"profile": str(profile or "")},
            )
        except Exception as exc:
            msg = f"Set guard profile failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        if isinstance(status, dict):
            self.apply_scenario_update(name, status)
        self._status.setText(f"{name} guard profile set to {profile}")
        self.scenarioMutated.emit(name)

    def show_storage_diagnosis(self, text: str) -> None:
        if self._guard_card is not None:
            self._guard_card.set_storage_diagnosis(text)

    def apply_training_progress(self, payload: dict[str, Any]) -> None:
        scenario = str(payload.get("scenario") or "").strip()
        if not scenario:
            return
        event_type = str(payload.get("event") or "epoch")
        if event_type in {"log", "log_batch"}:
            if event_type == "log_batch" and isinstance(payload.get("lines"), list):
                rows = [
                    (
                        str(item.get("line") or ""),
                        str(item.get("stream") or "stdout"),
                    )
                    for item in payload.get("lines") or []
                    if isinstance(item, dict)
                ]
            else:
                rows = [(str(payload.get("line") or ""), str(payload.get("stream") or "stdout"))]
            buf = self._console_buffers.setdefault(scenario, [])
            buf.extend(rows)
            if len(buf) > 2000:
                del buf[:-2000]
            if self.current_scenario() == scenario and self._train_console is not None:
                for line, stream in rows:
                    self._train_console.append_line(line, stream)
            job_id = str(payload.get("job_id") or "")
            if job_id:
                self._latest_train_jobs[scenario] = job_id
                if self.current_scenario() == scenario and hasattr(self, "_stop_btn"):
                    self._stop_btn.setEnabled(True)
                    self._render_scenario_flow(self._entries.get(scenario, {}) or self._flow_entry)
            return
        series = self._training_points.setdefault(scenario, [])
        point = dict(payload)
        if event_type == "start":
            series.clear()
        elif event_type == "epoch":
            epoch = payload.get("epoch")
            replaced = False
            for i, existing in enumerate(series):
                if existing.get("event") == "epoch" and existing.get("epoch") == epoch:
                    series[i] = point
                    replaced = True
                    break
            if not replaced:
                series.append(point)
        else:
            series.append(point)
        if len(series) > 600:
            del series[:-600]
        if self.current_scenario() == scenario:
            if str(payload.get("job_id") or "").strip() and hasattr(self, "_stop_btn"):
                self._stop_btn.setEnabled(True)
            self._render_training_live(scenario)
            self._render_scenario_flow(self._entries.get(scenario, {}) or self._flow_entry)

    def apply_tabular_cell_progress(self, payload: dict[str, Any]) -> None:
        """Mirror cell_progress training output into the Training console.

        YOLO detection training emits `training_progress` log events (handled by
        apply_training_progress). Non-YOLO training runs in the backbone cell
        executor and emits `cell_progress` events, which are mirrored here.
        """
        if str(payload.get("job_type") or "") != "train":
            return
        scenario = str(payload.get("scenario") or "").strip()
        if not scenario:
            return
        entry = self._entries.get(scenario, {}) or {}
        backbone_type = str(entry.get("backbone_type") or "")
        if backbone_type not in {"torch_tabular", "custom_code", "face_recognition", "audio_recognition", "llm_fine_tuning"}:
            return

        job_id = str(payload.get("job_id") or "")
        if job_id:
            self._latest_train_jobs[scenario] = job_id

        cell_name = str(payload.get("cell_name") or "").strip() or f"Cell {payload.get('cell_index', '')}"
        status = str(payload.get("cell_status") or "").strip() or "done"
        output = str(payload.get("output") or "").rstrip()

        buf = self._console_buffers.setdefault(scenario, [])

        # Announce cell state transitions.
        if status == "running":
            line = f"[cell] running: {cell_name}"
            buf.append((line, "stdout"))
            if self.current_scenario() == scenario and self._train_console is not None:
                self._train_console.append_line(line, "stdout")
            return

        if output:
            for ln in output.splitlines():
                line = f"[{cell_name}] {ln}"
                buf.append((line, "stdout"))
                if self.current_scenario() == scenario and self._train_console is not None:
                    self._train_console.append_line(line, "stdout")
                if backbone_type in ("torch_tabular", "custom_code"):
                    self._maybe_parse_tabular_epoch_line(scenario, ln)

        tail = f"[cell] {status}: {cell_name}"
        buf.append((tail, "stderr" if status == "error" else "stdout"))
        if self.current_scenario() == scenario and self._train_console is not None:
            self._train_console.append_line(tail, "stderr" if status == "error" else "stdout")

        if len(buf) > 2000:
            del buf[:-2000]

    _TAB_EPOCH_RE = re.compile(
        r"\\[epoch\\s+(?P<epoch>\\d+)\\s*/\\s*(?P<epochs>\\d+)\\]\\s+"
        r"train_loss=(?P<train_loss>[-+eE0-9\\.]+)\\s+"
        r"val_loss=(?P<val_loss>[-+eE0-9\\.]+)\\s+"
        r"val_(?P<metric_name>mae|acc)=(?P<metric_val>[-+eE0-9\\.]+)"
    )

    def _maybe_parse_tabular_epoch_line(self, scenario: str, line: str) -> None:
        m = self._TAB_EPOCH_RE.search(str(line or ""))
        if not m:
            return
        try:
            epoch = int(m.group("epoch"))
            epochs = int(m.group("epochs"))
        except Exception:
            return
        try:
            train_loss = float(m.group("train_loss"))
        except Exception:
            train_loss = None  # type: ignore[assignment]
        try:
            val_loss = float(m.group("val_loss"))
        except Exception:
            val_loss = None  # type: ignore[assignment]
        metric_name = str(m.group("metric_name") or "")
        try:
            metric_val = float(m.group("metric_val"))
        except Exception:
            metric_val = None  # type: ignore[assignment]

        progress = 0.0
        try:
            progress = (float(epoch) / float(max(1, epochs))) * 100.0
        except Exception:
            progress = 0.0

        point: dict[str, Any] = {
            "event": "epoch",
            "epoch": epoch,
            "epochs": epochs,
            "progress": progress,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if metric_name == "acc":
            point["val_acc"] = metric_val
        else:
            point["val_mae"] = metric_val

        series = self._training_points.setdefault(scenario, [])
        replaced = False
        for i, existing in enumerate(series):
            if existing.get("event") == "epoch" and existing.get("epoch") == epoch:
                series[i] = point
                replaced = True
                break
        if not replaced:
            series.append(point)
        if len(series) > 600:
            del series[:-600]
        if self.current_scenario() == scenario:
            self._render_training_live(scenario)

    def notify_training_job(self, scenario: str, job_id: str) -> None:
        scenario = str(scenario or "").strip()
        job_id = str(job_id or "").strip()
        if not scenario or not job_id:
            return
        self._latest_train_jobs[scenario] = job_id
        if self.current_scenario() == scenario:
            if self._artifacts_panel is not None:
                self._artifacts_panel.load_job(job_id)
            if self._history_panel is not None:
                self._history_panel.load_scenario(scenario)
            try:
                if hasattr(self, "_detail_main_split"):
                    self._detail_main_split.setSizes([760, 520])
            except Exception:
                pass

    def is_ready(self, scenario: str) -> bool:
        entry = self._entries.get(scenario)
        if not entry:
            return False
        return str(entry.get("status") or "") == "ready"

    def list_widget(self) -> QWidget:
        return self._list_widget

    def detail_widget(self) -> QWidget:
        return self._detail_widget

    # ---------- internal ----------

    def _select_by_name(self, name: str) -> None:
        r = self._scenario_table_find_row(name)
        if r >= 0:
            self._scenario_table.selectRow(r)

    def _set_op_slot(self, slot: str) -> None:
        want = str(slot or "all").strip().lower()
        if want not in {s for s, _label, _types in _OP_SLOTS}:
            want = "all"
        self._op_slot = want
        btn = self._op_buttons.get(want)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)
        self._apply_scenario_filter(self._filter_edit.text())

    def _apply_scenario_filter(self, query: str = "") -> None:
        if not hasattr(self, "_filter_edit"):
            return
        q = query if query is not None else self._filter_edit.text()
        q = str(q or "").strip().lower()
        total = self._scenario_table.rowCount()
        visible = 0
        for i in range(total):
            it = self._scenario_table.item(i, 0)
            if it is None:
                continue
            name = str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "")
            entry = self._entries.get(name, {}) or {}
            match = self._scenario_in_slot(entry, self._op_slot) and self._scenario_matches(entry, name, q)
            self._scenario_table.setRowHidden(i, not match)
            if match:
                visible += 1
        self._refresh_op_slot_counts()
        if not q:
            if self._op_slot == "all":
                self._filter_status.setVisible(False)
            else:
                label = self._op_slot_label(self._op_slot)
                self._filter_status.setText(f"{visible} {label.lower()} scenario{'s' if visible != 1 else ''}")
                self._filter_status.setVisible(True)
        else:
            self._filter_status.setText(f"{visible} of {total} match")
            self._filter_status.setVisible(True)

    def _refresh_op_slot_counts(self) -> None:
        for slot, label, _types in _OP_SLOTS:
            btn = self._op_buttons.get(slot)
            if btn is None:
                continue
            count = sum(1 for entry in self._entries.values() if self._scenario_in_slot(entry, slot))
            btn.setText(f"{label} {count}")

    @staticmethod
    def _op_slot_label(slot: str) -> str:
        for key, label, _types in _OP_SLOTS:
            if key == slot:
                return label
        return "All"

    @staticmethod
    def _scenario_in_slot(entry: dict[str, Any], slot: str) -> bool:
        key = str(slot or "all").strip().lower()
        if key == "all":
            return True
        backbone = str(entry.get("backbone_type") or "yolo_detection").strip().lower()
        for slot_key, _label, types in _OP_SLOTS:
            if slot_key == key:
                return backbone in types
        return True

    @staticmethod
    def _scenario_matches(entry: dict, name: str, q: str) -> bool:
        if not q:
            return True
        name_l = name.lower()
        disp_l = str(entry.get("display_name") or "").lower()
        status_l = str(entry.get("status") or "").lower()
        backbone_l = str(entry.get("backbone_type") or "").lower()
        dataset_l = str(entry.get("dataset") or "").lower()
        # Field-prefixed filter: "status:trained", "type:yolo", "ds:coco".
        if ":" in q:
            field, _, value = q.partition(":")
            field = field.strip()
            value = value.strip()
            if not value:
                return True
            if field in ("status", "s"):
                return value in status_l
            if field in ("type", "backbone", "t"):
                return value in backbone_l
            if field in ("dataset", "ds"):
                return value in dataset_l
            if field in ("name", "n"):
                return value in name_l or value in disp_l
            # Unknown field → fall through to substring match below.
        return (
            q in name_l
            or q in disp_l
            or q in status_l
            or q in backbone_l
            or q in dataset_l
        )

    def _route_dataset_panel(
        self,
        scenario: str,
        dataset_folder: str,
        backbone_type: str,
        backbone_config: dict[str, Any],
    ) -> None:
        """Pick the right panel for the scenario's backbone, hand off scenario state.

        Audio scenarios get the dedicated AudioStudioPanel (waveform editor +
        multi-region drafts + dataset ledger). Everything else falls back to the
        generic DatasetPanel.
        """
        bt = str(backbone_type or "").strip().lower()
        self._pending_dataset_context = (
            str(scenario or ""),
            str(dataset_folder or ""),
            bt,
            dict(backbone_config or {}),
        )
        if self._dataset_stack is None:
            self._dataset_card.set_title(
                "Audio Dataset Readiness" if bt == "audio_recognition" else "Dataset Readiness"
            )
            return
        if bt == "audio_recognition":
            target = self._audio_studio_panel
            self._dataset_card.set_title("Audio Dataset Readiness")
        else:
            target = self._dataset_panel
            self._dataset_card.set_title("Dataset Readiness")
        if target is None:
            return
        self._dataset_stack.setCurrentWidget(target)
        target.set_scenario(scenario, dataset_folder, bt, backbone_config or {})

    def _on_selection(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        entry = self._entries.get(name) or {}
        if str(entry.get("backbone_type") or "") != "custom_code":
            self._cc_loaded_scenario = ""
        self._render_detail(entry)
        self._route_dataset_panel(
            name,
            str(entry.get("dataset") or ""),
            str(entry.get("backbone_type") or ""),
            entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {},
        )
        # Swap the training console to this scenario's buffered output.
        if self._train_console is not None:
            self._train_console.set_lines(self._console_buffers.get(name, []))
        self.scenarioSelected.emit(name)
        self.emit_entity_selected("scenario", name)
        self._poll_guard_for_current()

    def _refresh_scenario_cards(self, scenario: str) -> None:
        scenario = str(scenario or "").strip()
        if not scenario:
            self._cards_browser.clear()
            return
        try:
            payload = self._http_get(f"/scenarios/{scenario}/cards")
        except Exception as exc:
            self._cards_browser.setMarkdown(f"_Unable to load cards: {exc}_")
            return
        if not isinstance(payload, dict):
            self._cards_browser.setMarkdown("_Invalid cards response._")
            return
        model_md = str(payload.get("model_card") or "").strip()
        ds_md = str(payload.get("dataset_card") or "").strip()
        combined = "\n\n---\n\n".join(part for part in (model_md, ds_md) if part)
        try:
            self._cards_browser.setMarkdown(combined or "_No card content._")
        except Exception as exc:
            try:
                self._cards_browser.setPlainText(combined or f"[Markdown render error: {exc}]")
            except Exception:
                pass

    def _render_detail(self, entry: dict[str, Any]) -> None:
        name = str(entry.get("name", ""))
        if name:
            self._show_detail_view()
        else:
            self._show_empty_state()
        self._title.setText(name or "—")
        status = str(entry.get("status") or "empty")
        self._detail_pill.set_status(status)
        self._update_readiness_strip(entry)
        btype = str(entry.get("backbone_type") or "yolo_detection")
        self._algo_override_wrap.setVisible(btype == "torch_tabular")
        self._custom_cells_wrap.setVisible(btype == "custom_code")
        try:
            self._model_controls_wrap.setVisible(btype == "yolo_detection")
        except Exception:
            pass

        lines: list[str] = []
        disp = entry.get("display_name")
        desc = entry.get("description")
        if disp:
            lines.append(str(disp))
        if desc:
            lines.append(str(desc))
        if btype == "torch_tabular":
            bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
            dataset_csv = str((bcfg or {}).get("dataset_csv") or "")
            if dataset_csv:
                lines.append(f"dataset_csv: {dataset_csv}")
            cells = (bcfg or {}).get("train_cells") or (bcfg or {}).get("cells") or []
            if isinstance(cells, list) and cells:
                lines.append(f"cells: {len(cells)}")
            else:
                lines.append("cells: (not configured)")
        elif btype == "custom_code":
            bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
            cells = (bcfg or {}).get("train_cells") or (bcfg or {}).get("cells") or []
            if isinstance(cells, list) and cells:
                lines.append(f"template cells: {len(cells)}")
            else:
                lines.append("template cells: (use draft editor or promote)")
            if self._custom_cells_editor is not None:
                self._custom_cells_editor.set_scenario(name, entry)
        elif btype == "audio_recognition":
            lines.append(
                f"dataset: {entry.get('dataset_count', 0)} audio clips   |   weights_ready: {entry.get('weights_ready')}   |   verified: {entry.get('verified')}"
            )
        elif btype == "llm_fine_tuning":
            bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
            lines.append(f"base_model: {bcfg.get('base_model') or entry.get('base_model', '')}")
            lines.append(f"ollama_base: {bcfg.get('ollama_base_model') or bcfg.get('base_model') or ''}")
            lines.append(
                f"dataset: {entry.get('dataset_count', 0)} examples   |   adapter_ready: {entry.get('weights_ready')}"
            )
        else:
            lines.append(f"base_model: {entry.get('base_model', '')}")
            if entry.get("base_model_resolved"):
                lines.append(f"resolved_model: {entry.get('base_model_resolved')}")
            lines.append(
                f"dataset: {entry.get('dataset_count', 0)}   |   weights_ready: {entry.get('weights_ready')}   |   verified: {entry.get('verified')}"
            )

        # Apply guard for both CV and ML modes — training_guard carries system
        # specs and hyperparam adjustments for all backbone types.
        guard = entry.get("training_guard") if isinstance(entry.get("training_guard"), dict) else None
        self._pending_guard_data = dict(guard) if isinstance(guard, dict) else None
        if self._guard_card is not None:
            self._guard_card.apply_guard(guard)
        if guard is None and name:
            self._fetch_guard_bg(name)
        err = str(entry.get("error") or "")
        if err:
            lines.append(f"error: {err}")
        self._meta.setText("\n".join(lines))

        run = entry.get("latest_run")
        if isinstance(run, dict):
            final_name = str(run.get("final_model_name") or "").strip()
            final_part = f"  |  model: {final_name}" if final_name else ""
            duration = format_duration_seconds(run.get("training_duration_seconds"), empty="")
            duration_part = f"  |  duration: {duration}" if duration else ""
            self._train_meta.setText(
                f"Latest run: {run.get('version')}{final_part}  |  map50: {run.get('map50')}  |  trained_at: {format_datetime_text(run.get('trained_at'), seconds=True)}{duration_part}  |  history: {entry.get('history_count', 0)}"
            )
        else:
            self._train_meta.setText(f"No training runs yet.  |  history: {entry.get('history_count', 0)}")
        if not str(self._final_model_name.text() or "").strip():
            default_final = str((run or {}).get("final_model_name") or entry.get("display_name") or name)
            self._final_model_name.setPlaceholderText(f"Optional; default {default_final}")

        can_train = status not in ("training",)
        has_existing_model = bool(run) or bool(entry.get("weights_ready"))
        if btype == "archival_ingestion":
            self._kick_btn.setText("Run Pipeline")
            self._update_btn.setText("Reconcile")
        else:
            self._kick_btn.setText("Start Training")
            self._update_btn.setText("Update Model")
        self._kick_btn.setEnabled(can_train)
        self._update_btn.setVisible(has_existing_model)
        self._update_btn.setEnabled(can_train and has_existing_model)
        active_job_id = str(self._latest_train_jobs.get(name, "") or "")
        is_training = status == "training" and bool(active_job_id)
        self._stop_btn.setEnabled(is_training)
        if self._train_console is not None:
            self._train_console.set_training_active(is_training)
        # Verified/weights readiness are YOLO concepts; hide for tabular.
        self._verify_btn.setVisible(btype == "yolo_detection")
        self._unverify_btn.setVisible(btype == "yolo_detection")
        self._verify_btn.setEnabled(bool(entry.get("weights_ready")) and not bool(entry.get("verified")))
        self._unverify_btn.setEnabled(bool(entry.get("verified")))
        self._set_model_btn.setEnabled(btype == "yolo_detection" and bool(name) and self._model_combo.count() > 0)
        self._sync_model_combo(str(entry.get("base_model") or ""))
        # Hyperparameter suite: YOLO detection only (the schema targets Ultralytics).
        if btype == "yolo_detection":
            self._hp_panel_host.setVisible(True)
            if self._hp_panel is not None:
                self._load_hyperparams_for(name)
        else:
            self._hp_panel_host.setVisible(False)
        self._render_training_live(name)
        preferred_version = str(run.get("version") or "") if isinstance(run, dict) else ""
        if self._history_panel is not None:
            self._history_panel.load_scenario(name, preferred_version=preferred_version)
        if self._run_compare_panel is not None:
            self._run_compare_panel.set_scenario(name)
        self._refresh_scenario_cards(name)
        if self._history_panel is not None and not self._history_panel.current_entry() and self._artifacts_panel is not None:
            self._artifacts_panel.clear()

        # Data visualization panel: auto-load the scenario CSV for tabular scenarios.
        self._sync_data_viz_panel(entry)
        self._render_scenario_flow(entry)
        if self._ci_cd_bar is not None:
            self._ci_cd_bar.set_scenario(name)

    def _sync_data_viz_panel(self, entry: dict[str, Any]) -> None:
        if self._data_viz_panel is None:
            return
        name = str(entry.get("name", ""))
        btype = str(entry.get("backbone_type") or "yolo_detection")
        try:
            if btype == "torch_tabular":
                bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
                dataset_csv = str((bcfg or {}).get("dataset_csv") or "").strip()
                self._data_viz_panel.set_scenario_csv(dataset_csv, str(ROOT_DIR))
            elif btype == "archival_ingestion":
                bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
                self._data_viz_panel.set_archive_context(
                    str((bcfg or {}).get("corpus_id") or ""),
                    str((bcfg or {}).get("dataset_version_id") or ""),
                    str((bcfg or {}).get("latest_snapshot_id") or entry.get("archive_snapshot_id") or ""),
                    scenario=name,
                )
            else:
                self._data_viz_panel.clear()
        except Exception:
            pass

    def _update_readiness_strip(self, entry: dict[str, Any]) -> None:
        if not entry:
            self._set_readiness_value(self._ready_dataset, "0", "idle")
            self._set_readiness_value(self._ready_model, "Unset", "idle")
            self._set_readiness_value(self._ready_train, "Empty", "idle")
            self._set_readiness_value(self._ready_latest, "None", "idle")
            self._set_readiness_value(self._ready_verified, "No", "idle")
            return

        btype = str(entry.get("backbone_type") or "yolo_detection")
        status = str(entry.get("status") or "empty")
        try:
            count = int(entry.get("dataset_count") or 0)
        except Exception:
            count = 0
        count_label = f"{count} item" if count == 1 else f"{count} items"
        self._set_readiness_value(
            self._ready_dataset,
            count_label,
            "ok" if count > 0 else "warning",
        )

        if btype == "yolo_detection":
            model_label = Path(str(entry.get("base_model") or "")).name or "Unset"
            model_ok = bool(entry.get("base_model_exists")) or bool(entry.get("base_model_resolved"))
            self._set_readiness_value(self._ready_model, model_label, "ok" if model_ok else "warning")
        elif btype in ("torch_tabular", "custom_code"):
            bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
            cells = bcfg.get("cells") or bcfg.get("train_cells") or []
            cell_count = len(cells) if isinstance(cells, list) else 0
            label = f"{cell_count} cell" if cell_count == 1 else f"{cell_count} cells"
            self._set_readiness_value(self._ready_model, label, "ok" if cell_count > 0 else "warning")
        elif btype == "llm_fine_tuning":
            bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
            model_label = Path(str(bcfg.get("base_model") or entry.get("base_model") or "LLM")).name
            self._set_readiness_value(self._ready_model, model_label or "LLM", "ok")
        else:
            self._set_readiness_value(self._ready_model, btype.replace("_", " "), "ok")

        train_state = status.title() if status else "Empty"
        train_ui_state = "ok" if status in {"trained", "ready"} else ("warning" if status in {"dataset", "training"} else "idle")
        if status == "error":
            train_ui_state = "error"
        self._set_readiness_value(self._ready_train, train_state, train_ui_state)

        run = entry.get("latest_run") if isinstance(entry.get("latest_run"), dict) else None
        if run:
            version = str(run.get("final_model_name") or run.get("version") or "").strip() or "Run"
            metric = run.get("map50")
            if metric is not None:
                try:
                    latest = f"{version} mAP50 {float(metric):.3f}"
                except Exception:
                    latest = f"{version} mAP50 {metric}"
            else:
                latest = version
            self._set_readiness_value(self._ready_latest, latest, "ok")
        else:
            self._set_readiness_value(self._ready_latest, "None", "idle")

        verified = bool(entry.get("verified"))
        self._set_readiness_value(self._ready_verified, "Yes" if verified else "No", "ok" if verified else "warning")

    def _load_models(self, *, force: bool) -> None:
        if self._models and not force:
            return
        try:
            payload = self._http_get("/models")
            models = payload.get("models") if isinstance(payload, dict) else []
        except Exception as exc:
            msg = f"Model list failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        self._models = [m for m in (models or []) if isinstance(m, dict)]
        prev = str(self._model_combo.currentData() or "")
        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for model in self._models:
            value = str(model.get("value") or model.get("path") or "")
            if not value:
                continue
            label = str(model.get("name") or Path(value).name)
            size_raw = model.get("size_bytes")
            try:
                size_mb = float(size_raw) / (1024.0 * 1024.0)
            except Exception:
                size_mb = 0.0
            self._model_combo.addItem(f"{label} ({size_mb:.1f} MB)", value)
        self._model_combo.blockSignals(False)
        if prev:
            idx = self._model_combo.findData(prev)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

    def _sync_model_combo(self, base_model: str) -> None:
        target = str(base_model or "").strip()
        if not target or self._model_combo.count() == 0:
            return
        idx = self._model_combo.findData(target)
        if idx < 0:
            target_name = Path(target).name
            for i in range(self._model_combo.count()):
                val = str(self._model_combo.itemData(i) or "")
                if Path(val).name == target_name or val.endswith("/" + target_name):
                    idx = i
                    break
        if idx >= 0:
            self._model_combo.setCurrentIndex(idx)

    def _set_model(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        model_value = str(self._model_combo.currentData() or "").strip()
        if not model_value:
            self._status.setText("No model selected.")
            return
        try:
            status = self._http_post(f"/scenarios/{name}/model", {"model": model_value})
        except Exception as exc:
            msg = f"Set model failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        if isinstance(status, dict):
            self.apply_scenario_update(name, status)
        self._status.setText(f"{name} base model set to {Path(model_value).name}")
        self.scenarioMutated.emit(name)

    def _populate_algo_catalog(self) -> None:
        selected_item = self._algo_catalog.currentItem()
        selected = str(selected_item.data(Qt.ItemDataRole.UserRole) or "") if selected_item is not None else ""
        list_algo_files = _lazy_symbol(".algo_catalog", "list_algo_files")
        entries = list_algo_files()
        self._algo_catalog_loaded = True
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
        self._add_algo_override_path(path)

    def _reveal_selected_catalog_algo(self) -> None:
        it = self._algo_catalog.currentItem()
        if it is None:
            return
        path = str(it.data(Qt.ItemDataRole.UserRole) or "").strip()
        if path:
            reveal_in_finder = _lazy_symbol(".algo_catalog", "reveal_in_finder")
            reveal_in_finder(path)

    def _add_algo_override_path(self, path: str) -> None:
        text = str(path or "").strip()
        if not text:
            return
        for i in range(self._algo_override_cells.count()):
            it = self._algo_override_cells.item(i)
            if it is not None and str(it.data(Qt.ItemDataRole.UserRole) or it.text()) == text:
                return
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, text)
        self._algo_override_cells.addItem(item)

    def _remove_selected_algo_override_cell(self) -> None:
        row = self._algo_override_cells.currentRow()
        if row >= 0:
            self._algo_override_cells.takeItem(row)

    def _reveal_selected_algo_override_cell(self) -> None:
        it = self._algo_override_cells.currentItem()
        if it is None:
            return
        path = str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "").strip()
        if path:
            reveal_in_finder = _lazy_symbol(".algo_catalog", "reveal_in_finder")
            reveal_in_finder(path)

    def _algo_override_specs(self) -> list[dict[str, str]]:
        specs: list[dict[str, str]] = []
        for i in range(self._algo_override_cells.count()):
            it = self._algo_override_cells.item(i)
            if it is None:
                continue
            path = str(it.data(Qt.ItemDataRole.UserRole) or it.text() or "").strip()
            if path:
                specs.append({"path": path})
        return specs

    def _cc_flush_editor_to_row(self, row: int) -> None:
        if row < 0 or row >= len(self._cc_cells_data):
            return
        cell = self._cc_cells_data[row]
        cell["code"] = self._cc_code.toPlainText()
        nm = str(self._cc_cell_name.text() or "").strip()
        if nm:
            cell["name"] = nm
        it = self._cc_list.item(row)
        if it is not None:
            it.setText(str(cell.get("name") or cell.get("id") or f"cell_{row}"))

    def _cc_on_list_row_changed(self, row: int) -> None:
        prev = self._cc_prev_list_row
        if prev >= 0:
            self._cc_flush_editor_to_row(prev)
        self._cc_prev_list_row = row
        if row < 0 or row >= len(self._cc_cells_data):
            self._cc_cell_name.clear()
            self._cc_code.clear()
            return
        cell = self._cc_cells_data[row]
        self._cc_cell_name.setText(str(cell.get("name") or ""))
        self._cc_code.setPlainText(str(cell.get("code") or ""))

    def _cc_refresh_list(self, *, select_row: int = 0) -> None:
        self._cc_list.blockSignals(True)
        self._cc_list.clear()
        for i, cell in enumerate(self._cc_cells_data):
            label = str(cell.get("name") or cell.get("id") or f"cell_{i}")
            self._cc_list.addItem(label)
        if self._cc_cells_data:
            self._cc_list.setCurrentRow(min(select_row, len(self._cc_cells_data) - 1))
        self._cc_list.blockSignals(False)
        self._cc_prev_list_row = self._cc_list.currentRow()
        if self._cc_cells_data and self._cc_prev_list_row >= 0:
            self._cc_on_list_row_changed(self._cc_prev_list_row)

    def _cc_default_cell(self) -> dict[str, Any]:
        cid = uuid.uuid4().hex[:10]
        code = (
            "def run(ctx, prev):\n"
            "    print('datasets', ctx.datasets)\n"
            "    print('active_cell', ctx.active_cell)\n"
            "    return {'data': {'ok': True}}\n"
        )
        return {
            "id": cid,
            "name": f"cell_{cid}",
            "entry": "run",
            "code": code,
            "datasets": [],
            "pasted_files": [],
        }

    def _cc_add_cell(self) -> None:
        cur = self._cc_list.currentRow()
        if cur >= 0:
            self._cc_flush_editor_to_row(cur)
        self._cc_cells_data.append(self._cc_default_cell())
        self._cc_refresh_list(select_row=len(self._cc_cells_data) - 1)

    def _cc_remove_cell(self) -> None:
        row = self._cc_list.currentRow()
        if row < 0 or row >= len(self._cc_cells_data):
            return
        del self._cc_cells_data[row]
        self._cc_prev_list_row = -1
        self._cc_refresh_list(select_row=max(0, row - 1))

    def _cc_add_pasted_file_to_cell(self) -> None:
        row = self._cc_list.currentRow()
        if row < 0 or row >= len(self._cc_cells_data):
            self._status.setText("Select a cell before attaching pasted data.")
            return
        name = str(self._cc_paste_name.text() or "").strip() or "pasted.txt"
        body = self._cc_paste_body.toPlainText()
        if not str(body or "").strip():
            self._status.setText("Paste content is empty.")
            return
        cell = self._cc_cells_data[row]
        blobs = cell.get("pasted_files")
        if not isinstance(blobs, list):
            blobs = []
        blobs.append({"name": name, "content": body, "format": "text"})
        cell["pasted_files"] = blobs
        self._cc_paste_body.clear()
        self._status.setText(f"Attached pasted file '{name}' to cell {row + 1} (saved on Save Draft).")

    def _cc_collect_put_body(self) -> dict[str, Any]:
        row = self._cc_list.currentRow()
        if row >= 0:
            self._cc_flush_editor_to_row(row)
        raw_sd = str(self._cc_scenario_ds.toPlainText() or "").strip() or "[]"
        try:
            scenario_datasets = json.loads(raw_sd)
        except Exception:
            scenario_datasets = []
        if not isinstance(scenario_datasets, list):
            scenario_datasets = []
        out_cells: list[dict[str, Any]] = []
        for cell in self._cc_cells_data:
            c = dict(cell)
            pf = c.get("pasted_files")
            if isinstance(pf, list) and pf:
                c["pasted_files"] = pf
            else:
                c.pop("pasted_files", None)
            out_cells.append(c)
        return {"cells": out_cells, "scenario_datasets": scenario_datasets}

    def _cc_save_draft(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        put = self._http_put
        if put is None:
            self._status.setText("Save draft requires HTTP PUT (update the app shell to pass http_put).")
            return
        try:
            put(f"/scenarios/{name}/custom_cells", self._cc_collect_put_body())
        except Exception as exc:
            msg = f"Save draft failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        self._status.setText(f"{name}: custom cell draft saved.")

    def _cc_load_from_server(self, scenario: str) -> None:
        self._cc_cells_data = []
        self._cc_prev_list_row = -1
        try:
            data = self._http_get(f"/scenarios/{scenario}/custom_cells")
        except Exception:
            self._cc_scenario_ds.setPlainText("[]")
            self._cc_refresh_list()
            return
        if not isinstance(data, dict):
            self._cc_scenario_ds.setPlainText("[]")
            self._cc_refresh_list()
            return
        for c in data.get("cells") or []:
            if isinstance(c, dict):
                self._cc_cells_data.append(dict(c))
        try:
            self._cc_scenario_ds.setPlainText(json.dumps(data.get("scenario_datasets") or [], indent=2))
        except Exception:
            self._cc_scenario_ds.setPlainText("[]")
        self._cc_refresh_list(select_row=0)

    def _cc_run_draft(self) -> None:
        self._cc_save_draft()
        self._kick_training()

    def _cc_promote_template(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        self._cc_save_draft()
        tpl, ok = QInputDialog.getText(self, "Promote to Template", "Template name (files go to mlops/algos/):")
        if not ok or not str(tpl or "").strip():
            return
        try:
            self._http_post(f"/scenarios/{name}/custom_cells/promote", {"template_name": str(tpl).strip()})
        except Exception as exc:
            msg = f"Promote failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        self._status.setText(f"Promoted template '{tpl.strip()}' for {name}.")
        self.scenarioMutated.emit(name)

    def _cc_apply_template(self) -> None:
        scen = self.current_scenario()
        if not scen:
            return
        entry = self._entries.get(scen, {}) or {}
        bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
        cells = bcfg.get("cells") or []
        if not isinstance(cells, list) or not cells:
            QMessageBox.information(self, "Apply Template", "Scenario has no backbone_config.cells to load.")
            return
        loaded: list[dict[str, Any]] = []
        for spec in cells:
            if not isinstance(spec, dict):
                continue
            path = str(spec.get("path") or "").strip()
            code = ""
            if path:
                p = Path(ROOT_DIR) / path
                try:
                    if p.is_file():
                        code = p.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    code = ""
            cid = str(spec.get("id") or Path(path).stem or uuid.uuid4().hex[:8])
            loaded.append(
                {
                    "id": cid,
                    "name": str(spec.get("name") or cid),
                    "entry": str(spec.get("entry") or "run"),
                    "code": code,
                    "datasets": list(spec.get("datasets") or []) if isinstance(spec.get("datasets"), list) else [],
                    "pasted_files": [],
                }
            )
        if not loaded:
            QMessageBox.warning(self, "Apply Template", "No readable cell files found.")
            return
        self._cc_cells_data = loaded
        self._cc_refresh_list(select_row=0)
        self._status.setText("Loaded cells from scenario template into the editor (save draft to persist).")

    def _custom_cells_train_override(self) -> Optional[dict[str, Any]]:
        if self._custom_cells_editor is not None:
            override = self._custom_cells_editor.collect_train_override()
            return override or None
        name = self.current_scenario()
        if not name:
            return None
        try:
            data = self._http_get(f"/scenarios/{name}/custom_cells")
        except Exception:
            return None
        if not isinstance(data, dict):
            return None
        cells_out: list[dict[str, Any]] = []
        for c in data.get("cells") or []:
            if not isinstance(c, dict):
                continue
            path = str(c.get("path") or "").strip()
            if not path:
                continue
            spec: dict[str, Any] = {
                "id": str(c.get("id") or ""),
                "name": str(c.get("name") or ""),
                "path": path,
                "entry": str(c.get("entry") or "run"),
            }
            ds = c.get("datasets")
            if isinstance(ds, list):
                spec["datasets"] = ds
            cells_out.append(spec)
        sd = data.get("scenario_datasets")
        scenario_datasets = list(sd) if isinstance(sd, list) else []
        if not cells_out:
            return None
        return {"cells": cells_out, "datasets": scenario_datasets}

    def _training_payload(self, btype: str) -> Optional[dict[str, Any]]:
        payload: dict[str, Any] = {}
        final_name = str(self._final_model_name.text() or "").strip()
        if final_name:
            payload["final_model_name"] = final_name
        payload["auto_fresh_on_completed_resume"] = bool(self._auto_fresh_resume.isChecked())
        device_override = str(getattr(self, "_train_device_override", "") or "").strip()
        if device_override:
            payload["device"] = device_override
        storage_override = str(getattr(self, "_train_storage_override", "") or "").strip()
        if storage_override:
            payload["training_assets_root"] = storage_override
        if btype == "yolo_detection":
            model_override = str(self._model_combo.currentData() or "").strip()
            if model_override:
                payload["base_model_override"] = model_override
        elif btype == "torch_tabular":
            cells = self._algo_override_specs()
            if cells:
                payload["backbone_config_override"] = {"cells": cells}
        elif btype == "custom_code":
            ov = self._custom_cells_train_override()
            if ov:
                payload["backbone_config_override"] = ov
        return payload or None

    def _kick_training(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        self._ensure_train_console()
        self._training_points[name] = []
        self._console_buffers[name] = []
        self._train_console.reset()
        self._render_training_live(name)
        entry = self._entries.get(name, {}) or {}
        btype = str(entry.get("backbone_type") or "yolo_detection")
        if btype == "archival_ingestion":
            bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
            corpus_id = str((bcfg or {}).get("corpus_id") or "")
            dataset_version_id = str((bcfg or {}).get("dataset_version_id") or "")
            if not corpus_id or not dataset_version_id:
                msg = "Archive scenario is not linked to an imported corpus yet."
                self._status.setText(msg)
                self.errorRaised.emit(msg)
                return
            try:
                result = self._http_post(
                    f"/archives/{corpus_id}/jobs",
                    {
                        "dataset_version_id": dataset_version_id,
                        "phase": "archive_pipeline",
                        "parent_snapshot_id": str((bcfg or {}).get("latest_snapshot_id") or entry.get("archive_snapshot_id") or ""),
                        "scenario": name,
                        "write_run_artifacts": True,
                    },
                )
            except Exception as exc:
                body = getattr(exc, "response_body", "") or str(exc)
                msg = f"Archive pipeline kick failed: {body}"
                self._status.setText(msg)
                self.errorRaised.emit(msg)
                return
            job_id = str(result.get("job_id") or "")
            if job_id:
                self._latest_train_jobs[name] = job_id
            self._status.setText(f"Archive job queued: {job_id}")
            self.trainKicked.emit(name, job_id)
            self.scenarioMutated.emit(name)
            return
        payload = self._training_payload(btype)
        try:
            result = self._http_post(f"/scenarios/{name}/train", payload)
        except Exception as exc:
            body = getattr(exc, "response_body", "") or str(exc)
            msg = f"Train kick failed: {body}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        guard = result.get("training_guard") if isinstance(result, dict) else None
        if isinstance(guard, dict):
            self._pending_guard_data = dict(guard)
            if self._guard_card is not None:
                self._guard_card.apply_guard(guard)
        job_id = str(result.get("job_id") or "")
        if job_id:
            self._latest_train_jobs[name] = job_id
        self._status.setText(f"Train job queued: {result.get('job_id', '')}")
        self._train_console.focus_for_training()
        self.trainKicked.emit(name, job_id)
        self.scenarioMutated.emit(name)

    def _kick_update_training(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        self._ensure_train_console()
        self._training_points[name] = []
        self._console_buffers[name] = []
        self._train_console.reset()
        self._render_training_live(name)
        entry = self._entries.get(name, {}) or {}
        btype = str(entry.get("backbone_type") or "yolo_detection")
        if btype == "archival_ingestion":
            bcfg = entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {}
            corpus_id = str((bcfg or {}).get("corpus_id") or "")
            dataset_version_id = str((bcfg or {}).get("dataset_version_id") or "")
            if not corpus_id or not dataset_version_id:
                msg = "Archive scenario is not linked to an imported corpus yet."
                self._status.setText(msg)
                self.errorRaised.emit(msg)
                return
            try:
                result = self._http_post(
                    f"/archives/{corpus_id}/jobs",
                    {
                        "dataset_version_id": dataset_version_id,
                        "phase": "archive_reconcile",
                        "parent_snapshot_id": str((bcfg or {}).get("latest_snapshot_id") or entry.get("archive_snapshot_id") or ""),
                        "scenario": name,
                        "write_run_artifacts": True,
                    },
                )
            except Exception as exc:
                body = getattr(exc, "response_body", "") or str(exc)
                msg = f"Archive reconcile kick failed: {body}"
                self._status.setText(msg)
                self.errorRaised.emit(msg)
                return
            job_id = str(result.get("job_id") or "")
            if job_id:
                self._latest_train_jobs[name] = job_id
            self._status.setText(f"Archive reconcile queued: {job_id}")
            self.trainKicked.emit(name, job_id)
            self.scenarioMutated.emit(name)
            return
        payload = self._training_payload(btype)
        try:
            result = self._http_post(f"/scenarios/{name}/update", payload)
        except Exception as exc:
            body = getattr(exc, "response_body", "") or str(exc)
            msg = f"Update kick failed: {body}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        guard = result.get("training_guard") if isinstance(result, dict) else None
        if isinstance(guard, dict):
            self._pending_guard_data = dict(guard)
            if self._guard_card is not None:
                self._guard_card.apply_guard(guard)
        job_id = str(result.get("job_id") or "")
        if job_id:
            self._latest_train_jobs[name] = job_id
        self._status.setText(f"Update job queued: {result.get('job_id', '')}")
        self._train_console.focus_for_training()
        self.trainKicked.emit(name, job_id)
        self.scenarioMutated.emit(name)

    def _stop_training(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        job_id = str(self._latest_train_jobs.get(name, "") or "")
        if not job_id:
            self._status.setText("No active training job found to stop.")
            return
        try:
            self._http_post(f"/jobs/{job_id}/cancel", None)
        except Exception as exc:
            msg = f"Stop training failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        self._status.setText(f"Stop requested for training job {job_id}.")
        self._stop_btn.setEnabled(False)
        self.scenarioMutated.emit(name)

    def _load_hyperparams_for(self, scenario: str) -> None:
        if self._hp_panel is None:
            return
        try:
            payload = self._http_get(f"/scenarios/{scenario}/hyperparams")
        except Exception as exc:
            self._status.setText(f"Load hyperparams failed: {exc}")
            return
        if not isinstance(payload, dict):
            return
        hp = payload.get("hyperparams") or {}
        schema = payload.get("schema") or {}
        if isinstance(schema, dict) and schema:
            self._hp_schema_cache = schema
        self._hp_panel.load(hp if isinstance(hp, dict) else {}, self._hp_schema_cache)

    def _on_hp_save_pressed(self, payload: dict[str, Any]) -> None:
        name = self.current_scenario()
        if not name:
            return
        reset = False
        updates = dict(payload or {})
        if updates.pop("__reset_all__", False):
            reset = True
            updates = {}
        body = {"updates": updates, "reset": reset}
        try:
            self._http_post(f"/scenarios/{name}/hyperparams", body)
        except Exception as exc:
            msg = f"Save hyperparams failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        self._status.setText(
            f"{name}: hyperparameters {'reset to defaults' if reset else 'saved'}."
        )
        self._load_hyperparams_for(name)
        self.scenarioMutated.emit(name)

    def _on_hp_reset_pressed(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        self._load_hyperparams_for(name)

    def _mark_verified(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        try:
            self._http_post(f"/scenarios/{name}/verify", {"note": "operator verified"})
        except Exception as exc:
            msg = f"Verify failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        self._status.setText(f"{name} marked verified.")
        self.scenarioMutated.emit(name)

    def _clear_verified(self) -> None:
        name = self.current_scenario()
        if not name:
            return
        try:
            self._http_delete(f"/scenarios/{name}/verify")
        except Exception as exc:
            msg = f"Clear verified failed: {exc}"
            self._status.setText(msg)
            self.errorRaised.emit(msg)
            return
        self._status.setText(f"{name} verification cleared.")
        self.scenarioMutated.emit(name)

    def _render_training_live(self, scenario: str) -> None:
        if self._train_console is None:
            return
        series = list(self._training_points.get(scenario, []))
        epoch_points = [pt for pt in series if str(pt.get("event") or "") == "epoch"]
        self._train_console.set_points(epoch_points)
        rev = list(reversed(series))
        if not series:
            self._train_console.set_metrics_text("No live metrics yet.")
            self._train_console.set_health("idle")
            self._train_console.set_health_info("N/A")
            return

        # Most recent non-batch_metrics event drives health/epoch/loss; keep the
        # latest batch_metrics separately so it can be appended to the readout.
        latest_batch: dict[str, Any] = {}
        latest: dict[str, Any] = {}
        for pt in rev:
            if not isinstance(pt, dict):
                continue
            ev = str(pt.get("event") or "").strip().lower()
            if ev == "batch_metrics" and not latest_batch:
                latest_batch = pt
                continue
            if not latest:
                latest = pt
            if latest and latest_batch:
                break
        if not latest and series:
            latest = series[-1] if isinstance(series[-1], dict) else {}
        event = str(latest.get("event") or "").strip().lower()
        progress = latest.get("progress")
        epoch = latest.get("epoch")
        epochs = latest.get("epochs")
        train_loss = latest.get("train_loss")
        val_loss = latest.get("val_loss")
        map50 = latest.get("map50")
        val_acc = latest.get("val_acc")
        val_mae = latest.get("val_mae")
        precision = latest.get("precision")
        recall = latest.get("recall")
        err = str(latest.get("error") or "").strip()

        parts: list[str] = []
        try:
            if epoch is not None and epochs is not None:
                parts.append(f"epoch {int(epoch) + 1}/{max(1, int(epochs))}")
        except Exception:
            pass
        try:
            if progress is not None:
                parts.append(f"{float(progress):.1f}%")
        except Exception:
            pass
        if map50 is not None:
            try:
                parts.append(f"mAP50 {float(map50):.4f}")
            except Exception:
                parts.append(f"mAP50 {map50}")
        elif val_acc is not None:
            try:
                parts.append(f"val_acc {float(val_acc):.4f}")
            except Exception:
                parts.append(f"val_acc {val_acc}")
        elif val_mae is not None:
            try:
                parts.append(f"val_mae {float(val_mae):.4f}")
            except Exception:
                parts.append(f"val_mae {val_mae}")
        if train_loss is not None:
            try:
                parts.append(f"train_loss {float(train_loss):.4f}")
            except Exception:
                parts.append(f"train_loss {train_loss}")
        if val_loss is not None:
            try:
                parts.append(f"val_loss {float(val_loss):.4f}")
            except Exception:
                parts.append(f"val_loss {val_loss}")
        if precision is not None:
            try:
                parts.append(f"precision {float(precision):.4f}")
            except Exception:
                parts.append(f"precision {precision}")
        if recall is not None:
            try:
                parts.append(f"recall {float(recall):.4f}")
            except Exception:
                parts.append(f"recall {recall}")
        eta_text = self._training_eta_text(series, epoch_points, latest)
        if eta_text:
            parts.append(eta_text)
        if latest_batch:
            stall = latest_batch.get("stall_pct")
            step_ms = latest_batch.get("step_time_ms")
            data_ms = latest_batch.get("data_time_ms")
            sps = latest_batch.get("samples_per_sec")
            try:
                if stall is not None:
                    parts.append(f"stall {float(stall):.0f}%")
            except Exception:
                pass
            try:
                if step_ms is not None and data_ms is not None:
                    parts.append(f"iter {float(step_ms):.0f}ms step / {float(data_ms):.0f}ms data")
            except Exception:
                pass
            try:
                if sps is not None:
                    parts.append(f"{float(sps):.0f} img/s")
            except Exception:
                pass
        if err:
            parts.append(f"error: {err}")

        formatted_metrics = "\n".join(parts) if parts else "Waiting for metrics..."
        self._train_console.set_metrics_text(formatted_metrics)

        latest_switch = next((p for p in rev if str(p.get("event") or "") == "overflow_switch"), None)
        start_evt = next((p for p in rev if str(p.get("event") or "") == "start"), None)
        overflow_note = ""
        if isinstance(latest_switch, dict):
            overflow_note = str(latest_switch.get("message") or "").strip()
        if not overflow_note and isinstance(start_evt, dict):
            tg = start_evt.get("training_guard")
            if isinstance(tg, dict):
                op = tg.get("overflow_protocol")
                if isinstance(op, dict) and op.get("overflowed"):
                    overflow_note = str(op.get("message") or "").strip()

        health_state = "idle"
        if event == "start":
            health_state = "starting"
        elif event == "completed":
            health_state = "completed"
            if self.current_scenario() == scenario and hasattr(self, "_stop_btn"):
                self._stop_btn.setEnabled(False)
        elif event == "failed":
            if "cancel" in err.lower():
                health_state = "cancelled"
            else:
                health_state = "failed"
            if self.current_scenario() == scenario and hasattr(self, "_stop_btn"):
                self._stop_btn.setEnabled(False)
        else:
            health_state = "healthy"
            if len(epoch_points) >= 2:
                prev = epoch_points[-2]
                last = epoch_points[-1]

                def _as_float(v: Any) -> Optional[float]:
                    try:
                        return float(v)
                    except Exception:
                        return None

                prev_train = _as_float(prev.get("train_loss"))
                last_train = _as_float(last.get("train_loss"))
                prev_metric = _as_float(prev.get("map50"))
                last_metric = _as_float(last.get("map50"))
                metric_higher_better = True
                if prev_metric is None or last_metric is None:
                    prev_metric = _as_float(prev.get("val_acc"))
                    last_metric = _as_float(last.get("val_acc"))
                if prev_metric is None or last_metric is None:
                    prev_metric = _as_float(prev.get("val_mae"))
                    last_metric = _as_float(last.get("val_mae"))
                    metric_higher_better = False

                worsening = 0
                if prev_train is not None and last_train is not None and last_train > prev_train * 1.03:
                    worsening += 1
                if prev_metric is not None and last_metric is not None:
                    if metric_higher_better and last_metric < prev_metric * 0.98:
                        worsening += 1
                    if (not metric_higher_better) and last_metric > prev_metric * 1.02:
                        worsening += 1

                if worsening >= 2:
                    health_state = "at_risk"
                elif worsening == 1:
                    health_state = "watch"
                else:
                    health_state = "watch" if overflow_note else "healthy"
            elif overflow_note:
                health_state = "watch"
        self._train_console.set_health(health_state)

        health_info_parts: list[str] = []
        if event == "start":
            health_info_parts.append("Status: Starting")
        elif event == "completed":
            health_info_parts.append("Status: Completed")
        elif event == "failed":
            health_info_parts.append("Status: Failed")
        else:
            health_info_parts.append(f"Status: Training")

        if overflow_note:
            health_info_parts.append(f"Overflow protocol: {overflow_note}")
        if isinstance(start_evt, dict):
            asset_root = str(start_evt.get("asset_root") or "").strip()
            if asset_root:
                health_info_parts.append(f"Training assets: {asset_root}")
        if event == "completed" and isinstance(latest, dict):
            rd = str(latest.get("run_dir") or "").strip()
            if rd:
                health_info_parts.append(f"Run output: {rd}")

        if epoch is not None and epochs is not None:
            health_info_parts.append(f"Epoch: {int(epoch) + 1}/{max(1, int(epochs))}")
        else:
            health_info_parts.append("Epoch: N/A")

        if progress is not None:
            try:
                health_info_parts.append(f"Progress: {float(progress):.1f}%")
            except Exception:
                health_info_parts.append("Progress: N/A")
        else:
            health_info_parts.append("Progress: N/A")

        if map50 is not None:
            try:
                health_info_parts.append(f"mAP50: {float(map50):.4f}")
            except Exception:
                health_info_parts.append(f"mAP50: {map50}")
        elif val_acc is not None:
            try:
                health_info_parts.append(f"Accuracy: {float(val_acc):.4f}")
            except Exception:
                health_info_parts.append(f"Accuracy: {val_acc}")
        elif val_mae is not None:
            try:
                health_info_parts.append(f"MAE: {float(val_mae):.4f}")
            except Exception:
                health_info_parts.append(f"MAE: {val_mae}")
        else:
            health_info_parts.append("Metric: N/A")

        if train_loss is not None:
            try:
                health_info_parts.append(f"Train Loss: {float(train_loss):.4f}")
            except Exception:
                health_info_parts.append(f"Train Loss: {train_loss}")
        else:
            health_info_parts.append("Train Loss: N/A")

        if val_loss is not None:
            try:
                health_info_parts.append(f"Val Loss: {float(val_loss):.4f}")
            except Exception:
                health_info_parts.append(f"Val Loss: {val_loss}")
        else:
            health_info_parts.append("Val Loss: N/A")

        health_info = "\n".join(health_info_parts)
        self._train_console.set_health_info(health_info)

    def _training_eta_text(
        self,
        series: list[dict[str, Any]],
        epoch_points: list[dict[str, Any]],
        latest: dict[str, Any],
    ) -> str:
        event = str(latest.get("event") or "").strip().lower()
        if event in {"completed", "failed", "canceled", "cancelled", "interrupted"}:
            return ""
        epochs_raw = latest.get("epochs")
        try:
            total_epochs = int(epochs_raw)
        except Exception:
            total_epochs = 0
        if total_epochs <= 0:
            return ""
        completed_epochs = 0
        for pt in epoch_points:
            try:
                completed_epochs = max(completed_epochs, int(pt.get("epoch")) + 1)
            except Exception:
                pass
        if completed_epochs >= total_epochs:
            return "ETA finishing"
        now = time.time()
        start_ts = None
        for pt in series:
            if str(pt.get("event") or "").strip().lower() == "start":
                try:
                    start_ts = float(pt.get("timestamp"))
                    break
                except Exception:
                    start_ts = None
        epoch_times: list[float] = []
        for pt in epoch_points:
            try:
                epoch_times.append(float(pt.get("timestamp")))
            except Exception:
                pass
        per_epoch = 0.0
        if len(epoch_times) >= 2:
            deltas = [
                max(0.0, epoch_times[i] - epoch_times[i - 1])
                for i in range(1, len(epoch_times))
            ]
            if deltas:
                per_epoch = sum(deltas[-3:]) / min(3, len(deltas))
        elif completed_epochs > 0 and epoch_times and start_ts is not None:
            per_epoch = max(0.0, epoch_times[-1] - start_ts) / max(1, completed_epochs)
        elif start_ts is not None:
            progress = latest.get("progress")
            try:
                pct = float(progress)
            except Exception:
                pct = 0.0
            if pct > 0.5:
                elapsed = max(0.0, now - start_ts)
                return f"ETA {format_duration_seconds(elapsed * ((100.0 - pct) / pct))}"
        if per_epoch <= 0:
            return "ETA calculating"
        remaining = (total_epochs - completed_epochs) * per_epoch
        return f"ETA {format_duration_seconds(remaining)}"

    def _on_history_selected(self, entry: object) -> None:
        self._ensure_artifacts_panel()
        if not isinstance(entry, dict):
            self._artifacts_panel.clear()
            return
        scenario = self.current_scenario()
        version = str(entry.get("version") or "").strip()
        if not scenario or not version:
            self._artifacts_panel.clear()
            return
        self._artifacts_panel.load_run(scenario, version)
