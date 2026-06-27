from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QDragEnterEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ...config import COLOR_SCHEME_CHOICES, GALLERY_DB_PATH, ROOT_DIR
from .collapsible_section import CollapsibleSection
from .cvops_theme import normalize_color_override, normalize_ui_scale_pct
from .path_actions import reveal_in_file_manager
from .patch_parallelogram_buttons import normalize_cvops_button_shape
from .time_format import TIME_FORMAT_12H, TIME_FORMAT_24H, normalize_time_format

_WORKSPACE_BG_PREFIX = "workspace_background"
_ALLOWED_WALLPAPER_SUFFIX = frozenset({".png", ".jpg", ".jpeg", ".webp"})
_LOGGER = logging.getLogger(__name__)
_density_deprecation_logged = False


def workspace_background_reserved_names() -> tuple[str, ...]:
    return tuple(f"{_WORKSPACE_BG_PREFIX}{ext}" for ext in (".png", ".jpg", ".jpeg", ".webp"))


def _normalize_optional_color(value: Any) -> str:
    return normalize_color_override(value)


@dataclass
class CvOpsSettings:
    color_scheme: str = "aurora"
    button_shape: str = "parallelogram"
    ui_scale_pct: int = 100
    time_format: str = TIME_FORMAT_24H
    title_text_color: str = ""
    """Optional #RRGGBB override for widgets marked isTitle."""
    title_background_color: str = ""
    """Optional #RRGGBB override for title chip/background fill."""
    ui_text_color: str = ""
    """Optional #RRGGBB override for primary UI text."""
    ui_muted_text_color: str = ""
    """Optional #RRGGBB override for secondary UI text."""
    ui_background_color: str = ""
    """Optional #RRGGBB override for the root workbench background."""
    ui_panel_background_color: str = ""
    """Optional #RRGGBB override for panels, cards, and pane surfaces."""
    ui_control_background_color: str = ""
    """Optional #RRGGBB override for inputs, menus, and button fills."""
    ui_accent_color: str = ""
    """Optional #RRGGBB override for accent, hover, and selection chrome."""
    auto_start_dashboard: bool = False
    dashboard_port: int = 8501
    health_poll_ms: int = 2000
    gallery_poll_ms: int = 3000
    dashboard_poll_ms: int = 1500
    last_scenario: str = ""
    show_event_pulse: bool = True
    custom_workspace_background: bool = False
    """When True and a saved image exists under state, stretch it as the CV Ops workbench backdrop."""
    workspace_background_asset: str = ""
    """Filename only, relative to CV Ops state dir (e.g. workspace_background.png)."""

    workspace_backdrop_wear_alpha: int = 5
    """Peel Insight global Wear fill under Cv Ops (**0–100**, lower = wallpaper shows more)."""
    workspace_backdrop_scale_tabs: int = 50
    workspace_backdrop_scale_frames: int = 50
    workspace_backdrop_scale_cells: int = 50
    workspace_backdrop_scale_controls: int = 50
    """Layer strength vs preset (**50** = default); scales rgba alpha per tier (10–180)."""
    ui_shell_plane: str = "workbench"
    """Last shell plane: ``ecosystem`` or ``workbench``."""
    ui_shell_mode: str = "explore"
    """Last workbench rail mode slug (explore, test, data, …)."""
    ui_shell_preset: str = "train"
    """Last explorer layout preset: train, eval, lineage."""
    ui_shell_split_outer_b64: str = ""
    """Base64 ``QSplitter`` state for workbench main/tray vertical split."""
    ui_shell_split_explorer_b64: str = ""
    """Base64 ``QSplitter`` state for workbench catalog/main horizontal split."""
    ui_shell_split_settings_b64: str = ""
    """Base64 ``QSplitter`` state for settings/diagnostics horizontal split."""
    ui_shell_closed_panes: str = ""
    """Comma-separated workbench catalog/tray panes hidden by the operator."""


def load_cvops_settings(path: Path) -> CvOpsSettings:
    settings = CvOpsSettings()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return settings
    if not isinstance(raw, dict):
        return settings

    color = str(raw.get("color_scheme") or settings.color_scheme).strip().lower()
    if color in COLOR_SCHEME_CHOICES:
        settings.color_scheme = color
    if "density_mode" in raw:
        global _density_deprecation_logged
        if not _density_deprecation_logged:
            _LOGGER.warning(
                "CvOps setting 'density_mode' is deprecated; layout is now driven by window width."
            )
            _density_deprecation_logged = True
    settings.button_shape = normalize_cvops_button_shape(raw.get("button_shape", settings.button_shape))
    settings.ui_scale_pct = normalize_ui_scale_pct(raw.get("ui_scale_pct"), default=settings.ui_scale_pct)
    settings.time_format = normalize_time_format(raw.get("time_format", settings.time_format))
    settings.title_text_color = _normalize_optional_color(raw.get("title_text_color"))
    settings.title_background_color = _normalize_optional_color(raw.get("title_background_color"))
    settings.ui_text_color = _normalize_optional_color(raw.get("ui_text_color"))
    settings.ui_muted_text_color = _normalize_optional_color(raw.get("ui_muted_text_color"))
    settings.ui_background_color = _normalize_optional_color(raw.get("ui_background_color"))
    settings.ui_panel_background_color = _normalize_optional_color(raw.get("ui_panel_background_color"))
    settings.ui_control_background_color = _normalize_optional_color(raw.get("ui_control_background_color"))
    settings.ui_accent_color = _normalize_optional_color(raw.get("ui_accent_color"))
    # Dashboard auto-start is retired; keep the field for settings-file compatibility
    # but force manual launch from the UI.
    settings.auto_start_dashboard = False
    settings.dashboard_port = _int_range(raw.get("dashboard_port"), 1024, 65535, settings.dashboard_port)
    settings.health_poll_ms = _int_range(raw.get("health_poll_ms"), 500, 60000, settings.health_poll_ms)
    settings.gallery_poll_ms = _int_range(raw.get("gallery_poll_ms"), 1000, 60000, settings.gallery_poll_ms)
    settings.dashboard_poll_ms = _int_range(raw.get("dashboard_poll_ms"), 500, 60000, settings.dashboard_poll_ms)
    settings.last_scenario = str(raw.get("last_scenario") or "").strip()
    settings.show_event_pulse = bool(raw.get("show_event_pulse", settings.show_event_pulse))
    settings.custom_workspace_background = bool(raw.get("custom_workspace_background", settings.custom_workspace_background))
    asset = str(raw.get("workspace_background_asset") or settings.workspace_background_asset or "").strip()
    if asset and ".." not in asset and "/" not in asset and "\\" not in asset:
        settings.workspace_background_asset = asset
    settings.workspace_backdrop_wear_alpha = _int_range(
        raw.get("workspace_backdrop_wear_alpha"), 0, 100, settings.workspace_backdrop_wear_alpha
    )
    settings.workspace_backdrop_scale_tabs = _int_range(
        raw.get("workspace_backdrop_scale_tabs"), 10, 180, settings.workspace_backdrop_scale_tabs
    )
    settings.workspace_backdrop_scale_frames = _int_range(
        raw.get("workspace_backdrop_scale_frames"), 10, 180, settings.workspace_backdrop_scale_frames
    )
    settings.workspace_backdrop_scale_cells = _int_range(
        raw.get("workspace_backdrop_scale_cells"), 10, 180, settings.workspace_backdrop_scale_cells
    )
    settings.workspace_backdrop_scale_controls = _int_range(
        raw.get("workspace_backdrop_scale_controls"), 10, 180, settings.workspace_backdrop_scale_controls
    )
    plane = str(raw.get("ui_shell_plane") or settings.ui_shell_plane).strip().lower()
    if plane in {"ecosystem", "workbench"}:
        settings.ui_shell_plane = plane
    mode = str(raw.get("ui_shell_mode") or settings.ui_shell_mode).strip().lower()
    if mode:
        settings.ui_shell_mode = mode
    preset = str(raw.get("ui_shell_preset") or settings.ui_shell_preset).strip().lower()
    if preset in {"train", "eval", "lineage"}:
        settings.ui_shell_preset = preset
    settings.ui_shell_split_outer_b64 = str(raw.get("ui_shell_split_outer_b64") or "").strip()
    settings.ui_shell_split_explorer_b64 = str(raw.get("ui_shell_split_explorer_b64") or "").strip()
    settings.ui_shell_split_settings_b64 = str(raw.get("ui_shell_split_settings_b64") or "").strip()
    settings.ui_shell_closed_panes = str(raw.get("ui_shell_closed_panes") or "").strip()
    return settings


def save_cvops_settings(path: Path, settings: CvOpsSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_workspace_wallpaper_path(settings: CvOpsSettings, state_dir: Path) -> Optional[Path]:
    if not settings.custom_workspace_background:
        return None
    base = Path(state_dir).resolve()
    asset = settings.workspace_background_asset.strip()
    if asset and "/" not in asset and "\\" not in asset and ".." not in asset:
        cand = base / asset
        if cand.is_file():
            return cand
    for name in workspace_background_reserved_names():
        p = base / name
        if p.is_file():
            return p
    return None


def _int_range(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        return default
    return max(lo, min(hi, parsed))


class _WorkspaceBackdropDropZone(QFrame):
    """Accepts image file drops and copies them into CV Ops state."""

    fileDropped = pyqtSignal(Path)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceBackdropDrop")
        self.setAcceptDrops(True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setMinimumHeight(104)
        self._label = QLabel("Drop an image here — PNG, JPEG, or WebP")
        self._label.setWordWrap(True)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ll = QVBoxLayout(self)
        ll.setContentsMargins(12, 12, 12, 12)
        ll.addWidget(self._label)

    def set_hint(self, text: str) -> None:
        self._label.setText(text)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if self._first_image_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # type: ignore[override]
        if self._first_image_path(event.mimeData()) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        path = self._first_image_path(event.mimeData())
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.fileDropped.emit(path)

    @staticmethod
    def _first_image_path(mime) -> Optional[Path]:  # noqa: ANN001 (QMimeData ring)
        if not mime.hasUrls():
            return None
        from PyQt6.QtCore import QUrl  # noqa: PLC0415

        for url in mime.urls():
            if isinstance(url, QUrl) and url.isLocalFile():
                p = Path(url.toLocalFile())
                if p.is_file() and p.suffix.lower() in _ALLOWED_WALLPAPER_SUFFIX:
                    return p
        return None


class CvOpsSettingsPanel(QWidget):
    colorSchemeChanged = pyqtSignal(str)
    buttonShapeChanged = pyqtSignal(str)
    uiScaleChanged = pyqtSignal(int)
    timeFormatChanged = pyqtSignal(str)
    themeOverridesChanged = pyqtSignal()
    titleStyleChanged = pyqtSignal()
    dashboardSettingsChanged = pyqtSignal(bool, int)
    dashboardOpenScopeRequested = pyqtSignal()
    pollIntervalsChanged = pyqtSignal(int, int, int)
    dashboardStartRequested = pyqtSignal()
    dashboardStopRequested = pyqtSignal()
    dashboardOpenRequested = pyqtSignal()
    dashboardReloadRequested = pyqtSignal()
    stateExportRequested = pyqtSignal()
    workspaceBackdropChanged = pyqtSignal()
    eventPulseVisibilityChanged = pyqtSignal(bool)

    def __init__(
        self,
        *,
        settings_path: Path,
        settings: CvOpsSettings,
        host: str,
        port: int,
        dashboard_url: str,
        state_dir: Path,
        jobs_db_path: Path,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsSettingsPanel")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self._settings_path = Path(settings_path)
        self._settings = settings
        self._state_dir = Path(state_dir)
        self._jobs_db_path = Path(jobs_db_path)
        self._dashboard_url = str(dashboard_url)
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self.setMinimumWidth(self._PANEL_MIN_W)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)

        scroll = QScrollArea()
        scroll.setObjectName("cvOpsSettingsScroll")
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.viewport().setObjectName("cvOpsSettingsViewport")
        scroll.setMinimumWidth(self._PANEL_MIN_W)
        inner = QWidget()
        inner.setObjectName("cvOpsSettingsInner")
        inner.setMinimumWidth(self._SECTION_W)
        inner.setMaximumWidth(self._SECTION_W)
        inner.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Keep the whole settings menu as a tidy left-aligned column instead of
        # cards that stretch to the window edge.
        section_w = self._SECTION_W
        for section in (
            self._build_appearance_section(),
            self._build_runtime_section(host=host, port=port),
            self._build_dashboard_section(),
            self._build_storage_section(),
        ):
            section.setMinimumWidth(section_w)
            section.setMaximumWidth(section_w)
            section.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Maximum)
            layout.addWidget(section, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setMaximumWidth(self._SECTION_W)
        outer.addWidget(self._status, alignment=Qt.AlignmentFlag.AlignLeft)

    @property
    def settings(self) -> CvOpsSettings:
        return self._settings

    def set_dashboard_url(self, dashboard_url: str) -> None:
        self._dashboard_url = str(dashboard_url)
        self._dashboard_url_label.setText(self._dashboard_url)

    def set_dashboard_status(self, text: str) -> None:
        label = getattr(self, "_dashboard_status", None)
        if label is not None:
            label.setText(str(text or ""))

    def _build_appearance_section(self) -> QWidget:
        section = CollapsibleSection("Appearance", expanded=True)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self._tune_form(form)

        self._color_combo = QComboBox()
        self._color_combo.addItems(list(COLOR_SCHEME_CHOICES))
        self._color_combo.setCurrentText(self._settings.color_scheme)
        self._color_combo.currentTextChanged.connect(self._on_color_changed)
        form.addRow("Color scheme", self._cap(self._color_combo))

        self._button_shape_combo = QComboBox()
        self._button_shape_combo.addItem("Square", "none")
        self._button_shape_combo.addItem("Radial", "radial")
        self._button_shape_combo.addItem("Parallelogram", "parallelogram")
        self._button_shape_combo.addItem("Octagon", "octagon")
        idx = self._button_shape_combo.findData(normalize_cvops_button_shape(self._settings.button_shape))
        self._button_shape_combo.setCurrentIndex(max(0, idx))
        self._button_shape_combo.currentIndexChanged.connect(self._on_button_shape_changed)
        form.addRow("Button shape", self._cap(self._button_shape_combo))

        self._ui_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._ui_scale_slider.setRange(70, 140)
        self._ui_scale_slider.setValue(normalize_ui_scale_pct(self._settings.ui_scale_pct))
        self._ui_scale_slider.valueChanged.connect(self._on_ui_scale_changed)
        self._ui_scale_value = QLabel("")
        self._ui_scale_value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._sync_ui_scale_label(self._ui_scale_slider.value())
        ui_scale_row = QWidget()
        ui_scale_layout = QHBoxLayout(ui_scale_row)
        ui_scale_layout.setContentsMargins(0, 0, 0, 0)
        ui_scale_layout.setSpacing(8)
        ui_scale_layout.addWidget(self._ui_scale_slider, stretch=1)
        ui_scale_layout.addWidget(self._ui_scale_value)
        form.addRow("UI scale", self._cap(ui_scale_row, self._CONTENT_W))

        self._time_format_combo = QComboBox()
        self._time_format_combo.addItem("24-hour", TIME_FORMAT_24H)
        self._time_format_combo.addItem("12-hour", TIME_FORMAT_12H)
        idx = self._time_format_combo.findData(normalize_time_format(self._settings.time_format))
        self._time_format_combo.setCurrentIndex(max(0, idx))
        self._time_format_combo.currentIndexChanged.connect(self._on_time_format_changed)
        form.addRow("Time format", self._cap(self._time_format_combo))

        self._theme_color_edits: dict[str, QLineEdit] = {}
        color_rows = (
            (
                "Title text",
                "title_text_color",
                "Optional title text color override. Leave empty to use the selected theme.",
            ),
            (
                "Title background",
                "title_background_color",
                "Optional title background override. Leave empty to use the selected theme.",
            ),
            (
                "UI text",
                "ui_text_color",
                "Optional primary text override for the wider UI.",
            ),
            (
                "Muted text",
                "ui_muted_text_color",
                "Optional secondary text override for labels, tabs, hints, and disabled controls.",
            ),
            (
                "Workbench background",
                "ui_background_color",
                "Optional root background override for the CV Ops workbench.",
            ),
            (
                "Panel background",
                "ui_panel_background_color",
                "Optional panel/card surface override.",
            ),
            (
                "Control background",
                "ui_control_background_color",
                "Optional input, menu, and button fill override.",
            ),
            (
                "Accent",
                "ui_accent_color",
                "Optional accent override for hover, selection, and active chrome.",
            ),
        )
        for label, field, tooltip in color_rows:
            row, edit = self._color_override_row(
                initial=str(getattr(self._settings, field, "") or ""),
                tooltip=tooltip,
                pick_cb=lambda f=field, lbl=label: self._pick_theme_color(f, lbl),
                reset_cb=lambda f=field, lbl=label: self._reset_theme_color(f, lbl),
                edit_cb=lambda f=field, lbl=label: self._on_theme_color_edited(f, lbl),
            )
            self._theme_color_edits[field] = edit
            form.addRow(label, row)

        section.body_layout().addLayout(form)

        self._event_pulse_chk = QCheckBox("Show scrolling notification bar")
        self._event_pulse_chk.setChecked(bool(self._settings.show_event_pulse))
        self._event_pulse_chk.stateChanged.connect(self._on_event_pulse_toggled)
        section.body_layout().addWidget(self._event_pulse_chk)

        self._custom_backdrop_chk = QCheckBox("Custom workbench background image")
        self._custom_backdrop_chk.setChecked(self._settings.custom_workspace_background)
        self._custom_backdrop_chk.stateChanged.connect(self._on_custom_workspace_backdrop_toggled)
        section.body_layout().addWidget(self._custom_backdrop_chk)

        self._backdrop_drop = _WorkspaceBackdropDropZone()
        self._backdrop_drop.fileDropped.connect(self._on_workspace_backdrop_dropped)
        self._backdrop_drop.setMinimumWidth(self._CONTENT_W)
        self._backdrop_drop.setMaximumWidth(self._CONTENT_W)
        section.body_layout().addWidget(self._backdrop_drop, alignment=Qt.AlignmentFlag.AlignLeft)

        rm_row = QHBoxLayout()
        self._backdrop_remove_btn = QPushButton("Remove saved backdrop")
        self._backdrop_remove_btn.clicked.connect(self._on_workspace_backdrop_removed)
        rm_row.addWidget(self._backdrop_remove_btn)
        rm_row.addStretch(1)
        section.body_layout().addLayout(rm_row)

        self._backdrop_footer = QLabel("")
        self._backdrop_footer.setWordWrap(True)
        self._backdrop_footer.setProperty("muted", True)
        section.body_layout().addWidget(self._backdrop_footer)

        self._backdrop_blend_rows: list[QWidget] = []
        self._backdrop_blend_reads: list[tuple[QSlider, QLabel]] = []

        layered = QLabel("Backdrop layering")
        layered.setWordWrap(True)
        layered.setProperty("muted", True)
        section.body_layout().addWidget(layered)
        lh = QLabel(
            "Global veil trims the Insight Wear stylesheet behind CV Ops "
            "(0 = peel it off entirely). Tier sliders scale Cv Ops chrome; 50 matches presets — "
            "lower shows more wallpaper, higher makes panels more solid."
        )
        lh.setWordWrap(True)
        lh.setProperty("muted", True)
        section.body_layout().addWidget(lh)

        w0, self._blend_wear_slider, _ = self._backdrop_blend_slider_row(
            "Global sheet / Wear peel",
            "Tint over the Insight global Wear root under Cv Ops. Use 0 to keep only your photo.",
            0,
            100,
            self._settings.workspace_backdrop_wear_alpha,
        )
        section.body_layout().addWidget(w0, alignment=Qt.AlignmentFlag.AlignLeft)
        w1, self._blend_tabs_slider, _ = self._backdrop_blend_slider_row(
            "Tab shell & orbit pane",
            "Alpha multiplier for main tab cavity vs backdrop preset (50 = default).",
            10,
            180,
            self._settings.workspace_backdrop_scale_tabs,
        )
        section.body_layout().addWidget(w1, alignment=Qt.AlignmentFlag.AlignLeft)
        w2, self._blend_frames_slider, _ = self._backdrop_blend_slider_row(
            "Frames / glass tier",
            "Generic QFrame and glass surfaces.",
            10,
            180,
            self._settings.workspace_backdrop_scale_frames,
        )
        section.body_layout().addWidget(w2, alignment=Qt.AlignmentFlag.AlignLeft)
        w3, self._blend_cells_slider, _ = self._backdrop_blend_slider_row(
            "Ops cells",
            "Primary card fills (opsCell).",
            10,
            180,
            self._settings.workspace_backdrop_scale_cells,
        )
        section.body_layout().addWidget(w3, alignment=Qt.AlignmentFlag.AlignLeft)
        w4, self._blend_controls_slider, _ = self._backdrop_blend_slider_row(
            "Controls & inputs",
            "Buttons, inputs, dense lists — highest alpha tier for readability.",
            10,
            180,
            self._settings.workspace_backdrop_scale_controls,
        )
        section.body_layout().addWidget(w4, alignment=Qt.AlignmentFlag.AlignLeft)

        self._refresh_workspace_backdrop_ui()

        hint = QLabel(
            "Theme changes apply immediately. Use Ctrl+=, Ctrl+-, and Ctrl+0 to zoom CV Ops in, out, or reset."
        )
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        section.body_layout().addWidget(hint)
        return section

    def _build_runtime_section(self, *, host: str, port: int) -> QWidget:
        section = CollapsibleSection("Runtime", expanded=True)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self._tune_form(form)
        form.addRow("API URL", self._cap(self._readonly_label(f"http://{host}:{port}"), self._CONTENT_W))
        form.addRow("WebSocket", self._cap(self._readonly_label(f"ws://{host}:{port}/events"), self._CONTENT_W))

        self._health_spin = self._seconds_spin(self._settings.health_poll_ms, 0.5, 60.0)
        self._dashboard_spin = self._seconds_spin(self._settings.dashboard_poll_ms, 0.5, 60.0)
        self._health_spin.valueChanged.connect(self._on_poll_intervals_changed)
        self._dashboard_spin.valueChanged.connect(self._on_poll_intervals_changed)
        form.addRow("Health poll", self._cap(self._health_spin, 140))
        form.addRow("Dashboard poll", self._cap(self._dashboard_spin, 140))
        section.body_layout().addLayout(form)
        return section

    def _build_dashboard_section(self) -> QWidget:
        section = CollapsibleSection("Dashboard", expanded=True)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self._tune_form(form)

        start_btn = QPushButton("Start Dashboard")
        start_btn.clicked.connect(self.dashboardStartRequested.emit)
        form.addRow("Launch", start_btn)

        self._dashboard_port_spin = QSpinBox()
        self._dashboard_port_spin.setRange(1024, 65535)
        self._dashboard_port_spin.setValue(int(self._settings.dashboard_port))
        self._dashboard_port_spin.valueChanged.connect(self._on_dashboard_settings_changed)
        form.addRow("Port", self._cap(self._dashboard_port_spin, 140))

        self._dashboard_url_label = self._readonly_label(self._dashboard_url)
        form.addRow("URL", self._cap(self._dashboard_url_label, self._CONTENT_W))
        section.body_layout().addLayout(form)

        open_scope_btn = QPushButton("Open in Scope")
        open_scope_btn.clicked.connect(self.dashboardOpenScopeRequested.emit)
        section.body_layout().addWidget(open_scope_btn, alignment=Qt.AlignmentFlag.AlignLeft)

        hint = QLabel("Use the Scope tab for the embedded dashboard web view.")
        hint.setWordWrap(True)
        hint.setProperty("muted", True)
        section.body_layout().addWidget(hint)
        return section

    def _build_storage_section(self) -> QWidget:
        section = CollapsibleSection("Storage", expanded=True)
        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(8)
        self._tune_form(form)
        form.addRow("Repo root", self._path_row(ROOT_DIR))
        form.addRow("Dataset library", self._path_row(ROOT_DIR / "database"))
        form.addRow("CV Ops state", self._path_row(self._state_dir))
        form.addRow("Jobs database", self._path_row(self._jobs_db_path))
        form.addRow("Face gallery DB (optional)", self._path_row(GALLERY_DB_PATH))
        form.addRow("Settings file", self._path_row(self._settings_path))
        section.body_layout().addLayout(form)

        export_btn = QPushButton("Export State")
        export_btn.clicked.connect(self.stateExportRequested.emit)
        section.body_layout().addWidget(export_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        return section

    def _path_row(self, path: Path) -> QWidget:
        row = QWidget()
        row.setMinimumWidth(self._CONTENT_W)
        row.setMaximumWidth(self._CONTENT_W)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = self._readonly_label(str(path))
        layout.addWidget(label, stretch=1)
        btn = QPushButton("Reveal")
        btn.setProperty("variant", "ghost")
        btn.clicked.connect(lambda _checked=False, p=path: self._reveal_path(p))
        layout.addWidget(btn)
        return row

    @staticmethod
    def _readonly_label(text: str) -> QLabel:
        label = QLabel(str(text))
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        return label

    # Width of the settings content column. Fields are left-aligned and kept to
    # this width instead of stretching to the window edge (which looks tacky).
    _CONTENT_W = 500
    _FIELD_W = 300
    _SECTION_W = 580
    _PANEL_MIN_W = 320

    @staticmethod
    def _tune_form(form: QFormLayout) -> None:
        """Left-align a form so labels start at the left edge and fields stay at
        their natural size instead of stretching across the whole window."""
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)

    @classmethod
    def _cap(cls, widget: QWidget, width: Optional[int] = None) -> QWidget:
        """Cap a control's width so it does not stretch to the window edge."""
        target = int(width if width is not None else cls._FIELD_W)
        widget.setMinimumWidth(max(96, int(round(target * 0.72))))
        widget.setMaximumWidth(target)
        return widget

    def _color_override_row(
        self,
        *,
        initial: str,
        tooltip: str,
        pick_cb: Callable[[], None],
        reset_cb: Callable[[], None],
        edit_cb: Callable[[], None],
    ) -> tuple[QWidget, QLineEdit]:
        row = QWidget()
        row.setMinimumWidth(self._CONTENT_W)
        row.setMaximumWidth(self._CONTENT_W)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        edit = QLineEdit()
        edit.setPlaceholderText("Theme default (#RRGGBB or rgb)")
        edit.setToolTip(tooltip)
        edit.setText(_normalize_optional_color(initial))
        edit.editingFinished.connect(edit_cb)
        self._apply_color_edit_style(edit, edit.text())
        layout.addWidget(edit, stretch=1)

        pick_btn = QPushButton("Pick")
        pick_btn.setToolTip(tooltip)
        pick_btn.clicked.connect(pick_cb)
        layout.addWidget(pick_btn)

        reset_btn = QPushButton("Reset")
        reset_btn.clicked.connect(reset_cb)
        layout.addWidget(reset_btn)
        return row, edit

    @staticmethod
    def _apply_color_edit_style(edit: QLineEdit, color: str) -> None:
        normalized = _normalize_optional_color(color)
        if not normalized:
            edit.setStyleSheet("")
            return
        swatch = QColor(normalized)
        fg = "#000000" if swatch.lightness() > 150 else "#FFFFFF"
        edit.setStyleSheet(
            "QLineEdit {"
            f" background: {normalized};"
            f" color: {fg};"
            f" border: 1px solid {normalized};"
            "}"
        )

    @staticmethod
    def _seconds_spin(ms: int, lo: float, hi: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(1)
        spin.setSingleStep(0.5)
        spin.setSuffix(" sec")
        spin.setValue(max(spin.minimum(), min(spin.maximum(), float(ms) / 1000.0)))
        return spin

    @staticmethod
    def _spin_ms(spin: QDoubleSpinBox) -> int:
        return int(round(spin.value() * 1000.0))

    def set_ui_scale_pct(self, value: int) -> None:
        self._ui_scale_slider.setValue(normalize_ui_scale_pct(value))

    def _sync_ui_scale_label(self, value: int) -> None:
        self._ui_scale_value.setText(f"{int(value)}%")

    def _backdrop_blend_slider_row(
        self,
        title: str,
        tooltip: str,
        lo: int,
        hi: int,
        initial: int,
    ) -> tuple[QWidget, QSlider, QLabel]:
        outer = QWidget()
        outer.setMinimumWidth(self._CONTENT_W)
        outer.setMaximumWidth(self._CONTENT_W)
        vl = QVBoxLayout(outer)
        vl.setContentsMargins(0, 6, 0, 2)
        vl.setSpacing(4)
        head = QLabel(title)
        head.setToolTip(tooltip)
        vl.addWidget(head)

        row = QHBoxLayout()
        row.setSpacing(8)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(lo, hi)
        sl.setValue(max(lo, min(hi, initial)))
        val_lbl = QLabel(str(sl.value()))
        val_lbl.setMinimumWidth(38)
        val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        def _upd_val(v: int, lbl: QLabel = val_lbl) -> None:
            lbl.setText(str(v))

        sl.valueChanged.connect(_upd_val)
        sl.valueChanged.connect(self._persist_backdrop_blend_from_sliders)
        row.addWidget(sl, stretch=1)
        row.addWidget(val_lbl)
        vl.addLayout(row)
        self._backdrop_blend_rows.append(outer)
        self._backdrop_blend_reads.append((sl, val_lbl))
        return outer, sl, val_lbl

    def _sync_backdrop_blend_sliders(self) -> None:
        wp_on = resolve_workspace_wallpaper_path(self._settings, self._state_dir) is not None
        for rw in self._backdrop_blend_rows:
            rw.setEnabled(wp_on)
        vals = (
            self._settings.workspace_backdrop_wear_alpha,
            self._settings.workspace_backdrop_scale_tabs,
            self._settings.workspace_backdrop_scale_frames,
            self._settings.workspace_backdrop_scale_cells,
            self._settings.workspace_backdrop_scale_controls,
        )
        pairs = getattr(self, "_backdrop_blend_reads", None)
        if not pairs or len(pairs) != len(vals):
            return
        for (sl, lbl), v in zip(pairs, vals):
            sl.blockSignals(True)
            sl.setValue(max(sl.minimum(), min(sl.maximum(), int(v))))
            sl.blockSignals(False)
            lbl.setText(str(sl.value()))

    def _persist_backdrop_blend_from_sliders(self, *_args: Any) -> None:
        if resolve_workspace_wallpaper_path(self._settings, self._state_dir) is None:
            return
        if not getattr(self, "_blend_wear_slider", None):
            return
        self._settings.workspace_backdrop_wear_alpha = int(self._blend_wear_slider.value())
        self._settings.workspace_backdrop_scale_tabs = int(self._blend_tabs_slider.value())
        self._settings.workspace_backdrop_scale_frames = int(self._blend_frames_slider.value())
        self._settings.workspace_backdrop_scale_cells = int(self._blend_cells_slider.value())
        self._settings.workspace_backdrop_scale_controls = int(self._blend_controls_slider.value())
        try:
            save_cvops_settings(self._settings_path, self._settings)
            self._status.setText("Backdrop layering saved.")
        except Exception as exc:
            self._status.setText(f"Failed to save layering: {exc}")
            return
        self.workspaceBackdropChanged.emit()

    def _stored_backdrop_file(self) -> Optional[Path]:
        base = self._state_dir
        asset = self._settings.workspace_background_asset.strip()
        if asset and "/" not in asset and "\\" not in asset and ".." not in asset:
            p = base / asset
            if p.is_file():
                return p
        for name in workspace_background_reserved_names():
            p = base / name
            if p.is_file():
                return p
        return None

    def _refresh_workspace_backdrop_ui(self) -> None:
        saved = self._stored_backdrop_file()
        self._backdrop_remove_btn.setEnabled(saved is not None)
        wp_active = resolve_workspace_wallpaper_path(self._settings, self._state_dir)
        if saved is not None:
            self._backdrop_drop.set_hint(
                f"Saved backdrop: {saved.name}\nDrop another image here to replace it."
            )
        else:
            self._backdrop_drop.set_hint("Drop an image here — PNG, JPEG, or WebP.")
        if wp_active is not None:
            self._backdrop_footer.setText(
                "The image is scaled to fill behind CV Ops. Uncheck “Custom workbench background image” "
                "to use the theme backdrop again (the file stays saved)."
            )
        elif self._settings.custom_workspace_background and saved is None:
            self._backdrop_footer.setText("Custom backdrop is enabled — drop an image to apply it.")
        elif saved is not None and not self._settings.custom_workspace_background:
            self._backdrop_footer.setText(
                "A backdrop file is saved but disabled — check the box above to apply it."
            )
        else:
            self._backdrop_footer.setText("")
        self._sync_backdrop_blend_sliders()

    def _on_custom_workspace_backdrop_toggled(self, _state: int) -> None:
        self._settings.custom_workspace_background = bool(self._custom_backdrop_chk.isChecked())
        self._persist("Workbench backdrop preference saved.")
        self._refresh_workspace_backdrop_ui()
        self.workspaceBackdropChanged.emit()

    def _on_workspace_backdrop_dropped(self, src: Path) -> None:
        try:
            ext = src.suffix.lower()
            if ext not in _ALLOWED_WALLPAPER_SUFFIX:
                self._status.setText("Backdrop: use PNG, JPEG, or WebP.")
                return
            self._state_dir.mkdir(parents=True, exist_ok=True)
            for name in workspace_background_reserved_names():
                (self._state_dir / name).unlink(missing_ok=True)
            dest = self._state_dir / f"{_WORKSPACE_BG_PREFIX}{ext}"
            shutil.copy2(src, dest)
            self._settings.workspace_background_asset = dest.name
            self._settings.custom_workspace_background = True
            self._custom_backdrop_chk.blockSignals(True)
            self._custom_backdrop_chk.setChecked(True)
            self._custom_backdrop_chk.blockSignals(False)
            save_cvops_settings(self._settings_path, self._settings)
            self._refresh_workspace_backdrop_ui()
            self.workspaceBackdropChanged.emit()
            self._status.setText("Workbench backdrop saved.")
        except Exception as exc:
            self._status.setText(f"Backdrop save failed: {exc}")

    def _on_workspace_backdrop_removed(self) -> None:
        try:
            for name in workspace_background_reserved_names():
                (self._state_dir / name).unlink(missing_ok=True)
            self._settings.workspace_background_asset = ""
            self._settings.custom_workspace_background = False
            self._custom_backdrop_chk.blockSignals(True)
            self._custom_backdrop_chk.setChecked(False)
            self._custom_backdrop_chk.blockSignals(False)
            save_cvops_settings(self._settings_path, self._settings)
            self._refresh_workspace_backdrop_ui()
            self.workspaceBackdropChanged.emit()
            self._status.setText("Workbench backdrop cleared.")
        except Exception as exc:
            self._status.setText(f"Remove backdrop failed: {exc}")

    def _on_color_changed(self, value: str) -> None:
        value = str(value or "default").strip().lower()
        if value not in COLOR_SCHEME_CHOICES:
            return
        self._settings.color_scheme = value
        self._persist("Color scheme saved.")
        self.colorSchemeChanged.emit(value)

    def _pick_color(self, *, current: str, title: str) -> str:
        initial = QColor(_normalize_optional_color(current) or "#C5FF46")
        chosen = QColorDialog.getColor(initial, self, title)
        if not chosen.isValid():
            return ""
        return chosen.name(QColor.NameFormat.HexRgb).upper()

    def _set_theme_color_override(self, field: str, value: str, edit: QLineEdit, label: str) -> None:
        normalized = _normalize_optional_color(value)
        raw = str(value or "").strip()
        if raw and not normalized:
            edit.blockSignals(True)
            edit.setText(str(getattr(self._settings, field, "") or ""))
            edit.blockSignals(False)
            self._apply_color_edit_style(edit, edit.text())
            self._status.setText(f"{label}: use #C5FF46, C5FF46, rgb(197,255,70), or 197,255,70.")
            return
        if str(getattr(self._settings, field, "") or "") == normalized:
            edit.blockSignals(True)
            edit.setText(normalized)
            edit.blockSignals(False)
            self._apply_color_edit_style(edit, normalized)
            return
        setattr(self._settings, field, normalized)
        edit.blockSignals(True)
        edit.setText(normalized)
        edit.blockSignals(False)
        self._apply_color_edit_style(edit, normalized)
        self._persist(f"{label} saved." if normalized else f"{label} reset to theme default.")
        self.themeOverridesChanged.emit()
        if field.startswith("title_"):
            self.titleStyleChanged.emit()

    def _on_theme_color_edited(self, field: str, label: str) -> None:
        edit = self._theme_color_edits.get(field)
        if edit is None:
            return
        self._set_theme_color_override(field, edit.text(), edit, f"{label} color")

    def _pick_theme_color(self, field: str, label: str) -> None:
        edit = self._theme_color_edits.get(field)
        if edit is None:
            return
        color = self._pick_color(
            current=str(getattr(self._settings, field, "") or ""),
            title=f"Choose {label.lower()} color",
        )
        if color:
            self._set_theme_color_override(field, color, edit, f"{label} color")

    def _reset_theme_color(self, field: str, label: str) -> None:
        edit = self._theme_color_edits.get(field)
        if edit is None:
            return
        self._set_theme_color_override(field, "", edit, f"{label} color")

    def _on_title_text_color_edited(self) -> None:
        self._on_theme_color_edited("title_text_color", "Title text")

    def _on_title_background_color_edited(self) -> None:
        self._on_theme_color_edited("title_background_color", "Title background")

    def _pick_title_text_color(self) -> None:
        self._pick_theme_color("title_text_color", "Title text")

    def _pick_title_background_color(self) -> None:
        self._pick_theme_color("title_background_color", "Title background")

    def _reset_title_text_color(self) -> None:
        self._reset_theme_color("title_text_color", "Title text")

    def _reset_title_background_color(self) -> None:
        self._reset_theme_color("title_background_color", "Title background")

    def _on_button_shape_changed(self, _index: int) -> None:
        value = normalize_cvops_button_shape(self._button_shape_combo.currentData())
        self._settings.button_shape = value
        self._persist("Button shape saved.")
        self.buttonShapeChanged.emit(value)

    def _on_ui_scale_changed(self, value: int) -> None:
        pct = normalize_ui_scale_pct(value)
        self._sync_ui_scale_label(pct)
        self._settings.ui_scale_pct = pct
        self._persist("UI scale saved.")
        self.uiScaleChanged.emit(pct)

    def _on_time_format_changed(self, _index: int) -> None:
        value = normalize_time_format(self._time_format_combo.currentData())
        self._settings.time_format = value
        self._persist("Time format saved.")
        self.timeFormatChanged.emit(value)

    def _on_event_pulse_toggled(self, _state: int) -> None:
        visible = bool(self._event_pulse_chk.isChecked())
        self._settings.show_event_pulse = visible
        self._persist("Notification bar preference saved.")
        self.eventPulseVisibilityChanged.emit(visible)

    def _on_dashboard_settings_changed(self, *_args: Any) -> None:
        self._settings.auto_start_dashboard = False
        self._settings.dashboard_port = int(self._dashboard_port_spin.value())
        self._persist("Dashboard settings saved.")
        self.dashboardSettingsChanged.emit(
            False,
            self._settings.dashboard_port,
        )

    def _on_poll_intervals_changed(self, *_args: Any) -> None:
        self._settings.health_poll_ms = self._spin_ms(self._health_spin)
        self._settings.dashboard_poll_ms = self._spin_ms(self._dashboard_spin)
        self._persist("Polling settings saved.")
        self.pollIntervalsChanged.emit(
            self._settings.health_poll_ms,
            self._settings.gallery_poll_ms,
            self._settings.dashboard_poll_ms,
        )

    def _persist(self, message: str) -> None:
        try:
            save_cvops_settings(self._settings_path, self._settings)
            self._status.setText(message)
        except Exception as exc:
            self._status.setText(f"Failed to save settings: {exc}")

    def _reveal_path(self, path: Path) -> None:
        target = Path(path)
        if not target.exists() and target.parent.exists():
            target = target.parent
        try:
            reveal_in_file_manager(target)
        except Exception as exc:
            self._status.setText(f"Failed to reveal path: {exc}")
