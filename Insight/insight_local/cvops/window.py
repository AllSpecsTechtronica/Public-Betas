from __future__ import annotations

import insight_local.cvops.ui.patch_parallelogram_buttons  # noqa: F401
from insight_local.cvops.ui.patch_parallelogram_buttons import set_cvops_button_shape

import argparse
import base64
import importlib.util
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from PyQt6.QtCore import QEvent, QObject, QPoint, QPointF, QRect, QRectF, QThread, QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QDragEnterEvent,
    QDropEvent,
    QFont,
    QKeySequence,
    QLinearGradient,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStyleFactory,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..config import COLOR_SCHEME_CHOICES, ROOT_DIR
from ..ui.theme import (
    apply_text_palette,
    configure_color_scheme,
    configure_text_mode,
    current_color_scheme,
    get_global_stylesheet,
)
from . import model_feedback_store
# Path constants come from the dependency-free paths module so importing window
# does NOT pull in service.py (fastapi + the mlops/torch stack). CvOpsServerHandle
# is imported lazily on a background thread (see _ServerBootWorker) so the window
# can paint immediately instead of waiting on the heavy import + app build.
from .paths import CVOPS_DB_PATH, CVOPS_STATE_DIR
from .ui.collapsible_section import CollapsibleSection
from .ui.notes_ai_keys import assistant_display_name
from .ui.catalog_panel import CatalogPanel
from .ui.cvops_theme import (
    apply_ui_scale_compaction,
    cvops_color,
    cvops_qcolor,
    get_cvops_stylesheet,
    install_cvops_chamfer_combo_style,
    install_cvops_font_substitutions,
    install_combo_popup_top_align,
    install_cvops_local_stylesheet_normalizer,
    install_selectable_labels,
    install_title_fit_filter,
    normalize_color_override,
    normalize_ui_scale_pct,
    refresh_cvops_theme_tree,
    repolish,
    resolve_ui_scale_factor,
    scale_qss_pixel_metrics,
    set_cvops_stylesheet,
)
from .ui.queue_panel import QueuePanel
from .ui.model_feedback_dialog import ModelFeedbackDialog
from .ui.backdrop_blend import blend_from_cvops_settings
from .ui.settings_panel import (
    CvOpsSettings,
    CvOpsSettingsPanel,
    load_cvops_settings,
    resolve_workspace_wallpaper_path,
    save_cvops_settings,
)
from .ui.time_format import format_timestamp, set_time_format
from .ui.memory_client import CvOpsMemoryClient
from .ui.activity_rail import ActivityRailWidget
from .ui.workbench_split_host import WorkbenchSplitHost, WorkbenchSplitRefs
from .ui.algo_catalog import reveal_in_finder
from .ui.schema_fix_dialog import SchemaFixDialog
from .ui.storage_diagnosis import (
    build_storage_diagnosis,
    format_storage_diagnosis,
    looks_like_storage_error,
)
from .ui.connection_overlay import ConnectionOverlay
from .ui.dropdown_pane_stack import DropdownPaneStack
from .ui.event_pulse_widget import EventPulseWidget
from .ui.notification_cards import (
    HeartbeatNotificationGate,
    NotificationCardTray,
    should_show_notification_card,
)
from .ui.notifications_panel import NotificationsPanel
# [LAZY-LOADED] Heavy panel modules are imported inside their factory methods
# to avoid blocking the UI thread at startup:
#   DatasetEditorDialog, DatasetPanel, TestRangePanel, VideoTestPanel,
#   SubmitPanel, ThreeDPanel, ScrapePanel, CellsPanel, DatabaseGodViewPanel,
#   DataVizHub, NotesPanel, OntologyPanel, DashboardOverviewWidget

_SHELL_CACHE_VERSION = 1
_SHELL_CACHE_MAX_TRAINING_EVENTS = 900


def _mark_boot(name: str, *, once: bool = False, **fields: object) -> None:
    try:
        from insight_local.cvops.__main__ import _boot_mark  # noqa: PLC0415
        _boot_mark(name, once=once, **fields)
    except Exception:
        pass


class _CvOpsWorkbenchRoot(QWidget):
    """Workbench root that paints the user-selected wallpaper behind all chrome."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._wallpaper = QPixmap()
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)

    def set_wallpaper(self, path: Optional[Path]) -> None:
        pixmap = QPixmap()
        if path is not None:
            try:
                resolved = Path(path).expanduser().resolve()
            except Exception:
                resolved = Path(path).expanduser()
            if resolved.is_file():
                pixmap = QPixmap(str(resolved))
        self._wallpaper = pixmap if not pixmap.isNull() else QPixmap()
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        if self._wallpaper.isNull() or self.width() <= 0 or self.height() <= 0:
            return
        scaled = self._wallpaper.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatioByExpanding,
            Qt.TransformationMode.SmoothTransformation,
        )
        if scaled.isNull():
            return
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.drawPixmap(x, y, scaled)


class _ScrollTabPage(QScrollArea):
    """Viewport wrapper that lets tab/page content grow vertically without compression."""

    def __init__(self, page: QWidget, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("cvOpsScrollPage")
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setWidgetResizable(True)
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        host = QWidget()
        host.setObjectName("cvOpsScrollHost")
        host_layout = QVBoxLayout(host)
        host_layout.setContentsMargins(0, 0, 0, 0)
        host_layout.setSpacing(0)
        host_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        host_layout.addWidget(page)
        self.setWidget(host)


class _DeferredPage(QWidget):
    """Container that lazy-builds its real content on first show.

    Pass a factory callable that returns the real QWidget. The factory is
    invoked exactly once — on the first showEvent — so heavy module imports
    and widget construction never run until the user actually opens that tab.

    For panels whose factory is itself slow (heavy ML imports, filesystem I/O,
    GPU detection), pass an optional ``preload_fn``.  When provided:
      1. ``preload_fn()`` runs on the main thread, deferred one extra event-loop
         tick so the "Loading..." placeholder paints first.
      2. Its return value is forwarded as the sole argument to ``factory()``.

    Preloads run on the main thread on purpose: ``preload_fn`` bodies import
    Python modules (cv2, matplotlib, torch, mlops, the ui.* panels), and
    CPython serializes imports on per-module locks. Running those imports on a
    background QThread concurrently with the in-process backend service's own
    lazy imports (it shares this process) can deadlock on the import lock and
    freeze the whole UI — exactly the hang seen when opening a tab mid-training.
    A background QThread also holds the GIL throughout an import, so it would
    not keep the main thread responsive anyway; the only thing it added was the
    deadlock risk.
    """

    def __init__(self, factory, preload_fn=None, parent=None, load_label: str = "panel"):
        super().__init__(parent)
        self._factory = factory
        self._preload_fn = preload_fn
        self._load_label = str(load_label or "panel")
        self._build_started = False
        lyt = QVBoxLayout(self)
        lyt.setContentsMargins(0, 0, 0, 0)
        lyt.setSpacing(0)
        self._placeholder: Optional[QWidget] = QWidget(self)
        self._placeholder.setObjectName("cvOpsDeferredPlaceholder")
        ph_layout = QVBoxLayout(self._placeholder)
        ph_layout.setContentsMargins(32, 28, 32, 28)
        ph_layout.setSpacing(10)
        ph_layout.addStretch(1)
        self._placeholder_title = QLabel("", self._placeholder)
        self._placeholder_title.setObjectName("cvOpsDeferredPlaceholderTitle")
        self._placeholder_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_title.setWordWrap(True)
        ph_layout.addWidget(self._placeholder_title)
        self._placeholder_status = QLabel("", self._placeholder)
        self._placeholder_status.setObjectName("cvOpsDeferredPlaceholderStatus")
        self._placeholder_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder_status.setWordWrap(True)
        ph_layout.addWidget(self._placeholder_status)
        self._placeholder_progress = QProgressBar(self._placeholder)
        self._placeholder_progress.setObjectName("cvOpsDeferredPlaceholderBar")
        self._placeholder_progress.setRange(0, 0)
        self._placeholder_progress.setTextVisible(False)
        self._placeholder_progress.setFixedHeight(10)
        set_cvops_stylesheet(
            self._placeholder_progress,
            lambda: (
                f"QProgressBar#cvOpsDeferredPlaceholderBar {{ "
                f"border: 1px solid {cvops_color('line_light')}; "
                f"background: {cvops_color('panel')}; "
                f"border-radius: 2px; }}"
                f"QProgressBar#cvOpsDeferredPlaceholderBar::chunk {{ "
                f"background: {cvops_color('line_bright')}; }}"
            ),
        )
        ph_layout.addWidget(self._placeholder_progress)
        ph_layout.addStretch(1)
        lyt.addWidget(self._placeholder, stretch=1)
        self._set_placeholder_state(
            f"Preparing {self._load_label}...",
            "This tab can finish loading in the background while you move elsewhere in CvOps.",
        )

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self.ensure_loading()

    def ensure_loading(self) -> None:
        if self._build_started:
            return
        self._build_started = True
        # Defer briefly so the placeholder paints and a quick click-away can
        # cancel the off-screen build before heavy imports start.
        QTimer.singleShot(120, self._do_build)

    def _do_build(self) -> None:
        if self._defer_build_until_visible():
            return
        if self._preload_fn is not None:
            self._start_preload()
        else:
            self._build_direct()

    def _defer_build_until_visible(self) -> bool:
        if self.isVisible():
            return False
        self._build_started = False
        self._set_placeholder_state(
            f"Preparing {self._load_label}...",
            "Open this page again to continue loading it.",
        )
        return True

    def _build_direct(self) -> None:
        self._set_placeholder_state(
            f"Loading {self._load_label}...",
            "Building this page now. You can switch tabs and come back when it finishes.",
        )
        try:
            real = self._factory()
        except Exception as exc:
            self._on_build_failed(str(exc))
            import traceback
            traceback.print_exc()
            return
        self._swap_in(real)

    def _start_preload(self) -> None:
        self._set_placeholder_state(
            f"Loading {self._load_label}...",
            "Preparing heavy resources for this tab. You can leave this tab and return later.",
        )
        # Defer briefly so the "Loading..." text paints and a quick click-away
        # can cancel before the main-thread preload starts. See the class
        # docstring for why the preload must not run on a background thread.
        QTimer.singleShot(120, self._run_preload)

    def _run_preload(self) -> None:
        if self._defer_build_until_visible():
            return
        try:
            preloaded = self._preload_fn()
        except Exception as exc:
            self._on_build_failed(str(exc))
            import traceback
            traceback.print_exc()
            return
        self._set_placeholder_state(
            f"Finalizing {self._load_label}...",
            "Resources ready. Attaching the page layout now.",
        )
        try:
            real = self._factory(preloaded)
        except Exception as exc:
            self._on_build_failed(str(exc))
            import traceback
            traceback.print_exc()
            return
        self._swap_in(real)

    def _on_build_failed(self, error: str) -> None:
        self._set_placeholder_state(
            f"{self._load_label} failed to load.",
            str(error or "Unknown error"),
            failed=True,
        )

    def _set_placeholder_state(self, title: str, detail: str, *, failed: bool = False) -> None:
        if self._placeholder is None:
            return
        self._placeholder_title.setText(str(title or ""))
        self._placeholder_status.setText(str(detail or ""))
        self._placeholder_progress.setRange(0, 1 if failed else 0)
        if failed:
            self._placeholder_progress.setValue(0)

    def _swap_in(self, real) -> None:
        ph = self._placeholder
        if ph is not None:
            self._placeholder = None
            ph.setParent(None)
        if real is not None:
            real.setParent(self)
            self.layout().addWidget(real, stretch=1)


class _DeferredResultPanel(QWidget):
    """Lightweight result pane that imports the real ResultPanel on first result use."""

    flagRequested = pyqtSignal(object)
    activeContextChanged = pyqtSignal(bool)

    def __init__(
        self,
        *,
        base_url: str,
        http_get: Callable[[str], dict[str, Any]],
        http_get_text: Callable[[str], str],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = base_url
        self._http_get = http_get
        self._http_get_text = http_get_text
        self._real: Optional[QWidget] = None
        self._active_context = False
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)
        self._placeholder = QLabel("No result selected.", self)
        self._placeholder.setObjectName("cvOpsDeferredPlaceholderTitle")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._layout.addStretch(1)
        self._layout.addWidget(self._placeholder)
        self._layout.addStretch(1)

    def _ensure_real(self) -> QWidget:
        if self._real is not None:
            return self._real
        _mark_boot("result_panel_import_start", once=True)
        from .ui.result_panel import ResultPanel  # noqa: PLC0415
        _mark_boot("result_panel_import_done", once=True)
        real = ResultPanel(
            base_url=self._base_url,
            http_get=self._http_get,
            http_get_text=self._http_get_text,
        )
        real.flagRequested.connect(self.flagRequested.emit)
        real.activeContextChanged.connect(self._on_real_active_context_changed)
        artifacts = getattr(real, "_artifacts_panel", None)
        if artifacts is not None:
            artifacts.flagRequested.connect(self.flagRequested.emit)
        while self._layout.count():
            item = self._layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addWidget(real, stretch=1)
        self._real = real
        return real

    def _set_active_context(self, active: bool) -> None:
        active = bool(active)
        if self._active_context == active:
            return
        self._active_context = active
        self.activeContextChanged.emit(active)

    def _on_real_active_context_changed(self, active: bool) -> None:
        self._set_active_context(active)

    def has_active_context(self) -> bool:
        if self._real is not None:
            checker = getattr(self._real, "has_active_context", None)
            if callable(checker):
                return bool(checker())
        return bool(self._active_context)

    def show_message(self, text: str) -> None:
        msg = str(text or "").strip() or "No result selected."
        if self._real is not None:
            getattr(self._real, "show_message")(msg)
            return
        self._placeholder.setText(msg)
        self._set_active_context(msg != "No result selected.")

    def clear(self) -> None:
        if self._real is not None:
            getattr(self._real, "clear")()
            return
        self._placeholder.setText("No result selected.")
        self._set_active_context(False)

    def select_job(self, job_id: str) -> None:
        getattr(self._ensure_real(), "select_job")(job_id)

    def apply_result(self, job_id: str, result: dict[str, Any]) -> None:
        getattr(self._ensure_real(), "apply_result")(job_id, result)

    def refresh_subroutine_models(self) -> None:
        if self._real is not None:
            getattr(self._real, "refresh_subroutine_models")()

    def refresh_responsive_layout(self) -> None:
        if self._real is not None:
            getattr(self._real, "refresh_responsive_layout")()

    def refresh_theme_styles(self) -> None:
        if self._real is not None:
            refresher = getattr(self._real, "refresh_theme_styles", None)
            if callable(refresher):
                refresher()


class _LazyLineagePanel(QWidget):
    """Placeholder lineage pane that builds the real catalog only on lineage use."""

    errorRaised = pyqtSignal(str)
    entitySelected = pyqtSignal(str, str)

    def __init__(
        self,
        *,
        http_get: Callable[..., Any],
        http_post: Callable[..., Any],
        http_delete: Callable[..., Any],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._http_get = http_get
        self._http_post = http_post
        self._http_delete = http_delete
        self._real: Optional[QWidget] = None
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 10, 10, 10)
        self._layout.setSpacing(8)
        self._placeholder = QLabel("Lineage catalog not loaded.", self)
        self._placeholder.setObjectName("cvOpsDeferredPlaceholderTitle")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._load_btn = QPushButton("Load Lineage", self)
        self._load_btn.clicked.connect(lambda _checked=False: self._ensure_real())
        self._layout.addStretch(1)
        self._layout.addWidget(self._placeholder)
        self._layout.addWidget(self._load_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._layout.addStretch(1)

    def _ensure_real(self) -> QWidget:
        if self._real is not None:
            return self._real
        _mark_boot("lineage_panel_import_start", once=True)
        from .ui.lineage_panel import LineageCatalogPanel  # noqa: PLC0415
        _mark_boot("lineage_panel_import_done", once=True)
        real = LineageCatalogPanel(
            http_get=self._http_get,
            http_post=self._http_post,
            http_delete=self._http_delete,
        )
        real.errorRaised.connect(self.errorRaised.emit)
        real.entitySelected.connect(self.entitySelected.emit)
        while self._layout.count():
            item = self._layout.takeAt(0)
            old = item.widget()
            if old is not None:
                old.setParent(None)
                old.deleteLater()
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(0)
        self._layout.addWidget(real, stretch=1)
        self._real = real
        return real

    def reload(self) -> None:
        getattr(self._ensure_real(), "reload")()

    def select_lineage(self, lineage_id: str) -> None:
        getattr(self._ensure_real(), "select_lineage")(lineage_id)

    def select_lineage_for_scenario(self, scenario: str) -> None:
        getattr(self._ensure_real(), "select_lineage_for_scenario")(scenario)

    def refresh_theme_styles(self) -> None:
        if self._real is not None:
            refresher = getattr(self._real, "refresh_theme_styles", None)
            if callable(refresher):
                refresher()


class _ServiceStripLabel(QLabel):
    """Service line beside the event pulse: hides when empty so the strip stays one tight row."""

    def __init__(self, text: str = "", parent: Optional[QWidget] = None) -> None:
        super().__init__(str(text), parent)
        self._sync_vis()

    def setText(self, text: str) -> None:  # type: ignore[override]
        super().setText(text)
        self._sync_vis()

    def _sync_vis(self) -> None:
        self.setVisible(bool(str(self.text() or "").strip()))


class _CvOpsLaunchSplash(QWidget):
    """Frameless launch splash with the same machine-panel language as the app chrome."""

    _RUN_BLUES = (
        "#2bd9ff",
        "#22b8f0",
        "#1d8ed8",
        "#55d9ff",
        "#0a8fa8",
        "#5a9fd6",
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(
            parent,
            Qt.WindowType.SplashScreen
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setObjectName("cvOpsLaunchSplash")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFixedSize(420, 260)
        self._tick = 0
        self._faces = [QColor(c) for c in self._RUN_BLUES[:3]]
        self._timer = QTimer(self)
        self._timer.setInterval(120)
        self._timer.timeout.connect(self._advance)

    def start(self) -> None:
        self._center_on_screen()
        self._timer.start()
        self.show()
        self.raise_()
        self.activateWindow()

    def finish(self, window: QWidget) -> None:
        self._timer.stop()
        self.hide()
        self.deleteLater()
        try:
            window.raise_()
            window.activateWindow()
        except Exception:
            pass

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        geo = screen.availableGeometry() if screen is not None else QRect(0, 0, 1180, 800)
        self.move(
            geo.x() + max(0, (geo.width() - self.width()) // 2),
            geo.y() + max(0, (geo.height() - self.height()) // 2),
        )

    def _advance(self) -> None:
        self._tick += 1
        # Training nodes in the ecosystem pulse through active blue/cyan hues.
        self._faces = [QColor(random.choice(self._RUN_BLUES)) for _ in range(3)]
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(1, 1, self.width() - 2, self.height() - 2)

        panel = QLinearGradient(rect.topLeft(), rect.bottomRight())
        panel.setColorAt(0.0, cvops_qcolor("bg_panel", 242))
        panel.setColorAt(0.56, cvops_qcolor("bg_void", 246))
        panel.setColorAt(1.0, cvops_qcolor("bg_graphite", 238))
        painter.setBrush(QBrush(panel))
        painter.setPen(QPen(cvops_qcolor("line_light", 230), 1.25))
        painter.drawRect(rect)

        painter.setPen(QPen(cvops_qcolor("line_med", 160), 1.0))
        painter.drawLine(18, 38, self.width() - 18, 38)
        painter.drawLine(18, self.height() - 36, self.width() - 18, self.height() - 36)

        self._draw_cube(painter)
        self._draw_text(painter)
        painter.end()

    def _draw_cube(self, painter: QPainter) -> None:
        cx = self.width() / 2.0
        cy = 96.0
        pulse = (self._tick % 16) / 16.0
        side = 122.0 + (2.0 if pulse < 0.5 else 0.0)
        x0 = cx - side / 2.0
        y0 = cy - side / 2.0
        outer = QRectF(x0, y0, side, side)

        outer_fill = QLinearGradient(outer.topLeft(), outer.bottomRight())
        outer_fill.setColorAt(0.0, QColor("#041723"))
        outer_fill.setColorAt(1.0, QColor("#0b2b3a"))
        painter.setPen(QPen(QColor("#8ceaff"), 1.2))
        painter.setBrush(QBrush(outer_fill))
        painter.drawRect(outer)

        cells = 6
        gap = 3.0
        inner_margin = 8.0
        inner_side = side - inner_margin * 2.0
        cell_side = (inner_side - gap * (cells - 1)) / cells
        x_cell = x0 + inner_margin
        y_cell = y0 + inner_margin
        painter.setPen(Qt.PenStyle.NoPen)
        for row in range(cells):
            for col in range(cells):
                r = random.random()
                if r > 0.7:
                    color = QColor(random.choice(self._RUN_BLUES))
                    color.setAlpha(220)
                elif r > 0.34:
                    color = QColor("#13465e")
                    color.setAlpha(180)
                else:
                    color = QColor("#0a2735")
                    color.setAlpha(148)
                rx = x_cell + col * (cell_side + gap)
                ry = y_cell + row * (cell_side + gap)
                painter.setBrush(QBrush(color))
                painter.drawRect(QRectF(rx, ry, cell_side, cell_side))

        glow = QColor(random.choice(self._RUN_BLUES))
        glow.setAlpha(72)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(glow, 3.5))
        painter.drawRect(outer.adjusted(-2, -2, 2, 2))

    def _draw_text(self, painter: QPainter) -> None:
        title_font = QFont(self.font())
        title_font.setFamily("IBM Plex Mono")
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        painter.setFont(title_font)
        painter.setPen(cvops_qcolor("text_bright"))
        painter.drawText(QRectF(0, 160, self.width(), 28), Qt.AlignmentFlag.AlignCenter, "test: loading")

        sub_font = QFont(self.font())
        sub_font.setFamily("IBM Plex Mono")
        sub_font.setPointSize(8)
        sub_font.setLetterSpacing(QFont.SpacingType.PercentageSpacing, 118)
        painter.setFont(sub_font)
        painter.setPen(QColor(cvops_color("accent_active")))
        phase = "." * ((self._tick % 3) + 1)
        painter.drawText(
            QRectF(0, 194, self.width(), 18),
            Qt.AlignmentFlag.AlignCenter,
            f"CVOPS BOOTSTRAP / TRAIN-RUN VISUAL{phase}",
        )


class _TopNavClickRecover(QObject):
    """Application-level event filter that re-dispatches mouse events that
    land inside the top-nav's global rect.

    Background: QtWebEngineView on macOS renders into a layer-backed NSView
    whose CALayer ends up on top of every other widget in the window
    regardless of Qt's z-order, so clicks "through" the web view to the
    top-nav above it never arrive at the buttons — they're swallowed by the
    web view. Setting ``WA_NativeWindow`` / ``raise_()`` is not enough on
    layer-backed builds. This filter intercepts mouse presses/releases at
    the QApplication level, checks whether the click position falls inside
    the top-nav's screen rectangle, and posts the event to the correct child
    widget directly, then consumes the original so the web view never sees
    it.
    """

    def __init__(self, top_nav: QWidget) -> None:
        super().__init__(top_nav)
        self._top_nav = top_nav
        self._recovering = False  # re-entrancy guard

    def _global_rect(self) -> QRect:
        if self._top_nav is None or not self._top_nav.isVisible():
            return QRect()
        try:
            origin = self._top_nav.mapToGlobal(QPoint(0, 0))
        except Exception:
            return QRect()
        return QRect(origin, self._top_nav.size())

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # type: ignore[override]
        if self._recovering:
            return False
        et = event.type()
        if et not in (
            QEvent.Type.MouseButtonPress,
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.MouseButtonDblClick,
        ):
            return False
        if not isinstance(event, QMouseEvent):
            return False
        rect = self._global_rect()
        if rect.isEmpty():
            return False
        try:
            gp = event.globalPosition().toPoint()
        except Exception:
            return False
        if not rect.contains(gp):
            return False
        # Find the deepest child under the cursor; fall back to the strip itself.
        local_in_strip = self._top_nav.mapFromGlobal(gp)
        target = self._top_nav.childAt(local_in_strip) or self._top_nav
        if watched is target:
            # The event is already on its way to the right widget; don't loop.
            return False
        local_pos = target.mapFromGlobal(gp)
        self._recovering = True
        try:
            forwarded = QMouseEvent(
                et,
                QPointF(local_pos),
                QPointF(gp),
                event.button(),
                event.buttons(),
                event.modifiers(),
            )
            QApplication.sendEvent(target, forwarded)
        finally:
            self._recovering = False
        # Swallow the original event so the web view (or anything below) never
        # gets to act on it.
        return True


class _ServerBootWorker(QThread):
    """Import + construct + start the backend server off the UI thread.

    Importing service.py (fastapi + the mlops stack) and building the FastAPI app
    is the heaviest part of startup. Doing it here lets the window paint its shell
    immediately; the window binds the live service when `ready` fires on the main
    thread. The handle is passed back as an opaque object."""

    ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, host: str, port: int, db_path, parent=None) -> None:
        super().__init__(parent)
        self._host = host
        self._port = port
        self._db_path = db_path

    def run(self) -> None:
        try:
            from .service import CvOpsServerHandle  # lazy: heavy import, off UI thread
            handle = CvOpsServerHandle(host=self._host, port=self._port, db_path=self._db_path)
            handle.start()
            self.ready.emit(handle)
        except Exception as exc:
            self.failed.emit(str(exc))


class _HttpFetchThread(QThread):
    """Run any blocking HTTP callable on a daemon thread; emit result or error on the main thread."""

    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fetch_fn, parent=None):
        super().__init__(parent)
        self._fetch_fn = fetch_fn
        self.setTerminationEnabled(True)

    def run(self) -> None:
        try:
            result = self._fetch_fn()
            self.done.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class _EcosystemWarmupWorker(QThread):
    """Warm the ontology graph cache without constructing Qt web widgets."""

    done = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        base_url: str,
        *,
        direct_loader: Optional[Callable[[str], dict[str, Any]]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = str(base_url).rstrip("/")
        self._direct_loader = direct_loader

    def run(self) -> None:
        try:
            core = self._fetch_layer("core")
            full = self._fetch_layer("full")
            deadline = time.time() + 8.0
            while self._is_pending(full) and time.time() < deadline:
                self.msleep(350)
                full = self._fetch_layer("full")
            self.done.emit({"core": core, "full": full})
        except Exception as exc:
            self.failed.emit(str(exc))

    @staticmethod
    def _is_pending(payload: dict[str, Any]) -> bool:
        meta = payload.get("cache") if isinstance(payload.get("cache"), dict) else {}
        return bool(meta.get("pending"))

    def _fetch_layer(self, layer: str) -> dict[str, Any]:
        if self._direct_loader is not None:
            payload = self._direct_loader(layer)
            return payload if isinstance(payload, dict) else {}
        url = f"{self._base_url}/ontology/graph?layer={urllib.parse.quote(layer)}"
        with urllib.request.urlopen(url, timeout=10.0) as resp:
            raw = resp.read().decode("utf-8")
        payload = json.loads(raw) if raw else {}
        return payload if isinstance(payload, dict) else {}


class _WsResyncWorker(QThread):
    """Fetch websocket reconnect state without blocking the Qt UI thread."""

    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        base_url: str,
        *,
        direct_resync: Optional[Callable[[], dict[str, Any]]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._base_url = str(base_url).rstrip("/")
        self._direct_resync = direct_resync

    def run(self) -> None:
        if self._direct_resync is not None:
            try:
                payload = self._direct_resync()
                self.completed.emit(payload if isinstance(payload, dict) else {})
            except Exception as exc:
                self.failed.emit(str(exc))
            return

        errors: list[str] = []
        jobs: list[dict[str, Any]] = []
        training_events: list[dict[str, Any]] = []
        scenarios: list[dict[str, Any]] = []

        try:
            payload = self._http_json("GET", "/jobs", timeout=4.0)
            jobs = [j for j in list(payload.get("jobs") or []) if isinstance(j, dict)]
        except Exception as exc:
            errors.append(f"/jobs: {exc}")

        for job in [j for j in jobs if str(j.get("job_type") or "") == "train"][:18]:
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            try:
                payload = self._http_json("GET", f"/jobs/{job_id}/training_progress", timeout=2.5)
            except Exception as exc:
                errors.append(f"/jobs/{job_id}/training_progress: {exc}")
                continue
            events = payload.get("events") if isinstance(payload, dict) else []
            if not isinstance(events, list):
                continue
            scenario = str(job.get("scenario") or "")
            for event in events:
                if not isinstance(event, dict):
                    continue
                merged = dict(event)
                if not merged.get("scenario"):
                    merged["scenario"] = scenario
                if not merged.get("job_id"):
                    merged["job_id"] = job_id
                training_events.append(merged)

        try:
            payload = self._http_json("GET", "/scenarios", timeout=6.0)
            scenarios = [s for s in list(payload.get("scenarios") or []) if isinstance(s, dict)]
        except Exception as exc:
            errors.append(f"/scenarios: {exc}")

        if not jobs and not scenarios and errors:
            self.failed.emit("; ".join(errors[:3]))
            return
        self.completed.emit(
            {
                "jobs": jobs,
                "training_events": training_events,
                "scenarios": scenarios,
                "errors": errors[:8],
            }
        )

    def _http_json(
        self,
        method: str,
        path: str,
        *,
        timeout: float,
    ) -> dict[str, Any]:
        req = urllib.request.Request(
            self._base_url + path,
            method=method.upper(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


class _DatasetImportDropZone(QFrame):
    """Drag-and-drop target for importing an already-made dataset folder.

    This is the expected ingestion path — dropping a folder never opens the slow
    native enumeration the old multi-select picker did, so it can't freeze. The
    "Import dataset" button stays as a fallback that uses the fast native OS
    chooser only when a picker is actually wanted."""

    foldersDropped = pyqtSignal(list)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("datasetImportDropZone")
        self.setAcceptDrops(True)
        self.setProperty("state", "idle")
        self.setMinimumHeight(46)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setStyleSheet(
            "#datasetImportDropZone {"
            " border: 1px dashed rgba(10,143,168,0.55); border-radius: 6px;"
            " background: rgba(10,143,168,0.05); }"
            "#datasetImportDropZone[state=\"dragover\"] {"
            " border: 1px solid rgba(10,143,168,0.95);"
            " background: rgba(10,143,168,0.18); }"
            "#datasetImportDropZone[state=\"busy\"] {"
            " border: 1px solid rgba(10,143,168,0.75);"
            " background: rgba(10,143,168,0.12); }"
            "#datasetImportDropZone QLabel { border: none; background: transparent;"
            " color: rgba(180,210,220,0.85); font-size: 11px; }"
        )
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        self._default_text = "Drag a dataset folder here to import  —  YOLO / ImageFolder / audio / CSV"
        self._label = QLabel(self._default_text)
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

    @staticmethod
    def _folders(event) -> list[str]:
        md = event.mimeData()
        out: list[str] = []
        if md is None or not md.hasUrls():
            return out
        for url in md.urls():
            if url.isLocalFile():
                p = Path(url.toLocalFile())
                if p.is_dir():
                    out.append(str(p))
        return out

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # type: ignore[override]
        if self._folders(event):
            self.setProperty("state", "dragover")
            self._refresh_style()
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:  # type: ignore[override]
        self.setProperty("state", "idle")
        self._refresh_style()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent) -> None:  # type: ignore[override]
        folders = self._folders(event)
        self.setProperty("state", "idle")
        self._refresh_style()
        if folders:
            self.foldersDropped.emit(folders)
            event.acceptProposedAction()

    def set_busy(self, busy: bool, text: str = "") -> None:
        self.setEnabled(not busy)
        self.setProperty("state", "busy" if busy else "idle")
        self._label.setText(str(text or self._default_text))
        self._refresh_style()

    def _refresh_style(self) -> None:
        self.style().unpolish(self)
        self.style().polish(self)


class CvOpsWindow(QMainWindow):
    def __init__(self, *, host: str, port: int, settings: Optional[CvOpsSettings] = None) -> None:
        _bprint("CvOpsWindow.__init__ — super().__init__")
        super().__init__()
        self._settings_path = CVOPS_STATE_DIR / "settings.json"
        self._shell_cache_path = CVOPS_STATE_DIR / "shell_cache.json"
        self._cvops_settings = settings or load_cvops_settings(self._settings_path)
        set_time_format(self._cvops_settings.time_format)
        set_cvops_button_shape(self._cvops_settings.button_shape)
        self._cvops_settings.ui_scale_pct = normalize_ui_scale_pct(
            os.environ.get("INSIGHT_CVOPS_UI_SCALE", self._cvops_settings.ui_scale_pct)
        )
        self.setObjectName("cvOpsWindow")
        self.setProperty("uiScalePct", self._cvops_settings.ui_scale_pct)
        # Make every QLabel in the entire UI text-selectable by the user.
        _app = QApplication.instance()
        if _app is not None:
            install_cvops_local_stylesheet_normalizer()
            install_selectable_labels(_app)
            install_title_fit_filter(_app)
            install_combo_popup_top_align(_app)
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/events"
        _bprint(f"booting backend server in background (http://{host}:{port})")
        # The server (service.py: fastapi + mlops/torch) boots on a background
        # thread so the window paints immediately. Until _on_server_ready fires,
        # self._server is None; direct reads fall back / no-op and the memory
        # client stays unbound. _on_server_ready binds it and resyncs live data.
        self._server = None
        self._server_boot = _ServerBootWorker(host, port, CVOPS_DB_PATH, parent=self)
        self._server_boot.ready.connect(self._on_server_ready)
        self._server_boot.failed.connect(self._on_server_boot_failed)
        self._prefer_direct_service_reads = str(
            os.environ.get("CVOPS_DIRECT_SERVICE_READS", "1")
        ).strip().lower() not in {"0", "false", "no", "off"}

        self._scenarios_cache: list[dict[str, Any]] = []
        self._shell_jobs_cache: list[dict[str, Any]] = []
        self._shell_training_events_cache: list[dict[str, Any]] = []
        self._last_event_seq = 0
        self._scenarios_refresh_failed = False
        self._result_cache: dict[str, dict[str, Any]] = {}
        self._last_error_key = ""
        self._last_error_ts = 0.0
        self._pending_error_lines: list[str] = []
        self._shown_storage_error_keys: set[str] = set()
        self._ws_resync_worker: Optional[QThread] = None
        self._ws_resync_pending = False
        self._ws_resync_last_started = 0.0
        self._suppress_next_connected_resync = False
        self._health_fetch_running = False
        self._scenarios_fetch_running = False
        self._collect_import_thread: Optional[_HttpFetchThread] = None
        self._collect_scenario_prep_thread: Optional[_HttpFetchThread] = None
        self._console_splitter: Optional[QSplitter] = None
        self._project_root = Path(__file__).resolve().parents[3]
        self._dashboard_port = int(self._cvops_settings.dashboard_port)
        self._dashboard_url = f"http://127.0.0.1:{self._dashboard_port}"
        self._dashboard_proc: Optional[subprocess.Popen] = None
        self._dashboard_status: Optional[QLabel] = None
        self._settings_panel: Optional[CvOpsSettingsPanel] = None
        self._dashboard_web_view: Optional[QWidget] = None
        self._dashboard_web_supported = False
        self._scope_dashboard_status: Optional[QLabel] = None
        self._scope_dashboard_host: Optional[QWidget] = None
        self._scope_dashboard_stack: Optional[QStackedWidget] = None
        self._scope_dashboard_launch_btn: Optional[QPushButton] = None
        self._scope_dashboard_reload_btn: Optional[QPushButton] = None
        self._scope_dashboard_open_btn: Optional[QPushButton] = None
        self._scope_dashboard_stop_btn: Optional[QPushButton] = None
        self._dashboard_overview: Optional[DashboardOverviewWidget] = None
        self._dashboard_health_summary: Optional[QLabel] = None
        self._dashboard_scenario_summary: Optional[QLabel] = None
        self._dashboard_jobs_table: Optional[QTableWidget] = None
        self._dashboard_jobs_rows: dict[str, int] = {}
        self._portal_proc: Optional[subprocess.Popen] = None
        self._portal_status: Optional[QLabel] = None
        self._portal_init_btn: Optional[QPushButton] = None
        self._portal_stop_btn: Optional[QPushButton] = None
        self._portal_loaded = False
        self._close_confirmed = False
        self._test_range_result_panel: Optional[QWidget] = None
        self._test_range_last_job_id = ""
        self._last_ui_style_scale: Optional[float] = None
        self._layout_refresh_timer = QTimer(self)
        self._layout_refresh_timer.setSingleShot(True)
        self._layout_refresh_timer.setInterval(0)
        self._layout_refresh_timer.timeout.connect(self._refresh_responsive_children)
        self._notification_tray_layout_timer = QTimer(self)
        self._notification_tray_layout_timer.setSingleShot(True)
        self._notification_tray_layout_timer.setInterval(0)
        self._notification_tray_layout_timer.timeout.connect(self._update_notification_tray_geometry)
        self._dashboard_timer = QTimer(self)
        self._dashboard_timer.setInterval(int(self._cvops_settings.dashboard_poll_ms))
        self._dashboard_timer.timeout.connect(self._refresh_dashboard_status)
        self._shell_cache_save_timer = QTimer(self)
        self._shell_cache_save_timer.setSingleShot(True)
        self._shell_cache_save_timer.setInterval(750)
        self._shell_cache_save_timer.timeout.connect(self._write_shell_cache)

        self.setWindowTitle("CV Ops")
        self.resize(1180, 800)
        # Prevent pathological tiny sizes where nested splitters and cards can no longer lay out cleanly.
        self.setMinimumSize(860, 560)

        root = _CvOpsWorkbenchRoot()
        root.setObjectName("cvOpsRoot")
        root.set_wallpaper(resolve_workspace_wallpaper_path(self._cvops_settings, CVOPS_STATE_DIR))
        self._workbench_root = root
        self.setCentralWidget(root)
        self._install_ui_scale_shortcuts()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._top_nav_row = QHBoxLayout()
        self._catalog_toggle_btn = QPushButton("Catalog")
        self._catalog_toggle_btn.setCheckable(True)
        self._catalog_toggle_btn.setProperty("navToggle", True)
        self._catalog_toggle_btn.setChecked(True)
        self._catalog_toggle_btn.setToolTip("Show or hide the Catalog pane.")
        self._catalog_toggle_btn.clicked.connect(self._on_catalog_toggle_clicked)
        self._top_nav_row.addWidget(self._catalog_toggle_btn)
        self._restore_panes_btn = QPushButton("Restore panes")
        self._restore_panes_btn.setToolTip("Reopen closed split panes.")
        self._restore_panes_btn.clicked.connect(self._restore_workbench_panes)
        self._top_nav_row.addWidget(self._restore_panes_btn)
        self._bottom_pane_toggle_btn = QPushButton("Bottom pane")
        self._bottom_pane_toggle_btn.setCheckable(True)
        self._bottom_pane_toggle_btn.setProperty("navToggle", True)
        self._bottom_pane_toggle_btn.setChecked(True)
        self._bottom_pane_toggle_btn.setToolTip("Show or hide the bottom pane of auxiliary pages.")
        self._bottom_pane_toggle_btn.clicked.connect(self._on_bottom_pane_toggle_clicked)
        self._top_nav_row.addWidget(self._bottom_pane_toggle_btn)
        self._ai_assistant_btn = QPushButton(assistant_display_name())
        self._ai_assistant_btn.setCheckable(True)
        self._ai_assistant_btn.setProperty("navToggle", True)
        self._ai_assistant_btn.setToolTip(f"Open {assistant_display_name()} for quick CV Ops questions.")
        self._ai_assistant_btn.clicked.connect(self._on_ai_assistant_clicked)
        self._top_nav_row.addWidget(self._ai_assistant_btn)

        self._ws_status = QLabel("[WS] connecting...")
        self._ws_status.setObjectName("wsStatus")
        self._ws_status.setProperty("state", "connecting")
        self._ws_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ws_status.setToolTip("WebSocket status. Click to reconnect and resync without restarting training.")
        self._ws_status.mousePressEvent = self._on_ws_status_clicked  # type: ignore[method-assign]

        self._ws_refresh_btn = QToolButton()
        self._ws_refresh_btn.setObjectName("wsRefreshButton")
        self._ws_refresh_btn.setText("↻")
        self._ws_refresh_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._ws_refresh_btn.setAutoRaise(False)
        self._ws_refresh_btn.setEnabled(True)
        self._ws_refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._ws_refresh_btn.setToolTip("Force WebSocket reconnect and refresh the live output cache.")
        self._ws_refresh_btn.clicked.connect(self._refresh_ws_output)

        self._notifications_panel = NotificationsPanel(
            stream_hint=self.ws_url,
            parent=self,
            top_toolbar_layout=None,
        )

        self._status = _ServiceStripLabel("")
        self._status.setObjectName("serviceStatus")
        self._status.setWordWrap(False)

        # Host the top-nav buttons inside their own QWidget so we can promote
        # it to a native NSView later. Without that promotion, the
        # QWebEngineView inside the ecosystem plane (Cytoscape graph) takes
        # over click delivery for everything stacked above it in AppKit's
        # z-order — that's the "buttons are dead until I leave Ecosystem" bug.
        self._top_nav_container = QWidget()
        self._top_nav_container.setObjectName("cvOpsTopNavContainer")
        _top_nav_container_l = QHBoxLayout(self._top_nav_container)
        _top_nav_container_l.setContentsMargins(0, 0, 0, 0)
        _top_nav_container_l.setSpacing(0)
        _top_nav_container_l.addLayout(self._top_nav_row)
        layout.addWidget(self._top_nav_container)

        sb = self.statusBar()
        sb.setSizeGripEnabled(False)
        sb.hide()

        self._toast = QLabel("")
        self._toast.setObjectName("cvOpsToast")
        self._toast.setWordWrap(True)
        self._toast.setVisible(False)
        self._toast.setCursor(Qt.CursorShape.PointingHandCursor)
        self._toast.mousePressEvent = self._on_toast_clicked  # type: ignore[method-assign]
        self._toast_target_scenario = ""
        layout.addWidget(self._toast)

        self._heartbeat_notification_gate = HeartbeatNotificationGate()
        self._notification_tray = NotificationCardTray(parent=self, max_cards=1)
        self._notification_tray.notificationActivated.connect(self._on_notification_card_activated)
        self._notification_tray.trayLayoutChanged.connect(self._schedule_notification_tray_layout)
        self._notification_tray.hide()

        workspace_host = QWidget()
        workspace_layout = QHBoxLayout(workspace_host)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(6)
        layout.addWidget(workspace_host, stretch=1)
        self._workspace_host = workspace_host
        self._workspace_layout = workspace_layout

        _bprint("building catalog panel (eager)")
        _mark_boot("eager_panels_start", once=True)
        # [EAGER] Panels always visible in the default explore mode.
        self._catalog_panel = CatalogPanel(
            base_url=self.base_url,
            http_get=lambda path: self._http_json("GET", path),
            http_post=lambda path, body: self._http_json("POST", path, body),
            http_delete=lambda path: self._http_json("DELETE", path),
            http_get_text=lambda path: self._http_text(path),
            http_put=lambda path, body: self._http_json("PUT", path, body),
        )
        self._catalog_panel.scenarioMutated.connect(lambda _n: self._refresh_scenarios())
        self._catalog_panel.trainKicked.connect(self._on_train_kicked)
        self._catalog_panel.scenarioSelected.connect(self._on_catalog_selection)
        self._catalog_panel.errorRaised.connect(lambda msg: self._append_error("catalog", msg))

        self._lineage_panel = _LazyLineagePanel(
            http_get=lambda path: self._http_json("GET", path),
            http_post=lambda path, body: self._http_json("POST", path, body),
            http_delete=lambda path: self._http_json("DELETE", path),
            parent=self,
        )
        self._lineage_panel.errorRaised.connect(lambda msg: self._append_error("continuous-learning", msg))

        self._queue_panel = QueuePanel()
        self._queue_panel.jobSelected.connect(self._on_queue_selection)
        self._queue_panel.cancelRequested.connect(self._on_queue_cancel)
        self._queue_panel.retryRequested.connect(self._on_queue_retry)

        self._result_panel = _DeferredResultPanel(
            base_url=self.base_url,
            http_get=lambda path: self._http_json("GET", path),
            http_get_text=lambda path: self._http_text(path),
            parent=self,
        )
        self._result_panel.flagRequested.connect(self._on_console_flag_requested)
        _mark_boot(
            "eager_panels_done",
            once=True,
            panels="catalog,queue,result_proxy,lineage_proxy",
        )

        # [LAZY] All other panels are None until their tab is first activated.
        # Each _build_* factory creates the panel, wires signals, and returns
        # the page widget; _DeferredPage calls it on first showEvent.
        self._range_panel = None
        self._video_test_panel = None
        self._submit_panel = None
        self._test_range_result_panel = None
        self._split_magnifier_window = None
        self._split_magnifier_cls = None
        self._ai_assistant_window = None
        self._ai_assistant_cls = None
        self._database_panel = None
        self._three_d_panel = None
        self._cells_panel = None
        self._scrape_panel = None
        self._collect_tabular_panel = None
        self._collect_carve_panel = None
        self._collect_mode_stack: Optional[QStackedWidget] = None
        self._collect_mode_chooser_group = None
        self._collect_stage_group = None
        self._collect_full_gallery_btn = None
        self._collect_create_model_btn: Optional[QPushButton] = None
        self._collect_active_dataset_slug = ""
        self._database_godview_panel = None
        self._data_viz_db_selector = None
        self._data_viz_standalone = None
        self._notes_panel = None
        self._ontology_panel = None
        self._ontology_warmup_worker: Optional[QThread] = None
        self._ontology_warmup_ready = False
        self._ontology_attach_after_warmup = False
        self._webengine_warmed = False
        self._webengine_ready = False
        self._webengine_warm_view: Optional[QWidget] = None
        self._ontology_build_pending = False
        self._collect_dataset_editor = None

        _bprint("registering deferred pages")
        # Mode pages: all deferred so __init__ returns without building Qt widgets.
        self._test_range_page = _DeferredPage(
            self._build_test_range_body_widget,
            preload_fn=self._preload_test_range_page,
            load_label="Range",
        )
        self._data_mode_page = _DeferredPage(
            self._build_data_mode_page,
            preload_fn=self._preload_data_mode_page,
            load_label="Database",
        )
        self._data_viz_mode_page = _DeferredPage(
            self._build_data_viz_mode_page,
            preload_fn=self._preload_data_viz_mode_page,
            load_label="Data Viz",
        )
        self._collect_mode_page = _DeferredPage(
            self._build_collect_mode_page,
            preload_fn=self._preload_collect_mode_page,
            load_label="Collect & Edit",
        )
        self._settings_mode_page = _DeferredPage(self._build_settings_subtab, load_label="Settings")
        self._diagnostics_page = _DeferredPage(self._build_diagnostics_tab, load_label="Diagnostics")
        self._portal_page = _DeferredPage(self._build_portal_tab, load_label="Scope")

        # Tray-slot deferred panels: DeferredPage wrappers stand in until the
        # relevant mode is first activated, at which point the real panel is
        # built and stored as self._database_panel / _collect_dataset_editor /
        # _data_viz_db_selector.
        _db_panel_deferred = _DeferredPage(
            self._build_database_panel_widget,
            preload_fn=self._preload_database_panel_widget,
            load_label="Dataset Library",
        )
        _dataset_editor_deferred = _DeferredPage(
            self._build_dataset_editor_widget,
            preload_fn=self._preload_dataset_editor_widget,
            load_label="Dataset Editor",
        )
        _data_viz_selector_deferred = _DeferredPage(
            self._build_data_viz_selector_widget,
            preload_fn=self._preload_data_viz_selector_widget,
            load_label="Data Viz Selector",
        )

        self._workbench_split_host = WorkbenchSplitHost(
            WorkbenchSplitRefs(
                catalog_list=self._catalog_panel.list_widget(),
                catalog_detail=self._catalog_panel.detail_widget(),
                result_panel=self._result_panel,
                lineage_panel=self._lineage_panel,
                test_range_page=self._test_range_page,
                data_page=self._data_mode_page,
                viz_page=self._data_viz_mode_page,
                collect_page=self._collect_mode_page,
                notes_page=_DeferredPage(
                    self._build_notes_page,
                    preload_fn=self._preload_notes_page,
                    load_label="Notes",
                ),
                settings_page=self._settings_mode_page,
                diagnostics_page=self._diagnostics_page,
                cells_page=_DeferredPage(
                    self._build_cells_tab,
                    preload_fn=self._preload_cells_tab,
                    load_label="Cells",
                ),
                three_d_page=self._wrap_scroll_page(
                    _DeferredPage(
                        self._build_three_d_tab,
                        preload_fn=self._preload_three_d_panel,
                        load_label="3D",
                    )
                ),
                notifications_page=self._wrap_scroll_page(self._notifications_panel),
                portal_page=self._portal_page,
                queue_panel=self._queue_panel,
                collect_database_panel=_db_panel_deferred,
                collect_dataset_editor=_dataset_editor_deferred,
                data_viz_selector=_data_viz_selector_deferred,
            ),
            on_splitter_moved=self._on_workbench_split_moved,
            parent=self,
        )

        # Ecosystem plane: placeholder until first ecosystem click.
        self._plane_ecosystem = QWidget()
        self._plane_ecosystem.setObjectName("cvOpsEcosystemPlane")
        eco_l = QVBoxLayout(self._plane_ecosystem)
        eco_l.setContentsMargins(0, 0, 0, 0)
        eco_l.setSpacing(0)
        self._eco_placeholder = QLabel("[Loading Ecosystem...]")
        self._eco_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        eco_l.addWidget(self._eco_placeholder, stretch=1)

        self._plane_stack = QStackedWidget()
        self._plane_stack.addWidget(self._plane_ecosystem)
        self._plane_stack.addWidget(self._workbench_split_host)

        self._activity_rail = ActivityRailWidget(orientation=Qt.Orientation.Horizontal)
        self._activity_rail.ecoPressed.connect(self._on_activity_rail_eco)
        self._activity_rail.modePressed.connect(self._on_activity_rail_mode)
        # Center the rail in the top nav row: stretches on both sides push it to the middle.
        restore_idx = self._top_nav_row.indexOf(self._restore_panes_btn)
        self._top_nav_row.insertStretch(restore_idx + 1, 1)
        self._top_nav_row.insertWidget(restore_idx + 2, self._activity_rail)
        self._top_nav_row.insertStretch(restore_idx + 3, 1)
        self._ws_status.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self._ws_status.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        self._top_nav_row.addWidget(
            self._ws_status, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )
        self._top_nav_row.addWidget(
            self._ws_refresh_btn, 0, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight
        )

        self._shell_center = QWidget()
        shell_l = QHBoxLayout(self._shell_center)
        shell_l.setContentsMargins(0, 0, 0, 0)
        shell_l.setSpacing(0)
        shell_l.addWidget(self._plane_stack, stretch=1)
        self._workspace_layout.addWidget(self._shell_center, stretch=1)

        self._restore_ui_shell_state()
        self._sync_catalog_toggle_btn()
        self._sync_bottom_pane_toggle_btn()

        self._bottom_pulse_bar = QWidget()
        self._bottom_pulse_bar.setObjectName("cvOpsBottomPulseBar")
        self._bottom_pulse_bar.setFixedHeight(28)
        self._bottom_pulse_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        bpl = QHBoxLayout(self._bottom_pulse_bar)
        bpl.setContentsMargins(0, 0, 0, 0)
        bpl.setSpacing(0)

        self._event_pulse = EventPulseWidget(parent=self._bottom_pulse_bar)
        self._event_pulse.setVisible(bool(self._cvops_settings.show_event_pulse))
        self._event_pulse.openNotificationsRequested.connect(self._open_notifications_center)

        self._status.setParent(self._bottom_pulse_bar)
        self._status.setMaximumWidth(520)
        self._status.setMaximumHeight(28)
        self._status.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._status.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)
        bpl.addWidget(self._event_pulse, stretch=1)
        bpl.addWidget(self._status, stretch=0)
        layout.addWidget(self._bottom_pulse_bar)
        self._sync_bottom_pulse_bar_visibility()

        # Transparent overlay — covers full window, draws entity-connection beziers.
        self._connection_overlay = ConnectionOverlay(
            base_url=self.base_url, parent=self
        )
        self._connection_overlay.resize(self.size())

        # Register eager panels by primary entity type. Deferred panels
        # (dataset, database, data_viz_selector) register themselves in their
        # factory methods when first built.
        self._connection_overlay.register_panel(
            "scenario", self._catalog_panel.list_widget()
        )
        self._connection_overlay.register_panel("lineage", self._lineage_panel)

        # Wire entity selection for eagerly-built panels only.
        self._catalog_panel.entitySelected.connect(
            self._connection_overlay.draw_connections
        )
        self._lineage_panel.entitySelected.connect(
            self._connection_overlay.draw_connections
        )

        _bprint("connecting in-process event client")
        # Created unbound — the service is attached in _on_server_ready once the
        # background boot finishes. Signals are wired now so nothing is missed.
        self._ws = CvOpsMemoryClient(parent=self)
        self._ws.connectedChanged.connect(self._on_ws_connected)
        self._ws.jobStatus.connect(self._on_ws_job_status)
        self._ws.jobResult.connect(self._on_ws_job_result)
        self._ws.scenarioUpdated.connect(self._on_ws_scenario_updated)
        self._ws.trainingProgress.connect(self._on_ws_training_progress)
        self._ws.cellProgress.connect(self._on_ws_cell_progress)
        self._ws.socketError.connect(self._on_ws_error)
        self._ws.rawEvent.connect(self._ingest_system_notification)
        self._schedule_notification_tray_layout()
        self._apply_cvops_stylesheet()
        self._restore_shell_cache()
        self._sync_catalog_toggle_btn()
        self._sync_bottom_pane_toggle_btn()
        QTimer.singleShot(0, self._sync_catalog_toggle_btn)
        # Kick the background server boot now that the UI is wired. self._ws.start()
        # is deferred to _on_server_ready (it needs the live service).
        self._server_boot.start()
        _bprint("backend server boot dispatched")

        self._health_timer = QTimer(self)
        self._health_timer.setInterval(int(self._cvops_settings.health_poll_ms))
        self._health_timer.timeout.connect(self._refresh_health)
        self._health_timer.start()

        self._dashboard_timer.start()

        # Initial population starts once the embedded service is ready. Avoid
        # racing the local backend with loopback HTTP calls during cold boot.
        if self._prefer_direct_service_reads:
            try:
                self._catalog_panel.show_scenarios_connecting()
            except Exception:
                pass
        else:
            QTimer.singleShot(200, self._initial_load)
        QTimer.singleShot(3000, self._maybe_retry_scenarios_boot)
        # Warm the QtWebEngine/Chromium runtime once startup has settled (disk
        # idle) so the first Ecosystem open does not pay the cold-init cost on
        # the UI thread -- that cost is what makes the Ecosystem hang behind the
        # macOS spinner when opened during a training run (see _warm_webengine).
        QTimer.singleShot(2500, self._warm_webengine)
        _bprint("CvOpsWindow.__init__ complete — waiting for event loop")

        # Lift the top-nav strip above the ecosystem QWebEngineView in AppKit's
        # z-order by giving it its own NSView. This alone is *not enough* on
        # layer-backed macOS builds (the web view's CALayer still wins), so we
        # also install a global event filter below that hard-reroutes any click
        # in the top-nav's screen rect back to the right child.
        try:
            self._top_nav_container.setAttribute(
                Qt.WidgetAttribute.WA_NativeWindow, True
            )
            self._top_nav_container.winId()
            self._top_nav_container.raise_()
        except Exception:
            pass

        # Failsafe: re-route clicks at the QApplication level so even if the
        # web view's NSView is painted on top, our top nav still gets the
        # presses.
        try:
            self._top_nav_click_recover = _TopNavClickRecover(self._top_nav_container)
            app = QApplication.instance()
            if app is not None:
                app.installEventFilter(self._top_nav_click_recover)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Lazy panel factories — called by _DeferredPage on first show
    # ------------------------------------------------------------------

    def _preload_symbol(self, module_path: str, symbol: str) -> object:
        module = importlib.import_module(module_path, package=__package__)
        return getattr(module, symbol)

    def _preload_database_panel_widget(self) -> object:
        return self._preload_symbol(".ui.dataset_panel", "DatasetPanel")

    def _preload_dataset_editor_widget(self) -> object:
        return self._preload_symbol(".ui.dataset_editor", "DatasetEditorDialog")

    def _preload_data_viz_selector_widget(self) -> object:
        return self._preload_symbol(".ui.database_godview_panel", "DatabaseGodViewPanel")

    def _preload_notes_page(self) -> object:
        return self._preload_symbol(".ui.notes_panel", "NotesPanel")

    def _preload_test_range_page(self) -> dict[str, object]:
        return {
            "TestRangePanel": self._preload_symbol(".ui.range_panel", "TestRangePanel"),
            "VideoTestPanel": self._preload_symbol(".ui.video_test_panel", "VideoTestPanel"),
            "SubmitPanel": self._preload_symbol(".ui.submit_panel", "SubmitPanel"),
            "ResultPanel": self._preload_symbol(".ui.result_panel", "ResultPanel"),
            "SplitMagnifierWindow": self._preload_symbol(".ui.split_magnifier_panel", "SplitMagnifierWindow"),
        }

    def _preload_data_mode_page(self) -> object:
        return self._preload_symbol(".ui.database_godview_panel", "DatabaseGodViewPanel")

    def _preload_data_viz_mode_page(self) -> object:
        return self._preload_symbol(".ui.data_viz_hub", "DataVizHub")

    def _preload_collect_mode_page(self) -> dict[str, object]:
        return {
            "ScrapePanel": self._preload_symbol(".ui.scrape_panel", "ScrapePanel"),
            "DatasetPanel": self._preload_symbol(".ui.dataset_panel", "DatasetPanel"),
        }

    def _preload_cells_tab(self) -> object:
        return self._preload_symbol(".ui.cells_panel", "CellsPanel")

    def _build_database_panel_widget(self, preloaded: Optional[object] = None) -> Optional[QWidget]:
        """Build DatasetPanel on first call and wire signals."""
        DatasetPanel = preloaded or self._preload_symbol(".ui.dataset_panel", "DatasetPanel")
        if self._database_panel is None:
            self._database_panel = DatasetPanel(
                base_url=self.base_url,
                http_get=lambda path: self._http_json("GET", path),
                http_post=lambda path, body: self._http_json("POST", path, body),
                http_delete=lambda path: self._http_json("DELETE", path),
            )
            self._database_panel.datasetChanged.connect(lambda _name: self._refresh_scenarios())
            self._database_panel.errorRaised.connect(lambda msg: self._append_error("database", msg))
            self._database_panel.entitySelected.connect(self._connection_overlay.draw_connections)
            self._connection_overlay.register_panel("dataset", self._database_panel)
        return self._database_panel

    def _build_dataset_editor_widget(self, preloaded: Optional[object] = None) -> Optional[QWidget]:
        """Build DatasetEditorDialog on first call."""
        DatasetEditorDialog = preloaded or self._preload_symbol(".ui.dataset_editor", "DatasetEditorDialog")
        if self._collect_dataset_editor is None:
            self._collect_dataset_editor = DatasetEditorDialog(
                base_url=self.base_url,
                dataset_slug="",
                parent=None,
            )
            self._collect_dataset_editor.setWindowFlags(Qt.WindowType.Widget)
            for btn_box in self._collect_dataset_editor.findChildren(QDialogButtonBox):
                btn_box.setVisible(False)
        return self._collect_dataset_editor

    def _build_data_viz_selector_widget(self, preloaded: Optional[object] = None) -> Optional[QWidget]:
        """Build the data-viz DatabaseGodViewPanel selector on first call."""
        DatabaseGodViewPanel = preloaded or self._preload_symbol(".ui.database_godview_panel", "DatabaseGodViewPanel")
        if self._data_viz_db_selector is None:
            self._data_viz_db_selector = DatabaseGodViewPanel(
                project_root=self._project_root,
                http_get=lambda path: self._http_json("GET", path),
                parent=None,
                selector_only=True,
            )
            self._data_viz_db_selector.errorRaised.connect(
                lambda msg: self._append_error("database-godview-dataviz", msg)
            )
            self._data_viz_db_selector.scenario_focused.connect(self._on_data_viz_scenario_focused)
            self._data_viz_db_selector.entitySelected.connect(
                self._connection_overlay.draw_connections
            )
            self._data_viz_db_selector.entitySelected.connect(
                self._on_data_viz_database_entity_selected
            )
        return self._data_viz_db_selector

    def _build_notes_page(self, preloaded: Optional[object] = None) -> Optional[QWidget]:
        """Build NotesPanel on first call."""
        NotesPanel = preloaded or self._preload_symbol(".ui.notes_panel", "NotesPanel")
        if self._notes_panel is None:
            self._notes_panel = NotesPanel(parent=self, http_json=self._http_json)
            self._notes_panel.errorRaised.connect(lambda msg: self._append_error("notes", msg))
        return self._notes_panel

    def _take_notes_ai_workspace_for_assistant(self):
        panel = self._build_notes_page()
        take = getattr(panel, "take_ai_workspace_for_assistant", None)
        if callable(take):
            return take()
        return getattr(panel, "_ai_workspace", None)

    def _restore_notes_ai_workspace_from_assistant(self) -> None:
        panel = getattr(self, "_notes_panel", None)
        restore = getattr(panel, "restore_ai_workspace_from_assistant", None)
        if callable(restore):
            restore()

    def _build_test_range_body_widget(self, preloaded: Optional[dict[str, object]] = None) -> QWidget:
        if preloaded is not None:
            TestRangePanel = preloaded["TestRangePanel"]
            VideoTestPanel = preloaded["VideoTestPanel"]
            SubmitPanel = preloaded["SubmitPanel"]
            _ResultPanel = preloaded["ResultPanel"]
            SplitMagnifierWindow = preloaded["SplitMagnifierWindow"]
        else:
            TestRangePanel = self._preload_symbol(".ui.range_panel", "TestRangePanel")
            VideoTestPanel = self._preload_symbol(".ui.video_test_panel", "VideoTestPanel")
            SubmitPanel = self._preload_symbol(".ui.submit_panel", "SubmitPanel")
            _ResultPanel = self._preload_symbol(".ui.result_panel", "ResultPanel")
            SplitMagnifierWindow = self._preload_symbol(".ui.split_magnifier_panel", "SplitMagnifierWindow")

        if self._range_panel is None:
            self._range_panel = TestRangePanel(
                http_get=lambda path: self._http_json("GET", path),
                http_post=lambda path, body: self._http_json("POST", path, body),
                http_delete=lambda path: self._http_json("DELETE", path),
            )
            self._range_panel.errorRaised.connect(
                lambda msg: self._append_error("continuous-learning", msg)
            )

        if self._video_test_panel is None:
            self._video_test_panel = VideoTestPanel(
                http_get=lambda path: self._http_json("GET", path),
                http_post=lambda path, body: self._http_json("POST", path, body),
                http_delete=lambda path: self._http_json("DELETE", path),
            )
            self._video_test_panel.errorRaised.connect(
                lambda msg: self._append_error("test-range", msg)
            )

        if self._test_range_result_panel is None:
            self._test_range_result_panel = _ResultPanel(
                base_url=self.base_url,
                http_get=lambda path: self._http_json("GET", path),
                http_get_text=lambda path: self._http_text(path),
                show_detection_table=False,
            )
            self._test_range_result_panel.flagRequested.connect(self._on_console_flag_requested)
            self._test_range_result_panel.show_message("Submit a single image to see the result here.")

        if self._submit_panel is None:
            self._submit_panel = SubmitPanel(
                base_url=self.base_url,
                scenarios_provider=lambda: self._scenarios_cache,
                http_get=lambda path: self._http_json("GET", path),
            )
            self._submit_panel.set_ready_check(self._catalog_panel.is_ready)
            self._submit_panel.jobSubmitted.connect(self._on_job_submitted)
            self._submit_panel.submissionFailed.connect(self._on_submission_failed)
            self._submit_panel.registryModelsChanged.connect(
                self._video_test_panel._populate_models
            )
            self._submit_panel.registryModelsChanged.connect(
                self._test_range_result_panel.refresh_subroutine_models
            )
            self._submit_panel.registryModelsChanged.connect(
                self._refresh_split_magnifier_models
            )
            self._submit_panel.registryResultReady.connect(
                self._on_registry_result_ready
            )

        tab = QWidget()
        vl = QVBoxLayout(tab)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(6)

        info = QLabel(
            "Range — Quick-test trained models on single inputs, video, or a golden test set."
        )
        info.setWordWrap(True)
        info.setObjectName("stageInfo")
        vl.addWidget(info)

        quick_test_shell = QWidget()
        quick_test_shell_layout = QVBoxLayout(quick_test_shell)
        quick_test_shell_layout.setContentsMargins(0, 0, 0, 0)
        quick_test_shell_layout.setSpacing(6)

        quick_tool_row = QHBoxLayout()
        quick_tool_row.setContentsMargins(0, 0, 0, 0)
        self._split_magnifier_cls = SplitMagnifierWindow
        split_mag_btn = QPushButton("[SPLIT MAGNIFIER]")
        split_mag_btn.setToolTip(
            "Open a transparent always-on-top Range lens that captures another screen surface and runs CV inference."
        )
        split_mag_btn.clicked.connect(self._open_split_magnifier)
        quick_tool_row.addWidget(split_mag_btn)
        split_mag_hint = QLabel("Transparent screen-surface tester for apps outside CV Ops.")
        split_mag_hint.setProperty("muted", True)
        split_mag_hint.setStyleSheet("font-size: 10px;")
        quick_tool_row.addWidget(split_mag_hint, stretch=1)
        quick_test_shell_layout.addLayout(quick_tool_row)

        quick_test_workspace = QWidget()
        quick_test_layout = QHBoxLayout(quick_test_workspace)
        quick_test_layout.setContentsMargins(0, 0, 0, 0)
        quick_test_layout.setSpacing(10)

        submit_host = QWidget()
        submit_host.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        submit_host.setMinimumWidth(320)
        submit_host_layout = QVBoxLayout(submit_host)
        submit_host_layout.setContentsMargins(0, 0, 0, 0)
        submit_host_layout.setSpacing(0)
        # Let the panel fill the host's height so the drop zone expands into the
        # dead space, instead of a trailing stretch absorbing it.
        submit_host_layout.addWidget(self._submit_panel, stretch=1)

        result_host = QWidget()
        result_host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        result_host.setMinimumWidth(420)
        result_host_layout = QVBoxLayout(result_host)
        result_host_layout.setContentsMargins(0, 0, 0, 0)
        result_host_layout.setSpacing(0)
        result_host_layout.addWidget(self._test_range_result_panel)

        quick_test_layout.addWidget(submit_host, stretch=2)
        quick_test_layout.addWidget(result_host, stretch=3)
        quick_test_shell_layout.addWidget(quick_test_workspace)

        submit_section = CollapsibleSection("Quick Test (Submit Job)", expanded=True)
        submit_section.body_layout().addWidget(quick_test_shell)

        video_section = CollapsibleSection("Video Test Bench", expanded=True)
        video_section.body_layout().addWidget(self._video_test_panel, stretch=1)

        range_section = CollapsibleSection("Range Catalog", expanded=True)
        range_section.body_layout().addWidget(self._range_panel, stretch=1)

        test_range_body = QWidget()
        body_layout = QVBoxLayout(test_range_body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(10)
        body_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinAndMaxSize)
        body_layout.addWidget(submit_section)
        body_layout.addWidget(video_section)
        body_layout.addWidget(range_section)
        submit_section.expandedChanged.connect(lambda _expanded: self._layout_refresh_timer.start())
        video_section.expandedChanged.connect(lambda _expanded: self._layout_refresh_timer.start())
        range_section.expandedChanged.connect(lambda _expanded: self._layout_refresh_timer.start())
        vl.addWidget(self._wrap_scroll_page(test_range_body), stretch=1)
        return tab

    def _open_split_magnifier(self) -> None:
        cls = getattr(self, "_split_magnifier_cls", None)
        if cls is None:
            cls = self._preload_symbol(".ui.split_magnifier_panel", "SplitMagnifierWindow")
            self._split_magnifier_cls = cls
        if self._split_magnifier_window is None:
            self._split_magnifier_window = cls(
                http_get=lambda path: self._http_json("GET", path),
                parent=None,
            )
            self._split_magnifier_window.destroyed.connect(
                lambda _obj=None: setattr(self, "_split_magnifier_window", None)
            )
        try:
            self._split_magnifier_window.refresh_models()
        except Exception as exc:
            self._append_error("split-magnifier", f"model refresh: {exc}")
        self._split_magnifier_window.show_for_parent(self)

    def _refresh_split_magnifier_models(self) -> None:
        win = getattr(self, "_split_magnifier_window", None)
        if win is None:
            return
        try:
            win.refresh_models()
        except Exception as exc:
            self._append_error("split-magnifier", f"model refresh: {exc}")

    def _build_data_mode_page(self, preloaded: Optional[object] = None) -> QWidget:
        """Database mode: god-view only. Scrape/import live in the separate Collect mode."""
        DatabaseGodViewPanel = preloaded or self._preload_symbol(".ui.database_godview_panel", "DatabaseGodViewPanel")
        if self._database_godview_panel is None:
            self._database_godview_panel = DatabaseGodViewPanel(
                project_root=self._project_root,
                http_get=lambda path: self._http_json("GET", path),
                http_post=lambda path, body: self._http_json("POST", path, body),
            )
            self._database_godview_panel.errorRaised.connect(
                lambda msg: self._append_error("database-godview", msg)
            )
            self._database_godview_panel.entitySelected.connect(
                self._connection_overlay.draw_connections
            )
            self._connection_overlay.register_panel("database", self._database_godview_panel)

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        info = QLabel(
            "Database — god's-eye view of every data store, scenarios mapped to their resources."
        )
        info.setWordWrap(True)
        info.setObjectName("stageInfo")
        layout.addWidget(info)
        layout.addWidget(self._wrap_scroll_page(self._database_godview_panel), stretch=1)
        return page

    def _build_data_viz_mode_page(self, preloaded: Optional[object] = None) -> QWidget:
        """CSV/data visualization workspace; source selection lives in the bottom tray."""
        DataVizHub = preloaded or self._preload_symbol(".ui.data_viz_hub", "DataVizHub")
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        if self._data_viz_standalone is None:
            self._data_viz_standalone = DataVizHub(
                http_get=lambda path: self._http_json("GET", path),
                http_post=lambda path, body: self._http_json("POST", path, body, timeout=120.0),
                get_import_progress=lambda corr_id: (
                    self._server.service._import_progress.get(str(corr_id or "").strip())
                    if self._server is not None else None
                ),
                parent=page,
            )
            self._data_viz_standalone.setMinimumWidth(0)
            try:
                pol = self._data_viz_standalone.sizePolicy()
                pol.setHorizontalPolicy(QSizePolicy.Policy.Ignored)
                self._data_viz_standalone.setSizePolicy(pol)
            except Exception:
                pass
        outer.addWidget(self._data_viz_standalone, stretch=1)
        return page

    def _collapse_data_viz_catalog(self) -> None:
        # Data source selection now lives in the bottom tray; leave it visible
        # unless the operator explicitly closes that tray card.
        return

    def _on_data_viz_splitter_moved(self, pos: int, _index: int) -> None:
        return

    def _build_collect_mode_page(self, preloaded: Optional[dict[str, object]] = None) -> QWidget:
        """Collect & Edit mode: unified onboarding hub for all training data.

        A type chooser routes between Images (scrape jobs + dataset-folder dump) and
        Tabular (single-file CSV/TSV upload + edit + schema). The dataset library and
        live editor trays follow the active selection across both branches.
        """
        if preloaded is not None:
            ScrapePanel = preloaded["ScrapePanel"]
            preloaded_dataset_panel = preloaded.get("DatasetPanel")
        else:
            ScrapePanel = self._preload_symbol(".ui.scrape_panel", "ScrapePanel")
            preloaded_dataset_panel = None
        if self._scrape_panel is None:
            self._scrape_panel = ScrapePanel()
            self._scrape_panel.errorRaised.connect(lambda msg: self._append_error("scrape", msg))
            self._scrape_panel.jobSelected.connect(self._on_scrape_job_selected)
            self._scrape_panel.stageChanged.connect(self._on_scrape_stage_changed)
            # Folder import is now a first-class ingestion action next to "New job"
            # (the old Dataset Dump side tab is gone); window owns the import + the
            # straight-to-model handoff.
            self._scrape_panel.importDatasetRequested.connect(
                self._on_collect_import_dataset_clicked
            )

        # Ensure database panel exists so the entity-selected signal can be wired.
        self._build_database_panel_widget(preloaded_dataset_panel)
        if self._database_panel is not None:
            try:
                self._database_panel.entitySelected.connect(self._on_collect_library_entity_selected)
            except Exception:
                pass

        # Tabular onboarding panel (single-file CSV/TSV upload + edit).
        if self._collect_tabular_panel is None:
            CollectTabularPanel = self._preload_symbol(".ui.collect_tabular_panel", "CollectTabularPanel")
            self._collect_tabular_panel = CollectTabularPanel(base_url=self.base_url)
            self._collect_tabular_panel.errorRaised.connect(
                lambda msg: self._append_error("tabular_upload", msg)
            )
            self._collect_tabular_panel.tabularDatasetUploaded.connect(
                self._on_collect_tabular_uploaded
            )

        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # --- Onboarding type chooser: Images vs Tabular ---
        chooser_row = QHBoxLayout()
        chooser_row.setSpacing(6)
        prompt = QLabel("What are you onboarding for training?")
        prompt.setObjectName("stageInfo")
        chooser_row.addWidget(prompt)
        self._collect_images_btn = QPushButton("Images")
        self._collect_images_btn.setCheckable(True)
        self._collect_images_btn.setChecked(True)
        self._collect_images_btn.setProperty("navToggle", True)
        self._collect_tabular_btn = QPushButton("Tabular")
        self._collect_tabular_btn.setCheckable(True)
        self._collect_tabular_btn.setProperty("navToggle", True)
        self._collect_carve_btn = QPushButton("Semantic Carve")
        self._collect_carve_btn.setCheckable(True)
        self._collect_carve_btn.setProperty("navToggle", True)
        chooser_group = QButtonGroup(page)
        chooser_group.setExclusive(True)
        chooser_group.addButton(self._collect_images_btn, 0)
        chooser_group.addButton(self._collect_tabular_btn, 1)
        chooser_group.addButton(self._collect_carve_btn, 2)
        chooser_group.idClicked.connect(self._on_collect_mode_chooser)
        self._collect_mode_chooser_group = chooser_group
        chooser_row.addWidget(self._collect_images_btn)
        chooser_row.addWidget(self._collect_tabular_btn)
        chooser_row.addWidget(self._collect_carve_btn)
        chooser_row.addStretch(1)
        layout.addLayout(chooser_row)

        self._collect_mode_stack = QStackedWidget()
        layout.addWidget(self._collect_mode_stack, stretch=1)

        # Images branch: a guided 3-stage flow (Collect / Label & Edit / Emit) that
        # drives the ScrapePanel's existing pages. The panel's own nav strip is hidden
        # so this strip is the single navigation, and the workspace lands on Collect
        # (the scrape dashboard + job catalog) rather than the flooded Label workbench.
        self._scrape_panel.set_nav_strip_visible(False)
        images_page = QWidget()
        images_layout = QVBoxLayout(images_page)
        images_layout.setContentsMargins(0, 0, 0, 0)
        images_layout.setSpacing(6)

        stage_row = QHBoxLayout()
        stage_row.setSpacing(6)
        info = QLabel("Images —")
        info.setObjectName("stageInfo")
        stage_row.addWidget(info)
        stage_group = QButtonGroup(images_page)
        stage_group.setExclusive(True)
        self._collect_stage_group = stage_group
        # Emit is folded into the Label & Edit editor (a collapsible section), so
        # it is no longer its own islanded stage.
        for stage_id, label in ((0, "Collect"), (1, "Label & Edit")):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setProperty("navToggle", True)
            if stage_id == 0:
                btn.setChecked(True)
            stage_group.addButton(btn, stage_id)
            stage_row.addWidget(btn)
        self._collect_full_gallery_btn = QPushButton("Full gallery")
        self._collect_full_gallery_btn.setCheckable(True)
        self._collect_full_gallery_btn.setProperty("navToggle", True)
        self._collect_full_gallery_btn.setVisible(False)
        self._collect_full_gallery_btn.toggled.connect(self._on_collect_full_gallery_toggled)
        stage_row.addWidget(self._collect_full_gallery_btn)
        stage_row.addStretch(1)
        # Ungated scenario creation: turn the active (or any) dataset into a
        # training scenario right here, instead of detouring to the Train tab.
        self._collect_create_model_btn = QPushButton("Create model")
        self._collect_create_model_btn.setProperty("isPrimary", True)
        self._collect_create_model_btn.setToolTip(
            "Create a training scenario from a dataset — pre-filled from the selected "
            "dataset so an already-made dataset just needs a confirm."
        )
        self._collect_create_model_btn.clicked.connect(self._on_collect_create_scenario_clicked)
        stage_row.addWidget(self._collect_create_model_btn)
        stage_group.idClicked.connect(self._on_collect_stage)
        images_layout.addLayout(stage_row)
        self._sync_collect_create_model_btn()

        # Drag-and-drop is the expected import path (no freezing dialog).
        self._collect_import_drop = _DatasetImportDropZone()
        self._collect_import_drop.foldersDropped.connect(self._import_dataset_folders)
        images_layout.addWidget(self._collect_import_drop)

        images_layout.addWidget(self._scrape_panel, stretch=1)
        self._collect_mode_stack.addWidget(images_page)

        # Land on the Collect stage (scrape dashboard / job catalog), not the Label flood.
        self._scrape_panel.select_stage("collect")

        # Tabular branch: single-file upload + edit; schema handoff via the library tray.
        tabular_page = QWidget()
        tabular_layout = QVBoxLayout(tabular_page)
        tabular_layout.setContentsMargins(0, 0, 0, 0)
        tabular_layout.setSpacing(6)
        tab_guide = QLabel(
            "Tabular — the home for any row/column signal you want to model: exported tables, "
            "playlist features, hardware and sensor logs. Today: upload a .csv/.tsv file, Edit the "
            "table, then set the label and feature columns in the Dataset Library (tabular mode) to "
            "make it train-ready for a torch_tabular scenario."
        )
        tab_guide.setWordWrap(True)
        tab_guide.setObjectName("stageInfo")
        tabular_layout.addWidget(tab_guide)
        tabular_section = CollapsibleSection("Tabular Upload", expanded=True)
        tabular_section.body_layout().addWidget(self._collect_tabular_panel)
        tabular_layout.addWidget(tabular_section, stretch=1)
        self._collect_mode_stack.addWidget(tabular_page)

        # Semantic Carve branch: folder of images -> ImageFolder dataset by meaning.
        if self._collect_carve_panel is None:
            SemanticCarvePanel = self._preload_symbol(".ui.semantic_carve_panel", "SemanticCarvePanel")
            self._collect_carve_panel = SemanticCarvePanel(
                http_get=lambda path: self._http_json("GET", path),
                http_post=lambda path, body: self._http_json("POST", path, body),
            )
            self._collect_carve_panel.errorRaised.connect(
                lambda msg: self._append_error("semantic_carve", msg)
            )
            self._collect_carve_panel.statusChanged.connect(
                lambda msg: self._status.setText(msg) if self._status is not None else None
            )
            self._collect_carve_panel.datasetCreated.connect(self._on_collect_carve_created)
        carve_page = QWidget()
        carve_layout = QVBoxLayout(carve_page)
        carve_layout.setContentsMargins(0, 0, 0, 0)
        carve_layout.setSpacing(6)
        carve_layout.addWidget(self._collect_carve_panel, stretch=1)
        self._collect_mode_stack.addWidget(carve_page)

        return page

    def _on_collect_mode_chooser(self, index: int) -> None:
        """Switch the Collect & Edit hub between Images / Tabular / Semantic Carve."""
        if self._collect_mode_stack is None:
            return
        idx = int(index)
        self._collect_mode_stack.setCurrentIndex(idx)
        if idx == 1 and self._database_panel is not None:
            # Flip the shared library tray into tabular mode for the schema step.
            try:
                self._database_panel.show_tabular_dataset("")
            except Exception:
                pass

    def _on_collect_carve_created(self, slug: str) -> None:
        """A carved ImageFolder dataset just landed; surface it in the library tray."""
        name = str(slug or "").strip()
        if not name:
            return
        self._status.setText(f"Semantic carve created dataset '{name}'.") if self._status is not None else None
        if self._database_panel is not None:
            try:
                self._database_panel.refresh()
            except Exception:
                pass

    def _on_collect_tabular_uploaded(self, slug: str) -> None:
        """Bridge: a fresh tabular upload selects itself in the library tray (tabular mode)."""
        name = str(slug or "").strip()
        if not name or self._database_panel is None:
            return
        ok = True
        detail = ""
        try:
            self._database_panel.show_tabular_dataset(name)
        except Exception as exc:
            ok = False
            detail = str(exc)
        panel = getattr(self, "_collect_tabular_panel", None)
        note = getattr(panel, "note_library_handoff", None)
        if callable(note):
            try:
                note(name, ok=ok, detail=detail)
            except Exception:
                pass

    def _on_scrape_job_selected(self, slug: str) -> None:
        """Bridge: scrape panel selection drives the DatasetPanel library catalog."""
        name = str(slug or "").strip()
        if not name:
            return
        # Land a freshly selected job on the Collect dashboard instead of auto-jumping
        # into the dense Label workbench (the ScrapePanel's default preferred tab).
        if self._scrape_panel is not None:
            try:
                self._scrape_panel.select_stage("collect")
            except Exception:
                pass
        try:
            self._database_panel.select_library(name)
        except Exception:
            pass
        self._point_collect_editor_at(name)

    def _on_collect_stage(self, stage_id: int) -> None:
        """Drive the ScrapePanel from the outer Collect/Label&Edit/Emit stage strip."""
        stage = {0: "collect", 1: "label"}.get(int(stage_id), "collect")
        if self._scrape_panel is not None:
            self._scrape_panel.select_stage(stage)
        if self._collect_full_gallery_btn is not None:
            self._collect_full_gallery_btn.setVisible(stage == "label")
            if stage != "label" and self._collect_full_gallery_btn.isChecked():
                self._collect_full_gallery_btn.blockSignals(True)
                self._collect_full_gallery_btn.setChecked(False)
                self._collect_full_gallery_btn.blockSignals(False)

    def _on_collect_full_gallery_toggled(self, checked: bool) -> None:
        """Label & Edit sub-toggle: flip between the annotate view and the full gallery."""
        if self._scrape_panel is not None:
            self._scrape_panel.select_stage("gallery" if checked else "label")

    def _on_scrape_stage_changed(self, stage: str) -> None:
        """Keep the outer stage strip in sync when the ScrapePanel switches stages itself."""
        group = self._collect_stage_group
        if group is None:
            return
        outer = {"collect": 0, "label": 1, "gallery": 1, "emit": 1}.get(str(stage or ""))
        if outer is None:
            return
        btn = group.button(outer)
        if btn is not None and not btn.isChecked():
            btn.blockSignals(True)
            btn.setChecked(True)
            btn.blockSignals(False)
        if self._collect_full_gallery_btn is not None:
            on_label_stage = str(stage or "") in ("label", "gallery")
            self._collect_full_gallery_btn.setVisible(on_label_stage)
            want_gallery = str(stage or "") == "gallery"
            if self._collect_full_gallery_btn.isChecked() != want_gallery:
                self._collect_full_gallery_btn.blockSignals(True)
                self._collect_full_gallery_btn.setChecked(want_gallery)
                self._collect_full_gallery_btn.blockSignals(False)

    def _on_collect_library_entity_selected(self, kind: str, value: str) -> None:
        if str(kind or "").strip() != "dataset":
            return
        slug = str(value or "").strip()
        self._point_collect_editor_at(slug)
        # In Tabular onboarding mode, bind the tabular panel to the selected dataset so
        # Data Health / Split / History operate on it (not just freshly-uploaded files).
        panel = getattr(self, "_collect_tabular_panel", None)
        tab_btn = getattr(self, "_collect_tabular_btn", None)
        if panel is not None and slug and tab_btn is not None and tab_btn.isChecked():
            binder = getattr(panel, "set_active_dataset", None)
            if callable(binder):
                try:
                    binder(slug)
                except Exception:
                    pass

    def _on_collect_dataset_imported(self, slug: str) -> None:
        name = str(slug or "").strip()
        if not name:
            return
        try:
            self._database_panel.select_library(name)
        except Exception:
            pass
        self._point_collect_editor_at(name)

    def _on_collect_import_dataset_clicked(self) -> None:
        """Fallback picker for importing a dataset folder.

        Drag-and-drop onto the Collect & Edit drop zone is the expected path; this
        button is the "if you really want a picker" route and uses the FAST native
        OS chooser. The old non-native multi-select dialog enumerated folders on
        the GUI thread and hard-froze on large drives, so it is gone.
        """
        folder = QFileDialog.getExistingDirectory(
            self,
            "Import dataset folder",
            "",
            QFileDialog.Option.ShowDirsOnly,  # native chooser; no DontUseNativeDialog
        )
        folder = str(folder or "").strip()
        if folder:
            self._import_dataset_folders([folder])

    def _import_dataset_folders(self, folders: list) -> None:
        """Import one or more dataset folders into the library, then offer a model.

        Shared by both the drag-and-drop drop zone and the native picker fallback.
        Folders are copied via /database/import_folder; the freshest import lands
        in the library + editor and can be turned straight into a scenario.
        """
        worker = getattr(self, "_collect_import_thread", None)
        if worker is not None and worker.isRunning():
            self._status.setText("Dataset import already running.")
            return
        clean: list[str] = []
        seen: set[str] = set()
        for raw in folders or []:
            text = str(raw or "").strip()
            if not text:
                continue
            path = Path(text).expanduser()
            if not path.is_dir():
                continue
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                clean.append(resolved)
        if not clean:
            return
        self._set_collect_import_busy(
            True,
            f"Importing {len(clean)} dataset folder{'s' if len(clean) != 1 else ''}...",
        )

        def _run_import() -> dict[str, Any]:
            imported: list[str] = []
            errors: list[str] = []
            last_payload: dict[str, Any] = {}
            for folder in clean:
                try:
                    payload = self._http_json(
                        "POST", "/database/import_folder", {"source_path": folder}, timeout=120.0
                    )
                except Exception as exc:
                    errors.append(f"{Path(folder).name}: {exc}")
                    continue
                slug = str((payload or {}).get("slug") or "").strip()
                if slug:
                    imported.append(slug)
                    last_payload = dict(payload or {})
            return {"imported": imported, "errors": errors, "last_payload": last_payload}

        thread = _HttpFetchThread(_run_import, parent=self)
        thread.done.connect(self._on_collect_import_done)
        thread.failed.connect(self._on_collect_import_failed)
        thread.finished.connect(lambda: setattr(self, "_collect_import_thread", None))
        thread.finished.connect(thread.deleteLater)
        self._collect_import_thread = thread
        thread.start()

    def _set_collect_import_busy(self, busy: bool, text: str = "") -> None:
        drop = getattr(self, "_collect_import_drop", None)
        if drop is not None and hasattr(drop, "set_busy"):
            try:
                drop.set_busy(busy, text)
            except Exception:
                pass
        if text:
            self._status.setText(text)

    def _on_collect_import_done(self, result: object) -> None:
        self._set_collect_import_busy(False)
        payload = result if isinstance(result, dict) else {}
        imported = [str(x) for x in list(payload.get("imported") or []) if str(x).strip()]
        errors = [str(x) for x in list(payload.get("errors") or []) if str(x).strip()]
        last_payload = payload.get("last_payload") if isinstance(payload.get("last_payload"), dict) else {}
        for msg in errors:
            self._append_error("dataset_import", msg)
        if not imported:
            if not errors:
                self._append_error("dataset_import", "No valid dataset folders were imported.")
            self._status.setText("Dataset import failed.")
            return
        slug = imported[-1]
        self._on_collect_dataset_imported(slug)
        self._status.setText(f"Imported dataset: {slug}")
        QTimer.singleShot(0, lambda s=slug, p=dict(last_payload or {}): self._offer_model_from_dataset(s, p))

    def _on_collect_import_failed(self, message: str) -> None:
        self._set_collect_import_busy(False)
        msg = str(message or "Dataset import failed.")
        self._append_error("dataset_import", msg)
        self._status.setText(f"Dataset import failed: {msg}")

    def _offer_model_from_dataset(self, slug: str, payload: dict[str, Any]) -> None:
        """After importing a dataset, open the scenario builder pre-filled with it.

        Trainable formats go straight to the (pre-filled) New Scenario dialog so the
        user can confirm/tweak and see any validation error inline — instead of the
        old silent one-shot POST that surfaced only an opaque 'HTTP 400'."""
        name = str(slug or "").strip()
        if not name:
            return
        trainable = {
            "yolo_detection",
            "csv_tabular",
            "face_csv",
            "audiofolder_classification",
            "llm_instruction_jsonl",
        }
        fmt = str((payload or {}).get("format") or "").strip().lower()
        if fmt not in trainable:
            QMessageBox.information(
                self,
                "Dataset imported",
                f"Imported '{name}'.\n\n"
                "This dataset format is not directly trainable yet. For image-classification "
                "folders, use the Dataset Library's 'Convert ImageFolder -> YOLO' action first, "
                "then create a model.",
            )
            return
        # Active slug was set by _on_collect_dataset_imported; open the pre-filled
        # builder so classes/backbone are detected and any error is shown clearly.
        self._collect_active_dataset_slug = name
        self._sync_collect_create_model_btn()
        self._on_collect_create_scenario_clicked()

    def _point_collect_editor_at(self, slug: str) -> None:
        self._collect_active_dataset_slug = str(slug or "").strip()
        self._sync_collect_create_model_btn()
        editor = getattr(self, "_collect_dataset_editor", None)
        if editor is None:
            return
        try:
            editor.set_dataset_slug(slug)
        except Exception:
            pass

    def _sync_collect_create_model_btn(self) -> None:
        """Reflect the active dataset on the Collect & Edit 'Create model' button."""
        btn = getattr(self, "_collect_create_model_btn", None)
        if btn is None:
            return
        slug = str(getattr(self, "_collect_active_dataset_slug", "") or "").strip()
        if slug:
            btn.setText(f"Create model from {slug}")
            btn.setToolTip(
                f"Create a training scenario from '{slug}'. The dialog opens pre-filled "
                "(backbone, classes, name) so you can just confirm — or tweak."
            )
        else:
            btn.setText("Create model")
            btn.setToolTip(
                "Create a training scenario from a dataset. Select a dataset first to "
                "have it pre-filled, or pick one in the dialog."
            )

    def _on_collect_create_scenario_clicked(self) -> None:
        """Open the scenario builder from Collect & Edit, pre-filled with the active
        dataset. This is the same dialog the Train tab uses — just no longer gated
        behind it — so onboarding an already-made dataset stays on this page."""
        slug = str(getattr(self, "_collect_active_dataset_slug", "") or "").strip()
        worker = getattr(self, "_collect_scenario_prep_thread", None)
        if worker is not None and worker.isRunning():
            self._status.setText("Scenario builder is already preparing.")
            return
        self._status.setText("Preparing scenario builder...")

        def _prepare() -> dict[str, Any]:
            errors: list[str] = []
            models: list[dict[str, Any]] = []
            datasets_payload: dict[str, Any] = {}
            dataset_info_cache: dict[str, dict[str, Any]] = {}
            try:
                models_payload = self._http_json("GET", "/models", timeout=20.0)
                if isinstance(models_payload, dict):
                    models = [m for m in list(models_payload.get("models") or []) if isinstance(m, dict)]
            except Exception as exc:
                errors.append(f"/models: {exc}")
            try:
                payload = self._http_json("GET", "/database", timeout=20.0)
                if isinstance(payload, dict):
                    datasets_payload = dict(payload)
            except Exception as exc:
                errors.append(f"/database: {exc}")
            if slug:
                try:
                    enc = urllib.parse.quote(slug, safe="")
                    payload = self._http_json("GET", f"/database/{enc}", timeout=30.0)
                    if isinstance(payload, dict):
                        dataset_info_cache[slug] = dict(payload)
                except Exception as exc:
                    errors.append(f"/database/{slug}: {exc}")
            return {
                "slug": slug,
                "models": models,
                "datasets_payload": datasets_payload,
                "dataset_info_cache": dataset_info_cache,
                "errors": errors,
            }

        thread = _HttpFetchThread(_prepare, parent=self)
        thread.done.connect(self._open_collect_scenario_dialog)
        thread.failed.connect(self._on_collect_scenario_prepare_failed)
        thread.finished.connect(lambda: setattr(self, "_collect_scenario_prep_thread", None))
        thread.finished.connect(thread.deleteLater)
        self._collect_scenario_prep_thread = thread
        thread.start()

    def _open_collect_scenario_dialog(self, prep: object) -> None:
        from .ui.new_scenario_dialog import NewScenarioDialog  # local: lazy import

        data = prep if isinstance(prep, dict) else {}
        slug = str(data.get("slug") or getattr(self, "_collect_active_dataset_slug", "") or "").strip()
        models = [m for m in list(data.get("models") or []) if isinstance(m, dict)]
        datasets_payload = data.get("datasets_payload") if isinstance(data.get("datasets_payload"), dict) else {}
        dataset_info_cache = (
            data.get("dataset_info_cache") if isinstance(data.get("dataset_info_cache"), dict) else {}
        )
        for msg in [str(x) for x in list(data.get("errors") or []) if str(x).strip()]:
            self._append_error("scenario_create", msg)

        def _cached_http_get(path: str) -> dict[str, Any]:
            raw = str(path or "").strip()
            if raw == "/database":
                return dict(datasets_payload or {})
            if raw.startswith("/database/"):
                name = urllib.parse.unquote(raw.split("/database/", 1)[1])
                cached = dataset_info_cache.get(name)
                if isinstance(cached, dict):
                    return dict(cached)
            return self._http_json("GET", raw)

        dlg = NewScenarioDialog(
            http_get=_cached_http_get,
            http_post=lambda path, body: self._http_json("POST", path, body),
            models=models,
            datasets_payload=dict(datasets_payload or {}),
            dataset_info_cache={
                str(k): dict(v)
                for k, v in (dataset_info_cache or {}).items()
                if str(k).strip() and isinstance(v, dict)
            },
            initial_dataset=slug,
            parent=self,
        )
        if slug:
            try:
                dlg.preselect_dataset(slug)
            except Exception as exc:
                self._append_error("scenario_create", f"prefill failed: {exc}")
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        created = dlg.created_scenario_name()
        if not created:
            return
        self._refresh_scenarios()
        self._status.setText(f"Scenario created: {created}")
        answer = QMessageBox.question(
            self,
            "Scenario created",
            f"Created scenario '{created}'.\n\nOpen the Train workspace now to start training?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._set_workbench_mode(WorkbenchSplitHost.MODE_EXPLORE)
            QTimer.singleShot(0, lambda: self._catalog_panel.select_scenario(created))

    def _on_collect_scenario_prepare_failed(self, message: str) -> None:
        msg = str(message or "Scenario builder preparation failed.")
        self._append_error("scenario_create", msg)
        self._status.setText(f"Scenario builder failed: {msg}")

    def _build_diagnostics_tab(self) -> QWidget:
        """Auxiliary diagnostics surface for overview and error review."""
        tab = QWidget()
        dl = QVBoxLayout(tab)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(6)

        inner = DropdownPaneStack()
        inner.addTab(self._wrap_scroll_page(self._build_overview_subtab()), "Overview")
        inner.addTab(self._wrap_scroll_page(self._build_errors_subtab()), "Errors")
        dl.addWidget(inner, stretch=1)
        return tab

    def _preload_three_d_panel(self) -> dict:
        """Heavy preload for the 3D tab (runs on the main thread, deferred a tick).

        Imports mlops.trellis2, runs detect()/JobStore()/resolve_depth_mlpackage(), and
        checks Apple MLX availability, then hands the result to _build_three_d_tab().
        This runs on the main thread on purpose — see _DeferredPage for why these
        imports must not run on a background thread.
        """
        from .ui.three_d_panel import ThreeDPanel as _cls
        from mlops.trellis2 import DEFAULT_PARAMS, JobStore, detect
        from mlops.trellis2.depth_anything_local import resolve_depth_mlpackage
        caps = detect()
        store = JobStore()
        depth_mlpackage = resolve_depth_mlpackage()
        try:
            from mlops.three_d.trellis2_apple import available as _apple_avail
            apple_ok, _ = _apple_avail()
        except Exception:
            apple_ok = False
        return {
            "_cls": _cls,
            "caps": caps,
            "store": store,
            "defaults": DEFAULT_PARAMS,
            "depth_mlpackage": depth_mlpackage,
            "apple_ok": apple_ok,
        }

    def _build_three_d_tab(self, preloaded: Optional[dict] = None) -> QWidget:
        if self._three_d_panel is None:
            if preloaded is not None:
                cls = preloaded.get("_cls")
            else:
                from .ui.three_d_panel import ThreeDPanel
                cls = ThreeDPanel
            self._three_d_panel = cls(preloaded=preloaded)
            self._three_d_panel.errorRaised.connect(lambda msg: self._append_error("3d-gen", msg))
        root = QWidget()
        vl = QVBoxLayout(root)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.addWidget(self._three_d_panel)
        return root

    def _build_cells_tab(self, preloaded: Optional[object] = None) -> QWidget:
        CellsPanel = preloaded or self._preload_symbol(".ui.cells_panel", "CellsPanel")
        if self._cells_panel is None:
            self._cells_panel = CellsPanel(
                http_get=lambda path: self._http_json("GET", path),
                http_put=lambda path, body: self._http_json("PUT", path, body),
                http_post=lambda path, body: self._http_json("POST", path, body),
            )
            self._cells_panel.errorRaised.connect(lambda msg: self._append_error("cells", msg))
            self._cells_panel.openWorkflowTrainRequested.connect(self._on_cells_open_workflow_train)
            self._cells_panel.scenarioMutated.connect(lambda _name: self._refresh_scenarios())
            self._cells_panel.trainKicked.connect(self._on_cells_train_kicked)
        root = QWidget()
        vl = QVBoxLayout(root)
        vl.setContentsMargins(8, 8, 8, 8)
        vl.addWidget(self._cells_panel)
        return root

    def _on_catalog_toggle_clicked(self, _checked: bool = False) -> None:
        host = getattr(self, "_workbench_split_host", None)
        if host is None:
            return
        if self._plane_stack.currentIndex() == 0:
            self._set_workbench_mode(self._cvops_settings.ui_shell_mode or WorkbenchSplitHost.MODE_EXPLORE)
            if not host.is_catalog_visible():
                host.toggle_catalog()
            self._sync_catalog_toggle_btn()
            self._persist_ui_shell_state()
            return
        host.toggle_catalog()
        self._sync_catalog_toggle_btn()

    def _on_bottom_pane_toggle_clicked(self, _checked: bool = False) -> None:
        host = getattr(self, "_workbench_split_host", None)
        if host is None:
            return
        if self._plane_stack.currentIndex() == 0:
            self._set_workbench_mode(self._cvops_settings.ui_shell_mode or WorkbenchSplitHost.MODE_EXPLORE)
            if host.has_bottom_panes() and not host.is_bottom_pane_visible():
                host.toggle_bottom_pane()
            self._sync_bottom_pane_toggle_btn()
            self._persist_ui_shell_state()
            return
        host.toggle_bottom_pane()
        self._sync_bottom_pane_toggle_btn()

    def _on_ai_assistant_clicked(self, checked: bool = False) -> None:
        if checked:
            self._open_ai_assistant()
            return
        win = getattr(self, "_ai_assistant_window", None)
        if win is not None:
            win.close()

    def _open_ai_assistant(self) -> None:
        cls = getattr(self, "_ai_assistant_cls", None)
        if cls is None:
            cls = self._preload_symbol(".ui.assistant_overlay", "AssistantOverlayWindow")
            self._ai_assistant_cls = cls
        if self._ai_assistant_window is None:
            self._ai_assistant_window = cls(
                workspace_provider=self._take_notes_ai_workspace_for_assistant,
                workspace_restorer=self._restore_notes_ai_workspace_from_assistant,
                parent=self,
            )
            closed = getattr(self._ai_assistant_window, "closed", None)
            if closed is not None:
                closed.connect(self._on_ai_assistant_closed)
            self._ai_assistant_window.destroyed.connect(
                self._on_ai_assistant_destroyed
            )
        self._ai_assistant_btn.setText(assistant_display_name())
        self._ai_assistant_btn.setToolTip(f"Open {assistant_display_name()} for quick CV Ops questions.")
        self._set_ai_assistant_button_visible(True)
        self._ai_assistant_window.show_for_parent(self)
        self._update_ai_assistant_geometry()

    def _set_ai_assistant_button_visible(self, visible: bool) -> None:
        if not hasattr(self, "_ai_assistant_btn"):
            return
        self._ai_assistant_btn.setChecked(bool(visible))
        self._ai_assistant_btn.setProperty("paneVisible", bool(visible))
        repolish(self._ai_assistant_btn)
        self._ai_assistant_btn.update()

    def _on_ai_assistant_closed(self) -> None:
        self._set_ai_assistant_button_visible(False)

    def _on_ai_assistant_destroyed(self, _obj=None) -> None:
        self._ai_assistant_window = None
        self._set_ai_assistant_button_visible(False)

    def _sync_catalog_toggle_btn(self) -> None:
        host = getattr(self, "_workbench_split_host", None)
        if host is None:
            return
        self._catalog_toggle_btn.setEnabled(True)
        visible = bool(host.is_catalog_visible())
        self._catalog_toggle_btn.setProperty("paneVisible", visible)
        self._catalog_toggle_btn.setChecked(visible)
        repolish(self._catalog_toggle_btn)
        self._catalog_toggle_btn.update()

    def _sync_bottom_pane_toggle_btn(self) -> None:
        host = getattr(self, "_workbench_split_host", None)
        if host is None:
            return
        has_panes = bool(host.has_bottom_panes())
        visible = bool(host.is_bottom_pane_visible())
        self._bottom_pane_toggle_btn.setEnabled(has_panes)
        self._bottom_pane_toggle_btn.setProperty("paneVisible", visible)
        self._bottom_pane_toggle_btn.setChecked(visible)
        repolish(self._bottom_pane_toggle_btn)
        self._bottom_pane_toggle_btn.update()

    def _open_workflow_stage(self, stage_index: int, cube_attr: str = "") -> None:
        self._show_plane_workbench()
        idx = int(stage_index)
        if idx == 1:
            self._set_workbench_mode(WorkbenchSplitHost.MODE_DATA)
        else:
            self._set_workbench_mode(WorkbenchSplitHost.MODE_EXPLORE)
            # Every explore stage must set an explicit preset; otherwise opening
            # e.g. the Train stage inherits a stale preset (Lineage), which puts
            # the Continuous Learning catalog in the main pane and the training
            # detail in the bottom tray. Stages 0 (scenario config) and 2
            # (training) are the train workbench, so they pin PRESET_TRAIN.
            if idx == 3:
                preset = WorkbenchSplitHost.PRESET_EVAL
            elif idx == 4:
                preset = WorkbenchSplitHost.PRESET_LINEAGE
            else:
                preset = WorkbenchSplitHost.PRESET_TRAIN
            self._workbench_split_host.apply_preset(preset)
        self._layout_refresh_timer.start()

    def _wrap_scroll_page(self, page: QWidget) -> QWidget:
        # Keep workbench pages unparented until a visible container explicitly
        # inserts them. If the main window owns an inactive wrapper directly,
        # Qt can paint it at (0, 0), which leaks panels over the top nav.
        return _ScrollTabPage(page)

    def _set_eco_placeholder(self, title: str, detail: str = "") -> None:
        ph = getattr(self, "_eco_placeholder", None)
        if ph is None:
            return
        text = str(title or "").strip()
        extra = str(detail or "").strip()
        ph.setText(f"{text}\n\n{extra}" if extra else text)

    def _warm_webengine(self) -> None:
        """Pre-initialize the QtWebEngine/Chromium runtime off the critical path.

        The first QWebEngineView the app constructs pays a heavy one-time cost:
        QtWebEngine spawns the Chromium helper process and loads ~100MB of
        resources from disk. Deferred lazily, that cost lands on the UI thread
        the moment the user first opens the Ecosystem graph -- and if a training
        run is in flight (hammering the disk with cache/checkpoint writes), the
        cold init stalls for many seconds, producing the macOS spinner and the
        'Ecosystem hangs until training finishes' symptom.

        Creating one throwaway view here warms the process-global Chromium
        context while the disk is idle; every later view (the real ontology
        graph) is then cheap. The throwaway is never parented into the visible
        hierarchy nor shown, so it cannot trigger the macOS web-view z-order
        bug the rest of this file guards against.

        Measured cost: the first QWebEngineView is ~4ms, but the first setHtml
        blocks the UI thread ~2.6s (render-process cold spin) before the rest of
        the page loads asynchronously. We gate the real Ecosystem panel build on
        this warm completing (see _ensure_ontology_panel), so that one-time stall
        is paid here -- on a responsive placeholder you can navigate away from --
        instead of trapping you the moment you land on the Ecosystem graph.
        Opt out with CVOPS_WARM_WEBENGINE=0 (the panel then builds eagerly)."""
        if self._webengine_warmed:
            return
        self._webengine_warmed = True
        if str(os.environ.get("CVOPS_WARM_WEBENGINE", "1")).strip() in {"0", "false", "False"}:
            # Opted out: treat the engine as ready so the panel builds on demand
            # (eager cold init -- the legacy behavior).
            self._webengine_ready = True
            _mark_boot("webengine_ready", once=True, source="warm_disabled")
            self._maybe_build_pending_ontology()
            return
        # If the real Ecosystem view already exists, the engine is already warm.
        if self._ontology_panel is not None:
            self._webengine_ready = True
            _mark_boot("webengine_ready", once=True, source="ontology_panel_exists")
            return
        try:
            _mark_boot("webengine_lazy_import_start", once=True)
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            _mark_boot("webengine_lazy_import_done", once=True)
        except Exception:
            # No WebEngine available: let the panel build attempt its own
            # fallback rather than blocking forever on a warm that can't happen.
            self._webengine_ready = True
            _mark_boot("webengine_ready", once=True, source="unavailable")
            self._maybe_build_pending_ontology()
            return
        try:
            warm = QWebEngineView()  # no parent, never shown
            warm.loadFinished.connect(self._on_webengine_warm_finished)
            warm.setHtml("<html><body></body></html>")
            self._webengine_warm_view = warm
            # Safety net: if loadFinished never arrives, force-ready so the
            # Ecosystem cannot get stuck on the placeholder indefinitely.
            QTimer.singleShot(12000, self._on_webengine_warm_finished)
            _mark_boot("webengine_warm_start", once=True)
            _bprint("webengine runtime warming")
        except Exception:
            self._webengine_warm_view = None
            self._webengine_ready = True
            _mark_boot("webengine_ready", once=True, source="warm_failed")
            self._maybe_build_pending_ontology()

    def _boot_warm_webengine_sync(self) -> None:
        """Warm QtWebEngine synchronously during startup (before the window is
        shown), used only when restoring directly into the Ecosystem plane.

        The first setHtml blocks ~2.6s for the Chromium render-process cold spin;
        paying it here means the restored Ecosystem graph builds instantly rather
        than freezing the moment it appears. The render process persists
        process-wide once spawned, so subsequent views are cheap. This is the
        deliberate startup-cost trade chosen for the restore-into-Ecosystem case;
        all other paths warm lazily via _warm_webengine. Opt out with
        CVOPS_WARM_WEBENGINE=0."""
        if self._webengine_warmed:
            return
        if str(os.environ.get("CVOPS_WARM_WEBENGINE", "1")).strip() in {"0", "false", "False"}:
            return
        try:
            _mark_boot("webengine_sync_import_start", once=True)
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            _mark_boot("webengine_sync_import_done", once=True)
        except Exception:
            return
        try:
            self._webengine_warmed = True
            warm = QWebEngineView()  # no parent, never shown
            warm.setHtml("<html><body></body></html>")  # synchronous render-process spawn
            self._webengine_warm_view = warm
            self._webengine_ready = True
            _mark_boot("webengine_ready", once=True, source="sync_warm")
            # Release the throwaway once the loop is running; the global Chromium
            # context stays initialized for the app lifetime.
            QTimer.singleShot(3000, self._release_webengine_warm)
            _bprint("webengine runtime warmed (boot, ecosystem restore)")
        except Exception:
            self._webengine_warm_view = None

    def _on_webengine_warm_finished(self, *_args: Any) -> None:
        if self._webengine_ready:
            return
        self._webengine_ready = True
        _mark_boot("webengine_ready", once=True, source="warm_finished")
        _bprint("webengine runtime warmed")
        self._release_webengine_warm()
        self._maybe_build_pending_ontology()

    def _release_webengine_warm(self) -> None:
        view = self._webengine_warm_view
        self._webengine_warm_view = None
        if view is not None:
            # Drop on the next tick; the global Chromium context persists for the
            # app lifetime once initialized, so the engine stays warm.
            try:
                QTimer.singleShot(0, view.deleteLater)
            except Exception:
                pass

    def _maybe_build_pending_ontology(self) -> None:
        """Build the Ecosystem panel now that the engine is warm, if the user is
        still on the Ecosystem plane and asked for it. If they navigated away,
        clear the pending flag -- returning to Ecosystem builds it then (fast,
        because the engine is already warm)."""
        if not self._ontology_build_pending:
            return
        self._ontology_build_pending = False
        if self._ontology_panel is None and self._plane_stack.currentIndex() == 0:
            QTimer.singleShot(0, self._ensure_ontology_panel)

    def _start_ontology_background_load(self, *, attach_when_ready: bool) -> None:
        if self._ontology_panel is not None:
            return
        # Start the WebEngine warm now so its one-time cold init runs in parallel
        # with the threaded graph fetch below, rather than serially after it.
        self._warm_webengine()
        self._ontology_attach_after_warmup = self._ontology_attach_after_warmup or bool(attach_when_ready)
        if self._ontology_warmup_ready:
            if attach_when_ready and self._plane_stack.currentIndex() == 0:
                self._set_eco_placeholder("Opening Ecosystem...")
                QTimer.singleShot(0, self._ensure_ontology_panel)
            return
        worker = self._ontology_warmup_worker
        if worker is not None and worker.isRunning():
            return

        service = getattr(getattr(self, "_server", None), "service", None)
        direct_loader = None
        if service is not None and self._prefer_direct_service_reads:
            direct_loader = lambda layer, svc=service: svc._get_ontology_graph_cached(layer=layer)

        self._set_eco_placeholder(
            "Preparing Ecosystem...",
            "Graph cache is warming in the background.",
        )
        worker = _EcosystemWarmupWorker(self.base_url, direct_loader=direct_loader, parent=self)
        worker.done.connect(self._on_ontology_warmup_done)
        worker.failed.connect(self._on_ontology_warmup_failed)
        worker.finished.connect(lambda: setattr(self, "_ontology_warmup_worker", None))
        worker.finished.connect(worker.deleteLater)
        self._ontology_warmup_worker = worker
        worker.start()

    def _on_ontology_warmup_done(self, payload: dict[str, Any]) -> None:
        self._ontology_warmup_ready = True
        core = payload.get("core") if isinstance(payload, dict) else {}
        full = payload.get("full") if isinstance(payload, dict) else {}
        core_count = len(core.get("nodes") or []) if isinstance(core, dict) else 0
        full_count = len(full.get("nodes") or []) if isinstance(full, dict) else 0
        self._set_eco_placeholder(
            "Ecosystem ready.",
            f"Core {core_count} nodes / full {full_count} nodes.",
        )
        if self._ontology_attach_after_warmup and self._plane_stack.currentIndex() == 0:
            QTimer.singleShot(0, self._ensure_ontology_panel)

    def _on_ontology_warmup_failed(self, message: str) -> None:
        self._ontology_attach_after_warmup = False
        self._set_eco_placeholder(
            "Ecosystem is still preparing.",
            "The backend was not ready yet. Click Ecosystem again to retry.",
        )
        self._append_error("ecosystem", f"background load failed: {message}")

    def _ensure_ontology_panel(self) -> None:
        """Build OntologyPanel on first call and swap it into the ecosystem plane."""
        if self._ontology_panel is not None:
            return
        # The heavy part of building the panel is the first QtWebEngine setHtml,
        # which blocks the UI thread ~2.6s (more under disk load) for the Chromium
        # render-process cold spin. Defer the build until the engine is warmed in
        # the background so we never trap the user on a frozen Ecosystem -- the
        # placeholder stays interactive and they can switch tabs meanwhile.
        if not self._webengine_ready:
            self._ontology_build_pending = True
            self._set_eco_placeholder(
                "Preparing Ecosystem...",
                "Initializing the graph engine. You can switch to another tab "
                "and come back -- it will keep loading.",
            )
            self._warm_webengine()
            return
        from .ui.ontology_panel import OntologyPanel
        self._ontology_panel = OntologyPanel(base_url=self.base_url, parent=self)
        self._ontology_warmup_ready = True
        self._ontology_panel.navigateRequested.connect(self._on_eco_navigate)
        self._ontology_panel.entitySelected.connect(self._on_eco_entity_selected)
        self._ontology_panel.jobSubmitted.connect(self._on_job_submitted)
        eco_l = self._plane_ecosystem.layout()
        if eco_l is not None:
            ph = getattr(self, "_eco_placeholder", None)
            if ph is not None:
                eco_l.removeWidget(ph)
                ph.setParent(None)
                self._eco_placeholder = None
            eco_l.addWidget(self._ontology_panel, stretch=1)

    def _show_plane_ecosystem(self) -> None:
        self._plane_stack.setCurrentIndex(0)
        self._activity_rail.set_plane_ecosystem(True)
        self._cvops_settings.ui_shell_plane = "ecosystem"
        if self._ontology_panel is None:
            if self._ontology_warmup_ready:
                self._set_eco_placeholder("Opening Ecosystem...")
                QTimer.singleShot(0, self._ensure_ontology_panel)
            else:
                self._start_ontology_background_load(attach_when_ready=True)
        self._layout_refresh_timer.start()
        # Showing the web-view plane is the exact moment AppKit re-stacks
        # NSViews and tries to demote the top-nav strip; re-pin now and once
        # more on the next event-loop tick after Qt finishes the swap.
        self._lock_top_nav_z_order()
        QTimer.singleShot(0, self._lock_top_nav_z_order)

    def _show_plane_workbench(self) -> None:
        self._plane_stack.setCurrentIndex(1)
        self._activity_rail.set_plane_ecosystem(False)
        self._cvops_settings.ui_shell_plane = "workbench"
        self._layout_refresh_timer.start()
        self._lock_top_nav_z_order()

    def _set_workbench_mode(self, mode_id: str) -> None:
        self._show_plane_workbench()
        mid = str(mode_id or WorkbenchSplitHost.MODE_EXPLORE).strip().lower()
        # Legacy "scrape" slug routes to the new Collect mode.
        if mid == "scrape":
            mid = WorkbenchSplitHost.MODE_COLLECT
        self._workbench_split_host.set_mode(mid)
        self._activity_rail.set_workbench_mode(mid)
        self._cvops_settings.ui_shell_mode = mid
        if mid == WorkbenchSplitHost.MODE_PORTAL:
            self._refresh_dashboard_status()
            self._refresh_portal_status()
        self._layout_refresh_timer.start()

    def _on_activity_rail_eco(self) -> None:
        if self._plane_stack.currentIndex() == 0:
            self._show_plane_workbench()
            self._set_workbench_mode(self._cvops_settings.ui_shell_mode or "explore")
        else:
            self._show_plane_ecosystem()
        self._persist_ui_shell_state()

    def _on_activity_rail_mode(self, mode_id: str) -> None:
        mid = str(mode_id or "").strip().lower()
        # "scrape" is the legacy slug for Collect; map it through transparently.
        if mid == "scrape":
            mid = "collect"
        self._set_workbench_mode(mid)
        # The "Train" tab is the explore mode; it must land on the Train layout
        # preset. Without this it inherits a stale preset (e.g. Lineage), which
        # swaps the panes — Continuous Learning catalog in the main pane and the
        # training detail in the bottom tray.
        if mid == WorkbenchSplitHost.MODE_EXPLORE:
            self._workbench_split_host.apply_preset(WorkbenchSplitHost.PRESET_TRAIN)
            self._cvops_settings.ui_shell_preset = WorkbenchSplitHost.PRESET_TRAIN
        self._persist_ui_shell_state()

    def _on_workbench_split_moved(self) -> None:
        self._sync_catalog_toggle_btn()
        self._sync_bottom_pane_toggle_btn()
        self._persist_ui_shell_state()

    def _restore_workbench_panes(self) -> None:
        try:
            self._workbench_split_host.reopen_all_panes()
            self._cvops_settings.ui_shell_closed_panes = ""
            self._sync_catalog_toggle_btn()
            self._sync_bottom_pane_toggle_btn()
            self._persist_ui_shell_state()
            self._layout_refresh_timer.start()
        except Exception:
            pass

    def _restore_ui_shell_state(self) -> None:
        plane = str(self._cvops_settings.ui_shell_plane or "workbench").strip().lower()
        if plane == "ecosystem":
            self._plane_stack.setCurrentIndex(0)
            self._activity_rail.set_plane_ecosystem(True)
            self._set_eco_placeholder(
                "Preparing Ecosystem...",
                "The graph will finish loading after the backend is ready.",
            )
            QTimer.singleShot(
                750,
                lambda: self._start_ontology_background_load(attach_when_ready=True),
            )
        else:
            self._plane_stack.setCurrentIndex(1)
            self._activity_rail.set_plane_ecosystem(False)
        mode = str(self._cvops_settings.ui_shell_mode or "explore").strip().lower()
        if mode in {"collect", "scrape"}:
            self._workbench_split_host.set_mode(WorkbenchSplitHost.MODE_COLLECT)
            self._activity_rail.set_workbench_mode("collect")
        else:
            self._workbench_split_host.set_mode(mode)
            self._activity_rail.set_workbench_mode(mode)
        preset = str(self._cvops_settings.ui_shell_preset or "train").strip().lower()
        if preset not in {"train", "eval", "lineage"}:
            preset = "train"
        self._workbench_split_host.apply_preset(preset)
        data: dict[str, bytes] = {}
        try:
            o = self._cvops_settings.ui_shell_split_outer_b64
            if o:
                data["outer"] = base64.standard_b64decode(o.encode("ascii"))
            e = self._cvops_settings.ui_shell_split_explorer_b64
            if e:
                data["explorer_tri"] = base64.standard_b64decode(e.encode("ascii"))
            s = self._cvops_settings.ui_shell_split_settings_b64
            if s:
                data["settings_diag"] = base64.standard_b64decode(s.encode("ascii"))
            c = self._cvops_settings.ui_shell_closed_panes
            if c:
                data["closed_panes"] = c.encode("ascii", errors="ignore")
        except Exception:
            data = {}
        if data:
            self._workbench_split_host.restore_split_state(data)

    def _persist_ui_shell_state(self) -> None:
        try:
            st = self._workbench_split_host.save_split_state()
            self._cvops_settings.ui_shell_split_outer_b64 = base64.standard_b64encode(
                st.get("outer", b"")
            ).decode("ascii")
            self._cvops_settings.ui_shell_split_explorer_b64 = base64.standard_b64encode(
                st.get("explorer_tri", b"")
            ).decode("ascii")
            self._cvops_settings.ui_shell_split_settings_b64 = base64.standard_b64encode(
                st.get("settings_diag", b"")
            ).decode("ascii")
            closed = st.get("closed_panes", b"")
            if isinstance(closed, bytes):
                self._cvops_settings.ui_shell_closed_panes = closed.decode("ascii", errors="ignore")
            else:
                self._cvops_settings.ui_shell_closed_panes = ""
            save_cvops_settings(self._settings_path, self._cvops_settings)
        except Exception:
            pass

    def _on_initialize_scope_clicked(self) -> None:
        if self._portal_proc is not None and self._portal_proc.poll() is None:
            self._refresh_portal_status("Scope is already running.")
            return
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Initialize Scope")
        msg.setText("Start the separate Scope window?")
        msg.setInformativeText(
            "This spawns a new Insight Local process in its own window (outside CV Ops). "
            "Use it when you want live video and previews without blocking this UI. "
            "You can leave this tab anytime; the Scope window runs independently until you stop it here."
        )
        msg.setStandardButtons(
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
        )
        msg.setDefaultButton(QMessageBox.StandardButton.Cancel)
        if msg.exec() != QMessageBox.StandardButton.Ok:
            return
        self._portal_loaded = True
        self._start_portal()

    def _portal_env(self) -> dict[str, str]:
        env = dict(os.environ)
        package_root = str((self._project_root / "Insight").resolve())
        existing = str(env.get("PYTHONPATH", "") or "")
        env["PYTHONPATH"] = package_root if not existing else f"{package_root}{os.pathsep}{existing}"
        return env

    def _start_portal(self) -> None:
        if self._portal_proc is not None and self._portal_proc.poll() is not None:
            self._portal_proc = None
        if self._portal_proc is not None and self._portal_proc.poll() is None:
            self._refresh_portal_status("Scope is already running.")
            return
        cmd = [sys.executable, "-m", "insight_local"]
        try:
            self._portal_proc = subprocess.Popen(
                cmd,
                cwd=str(self._project_root),
                env=self._portal_env(),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._refresh_portal_status("Scope process started.")
        except Exception as exc:
            self._portal_proc = None
            msg = f"Failed to start Scope: {exc}"
            self._append_error("portal", msg)
            self._refresh_portal_status(msg)

    def _stop_portal(self) -> None:
        proc = self._portal_proc
        if proc is None:
            self._refresh_portal_status("Scope is already stopped.")
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            except Exception as exc:
                self._append_error("portal", f"Failed to stop Scope: {exc}")
        self._portal_proc = None
        self._portal_loaded = False
        self._refresh_portal_status("Scope stopped.")

    def _refresh_portal_status(self, note: str = "") -> None:
        proc = self._portal_proc
        prefix = f"{note} " if note else ""
        if proc is not None and proc.poll() is None:
            text = f"{prefix}Status: running (PID {proc.pid})"
            running = True
        elif proc is not None and proc.poll() is not None:
            text = f"{prefix}Status: stopped (exit {proc.poll()})"
            running = False
        else:
            text = f"{prefix}Status: stopped"
            running = False
        if self._portal_status is not None:
            self._portal_status.setText(text)
        if self._portal_init_btn is not None:
            self._portal_init_btn.setEnabled(not running)
        if self._portal_stop_btn is not None:
            self._portal_stop_btn.setEnabled(running)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._layout_refresh_timer.start()
        self._schedule_notification_tray_layout()
        self._update_ai_assistant_geometry()
        if hasattr(self, "_connection_overlay"):
            self._connection_overlay.resize(self.size())
            if self._connection_overlay.isVisible():
                # Screen/DPI changes can promote transparent child widgets into
                # AppKit's hit-test stack. Drop transient connection lines
                # during resize so the full-window overlay cannot trap input.
                self._connection_overlay.clear()
        self._lock_top_nav_z_order()

    def moveEvent(self, event) -> None:  # type: ignore[override]
        super().moveEvent(event)
        if hasattr(self, "_connection_overlay") and self._connection_overlay.isVisible():
            self._connection_overlay.clear()
        self._schedule_notification_tray_layout()
        self._update_ai_assistant_geometry()
        QTimer.singleShot(0, self._lock_top_nav_z_order)

    def _schedule_notification_tray_layout(self) -> None:
        tray = getattr(self, "_notification_tray", None)
        if tray is not None and tray.card_count() > 0 and tray.isVisible():
            self._update_notification_tray_geometry()
        timer = getattr(self, "_notification_tray_layout_timer", None)
        if timer is not None:
            timer.start()
        self._update_ai_assistant_geometry()

    def _update_notification_tray_geometry(self) -> None:
        tray = getattr(self, "_notification_tray", None)
        if tray is None:
            return
        if tray.isHidden() or tray.card_count() <= 0:
            return

        root = getattr(self, "_workbench_root", None)
        anchor_widget = getattr(self, "_workspace_host", None)
        if root is None or anchor_widget is None:
            return

        try:
            anchor_top_left = anchor_widget.mapTo(self, QPoint(0, 0))
        except Exception:
            anchor_top_left = QPoint(0, 0)
        anchor_rect = QRect(anchor_top_left, anchor_widget.size())
        if anchor_rect.width() <= 0 or anchor_rect.height() <= 0:
            anchor_rect = self.contentsRect()

        tray.adjustSize()
        hint = tray.sizeHint().expandedTo(tray.minimumSizeHint())
        margin = 12
        max_width = max(240, anchor_rect.width() - (margin * 2))
        width = min(hint.width(), max_width)
        height = hint.height()
        if width <= 0 or height <= 0:
            return

        x = anchor_rect.right() - width - margin
        y = anchor_rect.top() + margin
        min_x = anchor_rect.left() + margin
        min_y = anchor_rect.top() + margin
        tray.resize(width, height)
        tray.move(max(min_x, x), max(min_y, y))
        tray.raise_()

    def _update_ai_assistant_geometry(self) -> None:
        win = getattr(self, "_ai_assistant_window", None)
        if win is None or win.isHidden():
            return

        anchor_widget = getattr(self, "_workspace_host", None)
        if anchor_widget is None:
            anchor_rect = self.contentsRect()
        else:
            try:
                anchor_top_left = anchor_widget.mapTo(self, QPoint(0, 0))
            except Exception:
                anchor_top_left = QPoint(0, 0)
            anchor_rect = QRect(anchor_top_left, anchor_widget.size())
            if anchor_rect.width() <= 0 or anchor_rect.height() <= 0:
                anchor_rect = self.contentsRect()

        place_in_parent = getattr(win, "place_in_parent", None)
        if callable(place_in_parent):
            place_in_parent(anchor_rect, margin=12)
            return

        hint = win.sizeHint().expandedTo(win.minimumSizeHint())
        margin = 12
        width = min(max(hint.width(), win.minimumWidth()), max(1, anchor_rect.width() - (margin * 2)))
        height = min(max(hint.height(), win.minimumHeight()), max(1, anchor_rect.height() - (margin * 2)))
        x = anchor_rect.right() - width - margin
        y = max(anchor_rect.top() + margin, anchor_rect.bottom() - height - margin)
        win.resize(width, height)
        win.move(max(anchor_rect.left() + margin, x), max(anchor_rect.top() + margin, y))
        win.raise_()

    def _lock_top_nav_z_order(self) -> None:
        """Re-pin the top-nav strip above the ecosystem QWebEngineView.

        AppKit can re-stack NSViews when siblings get added/removed/resized
        (collapsing the notification strip, switching planes, dragging a
        splitter). Each of those events can quietly drop the top-nav NSView
        behind the web view's NSView, killing click delivery to the tabs.
        Re-raising on every reshuffle keeps it pinned.
        """
        container = getattr(self, "_top_nav_container", None)
        if container is None:
            return
        try:
            if not container.testAttribute(Qt.WidgetAttribute.WA_NativeWindow):
                container.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
                container.winId()
            container.raise_()
        except Exception:
            pass

    def _build_overview_subtab(self) -> QWidget:
        from .ui.dashboard_widgets import DashboardOverviewWidget
        from .ui.getting_started_guide import GettingStartedGuide
        tab = QWidget()
        dl = QVBoxLayout(tab)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(6)

        # New-user guide sits at the top of the overview, above Live Activity.
        guide_section = CollapsibleSection("Getting Started — How to use CV Ops", expanded=True)
        self._getting_started_guide = GettingStartedGuide()
        guide_section.body_layout().addWidget(self._getting_started_guide)
        dl.addWidget(guide_section)

        section = CollapsibleSection("Live Activity", expanded=True)
        root = section.body_layout()

        info = QLabel(
            "Native service, scenario, and job visuals stay here. Open the full Streamlit dashboard in Scope when you need the web view."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        controls = QHBoxLayout()
        start_btn = QPushButton("Start Dashboard")
        start_btn.clicked.connect(self._start_dashboard)
        controls.addWidget(start_btn)
        scope_btn = QPushButton("Open in Scope")
        scope_btn.clicked.connect(self._open_dashboard_in_scope)
        controls.addWidget(scope_btn)
        reload_btn = QPushButton("Reload")
        reload_btn.clicked.connect(self._reload_dashboard_view)
        controls.addWidget(reload_btn)
        open_btn = QPushButton("Open in Browser")
        open_btn.clicked.connect(self._open_dashboard)
        controls.addWidget(open_btn)
        stop_btn = QPushButton("Stop Dashboard")
        stop_btn.clicked.connect(self._stop_dashboard)
        controls.addWidget(stop_btn)
        controls.addStretch(1)
        root.addLayout(controls)

        self._dashboard_overview = DashboardOverviewWidget()
        root.addWidget(self._dashboard_overview)

        self._dashboard_status = QLabel("")
        self._dashboard_status.setObjectName("dashboardStatus")
        self._dashboard_status.setWordWrap(True)
        root.addWidget(self._dashboard_status)

        monitor_host = QWidget()
        monitor_host.setObjectName("dashboardNativePanel")
        monitor_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        monitor_layout = QVBoxLayout(monitor_host)
        monitor_layout.setContentsMargins(8, 7, 8, 8)
        monitor_layout.setSpacing(6)
        monitor_title = QLabel("Live Activity")
        monitor_title.setProperty("isTitle", True)
        monitor_layout.addWidget(monitor_title)

        self._dashboard_health_summary = QLabel("Service health: waiting for data.")
        self._dashboard_health_summary.setObjectName("dashboardSummaryLine")
        self._dashboard_health_summary.setWordWrap(True)
        monitor_layout.addWidget(self._dashboard_health_summary)
        self._dashboard_scenario_summary = QLabel("Scenario readiness: waiting for data.")
        self._dashboard_scenario_summary.setObjectName("dashboardSummaryLine")
        self._dashboard_scenario_summary.setWordWrap(True)
        monitor_layout.addWidget(self._dashboard_scenario_summary)

        self._dashboard_jobs_table = QTableWidget(0, 6)
        self._dashboard_jobs_table.setHorizontalHeaderLabels(
            ["Job ID", "Type", "Scenario", "State", "Created", "Finished"]
        )
        self._dashboard_jobs_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._dashboard_jobs_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._dashboard_jobs_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._dashboard_jobs_table.setAlternatingRowColors(True)
        dh = self._dashboard_jobs_table.horizontalHeader()
        dh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        dh.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        dh.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        dh.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        dh.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        dh.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        monitor_layout.addWidget(self._dashboard_jobs_table, stretch=1)
        root.addWidget(monitor_host, stretch=1)

        dl.addWidget(section, stretch=1)
        self._refresh_dashboard_status()
        self._refresh_dashboard_scenario_summary()
        return tab

    def _build_settings_subtab(self) -> QWidget:
        tab = QWidget()
        sl = QVBoxLayout(tab)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.setSpacing(6)
        self._settings_panel = CvOpsSettingsPanel(
            settings_path=self._settings_path,
            settings=self._cvops_settings,
            host=self.host,
            port=self.port,
            dashboard_url=self._dashboard_url,
            state_dir=CVOPS_STATE_DIR,
            jobs_db_path=CVOPS_DB_PATH,
        )
        self._settings_panel.colorSchemeChanged.connect(self._on_settings_color_scheme)
        self._settings_panel.buttonShapeChanged.connect(self._on_settings_button_shape)
        self._settings_panel.uiScaleChanged.connect(self._on_settings_ui_scale)
        self._settings_panel.themeOverridesChanged.connect(self._on_settings_theme_overrides)
        self._settings_panel.timeFormatChanged.connect(self._on_settings_time_format)
        self._settings_panel.dashboardSettingsChanged.connect(self._on_settings_dashboard)
        self._settings_panel.dashboardOpenScopeRequested.connect(self._open_dashboard_in_scope)
        self._settings_panel.pollIntervalsChanged.connect(self._on_settings_poll_intervals)
        self._settings_panel.dashboardStartRequested.connect(self._start_dashboard)
        self._settings_panel.dashboardStopRequested.connect(self._stop_dashboard)
        self._settings_panel.dashboardOpenRequested.connect(self._open_dashboard)
        self._settings_panel.dashboardReloadRequested.connect(self._reload_dashboard_view)
        self._settings_panel.workspaceBackdropChanged.connect(self._apply_cvops_stylesheet)
        self._settings_panel.eventPulseVisibilityChanged.connect(self._on_settings_event_pulse_visibility)
        sl.addWidget(self._settings_panel, stretch=1)
        return tab

    def _dashboard_embed_supported(self) -> bool:
        if self._dashboard_web_supported:
            return True
        try:
            from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings  # noqa: F401
            from PyQt6.QtWebEngineWidgets import QWebEngineView  # noqa: F401
        except Exception:
            return False
        self._dashboard_web_supported = True
        return True

    def _build_dashboard_web_widget(self, autoload: bool = True) -> QWidget:
        try:
            from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
            from PyQt6.QtWebEngineWidgets import QWebEngineView
        except Exception as exc:
            self._dashboard_web_supported = False
            fallback = QLabel(
                "Embedded browser unavailable. Install PyQt6-WebEngine or run CV Ops via package entrypoint.\n"
                f"Reason: {exc}"
            )
            fallback.setObjectName("dashboardFallback")
            fallback.setWordWrap(True)
            fallback.setAlignment(Qt.AlignmentFlag.AlignCenter)
            return fallback

        class _DashboardPage(QWebEnginePage):
            def javaScriptConsoleMessage(
                self, level, message, line_number, source_id
            ) -> None:  # type: ignore[override]
                text = str(message or "")
                if "generate_204" in text and "preloaded" in text:
                    return
                print(f"[CVOPS DASHBOARD JS] {source_id}:{line_number} {text}", flush=True)

            def acceptNavigationRequest(
                self, url: QUrl, navigation_type, is_main_frame: bool
            ) -> bool:  # type: ignore[override]
                scheme = url.scheme().lower()
                if scheme in {"", "about", "data", "file", "http", "https", "qrc"}:
                    return super().acceptNavigationRequest(url, navigation_type, is_main_frame)
                QDesktopServices.openUrl(url)
                return False

            def createWindow(self, window_type):  # type: ignore[override]
                class _PopupPage(QWebEnginePage):
                    def acceptNavigationRequest(
                        self, url: QUrl, navigation_type, is_main_frame: bool
                    ) -> bool:  # type: ignore[override]
                        if url.scheme().lower() in {"http", "https", "file"}:
                            QDesktopServices.openUrl(url)
                        self.deleteLater()
                        return False

                return _PopupPage(self)

        view = QWebEngineView()
        view.setObjectName("dashboardWebView")
        view.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
        view.setPage(_DashboardPage(view))
        settings = view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, True)
        self._dashboard_web_supported = True
        if autoload:
            view.load(QUrl(self._dashboard_url))
        return view

    def _build_portal_tab(self) -> QWidget:
        root = QWidget()
        root.setObjectName("portalTab")
        layout = QVBoxLayout(root)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(12)

        title = QLabel("Scope")
        title.setProperty("isTitle", True)
        layout.addWidget(title)

        detail = QLabel(
            "Choose which surface to launch here: the separate Insight Scope window, "
            "or the embedded web-based dashboard."
        )
        detail.setWordWrap(True)
        detail.setProperty("muted", True)
        layout.addWidget(detail)

        card_row = QHBoxLayout()
        card_row.setSpacing(12)

        portal_shell = QWidget()
        portal_shell.setObjectName("dashboardNativePanel")
        portal_shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        portal_shell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        portal_layout = QVBoxLayout(portal_shell)
        portal_layout.setContentsMargins(8, 7, 8, 8)
        portal_layout.setSpacing(6)
        portal_title = QLabel("Separate Scope Window")
        portal_title.setProperty("isTitle", True)
        portal_layout.addWidget(portal_title)
        portal_hint = QLabel(
            "Spawns the standalone Insight Scope window in a separate process."
        )
        portal_hint.setWordWrap(True)
        portal_hint.setProperty("muted", True)
        portal_layout.addWidget(portal_hint)
        portal_controls = QHBoxLayout()
        portal_controls.setSpacing(6)
        self._portal_init_btn = QPushButton("Launch Scope")
        self._portal_init_btn.setProperty("isPrimary", True)
        self._portal_init_btn.setMinimumHeight(42)
        self._portal_init_btn.clicked.connect(self._on_initialize_scope_clicked)
        portal_controls.addWidget(self._portal_init_btn)
        self._portal_stop_btn = QPushButton("Stop Scope")
        self._portal_stop_btn.clicked.connect(self._stop_portal)
        portal_controls.addWidget(self._portal_stop_btn)
        portal_controls.addStretch(1)
        portal_layout.addLayout(portal_controls)
        self._portal_status = QLabel("")
        self._portal_status.setObjectName("dashboardStatus")
        self._portal_status.setWordWrap(True)
        portal_layout.addWidget(self._portal_status)
        portal_preview = QWidget()
        portal_preview.setObjectName("dashboardEmbedPanel")
        portal_preview.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        portal_preview.setMinimumHeight(300)
        portal_preview_layout = QVBoxLayout(portal_preview)
        portal_preview_layout.setContentsMargins(18, 18, 18, 18)
        portal_preview_layout.setSpacing(8)
        portal_preview_label = QLabel(
            "Launch the standalone Insight Scope here.\n"
            "It opens in a separate window, while this page remains the control surface."
        )
        portal_preview_label.setWordWrap(True)
        portal_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        portal_preview_label.setProperty("muted", True)
        portal_preview_layout.addStretch(1)
        portal_preview_layout.addWidget(portal_preview_label)
        portal_preview_layout.addStretch(1)
        portal_layout.addWidget(portal_preview, stretch=1)
        card_row.addWidget(portal_shell, stretch=1)

        dashboard_shell = QWidget()
        dashboard_shell.setObjectName("dashboardEmbedPanel")
        dashboard_shell.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        dashboard_shell.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        dashboard_layout = QVBoxLayout(dashboard_shell)
        dashboard_layout.setContentsMargins(8, 7, 8, 8)
        dashboard_layout.setSpacing(6)
        dashboard_title = QLabel("Web-Based Dashboard")
        dashboard_title.setProperty("isTitle", True)
        dashboard_layout.addWidget(dashboard_title)
        dashboard_hint = QLabel(
            "Loads the local Streamlit dashboard inside this tab."
        )
        dashboard_hint.setWordWrap(True)
        dashboard_hint.setProperty("muted", True)
        dashboard_layout.addWidget(dashboard_hint)
        dashboard_controls = QHBoxLayout()
        dashboard_controls.setSpacing(6)
        self._scope_dashboard_launch_btn = QPushButton("Launch Web Dashboard")
        self._scope_dashboard_launch_btn.setProperty("isPrimary", True)
        self._scope_dashboard_launch_btn.setMinimumHeight(42)
        self._scope_dashboard_launch_btn.clicked.connect(self._launch_dashboard_from_scope)
        dashboard_controls.addWidget(self._scope_dashboard_launch_btn)
        self._scope_dashboard_reload_btn = QPushButton("Reload")
        self._scope_dashboard_reload_btn.clicked.connect(self._reload_dashboard_view)
        dashboard_controls.addWidget(self._scope_dashboard_reload_btn)
        self._scope_dashboard_open_btn = QPushButton("Open in Browser")
        self._scope_dashboard_open_btn.clicked.connect(self._open_dashboard)
        dashboard_controls.addWidget(self._scope_dashboard_open_btn)
        self._scope_dashboard_stop_btn = QPushButton("Stop Dashboard")
        self._scope_dashboard_stop_btn.clicked.connect(self._stop_dashboard)
        dashboard_controls.addWidget(self._scope_dashboard_stop_btn)
        dashboard_controls.addStretch(1)
        dashboard_layout.addLayout(dashboard_controls)
        self._scope_dashboard_status = QLabel("")
        self._scope_dashboard_status.setObjectName("dashboardStatus")
        self._scope_dashboard_status.setWordWrap(True)
        dashboard_layout.addWidget(self._scope_dashboard_status)

        embed_host = QWidget()
        embed_host.setObjectName("dashboardEmbedPanel")
        embed_host.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        embed_host.setMinimumHeight(300)
        embed_layout = QVBoxLayout(embed_host)
        embed_layout.setContentsMargins(0, 0, 0, 0)
        embed_layout.setSpacing(6)
        dashboard_running = self._dashboard_proc is not None and self._dashboard_proc.poll() is None
        dashboard_placeholder = QLabel(
            "Launch Web Dashboard to load the embedded Streamlit view here."
        )
        dashboard_placeholder.setWordWrap(True)
        dashboard_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        dashboard_placeholder.setProperty("muted", True)
        self._dashboard_web_view = self._build_dashboard_web_widget(autoload=dashboard_running)
        self._scope_dashboard_stack = QStackedWidget(embed_host)
        self._scope_dashboard_stack.addWidget(dashboard_placeholder)
        self._scope_dashboard_stack.addWidget(self._dashboard_web_view)
        self._scope_dashboard_stack.setCurrentIndex(1 if dashboard_running else 0)
        embed_layout.addWidget(self._scope_dashboard_stack, stretch=1)
        self._scope_dashboard_host = embed_host
        dashboard_layout.addWidget(embed_host, stretch=1)
        card_row.addWidget(dashboard_shell, stretch=1)
        layout.addLayout(card_row, stretch=1)

        self._refresh_portal_status("Choose Launch Scope to open the separate window.")
        self._refresh_dashboard_status(
            "Choose Launch Web Dashboard to load the embedded dashboard."
            if not dashboard_running
            else "Dashboard ready in Scope."
        )
        return root

    def _dashboard_app_path(self) -> Path:
        return self._project_root / "mlops" / "dashboard" / "app.py"

    def _start_dashboard(self) -> None:
        if self._dashboard_proc is not None and self._dashboard_proc.poll() is not None:
            self._dashboard_proc = None
        if self._dashboard_proc is not None and self._dashboard_proc.poll() is None:
            self._refresh_dashboard_status()
            return
        app_path = self._dashboard_app_path()
        if not app_path.exists():
            msg = f"Dashboard app not found: {app_path}"
            self._append_error("dashboard", msg)
            self._refresh_dashboard_status(msg)
            return
        if importlib.util.find_spec("streamlit") is None:
            msg = (
                "Streamlit is not installed. Run: "
                "python -m pip install -r mlops/dashboard/requirements.txt"
            )
            self._append_error("dashboard", msg)
            self._refresh_dashboard_status(msg)
            return
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--server.headless=true",
            "--server.port",
            str(self._dashboard_port),
            "--browser.gatherUsageStats=false",
        ]
        try:
            self._dashboard_proc = subprocess.Popen(
                cmd,
                cwd=str(self._project_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._load_dashboard_in_view()
            self._refresh_dashboard_status("Dashboard process started.")
        except Exception as exc:
            msg = f"Failed to start dashboard: {exc}"
            self._append_error("dashboard", msg)
            self._refresh_dashboard_status(msg)

    def _open_dashboard(self) -> None:
        if not QDesktopServices.openUrl(QUrl(self._dashboard_url)):
            self._append_error("dashboard", f"Failed to open URL: {self._dashboard_url}")
        self._refresh_dashboard_status()

    def _launch_dashboard_from_scope(self) -> None:
        host = self._scope_dashboard_host
        if host is not None:
            host.setVisible(True)
        if self._dashboard_proc is not None and self._dashboard_proc.poll() is None:
            self._load_dashboard_in_view()
            self._refresh_dashboard_status("Dashboard opened in Scope.")
        else:
            self._start_dashboard()
        self._layout_refresh_timer.start()

    def _open_dashboard_in_scope(self) -> None:
        self._set_workbench_mode(WorkbenchSplitHost.MODE_PORTAL)
        self._persist_ui_shell_state()
        host = self._scope_dashboard_host
        if self._dashboard_proc is not None and self._dashboard_proc.poll() is None and host is not None:
            host.setVisible(True)
        if self._dashboard_proc is not None and self._dashboard_proc.poll() is None:
            self._load_dashboard_in_view()
            self._refresh_dashboard_status("Dashboard opened in Scope.")
        else:
            self._refresh_dashboard_status("Scope ready. Launch Web Dashboard when you want the embedded dashboard.")

    def _stop_dashboard(self) -> None:
        proc = self._dashboard_proc
        if proc is None:
            self._refresh_dashboard_status("Dashboard is already stopped.")
            return
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    pass
            except Exception as exc:
                self._append_error("dashboard", f"Failed to stop dashboard: {exc}")
        self._dashboard_proc = None
        self._refresh_dashboard_status("Dashboard stopped.")

    def _reload_dashboard_view(self) -> None:
        if self._dashboard_web_view is None:
            self._refresh_dashboard_status("Open Scope to load the embedded dashboard browser.")
            return
        if self._dashboard_web_supported and hasattr(self._dashboard_web_view, "reload"):
            try:
                self._dashboard_web_view.reload()  # type: ignore[attr-defined]
            except Exception:
                pass
            return
        self._refresh_dashboard_status("Embedded browser unavailable; use Open in Browser.")

    def _load_dashboard_in_view(self) -> None:
        if (
            self._scope_dashboard_stack is not None
            and self._dashboard_web_view is not None
        ):
            self._scope_dashboard_stack.setCurrentWidget(self._dashboard_web_view)
        if self._dashboard_web_view is None or not self._dashboard_web_supported:
            return
        if hasattr(self._dashboard_web_view, "load"):
            try:
                self._dashboard_web_view.load(QUrl(self._dashboard_url))  # type: ignore[attr-defined]
            except Exception:
                pass

    def _refresh_dashboard_status(self, note: str = "") -> None:
        prefix = f"{note} " if note else ""
        proc = self._dashboard_proc
        suffix = ""
        embed_supported = self._dashboard_embed_supported()
        running = proc is not None and proc.poll() is None
        if not embed_supported:
            suffix = " · embed unavailable"
        if running:
            text = f"{prefix}Status: running (PID {proc.pid}) · URL: {self._dashboard_url}{suffix}"
        elif proc is not None and proc.poll() is not None:
            text = f"{prefix}Status: stopped (exit {proc.poll()}) · URL: {self._dashboard_url}{suffix}"
        else:
            text = f"{prefix}Status: stopped · URL: {self._dashboard_url}{suffix}"

        if self._dashboard_status is not None:
            self._dashboard_status.setText(text)
        if self._scope_dashboard_status is not None:
            self._scope_dashboard_status.setText(text)
        if self._settings_panel is not None:
            self._settings_panel.set_dashboard_status(text)
        if self._dashboard_overview is not None:
            self._dashboard_overview.set_service(
                running=running,
                url=self._dashboard_url,
                embed_supported=embed_supported,
                detail=text,
            )
        if self._scope_dashboard_launch_btn is not None:
            self._scope_dashboard_launch_btn.setEnabled(not running)
        if self._scope_dashboard_reload_btn is not None:
            self._scope_dashboard_reload_btn.setEnabled(
                running and embed_supported and self._dashboard_web_view is not None
            )
        if self._scope_dashboard_open_btn is not None:
            self._scope_dashboard_open_btn.setEnabled(True)
        if self._scope_dashboard_stop_btn is not None:
            self._scope_dashboard_stop_btn.setEnabled(running)
        if self._scope_dashboard_stack is not None:
            if running and self._dashboard_web_view is not None:
                self._scope_dashboard_stack.setCurrentWidget(self._dashboard_web_view)
            else:
                self._scope_dashboard_stack.setCurrentIndex(0)

    def _build_errors_subtab(self) -> QWidget:
        tab = QWidget()
        el = QVBoxLayout(tab)
        el.setContentsMargins(0, 0, 0, 0)
        el.setSpacing(6)
        section = CollapsibleSection("Errors", expanded=True)
        body = section.body_layout()
        row = QHBoxLayout()
        clear_btn = QPushButton("Clear Errors")
        clear_btn.clicked.connect(lambda: self._errors_text.clear())
        row.addWidget(clear_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        row.addStretch(1)
        body.addLayout(row)
        self._errors_text = QTextEdit()
        self._errors_text.setReadOnly(True)
        self._errors_text.setPlaceholderText("Runtime errors will appear here.")
        for line in self._pending_error_lines:
            self._errors_text.append(line)
        self._pending_error_lines.clear()
        body.addWidget(self._errors_text, stretch=1)
        el.addWidget(section, stretch=1)
        return tab

    def _confirm_close_program(self) -> bool:
        if self._close_confirmed:
            return True
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Question)
        msg.setWindowTitle("Close Program")
        msg.setText("Do you want to close the program?")
        close_btn = msg.addButton("Close", QMessageBox.ButtonRole.AcceptRole)
        return_btn = msg.addButton("Return", QMessageBox.ButtonRole.RejectRole)
        msg.setDefaultButton(return_btn)
        msg.setEscapeButton(return_btn)
        msg.exec()
        if msg.clickedButton() is not close_btn:
            return False
        self._close_confirmed = True
        return True

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if not self._confirm_close_program():
            event.ignore()
            return
        try:
            self._stop_portal()
        except Exception:
            pass
        try:
            self._health_timer.stop()
            self._dashboard_timer.stop()
        except Exception:
            pass
        try:
            self._stop_dashboard()
        except Exception:
            pass
        try:
            self._ws.stop()
        except Exception:
            pass
        try:
            worker = self._ws_resync_worker
            if worker is not None and worker.isRunning():
                worker.quit()
                worker.wait(500)
                if worker.isRunning():
                    worker.terminate()
                    worker.wait(500)
        except Exception:
            pass
        try:
            boot = getattr(self, "_server_boot", None)
            if boot is not None and boot.isRunning():
                boot.wait(1500)
        except Exception:
            pass
        try:
            if self._server is not None:
                self._server.stop(timeout=1.0)
        except Exception:
            pass
        if self._video_test_panel is not None:
            try:
                self._video_test_panel.shutdown_background_processes()
            except Exception:
                pass
        if self._split_magnifier_window is not None:
            try:
                self._split_magnifier_window.close()
            except Exception:
                pass
        if self._ai_assistant_window is not None:
            try:
                self._ai_assistant_window.close()
            except Exception:
                pass
        try:
            self._persist_ui_shell_state()
        except Exception:
            pass
        super().closeEvent(event)

    def _refresh_responsive_children(self) -> None:
        scale = resolve_ui_scale_factor(self, self._cvops_settings.ui_scale_pct)
        if self._last_ui_style_scale is None or abs(scale - self._last_ui_style_scale) >= 0.03:
            self._apply_cvops_stylesheet()
        else:
            apply_ui_scale_compaction(
                self,
                ui_scale_pct=self._cvops_settings.ui_scale_pct,
            )
        try:
            self._catalog_panel.refresh_responsive_layout()
        except Exception:
            pass
        if self._database_panel is not None:
            try:
                self._database_panel.refresh_responsive_layout()
            except Exception:
                pass
        try:
            self._result_panel.refresh_responsive_layout()
        except Exception:
            pass
        try:
            if self._test_range_result_panel is not None:
                self._test_range_result_panel.refresh_responsive_layout()
        except Exception:
            pass
        # The timer is started after every layout-affecting event (resize,
        # plane swap, splitter drag, section collapse, notification strip
        # hide/show). Re-pinning here covers all of them in one place.
        self._schedule_notification_tray_layout()
        self._lock_top_nav_z_order()

    # ---------- Settings ----------

    def _install_ui_scale_shortcuts(self) -> None:
        self._zoom_in_shortcuts: list[QShortcut] = []
        for seq in ("Ctrl+=", "Ctrl++"):
            shortcut = QShortcut(QKeySequence(seq), self)
            shortcut.activated.connect(lambda _seq=seq: self._nudge_ui_scale(5))
            self._zoom_in_shortcuts.append(shortcut)
        self._zoom_out_shortcut = QShortcut(QKeySequence("Ctrl+-"), self)
        self._zoom_out_shortcut.activated.connect(lambda: self._nudge_ui_scale(-5))
        self._zoom_reset_shortcut = QShortcut(QKeySequence("Ctrl+0"), self)
        self._zoom_reset_shortcut.activated.connect(self._reset_ui_scale)

    def _combined_stylesheet(self) -> str:
        wp = resolve_workspace_wallpaper_path(self._cvops_settings, CVOPS_STATE_DIR)
        blend = blend_from_cvops_settings(self._cvops_settings) if wp is not None else None
        rail_bg = (
            normalize_color_override(self._cvops_settings.ui_panel_background_color)
            or normalize_color_override(self._cvops_settings.ui_background_color)
            or "rgba(18,18,20,0.96)"
        )
        rail_text = normalize_color_override(self._cvops_settings.ui_muted_text_color) or "#999"
        rail_checked_text = cvops_color("selection_text")
        rail_accent = cvops_color("selection_active")
        rail_edge = cvops_color("selection_edge")
        rail_accent_color = QColor(rail_accent)
        rail_edge_color = QColor(rail_edge)
        rail_checked_bg = (
            f"rgba({rail_accent_color.red()},{rail_accent_color.green()},"
            f"{rail_accent_color.blue()},0.88)"
        )
        rail_checked_edge = (
            f"rgba({rail_edge_color.red()},{rail_edge_color.green()},"
            f"{rail_edge_color.blue()},0.92)"
        )
        css = (
            get_global_stylesheet(cv_ops_wallpaper_blend=blend)
            + get_cvops_stylesheet(
                workspace_wallpaper=wp,
                backdrop_blend=blend,
                title_text_color=self._cvops_settings.title_text_color,
                title_background_color=self._cvops_settings.title_background_color,
                ui_text_color=self._cvops_settings.ui_text_color,
                ui_muted_text_color=self._cvops_settings.ui_muted_text_color,
                ui_background_color=self._cvops_settings.ui_background_color,
                ui_panel_background_color=self._cvops_settings.ui_panel_background_color,
                ui_control_background_color=self._cvops_settings.ui_control_background_color,
                ui_accent_color=self._cvops_settings.ui_accent_color,
            )
            + (
                f"QFrame#cvOpsActivityRail{{background:{rail_bg};border-right:1px solid #333;}}"
                "QFrame#cvOpsActivityRail QPushButton{font-family:'JetBrains Mono','Consolas',monospace;"
                f"font-size:9px;padding:6px 12px;border:none;color:{rail_text};}}"
                "QFrame#cvOpsActivityRail QPushButton:checked{"
                f"background:{rail_checked_bg};color:{rail_checked_text};"
                f"border-left:1px solid {rail_checked_edge};border-right:1px solid {rail_checked_edge};"
                "}"
            )
        )
        scale = resolve_ui_scale_factor(self, self._cvops_settings.ui_scale_pct)
        self._last_ui_style_scale = scale
        self._workbench_root.set_wallpaper(wp)
        return scale_qss_pixel_metrics(css, scale)

    def _apply_cvops_stylesheet(self) -> None:
        configure_color_scheme(self._cvops_settings.color_scheme)
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(self._combined_stylesheet())
        refresh_cvops_theme_tree(self)
        apply_text_palette(self)
        repolish(self)
        root = self.centralWidget()
        if root is not None:
            repolish(root)
        apply_ui_scale_compaction(
            self,
            ui_scale_pct=self._cvops_settings.ui_scale_pct,
        )
        install_cvops_chamfer_combo_style(self)

    def _on_settings_color_scheme(self, scheme: str) -> None:
        _ = scheme
        self._apply_cvops_stylesheet()
        self._status.setText(f"Color scheme: {self._cvops_settings.color_scheme}")

    def _on_settings_theme_overrides(self) -> None:
        self._apply_cvops_stylesheet()
        self._status.setText("Theme colors updated.")

    def _on_settings_button_shape(self, button_shape: str) -> None:
        shape = set_cvops_button_shape(button_shape)
        self._apply_cvops_stylesheet()
        self._status.setText(f"Button shape: {shape}")

    def _on_settings_ui_scale(self, ui_scale_pct: int) -> None:
        pct = normalize_ui_scale_pct(ui_scale_pct)
        self._cvops_settings.ui_scale_pct = pct
        self.setProperty("uiScalePct", pct)
        self._last_ui_style_scale = None
        self._apply_cvops_stylesheet()
        self._layout_refresh_timer.start()
        self._status.setText(f"UI scale: {pct}%")

    def _nudge_ui_scale(self, delta_pct: int) -> None:
        target = normalize_ui_scale_pct(self._cvops_settings.ui_scale_pct + int(delta_pct))
        if target == self._cvops_settings.ui_scale_pct:
            return
        if self._settings_panel is not None:
            self._settings_panel.set_ui_scale_pct(target)
            return
        self._cvops_settings.ui_scale_pct = target
        try:
            save_cvops_settings(self._settings_path, self._cvops_settings)
        except Exception:
            pass
        self._on_settings_ui_scale(target)

    def _reset_ui_scale(self) -> None:
        if self._cvops_settings.ui_scale_pct == 100:
            return
        if self._settings_panel is not None:
            self._settings_panel.set_ui_scale_pct(100)
            return
        self._cvops_settings.ui_scale_pct = 100
        try:
            save_cvops_settings(self._settings_path, self._cvops_settings)
        except Exception:
            pass
        self._on_settings_ui_scale(100)

    def _on_settings_time_format(self, time_format: str) -> None:
        mode = set_time_format(time_format)
        self._cvops_settings.time_format = mode
        self._start_ws_resync(force=True)
        self._refresh_scenarios()
        if self._range_panel is not None:
            try:
                self._range_panel.reload()
            except Exception:
                pass
        try:
            self._lineage_panel.reload()
        except Exception:
            pass
        self._status.setText(f"Time format: {'12-hour' if mode == '12h' else '24-hour'}")

    def _on_settings_dashboard(self, auto_start: bool, dashboard_port: int) -> None:
        old_url = self._dashboard_url
        self._cvops_settings.auto_start_dashboard = False
        self._cvops_settings.dashboard_port = int(dashboard_port)
        self._dashboard_port = int(dashboard_port)
        self._dashboard_url = f"http://127.0.0.1:{self._dashboard_port}"
        if self._settings_panel is not None:
            self._settings_panel.set_dashboard_url(self._dashboard_url)
        if self._dashboard_url != old_url and (
            self._dashboard_proc is None or self._dashboard_proc.poll() is not None
        ):
            self._load_dashboard_in_view()
        if self._dashboard_url != old_url and self._dashboard_proc is not None and self._dashboard_proc.poll() is None:
            self._refresh_dashboard_status("Dashboard port saved; stop/start the dashboard to use the new port.")
        else:
            self._refresh_dashboard_status("Dashboard settings saved.")

    def _on_settings_poll_intervals(
        self,
        health_poll_ms: int,
        gallery_poll_ms: int,
        dashboard_poll_ms: int,
    ) -> None:
        self._health_timer.setInterval(int(health_poll_ms))
        _ = gallery_poll_ms  # retained for settings file compatibility; gallery tab removed
        self._dashboard_timer.setInterval(int(dashboard_poll_ms))
        self._status.setText("Polling settings saved.")

    def _on_settings_event_pulse_visibility(self, visible: bool) -> None:
        self._cvops_settings.show_event_pulse = bool(visible)
        if hasattr(self, "_event_pulse"):
            self._event_pulse.setVisible(bool(visible))
        self._sync_bottom_pulse_bar_visibility()
        self._status.setText(
            "Scrolling notification bar shown." if visible else "Scrolling notification bar hidden."
        )

    def _sync_bottom_pulse_bar_visibility(self) -> None:
        bar = getattr(self, "_bottom_pulse_bar", None)
        if bar is not None:
            bar.setVisible(bool(self._cvops_settings.show_event_pulse))

    def _open_notifications_center(self) -> None:
        self._set_workbench_mode(WorkbenchSplitHost.MODE_NOTIFICATIONS)

    def _export_cvops_state(self) -> None:
        url = QUrl(self.base_url + "/cvops/state/export")
        if not QDesktopServices.openUrl(url):
            self._append_error("settings", f"Failed to open export URL: {url.toString()}")

    # ---------- Shell cache ----------

    def _restore_shell_cache(self) -> None:
        try:
            payload = json.loads(self._shell_cache_path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict) or int(payload.get("version") or 0) != _SHELL_CACHE_VERSION:
            return
        self._last_event_seq = max(self._last_event_seq, int(payload.get("event_seq") or 0))
        cache_payload = {
            "jobs": payload.get("jobs") if isinstance(payload.get("jobs"), list) else [],
            "training_events": payload.get("training_events") if isinstance(payload.get("training_events"), list) else [],
            "scenarios": payload.get("scenarios") if isinstance(payload.get("scenarios"), list) else [],
            "event_seq": self._last_event_seq,
            "errors": [],
        }
        self._apply_resync_payload(cache_payload, persist=False)
        try:
            self._status.setText("Loaded last-known CV Ops state.")
        except Exception:
            pass

    def _schedule_shell_cache_save(self) -> None:
        if hasattr(self, "_shell_cache_save_timer"):
            self._shell_cache_save_timer.start()

    def _write_shell_cache(self) -> None:
        payload = {
            "version": _SHELL_CACHE_VERSION,
            "saved_at": time.time(),
            "event_seq": int(self._last_event_seq),
            "jobs": list(self._shell_jobs_cache[:500]),
            "training_events": list(self._shell_training_events_cache[-_SHELL_CACHE_MAX_TRAINING_EVENTS:]),
            "scenarios": list(self._scenarios_cache),
        }
        try:
            self._shell_cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self._shell_cache_path.with_name(
                f"{self._shell_cache_path.name}.tmp.{os.getpid()}"
            )
            tmp_path.write_text(
                json.dumps(payload, ensure_ascii=True, separators=(",", ":"), default=str),
                encoding="utf-8",
            )
            tmp_path.replace(self._shell_cache_path)
        except Exception:
            pass

    def _update_last_event_seq(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        try:
            seq = int(payload.get("seq") or payload.get("event_seq") or 0)
        except Exception:
            seq = 0
        if seq > self._last_event_seq:
            self._last_event_seq = seq
            try:
                self._ws.set_last_seq(seq)
            except Exception:
                pass

    def _cache_job_event(self, job: dict[str, Any]) -> None:
        job_id = str(job.get("job_id") or "")
        if not job_id:
            return
        clean = dict(job)
        for i, existing in enumerate(self._shell_jobs_cache):
            if str(existing.get("job_id") or "") == job_id:
                merged = dict(existing)
                merged.update(clean)
                self._shell_jobs_cache[i] = merged
                break
        else:
            self._shell_jobs_cache.insert(0, clean)
        del self._shell_jobs_cache[500:]
        self._schedule_shell_cache_save()

    def _cache_training_event(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self._shell_training_events_cache.append(dict(payload))
        if len(self._shell_training_events_cache) > _SHELL_CACHE_MAX_TRAINING_EVENTS:
            del self._shell_training_events_cache[
                : len(self._shell_training_events_cache) - _SHELL_CACHE_MAX_TRAINING_EVENTS
            ]
        self._schedule_shell_cache_save()

    def _replay_ws_events(self) -> bool:
        try:
            return bool(self._ws.replay_since(self._last_event_seq))
        except Exception as exc:
            self._append_error("websocket-replay", str(exc))
            return False

    # ---------- HTTP ----------

    def _direct_service_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
    ) -> Optional[dict[str, Any]]:
        if not self._prefer_direct_service_reads:
            return None
        if payload is not None or method.upper() != "GET":
            return None
        service = getattr(getattr(self, "_server", None), "service", None)
        if service is None:
            return None
        clean_path = str(path or "").split("?", 1)[0]
        parts = [urllib.parse.unquote(p) for p in clean_path.strip("/").split("/") if p]
        n = len(parts)
        # Read directly from the in-process service instead of the loopback HTTP
        # socket. Any failure (not-found, registry error) returns None so the
        # caller falls back to the real HTTP handler, keeping error/status codes
        # identical. Only covers pure, query-less GET reads -- binary/image/text
        # and query-parameterized endpoints still use HTTP.
        try:
            if clean_path == "/health":
                return service._health_snapshot()
            if clean_path == "/scenarios":
                return service.scenarios_payload()
            if clean_path == "/jobs":
                return service.jobs_payload()
            if clean_path == "/models":
                return service.models_payload()
            if clean_path == "/database":
                return service.database_list_payload()
            if n == 2 and parts[0] == "database":
                return service.database_dataset_payload(parts[1])
            if n == 2 and parts[0] == "jobs":
                return service.job_payload(parts[1])
            if n == 3 and parts[0] == "jobs" and parts[2] == "result":
                return service.job_result_payload(parts[1])
            if n == 3 and parts[0] == "jobs" and parts[2] == "training_progress":
                return service.training_progress_payload(parts[1])
            if n == 3 and parts[0] == "scenarios" and parts[2] == "status":
                return service.scenario_status_payload(parts[1])
            if n == 3 and parts[0] == "scenarios" and parts[2] == "history":
                return service.scenario_history_payload(parts[1])
            if n == 3 and parts[0] == "scenarios" and parts[2] == "cards":
                return service.scenario_cards_payload(parts[1])
            if n == 3 and parts[0] == "scenarios" and parts[2] == "custom_cells":
                return service.custom_cells_payload(parts[1])
        except Exception:
            return None
        return None

    def _http_json(
        self,
        method: str,
        path: str,
        payload: Optional[dict[str, Any]] = None,
        *,
        timeout: float = 4.0,
    ) -> dict[str, Any]:
        direct = self._direct_service_json(method, path, payload)
        if direct is not None:
            return direct
        url = self.base_url + path
        data = None
        headers = {"Content-Type": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, method=method.upper(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as http_err:
            body_raw = ""
            try:
                body_raw = http_err.read().decode("utf-8")
            except Exception:
                body_raw = ""
            detail = body_raw
            try:
                parsed = json.loads(body_raw) if body_raw else {}
                if isinstance(parsed, dict) and parsed.get("detail"):
                    d = parsed["detail"]
                    if isinstance(d, dict):
                        detail = str(d.get("detail") or json.dumps(d))
                    else:
                        detail = str(d)
            except Exception:
                pass
            http_err.response_body = detail  # type: ignore[attr-defined]
            raise
        return json.loads(raw) if raw else {}

    def _http_text(self, path: str, timeout: float = 4.0) -> str:
        url = self.base_url + path
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")

    # ---------- Loaders ----------

    def _initial_load(self) -> None:
        _bprint("_initial_load fired — fetching health + scenarios")
        self._refresh_health()
        last = str(self._cvops_settings.last_scenario or "").strip()
        if last:
            self._catalog_panel.select_scenario(last)
        if self._prefer_direct_service_reads:
            try:
                self._catalog_panel.show_scenarios_connecting()
            except Exception:
                pass
        else:
            self._refresh_scenarios()
        # _WsResyncWorker covers jobs + training-progress + scenarios on a background
        # thread — replaces the old synchronous _seed_jobs() call that blocked the UI.
        self._start_ws_resync(force=True)

    def _maybe_retry_scenarios_boot(self) -> None:
        """If the first /scenarios fetch failed (server still starting), try once more."""
        if self._scenarios_refresh_failed:
            self._refresh_scenarios()

    def _refresh_health(self) -> None:
        if self._health_fetch_running:
            return
        self._health_fetch_running = True
        t = _HttpFetchThread(
            lambda: self._http_json("GET", "/health", timeout=2.5),
            parent=self,
        )
        t.done.connect(self._on_health_done)
        t.failed.connect(self._on_health_failed)
        t.finished.connect(lambda: setattr(self, "_health_fetch_running", False))
        t.finished.connect(t.deleteLater)
        t.start()

    def _on_health_done(self, health: object) -> None:
        if not isinstance(health, dict):
            health = {}
        self._refresh_dashboard_health_summary(health=health)
        st = (self._status.text() or "").strip()
        if st.startswith("Service unavailable"):
            self._status.setText("")

    def _on_health_failed(self, msg: str) -> None:
        self._status.setText(f"Service unavailable: {msg}")
        self._refresh_dashboard_health_summary(error=msg)
        self._append_error("health", msg)

    def _refresh_scenarios(self) -> None:
        if self._scenarios_fetch_running:
            return
        self._scenarios_fetch_running = True
        # Surface a "connecting and refreshing" message in the catalog while there
        # is nothing valid to show yet (first load, or after a prior failure) so a
        # background refresh of an already-populated table doesn't flicker.
        if not self._scenarios_cache or self._scenarios_refresh_failed:
            try:
                self._catalog_panel.show_scenarios_connecting()
            except Exception:
                pass
        timeout_s = 6.0
        t = _HttpFetchThread(
            lambda: self._http_json("GET", "/scenarios", timeout=timeout_s),
            parent=self,
        )
        t.done.connect(self._on_scenarios_done)
        t.failed.connect(lambda msg: self._on_scenarios_failed(msg, timeout_s))
        t.finished.connect(lambda: setattr(self, "_scenarios_fetch_running", False))
        t.finished.connect(t.deleteLater)
        t.start()

    def _on_scenarios_done(self, payload: object) -> None:
        if not isinstance(payload, dict):
            payload = {}
        scenarios = [s for s in list(payload.get("scenarios") or []) if isinstance(s, dict)]
        self._apply_scenarios(scenarios)

    def _on_scenarios_failed(self, msg: str, timeout_s: float) -> None:
        self._scenarios_refresh_failed = True
        if "timed out" in msg.lower():
            msg = f"/scenarios timed out after {timeout_s:.1f}s ({self.base_url})."
        # Only take over the catalog with the error card when there is no usable
        # list already on screen; a failed background poll shouldn't wipe a good table.
        if not self._scenarios_cache:
            try:
                self._catalog_panel.show_scenarios_error(msg)
            except Exception:
                pass
        self._refresh_dashboard_scenario_summary(error=msg)
        self._append_error("scenarios", msg)
        self._apply_stage_gating()

    def _apply_scenarios(self, scenarios: list[dict[str, Any]]) -> None:
        self._scenarios_refresh_failed = False
        self._scenarios_cache = scenarios
        if self._submit_panel is not None:
            self._submit_panel.refresh_scenarios()
        self._catalog_panel.apply_scenarios(scenarios)
        if self._cells_panel is not None:
            self._cells_panel.set_scenarios(scenarios)
        try:
            if self._database_godview_panel is not None:
                self._database_godview_panel.apply_scenarios(scenarios)
            if self._data_viz_db_selector is not None:
                self._data_viz_db_selector.apply_scenarios(scenarios)
        except Exception:
            pass
        self._sync_database_panel_context(self._catalog_panel.current_scenario())
        self._refresh_dashboard_scenario_summary()
        self._apply_stage_gating()
        self._schedule_shell_cache_save()
        _mark_boot(
            "first_data_paint",
            once=True,
            source="scenarios",
            count=len(scenarios),
            visible=bool(self.isVisible()),
        )

    def _apply_training_progress_events(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            if not isinstance(event, dict):
                continue
            self._forward_notes_ai_cvops_event(event)
            self._catalog_panel.apply_training_progress(event)
            if self._ontology_panel is not None:
                try:
                    self._ontology_panel.apply_training_progress(event)
                except Exception:
                    pass

    def _apply_stage_gating(self) -> None:
        pass

    @staticmethod
    def _scenario_is_trained(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        if bool(entry.get("weights_ready")):
            return True
        return str(entry.get("status") or "").strip().lower() in {"ready", "trained"}

    @staticmethod
    def _scenario_is_verified(entry: Any) -> bool:
        if not isinstance(entry, dict):
            return False
        if bool(entry.get("verified")):
            return True
        return bool(entry.get("weights_ready")) and bool(entry.get("verified", False))

    def _seed_jobs(self) -> None:
        try:
            payload = self._http_json("GET", "/jobs")
        except Exception:
            return
        jobs = list(payload.get("jobs") or [])
        self._queue_panel.seed_jobs(jobs)
        self._seed_dashboard_jobs(jobs)
        self._seed_training_progress(jobs)

    def _seed_training_progress(self, jobs: list[dict[str, Any]]) -> None:
        train_jobs = [j for j in jobs if str(j.get("job_type") or "") == "train"][:18]
        for job in train_jobs:
            job_id = str(job.get("job_id") or "")
            if not job_id:
                continue
            try:
                payload = self._http_json("GET", f"/jobs/{job_id}/training_progress")
            except Exception:
                continue
            events = payload.get("events") if isinstance(payload, dict) else []
            if not isinstance(events, list):
                continue
            scenario = str(job.get("scenario") or "")
            merged_events: list[dict[str, Any]] = []
            for event in events:
                if not isinstance(event, dict):
                    continue
                merged = dict(event)
                if not merged.get("scenario"):
                    merged["scenario"] = scenario
                if not merged.get("job_id"):
                    merged["job_id"] = job_id
                merged_events.append(merged)
            self._apply_training_progress_events(merged_events)

    def _on_ws_status_clicked(self, event) -> None:  # type: ignore[override]
        try:
            event.accept()
        except Exception:
            pass
        self._refresh_ws_output()

    def _refresh_ws_output(self) -> None:
        self._ws_refresh_btn.setEnabled(True)
        self._ws_status.setProperty("state", "disconnected")
        repolish(self._ws_status)
        self._ws_status.setText("[WS] reconnecting...")
        try:
            self._suppress_next_connected_resync = True
            self._ws.reconnect_now()
        except Exception as exc:
            self._suppress_next_connected_resync = False
            self._append_error("websocket", f"manual reconnect failed: {exc}")
        if not self._replay_ws_events():
            self._start_ws_resync(force=True)

    def _start_ws_resync(self, *, force: bool = False) -> None:
        worker = self._ws_resync_worker
        if worker is not None and worker.isRunning():
            self._ws_resync_pending = True
            return
        now = time.time()
        if not force and (now - self._ws_resync_last_started) < 1.0:
            return
        self._ws_resync_last_started = now
        self._ws_resync_pending = False
        direct_resync = None
        if self._prefer_direct_service_reads:
            service = getattr(getattr(self, "_server", None), "service", None)
            if service is not None:
                direct_resync = service.startup_resync_payload
        worker = _WsResyncWorker(self.base_url, direct_resync=direct_resync, parent=self)
        worker.completed.connect(self._on_ws_resync_complete)
        worker.failed.connect(self._on_ws_resync_failed)
        worker.finished.connect(self._on_ws_resync_finished)
        self._ws_resync_worker = worker
        worker.start()

    def _on_ws_resync_complete(self, payload: dict[str, Any]) -> None:
        self._apply_resync_payload(payload, persist=True)

    def _apply_resync_payload(self, payload: dict[str, Any], *, persist: bool) -> None:
        self._update_last_event_seq(payload)
        jobs = payload.get("jobs")
        if isinstance(jobs, list):
            clean_jobs = [j for j in jobs if isinstance(j, dict)]
            self._shell_jobs_cache = [dict(j) for j in clean_jobs[:500]]
            self._queue_panel.seed_jobs(clean_jobs)
            self._seed_dashboard_jobs(clean_jobs)
            _mark_boot(
                "first_data_paint",
                once=True,
                source="jobs",
                count=len(clean_jobs),
                visible=bool(self.isVisible()),
            )
            for job in clean_jobs:
                self._forward_notes_ai_cvops_event(job)
        events = payload.get("training_events")
        if isinstance(events, list):
            clean_events = [e for e in events if isinstance(e, dict)]
            self._shell_training_events_cache = [
                dict(e) for e in clean_events[-_SHELL_CACHE_MAX_TRAINING_EVENTS:]
            ]
            self._apply_training_progress_events(clean_events)
        scenarios = payload.get("scenarios")
        if isinstance(scenarios, list):
            self._apply_scenarios([s for s in scenarios if isinstance(s, dict)])
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            self._append_error("websocket-resync", "; ".join(str(e) for e in errors[:3]))
        if persist:
            self._schedule_shell_cache_save()

    def _on_ws_resync_failed(self, message: str) -> None:
        self._append_error("websocket-resync", str(message or "resync failed"))

    def _on_ws_resync_finished(self) -> None:
        worker = self.sender()
        if worker is self._ws_resync_worker:
            self._ws_resync_worker = None
        if isinstance(worker, QThread):
            worker.deleteLater()
        if self._ws_resync_pending:
            self._ws_resync_pending = False
            QTimer.singleShot(250, self._start_ws_resync)

    def _ingest_system_notification(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            return
        self._update_last_event_seq(payload)
        event = dict(payload)
        event.setdefault("emitted_at", time.time())
        self._event_pulse.ingest(event)
        self._notifications_panel.ingest(event)
        if should_show_notification_card(event, self._heartbeat_notification_gate):
            self._notification_tray.push(event)
        self._schedule_shell_cache_save()

    def _forward_notes_ai_cvops_event(self, payload: dict[str, Any]) -> None:
        panel = getattr(self, "_notes_panel", None)
        workspace = getattr(panel, "_ai_workspace", None)
        apply_event = getattr(workspace, "apply_cvops_event", None)
        if callable(apply_event):
            try:
                apply_event(payload)
            except Exception:
                pass

    def _emit_local_notification(
        self,
        event_type: str,
        *,
        scope: str = "",
        state: str = "",
        message: str = "",
        scenario: str = "",
        **extra: Any,
    ) -> None:
        payload: dict[str, Any] = {
            "type": str(event_type or "local_event"),
            "emitted_at": time.time(),
        }
        if scope:
            payload["scope"] = scope
        if state:
            payload["state"] = state
        if message:
            payload["message"] = message
            if state in {"error", "failed"}:
                payload["error"] = message
        if scenario:
            payload["scenario"] = scenario
        payload.update(extra)
        self._ingest_system_notification(payload)

    # ---------- Background server boot ----------

    def _on_server_ready(self, handle: object) -> None:
        """The backend finished booting on the worker thread — bind it live.

        Runs on the main thread (queued signal). Binds the in-process event
        client to the live service and starts it; that emits connectedChanged,
        which drives the existing resync that populates every panel."""
        self._server = handle
        _bprint("backend server ready")
        _mark_boot("backend_ready", once=True)
        try:
            self._ws.set_service(handle.service)
            self._suppress_next_connected_resync = True
            self._ws.start(replay_since=self._last_event_seq)
        except Exception as exc:
            self._suppress_next_connected_resync = False
            self._append_error("server", f"event client bind failed: {exc}")
        try:
            self._initial_load()
        except Exception:
            pass
        if (
            getattr(self, "_plane_stack", None) is not None
            and self._plane_stack.currentIndex() == 0
            and self._ontology_panel is None
        ):
            self._start_ontology_background_load(attach_when_ready=True)

    def _on_server_boot_failed(self, message: str) -> None:
        self._append_error("server", f"backend failed to start: {message}")
        try:
            self._status.setText(f"Backend failed to start: {message}")
        except Exception:
            pass

    # ---------- WS handlers ----------

    def _on_ws_connected(self, connected: bool) -> None:
        if connected:
            _bprint("WebSocket connected — live")
            try:
                from insight_local.cvops.__main__ import _boot_finish  # noqa: PLC0415
                _boot_finish()
            except Exception:
                pass
            self._ws_status.setProperty("state", "live")
            repolish(self._ws_status)
            self._ws_status.setText("[WS] live")
            # Re-seed on reconnect to resync state, but keep network I/O off
            # the Qt event loop so a flapping socket cannot freeze the UI.
            if self._suppress_next_connected_resync:
                self._suppress_next_connected_resync = False
            else:
                if not self._replay_ws_events():
                    self._start_ws_resync()
        else:
            self._ws_status.setProperty("state", "disconnected")
            repolish(self._ws_status)
            self._ws_status.setText("[WS] reconnecting...")
            self._append_error("websocket", "disconnected; reconnecting")

    def _on_ws_job_status(self, payload: dict) -> None:
        self._update_last_event_seq(payload)
        self._cache_job_event(payload)
        self._forward_notes_ai_cvops_event(payload)
        self._queue_panel.upsert_job(payload)
        self._dashboard_upsert_job(payload)
        if self._ontology_panel is not None:
            try:
                self._ontology_panel.apply_job_status(payload)
            except Exception:
                pass
        state = str(payload.get("state") or "").strip().lower()
        if state == "error":
            err = str(payload.get("error") or "").strip()
            job_id = str(payload.get("job_id") or "")
            scen = str(payload.get("scenario") or "")
            self._append_error(
                "job",
                f"{job_id} ({scen}): {err or 'job entered error state'}",
                mirror_notification=False,
            )
        # Training jobs flip catalog status; refresh on any train transition.
        if str(payload.get("job_type") or "") == "train":
            scen = str(payload.get("scenario") or "")
            if scen:
                try:
                    status = self._http_json("GET", f"/scenarios/{scen}/status")
                    self._catalog_panel.apply_scenario_update(scen, status)
                    self._update_scenarios_cache_entry(status)
                    if self._submit_panel is not None:
                        self._submit_panel.refresh_scenarios()
                    self._refresh_dashboard_scenario_summary()
                    self._apply_stage_gating()
                except Exception:
                    pass
            if state in {"done", "completed", "complete", "succeeded", "success"} and scen:
                self._show_verify_toast(scen)

    def _show_verify_toast(self, scenario: str) -> None:
        if not scenario:
            return
        self._toast_target_scenario = scenario
        message = f"Training finished for {scenario} - click here to open in Range"
        self._toast.setText(message)
        self._toast.setVisible(True)
        self._schedule_notification_tray_layout()
        self._emit_local_notification(
            "toast",
            scope="train",
            state="info",
            scenario=scenario,
            message=message,
            action="open_test_range",
        )

    def _on_toast_clicked(self, _event) -> None:
        scen = self._toast_target_scenario
        self._toast.setVisible(False)
        self._toast_target_scenario = ""
        self._schedule_notification_tray_layout()
        self._open_test_range_for_scenario(scen)

    def _on_notification_card_activated(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            self._open_notifications_center()
            return
        action = str(payload.get("action") or "").strip().lower()
        scenario = str(payload.get("scenario") or "").strip()
        if action == "open_test_range" and scenario:
            self._open_test_range_for_scenario(scenario)
            return
        self._open_notifications_center()

    def _open_test_range_for_scenario(self, scenario: str) -> None:
        scen = str(scenario or "").strip()
        if not scen:
            return
        if hasattr(self, "_plane_stack"):
            self._set_workbench_mode(WorkbenchSplitHost.MODE_TEST)
        self._catalog_panel.select_scenario(scen)

    def _on_cells_open_workflow_train(self, scenario: str) -> None:
        """Cells tab bridge: jump to Workbench Train where drafts and runs live."""
        name = str(scenario or "").strip()
        if not name:
            return
        try:
            self._set_workbench_mode(WorkbenchSplitHost.MODE_EXPLORE)
            self._workbench_split_host.apply_preset(WorkbenchSplitHost.PRESET_TRAIN)
            self._catalog_panel.select_scenario(name)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Ecosystem (Ontology graph) navigation router
    # ------------------------------------------------------------------

    def _on_eco_navigate(self, target: str, focus_id: str, scenario_hint: str) -> None:
        """Route a quick-nav request from the Ecosystem detail card.

        target        — slug like 'training', 'test_range', 'jobs', etc.
        focus_id      — the id segment from the URL; meaning is per-target
                        (scenario name, job_id, lineage_id, range_id...)
        scenario_hint — meta.scenario from the source node, when present.
        """
        target = str(target or "").strip().lower()
        focus_id = str(focus_id or "").strip()
        scenario_hint = str(scenario_hint or "").strip()
        if not target:
            return

        # Explorer stages (legacy workflow indices): 0 Scenario  1 Data  2 Train  3 Results  4 Charts  5 Jobs
        WORKFLOW_TARGETS = {
            "scenario_config": 0,
            "data_selection":  1,
            "training":        2,
            "results":         3,
            "charts":          4,
            "jobs":            5,
        }

        try:
            if target in WORKFLOW_TARGETS:
                self._open_workflow_stage(WORKFLOW_TARGETS[target], "")
            elif target == "test_range":
                self._set_workbench_mode(WorkbenchSplitHost.MODE_TEST)
            elif target == "database":
                self._set_workbench_mode(WorkbenchSplitHost.MODE_DATA)
            elif target in ("viz", "data_viz", "visualization"):
                self._set_workbench_mode(WorkbenchSplitHost.MODE_VIZ)
            elif target in ("scrape", "collect"):
                self._set_workbench_mode(WorkbenchSplitHost.MODE_COLLECT)
            else:
                return
        except Exception:
            return

        # Pre-focus the panel where we can. Fall back silently if the
        # focus method is missing — switching the tab alone is still useful.
        try:
            if target in ("scenario_config", "training") and focus_id:
                self._catalog_panel.select_scenario(focus_id)
            elif target == "data_selection" and focus_id and self._database_panel is not None:
                self._database_panel.set_scenario(focus_id)
            elif target == "results" and focus_id:
                self._result_panel.select_job(focus_id)
            elif target == "charts" and focus_id:
                # focus_id may be a scenario name (from a scenario node) or a
                # lineage_id (from a lineage node). The panel exposes
                # select_lineage; scenario-based filtering is not implemented
                # yet so we attempt lineage select and let it no-op silently.
                self._lineage_panel.select_lineage(focus_id)
            elif target == "jobs" and focus_id:
                # focus_id may be a job_id (from a job node) or a scenario
                # name (from a scenario node). select_job no-ops if missing.
                self._queue_panel.select_job(focus_id)
            elif target == "test_range" and focus_id and self._range_panel is not None:
                self._range_panel.select_range(focus_id)
        except Exception:
            # Pre-focus is best-effort. The tab switch already happened.
            pass

    def _on_eco_entity_selected(self, entity_type: str, entity_id: str) -> None:
        """Fallback for the [NAVIGATE →] button on the Ecosystem detail card.

        Maps each entity type to a sensible default quick-nav target and
        delegates to _on_eco_navigate.
        """
        et = str(entity_type or "").strip().lower()
        eid = str(entity_id or "").strip()
        DEFAULT_TARGETS = {
            "scenario":         "scenario_config",
            "scenario_group":   "scenario_config",
            "job":              "jobs",
            "dataset":          "data_selection",
            "dataset_snapshot": "data_selection",
            "model_version":    "training",
            "model_snapshot":   "charts",
            "lineage":          "charts",
            "range":            "test_range",
            "catalog_asset":    "scenario_config",
            "database":         "database",
        }
        target = DEFAULT_TARGETS.get(et)
        if not target:
            return
        self._on_eco_navigate(target, eid, "")

    def _on_ws_scenario_updated(self, payload: dict) -> None:
        self._update_last_event_seq(payload)
        scen = str(payload.get("scenario") or "")
        status = payload.get("status_payload") if isinstance(payload.get("status_payload"), dict) else None
        if scen and status is not None:
            self._catalog_panel.apply_scenario_update(scen, status)
            self._update_scenarios_cache_entry(status)
            if self._submit_panel is not None:
                self._submit_panel.refresh_scenarios()
            self._refresh_dashboard_scenario_summary()
            if self._catalog_panel.current_scenario() == scen:
                self._sync_database_panel_context(scen)
        else:
            self._refresh_scenarios()

    def _update_scenarios_cache_entry(self, status: dict[str, Any]) -> None:
        name = str(status.get("name") or "")
        if not name:
            return
        for i, entry in enumerate(self._scenarios_cache):
            if str(entry.get("name") or "") == name:
                merged = dict(entry)
                merged.update(status)
                self._scenarios_cache[i] = merged
                self._schedule_shell_cache_save()
                return
        self._scenarios_cache.append(dict(status))
        self._schedule_shell_cache_save()

    def _on_data_viz_scenario_focused(self, scen: dict[str, Any]) -> None:
        panel = getattr(self, "_data_viz_standalone", None)
        if panel is None:
            return
        try:
            btype = str(scen.get("backbone_type") or "")
            if btype == "torch_tabular":
                bcfg = scen.get("backbone_config") if isinstance(scen.get("backbone_config"), dict) else {}
                dataset_csv = str((bcfg or {}).get("dataset_csv") or "").strip()
                panel.set_scenario_csv(dataset_csv, str(ROOT_DIR))
                self._collapse_data_viz_catalog()
            elif btype == "archival_ingestion":
                bcfg = scen.get("backbone_config") if isinstance(scen.get("backbone_config"), dict) else {}
                panel.set_archive_context(
                    str((bcfg or {}).get("corpus_id") or ""),
                    str((bcfg or {}).get("dataset_version_id") or ""),
                    str((bcfg or {}).get("latest_snapshot_id") or scen.get("archive_snapshot_id") or ""),
                    scenario=str(scen.get("name") or ""),
                )
                self._collapse_data_viz_catalog()
            else:
                panel.clear()
        except Exception:
            pass

    def _on_data_viz_database_entity_selected(self, entity_type: str, entity_id: str) -> None:
        if str(entity_type or "") != "database":
            return
        panel = getattr(self, "_data_viz_standalone", None)
        if panel is None:
            return
        try:
            panel.set_data_source_path(Path(str(entity_id or "")))
            self._collapse_data_viz_catalog()
        except Exception as exc:
            self._append_error("data-viz", f"Could not load selected data source: {exc}")

    def _on_catalog_selection(self, _scenario: str) -> None:
        if self._submit_panel is not None:
            self._submit_panel.refresh_scenarios()
        self._sync_database_panel_context(_scenario)
        self._persist_last_scenario(_scenario)
        try:
            self._lineage_panel.select_lineage_for_scenario(_scenario)
        except Exception:
            pass

    def _persist_last_scenario(self, scenario: str) -> None:
        name = str(scenario or "").strip()
        if name == self._cvops_settings.last_scenario:
            return
        self._cvops_settings.last_scenario = name
        try:
            from .ui.settings_panel import save_cvops_settings
            save_cvops_settings(self._settings_path, self._cvops_settings)
        except Exception:
            pass

    def _sync_database_panel_context(self, scenario: str = "") -> None:
        if self._database_panel is None:
            return
        name = str(scenario or "").strip()
        if not name:
            self._database_panel.set_scenario("", "", "", {})
            return
        for entry in self._scenarios_cache:
            if str(entry.get("name") or "") != name:
                continue
            self._database_panel.set_scenario(
                name,
                str(entry.get("dataset") or ""),
                str(entry.get("backbone_type") or ""),
                entry.get("backbone_config") if isinstance(entry.get("backbone_config"), dict) else {},
            )
            return
        self._database_panel.set_scenario("", "", "", {})

    def _on_ws_job_result(self, job_id: str, result: dict) -> None:
        self._update_last_event_seq(result)
        self._forward_notes_ai_cvops_event(
            {**dict(result or {}), "type": "job_result", "job_id": job_id, "result": result}
        )
        self._result_cache[job_id] = result
        if self._ontology_panel is not None:
            try:
                self._ontology_panel.apply_job_result(job_id, result)
            except Exception:
                pass
        if self._queue_panel.selected_job_id() == job_id:
            self._result_panel.apply_result(job_id, result)
        if self._test_range_result_panel is not None and job_id == self._test_range_last_job_id:
            self._test_range_result_panel.apply_result(job_id, result)
        err = str(result.get("error") or "").strip()
        if err:
            if self._maybe_handle_schema_prompt(result, err):
                return
            scen = str(result.get("scenario") or "")
            self._append_error("job_result", f"{job_id} ({scen}): {err}", mirror_notification=False)
            self._maybe_surface_storage_pressure(
                err,
                job_id=job_id,
                scenario=scen,
                payload=result,
            )
        # Completed training jobs should surface their run artifacts in the catalog panel.
        scen = str(result.get("scenario") or "")
        if scen and not err and str(result.get("result_path") or ""):
            self._catalog_panel.notify_training_job(scen, job_id)

    def _on_ws_training_progress(self, payload: dict) -> None:
        self._update_last_event_seq(payload)
        self._cache_training_event(payload)
        self._forward_notes_ai_cvops_event(payload)
        self._catalog_panel.apply_training_progress(payload)
        if self._ontology_panel is not None:
            try:
                self._ontology_panel.apply_training_progress(payload)
            except Exception:
                pass
        if str(payload.get("event") or "") == "failed":
            err = str(payload.get("error") or "").strip()
            job_id = str(payload.get("job_id") or "")
            scen = str(payload.get("scenario") or "")
            self._append_error(
                "training",
                f"{job_id} ({scen}): {err or 'training failed'}",
                mirror_notification=False,
            )
            self._maybe_surface_storage_pressure(
                err,
                job_id=job_id,
                scenario=scen,
                payload=payload,
            )

    def _maybe_surface_storage_pressure(
        self,
        error: str,
        *,
        job_id: str = "",
        scenario: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        if not looks_like_storage_error(error):
            return
        key = f"{job_id}:{scenario}:{error[:160]}"
        if key in self._shown_storage_error_keys:
            return
        self._shown_storage_error_keys.add(key)

        payload = payload if isinstance(payload, dict) else {}
        asset_root = str(payload.get("asset_root") or payload.get("training_assets_root") or "").strip()
        extra_paths: list[str] = []
        pinned = payload.get("pinned_cache_paths")
        if isinstance(pinned, dict):
            extra_paths.extend(str(v) for v in pinned.values() if v)
        diagnosis = build_storage_diagnosis(
            message=error,
            asset_root=asset_root,
            extra_paths=extra_paths,
        )
        detail = format_storage_diagnosis(diagnosis)
        try:
            self._catalog_panel.show_storage_diagnosis(detail)
        except Exception:
            pass

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Icon.Warning)
        msg.setWindowTitle("Training Storage Pressure")
        msg.setText("Training failed because the system reported a disk/cache space problem.")
        context_bits = []
        if job_id:
            context_bits.append(f"job {job_id}")
        if scenario:
            context_bits.append(f"scenario {scenario}")
        context = " / ".join(context_bits)
        msg.setInformativeText(
            (context + "\n\n" if context else "")
            + "System Guard now shows the largest cache and drive pressure targets. "
            + "Free space on the listed system/cache paths, then restart the training run."
        )
        msg.setDetailedText(f"Original error:\n{error}\n\n{detail}")
        msg.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg.exec()

    def _maybe_handle_schema_prompt(self, result: dict[str, Any], err: str) -> bool:
        """Intercept structured schema prompts embedded in error strings."""
        prefix = "__CVOPS_PROMPT__:"
        idx = err.find(prefix)
        if idx < 0:
            return False
        raw = err[idx + len(prefix):].strip()
        try:
            payload = json.loads(raw)
        except Exception:
            return False
        if not isinstance(payload, dict):
            return False
        if str(payload.get("kind") or "") != "label_col_missing":
            return False

        scenario = str(result.get("scenario") or "").strip()
        dataset_csv = str(payload.get("dataset_csv") or "").strip()
        attempted = str(payload.get("attempted_label_col") or "").strip()
        columns = payload.get("columns") if isinstance(payload.get("columns"), list) else []
        columns = [str(c) for c in columns if str(c)]
        suggested = payload.get("suggested") if isinstance(payload.get("suggested"), list) else []
        suggested = [str(c) for c in suggested if str(c)]

        if not columns:
            return False

        dlg = SchemaFixDialog(
            scenario=scenario,
            dataset_csv=dataset_csv,
            attempted_label_col=attempted,
            columns=columns,
            suggested_label_cols=suggested,
            parent=self,
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return True
        action = dlg.action()
        chosen = dlg.chosen_label_col()
        feat_cols = dlg.chosen_feature_cols()
        if action == "reveal":
            if dataset_csv:
                reveal_in_finder(dataset_csv)
            return True
        if action in ("apply", "apply_rerun") and scenario and chosen:
            patch: dict[str, Any] = {"label_col": chosen}
            if feat_cols:
                patch["feature_cols"] = feat_cols
            try:
                self._http_json("POST", f"/scenarios/{scenario}/backbone_config", {"patch": patch})
                self._refresh_scenarios()
            except Exception as exc:
                self._append_error("schema", f"Failed to apply schema patch: {exc}")
                return True
            if action == "apply_rerun":
                try:
                    self._http_json("POST", f"/scenarios/{scenario}/train", None)
                except Exception as exc:
                    self._append_error("schema", f"Failed to re-run training: {exc}")
        return True

    def _on_ws_cell_progress(self, payload: dict) -> None:
        self._update_last_event_seq(payload)
        try:
            self._catalog_panel.apply_tabular_cell_progress(payload)
        except Exception:
            pass
        # Mirror the same stream into the Cell Space inline terminal (Colab-style).
        if self._cells_panel is not None:
            try:
                self._cells_panel.apply_cell_progress(payload)
            except Exception:
                pass
        # Per-job cell display removed; cell output is mirrored into the per-scenario Training console.

    def _on_ws_error(self, message: str) -> None:
        self._append_error("websocket", message)

    def _on_job_submitted(self, payload: dict[str, Any]) -> None:
        self._queue_panel.upsert_job(payload)
        self._dashboard_upsert_job(payload)
        job_id = str(payload.get("job_id") or "")
        scenario = str(payload.get("scenario") or "")
        if job_id:
            self._queue_panel.select_job(job_id)
            self._test_range_last_job_id = job_id
            # Local "RUN REGISTRY MODEL" inference emits both registryResultReady
            # (which already rendered the result + fed the subroutine) and
            # jobSubmitted with state "complete". Don't show the "Running..."
            # placeholder for it — show_message() calls clear() and would wipe
            # the freshly-rendered result. Only pending/queued jobs need it.
            already_complete = str(payload.get("state") or "").strip().lower() == "complete"
            if self._test_range_result_panel is not None and not already_complete:
                self._test_range_result_panel.show_message(
                    f"Running quick test job {job_id} for {scenario or 'selected scenario'}..."
                )
        self._status.setText(
            f"Queued {payload.get('job_type', 'infer')} job {job_id} for {payload.get('scenario', '')}"
        )

    def _on_registry_result_ready(self, result: dict) -> None:
        job_id = str(result.get("job_id") or "")
        if self._test_range_result_panel is not None:
            self._test_range_result_panel.apply_result(job_id, result)
        self._status.setText(result.get("summary", "Registry inference complete."))

    def _on_train_kicked(self, scenario: str, job_id: str) -> None:
        if not job_id:
            return
        try:
            payload = self._http_json("GET", f"/jobs/{job_id}")
        except Exception:
            payload = {
                "job_id": job_id,
                "job_type": "train",
                "scenario": scenario,
                "state": "queued",
                "source": "cvops_ui",
                "created_at": time.time(),
            }
        self._queue_panel.upsert_job(payload)
        self._dashboard_upsert_job(payload)
        self._queue_panel.select_job(job_id)
        try:
            self._set_workbench_mode(WorkbenchSplitHost.MODE_EXPLORE)
            self._workbench_split_host.apply_preset(WorkbenchSplitHost.PRESET_TRAIN)
        except Exception:
            pass

    def _on_cells_train_kicked(self, scenario: str, job_id: str) -> None:
        if not job_id:
            return
        try:
            payload = self._http_json("GET", f"/jobs/{job_id}")
        except Exception:
            payload = {
                "job_id": job_id,
                "job_type": "train",
                "scenario": scenario,
                "state": "queued",
                "source": "cvops_ui",
                "created_at": time.time(),
            }
        self._queue_panel.upsert_job(payload)
        self._dashboard_upsert_job(payload)
        self._status.setText(f"Cells queued notebook run {job_id} for {scenario}")

    # ---------- Queue selection ----------

    def _on_queue_cancel(self, job_id: str) -> None:
        if not job_id:
            return
        try:
            self._http_json("POST", f"/jobs/{job_id}/cancel")
        except Exception as exc:
            self._append_error("job_cancel", f"{job_id}: {exc}")
            return
        try:
            job = self._http_json("GET", f"/jobs/{job_id}")
        except Exception:
            job = {"job_id": job_id, "state": "error"}
        self._queue_panel.upsert_job(job)
        self._dashboard_upsert_job(job)

    def _on_queue_retry(self, job_id: str) -> None:
        if not job_id:
            return
        try:
            job = self._http_json("POST", f"/jobs/{job_id}/retry")
        except Exception as exc:
            self._append_error("job_retry", f"{job_id}: {exc}")
            return
        self._queue_panel.upsert_job(job)
        self._dashboard_upsert_job(job)
        new_id = str(job.get("job_id") or "")
        if new_id:
            self._queue_panel.select_job(new_id)

    def _on_queue_selection(self, job_id: str) -> None:
        if not job_id:
            self._result_panel.clear()
            return
        cached = self._result_cache.get(job_id)
        if cached is not None:
            self._result_panel.apply_result(job_id, cached)
            return
        try:
            result = self._http_json("GET", f"/jobs/{job_id}/result")
        except urllib.error.HTTPError:
            self._result_panel.show_message(f"Job {job_id}: no result yet.")
            return
        except Exception as exc:
            self._result_panel.show_message(f"Failed to load result: {exc}")
            self._append_error("result_fetch", f"{job_id}: {exc}")
            return
        self._result_cache[job_id] = result
        self._result_panel.apply_result(job_id, result)

    # ---------- Dashboard monitor ----------

    def _refresh_dashboard_health_summary(
        self,
        *,
        health: Optional[dict[str, Any]] = None,
        error: str = "",
    ) -> None:
        if self._dashboard_health_summary is None:
            return
        if error:
            self._dashboard_health_summary.setText(f"Service health: unavailable ({error})")
            if self._dashboard_overview is not None:
                self._dashboard_overview.set_health(
                    status="unavailable",
                    queued=0,
                    running=0,
                    done=0,
                    failed=0,
                )
            return
        if health is None:
            self._dashboard_health_summary.setText("Service health: waiting for data.")
            if self._dashboard_overview is not None:
                self._dashboard_overview.set_health(
                    status="waiting",
                    queued=0,
                    running=0,
                    done=0,
                    failed=0,
                )
            return
        status = str(health.get("status") or "unknown").upper()
        queued = int(health.get("queued") or 0)
        running = int(health.get("running") or 0)
        done = int(health.get("done") or 0)
        failed = int(health.get("error") or 0)
        slots_free = health.get("slots_free", "")
        max_workers = health.get("max_workers", "")
        workers = (
            f"{slots_free}/{max_workers} workers free"
            if slots_free != "" and max_workers != ""
            else "workers unknown"
        )
        self._dashboard_health_summary.setText(
            f"Service health: {status} · {workers} · {queued} queued · {running} running · {done} done · {failed} error"
        )
        if self._dashboard_overview is not None:
            self._dashboard_overview.set_health(
                status=status,
                queued=queued,
                running=running,
                done=done,
                failed=failed,
                slots_free=slots_free,
                max_workers=max_workers,
            )

    def _refresh_dashboard_scenario_summary(self, *, error: str = "") -> None:
        if self._dashboard_scenario_summary is None:
            return
        if error:
            self._dashboard_scenario_summary.setText(f"Scenario readiness: unavailable ({error})")
            if self._dashboard_overview is not None:
                self._dashboard_overview.set_scenarios(
                    total=0,
                    ready=0,
                    partial=0,
                    failed=0,
                )
            return
        total = len(self._scenarios_cache)
        ready = 0
        partial = 0
        failed = 0
        for item in self._scenarios_cache:
            if not isinstance(item, dict):
                continue
            status = str(item.get("status") or "").strip().lower()
            if status in {"ready", "trained"}:
                ready += 1
            elif status in {"error"}:
                failed += 1
            elif status in {"partial", "metrics_only"}:
                partial += 1
            elif bool(item.get("weights_ready")) and bool(item.get("verified", True)):
                ready += 1
            elif item.get("error"):
                failed += 1
            else:
                partial += 1
        self._dashboard_scenario_summary.setText(
            f"Scenario readiness: {ready}/{total} ready · {partial} partial · {failed} error"
        )
        if self._dashboard_overview is not None:
            self._dashboard_overview.set_scenarios(
                total=total,
                ready=ready,
                partial=partial,
                failed=failed,
            )

    def _seed_dashboard_jobs(self, jobs: list[dict[str, Any]]) -> None:
        if self._dashboard_jobs_table is None:
            return
        self._dashboard_jobs_rows.clear()
        self._dashboard_jobs_table.setRowCount(0)
        for job in jobs:
            if not isinstance(job, dict):
                continue
            self._dashboard_upsert_job(job, prepend=False)
        self._dashboard_jobs_table.resizeRowsToContents()
        self._refresh_dashboard_jobs_visual()

    def _dashboard_upsert_job(self, job: dict[str, Any], prepend: bool = True) -> None:
        table = self._dashboard_jobs_table
        if table is None:
            return
        job_id = str(job.get("job_id") or "")
        if not job_id:
            return
        row = self._dashboard_jobs_rows.get(job_id)
        if row is None:
            row = 0 if prepend else table.rowCount()
            table.insertRow(row)
            if prepend:
                self._dashboard_jobs_rows = {k: (v + 1) for k, v in self._dashboard_jobs_rows.items()}
            self._dashboard_jobs_rows[job_id] = row
        values = [
            job_id,
            str(job.get("job_type") or ""),
            str(job.get("scenario") or ""),
            str(job.get("state") or ""),
            self._fmt_dashboard_timestamp(job.get("created_at")),
            self._fmt_dashboard_timestamp(job.get("finished_at")),
        ]
        for col, value in enumerate(values):
            item = QTableWidgetItem(value)
            if value:
                item.setToolTip(value)
            table.setItem(row, col, item)
        # Keep only most-recent rows for responsiveness.
        max_rows = 250
        while table.rowCount() > max_rows:
            drop_row = table.rowCount() - 1
            stale_job_id = next((k for k, v in self._dashboard_jobs_rows.items() if v == drop_row), "")
            table.removeRow(drop_row)
            if stale_job_id:
                self._dashboard_jobs_rows.pop(stale_job_id, None)
        table.resizeRowsToContents()
        self._refresh_dashboard_jobs_visual()

    def _refresh_dashboard_jobs_visual(self) -> None:
        if self._dashboard_overview is None or self._dashboard_jobs_table is None:
            return
        counts = {"queued": 0, "running": 0, "done": 0, "error": 0, "other": 0}
        table = self._dashboard_jobs_table
        for row in range(table.rowCount()):
            item = table.item(row, 3)
            state = str(item.text() if item is not None else "").strip().lower()
            if state in {"queued", "pending", "submitted", "created", "staged"}:
                counts["queued"] += 1
            elif state in {"running", "active", "in_progress", "processing"}:
                counts["running"] += 1
            elif state in {"done", "completed", "complete", "succeeded", "success"}:
                counts["done"] += 1
            elif state in {"error", "failed", "failure", "canceled", "cancelled"}:
                counts["error"] += 1
            else:
                counts["other"] += 1
        self._dashboard_overview.set_jobs(counts)

    @staticmethod
    def _fmt_dashboard_timestamp(value: Any) -> str:
        return format_timestamp(value, seconds=True, empty="")

    # ---------- Errors ----------

    def _on_submission_failed(self, msg: str) -> None:
        self._status.setText(f"Submit failed: {msg}")
        self._test_range_last_job_id = ""
        if self._test_range_result_panel is not None:
            self._test_range_result_panel.show_message(f"Quick test submit failed: {msg}")
        self._append_error("submit", msg)

    def _on_console_flag_requested(self, payload: object) -> None:
        """Save Console-level feedback for a run/model artifact."""
        if isinstance(payload, dict):
            context = dict(payload)
        else:
            context = {"source": "console", "weights_path": str(payload or "").strip()}
        weights_path = str(context.get("weights_path") or "").strip()
        if not weights_path:
            return
        dlg = ModelFeedbackDialog(payload=context, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted or dlg.result_payload is None:
            return
        try:
            feedback = model_feedback_store.ConsoleModelFeedback.new(
                payload=context,
                issue_type=dlg.result_payload.issue_type,
                severity=dlg.result_payload.severity,
                notes=dlg.result_payload.notes,
                recommendation=dlg.result_payload.recommendation,
            )
            model_feedback_store.append_feedback(feedback)
        except Exception as exc:
            self._append_error("model-feedback", f"Could not save feedback: {exc}")
            return
        weight_name = Path(weights_path).name or weights_path
        self._status.setText(f"Saved Console feedback for {weight_name}.")

    def _append_error(self, source: str, message: str, *, mirror_notification: bool = True) -> None:
        text = str(message or "").strip()
        if not text:
            return
        now = time.time()
        key = f"{source}|{text}"
        if key == self._last_error_key and (now - self._last_error_ts) < 8.0:
            return
        self._last_error_key = key
        self._last_error_ts = now
        stamp = format_timestamp(now, seconds=True, empty="")
        line = f"[{stamp}] [{source}] {text}"
        if mirror_notification:
            self._emit_local_notification(
                "local_error",
                scope=source,
                state="error",
                message=text,
            )
        errors_text = getattr(self, "_errors_text", None)
        if errors_text is None:
            self._pending_error_lines.append(line)
            return
        errors_text.append(line)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    default_scheme = str(os.environ.get("INSIGHT_CVOPS_COLOR_SCHEME", "aurora")).strip().lower()
    if default_scheme not in COLOR_SCHEME_CHOICES:
        default_scheme = "aurora"
    parser = argparse.ArgumentParser(description="CV Ops training control window")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--color-scheme", choices=COLOR_SCHEME_CHOICES, default=default_scheme)
    return parser.parse_args(argv)


def _bprint(msg: str) -> None:
    """Advance the boot loading bar by one step; falls back to a plain print if unavailable."""
    try:
        from insight_local.cvops.__main__ import _boot_step  # noqa: PLC0415
        _boot_step(msg)
    except Exception:
        print(f"[BOOT] {msg}", flush=True)


def _preinit_webengine() -> None:
    """Configure Qt WebEngine before QApplication for direct window.main callers."""
    if QApplication.instance() is not None:
        return
    _mark_boot("webengine_config_start", once=True)
    try:
        existing = os.environ.get("QTWEBENGINE_CHROMIUM_FLAGS", "")
        flags = [
            "--enable-gpu-rasterization",
            "--enable-zero-copy",
            "--ignore-gpu-blocklist",
            "--enable-accelerated-video-decode",
            "--enable-features=VaapiVideoDecoder",
        ]
        os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (existing + " " + " ".join(flags)).strip()
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
        import PyQt6.QtWebEngineWidgets as _webengine  # noqa: F401, PLC0415
        _ = _webengine
    except Exception as exc:
        _mark_boot("webengine_config_failed", once=True, error=str(exc))
    _mark_boot("webengine_config_done", once=True)


def main(argv: Optional[list[str]] = None) -> int:
    _bprint("main() entered — parsing args + loading settings")
    args = _parse_args(argv)
    settings = load_cvops_settings(CVOPS_STATE_DIR / "settings.json")
    requested_scheme = str(args.color_scheme or "").strip().lower()
    env_scheme = str(os.environ.get("INSIGHT_CVOPS_COLOR_SCHEME", "")).strip().lower()
    if requested_scheme and (env_scheme or requested_scheme != "aurora" or settings.color_scheme == "default"):
        settings.color_scheme = requested_scheme
    settings.ui_scale_pct = normalize_ui_scale_pct(
        os.environ.get("INSIGHT_CVOPS_UI_SCALE", settings.ui_scale_pct)
    )
    _bprint(f"color scheme: {settings.color_scheme!r}  scale: {settings.ui_scale_pct!r}")
    configure_color_scheme(settings.color_scheme)

    _bprint("creating QApplication")
    _mark_boot("qapplication_start", once=True)
    _preinit_webengine()
    app = QApplication.instance()
    owns_app = app is None
    if app is None:
        prog = ""
        try:
            prog = str(sys.argv[0] or "").strip()
        except Exception:
            prog = ""
        app = QApplication([prog or "insight-cvops"])
    app.setStyle(QStyleFactory.create("Fusion"))
    app.setApplicationName("CV Ops")
    _mark_boot("qapplication_done", once=True)

    # Map the design's preferred font families onto concrete installed faces so
    # text (QSS included) renders crisply instead of collapsing onto the hidden
    # system font. Must run before the default font is chosen below; affects QSS
    # font matching live, so it also fixes the stylesheet applied just after.
    base_ui_family = install_cvops_font_substitutions()

    _bprint("building stylesheet (wallpaper + color scale)")
    _mark_boot("stylesheet_start", once=True)
    wp_boot = resolve_workspace_wallpaper_path(settings, CVOPS_STATE_DIR)
    blend_boot = blend_from_cvops_settings(settings) if wp_boot is not None else None
    boot_scale = resolve_ui_scale_factor((1180, 800), settings.ui_scale_pct)
    app.setStyleSheet(
        scale_qss_pixel_metrics(
            get_global_stylesheet(cv_ops_wallpaper_blend=blend_boot)
            + get_cvops_stylesheet(
                workspace_wallpaper=wp_boot,
                backdrop_blend=blend_boot,
                title_text_color=settings.title_text_color,
                title_background_color=settings.title_background_color,
                ui_text_color=settings.ui_text_color,
                ui_muted_text_color=settings.ui_muted_text_color,
                ui_background_color=settings.ui_background_color,
                ui_panel_background_color=settings.ui_panel_background_color,
                ui_control_background_color=settings.ui_control_background_color,
                ui_accent_color=settings.ui_accent_color,
            ),
            boot_scale,
        )
    )
    _mark_boot("stylesheet_done", once=True)
    # Use a concrete installed sans family (resolved above) so the default font is
    # a real face, not the hidden '.AppleSystemUIFont' that bare substitution picks.
    font = QFont(base_ui_family or "IBM Plex Sans", 10)
    if not base_ui_family and not font.exactMatch():
        font = QFont("Roboto", 10)
    if not base_ui_family and not font.exactMatch():
        font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    # Force the anti-alias rasterizer for every QFont derived from this default
    # — without it, Qt can pick a bitmap fallback for small (10 px) sizes which
    # is what produces the jaggy look on title chips.
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)

    _bprint(f"constructing CvOpsWindow (host={args.host} port={args.port})")
    win = CvOpsWindow(host=str(args.host), port=int(args.port), settings=settings)
    _bprint("CvOpsWindow constructed — calling win.show()")
    apply_text_palette(win)
    win.show()
    app.processEvents()
    _mark_boot("first_show", once=True)
    _bprint("window visible — entering event loop")

    # Safety-net: seal the bar after 8 s regardless of WS state so it never
    # hangs on-screen if the WebSocket takes a long time to first connect.
    def _finish_bar_fallback() -> None:
        try:
            from insight_local.cvops.__main__ import _boot_finish  # noqa: PLC0415
            _boot_finish()
        except Exception:
            pass

    QTimer.singleShot(8000, _finish_bar_fallback)

    if owns_app:
        rc = int(app.exec())
        # The Qt event loop has ended (window closed). closeEvent() has already
        # persisted settings and stopped the backend server, WS client, and
        # timers. We must NOT fall through to normal Python finalization here:
        # PyQt6/sip's atexit handler walks every surviving Qt wrapper and frees
        # it, and with QtWebEngine's render/IO/GPU threads still tearing down,
        # that teardown order segfaults inside QToolBar/QWidgetAction (see
        # crash report: Py_FinalizeEx -> sip cleanup_on_exit -> QWidget dtors ->
        # QMetaObject::cast on freed memory). Force an immediate, hard exit so
        # the OS reclaims everything instead of sip/WebEngine teardown.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
